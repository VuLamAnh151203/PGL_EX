r"""
DUAL_MODALITY
################################################

Four-branch extension of PGL with modality-specific interaction masks:
image-full, image-masked, text-full, and text-masked. Full/masked outputs
can be fused by separate modality gates or by one joint gate after image/text
concatenation. The multimodal I-I path remains parallel to U-I propagation.

The public method layout intentionally follows ``models/pgl.py`` so the
model remains easy to compare with the original implementation.
"""

import math
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


def _config_value(config, key, default):
    value = config[key]
    return default if value is None else value


class _ObservedEdgeSparseMM(torch.autograd.Function):
    """Sparse MM whose adjacency gradient is evaluated only on COO edges."""

    @staticmethod
    def forward(ctx, indices, values, size, embeddings):
        adjacency = torch.sparse_coo_tensor(
            indices,
            values,
            size,
            dtype=values.dtype,
            device=values.device,
        ).coalesce()
        ctx.adjacency_size = tuple(size)
        ctx.save_for_backward(
            adjacency.indices(), adjacency.values(), embeddings
        )
        return torch.sparse.mm(adjacency, embeddings)

    @staticmethod
    def backward(ctx, grad_output):
        indices, values, embeddings = ctx.saved_tensors
        rows, columns = indices

        grad_values = None
        if ctx.needs_input_grad[1]:
            output_gradients = grad_output.index_select(0, rows)
            source_embeddings = embeddings.index_select(0, columns)
            grad_values = (
                output_gradients * source_embeddings
            ).sum(dim=1)

        grad_embeddings = None
        if ctx.needs_input_grad[3]:
            transpose_indices = torch.stack((columns, rows), dim=0)
            transpose_adjacency = torch.sparse_coo_tensor(
                transpose_indices,
                values,
                (ctx.adjacency_size[1], ctx.adjacency_size[0]),
                dtype=values.dtype,
                device=values.device,
            ).coalesce()
            grad_embeddings = torch.sparse.mm(
                transpose_adjacency, grad_output
            )

        return None, grad_values, None, grad_embeddings


class DUAL_MODALITY(GeneralRecommender):
    """PGL with full/masked U-I branches for image and text separately."""

    MM_CACHE_VERSION = 2

    def __init__(self, config, dataset):
        super(DUAL_MODALITY, self).__init__(config, dataset)

        self.mode = str(_config_value(config, 'mode', 'dual_modality'))
        self.embedding_dim = int(_config_value(config, 'embedding_size', 64))
        self.feat_embed_dim = int(
            _config_value(config, 'feat_embed_dim', self.embedding_dim)
        )
        self.knn_k = int(_config_value(config, 'knn_k', 10))
        self.n_layers = int(_config_value(config, 'n_mm_layers', 1))
        self.n_ui_layers = int(_config_value(config, 'n_ui_layers', 2))
        self.mm_image_weight = float(
            _config_value(config, 'mm_image_weight', 0.1)
        )
        self.mm_graph_mode = str(
            _config_value(config, 'mm_graph_mode', 'mixed')
        ).lower()
        self.cl_weight = float(
            _config_value(
                config,
                'cl_weight',
                _config_value(config, 'reg_weight', 0.0),
            )
        )
        self.cl_temperature = float(
            _config_value(config, 'cl_temperature', 0.2)
        )
        self.cl_mode = str(
            _config_value(config, 'cl_mode', 'pgl_dropout')
        ).lower()
        self.aux_bpr_mode = str(
            _config_value(config, 'aux_bpr_mode', 'none')
        ).lower()
        self.aux_bpr_weight = float(
            _config_value(config, 'aux_bpr_weight', 0.0)
        )
        self.mask_keep_ratio = float(
            _config_value(config, 'mask_keep_ratio', 0.3)
        )
        self.mask_weight = float(_config_value(config, 'mask_weight', 0.1))
        self.mask_binary_weight = float(
            _config_value(config, 'mask_binary_weight', 0.1)
        )
        self.mask_graph_mode = str(
            _config_value(config, 'mask_graph_mode', 'hard')
        ).lower()
        self.mask_generation_mode = str(
            _config_value(config, 'mask_generation_mode', 'edge_logits')
        ).lower()
        self.mask_hidden_dim = int(
            _config_value(config, 'mask_hidden_dim', self.embedding_dim)
        )
        self.mask_sharing_mode = str(
            _config_value(config, 'mask_sharing_mode', 'separate')
        ).lower()
        self.fusion_gate_mode = str(
            _config_value(config, 'fusion_gate_mode', 'separate')
        ).lower()
        self.mask_degree_mode = str(
            _config_value(config, 'mask_degree_mode', 'full')
        ).lower()
        self.hard_mask_temperature = float(
            _config_value(config, 'hard_mask_temperature', 1.0)
        )
        self.cl_dropout = float(_config_value(config, 'dropout', 0.2))

        if self.embedding_dim <= 0 or self.feat_embed_dim <= 0:
            raise ValueError('Embedding dimensions must be positive.')
        if self.embedding_dim != self.feat_embed_dim:
            raise ValueError(
                'DUAL_MODALITY requires embedding_size == feat_embed_dim '
                'so the PGL U-I and I-I paths have the same width.'
            )
        if self.knn_k <= 0:
            raise ValueError('knn_k must be positive.')
        if self.n_layers < 0 or self.n_ui_layers < 0:
            raise ValueError('Propagation layer counts must be non-negative.')
        if not 0.0 <= self.mm_image_weight <= 1.0:
            raise ValueError('mm_image_weight must be in [0, 1].')
        if self.mm_graph_mode not in {'mixed', 'separate'}:
            raise ValueError(
                "mm_graph_mode must be 'mixed' or 'separate'."
            )
        if not 0.0 < self.mask_keep_ratio < 1.0:
            raise ValueError('mask_keep_ratio must be between 0 and 1.')
        if self.mask_weight < 0.0 or self.mask_binary_weight < 0.0:
            raise ValueError('Mask loss weights must be non-negative.')
        if self.cl_weight < 0.0 or self.cl_temperature <= 0.0:
            raise ValueError('Invalid contrastive-learning configuration.')
        if self.cl_mode not in {'pgl_dropout', 'full_masked_concat'}:
            raise ValueError(
                "cl_mode must be 'pgl_dropout' or "
                "'full_masked_concat'."
            )
        if self.aux_bpr_mode not in {
            'none', 'modality', 'masked_branch'
        }:
            raise ValueError(
                "aux_bpr_mode must be 'none', 'modality', or "
                "'masked_branch'."
            )
        if self.aux_bpr_weight < 0.0:
            raise ValueError('aux_bpr_weight must be non-negative.')
        if not 0.0 <= self.cl_dropout < 1.0:
            raise ValueError('dropout must be in [0, 1).')
        if self.mask_graph_mode not in {'soft', 'hard'}:
            raise ValueError("mask_graph_mode must be 'soft' or 'hard'.")
        if self.mask_generation_mode not in {
            'edge_logits', 'feature_network'
        }:
            raise ValueError(
                "mask_generation_mode must be 'edge_logits' or "
                "'feature_network'."
            )
        if self.mask_hidden_dim <= 0:
            raise ValueError('mask_hidden_dim must be positive.')
        if self.mask_sharing_mode not in {'shared', 'separate'}:
            raise ValueError(
                "mask_sharing_mode must be 'shared' or 'separate'."
            )
        if (
            self.mask_generation_mode == 'feature_network'
            and self.mask_sharing_mode != 'separate'
        ):
            raise ValueError(
                "mask_generation_mode='feature_network' requires "
                "mask_sharing_mode='separate' because image and text "
                "use different mask networks."
            )
        if self.fusion_gate_mode not in {'shared', 'separate'}:
            raise ValueError(
                "fusion_gate_mode must be 'shared' or 'separate'."
            )
        if self.mask_degree_mode not in {'full', 'masked'}:
            raise ValueError(
                "mask_degree_mode must be 'full' or 'masked'."
            )
        if self.hard_mask_temperature <= 0.0:
            raise ValueError('hard_mask_temperature must be positive.')
        if self.v_feat is None or self.t_feat is None:
            raise ValueError(
                'DUAL_MODALITY requires both visual and textual features.'
            )
        if self.v_feat.dim() != 2 or self.t_feat.dim() != 2:
            raise ValueError('Visual and textual features must be 2-D.')
        if self.v_feat.size(0) != self.n_items:
            raise ValueError(
                'Visual feature count ({}) does not match n_items ({}).'.format(
                    self.v_feat.size(0), self.n_items
                )
            )
        if self.t_feat.size(0) != self.n_items:
            raise ValueError(
                'Text feature count ({}) does not match n_items ({}).'.format(
                    self.t_feat.size(0), self.n_items
                )
            )

        self.n_nodes = self.n_users + self.n_items
        self.final_embedding_dim = 2 * self.embedding_dim

        interaction_matrix = dataset.inter_matrix(form='coo').astype(
            np.float32
        )
        interaction_matrix = interaction_matrix.tocsr()
        interaction_matrix.eliminate_zeros()
        interaction_matrix.data.fill(1.0)
        self.interaction_matrix = interaction_matrix.tocoo()
        if self.interaction_matrix.nnz == 0:
            raise ValueError('DUAL_MODALITY requires at least one interaction.')
        self.num_interactions = self.interaction_matrix.nnz

        edge_indices, edge_values = self.get_edge_info()
        edge_indices = edge_indices.to(self.device)
        edge_values = edge_values.to(self.device)
        self.register_buffer('edge_indices', edge_indices)
        self.register_buffer('edge_values', edge_values)

        forward_edges = torch.stack(
            (edge_indices[0], edge_indices[1] + self.n_users), dim=0
        )
        reverse_edges = forward_edges.flip(0)
        ui_edge_index = torch.cat((forward_edges, reverse_edges), dim=1)
        full_ui_values = torch.cat((edge_values, edge_values), dim=0)
        self.register_buffer('ui_edge_index', ui_edge_index)
        self.register_buffer('full_ui_values', full_ui_values)
        self.register_buffer(
            'norm_adj',
            torch.sparse_coo_tensor(
                ui_edge_index,
                full_ui_values,
                (self.n_nodes, self.n_nodes),
                device=self.device,
            ).coalesce(),
        )

        initial_logit = math.log(
            self.mask_keep_ratio / (1.0 - self.mask_keep_ratio)
        )
        if self.mask_generation_mode == 'feature_network':
            self.register_parameter('shared_mask_logits', None)
            self.register_parameter('image_mask_logits', None)
            self.register_parameter('text_mask_logits', None)
            self.image_mask_net = self._build_mask_network(initial_logit)
            self.text_mask_net = self._build_mask_network(initial_logit)
        else:
            self.image_mask_net = None
            self.text_mask_net = None
            mask_template = torch.full(
                (self.num_interactions,),
                initial_logit,
                dtype=torch.float32,
                device=self.device,
            )
            if self.mask_sharing_mode == 'shared':
                self.shared_mask_logits = nn.Parameter(mask_template)
                self.register_parameter('image_mask_logits', None)
                self.register_parameter('text_mask_logits', None)
            else:
                self.register_parameter('shared_mask_logits', None)
                self.image_mask_logits = nn.Parameter(mask_template.clone())
                self.text_mask_logits = nn.Parameter(mask_template.clone())

        for modality in ('shared', 'image', 'text'):
            for split in ('train', 'eval'):
                self.register_buffer(
                    '{}_hard_{}_indices'.format(modality, split),
                    torch.empty(0, dtype=torch.long, device=self.device),
                    persistent=False,
                )

        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        self.masked_user_image = nn.Embedding(
            self.n_users, self.embedding_dim
        )
        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        self.masked_user_text = nn.Embedding(
            self.n_users, self.embedding_dim
        )
        for user_table in (
            self.user_image,
            self.masked_user_image,
            self.user_text,
            self.masked_user_text,
        ):
            nn.init.xavier_uniform_(user_table.weight)

        self.image_embedding = nn.Embedding.from_pretrained(
            self.v_feat, freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            self.t_feat, freeze=False
        )
        self.image_trs = nn.Linear(
            self.v_feat.size(1), self.feat_embed_dim
        )
        self.text_trs = nn.Linear(
            self.t_feat.size(1), self.feat_embed_dim
        )

        if self.fusion_gate_mode == 'shared':
            self.shared_fusion_gate = nn.Linear(
                4 * self.embedding_dim, 2 * self.embedding_dim
            )
            self.image_fusion_gate = None
            self.text_fusion_gate = None
            nn.init.xavier_uniform_(self.shared_fusion_gate.weight)
            nn.init.zeros_(self.shared_fusion_gate.bias)
        else:
            self.shared_fusion_gate = None
            self.image_fusion_gate = nn.Linear(
                2 * self.embedding_dim, self.embedding_dim
            )
            self.text_fusion_gate = nn.Linear(
                2 * self.embedding_dim, self.embedding_dim
            )
            for fusion_gate in (
                self.image_fusion_gate,
                self.text_fusion_gate,
            ):
                nn.init.xavier_uniform_(fusion_gate.weight)
                nn.init.zeros_(fusion_gate.bias)

        mm_adj, image_mm_adj, text_mm_adj = (
            self._build_or_load_mm_graph(config)
        )
        self.register_buffer('mm_adj', mm_adj.coalesce())
        self.register_buffer('image_mm_adj', image_mm_adj.coalesce())
        self.register_buffer('text_mm_adj', text_mm_adj.coalesce())
        self.dropoutf = nn.Dropout(self.cl_dropout)
        self.latest_loss_components = {}
        self.latest_representations = None

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        return torch.sparse_coo_tensor(
            indices, values, torch.Size(sparse_mx.shape)
        ).coalesce()

    def get_knn_adj_mat(self, mm_embeddings):
        if self.n_items == 0:
            raise ValueError('Cannot build an item graph without items.')
        topk = min(self.knn_k, self.n_items)
        context_norm = F.normalize(
            mm_embeddings, p=2, dim=-1, eps=1e-12
        )
        similarity = torch.mm(context_norm, context_norm.transpose(0, 1))
        _, knn_indices = torch.topk(similarity, topk, dim=-1)
        del similarity

        rows = torch.arange(
            knn_indices.size(0), device=mm_embeddings.device
        ).unsqueeze(1).expand(-1, topk)
        indices = torch.stack(
            (rows.reshape(-1), knn_indices.reshape(-1)), dim=0
        )
        adjacency_size = torch.Size((self.n_items, self.n_items))
        return indices, self.compute_normalized_laplacian(
            indices, adjacency_size
        )

    def compute_normalized_laplacian(self, indices, adj_size):
        values = torch.ones(
            indices.size(1), dtype=torch.float32, device=indices.device
        )
        adjacency = torch.sparse_coo_tensor(
            indices, values, adj_size, device=indices.device
        ).coalesce()
        row_sum = torch.sparse.sum(adjacency, dim=1).to_dense() + 1e-7
        degree_inv_sqrt = row_sum.pow(-0.5)
        row, col = adjacency.indices()
        normalized_values = degree_inv_sqrt[row] * degree_inv_sqrt[col]
        return torch.sparse_coo_tensor(
            adjacency.indices(),
            normalized_values,
            adj_size,
            device=indices.device,
        ).coalesce()

    def get_norm_adj_mat(self):
        rows = np.concatenate(
            (
                self.interaction_matrix.row,
                self.interaction_matrix.col + self.n_users,
            )
        )
        cols = np.concatenate(
            (
                self.interaction_matrix.col + self.n_users,
                self.interaction_matrix.row,
            )
        )
        adjacency = sp.coo_matrix(
            (np.ones(rows.size, dtype=np.float32), (rows, cols)),
            shape=(self.n_nodes, self.n_nodes),
        ).tocsr()
        adjacency.data.fill(1.0)
        degree = np.asarray((adjacency > 0).sum(axis=1)).reshape(-1) + 1e-7
        degree_inv_sqrt = np.power(degree, -0.5)
        normalized = sp.diags(degree_inv_sqrt) @ adjacency @ sp.diags(
            degree_inv_sqrt
        )
        return self.sparse_mx_to_torch_sparse_tensor(normalized)

    def alignment(self, x, y):
        users, items = self.interaction_matrix.nonzero()
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x[users] - y[items]).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x, t=2):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def _build_mask_network(self, initial_logit):
        mask_network = nn.Sequential(
            nn.Linear(3 * self.embedding_dim, self.mask_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mask_hidden_dim, 1),
        )
        for layer in mask_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.constant_(mask_network[-1].bias, initial_logit)
        return mask_network

    def _project_item_features(self):
        image_features = F.normalize(
            self.image_trs(self.image_embedding.weight),
            p=2,
            dim=-1,
            eps=1e-12,
        )
        text_features = F.normalize(
            self.text_trs(self.text_embedding.weight),
            p=2,
            dim=-1,
            eps=1e-12,
        )
        return image_features, text_features

    def _feature_network_mask_logits(self, modality, item_features):
        if modality == 'image':
            user_table = self.masked_user_image.weight
            mask_network = self.image_mask_net
        elif modality == 'text':
            user_table = self.masked_user_text.weight
            mask_network = self.text_mask_net
        else:
            raise ValueError("modality must be 'image' or 'text'.")

        edge_users, edge_items = self.edge_indices
        user_features = F.normalize(
            user_table.index_select(0, edge_users),
            p=2,
            dim=-1,
            eps=1e-12,
        )
        edge_item_features = item_features.index_select(0, edge_items)
        mask_input = torch.cat(
            (
                user_features,
                edge_item_features,
                user_features * edge_item_features,
            ),
            dim=1,
        )
        return mask_network(mask_input).squeeze(-1)

    def save(self):
        pass

    @property
    def hard_keep_count(self):
        return max(
            1,
            min(
                self.num_interactions,
                int(round(self.num_interactions * self.mask_keep_ratio)),
            ),
        )

    @torch.no_grad()
    def _sample_hard_indices(self, mask_logits):
        uniform_noise = torch.rand_like(mask_logits).clamp_(
            1e-8, 1.0 - 1e-8
        )
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        scores = mask_logits / self.hard_mask_temperature + gumbel_noise
        return torch.topk(
            scores, self.hard_keep_count, sorted=False
        ).indices

    @torch.no_grad()
    def _select_hard_indices(self, mask_logits):
        return torch.topk(
            mask_logits, self.hard_keep_count, sorted=False
        ).indices

    def pre_epoch_processing(self):
        if self.mask_graph_mode == 'hard':
            with torch.no_grad():
                if self.mask_generation_mode == 'feature_network':
                    image_features, text_features = (
                        self._project_item_features()
                    )
                else:
                    image_features, text_features = None, None
                for modality, mask_logits in self._unique_mask_logits(
                    image_features, text_features
                ):
                    setattr(
                        self,
                        '{}_hard_train_indices'.format(modality),
                        self._sample_hard_indices(mask_logits),
                    )

    def post_epoch_processing(self):
        if self.mask_graph_mode == 'hard':
            with torch.no_grad():
                if self.mask_generation_mode == 'feature_network':
                    image_features, text_features = (
                        self._project_item_features()
                    )
                else:
                    image_features, text_features = None, None
                for modality, mask_logits in self._unique_mask_logits(
                    image_features, text_features
                ):
                    setattr(
                        self,
                        '{}_hard_eval_indices'.format(modality),
                        self._select_hard_indices(mask_logits),
                    )

    def _normalize_adj_m(self, indices, adj_size, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(
                indices.size(1), dtype=torch.float32, device=indices.device
            )
        users, items = indices
        user_degree = torch.zeros(
            adj_size[0], dtype=edge_weights.dtype, device=edge_weights.device
        )
        item_degree = torch.zeros(
            adj_size[1], dtype=edge_weights.dtype, device=edge_weights.device
        )
        user_degree.index_add_(0, users, edge_weights)
        item_degree.index_add_(0, items, edge_weights)
        user_inv_sqrt = (user_degree + 1e-7).pow(-0.5)
        item_inv_sqrt = (item_degree + 1e-7).pow(-0.5)
        return (
            user_inv_sqrt[users]
            * edge_weights
            * item_inv_sqrt[items]
        )

    def get_edge_info(self):
        rows = torch.from_numpy(
            self.interaction_matrix.row.astype(np.int64, copy=False)
        )
        cols = torch.from_numpy(
            self.interaction_matrix.col.astype(np.int64, copy=False)
        )
        edges = torch.stack((rows, cols), dim=0)
        values = self._normalize_adj_m(
            edges, torch.Size((self.n_users, self.n_items))
        )
        return edges, values

    def _get_mask_logits(self, modality, item_features=None):
        if self.mask_generation_mode == 'feature_network':
            if item_features is None:
                raise ValueError(
                    'item_features are required for feature-network masks.'
                )
            return self._feature_network_mask_logits(
                modality, item_features
            )
        if self.mask_sharing_mode == 'shared':
            return self.shared_mask_logits
        if modality == 'image':
            return self.image_mask_logits
        if modality == 'text':
            return self.text_mask_logits
        raise ValueError("modality must be 'image' or 'text'.")

    def _unique_mask_logits(
        self, image_features=None, text_features=None
    ):
        if self.mask_generation_mode == 'feature_network':
            return (
                (
                    'image',
                    self._get_mask_logits('image', image_features),
                ),
                (
                    'text',
                    self._get_mask_logits('text', text_features),
                ),
            )
        if self.mask_sharing_mode == 'shared':
            return (('shared', self.shared_mask_logits),)
        return (
            ('image', self.image_mask_logits),
            ('text', self.text_mask_logits),
        )

    def _latest_unique_mask_logits(self):
        if self.mask_generation_mode == 'feature_network':
            if self.latest_representations is None:
                raise RuntimeError(
                    'forward() must run before reading feature-network masks.'
                )
            return (
                (
                    'image',
                    self.latest_representations['image_mask_logits'],
                ),
                (
                    'text',
                    self.latest_representations['text_mask_logits'],
                ),
            )
        return self._unique_mask_logits()

    def _current_hard_indices(self, modality, mask_logits):
        if self.mask_sharing_mode == 'shared':
            modality = 'shared'
        split = 'train' if self.training else 'eval'
        buffer_name = '{}_hard_{}_indices'.format(modality, split)
        indices = getattr(self, buffer_name)
        if indices.numel() == 0:
            if self.training:
                indices = self._sample_hard_indices(mask_logits)
            else:
                indices = self._select_hard_indices(mask_logits)
            setattr(self, buffer_name, indices)
        return indices

    def _masked_ui_adjacency(self, modality, mask_logits):
        probabilities = torch.sigmoid(mask_logits)
        if self.mask_graph_mode == 'hard':
            kept = self._current_hard_indices(modality, mask_logits)
            reverse = kept + self.num_interactions
            kept_undirected = torch.cat((kept, reverse), dim=0)
            edge_index = self.ui_edge_index[:, kept_undirected]
            selected = probabilities.index_select(0, kept)
            straight_through = (
                torch.ones_like(selected) + selected - selected.detach()
            )
            edge_mask = torch.cat(
                (straight_through, straight_through), dim=0
            )
            if self.mask_degree_mode == 'full':
                values = (
                    self.full_ui_values[kept_undirected] * edge_mask
                )
            else:
                bipartite_edges = torch.stack(
                    (
                        edge_index[0, :kept.numel()],
                        edge_index[1, :kept.numel()] - self.n_users,
                    ),
                    dim=0,
                )
                one_direction_values = self._normalize_adj_m(
                    bipartite_edges,
                    torch.Size((self.n_users, self.n_items)),
                    straight_through,
                )
                values = torch.cat(
                    (one_direction_values, one_direction_values), dim=0
                )
        else:
            edge_index = self.ui_edge_index
            edge_mask = torch.cat(
                (probabilities, probabilities), dim=0
            )
            if self.mask_degree_mode == 'full':
                values = self.full_ui_values * edge_mask
            else:
                one_direction_values = self._normalize_adj_m(
                    self.edge_indices,
                    torch.Size((self.n_users, self.n_items)),
                    probabilities,
                )
                values = torch.cat(
                    (one_direction_values, one_direction_values), dim=0
                )

        adjacency = torch.sparse_coo_tensor(
            edge_index,
            values,
            (self.n_nodes, self.n_nodes),
            device=values.device,
        ).coalesce()
        return adjacency, probabilities

    @staticmethod
    def _memory_safe_sparse_mm(adjacency, embeddings):
        adjacency = adjacency.coalesce()
        return _ObservedEdgeSparseMM.apply(
            adjacency.indices(),
            adjacency.values(),
            tuple(adjacency.shape),
            embeddings,
        )

    @staticmethod
    def _propagate_ui_graph(adjacency, initial_embeddings, n_layers):
        adjacency = adjacency.coalesce()
        differentiable_adjacency = adjacency.requires_grad
        all_embeddings = [initial_embeddings]
        current_embeddings = initial_embeddings
        for _ in range(n_layers):
            if differentiable_adjacency:
                current_embeddings = DUAL_MODALITY._memory_safe_sparse_mm(
                    adjacency, current_embeddings
                )
            else:
                current_embeddings = torch.sparse.mm(
                    adjacency, current_embeddings
                )
            all_embeddings.append(current_embeddings)
        return torch.stack(all_embeddings, dim=1).mean(dim=1)

    @staticmethod
    def _gated_sum(full_embeddings, masked_embeddings, fusion_gate):
        gate = torch.sigmoid(
            fusion_gate(
                torch.cat((full_embeddings, masked_embeddings), dim=1)
            )
        )
        fused = (
            gate * full_embeddings
            + (1.0 - gate) * masked_embeddings
        )
        return fused, gate

    def _fuse_full_masked_modalities(
        self,
        image_full,
        image_masked,
        text_full,
        text_masked,
    ):
        if self.fusion_gate_mode == 'shared':
            full_embeddings = torch.cat((image_full, text_full), dim=1)
            masked_embeddings = torch.cat(
                (image_masked, text_masked), dim=1
            )
            fused, shared_gate = self._gated_sum(
                full_embeddings,
                masked_embeddings,
                self.shared_fusion_gate,
            )
            #???
            image_gate, text_gate = torch.split(
                shared_gate,
                [self.embedding_dim, self.embedding_dim],
                dim=1,
            )
            return fused, image_gate, text_gate, shared_gate

        image_embeddings, image_gate = self._gated_sum(
            image_full,
            image_masked,
            self.image_fusion_gate,
        )
        text_embeddings, text_gate = self._gated_sum(
            text_full,
            text_masked,
            self.text_fusion_gate,
        )
        fused = torch.cat((image_embeddings, text_embeddings), dim=1)
        return fused, image_gate, text_gate, None

    def _mm_cache_metadata(self, config):
        return {
            'version': self.MM_CACHE_VERSION,
            'num_items': self.n_items,
            'knn_k': self.knn_k,
            'mm_image_weight': self.mm_image_weight,
            'mm_graph_mode': self.mm_graph_mode,
            'vision_feature_file': str(config['vision_feature_file']),
            'text_feature_file': str(config['text_feature_file']),
            'vision_shape': list(self.v_feat.shape),
            'text_shape': list(self.t_feat.shape),
        }

    @staticmethod
    def _load_cache(file_path, device):
        try:
            return torch.load(file_path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(file_path, map_location=device)

    @staticmethod
    def _atomic_torch_save(payload, file_path):
        temporary_file = file_path + '.tmp'
        torch.save(payload, temporary_file)
        os.replace(temporary_file, file_path)

    def _build_or_load_mm_graph(self, config):
        dataset_path = os.path.abspath(
            os.path.join(str(config['data_path']), str(config['dataset']))
        )
        weight_label = format(self.mm_image_weight, '.8g').replace('.', 'p')
        cache_name = 'dual_modality_mm_v{}_{}_k{}_w{}.pt'.format(
            self.MM_CACHE_VERSION,
            self.mm_graph_mode,
            self.knn_k,
            weight_label,
        )
        self.mm_cache_file = os.path.join(dataset_path, cache_name)
        expected_metadata = self._mm_cache_metadata(config)

        if os.path.exists(self.mm_cache_file):
            payload = self._load_cache(self.mm_cache_file, self.device)
            adjacency_keys = (
                'adjacency', 'image_adjacency', 'text_adjacency'
            )
            valid_adjacencies = all(
                torch.is_tensor(payload.get(key))
                and tuple(payload[key].shape)
                == (self.n_items, self.n_items)
                for key in adjacency_keys
            ) if isinstance(payload, dict) else False
            if (
                isinstance(payload, dict)
                and payload.get('metadata') == expected_metadata
                and valid_adjacencies
            ):
                return tuple(
                    payload[key].to(self.device).coalesce()
                    for key in adjacency_keys
                )

        with torch.no_grad():
            _, image_adj = self.get_knn_adj_mat(self.v_feat)
            _, text_adj = self.get_knn_adj_mat(self.t_feat)
            mm_adj = (
                self.mm_image_weight * image_adj
                + (1.0 - self.mm_image_weight) * text_adj
            ).coalesce()
        os.makedirs(dataset_path, exist_ok=True)
        self._atomic_torch_save(
            {
                'metadata': expected_metadata,
                'adjacency': mm_adj.detach().cpu(),
                'image_adjacency': image_adj.detach().cpu(),
                'text_adjacency': text_adj.detach().cpu(),
            },
            self.mm_cache_file,
        )
        return (
            mm_adj.to(self.device).coalesce(),
            image_adj.to(self.device).coalesce(),
            text_adj.to(self.device).coalesce(),
        )

    def forward(self, adj):
        image_feats, text_feats = self._project_item_features()

        image_full_initial = torch.cat(
            (self.user_image.weight, image_feats), dim=0
        )
        image_masked_initial = torch.cat(
            (self.masked_user_image.weight, image_feats), dim=0
        )
        text_full_initial = torch.cat(
            (self.user_text.weight, text_feats), dim=0
        )
        text_masked_initial = torch.cat(
            (self.masked_user_text.weight, text_feats), dim=0
        )

        unique_mask_logits = dict(
            self._unique_mask_logits(image_feats, text_feats)
        )
        if self.mask_sharing_mode == 'shared':
            image_mask_logits = unique_mask_logits['shared']
            text_mask_logits = unique_mask_logits['shared']
        else:
            image_mask_logits = unique_mask_logits['image']
            text_mask_logits = unique_mask_logits['text']
        image_masked_adj, image_mask = self._masked_ui_adjacency(
            'image', image_mask_logits
        )
        text_masked_adj, text_mask = self._masked_ui_adjacency(
            'text', text_mask_logits
        )
        full_initial = torch.cat(
            (image_full_initial, text_full_initial), dim=1
        )
        full = self._propagate_ui_graph(
            adj, full_initial, self.n_ui_layers
        )
        image_full, text_full = torch.split(
            full,
            [self.embedding_dim, self.embedding_dim],
            dim=1,
        )
        image_masked = self._propagate_ui_graph(
            image_masked_adj, image_masked_initial, self.n_ui_layers
        )
        text_masked = self._propagate_ui_graph(
            text_masked_adj, text_masked_initial, self.n_ui_layers
        )

        image_full_users, image_full_items = torch.split(
            image_full, [self.n_users, self.n_items], dim=0
        )
        image_masked_users, image_masked_items = torch.split(
            image_masked, [self.n_users, self.n_items], dim=0
        )
        text_full_users, text_full_items = torch.split(
            text_full, [self.n_users, self.n_items], dim=0
        )
        text_masked_users, text_masked_items = torch.split(
            text_masked, [self.n_users, self.n_items], dim=0
        )

        user_embeddings, image_user_gate, text_user_gate, shared_user_gate = (
            self._fuse_full_masked_modalities(
                image_full_users,
                image_masked_users,
                text_full_users,
                text_masked_users,
            )
        )
        ui_item_embeddings, image_item_gate, text_item_gate, shared_item_gate = (
            self._fuse_full_masked_modalities(
                image_full_items,
                image_masked_items,
                text_full_items,
                text_masked_items,
            )
        )

        image_user_embeddings = (
            image_user_gate * image_full_users
            + (1.0 - image_user_gate) * image_masked_users
        )
        text_user_embeddings = (
            text_user_gate * text_full_users
            + (1.0 - text_user_gate) * text_masked_users
        )
        image_ui_item_embeddings = (
            image_item_gate * image_full_items
            + (1.0 - image_item_gate) * image_masked_items
        )
        text_ui_item_embeddings = (
            text_item_gate * text_full_items
            + (1.0 - text_item_gate) * text_masked_items
        )

        full_user_embeddings = torch.cat(
            (image_full_users, text_full_users), dim=1
        )
        masked_user_embeddings = torch.cat(
            (image_masked_users, text_masked_users), dim=1
        )
        full_item_embeddings = torch.cat(
            (image_full_items, text_full_items), dim=1
        )
        masked_item_embeddings = torch.cat(
            (image_masked_items, text_masked_items), dim=1
        )

        if self.mm_graph_mode == 'separate':
            image_mm_item_embeddings = image_feats
            text_mm_item_embeddings = text_feats
            for _ in range(self.n_layers):
                image_mm_item_embeddings = torch.sparse.mm(
                    self.image_mm_adj, image_mm_item_embeddings
                )
                text_mm_item_embeddings = torch.sparse.mm(
                    self.text_mm_adj, text_mm_item_embeddings
                )
            mm_item_embeddings = torch.cat(
                (image_mm_item_embeddings, text_mm_item_embeddings), dim=1
            )
        else:
            item_embeddings = torch.cat((image_feats, text_feats), dim=1)
            mm_item_embeddings = item_embeddings
            for _ in range(self.n_layers):
                mm_item_embeddings = torch.sparse.mm(
                    self.mm_adj, mm_item_embeddings
                )
            image_mm_item_embeddings, text_mm_item_embeddings = torch.split(
                mm_item_embeddings,
                [self.embedding_dim, self.embedding_dim],
                dim=1,
            )
        image_item_embeddings = (
            image_ui_item_embeddings + image_mm_item_embeddings
        )
        text_item_embeddings = (
            text_ui_item_embeddings + text_mm_item_embeddings
        )
        final_item_embeddings = torch.cat(
            (image_item_embeddings, text_item_embeddings), dim=1
        )

        self.latest_representations = {
            'users': user_embeddings,
            'items': final_item_embeddings,
            'ui_items': ui_item_embeddings,
            'mm_items': mm_item_embeddings,
            'image_users': image_user_embeddings,
            'text_users': text_user_embeddings,
            'image_items': image_item_embeddings,
            'text_items': text_item_embeddings,
            'image_ui_items': image_ui_item_embeddings,
            'text_ui_items': text_ui_item_embeddings,
            'image_mm_items': image_mm_item_embeddings,
            'text_mm_items': text_mm_item_embeddings,
            'full_users': full_user_embeddings,
            'masked_users': masked_user_embeddings,
            'full_items': full_item_embeddings,
            'masked_items': masked_item_embeddings,
            'image_full_users': image_full_users,
            'image_full_items': image_full_items,
            'image_masked_users': image_masked_users,
            'image_masked_items': image_masked_items,
            'text_full_users': text_full_users,
            'text_full_items': text_full_items,
            'text_masked_users': text_masked_users,
            'text_masked_items': text_masked_items,
            'image_user_gate': image_user_gate,
            'image_item_gate': image_item_gate,
            'text_user_gate': text_user_gate,
            'text_item_gate': text_item_gate,
            'shared_user_gate': shared_user_gate,
            'shared_item_gate': shared_item_gate,
            'image_mask_logits': image_mask_logits,
            'text_mask_logits': text_mask_logits,
            'image_mask': image_mask,
            'text_mask': text_mask,
        }
        return user_embeddings, final_item_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def InfoNCE(self, view1, view2, temperature):
        view1 = F.normalize(view1, p=2, dim=1, eps=1e-12)
        view2 = F.normalize(view2, p=2, dim=1, eps=1e-12)
        logits = torch.matmul(view1, view2.transpose(0, 1)) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        all_user_embeddings, all_item_embeddings = self.forward(self.norm_adj)
        user_embeddings = all_user_embeddings[users]
        positive_embeddings = all_item_embeddings[pos_items]
        negative_embeddings = all_item_embeddings[neg_items]
        ranking_loss = self.bpr_loss(
            user_embeddings, positive_embeddings, negative_embeddings
        )

        if (
            self.aux_bpr_mode != 'none'
            and self.aux_bpr_weight > 0.0
        ):
            representations = self.latest_representations
            if self.aux_bpr_mode == 'masked_branch':
                image_users = representations['image_masked_users']
                image_items = representations['image_masked_items']
                text_users = representations['text_masked_users']
                text_items = representations['text_masked_items']
            else:
                image_users = representations['image_users']
                image_items = representations['image_items']
                text_users = representations['text_users']
                text_items = representations['text_items']

            image_ranking_loss = self.bpr_loss(
                image_users[users],
                image_items[pos_items],
                image_items[neg_items],
            )
            text_ranking_loss = self.bpr_loss(
                text_users[users],
                text_items[pos_items],
                text_items[neg_items],
            )
            auxiliary_ranking_loss = 0.5 * (
                image_ranking_loss + text_ranking_loss
            )
        else:
            image_ranking_loss = ranking_loss.new_zeros(())
            text_ranking_loss = ranking_loss.new_zeros(())
            auxiliary_ranking_loss = ranking_loss.new_zeros(())

        if self.cl_weight > 0.0:
            # if self.cl_mode == 'full_masked_concat':
            #     representations = self.latest_representations
            #     user_cl_loss = self.InfoNCE(
            #         representations['full_users'][users],
            #         representations['masked_users'][users],
            #         self.cl_temperature,
            #     )
            #     item_cl_loss = self.InfoNCE(
            #         representations['full_items'][pos_items],
            #         representations['masked_items'][pos_items],
            #         self.cl_temperature,
            #     )

            if self.cl_mode == 'full_masked_concat':
                representations = self.latest_representations
                unique_users = torch.unique(users)
                unique_items = torch.unique(pos_items)

                user_cl_loss = self.symmetric_info_nce(
                    representations['full_users'][unique_users],
                    representations['masked_users'][unique_users],
                )
                item_cl_loss = self.symmetric_info_nce(
                    representations['full_items'][unique_items],
                    representations['masked_items'][unique_items],
                )
            else:
                user_cl_loss = self.InfoNCE(
                    self.dropoutf(user_embeddings),
                    self.dropoutf(user_embeddings),
                    self.cl_temperature,
                )
                item_cl_loss = self.InfoNCE(
                    self.dropoutf(positive_embeddings),
                    self.dropoutf(positive_embeddings),
                    self.cl_temperature,
                )
            contrastive_loss = 0.5 * (user_cl_loss + item_cl_loss)
        else:
            contrastive_loss = ranking_loss.new_zeros(())

        mask_losses = []
        mask_means = []
        for _, mask_logits in self._latest_unique_mask_logits():
            probabilities = torch.sigmoid(mask_logits)
            mask_mean = probabilities.mean()
            budget_loss = (mask_mean - self.mask_keep_ratio).pow(2)
            binary_loss = (probabilities * (1.0 - probabilities)).mean()
            mask_losses.append(
                budget_loss + self.mask_binary_weight * binary_loss
            )
            mask_means.append(mask_mean)
        mask_loss = torch.stack(mask_losses).mean()
        mean_mask_probability = torch.stack(mask_means).mean()

        total_loss = (
            ranking_loss
            + self.aux_bpr_weight * auxiliary_ranking_loss
            + self.cl_weight * contrastive_loss
            + self.mask_weight * mask_loss
        )
        self.latest_loss_components = {
            'bpr': ranking_loss.detach(),
            'aux_bpr': auxiliary_ranking_loss.detach(),
            'image_bpr': image_ranking_loss.detach(),
            'text_bpr': text_ranking_loss.detach(),
            'contrastive': contrastive_loss.detach(),
            'mask': mask_loss.detach(),
            'mask_mean': mean_mask_probability.detach(),
        }
        return total_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        all_user_embeddings, all_item_embeddings = self.forward(self.norm_adj)
        user_embeddings = all_user_embeddings[users]
        return torch.matmul(
            user_embeddings, all_item_embeddings.transpose(0, 1)
        )

    @torch.no_grad()
    def full_sort_predict_modalities(self, interaction):
        users = interaction[0]
        self.forward(self.norm_adj)
        representations = self.latest_representations
        image_scores = torch.matmul(
            representations['image_users'][users],
            representations['image_items'].transpose(0, 1),
        )
        text_scores = torch.matmul(
            representations['text_users'][users],
            representations['text_items'].transpose(0, 1),
        )
        return {
            'image': image_scores,
            'text': text_scores,
            'joint': image_scores + text_scores,
        }

    @torch.no_grad()
    def modality_triplet_margins(self, interaction):
        users, positive_items, negative_items = interaction[:3]
        self.forward(self.norm_adj)
        representations = self.latest_representations

        def margin(user_table, item_table):
            user_embeddings = user_table[users]
            positive_embeddings = item_table[positive_items]
            negative_embeddings = item_table[negative_items]
            positive_scores = torch.sum(
                user_embeddings * positive_embeddings, dim=1
            )
            negative_scores = torch.sum(
                user_embeddings * negative_embeddings, dim=1
            )
            return positive_scores - negative_scores

        image_margin = margin(
            representations['image_users'],
            representations['image_items'],
        )
        text_margin = margin(
            representations['text_users'],
            representations['text_items'],
        )
        return {
            'image': image_margin,
            'text': text_margin,
            'joint': image_margin + text_margin,
        }

    def symmetric_info_nce(self, first_view, second_view):
        first_view = F.normalize(first_view, dim=1)
        second_view = F.normalize(second_view, dim=1)

        logits = first_view @ second_view.T
        logits = logits / self.cl_temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

    @torch.no_grad()
    def get_analysis_artifacts(self):
        was_training = self.training
        self.eval()
        self.forward(self.norm_adj)
        representations = self.latest_representations

        masks = {}
        for modality, logits in self._latest_unique_mask_logits():
            selected_indices = self._select_hard_indices(logits)
            selected = torch.zeros_like(logits, dtype=torch.bool)
            selected[selected_indices] = True
            masks[modality] = {
                'logits': logits.detach().cpu(),
                'probabilities': torch.sigmoid(logits).detach().cpu(),
                'selected_at_keep_ratio': selected.detach().cpu(),
            }

        artifacts = {
            'metadata': {
                'model': self.__class__.__name__,
                'ui_branch_mode': 'four_branch_dual_modality',
                'mask_graph_mode': self.mask_graph_mode,
                'mask_generation_mode': self.mask_generation_mode,
                'mask_hidden_dim': self.mask_hidden_dim,
                'mask_sharing_mode': self.mask_sharing_mode,
                'fusion_gate_mode': self.fusion_gate_mode,
                'mask_degree_mode': self.mask_degree_mode,
                'mask_keep_ratio': self.mask_keep_ratio,
                'n_ui_layers': self.n_ui_layers,
                'n_mm_layers': self.n_layers,
                'mm_graph_mode': self.mm_graph_mode,
                'mm_image_weight': self.mm_image_weight,
                'cl_weight': self.cl_weight,
                'cl_mode': self.cl_mode,
                'aux_bpr_mode': self.aux_bpr_mode,
                'aux_bpr_weight': self.aux_bpr_weight,
                'num_users': self.n_users,
                'num_items': self.n_items,
                'num_interactions': self.num_interactions,
                'final_embedding_dim': self.final_embedding_dim,
            },
            'ui_edges': {
                'user_ids': self.edge_indices[0].detach().cpu(),
                'item_ids': self.edge_indices[1].detach().cpu(),
            },
            'masks': masks,
            'embedding_tables': {
                'user_image.weight': self.user_image.weight.detach().cpu(),
                'masked_user_image.weight': (
                    self.masked_user_image.weight.detach().cpu()
                ),
                'user_text.weight': self.user_text.weight.detach().cpu(),
                'masked_user_text.weight': (
                    self.masked_user_text.weight.detach().cpu()
                ),
                'image_embedding.weight': (
                    self.image_embedding.weight.detach().cpu()
                ),
                'text_embedding.weight': (
                    self.text_embedding.weight.detach().cpu()
                ),
            },
            'representations': {
                key: value.detach().cpu()
                for key, value in representations.items()
                if torch.is_tensor(value) and key not in {
                    'image_mask',
                    'text_mask',
                    'image_mask_logits',
                    'text_mask_logits',
                }
            },
        }
        if was_training:
            self.train()
        return artifacts
