"""Multimodal, memory-safe MASKED_GLORIA for the PGL pipeline.

This implementation keeps the original GLORIA layout: two independent
ID-embedding user-item branches (full and softly masked), three propagation
steps summed with the initial embeddings, concatenation of both branches,
and item-only propagation on a feature-derived graph.  The item graph fuses
visual and textual kNN graphs, while edge-index propagation avoids dense
``(n_users + n_items) ** 2`` gradients for the learnable interaction mask.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


def _config_value(config, key, default):
    value = config[key]
    return default if value is None else value


class MASKED_GLORIA(GeneralRecommender):
    """GLORIA with full/masked U-I branches and a multimodal item graph."""

    MM_CACHE_VERSION = 1

    def __init__(self, config, dataset):
        super(MASKED_GLORIA, self).__init__(config, dataset)

        self.embedding_dim = int(_config_value(config, 'embedding_size', 64))
        self.feat_embed_dim = int(
            _config_value(config, 'feat_embed_dim', self.embedding_dim)
        )
        self.n_layers = int(_config_value(config, 'n_mm_layers', 1))
        self.knn_k = int(_config_value(config, 'knn_k', 10))
        self.mm_image_weight = float(
            _config_value(config, 'mm_image_weight', 0.1)
        )
        self.mask_keep_ratio = float(
            _config_value(config, 'mask_keep_ratio', 0.3)
        )
        self.mask_weight = float(_config_value(config, 'mask_weight', 0.1))
        self.mask_binary_weight = float(
            _config_value(config, 'mask_binary_weight', 0.1)
        )
        self.cl_weight = float(_config_value(config, 'cl_weight', 0.0))
        self.cl_temperature = float(
            _config_value(config, 'cl_temperature', 0.2)
        )
        self.cl_dropout = float(_config_value(config, 'dropout', 0.2))
        self.aggr_mode = str(_config_value(config, 'aggr_mode', 'add')).lower()
        self.fusion = str(_config_value(config, 'fusion', 'concat')).lower()

        if self.embedding_dim <= 0 or self.feat_embed_dim <= 0:
            raise ValueError('Embedding dimensions must be positive.')
        if self.embedding_dim != self.feat_embed_dim:
            raise ValueError(
                'MASKED_GLORIA requires embedding_size == feat_embed_dim.'
            )
        if self.n_layers < 0:
            raise ValueError('n_mm_layers must be non-negative.')
        if self.knn_k <= 0:
            raise ValueError('knn_k must be positive.')
        if not 0.0 <= self.mm_image_weight <= 1.0:
            raise ValueError('mm_image_weight must be in [0, 1].')
        if not 0.0 < self.mask_keep_ratio < 1.0:
            raise ValueError('mask_keep_ratio must be between 0 and 1.')
        if self.mask_weight < 0.0 or self.mask_binary_weight < 0.0:
            raise ValueError('Mask loss weights must be non-negative.')
        if self.cl_weight < 0.0:
            raise ValueError('cl_weight must be non-negative.')
        if self.cl_temperature <= 0.0:
            raise ValueError('cl_temperature must be positive.')
        if not 0.0 <= self.cl_dropout < 1.0:
            raise ValueError('dropout must be in the interval [0, 1).')
        if self.aggr_mode != 'add':
            raise ValueError("MASKED_GLORIA only supports aggr_mode='add'.")
        if self.fusion != 'concat':
            raise ValueError("MASKED_GLORIA only supports fusion='concat'.")
        if self.v_feat is None or self.t_feat is None:
            raise ValueError(
                'MASKED_GLORIA requires both visual and textual item features.'
            )
        if self.v_feat.dim() != 2 or self.t_feat.dim() != 2:
            raise ValueError('Visual and textual features must be 2-D tensors.')
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

        self.num_user = self.n_users
        self.num_item = self.n_items
        self.final_embedding_dim = 2 * self.feat_embed_dim

        interaction_matrix = dataset.inter_matrix(form='coo').astype(
            np.float32
        )
        interaction_matrix = interaction_matrix.tocsr()
        interaction_matrix.eliminate_zeros()
        interaction_matrix.data.fill(1.0)
        self.interaction_matrix = interaction_matrix.tocoo()
        if self.interaction_matrix.nnz == 0:
            raise ValueError('MASKED_GLORIA requires at least one interaction.')

        packed_edges = self.pack_edge_index(self.interaction_matrix)
        forward_edges = torch.from_numpy(packed_edges).to(
            device=self.device, dtype=torch.long
        ).transpose(0, 1).contiguous()
        reverse_edges = forward_edges.flip(0)
        edge_index = torch.cat((forward_edges, reverse_edges), dim=1)
        self.register_buffer('edge_index', edge_index)

        self.num_interactions = self.interaction_matrix.nnz
        full_edge_norm = self._full_edge_normalization(edge_index)
        self.register_buffer('full_edge_norm', full_edge_norm)

        initial_logit = math.log(
            self.mask_keep_ratio / (1.0 - self.mask_keep_ratio)
        )
        self.mask_logits = nn.Parameter(
            torch.full(
                (self.num_interactions,),
                initial_logit,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.id_embedding_full = nn.Embedding(
            self.n_items, self.feat_embed_dim
        )
        self.id_embedding_masked = nn.Embedding(
            self.n_items, self.feat_embed_dim
        )
        nn.init.xavier_uniform_(self.id_embedding_full.weight)
        nn.init.xavier_uniform_(self.id_embedding_masked.weight)

        gcn_kwargs = {
            'datasets': dataset,
            'batch_size': self.batch_size,
            'num_user': self.n_users,
            'num_item': self.n_items,
            'dim_id': self.embedding_dim,
            'aggr_mode': self.aggr_mode,
            'num_layer': 3,
            'has_feature': False,
            'dropout': 0.0,
            'dim_latent': self.feat_embed_dim,
            'device': self.device,
        }
        self.full_gcn = GCN(
            features=self.id_embedding_full.weight, **gcn_kwargs
        )
        self.mask_gcn = GCN(
            features=self.id_embedding_masked.weight, **gcn_kwargs
        )

        mm_adj = self._build_or_load_mm_graph(config)
        self.register_buffer('mm_adj', mm_adj.coalesce())
        self.cl_dropout_layer = nn.Dropout(self.cl_dropout)
        self.latest_loss_components = {}
        self.result_embed = None

    def _full_edge_normalization(self, edge_index):
        source, target = edge_index
        degree = torch.zeros(
            self.n_users + self.n_items,
            dtype=torch.float32,
            device=edge_index.device,
        )
        degree.index_add_(0, source, torch.ones_like(source, dtype=torch.float32))
        degree_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
        degree_inv_sqrt = torch.where(
            degree > 0,
            degree_inv_sqrt,
            torch.zeros_like(degree_inv_sqrt),
        )
        return degree_inv_sqrt[source] * degree_inv_sqrt[target]

    def _mm_cache_metadata(self, config):
        return {
            'version': self.MM_CACHE_VERSION,
            'num_items': self.n_items,
            'knn_k': self.knn_k,
            'mm_image_weight': self.mm_image_weight,
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
        cache_name = 'masked_gloria_mm_v{}_k{}_w{}.pt'.format(
            self.MM_CACHE_VERSION, self.knn_k, weight_label
        )
        self.mm_cache_file = os.path.join(dataset_path, cache_name)
        expected_metadata = self._mm_cache_metadata(config)

        if os.path.isfile(self.mm_cache_file):
            cached = self._load_cache(self.mm_cache_file, self.device)
            if isinstance(cached, dict):
                cached_metadata = cached.get('metadata')
                cached_adjacency = cached.get('adjacency')
                if (
                    cached_metadata == expected_metadata
                    and torch.is_tensor(cached_adjacency)
                    and cached_adjacency.layout == torch.sparse_coo
                    and tuple(cached_adjacency.shape)
                    == (self.n_items, self.n_items)
                ):
                    return cached_adjacency.to(self.device).coalesce()

        with torch.no_grad():
            _, image_adj = self.get_knn_adj_mat(self.v_feat.detach())
            _, text_adj = self.get_knn_adj_mat(self.t_feat.detach())
            mm_adj = (
                self.mm_image_weight * image_adj
                + (1.0 - self.mm_image_weight) * text_adj
            ).coalesce()

        payload = {
            'metadata': expected_metadata,
            'adjacency': mm_adj.detach().cpu().coalesce(),
        }
        self._atomic_torch_save(payload, self.mm_cache_file)
        return mm_adj.to(self.device).coalesce()

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
        row_sum = torch.sparse.sum(adjacency, dim=1).to_dense()
        degree_inv_sqrt = row_sum.clamp_min(1e-12).pow(-0.5)
        row, col = adjacency.indices()
        normalized_values = (
            degree_inv_sqrt[row]
            * adjacency.values()
            * degree_inv_sqrt[col]
        )
        return torch.sparse_coo_tensor(
            adjacency.indices(),
            normalized_values,
            adj_size,
            device=indices.device,
        ).coalesce()

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row.astype(np.int64, copy=False)
        cols = inter_mat.col.astype(np.int64, copy=False) + self.n_users
        return np.column_stack((rows, cols))

    def item_item(self, rep):
        propagated = rep
        for _ in range(self.n_layers):
            propagated = torch.sparse.mm(self.mm_adj, propagated)
        return rep + propagated

    def _encode(self):
        full_rep, full_preference = self.full_gcn(
            self.edge_index,
            self.id_embedding_full.weight,
            edge_norm=self.full_edge_norm,
        )

        interaction_mask = torch.sigmoid(self.mask_logits)
        edge_mask = torch.cat((interaction_mask, interaction_mask), dim=0)
        masked_rep, masked_preference = self.mask_gcn(
            self.edge_index,
            self.id_embedding_masked.weight,
            edge_mask=edge_mask,
            edge_norm=self.full_edge_norm,
        )

        full_users, full_items = torch.split(
            full_rep, [self.n_users, self.n_items], dim=0
        )
        masked_users, masked_items = torch.split(
            masked_rep, [self.n_users, self.n_items], dim=0
        )
        users = torch.cat((full_users, masked_users), dim=1)
        collaborative_items = torch.cat(
            (full_items, masked_items), dim=1
        )
        items = self.item_item(collaborative_items)

        return {
            'users': users,
            'items': items,
            'full_users': full_users,
            'full_items': full_items,
            'masked_users': masked_users,
            'masked_items': masked_items,
            'full_preference': full_preference,
            'masked_preference': masked_preference,
            'mask': interaction_mask,
        }

    def forward(self, interaction):
        representations = self._encode()
        self.result_embed = torch.cat(
            (representations['users'], representations['items']), dim=0
        )

        user_nodes = interaction[0]
        positive_item_nodes = interaction[1] + self.n_users
        negative_item_nodes = interaction[2] + self.n_users
        user_tensor = self.result_embed[user_nodes]
        positive_item_tensor = self.result_embed[positive_item_nodes]
        negative_item_tensor = self.result_embed[negative_item_nodes]
        positive_scores = torch.sum(
            user_tensor * positive_item_tensor, dim=1
        )
        negative_scores = torch.sum(
            user_tensor * negative_item_tensor, dim=1
        )
        return positive_scores, negative_scores

    def InfoNCE(self, view1, view2, temperature=None):
        """One-direction dropout-view InfoNCE used by the original PGL."""
        if temperature is None:
            temperature = self.cl_temperature
        view1 = F.normalize(view1, p=2, dim=1, eps=1e-12)
        view2 = F.normalize(view2, p=2, dim=1, eps=1e-12)
        logits = torch.matmul(view1, view2.transpose(0, 1)) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def calculate_loss(self, interaction):
        positive_scores, negative_scores = self.forward(interaction)
        ranking_loss = -F.logsigmoid(
            positive_scores - negative_scores
        ).mean()

        interaction_mask = torch.sigmoid(self.mask_logits)
        mask_mean = interaction_mask.mean()
        budget_loss = (mask_mean - self.mask_keep_ratio).pow(2)
        binary_loss = (
            interaction_mask * (1.0 - interaction_mask)
        ).mean()
        mask_loss = budget_loss + self.mask_binary_weight * binary_loss

        if self.cl_weight > 0.0:
            user_embeddings = self.result_embed[interaction[0]]
            positive_item_embeddings = self.result_embed[
                interaction[1] + self.n_users
            ]
            user_cl_loss = self.InfoNCE(
                self.cl_dropout_layer(user_embeddings),
                self.cl_dropout_layer(user_embeddings),
            )
            item_cl_loss = self.InfoNCE(
                self.cl_dropout_layer(positive_item_embeddings),
                self.cl_dropout_layer(positive_item_embeddings),
            )
            contrastive_loss = 0.5 * (user_cl_loss + item_cl_loss)
        else:
            contrastive_loss = ranking_loss.new_zeros(())

        total_loss = (
            ranking_loss
            + self.cl_weight * contrastive_loss
            + self.mask_weight * mask_loss
        )

        self.latest_loss_components = {
            'bpr': ranking_loss.detach(),
            'contrastive': contrastive_loss.detach(),
            'mask': mask_loss.detach(),
            'mask_mean': mask_mean.detach(),
        }
        return total_loss

    def full_sort_predict(self, interaction):
        representations = self._encode()
        user_tensor = representations['users'][interaction[0]]
        return torch.matmul(user_tensor, representations['items'].transpose(0, 1))

    @torch.no_grad()
    def get_analysis_artifacts(self):
        was_training = self.training
        self.eval()
        representations = self._encode()

        probabilities = torch.sigmoid(self.mask_logits)
        keep_count = max(
            1,
            min(
                self.num_interactions,
                int(round(self.num_interactions * self.mask_keep_ratio)),
            ),
        )
        selected_indices = torch.topk(
            self.mask_logits, keep_count, sorted=False
        ).indices
        selected = torch.zeros_like(probabilities, dtype=torch.bool)
        selected[selected_indices] = True

        forward_edges = self.edge_index[:, :self.num_interactions]
        artifacts = {
            'metadata': {
                'model': self.__class__.__name__,
                'mask_graph_mode': 'soft',
                'mask_degree_mode': 'full',
                'ui_branch_mode': 'dual',
                'ui_fusion_mode': 'concat',
                'user_embedding_mode': 'separate',
                'mask_keep_ratio': self.mask_keep_ratio,
                'cl_weight': self.cl_weight,
                'cl_temperature': self.cl_temperature,
                'mm_image_weight': self.mm_image_weight,
                'knn_k': self.knn_k,
                'n_mm_layers': self.n_layers,
                'num_users': self.n_users,
                'num_items': self.n_items,
                'num_interactions': self.num_interactions,
            },
            'ui_edges': {
                'user_ids': forward_edges[0].detach().cpu(),
                'item_ids': (
                    forward_edges[1] - self.n_users
                ).detach().cpu(),
            },
            'masks': {
                'masked_branch': {
                    'logits': self.mask_logits.detach().cpu(),
                    'probabilities': probabilities.detach().cpu(),
                    'selected_at_keep_ratio': selected.detach().cpu(),
                }
            },
            'embedding_tables': {
                'id_embedding_full.weight': (
                    self.id_embedding_full.weight.detach().cpu()
                ),
                'id_embedding_masked.weight': (
                    self.id_embedding_masked.weight.detach().cpu()
                ),
                'full_gcn.preference': (
                    self.full_gcn.preference.detach().cpu()
                ),
                'mask_gcn.preference': (
                    self.mask_gcn.preference.detach().cpu()
                ),
            },
            'representations': {
                key: value.detach().cpu()
                for key, value in representations.items()
                if torch.is_tensor(value) and key != 'mask'
            },
        }

        if was_training:
            self.train()
        return artifacts


class GCN(nn.Module):
    """Three-hop parameter-free graph convolution used by GLORIA."""

    def __init__(
        self,
        datasets,
        batch_size,
        num_user,
        num_item,
        dim_id,
        aggr_mode,
        num_layer,
        has_feature,
        dropout,
        dim_latent=None,
        device=None,
        features=None,
        user_profile=None,
    ):
        super(GCN, self).__init__()
        if features is None:
            raise ValueError('GCN requires an item feature/embedding table.')

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.has_feature = has_feature
        self.dropout = dropout
        self.device = device
        self.userprofile = user_profile
        self.num_layer = num_layer

        preference_dim = dim_latent if has_feature else self.dim_feat
        self.preference = nn.Parameter(
            torch.empty(num_user, preference_dim, device=features.device)
        )
        nn.init.xavier_normal_(self.preference)
        self.conv_embed_1 = Base_gcn(
            preference_dim, preference_dim, aggr=self.aggr_mode
        )

    def forward(self, edge_index, features, edge_mask=None, edge_norm=None):
        x = torch.cat((self.preference, features), dim=0)
        x = F.normalize(x, p=2, dim=-1, eps=1e-12)
        h = self.conv_embed_1(
            x, edge_index, edge_mask=edge_mask, edge_norm=edge_norm
        )
        h_1 = self.conv_embed_1(
            h, edge_index, edge_mask=edge_mask, edge_norm=edge_norm
        )
        h_2 = self.conv_embed_1(
            h_1, edge_index, edge_mask=edge_mask, edge_norm=edge_norm
        )
        return x + h + h_1 + h_2, self.preference


class Base_gcn(nn.Module):
    """Memory-safe normalized message passing over an edge list."""

    def __init__(
        self,
        in_channels,
        out_channels,
        normalize=True,
        bias=True,
        aggr='add',
        **kwargs
    ):
        super(Base_gcn, self).__init__()
        if aggr != 'add':
            raise ValueError("Base_gcn only supports aggr='add'.")
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.bias = bias

    def forward(
        self,
        x,
        edge_index,
        edge_mask=None,
        size=None,
        edge_norm=None,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if size is None:
            size = (x.size(0), x.size(0))

        source, target = edge_index
        if edge_mask is None:
            edge_mask = torch.ones(
                edge_index.size(1), dtype=x.dtype, device=x.device
            )
        else:
            edge_mask = edge_mask.to(dtype=x.dtype, device=x.device)

        if edge_norm is None:
            degree = torch.zeros(size[0], dtype=x.dtype, device=x.device)
            degree.index_add_(
                0, source, torch.ones_like(source, dtype=x.dtype)
            )
            degree_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
            degree_inv_sqrt = torch.where(
                degree > 0,
                degree_inv_sqrt,
                torch.zeros_like(degree_inv_sqrt),
            )
            edge_norm = degree_inv_sqrt[source] * degree_inv_sqrt[target]
        else:
            edge_norm = edge_norm.to(dtype=x.dtype, device=x.device)

        messages = self.message(
            x.index_select(0, source),
            edge_index,
            size,
            edge_mask,
            edge_norm,
        )
        aggregated = x.new_zeros((size[1], x.size(1)))
        aggregated.index_add_(0, target, messages)
        return self.update(aggregated)

    def message(self, x_j, edge_index, size, edge_mask, edge_norm):
        return edge_norm.view(-1, 1) * edge_mask.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({},{})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
