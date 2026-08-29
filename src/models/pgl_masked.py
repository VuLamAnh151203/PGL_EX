r"""
PGL with two-branch masked user-item graph learning.

The two branches share the same initial node embeddings. The second branch
can use a soft mask, a hard sparse mask, or the complete graph as an ablation.
Masked modes can use either full-graph degrees or degrees recomputed from the
masked weights. Branch outputs are combined by a learnable gate before adding
the multimodal item-item representation.
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


class PGL_MASKED(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PGL_MASKED, self).__init__(config, dataset)

        self.embedding_dim = _config_value(config, 'embedding_size', 64)
        self.feat_embed_dim = _config_value(config, 'feat_embed_dim', 64)
        self.knn_k = _config_value(config, 'knn_k', 10)
        self.n_mm_layers = _config_value(config, 'n_mm_layers', 1)
        self.n_ui_layers = _config_value(config, 'n_ui_layers', 2)
        self.mm_image_weight = _config_value(config, 'mm_image_weight', 0.1)

        self.cl_weight = _config_value(config, 'cl_weight', 0.05)
        self.cl_temperature = _config_value(config, 'cl_temperature', 0.2)
        self.mask_weight = _config_value(config, 'mask_weight', 0.1)
        self.mask_keep_ratio = _config_value(config, 'mask_keep_ratio', 0.3)
        self.mask_binary_weight = _config_value(
            config, 'mask_binary_weight', 0.1
        )
        self.mask_degree_mode = str(
            _config_value(config, 'mask_degree_mode', 'masked')
        ).lower()
        self.mask_graph_mode = str(
            _config_value(config, 'mask_graph_mode', 'soft')
        ).lower()
        self.hard_mask_temperature = _config_value(
            config, 'hard_mask_temperature', 1.0
        )

        if not 0.0 < self.mask_keep_ratio < 1.0:
            raise ValueError('mask_keep_ratio must be between 0 and 1.')
        if self.cl_temperature <= 0.0:
            raise ValueError('cl_temperature must be positive.')
        if self.knn_k <= 0:
            raise ValueError('knn_k must be positive.')
        if self.mask_degree_mode not in {'full', 'masked'}:
            raise ValueError(
                "mask_degree_mode must be either 'full' or 'masked'."
            )
        if self.mask_graph_mode not in {'soft', 'hard', 'double_full'}:
            raise ValueError(
                "mask_graph_mode must be 'soft', 'hard', or 'double_full'."
            )
        if self.hard_mask_temperature <= 0.0:
            raise ValueError('hard_mask_temperature must be positive.')
        if self.v_feat is None or self.t_feat is None:
            raise ValueError(
                'PGL_MASKED requires both image_feat.npy and text_feat.npy.'
            )

        self.n_nodes = self.n_users + self.n_items
        self.ui_embedding_dim = 2 * self.embedding_dim
        self.mm_embedding_dim = 2 * self.feat_embed_dim

        # Use a binary, duplicate-free interaction matrix so one learnable
        # logit always corresponds to exactly one undirected interaction.
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        interaction_matrix = interaction_matrix.tocsr()
        interaction_matrix.eliminate_zeros()
        interaction_matrix.data.fill(1.0)
        self.interaction_matrix = interaction_matrix.tocoo()
        if self.interaction_matrix.nnz == 0:
            raise ValueError('PGL_MASKED requires at least one interaction.')

        self._build_ui_graph()

        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_text.weight)
        nn.init.xavier_uniform_(self.user_image.weight)

        self.image_embedding = nn.Embedding.from_pretrained(
            self.v_feat, freeze=False
        )
        self.text_embedding = nn.Embedding.from_pretrained(
            self.t_feat, freeze=False
        )
        self.image_trs = nn.Linear(
            self.v_feat.shape[1], self.feat_embed_dim
        )
        self.text_trs = nn.Linear(
            self.t_feat.shape[1], self.feat_embed_dim
        )

        if self.mm_embedding_dim == self.ui_embedding_dim:
            self.item_ui_projection = nn.Identity()
            self.mm_output_projection = nn.Identity()
        else:
            self.item_ui_projection = nn.Linear(
                self.mm_embedding_dim, self.ui_embedding_dim
            )
            self.mm_output_projection = nn.Linear(
                self.mm_embedding_dim, self.ui_embedding_dim
            )

        self.fusion_gate = nn.Linear(
            2 * self.ui_embedding_dim, self.ui_embedding_dim
        )
        nn.init.xavier_uniform_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)

        self._build_or_load_mm_graph(config)
        self.latest_loss_components = {}

    def _build_ui_graph(self):
        users = torch.from_numpy(
            self.interaction_matrix.row.astype(np.int64, copy=False)
        )
        items = torch.from_numpy(
            self.interaction_matrix.col.astype(np.int64, copy=False)
        ) + self.n_users

        forward_edges = torch.stack((users, items), dim=0)
        reverse_edges = torch.stack((items, users), dim=0)
        edge_index = torch.cat((forward_edges, reverse_edges), dim=1)
        self.register_buffer('ui_edge_index', edge_index)

        self.num_interactions = self.interaction_matrix.nnz
        initial_logit = math.log(
            self.mask_keep_ratio / (1.0 - self.mask_keep_ratio)
        )
        if self.mask_graph_mode == 'double_full':
            self.register_parameter('mask_logits', None)
        else:
            self.mask_logits = nn.Parameter(
                torch.full((self.num_interactions,), initial_logit)
            )
        self.register_buffer(
            'hard_train_indices', torch.empty(0, dtype=torch.long)
        )
        self.register_buffer(
            'hard_eval_indices', torch.empty(0, dtype=torch.long)
        )

        full_edge_weights = torch.ones(edge_index.size(1), dtype=torch.float32)
        full_norm_edge_weights = self._normalized_ui_edge_weights(
            full_edge_weights
        )
        self.register_buffer(
            'full_norm_edge_weights', full_norm_edge_weights
        )
        norm_adj = self._ui_adjacency_from_weights(full_norm_edge_weights)
        self.register_buffer('norm_adj', norm_adj)

    def _normalized_ui_edge_weights(self, edge_weights, edge_index=None):
        if edge_index is None:
            edge_index = self.ui_edge_index
        row, col = edge_index
        degree = torch.zeros(
            self.n_nodes,
            dtype=edge_weights.dtype,
            device=edge_weights.device,
        )
        degree = degree.index_add(0, row, edge_weights)
        degree_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
        degree_inv_sqrt = torch.where(
            degree > 0,
            degree_inv_sqrt,
            torch.zeros_like(degree_inv_sqrt),
        )
        normalized_weights = (
            degree_inv_sqrt[row] * edge_weights * degree_inv_sqrt[col]
        )
        return normalized_weights

    def _ui_adjacency_from_weights(self, edge_weights, edge_index=None):
        if edge_index is None:
            edge_index = self.ui_edge_index
        return torch.sparse_coo_tensor(
            edge_index,
            edge_weights,
            (self.n_nodes, self.n_nodes),
            device=edge_weights.device,
        ).coalesce()

    def _normalized_ui_adjacency(self, edge_weights, edge_index=None):
        normalized_weights = self._normalized_ui_edge_weights(
            edge_weights, edge_index
        )
        return self._ui_adjacency_from_weights(
            normalized_weights, edge_index
        )

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
    def _sample_hard_train_indices(self):
        uniform_noise = torch.rand_like(self.mask_logits).clamp_(
            1e-8, 1.0 - 1e-8
        )
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        selection_scores = (
            self.mask_logits / self.hard_mask_temperature + gumbel_noise
        )
        return torch.topk(
            selection_scores,
            self.hard_keep_count,
            sorted=False,
        ).indices

    @torch.no_grad()
    def _select_hard_eval_indices(self):
        return torch.topk(
            self.mask_logits,
            self.hard_keep_count,
            sorted=False,
        ).indices

    def pre_epoch_processing(self):
        if self.mask_graph_mode == 'hard':
            self.hard_train_indices = self._sample_hard_train_indices()

    def post_epoch_processing(self):
        if self.mask_graph_mode == 'hard':
            self.hard_eval_indices = self._select_hard_eval_indices()

    def _current_hard_indices(self):
        if self.training:
            if self.hard_train_indices.numel() == 0:
                self.hard_train_indices = self._sample_hard_train_indices()
            return self.hard_train_indices

        if self.hard_eval_indices.numel() == 0:
            self.hard_eval_indices = self._select_hard_eval_indices()
        return self.hard_eval_indices

    def _hard_masked_ui_adjacency(self, interaction_mask):
        kept_interactions = self._current_hard_indices()
        reverse_interactions = kept_interactions + self.num_interactions
        kept_undirected = torch.cat(
            (kept_interactions, reverse_interactions), dim=0
        )
        hard_edge_index = self.ui_edge_index[:, kept_undirected]

        selected_soft_mask = interaction_mask[kept_interactions]
        selected_hard_mask = (
            torch.ones_like(selected_soft_mask)
            + selected_soft_mask
            - selected_soft_mask.detach()
        )
        hard_undirected_mask = torch.cat(
            (selected_hard_mask, selected_hard_mask), dim=0
        )

        if self.mask_degree_mode == 'full':
            masked_edge_weights = (
                self.full_norm_edge_weights[kept_undirected]
                * hard_undirected_mask
            )
            return self._ui_adjacency_from_weights(
                masked_edge_weights, hard_edge_index
            )

        return self._normalized_ui_adjacency(
            hard_undirected_mask, hard_edge_index
        )

    def _masked_ui_adjacency(self):
        if self.mask_graph_mode == 'double_full':
            return self.norm_adj, None

        interaction_mask = torch.sigmoid(self.mask_logits)
        if self.mask_graph_mode == 'hard':
            masked_adj = self._hard_masked_ui_adjacency(interaction_mask)
            return masked_adj, interaction_mask

        undirected_mask = torch.cat(
            (interaction_mask, interaction_mask), dim=0
        )
        if self.mask_degree_mode == 'full':
            masked_edge_weights = (
                self.full_norm_edge_weights * undirected_mask
            )
            masked_adj = self._ui_adjacency_from_weights(
                masked_edge_weights
            )
        else:
            masked_adj = self._normalized_ui_adjacency(undirected_mask)
        return masked_adj, interaction_mask

    def _build_or_load_mm_graph(self, config):
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        cache_name = 'mm_adj_freedomdsp_{}_{}.pt'.format(
            self.knn_k, int(10 * self.mm_image_weight)
        )
        mm_adj_file = os.path.join(dataset_path, cache_name)

        if os.path.exists(mm_adj_file):
            mm_adj = torch.load(mm_adj_file, map_location=self.device)
            if tuple(mm_adj.shape) != (self.n_items, self.n_items):
                raise ValueError(
                    'Cached multimodal graph has shape {}, expected {}.'.format(
                        tuple(mm_adj.shape), (self.n_items, self.n_items)
                    )
                )
        else:
            with torch.no_grad():
                image_adj = self.get_knn_adj_mat(
                    self.image_embedding.weight.detach()
                )
                text_adj = self.get_knn_adj_mat(
                    self.text_embedding.weight.detach()
                )
                mm_adj = (
                    self.mm_image_weight * image_adj
                    + (1.0 - self.mm_image_weight) * text_adj
                ).coalesce()
            torch.save(mm_adj.cpu(), mm_adj_file)
            mm_adj = mm_adj.to(self.device)

        self.register_buffer('mm_adj', mm_adj.coalesce())

    def get_knn_adj_mat(self, mm_embeddings):
        if self.n_items == 0:
            raise ValueError('Cannot build an item graph without items.')

        topk = min(self.knn_k, self.n_items)
        context_norm = F.normalize(mm_embeddings, p=2, dim=-1, eps=1e-12)
        similarity = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_indices = torch.topk(similarity, topk, dim=-1)
        del similarity

        rows = torch.arange(
            knn_indices.size(0), device=mm_embeddings.device
        ).unsqueeze(1).expand(-1, topk)
        indices = torch.stack(
            (rows.reshape(-1), knn_indices.reshape(-1)), dim=0
        )
        return self._normalized_item_adjacency(
            indices, torch.Size((self.n_items, self.n_items))
        )

    @staticmethod
    def _normalized_item_adjacency(indices, size):
        values = torch.ones(
            indices.size(1), dtype=torch.float32, device=indices.device
        )
        adjacency = torch.sparse_coo_tensor(
            indices, values, size, device=indices.device
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
            size,
            device=indices.device,
        ).coalesce()

    def _initial_node_embeddings(self):
        image_features = F.normalize(
            self.image_trs(self.image_embedding.weight), dim=-1
        )
        text_features = F.normalize(
            self.text_trs(self.text_embedding.weight), dim=-1
        )
        multimodal_items = torch.cat(
            (image_features, text_features), dim=1
        )

        user_embeddings = torch.cat(
            (self.user_image.weight, self.user_text.weight), dim=1
        )
        ui_item_embeddings = self.item_ui_projection(multimodal_items)
        initial_embeddings = torch.cat(
            (user_embeddings, ui_item_embeddings), dim=0
        )
        return initial_embeddings, multimodal_items

    def _propagate_ui_graph(self, adjacency, initial_embeddings):
        embeddings = [initial_embeddings]
        current_embeddings = initial_embeddings
        for _ in range(self.n_ui_layers):
            current_embeddings = torch.sparse.mm(
                adjacency, current_embeddings
            )
            embeddings.append(current_embeddings)
        return torch.stack(embeddings, dim=1).mean(dim=1)

    def _propagate_mm_graph(self, item_embeddings):
        propagated_items = item_embeddings
        for _ in range(self.n_mm_layers):
            propagated_items = torch.sparse.mm(
                self.mm_adj, propagated_items
            )
        return self.mm_output_projection(propagated_items)

    def _encode(self):
        initial_embeddings, multimodal_items = self._initial_node_embeddings()

        full_embeddings = self._propagate_ui_graph(
            self.norm_adj, initial_embeddings
        )
        masked_adj, interaction_mask = self._masked_ui_adjacency()
        masked_embeddings = self._propagate_ui_graph(
            masked_adj, initial_embeddings
        )

        branch_embeddings = torch.cat(
            (full_embeddings, masked_embeddings), dim=1
        )
        gate = torch.sigmoid(self.fusion_gate(branch_embeddings))
        fused_embeddings = (
            gate * full_embeddings
            + (1.0 - gate) * masked_embeddings
        )

        full_users, full_items = torch.split(
            full_embeddings, [self.n_users, self.n_items], dim=0
        )
        masked_users, masked_items = torch.split(
            masked_embeddings, [self.n_users, self.n_items], dim=0
        )
        fused_users, fused_items = torch.split(
            fused_embeddings, [self.n_users, self.n_items], dim=0
        )

        mm_items = self._propagate_mm_graph(multimodal_items)
        final_items = fused_items + mm_items

        return {
            'users': fused_users,
            'items': final_items,
            'full_users': full_users,
            'full_items': full_items,
            'masked_users': masked_users,
            'masked_items': masked_items,
            'mask': interaction_mask,
        }

    def forward(self):
        representations = self._encode()
        return representations['users'], representations['items']

    @staticmethod
    def bpr_loss(users, positive_items, negative_items):
        positive_scores = torch.sum(users * positive_items, dim=1)
        negative_scores = torch.sum(users * negative_items, dim=1)
        return -F.logsigmoid(positive_scores - negative_scores).mean()

    def info_nce(self, first_view, second_view):
        first_view = F.normalize(first_view, dim=1)
        second_view = F.normalize(second_view, dim=1)
        logits = torch.matmul(first_view, second_view.transpose(0, 1))
        logits = logits / self.cl_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )

    def calculate_loss(self, interaction):
        users = interaction[0]
        positive_items = interaction[1]
        negative_items = interaction[2]
        representations = self._encode()

        user_embeddings = representations['users'][users]
        positive_embeddings = representations['items'][positive_items]
        negative_embeddings = representations['items'][negative_items]
        ranking_loss = self.bpr_loss(
            user_embeddings, positive_embeddings, negative_embeddings
        )

        unique_users = torch.unique(users)
        unique_items = torch.unique(positive_items)
        user_cl_loss = self.info_nce(
            representations['full_users'][unique_users],
            representations['masked_users'][unique_users],
        )
        item_cl_loss = self.info_nce(
            representations['full_items'][unique_items],
            representations['masked_items'][unique_items],
        )
        contrastive_loss = 0.5 * (user_cl_loss + item_cl_loss)

        interaction_mask = representations['mask']
        if interaction_mask is None:
            mask_loss = ranking_loss.new_zeros(())
            mask_mean = ranking_loss.new_ones(())
        else:
            budget_loss = (
                interaction_mask.mean() - self.mask_keep_ratio
            ).pow(2)
            binary_loss = (
                interaction_mask * (1.0 - interaction_mask)
            ).mean()
            mask_loss = budget_loss + self.mask_binary_weight * binary_loss
            mask_mean = interaction_mask.mean()

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
        user_embeddings, item_embeddings = self.forward()
        batch_user_embeddings = user_embeddings[interaction[0]]
        return torch.matmul(
            batch_user_embeddings, item_embeddings.transpose(0, 1)
        )
