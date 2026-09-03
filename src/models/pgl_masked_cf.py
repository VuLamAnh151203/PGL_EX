r"""Counterfactual-guided edge and semantic-block masking for PGL."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pgl_masked import _config_value
from models.pgl_masked_ex import PGL_MASKED_EX


class PGL_MASKED_CF(PGL_MASKED_EX):
    """PGL with sampled edge- or semantic-block CF supervision."""

    CAUSAL_FEATURE_NAMES = (
        'visual_affinity',
        'textual_affinity',
        'normalized_log_user_degree',
        'normalized_log_item_degree',
        'full_uncertainty',
        'full_mask_disagreement',
    )

    def __init__(self, config, dataset):
        super(PGL_MASKED_CF, self).__init__(config, dataset)

        self.lambda_c = float(_config_value(config, 'lambda_c', 1.0))
        self.lambda_s = float(_config_value(config, 'lambda_s', 0.0))
        self.lambda_cf = float(_config_value(config, 'lambda_cf', 0.1))
        self.cf_samples_per_batch = int(
            _config_value(config, 'cf_samples_per_batch', 2)
        )
        self.cf_warmup_epochs = int(
            _config_value(config, 'cf_warmup_epochs', 10)
        )
        self.cf_hidden_dim = int(
            _config_value(config, 'cf_hidden_dim', 32)
        )
        self.cf_huber_beta = float(
            _config_value(config, 'cf_huber_beta', 0.1)
        )
        self.cf_rank_temperature = float(
            _config_value(config, 'cf_rank_temperature', 0.2)
        )
        self.cf_min_tau_gap = float(
            _config_value(config, 'cf_min_tau_gap', 1e-6)
        )
        self.cf_target_mode = str(
            _config_value(config, 'cf_target_mode', 'synergy')
        ).lower()
        self.cf_intervention_mode = str(
            _config_value(config, 'cf_intervention_mode', 'edge')
        ).lower()
        self.cf_block_num_prototypes = int(
            _config_value(config, 'cf_block_num_prototypes', 8)
        )
        self.cf_block_visual_weight = float(
            _config_value(config, 'cf_block_visual_weight', 0.5)
        )
        self.cf_block_min_edges = int(
            _config_value(config, 'cf_block_min_edges', 2)
        )
        self.cf_block_queries_per_target = int(
            _config_value(config, 'cf_block_queries_per_target', 10)
        )
        self.cf_block_full_temperature = float(
            _config_value(config, 'cf_block_full_temperature', 1.0)
        )
        self.cf_block_kmeans_seed = int(
            _config_value(config, 'cf_block_kmeans_seed', 999)
        )
        self.cf_block_kmeans_iterations = int(
            _config_value(config, 'cf_block_kmeans_iterations', 25)
        )
        self.cf_block_kmeans_tolerance = float(
            _config_value(config, 'cf_block_kmeans_tolerance', 1e-4)
        )

        if self.ui_branch_mode != 'dual':
            raise ValueError(
                "PGL_MASKED_CF only supports ui_branch_mode='dual'."
            )
        if self.mask_graph_mode != 'hard':
            raise ValueError(
                "PGL_MASKED_CF only supports mask_graph_mode='hard'."
            )
        self._validate_counterfactual_config()

        # Keep the original edge-specific residual. It is optimized through
        # the BPR/mask path, whereas the detached causal correction is learned
        # only from the counterfactual losses.
        self.mask_logits.requires_grad_(True)

        forward_edges = self.ui_edge_index[:, :self.num_interactions]
        edge_users = forward_edges[0]
        edge_items = forward_edges[1] - self.n_users
        user_degrees = torch.bincount(
            edge_users, minlength=self.n_users
        )
        item_degrees = torch.bincount(
            edge_items, minlength=self.n_items
        )

        edge_user_degrees = user_degrees[edge_users]
        edge_item_degrees = item_degrees[edge_items]
        normalized_log_user_degree = self._normalized_log_degree(
            edge_user_degrees
        )
        normalized_log_item_degree = self._normalized_log_degree(
            edge_item_degrees
        )
        self.register_buffer(
            'edge_user_degree', edge_user_degrees, persistent=True
        )
        self.register_buffer(
            'edge_item_degree', edge_item_degrees, persistent=True
        )
        self.register_buffer(
            'user_interaction_degree', user_degrees, persistent=True
        )
        self.register_buffer(
            'item_interaction_degree', item_degrees, persistent=True
        )
        self.register_buffer(
            'normalized_log_user_degree',
            normalized_log_user_degree,
            persistent=True,
        )
        self.register_buffer(
            'normalized_log_item_degree',
            normalized_log_item_degree,
            persistent=True,
        )

        # CSR-style user-to-edge lookup.  user_edge_ids preserves the global
        # edge IDs used by semantic affinities and mask scores.
        user_edge_ids = torch.argsort(edge_users, stable=True)
        user_edge_ptr = torch.zeros(
            self.n_users + 1, dtype=torch.long
        )
        user_edge_ptr[1:] = torch.cumsum(user_degrees, dim=0)
        self.register_buffer(
            'user_edge_ptr', user_edge_ptr, persistent=True
        )
        self.register_buffer(
            'user_edge_ids', user_edge_ids, persistent=True
        )

        if self.cf_intervention_mode == 'semantic_block':
            self._initialize_semantic_blocks(edge_users, edge_items)
        else:
            self._register_empty_semantic_blocks()

        self.register_buffer(
            'full_uncertainty',
            torch.ones(self.num_interactions, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            'full_mask_disagreement',
            torch.zeros(self.num_interactions, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            'observed_tau_syn_sum',
            torch.zeros(self.num_interactions, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            'observed_tau_syn_count',
            torch.zeros(self.num_interactions, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            'observed_block_target_sum',
            torch.zeros(self.num_semantic_blocks, dtype=torch.float32),
            persistent=self.cf_intervention_mode == 'semantic_block',
        )
        self.register_buffer(
            'observed_block_target_count',
            torch.zeros(self.num_semantic_blocks, dtype=torch.long),
            persistent=self.cf_intervention_mode == 'semantic_block',
        )
        self.register_buffer(
            'observed_block_add_effect_sum',
            torch.zeros(self.num_semantic_blocks, dtype=torch.float32),
            persistent=self.cf_intervention_mode == 'semantic_block',
        )
        self.register_buffer(
            'observed_block_remove_effect_sum',
            torch.zeros(self.num_semantic_blocks, dtype=torch.float32),
            persistent=self.cf_intervention_mode == 'semantic_block',
        )
        self.register_buffer(
            'observed_block_query_count',
            torch.zeros(self.num_semantic_blocks, dtype=torch.long),
            persistent=self.cf_intervention_mode == 'semantic_block',
        )
        self.register_buffer(
            'cf_epoch', torch.tensor(-1, dtype=torch.long), persistent=True
        )

        self.causal_scorer = nn.Sequential(
            nn.Linear(len(self.CAUSAL_FEATURE_NAMES), self.cf_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cf_hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.causal_scorer[0].weight)
        nn.init.zeros_(self.causal_scorer[0].bias)
        # Initial g_phi(e)=0 leaves the first mask to theta_e and, when
        # enabled, the optional semantic prior.
        nn.init.zeros_(self.causal_scorer[2].weight)
        nn.init.zeros_(self.causal_scorer[2].bias)

        self._last_cf_propagation_count = 0
        self._last_block_add_effects = None
        self._last_block_remove_effects = None

    def _validate_counterfactual_config(self):
        finite_non_negative = {
            'lambda_c': self.lambda_c,
            'lambda_s': self.lambda_s,
            'lambda_cf': self.lambda_cf,
            'cf_min_tau_gap': self.cf_min_tau_gap,
        }
        for name, value in finite_non_negative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    '{} must be a finite non-negative value.'.format(name)
                )
        if self.cf_samples_per_batch <= 0:
            raise ValueError('cf_samples_per_batch must be positive.')
        if self.cf_warmup_epochs < 0:
            raise ValueError('cf_warmup_epochs must be non-negative.')
        if self.cf_hidden_dim <= 0:
            raise ValueError('cf_hidden_dim must be positive.')
        if self.cf_target_mode not in {'synergy', 'fused_effect'}:
            raise ValueError(
                "cf_target_mode must be 'synergy' or 'fused_effect'."
            )
        if self.cf_intervention_mode not in {'edge', 'semantic_block'}:
            raise ValueError(
                "cf_intervention_mode must be 'edge' or "
                "'semantic_block'."
            )
        if (
            self.cf_intervention_mode == 'semantic_block'
            and self.cf_target_mode != 'fused_effect'
        ):
            raise ValueError(
                "semantic_block intervention requires "
                "cf_target_mode='fused_effect'."
            )
        if self.cf_intervention_mode == 'semantic_block':
            if not 2 <= self.cf_block_num_prototypes <= self.n_items:
                raise ValueError(
                    'cf_block_num_prototypes must be between 2 and '
                    'n_items.'
                )
            if (
                not math.isfinite(self.cf_block_visual_weight)
                or not 0.0 <= self.cf_block_visual_weight <= 1.0
            ):
                raise ValueError(
                    'cf_block_visual_weight must be finite and in [0, 1].'
                )
            if self.cf_block_min_edges <= 0:
                raise ValueError('cf_block_min_edges must be positive.')
            if self.cf_block_queries_per_target <= 0:
                raise ValueError(
                    'cf_block_queries_per_target must be positive.'
                )
            if (
                not math.isfinite(self.cf_block_full_temperature)
                or self.cf_block_full_temperature <= 0.0
            ):
                raise ValueError(
                    'cf_block_full_temperature must be finite and '
                    'positive.'
                )
            if self.cf_block_kmeans_iterations <= 0:
                raise ValueError(
                    'cf_block_kmeans_iterations must be positive.'
                )
            if (
                not math.isfinite(self.cf_block_kmeans_tolerance)
                or self.cf_block_kmeans_tolerance < 0.0
            ):
                raise ValueError(
                    'cf_block_kmeans_tolerance must be finite and '
                    'non-negative.'
                )
        if not math.isfinite(self.cf_huber_beta) or self.cf_huber_beta <= 0:
            raise ValueError('cf_huber_beta must be finite and positive.')
        if (
            not math.isfinite(self.cf_rank_temperature)
            or self.cf_rank_temperature <= 0
        ):
            raise ValueError(
                'cf_rank_temperature must be finite and positive.'
            )

    def _register_empty_semantic_blocks(self):
        self.num_semantic_blocks = 0
        empty_long = torch.empty(0, dtype=torch.long)
        self.register_buffer(
            'item_prototype_ids', empty_long.clone(), persistent=False
        )
        self.register_buffer(
            'edge_block_ids', empty_long.clone(), persistent=False
        )
        self.register_buffer(
            'block_edge_ptr', torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'block_edge_ids', empty_long.clone(), persistent=False
        )
        self.register_buffer(
            'block_user_ids', empty_long.clone(), persistent=False
        )
        self.register_buffer(
            'block_prototype_ids', empty_long.clone(), persistent=False
        )
        self.register_buffer(
            'block_sizes', empty_long.clone(), persistent=False
        )
        self.register_buffer(
            'block_eligible_for_intervention',
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            'user_block_ptr',
            torch.zeros(self.n_users + 1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'user_block_ids', empty_long.clone(), persistent=False
        )
        self.register_buffer(
            'user_eligible_block_count',
            torch.zeros(self.n_users, dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def _semantic_item_features(visual_features, textual_features, weight):
        visual = F.normalize(
            visual_features.to(dtype=torch.float32),
            p=2,
            dim=1,
            eps=1e-12,
        )
        textual = F.normalize(
            textual_features.to(dtype=torch.float32),
            p=2,
            dim=1,
            eps=1e-12,
        )
        combined = torch.cat(
            (
                math.sqrt(weight) * visual,
                math.sqrt(1.0 - weight) * textual,
            ),
            dim=1,
        )
        return F.normalize(combined, p=2, dim=1, eps=1e-12)

    @staticmethod
    def _spherical_kmeans(
        features,
        num_clusters,
        seed,
        max_iterations,
        tolerance,
    ):
        """Deterministic cosine K-means with K-means++ initialization."""
        if features.ndim != 2 or features.size(0) < num_clusters:
            raise ValueError(
                'features must contain at least num_clusters rows.'
            )
        features = F.normalize(
            features.detach().cpu().to(dtype=torch.float32),
            p=2,
            dim=1,
            eps=1e-12,
        )
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(seed))
        num_items = features.size(0)

        first = int(torch.randint(
            num_items, (1,), generator=generator
        ).item())
        centroid_rows = [first]
        closest_distance = (
            1.0 - torch.mv(features, features[first])
        ).clamp_min(0.0)

        for _ in range(1, num_clusters):
            probabilities = closest_distance.clone()
            probabilities[torch.tensor(centroid_rows)] = 0.0
            if float(probabilities.sum().item()) <= 1e-12:
                available = torch.ones(num_items, dtype=torch.bool)
                available[torch.tensor(centroid_rows)] = False
                next_row = int(torch.nonzero(
                    available, as_tuple=False
                )[0].item())
            else:
                next_row = int(torch.multinomial(
                    probabilities,
                    1,
                    generator=generator,
                ).item())
            centroid_rows.append(next_row)
            distance = (
                1.0 - torch.mv(features, features[next_row])
            ).clamp_min(0.0)
            closest_distance = torch.minimum(
                closest_distance, distance
            )

        centroids = features[torch.tensor(centroid_rows)].clone()
        previous_assignments = None
        for _ in range(max_iterations):
            similarities = torch.matmul(features, centroids.t())
            assignments = torch.argmax(similarities, dim=1)
            new_centroids = torch.zeros_like(centroids)
            new_centroids.index_add_(0, assignments, features)
            counts = torch.bincount(
                assignments, minlength=num_clusters
            )

            empty_clusters = torch.nonzero(
                counts == 0, as_tuple=False
            ).flatten()
            if empty_clusters.numel() > 0:
                farthest_rows = torch.argsort(
                    similarities.max(dim=1).values
                )
                used_rows = set()
                cursor = 0
                for cluster in empty_clusters.tolist():
                    while int(farthest_rows[cursor]) in used_rows:
                        cursor += 1
                    row = int(farthest_rows[cursor])
                    used_rows.add(row)
                    new_centroids[cluster] = features[row]
                    cursor += 1

            new_centroids = F.normalize(
                new_centroids, p=2, dim=1, eps=1e-12
            )
            centroid_shift = (
                1.0 - (centroids * new_centroids).sum(dim=1)
            ).abs().max()
            assignments_stable = (
                previous_assignments is not None
                and torch.equal(assignments, previous_assignments)
            )
            centroids = new_centroids
            previous_assignments = assignments
            if assignments_stable or float(centroid_shift) <= tolerance:
                break

        return torch.argmax(
            torch.matmul(features, centroids.t()), dim=1
        )

    def _initialize_semantic_blocks(self, edge_users, edge_items):
        with torch.no_grad():
            item_features = self._semantic_item_features(
                self.v_feat.detach().cpu(),
                self.t_feat.detach().cpu(),
                self.cf_block_visual_weight,
            )
            item_prototypes = self._spherical_kmeans(
                item_features,
                self.cf_block_num_prototypes,
                self.cf_block_kmeans_seed,
                self.cf_block_kmeans_iterations,
                self.cf_block_kmeans_tolerance,
            )

        users = edge_users.detach().cpu().to(dtype=torch.long)
        items = edge_items.detach().cpu().to(dtype=torch.long)
        edge_prototypes = item_prototypes[items]
        block_keys = (
            users * self.cf_block_num_prototypes + edge_prototypes
        )
        unique_keys, edge_block_ids, block_sizes = torch.unique(
            block_keys,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        self.num_semantic_blocks = int(unique_keys.numel())
        block_user_ids = torch.div(
            unique_keys,
            self.cf_block_num_prototypes,
            rounding_mode='floor',
        )
        block_prototype_ids = (
            unique_keys % self.cf_block_num_prototypes
        )
        block_edge_ids = torch.argsort(edge_block_ids, stable=True)
        block_edge_ptr = torch.zeros(
            self.num_semantic_blocks + 1, dtype=torch.long
        )
        block_edge_ptr[1:] = torch.cumsum(block_sizes, dim=0)

        user_degrees = torch.bincount(users, minlength=self.n_users)
        eligible = (
            (block_sizes >= self.cf_block_min_edges)
            & (user_degrees[block_user_ids] < self.n_items)
        )

        user_block_counts = torch.bincount(
            block_user_ids, minlength=self.n_users
        )
        user_block_ptr = torch.zeros(
            self.n_users + 1, dtype=torch.long
        )
        user_block_ptr[1:] = torch.cumsum(user_block_counts, dim=0)
        user_block_ids = torch.arange(
            self.num_semantic_blocks, dtype=torch.long
        )
        eligible_counts = torch.zeros(self.n_users, dtype=torch.long)
        eligible_counts.index_add_(
            0, block_user_ids, eligible.to(dtype=torch.long)
        )

        self.register_buffer(
            'item_prototype_ids', item_prototypes, persistent=True
        )
        self.register_buffer(
            'edge_block_ids', edge_block_ids, persistent=True
        )
        self.register_buffer(
            'block_edge_ptr', block_edge_ptr, persistent=True
        )
        self.register_buffer(
            'block_edge_ids', block_edge_ids, persistent=True
        )
        self.register_buffer(
            'block_user_ids', block_user_ids, persistent=True
        )
        self.register_buffer(
            'block_prototype_ids', block_prototype_ids, persistent=True
        )
        self.register_buffer(
            'block_sizes', block_sizes, persistent=True
        )
        self.register_buffer(
            'block_eligible_for_intervention', eligible, persistent=True
        )
        self.register_buffer(
            'user_block_ptr', user_block_ptr, persistent=True
        )
        self.register_buffer(
            'user_block_ids', user_block_ids, persistent=True
        )
        self.register_buffer(
            'user_eligible_block_count', eligible_counts, persistent=True
        )

    @staticmethod
    def _normalized_log_degree(edge_degrees):
        values = torch.log1p(edge_degrees.to(dtype=torch.float32))
        return values / values.max().clamp_min(1e-12)

    @property
    def counterfactual_training_active(self):
        return int(self.cf_epoch.item()) >= self.cf_warmup_epochs

    def causal_edge_features(self, edge_ids=None):
        features = torch.stack(
            (
                self.semantic_visual_affinity,
                self.semantic_textual_affinity,
                self.normalized_log_user_degree,
                self.normalized_log_item_degree,
                self.full_uncertainty,
                self.full_mask_disagreement,
            ),
            dim=1,
        )
        if edge_ids is not None:
            features = features[edge_ids]
        return features.detach()

    def causal_block_features(self, block_ids=None):
        if self.cf_intervention_mode != 'semantic_block':
            return self.mask_logits.new_empty(
                (0, len(self.CAUSAL_FEATURE_NAMES))
            )
        edge_features = self.causal_edge_features()
        pooled = edge_features.new_zeros(
            (self.num_semantic_blocks, edge_features.size(1))
        )
        pooled.index_add_(0, self.edge_block_ids, edge_features)
        pooled = pooled / self.block_sizes.to(
            dtype=pooled.dtype
        ).unsqueeze(1).clamp_min(1.0)
        if block_ids is not None:
            pooled = pooled[block_ids]
        return pooled.detach()

    def block_semantic_prior(self, block_ids=None):
        if self.cf_intervention_mode != 'semantic_block':
            return self.mask_logits.new_empty(0)
        edge_prior = self.semantic_prior()
        pooled = edge_prior.new_zeros(self.num_semantic_blocks)
        pooled.index_add_(0, self.edge_block_ids, edge_prior)
        pooled = pooled / self.block_sizes.to(
            dtype=pooled.dtype
        ).clamp_min(1.0)
        if block_ids is not None:
            pooled = pooled[block_ids]
        return pooled

    def block_residual_logit(self, block_ids=None):
        """Mean edge-specific residual for each semantic block."""
        if self.cf_intervention_mode != 'semantic_block':
            return self.mask_logits.new_empty(0)
        pooled = self.mask_logits.new_zeros(self.num_semantic_blocks)
        pooled.index_add_(0, self.edge_block_ids, self.mask_logits)
        pooled = pooled / self.block_sizes.to(
            dtype=pooled.dtype
        ).clamp_min(1.0)
        if block_ids is not None:
            pooled = pooled[block_ids]
        return pooled

    def block_causal_score(self, block_ids=None):
        return self.causal_scorer(
            self.causal_block_features(block_ids)
        ).squeeze(1)

    def causal_score(self, edge_ids=None):
        if self.cf_intervention_mode == 'semantic_block':
            block_scores = self.block_causal_score()
            edge_scores = block_scores[self.edge_block_ids]
            if edge_ids is not None:
                edge_scores = edge_scores[edge_ids]
            return edge_scores
        return self.causal_scorer(
            self.causal_edge_features(edge_ids)
        ).squeeze(1)

    def _causal_unit_score(self, unit_ids):
        if self.cf_intervention_mode == 'semantic_block':
            return self.block_causal_score(unit_ids)
        return self.causal_score(unit_ids)

    def effective_mask_logits(self):
        """Mask score used by BPR; causal scorer is detached on this path."""
        return (
            self.mask_logits
            + self.lambda_s * self.semantic_prior()
            + self.lambda_c * self.causal_score().detach()
        )

    def causal_training_logits(self, unit_ids):
        """Non-detached score used only by counterfactual supervision."""
        if self.cf_intervention_mode == 'semantic_block':
            return (
                self.block_residual_logit(unit_ids).detach()
                + self.lambda_s
                * self.block_semantic_prior(unit_ids).detach()
                + self.lambda_c * self.block_causal_score(unit_ids)
            )
        return (
            self.mask_logits[unit_ids].detach()
            + self.lambda_s * self.semantic_prior()[unit_ids].detach()
            + self.lambda_c * self.causal_score(unit_ids)
        )

    @torch.no_grad()
    def pre_epoch_processing(self):
        self.cf_epoch.add_(1)
        self.hard_train_indices = self._sample_hard_train_indices(
            self.effective_mask_logits()
        )

    @torch.no_grad()
    def post_epoch_processing(self):
        was_training = self.training
        self.eval()

        # First obtain branch representations with the previous epoch's
        # detached dynamic features, then refresh those features and select
        # the deterministic evaluation mask once more.
        self.hard_eval_indices = self._select_hard_eval_indices(
            self.effective_mask_logits()
        )
        representations = self._encode()
        self._update_dynamic_edge_features(representations)
        self.hard_eval_indices = self._select_hard_eval_indices(
            self.effective_mask_logits()
        )

        if was_training:
            self.train()
        return 'CF epoch {}, counterfactual supervision {}'.format(
            int(self.cf_epoch.item()),
            'active' if self.counterfactual_training_active else 'warm-up',
        )

    @torch.no_grad()
    def _update_dynamic_edge_features(self, representations):
        forward_edges = self.ui_edge_index[:, :self.num_interactions]
        edge_users = forward_edges[0]
        edge_items = forward_edges[1] - self.n_users

        full_similarity = F.cosine_similarity(
            representations['full_users'][edge_users],
            representations['full_items'][edge_items],
            dim=1,
            eps=1e-12,
        )
        masked_similarity = F.cosine_similarity(
            representations['masked_users'][edge_users],
            representations['masked_items'][edge_items],
            dim=1,
            eps=1e-12,
        )
        uncertainty = (1.0 - full_similarity.abs()).clamp(0.0, 1.0)
        disagreement = (
            0.5 * (full_similarity - masked_similarity).abs()
        ).clamp(0.0, 1.0)
        self.full_uncertainty.copy_(
            torch.nan_to_num(uncertainty, nan=1.0)
        )
        self.full_mask_disagreement.copy_(
            torch.nan_to_num(disagreement, nan=0.0)
        )

    @torch.no_grad()
    def _sample_edge_counterfactual_candidates(
        self, users, positive_items, negative_items
    ):
        eligible_rows = torch.nonzero(
            self.user_interaction_degree[users] >= 2,
            as_tuple=False,
        ).flatten()
        if eligible_rows.numel() == 0:
            empty = users.new_empty(0)
            return empty, empty, empty, empty

        order = eligible_rows[
            torch.randperm(eligible_rows.numel(), device=users.device)
        ]
        selected_rows = []
        selected_edges = []
        used_edges = set()
        forward_items = (
            self.ui_edge_index[1, :self.num_interactions] - self.n_users
        )

        for row_tensor in order:
            row = int(row_tensor.item())
            user = int(users[row].item())
            positive_item = positive_items[row]
            start = int(self.user_edge_ptr[user].item())
            stop = int(self.user_edge_ptr[user + 1].item())
            choices = self.user_edge_ids[start:stop]
            choices = choices[forward_items[choices] != positive_item]
            if used_edges:
                unused_mask = torch.tensor(
                    [int(edge.item()) not in used_edges for edge in choices],
                    dtype=torch.bool,
                    device=choices.device,
                )
                choices = choices[unused_mask]
            if choices.numel() == 0:
                continue

            choice_index = torch.randint(
                choices.numel(), (1,), device=choices.device
            )
            edge_id = int(choices[choice_index].item())
            used_edges.add(edge_id)
            selected_rows.append(row)
            selected_edges.append(edge_id)
            if len(selected_edges) >= self.cf_samples_per_batch:
                break

        if not selected_edges:
            empty = users.new_empty(0)
            return empty, empty, empty, empty

        rows = torch.tensor(
            selected_rows, dtype=torch.long, device=users.device
        )
        edge_ids = torch.tensor(
            selected_edges, dtype=torch.long, device=users.device
        )
        return (
            edge_ids,
            users[rows],
            positive_items[rows],
            negative_items[rows],
        )

    @torch.no_grad()
    def _sample_block_queries(self, block_id, user):
        block_id = int(block_id)
        user = int(user)
        history_start = int(self.user_edge_ptr[user].item())
        history_stop = int(self.user_edge_ptr[user + 1].item())
        history_edges = self.user_edge_ids[history_start:history_stop]
        forward_items = (
            self.ui_edge_index[1, :self.num_interactions] - self.n_users
        )
        # A positive may belong to the intervened block. Its own training
        # edge is excluded from the treatment later, so the query cannot
        # benefit trivially from propagating through itself.
        positive_pool = forward_items[history_edges]
        if positive_pool.numel() == 0:
            empty = history_edges.new_empty(0)
            return empty, empty

        query_count = self.cf_block_queries_per_target
        if positive_pool.numel() >= query_count:
            positive_indices = torch.randperm(
                positive_pool.numel(), device=positive_pool.device
            )[:query_count]
        else:
            positive_indices = torch.randint(
                positive_pool.numel(),
                (query_count,),
                device=positive_pool.device,
            )
        positives = positive_pool[positive_indices]

        history_items = forward_items[history_edges]
        negative_parts = []
        sampled = 0
        while sampled < query_count:
            proposal_count = max(8, 2 * (query_count - sampled))
            proposals = torch.randint(
                self.n_items,
                (proposal_count,),
                device=history_items.device,
            )
            valid = proposals[~torch.isin(proposals, history_items)]
            if valid.numel() > 0:
                negative_parts.append(valid)
                sampled += valid.numel()
        negatives = torch.cat(negative_parts)[:query_count]
        return positives, negatives

    @torch.no_grad()
    def _sample_block_counterfactual_candidates(self, users):
        eligible_rows = torch.nonzero(
            self.user_eligible_block_count[users] > 0,
            as_tuple=False,
        ).flatten()
        if eligible_rows.numel() == 0:
            empty = users.new_empty(0)
            empty_queries = users.new_empty(
                (0, self.cf_block_queries_per_target)
            )
            return empty, empty, empty_queries, empty_queries

        order = eligible_rows[torch.randperm(
            eligible_rows.numel(), device=users.device
        )]
        selected_blocks = []
        selected_users = []
        positive_queries = []
        negative_queries = []
        used_blocks = set()

        for row_tensor in order:
            user = int(users[int(row_tensor.item())].item())
            start = int(self.user_block_ptr[user].item())
            stop = int(self.user_block_ptr[user + 1].item())
            choices = self.user_block_ids[start:stop]
            choices = choices[
                self.block_eligible_for_intervention[choices]
            ]
            if used_blocks:
                choices = choices[torch.tensor(
                    [
                        int(block.item()) not in used_blocks
                        for block in choices
                    ],
                    dtype=torch.bool,
                    device=choices.device,
                )]
            if choices.numel() == 0:
                continue

            choice = choices[torch.randint(
                choices.numel(), (1,), device=choices.device
            )]
            block_id = int(choice.item())
            positives, negatives = self._sample_block_queries(
                block_id, user
            )
            if positives.numel() == 0:
                continue

            used_blocks.add(block_id)
            selected_blocks.append(block_id)
            selected_users.append(user)
            positive_queries.append(positives)
            negative_queries.append(negatives)
            if len(selected_blocks) >= self.cf_samples_per_batch:
                break

        if not selected_blocks:
            empty = users.new_empty(0)
            empty_queries = users.new_empty(
                (0, self.cf_block_queries_per_target)
            )
            return empty, empty, empty_queries, empty_queries

        return (
            torch.tensor(
                selected_blocks, dtype=torch.long, device=users.device
            ),
            torch.tensor(
                selected_users, dtype=torch.long, device=users.device
            ),
            torch.stack(positive_queries),
            torch.stack(negative_queries),
        )

    @torch.no_grad()
    def _sample_counterfactual_candidates(
        self, users, positive_items, negative_items
    ):
        if self.cf_intervention_mode == 'semantic_block':
            return self._sample_block_counterfactual_candidates(users)
        return self._sample_edge_counterfactual_candidates(
            users, positive_items, negative_items
        )

    def _forced_interaction_indices(self, edge_id, keep):
        base_indices = self.hard_train_indices
        edge_id = edge_id.to(
            device=base_indices.device, dtype=torch.long
        ).reshape(())
        is_present = torch.any(base_indices == edge_id)
        if keep:
            if bool(is_present.item()):
                return base_indices
            return torch.cat((base_indices, edge_id.reshape(1)), dim=0)
        return base_indices[base_indices != edge_id]

    def _forced_block_interaction_indices(
        self, block_id, keep, excluded_edge_id=None
    ):
        base_indices = self.hard_train_indices
        block_id = int(block_id.item()) if torch.is_tensor(block_id) else int(
            block_id
        )
        start = int(self.block_edge_ptr[block_id].item())
        stop = int(self.block_edge_ptr[block_id + 1].item())
        block_edges = self.block_edge_ids[start:stop]
        if excluded_edge_id is not None:
            excluded_edge_id = (
                int(excluded_edge_id.item())
                if torch.is_tensor(excluded_edge_id)
                else int(excluded_edge_id)
            )
            if excluded_edge_id >= 0:
                block_edges = block_edges[
                    block_edges != excluded_edge_id
                ]
        base_in_block = torch.isin(base_indices, block_edges)
        if not keep:
            return base_indices[~base_in_block]
        missing_edges = block_edges[~torch.isin(block_edges, base_indices)]
        if missing_edges.numel() == 0:
            return base_indices
        return torch.cat((base_indices, missing_edges), dim=0)

    def _block_query_excluded_edge_ids(
        self, block_id, positive_items
    ):
        """Return the edge (u, p) to leave out for each block query."""
        block_id = int(block_id)
        start = int(self.block_edge_ptr[block_id].item())
        stop = int(self.block_edge_ptr[block_id + 1].item())
        block_edges = self.block_edge_ids[start:stop]
        block_items = (
            self.ui_edge_index[1, block_edges] - self.n_users
        )
        matches = positive_items.unsqueeze(1) == block_items.unsqueeze(0)
        excluded = torch.full_like(positive_items, -1)
        has_match = matches.any(dim=1)
        if torch.any(has_match):
            member_positions = matches[has_match].to(
                dtype=torch.long
            ).argmax(dim=1)
            excluded[has_match] = block_edges[member_positions]
        return excluded

    def _counterfactual_adjacency(self, kept_interactions):
        reverse_interactions = kept_interactions + self.num_interactions
        kept_undirected = torch.cat(
            (kept_interactions, reverse_interactions), dim=0
        )
        edge_index = self.ui_edge_index[:, kept_undirected]
        if self.mask_degree_mode == 'full':
            edge_weights = self.full_norm_edge_weights[kept_undirected]
            return self._ui_adjacency_from_weights(
                edge_weights, edge_index
            )

        edge_weights = torch.ones(
            kept_undirected.numel(),
            dtype=self.full_norm_edge_weights.dtype,
            device=kept_undirected.device,
        )
        return self._normalized_ui_adjacency(edge_weights, edge_index)

    def _counterfactual_margin(
        self,
        full_nodes,
        masked_nodes,
        mm_items,
        user,
        positive_item,
        negative_item,
    ):
        node_ids = torch.stack(
            (
                user,
                positive_item + self.n_users,
                negative_item + self.n_users,
            )
        )
        selected_full = full_nodes[node_ids]
        selected_masked = masked_nodes[node_ids]
        fused, _ = self._fuse_ui_branches(
            selected_full, selected_masked
        )

        fused_user = fused[0]
        fused_positive = fused[1] + mm_items[positive_item]
        fused_negative = fused[2] + mm_items[negative_item]
        fused_margin = torch.sum(fused_user * fused_positive) - torch.sum(
            fused_user * fused_negative
        )

        if self.cf_target_mode == 'fused_effect':
            return fused_margin, None

        masked_user = selected_masked[0]
        masked_positive = selected_masked[1] + mm_items[positive_item]
        masked_negative = selected_masked[2] + mm_items[negative_item]
        masked_margin = torch.sum(
            masked_user * masked_positive
        ) - torch.sum(masked_user * masked_negative)
        return fused_margin, masked_margin

    def _counterfactual_fused_margins(
        self,
        full_nodes,
        masked_nodes,
        mm_items,
        user,
        positive_items,
        negative_items,
    ):
        user_node = user.reshape(())
        positive_nodes = positive_items + self.n_users
        negative_nodes = negative_items + self.n_users

        full_user = full_nodes[user_node]
        masked_user = masked_nodes[user_node]
        fused_user, _ = self._fuse_ui_branches(
            full_user, masked_user
        )

        full_positive = full_nodes[positive_nodes]
        masked_positive = masked_nodes[positive_nodes]
        fused_positive, _ = self._fuse_ui_branches(
            full_positive, masked_positive
        )
        fused_positive = fused_positive + mm_items[positive_items]

        full_negative = full_nodes[negative_nodes]
        masked_negative = masked_nodes[negative_nodes]
        fused_negative, _ = self._fuse_ui_branches(
            full_negative, masked_negative
        )
        fused_negative = fused_negative + mm_items[negative_items]
        return (
            (fused_positive * fused_user).sum(dim=1)
            - (fused_negative * fused_user).sum(dim=1)
        )

    @staticmethod
    def _full_only_margins(
        full_nodes,
        mm_items,
        num_users,
        user,
        positive_items,
        negative_items,
    ):
        full_user = full_nodes[user.reshape(())]
        full_positive = (
            full_nodes[positive_items + num_users]
            + mm_items[positive_items]
        )
        full_negative = (
            full_nodes[negative_items + num_users]
            + mm_items[negative_items]
        )
        return (
            (full_positive * full_user).sum(dim=1)
            - (full_negative * full_user).sum(dim=1)
        )

    @staticmethod
    def _weighted_block_effect(
        full_margins,
        fused_plus_margins,
        fused_minus_margins,
        temperature,
    ):
        weights = torch.sigmoid(-full_margins / temperature)
        outcome_difference = (
            F.logsigmoid(fused_plus_margins)
            - F.logsigmoid(fused_minus_margins)
        )
        return (
            (weights * outcome_difference).sum()
            / weights.sum().clamp_min(1e-12)
        )

    @staticmethod
    def _weighted_block_effect_components(
        full_margins,
        fused_plus_margins,
        fused_base_margins,
        fused_minus_margins,
        temperature,
    ):
        """Return total, add and remove effects on log-sigmoid outcomes."""
        weights = torch.sigmoid(-full_margins / temperature)
        denominator = weights.sum().clamp_min(1e-12)
        plus_outcome = F.logsigmoid(fused_plus_margins)
        base_outcome = F.logsigmoid(fused_base_margins)
        minus_outcome = F.logsigmoid(fused_minus_margins)
        add_effect = (
            weights * (plus_outcome - base_outcome)
        ).sum() / denominator
        remove_effect = (
            weights * (base_outcome - minus_outcome)
        ).sum() / denominator
        return add_effect + remove_effect, add_effect, remove_effect

    @staticmethod
    def _synergy_from_outcomes(
        fused_plus, fused_minus, masked_plus, masked_minus
    ):
        fused_effect = fused_plus - fused_minus
        masked_effect = masked_plus - masked_minus
        return fused_effect - masked_effect

    def _counterfactual_target_from_outcomes(
        self,
        fused_plus,
        fused_minus,
        masked_plus,
        masked_minus,
    ):
        """Return the configured intervention target.

        ``fused_effect`` directly measures the keep/remove effect on the
        deployed Full+Mask scoring path. ``synergy`` retains the original
        difference-in-differences target by subtracting the Mask-only effect.
        """
        if self.cf_target_mode == 'fused_effect':
            return fused_plus - fused_minus
        return self._synergy_from_outcomes(
            fused_plus,
            fused_minus,
            masked_plus,
            masked_minus,
        )

    def _encode_with_counterfactual_context(self):
        """Encode once and retain tensors shared by all interventions."""
        (
            full_initial,
            masked_initial,
            multimodal_items,
            _,
            _,
        ) = self._initial_node_embeddings()
        masked_adj, interaction_mask = self._masked_ui_adjacency()
        masked_nodes = self._propagate_ui_graph(
            masked_adj, masked_initial
        )
        full_nodes = self._propagate_ui_graph(
            self.norm_adj, full_initial
        )

        fused_nodes, _ = self._fuse_ui_branches(
            full_nodes, masked_nodes
        )
        full_users, full_items = torch.split(
            full_nodes, [self.n_users, self.n_items], dim=0
        )
        masked_users, masked_items = torch.split(
            masked_nodes, [self.n_users, self.n_items], dim=0
        )
        fused_users, fused_items = torch.split(
            fused_nodes, [self.n_users, self.n_items], dim=0
        )
        mm_items = self._propagate_mm_graph(multimodal_items)

        representations = {
            'users': fused_users,
            'items': fused_items + mm_items,
            'full_users': full_users,
            'full_items': full_items,
            'masked_users': masked_users,
            'masked_items': masked_items,
            'mask': interaction_mask,
            'second_mask': None,
        }
        context = {
            'masked_initial': masked_initial,
            'mm_items': mm_items,
        }
        return representations, context

    @torch.no_grad()
    def _counterfactual_targets(
        self,
        representations,
        counterfactual_context,
        edge_ids,
        users,
        positive_items,
        negative_items,
    ):
        if edge_ids.numel() == 0:
            return self.mask_logits.new_empty(0)

        full_nodes = torch.cat(
            (
                representations['full_users'].detach(),
                representations['full_items'].detach(),
            ),
            dim=0,
        )
        masked_initial = counterfactual_context[
            'masked_initial'
        ].detach()
        mm_items = counterfactual_context['mm_items'].detach()

        targets = []
        self._last_cf_propagation_count = 0
        for edge_id, user, positive_item, negative_item in zip(
            edge_ids, users, positive_items, negative_items
        ):
            plus_indices = self._forced_interaction_indices(
                edge_id, keep=True
            )
            minus_indices = self._forced_interaction_indices(
                edge_id, keep=False
            )
            plus_nodes = self._propagate_ui_graph(
                self._counterfactual_adjacency(plus_indices),
                masked_initial,
            )
            minus_nodes = self._propagate_ui_graph(
                self._counterfactual_adjacency(minus_indices),
                masked_initial,
            )
            self._last_cf_propagation_count += 2

            fused_plus, masked_plus = self._counterfactual_margin(
                full_nodes,
                plus_nodes,
                mm_items,
                user,
                positive_item,
                negative_item,
            )
            fused_minus, masked_minus = self._counterfactual_margin(
                full_nodes,
                minus_nodes,
                mm_items,
                user,
                positive_item,
                negative_item,
            )
            targets.append(
                self._counterfactual_target_from_outcomes(
                    fused_plus,
                    fused_minus,
                    masked_plus,
                    masked_minus,
                )
            )

        tau_syn = torch.stack(targets).detach()
        self.observed_tau_syn_sum.index_add_(
            0, edge_ids, tau_syn.to(self.observed_tau_syn_sum.dtype)
        )
        self.observed_tau_syn_count.index_add_(
            0,
            edge_ids,
            torch.ones_like(edge_ids, dtype=torch.long),
        )
        return tau_syn

    @torch.no_grad()
    def _block_counterfactual_targets(
        self,
        representations,
        counterfactual_context,
        block_ids,
        users,
        positive_items,
        negative_items,
    ):
        if block_ids.numel() == 0:
            self._last_block_add_effects = self.mask_logits.new_empty(0)
            self._last_block_remove_effects = self.mask_logits.new_empty(0)
            return self.mask_logits.new_empty(0)

        full_nodes = torch.cat(
            (
                representations['full_users'].detach(),
                representations['full_items'].detach(),
            ),
            dim=0,
        )
        masked_initial = counterfactual_context[
            'masked_initial'
        ].detach()
        base_masked_nodes = torch.cat(
            (
                representations['masked_users'].detach(),
                representations['masked_items'].detach(),
            ),
            dim=0,
        )
        mm_items = counterfactual_context['mm_items'].detach()

        targets = []
        add_effects = []
        remove_effects = []
        self._last_cf_propagation_count = 0
        for block_id, user, positives, negatives in zip(
            block_ids, users, positive_items, negative_items
        ):
            full_margins = self._full_only_margins(
                full_nodes,
                mm_items,
                self.n_users,
                user,
                positives,
                negatives,
            )
            base_margins = self._counterfactual_fused_margins(
                full_nodes,
                base_masked_nodes,
                mm_items,
                user,
                positives,
                negatives,
            )
            plus_margins = torch.empty_like(base_margins)
            minus_margins = torch.empty_like(base_margins)

            # B^(-p) differs only when p itself is a member of the block.
            # Reuse a propagation for queries sharing the same excluded edge.
            excluded_edges = self._block_query_excluded_edge_ids(
                block_id, positives
            )
            for excluded_edge in torch.unique(excluded_edges):
                query_rows = torch.nonzero(
                    excluded_edges == excluded_edge, as_tuple=False
                ).flatten()
                plus_indices = self._forced_block_interaction_indices(
                    block_id,
                    keep=True,
                    excluded_edge_id=excluded_edge,
                )
                minus_indices = self._forced_block_interaction_indices(
                    block_id,
                    keep=False,
                    excluded_edge_id=excluded_edge,
                )
                plus_nodes = self._propagate_ui_graph(
                    self._counterfactual_adjacency(plus_indices),
                    masked_initial,
                )
                minus_nodes = self._propagate_ui_graph(
                    self._counterfactual_adjacency(minus_indices),
                    masked_initial,
                )
                self._last_cf_propagation_count += 2
                plus_margins[query_rows] = (
                    self._counterfactual_fused_margins(
                        full_nodes,
                        plus_nodes,
                        mm_items,
                        user,
                        positives[query_rows],
                        negatives[query_rows],
                    )
                )
                minus_margins[query_rows] = (
                    self._counterfactual_fused_margins(
                        full_nodes,
                        minus_nodes,
                        mm_items,
                        user,
                        positives[query_rows],
                        negatives[query_rows],
                    )
                )

            target, add_effect, remove_effect = (
                self._weighted_block_effect_components(
                    full_margins,
                    plus_margins,
                    base_margins,
                    minus_margins,
                    self.cf_block_full_temperature,
                )
            )
            targets.append(target)
            add_effects.append(add_effect)
            remove_effects.append(remove_effect)

        target_tensor = torch.stack(targets).detach()
        add_tensor = torch.stack(add_effects).detach()
        remove_tensor = torch.stack(remove_effects).detach()
        self._last_block_add_effects = add_tensor
        self._last_block_remove_effects = remove_tensor
        self.observed_block_target_sum.index_add_(
            0,
            block_ids,
            target_tensor.to(self.observed_block_target_sum.dtype),
        )
        self.observed_block_add_effect_sum.index_add_(
            0,
            block_ids,
            add_tensor.to(self.observed_block_add_effect_sum.dtype),
        )
        self.observed_block_remove_effect_sum.index_add_(
            0,
            block_ids,
            remove_tensor.to(self.observed_block_remove_effect_sum.dtype),
        )
        self.observed_block_target_count.index_add_(
            0,
            block_ids,
            torch.ones_like(block_ids, dtype=torch.long),
        )
        self.observed_block_query_count.index_add_(
            0,
            block_ids,
            torch.full_like(
                block_ids,
                self.cf_block_queries_per_target,
                dtype=torch.long,
            ),
        )
        return target_tensor

    def _causal_alignment_loss(self, unit_ids, tau_syn):
        predictions = self._causal_unit_score(unit_ids)
        huber_loss = F.smooth_l1_loss(
            predictions,
            tau_syn.detach(),
            reduction='mean',
            beta=self.cf_huber_beta,
        )
        rank_loss = huber_loss.new_zeros(())
        if unit_ids.numel() >= 2:
            pair_indices = torch.triu_indices(
                unit_ids.numel(),
                unit_ids.numel(),
                offset=1,
                device=unit_ids.device,
            )
            tau_difference = (
                tau_syn[pair_indices[0]] - tau_syn[pair_indices[1]]
            )
            valid_pairs = tau_difference.abs() >= self.cf_min_tau_gap
            if torch.any(valid_pairs):
                training_logits = self.causal_training_logits(unit_ids)
                score_difference = (
                    training_logits[pair_indices[0]]
                    - training_logits[pair_indices[1]]
                )
                direction = tau_difference.sign()
                rank_loss = F.softplus(
                    -direction[valid_pairs]
                    * score_difference[valid_pairs]
                    / self.cf_rank_temperature
                ).mean()
        return huber_loss, rank_loss, predictions

    def _base_loss_from_representations(self, interaction, representations):
        users = interaction[0]
        positive_items = interaction[1]
        negative_items = interaction[2]

        user_embeddings = representations['users'][users]
        positive_embeddings = representations['items'][positive_items]
        negative_embeddings = representations['items'][negative_items]
        ranking_loss = self.bpr_loss(
            user_embeddings, positive_embeddings, negative_embeddings
        )

        if self.cl_weight == 0.0:
            contrastive_loss = ranking_loss.new_zeros(())
        else:
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
        mask_mean = interaction_mask.mean()
        budget_loss = (mask_mean - self.mask_keep_ratio).pow(2)
        binary_loss = (
            interaction_mask * (1.0 - interaction_mask)
        ).mean()
        mask_loss = budget_loss + self.mask_binary_weight * binary_loss
        base_loss = (
            ranking_loss
            + self.cl_weight * contrastive_loss
            + self.mask_weight * mask_loss
        )
        return (
            base_loss,
            ranking_loss,
            contrastive_loss,
            mask_loss,
            mask_mean,
        )

    def calculate_loss(self, interaction):
        representations, counterfactual_context = (
            self._encode_with_counterfactual_context()
        )
        (
            base_loss,
            ranking_loss,
            contrastive_loss,
            mask_loss,
            mask_mean,
        ) = self._base_loss_from_representations(
            interaction, representations
        )

        cf_huber = base_loss.new_zeros(())
        cf_rank = base_loss.new_zeros(())
        cf_samples = base_loss.new_zeros(())
        tau_mean = base_loss.new_zeros(())
        tau_positive_fraction = base_loss.new_zeros(())
        cf_add_effect_mean = base_loss.new_zeros(())
        cf_remove_effect_mean = base_loss.new_zeros(())
        self._last_cf_propagation_count = 0
        self._last_block_add_effects = None
        self._last_block_remove_effects = None

        if (
            self.training
            and self.counterfactual_training_active
            and self.lambda_cf > 0.0
        ):
            unit_ids, users, positives, negatives = (
                self._sample_counterfactual_candidates(
                    interaction[0], interaction[1], interaction[2]
                )
            )
            if unit_ids.numel() > 0:
                if self.cf_intervention_mode == 'semantic_block':
                    tau_syn = self._block_counterfactual_targets(
                        representations,
                        counterfactual_context,
                        unit_ids,
                        users,
                        positives,
                        negatives,
                    )
                    cf_add_effect_mean = (
                        self._last_block_add_effects.mean()
                    )
                    cf_remove_effect_mean = (
                        self._last_block_remove_effects.mean()
                    )
                else:
                    tau_syn = self._counterfactual_targets(
                        representations,
                        counterfactual_context,
                        unit_ids,
                        users,
                        positives,
                        negatives,
                    )
                cf_huber, cf_rank, _ = self._causal_alignment_loss(
                    unit_ids, tau_syn
                )
                cf_samples = base_loss.new_tensor(float(unit_ids.numel()))
                tau_mean = tau_syn.mean()
                tau_positive_fraction = (tau_syn > 0).float().mean()

        cf_total = cf_huber + cf_rank
        total_loss = base_loss + self.lambda_cf * cf_total
        self.latest_loss_components = {
            'bpr': ranking_loss.detach(),
            'contrastive': contrastive_loss.detach(),
            'mask': mask_loss.detach(),
            'mask_mean': mask_mean.detach(),
            'cf_huber': cf_huber.detach(),
            'cf_rank': cf_rank.detach(),
            'cf_total': cf_total.detach(),
            'cf_samples': cf_samples.detach(),
            'tau_syn_mean': tau_mean.detach(),
            'tau_syn_positive_fraction': (
                tau_positive_fraction.detach()
            ),
            # Generic aliases describe both supported target modes. Keep the
            # tau_syn keys above so existing training logs remain compatible.
            'cf_target_mean': tau_mean.detach(),
            'cf_target_positive_fraction': (
                tau_positive_fraction.detach()
            ),
            'cf_add_effect_mean': cf_add_effect_mean.detach(),
            'cf_remove_effect_mean': cf_remove_effect_mean.detach(),
        }
        return total_loss

    @torch.no_grad()
    def get_analysis_artifacts(self):
        artifacts = super(PGL_MASKED_CF, self).get_analysis_artifacts()
        features = self.causal_edge_features()
        predictions = self.causal_score()
        counts = self.observed_tau_syn_count
        observed_mean = torch.full_like(
            self.observed_tau_syn_sum, float('nan')
        )
        observed = counts > 0
        observed_mean[observed] = (
            self.observed_tau_syn_sum[observed]
            / counts[observed].to(self.observed_tau_syn_sum.dtype)
        )

        mask_artifact = artifacts['masks']['masked_branch']
        for inherited_residual_field in (
            'bounded_residual_correction',
            'lambda_sem',
            'lambda_r',
            'residual_temperature',
            'semantic_mask_variant',
        ):
            mask_artifact.pop(inherited_residual_field, None)
        mask_artifact.update({
            'residual_mask_logits': self.mask_logits.detach().cpu(),
            'semantic_contribution': (
                self.lambda_s * self.semantic_prior()
            ).detach().cpu(),
            'causal_score': predictions.detach().cpu(),
            'causal_contribution': (
                self.lambda_c * predictions
            ).detach().cpu(),
            'causal_feature_names': list(self.CAUSAL_FEATURE_NAMES),
            'causal_features': features.detach().cpu(),
            'full_uncertainty': self.full_uncertainty.detach().cpu(),
            'full_mask_disagreement': (
                self.full_mask_disagreement.detach().cpu()
            ),
            'observed_tau_syn_mean': observed_mean.detach().cpu(),
            'observed_tau_syn_count': counts.detach().cpu(),
            # The legacy tau_syn fields contain the configured target. These
            # generic aliases avoid mislabelling fused-effect experiments.
            'observed_cf_target_mean': observed_mean.detach().cpu(),
            'observed_cf_target_count': counts.detach().cpu(),
            'cf_target_mode': self.cf_target_mode,
            'lambda_s': self.lambda_s,
            'lambda_c': self.lambda_c,
            'lambda_cf': self.lambda_cf,
            'cf_warmup_epochs': self.cf_warmup_epochs,
            'cf_samples_per_batch': self.cf_samples_per_batch,
            'cf_intervention_mode': self.cf_intervention_mode,
            'mask_score_mode': (
                'edge_residual_semantic_causal'
                if self.lambda_s > 0.0
                else 'edge_residual_causal'
            ),
        })
        if self.cf_intervention_mode == 'semantic_block':
            block_counts = self.observed_block_target_count
            block_mean = torch.full_like(
                self.observed_block_target_sum, float('nan')
            )
            observed_blocks = block_counts > 0
            block_mean[observed_blocks] = (
                self.observed_block_target_sum[observed_blocks]
                / block_counts[observed_blocks].to(
                    self.observed_block_target_sum.dtype
                )
            )
            block_add_mean = torch.full_like(
                self.observed_block_add_effect_sum, float('nan')
            )
            block_remove_mean = torch.full_like(
                self.observed_block_remove_effect_sum, float('nan')
            )
            block_add_mean[observed_blocks] = (
                self.observed_block_add_effect_sum[observed_blocks]
                / block_counts[observed_blocks].to(
                    self.observed_block_add_effect_sum.dtype
                )
            )
            block_remove_mean[observed_blocks] = (
                self.observed_block_remove_effect_sum[observed_blocks]
                / block_counts[observed_blocks].to(
                    self.observed_block_remove_effect_sum.dtype
                )
            )
            artifacts['counterfactual_blocks'] = {
                'item_prototype_ids': (
                    self.item_prototype_ids.detach().cpu()
                ),
                'edge_block_ids': self.edge_block_ids.detach().cpu(),
                'block_user_ids': self.block_user_ids.detach().cpu(),
                'block_prototype_ids': (
                    self.block_prototype_ids.detach().cpu()
                ),
                'block_sizes': self.block_sizes.detach().cpu(),
                'eligible_for_intervention': (
                    self.block_eligible_for_intervention.detach().cpu()
                ),
                'block_feature_names': list(self.CAUSAL_FEATURE_NAMES),
                'block_features': (
                    self.causal_block_features().detach().cpu()
                ),
                'block_causal_scores': (
                    self.block_causal_score().detach().cpu()
                ),
                'block_residual_logits': (
                    self.block_residual_logit().detach().cpu()
                ),
                'observed_target_mean': block_mean.detach().cpu(),
                'observed_add_effect_mean': (
                    block_add_mean.detach().cpu()
                ),
                'observed_remove_effect_mean': (
                    block_remove_mean.detach().cpu()
                ),
                'observed_target_count': block_counts.detach().cpu(),
                'observed_query_count': (
                    self.observed_block_query_count.detach().cpu()
                ),
            }
        for inherited_residual_field in (
            'semantic_mask_variant',
            'lambda_sem',
            'lambda_r',
            'residual_temperature',
        ):
            artifacts['metadata'].pop(inherited_residual_field, None)
        artifacts['metadata'].update({
            'mask_score_mode': (
                'edge_residual_semantic_causal'
                if self.lambda_s > 0.0
                else 'edge_residual_causal'
            ),
            'cf_intervention_mode': self.cf_intervention_mode,
            'lambda_s': self.lambda_s,
            'lambda_c': self.lambda_c,
            'lambda_cf': self.lambda_cf,
            'cf_samples_per_batch': self.cf_samples_per_batch,
            'cf_warmup_epochs': self.cf_warmup_epochs,
            'cf_hidden_dim': self.cf_hidden_dim,
            'cf_huber_beta': self.cf_huber_beta,
            'cf_rank_temperature': self.cf_rank_temperature,
            'cf_min_tau_gap': self.cf_min_tau_gap,
            'cf_target_mode': self.cf_target_mode,
            'cf_epoch': int(self.cf_epoch.item()),
        })
        if self.cf_intervention_mode == 'semantic_block':
            artifacts['metadata'].update({
                'cf_block_num_prototypes': (
                    self.cf_block_num_prototypes
                ),
                'cf_block_visual_weight': (
                    self.cf_block_visual_weight
                ),
                'cf_block_min_edges': self.cf_block_min_edges,
                'cf_block_queries_per_target': (
                    self.cf_block_queries_per_target
                ),
                'cf_block_full_temperature': (
                    self.cf_block_full_temperature
                ),
                'cf_block_kmeans_seed': self.cf_block_kmeans_seed,
                'cf_block_kmeans_iterations': (
                    self.cf_block_kmeans_iterations
                ),
                'cf_block_kmeans_tolerance': (
                    self.cf_block_kmeans_tolerance
                ),
                'num_semantic_blocks': self.num_semantic_blocks,
                'num_eligible_semantic_blocks': int(
                    self.block_eligible_for_intervention.sum().item()
                ),
                'semantic_block_coverage': float(
                    self.block_eligible_for_intervention.float()
                    .mean().item()
                ),
            })
        return artifacts
