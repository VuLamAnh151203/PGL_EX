r"""
PGL_MASKED with a semantic-guided learnable interaction mask.

The full U-I branch is unchanged.  The masked branch ranks or weights an
interaction with an effective logit made from a fixed leave-one-out semantic
prior and, optionally, the residual per-edge logit learned by PGL_MASKED.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pgl_masked import PGL_MASKED, _config_value


class PGL_MASKED_EX(PGL_MASKED):
    """Semantic-prior extension of :class:`PGL_MASKED`.

    Supported variants are ``semantic_only`` and ``prior_residual``.  The
    semantic affinities are computed once from the original pretrained item
    features.  They are buffers rather than parameters, so the mask cannot
    change the features from which its prior was derived.
    """

    _SEMANTIC_VARIANTS = {'semantic_only', 'prior_residual'}
    _AFFINITY_CHUNK_SIZE = 2048

    def __init__(self, config, dataset):
        super(PGL_MASKED_EX, self).__init__(config, dataset)

        self.semantic_mask_variant = str(
            _config_value(
                config, 'semantic_mask_variant', 'prior_residual'
            )
        ).lower()
        self.lambda_sem = float(_config_value(config, 'lambda_sem', 1.0))

        if self.ui_branch_mode != 'dual':
            raise ValueError(
                "PGL_MASKED_EX only supports ui_branch_mode='dual'; "
                "masked_only and dual_modal are not supported."
            )
        if self.mask_graph_mode not in {'hard', 'soft'}:
            raise ValueError(
                "PGL_MASKED_EX only supports mask_graph_mode='hard' or "
                "'soft'; double_full is not supported."
            )
        if self.semantic_mask_variant not in self._SEMANTIC_VARIANTS:
            raise ValueError(
                "semantic_mask_variant must be 'semantic_only' or "
                "'prior_residual'."
            )
        if not math.isfinite(self.lambda_sem) or self.lambda_sem < 0.0:
            raise ValueError('lambda_sem must be a finite non-negative value.')
        if self.mask_logits is None:
            raise RuntimeError(
                'PGL_MASKED_EX requires residual mask_logits to be present.'
            )

        # Equal visual/text contribution at initialization.  Softmax keeps
        # both learned weights non-negative and summing to one.
        self.semantic_gamma = nn.Parameter(torch.zeros(2))

        forward_edges = self.ui_edge_index[:, :self.num_interactions]
        edge_users = forward_edges[0]
        edge_items = forward_edges[1] - self.n_users
        with torch.no_grad():
            # Raw visual features can be wide (for example 4096D).  Computing
            # fixed affinities on CPU avoids consuming training GPU memory
            # during construction; the small per-edge buffers move with the
            # model afterwards.
            visual_affinity = self._leave_one_out_affinity(
                self.v_feat.detach().cpu(),
                edge_users,
                edge_items,
                self.n_users,
            )
            textual_affinity = self._leave_one_out_affinity(
                self.t_feat.detach().cpu(),
                edge_users,
                edge_items,
                self.n_users,
            )

        self.register_buffer(
            'semantic_visual_affinity', visual_affinity, persistent=True
        )
        self.register_buffer(
            'semantic_textual_affinity', textual_affinity, persistent=True
        )

        # Keep the residual in the state dict for comparable artifacts, but
        # exclude it from learning in the semantic-only ablation.
        if self.semantic_mask_variant == 'semantic_only':
            self.mask_logits.requires_grad_(False)

    @staticmethod
    def _leave_one_out_affinity(
        item_features,
        edge_users,
        edge_items,
        num_users,
        chunk_size=None,
    ):
        """Return per-edge cosine affinity to the rest of a user's history."""
        if item_features.ndim != 2:
            raise ValueError('item_features must be a two-dimensional tensor.')
        if edge_users.ndim != 1 or edge_items.ndim != 1:
            raise ValueError('edge_users and edge_items must be vectors.')
        if edge_users.numel() != edge_items.numel():
            raise ValueError('edge_users and edge_items must have equal size.')
        if num_users <= 0:
            raise ValueError('num_users must be positive.')
        if chunk_size is None:
            chunk_size = PGL_MASKED_EX._AFFINITY_CHUNK_SIZE
        if chunk_size <= 0:
            raise ValueError('chunk_size must be positive.')

        feature_device = item_features.device
        users = edge_users.to(device=feature_device, dtype=torch.long)
        items = edge_items.to(device=feature_device, dtype=torch.long)
        if users.numel() == 0:
            return item_features.new_empty(0)
        if users.min() < 0 or users.max() >= num_users:
            raise ValueError('An edge user ID is outside the user table.')
        if items.min() < 0 or items.max() >= item_features.size(0):
            raise ValueError('An edge item ID is outside the feature table.')

        normalized_items = F.normalize(
            item_features, p=2, dim=1, eps=1e-12
        )
        user_feature_sums = item_features.new_zeros(
            (num_users, item_features.size(1))
        )
        for start in range(0, users.numel(), chunk_size):
            stop = min(start + chunk_size, users.numel())
            user_feature_sums.index_add_(
                0,
                users[start:stop],
                normalized_items[items[start:stop]],
            )

        user_degrees = torch.bincount(
            users, minlength=num_users
        ).to(dtype=item_features.dtype)
        affinities = item_features.new_empty(users.numel())
        for start in range(0, users.numel(), chunk_size):
            stop = min(start + chunk_size, users.numel())
            chunk_users = users[start:stop]
            edge_features = normalized_items[items[start:stop]]
            edge_degrees = user_degrees[chunk_users]
            other_feature_sums = (
                user_feature_sums[chunk_users] - edge_features
            )
            denominator = (
                (edge_degrees - 1.0).clamp_min(1.0).unsqueeze(1)
            )
            leave_one_out_preferences = (
                other_feature_sums / denominator
            )
            chunk_affinities = F.cosine_similarity(
                leave_one_out_preferences,
                edge_features,
                dim=1,
                eps=1e-12,
            )
            affinities[start:stop] = torch.where(
                edge_degrees > 1.0,
                chunk_affinities,
                torch.zeros_like(chunk_affinities),
            )
        return torch.nan_to_num(
            affinities, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(-1.0, 1.0)

    def semantic_weights(self):
        """Return ``[alpha_visual, alpha_textual]``."""
        return F.softmax(self.semantic_gamma, dim=0)

    def semantic_prior(self):
        """Fuse fixed visual and textual affinities with learned weights."""
        alpha = self.semantic_weights()
        return (
            alpha[0] * self.semantic_visual_affinity.detach()
            + alpha[1] * self.semantic_textual_affinity.detach()
        )

    def effective_mask_logits(self):
        """Return the score actually used to weight or select U-I edges."""
        semantic_component = self.lambda_sem * self.semantic_prior()
        if self.semantic_mask_variant == 'semantic_only':
            return semantic_component
        return self.mask_logits + semantic_component

    @property
    def hard_keep_count(self):
        # The experiment specification defines K with floor for positive E.
        return max(
            1,
            min(
                self.num_interactions,
                int(self.num_interactions * self.mask_keep_ratio),
            ),
        )

    @torch.no_grad()
    def pre_epoch_processing(self):
        if self.mask_graph_mode == 'hard':
            self.hard_train_indices = self._sample_hard_train_indices(
                self.effective_mask_logits()
            )

    @torch.no_grad()
    def post_epoch_processing(self):
        if self.mask_graph_mode == 'hard':
            self.hard_eval_indices = self._select_hard_eval_indices(
                self.effective_mask_logits()
            )

    def _current_hard_indices(self, second_branch=False):
        if second_branch:
            raise ValueError('PGL_MASKED_EX has only one masked branch.')

        if self.training:
            if self.hard_train_indices.numel() == 0:
                with torch.no_grad():
                    self.hard_train_indices = self._sample_hard_train_indices(
                        self.effective_mask_logits()
                    )
            return self.hard_train_indices

        if self.hard_eval_indices.numel() == 0:
            with torch.no_grad():
                self.hard_eval_indices = self._select_hard_eval_indices(
                    self.effective_mask_logits()
                )
        return self.hard_eval_indices

    def _masked_ui_adjacency(self, second_branch=False):
        if second_branch:
            raise ValueError('PGL_MASKED_EX has only one masked branch.')

        interaction_mask = torch.sigmoid(self.effective_mask_logits())
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

    @torch.no_grad()
    def get_analysis_artifacts(self):
        """Export the effective mask and its semantic/residual components."""
        artifacts = super(PGL_MASKED_EX, self).get_analysis_artifacts()

        effective_logits = self.effective_mask_logits()
        probabilities = torch.sigmoid(effective_logits)
        topk_indices = self._select_hard_eval_indices(effective_logits)
        selected = torch.zeros_like(probabilities, dtype=torch.bool)
        selected[topk_indices] = True
        alpha = self.semantic_weights()
        prior = self.semantic_prior()

        artifacts['masks']['masked_branch'] = {
            # Existing notebook-facing fields describe the actual selection
            # score rather than the residual theta parameter.
            'logits': effective_logits.detach().cpu(),
            'probabilities': probabilities.detach().cpu(),
            'selected_at_keep_ratio': selected.detach().cpu(),
            # Extra decomposition for semantic-guidance analysis.
            'residual_mask_logits': self.mask_logits.detach().cpu(),
            'semantic_visual_affinity': (
                self.semantic_visual_affinity.detach().cpu()
            ),
            'semantic_textual_affinity': (
                self.semantic_textual_affinity.detach().cpu()
            ),
            'semantic_prior': prior.detach().cpu(),
            'semantic_alpha_visual': alpha[0].detach().cpu(),
            'semantic_alpha_textual': alpha[1].detach().cpu(),
            'lambda_sem': self.lambda_sem,
            'semantic_mask_variant': self.semantic_mask_variant,
        }
        artifacts['metadata'].update({
            'semantic_mask_variant': self.semantic_mask_variant,
            'semantic_feature_source': 'fixed_original_pretrained',
            'lambda_sem': self.lambda_sem,
            'semantic_alpha_visual': float(alpha[0].item()),
            'semantic_alpha_textual': float(alpha[1].item()),
        })
        return artifacts
