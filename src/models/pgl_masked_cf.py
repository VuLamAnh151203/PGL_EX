r"""
Counterfactual synergy-guided edge masking for PGL.

The hard mask is scored by a fixed semantic prior plus a lightweight causal
scorer trained from sampled keep/remove interventions.  Counterfactual labels
measure whether an edge is especially useful when the masked branch is fused
with the full branch, rather than merely useful to the masked branch alone.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pgl_masked import _config_value
from models.pgl_masked_ex import PGL_MASKED_EX


class PGL_MASKED_CF(PGL_MASKED_EX):
    """PGL with sampled counterfactual synergy supervision."""

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

        if self.ui_branch_mode != 'dual':
            raise ValueError(
                "PGL_MASKED_CF only supports ui_branch_mode='dual'."
            )
        if self.mask_graph_mode != 'hard':
            raise ValueError(
                "PGL_MASKED_CF only supports mask_graph_mode='hard'."
            )
        self._validate_counterfactual_config()

        # PGL_MASKED_CF replaces the per-edge learned residual entirely.
        # Retain the parameter only for checkpoint compatibility with the
        # parent class, but never optimize or use it in the mask score.
        with torch.no_grad():
            self.mask_logits.zero_()
        self.mask_logits.requires_grad_(False)

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
            'cf_epoch', torch.tensor(-1, dtype=torch.long), persistent=True
        )

        self.causal_scorer = nn.Sequential(
            nn.Linear(len(self.CAUSAL_FEATURE_NAMES), self.cf_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cf_hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.causal_scorer[0].weight)
        nn.init.zeros_(self.causal_scorer[0].bias)
        # Initial g_phi(e)=0 makes the first mask purely semantic.
        nn.init.zeros_(self.causal_scorer[2].weight)
        nn.init.zeros_(self.causal_scorer[2].bias)

        self._last_cf_propagation_count = 0

    def _validate_counterfactual_config(self):
        finite_non_negative = {
            'lambda_c': self.lambda_c,
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
        if not math.isfinite(self.cf_huber_beta) or self.cf_huber_beta <= 0:
            raise ValueError('cf_huber_beta must be finite and positive.')
        if (
            not math.isfinite(self.cf_rank_temperature)
            or self.cf_rank_temperature <= 0
        ):
            raise ValueError(
                'cf_rank_temperature must be finite and positive.'
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

    def causal_score(self, edge_ids=None):
        return self.causal_scorer(
            self.causal_edge_features(edge_ids)
        ).squeeze(1)

    def effective_mask_logits(self):
        """Mask score used by BPR; causal scorer is detached on this path."""
        return (
            self.semantic_prior()
            + self.lambda_c * self.causal_score().detach()
        )

    def causal_training_logits(self, edge_ids):
        """Non-detached score used only by counterfactual supervision."""
        return (
            self.semantic_prior()[edge_ids]
            + self.lambda_c * self.causal_score(edge_ids)
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
    def _sample_counterfactual_candidates(
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
        gate = torch.sigmoid(
            self.fusion_gate(
                torch.cat((selected_full, selected_masked), dim=1)
            )
        )
        fused = gate * selected_full + (1.0 - gate) * selected_masked

        fused_user = fused[0]
        fused_positive = fused[1] + mm_items[positive_item]
        fused_negative = fused[2] + mm_items[negative_item]
        fused_margin = torch.sum(fused_user * fused_positive) - torch.sum(
            fused_user * fused_negative
        )

        masked_user = selected_masked[0]
        masked_positive = selected_masked[1] + mm_items[positive_item]
        masked_negative = selected_masked[2] + mm_items[negative_item]
        masked_margin = torch.sum(
            masked_user * masked_positive
        ) - torch.sum(masked_user * masked_negative)
        return fused_margin, masked_margin

    @staticmethod
    def _synergy_from_outcomes(
        fused_plus, fused_minus, masked_plus, masked_minus
    ):
        fused_effect = fused_plus - fused_minus
        masked_effect = masked_plus - masked_minus
        return fused_effect - masked_effect

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

        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((full_nodes, masked_nodes), dim=1))
        )
        fused_nodes = gate * full_nodes + (1.0 - gate) * masked_nodes
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
                self._synergy_from_outcomes(
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

    def _causal_alignment_loss(self, edge_ids, tau_syn):
        predictions = self.causal_score(edge_ids)
        huber_loss = F.smooth_l1_loss(
            predictions,
            tau_syn.detach(),
            reduction='mean',
            beta=self.cf_huber_beta,
        )
        rank_loss = huber_loss.new_zeros(())
        if edge_ids.numel() >= 2:
            pair_indices = torch.triu_indices(
                edge_ids.numel(),
                edge_ids.numel(),
                offset=1,
                device=edge_ids.device,
            )
            tau_difference = (
                tau_syn[pair_indices[0]] - tau_syn[pair_indices[1]]
            )
            valid_pairs = tau_difference.abs() >= self.cf_min_tau_gap
            if torch.any(valid_pairs):
                training_logits = self.causal_training_logits(edge_ids)
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
        self._last_cf_propagation_count = 0

        if (
            self.training
            and self.counterfactual_training_active
            and self.lambda_cf > 0.0
        ):
            edge_ids, users, positives, negatives = (
                self._sample_counterfactual_candidates(
                    interaction[0], interaction[1], interaction[2]
                )
            )
            if edge_ids.numel() > 0:
                tau_syn = self._counterfactual_targets(
                    representations,
                    counterfactual_context,
                    edge_ids,
                    users,
                    positives,
                    negatives,
                )
                cf_huber, cf_rank, _ = self._causal_alignment_loss(
                    edge_ids, tau_syn
                )
                cf_samples = base_loss.new_tensor(float(edge_ids.numel()))
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
            'residual_mask_logits',
            'bounded_residual_correction',
            'lambda_sem',
            'lambda_r',
            'residual_temperature',
            'semantic_mask_variant',
        ):
            mask_artifact.pop(inherited_residual_field, None)
        mask_artifact.update({
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
            'lambda_c': self.lambda_c,
            'lambda_cf': self.lambda_cf,
            'cf_warmup_epochs': self.cf_warmup_epochs,
            'cf_samples_per_batch': self.cf_samples_per_batch,
            'mask_score_mode': 'counterfactual_synergy',
        })
        for inherited_residual_field in (
            'semantic_mask_variant',
            'lambda_sem',
            'lambda_r',
            'residual_temperature',
        ):
            artifacts['metadata'].pop(inherited_residual_field, None)
        artifacts['metadata'].update({
            'mask_score_mode': 'counterfactual_synergy',
            'lambda_c': self.lambda_c,
            'lambda_cf': self.lambda_cf,
            'cf_samples_per_batch': self.cf_samples_per_batch,
            'cf_warmup_epochs': self.cf_warmup_epochs,
            'cf_hidden_dim': self.cf_hidden_dim,
            'cf_huber_beta': self.cf_huber_beta,
            'cf_rank_temperature': self.cf_rank_temperature,
            'cf_min_tau_gap': self.cf_min_tau_gap,
            'cf_epoch': int(self.cf_epoch.item()),
        })
        return artifacts
