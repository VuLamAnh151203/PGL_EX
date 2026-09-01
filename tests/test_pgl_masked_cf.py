import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.pgl_masked_cf import PGL_MASKED_CF  # noqa: E402
from utils.utils import get_model  # noqa: E402


class NullableConfig(dict):
    def __getitem__(self, key):
        return self.get(key)


class FakeDatasetStats:
    def get_user_num(self):
        return 3

    def get_item_num(self):
        return 4


class FakeTrainData:
    def __init__(self):
        self.dataset = FakeDatasetStats()
        self._interactions = sp.coo_matrix(
            (
                np.ones(5, dtype=np.float32),
                (
                    np.array([0, 0, 1, 2, 2]),
                    np.array([0, 1, 2, 1, 3]),
                ),
            ),
            shape=(3, 4),
        )

    def inter_matrix(self, form='coo'):
        return self._interactions.asformat(form)


class FakeBlockTrainData(FakeTrainData):
    def __init__(self):
        self.dataset = FakeDatasetStats()
        self._interactions = sp.coo_matrix(
            (
                np.ones(7, dtype=np.float32),
                (
                    np.array([0, 0, 0, 1, 2, 2, 2]),
                    np.array([0, 1, 2, 2, 0, 1, 3]),
                ),
            ),
            shape=(3, 4),
        )


class TestablePGLMaskedCF(PGL_MASKED_CF):
    def _build_or_load_mm_graph(self, config):
        indices = torch.arange(self.n_items).repeat(2, 1)
        values = torch.ones(self.n_items)
        adjacency = torch.sparse_coo_tensor(
            indices, values, (self.n_items, self.n_items)
        ).coalesce()
        self.register_buffer('mm_adj', adjacency)


class PGLMaskedCFTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        dataset_dir = self.root / 'toy'
        dataset_dir.mkdir()
        np.save(
            dataset_dir / 'image_feat.npy',
            np.array([
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),
        )
        np.save(
            dataset_dir / 'text_feat.npy',
            np.array([
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.2, 0.8],
            ], dtype=np.float32),
        )
        self.train_data = FakeTrainData()
        self.block_train_data = FakeBlockTrainData()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def config(
        self,
        warmup=10,
        lambda_cf=0.1,
        target_mode='synergy',
        intervention_mode='edge',
    ):
        return NullableConfig({
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'NEG_PREFIX': 'neg_',
            'train_batch_size': 3,
            'device': torch.device('cpu'),
            'end2end': False,
            'is_multimodal_model': True,
            'data_path': str(self.root) + os.sep,
            'dataset': 'toy',
            'vision_feature_file': 'image_feat.npy',
            'text_feature_file': 'text_feat.npy',
            'embedding_size': 2,
            'feat_embed_dim': 2,
            'knn_k': 2,
            'n_mm_layers': 1,
            'n_ui_layers': 1,
            'mask_keep_ratio': 0.4,
            'mask_degree_mode': 'full',
            'mask_graph_mode': 'hard',
            'user_embedding_mode': 'separate',
            'ui_branch_mode': 'dual',
            'hard_mask_temperature': 1.0,
            'semantic_mask_variant': 'prior_residual',
            'lambda_sem': 0.0,
            'lambda_s': 0.0,
            'lambda_c': 1.0,
            'lambda_cf': lambda_cf,
            'cf_samples_per_batch': 2,
            'cf_warmup_epochs': warmup,
            'cf_hidden_dim': 8,
            'cf_huber_beta': 0.1,
            'cf_rank_temperature': 0.2,
            'cf_min_tau_gap': 1e-6,
            'cf_target_mode': target_mode,
            'cf_intervention_mode': intervention_mode,
            'cf_block_num_prototypes': 2,
            'cf_block_visual_weight': 0.5,
            'cf_block_min_edges': 2,
            'cf_block_queries_per_target': 3,
            'cf_block_full_temperature': 1.0,
            'cf_block_kmeans_seed': 999,
            'cf_block_kmeans_iterations': 25,
            'cf_block_kmeans_tolerance': 1e-4,
            'cl_weight': 0.05,
            'cl_temperature': 0.2,
            'dropout': 0.0,
            'mask_weight': 0.1,
            'mask_binary_weight': 0.1,
        })

    @staticmethod
    def interaction():
        # User 1 has degree one and must never be selected for intervention.
        return (
            torch.tensor([1, 0, 2]),
            torch.tensor([2, 0, 1]),
            torch.tensor([3, 2, 0]),
        )

    def model(
        self,
        warmup=10,
        lambda_cf=0.1,
        target_mode='synergy',
        intervention_mode='edge',
    ):
        train_data = (
            self.block_train_data
            if intervention_mode == 'semantic_block'
            else self.train_data
        )
        return TestablePGLMaskedCF(
            self.config(
                warmup=warmup,
                lambda_cf=lambda_cf,
                target_mode=target_mode,
                intervention_mode=intervention_mode,
            ),
            train_data,
        )

    def block_model(self, warmup=0, lambda_cf=0.1):
        return self.model(
            warmup=warmup,
            lambda_cf=lambda_cf,
            target_mode='fused_effect',
            intervention_mode='semantic_block',
        )

    def test_loader_initial_score_features_and_learnable_residual(self):
        self.assertIs(get_model('PGL_MASKED_CF'), PGL_MASKED_CF)
        model = self.model()

        self.assertTrue(model.mask_logits.requires_grad)
        torch.testing.assert_close(
            model.causal_score(), torch.zeros(model.num_interactions)
        )
        torch.testing.assert_close(
            model.effective_mask_logits(), model.mask_logits
        )

        model.lambda_s = 0.5
        torch.testing.assert_close(
            model.effective_mask_logits(),
            model.mask_logits + 0.5 * model.semantic_prior(),
        )

        features = model.causal_edge_features()
        self.assertEqual(tuple(features.shape), (5, 6))
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(
            features[:, 4], torch.ones(model.num_interactions)
        )
        torch.testing.assert_close(
            features[:, 5], torch.zeros(model.num_interactions)
        )
        self.assertTrue((features[:, 2:4] >= 0).all())
        self.assertTrue((features[:, 2:4] <= 1).all())

    def test_candidate_sampling_skips_degree_one_and_target_edge(self):
        model = self.model(warmup=0)
        model.pre_epoch_processing()
        interaction = self.interaction()
        edge_ids, users, positives, _ = (
            model._sample_counterfactual_candidates(*interaction)
        )
        forward_items = (
            model.ui_edge_index[1, :model.num_interactions] - model.n_users
        )
        forward_users = model.ui_edge_index[0, :model.num_interactions]

        self.assertEqual(edge_ids.numel(), 2)
        self.assertEqual(torch.unique(edge_ids).numel(), 2)
        self.assertFalse(torch.any(users == 1))
        torch.testing.assert_close(forward_users[edge_ids], users)
        self.assertTrue(torch.all(forward_items[edge_ids] != positives))

    def test_forced_graph_changes_only_the_candidate_interaction(self):
        model = self.model(warmup=0)
        model.hard_train_indices = torch.tensor([0, 2])

        plus_unselected = model._forced_interaction_indices(
            torch.tensor(1), keep=True
        )
        minus_unselected = model._forced_interaction_indices(
            torch.tensor(1), keep=False
        )
        plus_selected = model._forced_interaction_indices(
            torch.tensor(0), keep=True
        )
        minus_selected = model._forced_interaction_indices(
            torch.tensor(0), keep=False
        )

        self.assertEqual(set(plus_unselected.tolist()), {0, 1, 2})
        self.assertEqual(set(minus_unselected.tolist()), {0, 2})
        self.assertEqual(set(plus_selected.tolist()), {0, 2})
        self.assertEqual(set(minus_selected.tolist()), {2})

        adjacency = model._counterfactual_adjacency(plus_unselected)
        dense = adjacency.to_dense()
        for edge_id in plus_unselected:
            user = model.ui_edge_index[0, edge_id]
            item_node = model.ui_edge_index[1, edge_id]
            self.assertGreater(dense[user, item_node].item(), 0.0)
            self.assertGreater(dense[item_node, user].item(), 0.0)

    def test_synergy_difference_in_differences(self):
        synergy = PGL_MASKED_CF._synergy_from_outcomes(
            torch.tensor(0.61),
            torch.tensor(0.49),
            torch.tensor(0.22),
            torch.tensor(0.20),
        )
        torch.testing.assert_close(synergy, torch.tensor(0.10))

    def test_configurable_counterfactual_target(self):
        outcomes = (
            torch.tensor(0.61),
            torch.tensor(0.49),
            torch.tensor(0.22),
            torch.tensor(0.20),
        )
        synergy_model = self.model(target_mode='synergy')
        fused_effect_model = self.model(target_mode='fused_effect')

        torch.testing.assert_close(
            synergy_model._counterfactual_target_from_outcomes(*outcomes),
            torch.tensor(0.10),
        )
        torch.testing.assert_close(
            fused_effect_model._counterfactual_target_from_outcomes(
                *outcomes
            ),
            torch.tensor(0.12),
        )

    def test_ten_complete_warmup_epochs(self):
        model = self.model(warmup=10)
        for expected_epoch in range(10):
            model.pre_epoch_processing()
            self.assertEqual(model.cf_epoch.item(), expected_epoch)
            self.assertFalse(model.counterfactual_training_active)

        model.pre_epoch_processing()
        self.assertEqual(model.cf_epoch.item(), 10)
        self.assertTrue(model.counterfactual_training_active)

    def test_dynamic_features_update_only_at_post_epoch(self):
        model = self.model(warmup=0)
        model.pre_epoch_processing()
        uncertainty_before = model.full_uncertainty.clone()
        disagreement_before = model.full_mask_disagreement.clone()

        model.forward()
        torch.testing.assert_close(
            model.full_uncertainty, uncertainty_before
        )
        torch.testing.assert_close(
            model.full_mask_disagreement, disagreement_before
        )

        model.post_epoch_processing()
        self.assertTrue(torch.isfinite(model.full_uncertainty).all())
        self.assertTrue(torch.isfinite(model.full_mask_disagreement).all())
        self.assertTrue((model.full_uncertainty >= 0).all())
        self.assertTrue((model.full_uncertainty <= 1).all())
        self.assertTrue((model.full_mask_disagreement >= 0).all())
        self.assertTrue((model.full_mask_disagreement <= 1).all())
        self.assertEqual(
            model.hard_eval_indices.numel(), model.hard_keep_count
        )

    def test_bpr_path_does_not_update_causal_scorer(self):
        model = self.model(warmup=100, lambda_cf=0.0)
        model.train()
        model.pre_epoch_processing()
        loss = model.calculate_loss(self.interaction())
        loss.backward()

        for parameter in model.causal_scorer.parameters():
            self.assertIsNone(parameter.grad)
        self.assertIsNotNone(model.mask_logits.grad)
        self.assertTrue(torch.isfinite(model.mask_logits.grad).all())

    def test_causal_losses_update_scorer_and_skip_tied_rank(self):
        model = self.model(warmup=0)
        edge_ids = torch.tensor([0, 1])
        targets = torch.tensor([0.4, -0.2])
        huber, rank, _ = model._causal_alignment_loss(edge_ids, targets)
        (huber + rank).backward()

        self.assertGreater(rank.item(), 0.0)
        self.assertIsNotNone(model.causal_scorer[2].weight.grad)
        self.assertTrue(
            torch.isfinite(model.causal_scorer[2].weight.grad).all()
        )
        self.assertIsNone(model.mask_logits.grad)
        self.assertIsNone(model.semantic_gamma.grad)

        model.zero_grad(set_to_none=True)
        _, tied_rank, _ = model._causal_alignment_loss(
            edge_ids, torch.tensor([0.1, 0.1])
        )
        self.assertEqual(tied_rank.item(), 0.0)

    def test_counterfactual_targets_are_detached_and_use_four_propagations(self):
        model = self.model(warmup=0)
        model.train()
        model.pre_epoch_processing()
        interaction = self.interaction()
        representations, context = (
            model._encode_with_counterfactual_context()
        )
        edge_ids, users, positives, negatives = (
            model._sample_counterfactual_candidates(*interaction)
        )
        targets = model._counterfactual_targets(
            representations,
            context,
            edge_ids,
            users,
            positives,
            negatives,
        )

        self.assertEqual(targets.numel(), 2)
        self.assertFalse(targets.requires_grad)
        self.assertTrue(torch.isfinite(targets).all())
        self.assertEqual(model._last_cf_propagation_count, 4)
        self.assertEqual(
            model.observed_tau_syn_count.sum().item(), 2
        )

    def test_semantic_block_prototypes_and_membership_are_deterministic(self):
        first = self.block_model()
        second = self.block_model()
        torch.testing.assert_close(
            first.item_prototype_ids, second.item_prototype_ids
        )
        self.assertTrue((first.item_prototype_ids >= 0).all())
        self.assertTrue((first.item_prototype_ids < 2).all())
        self.assertEqual(first.edge_block_ids.numel(), 7)
        self.assertEqual(
            torch.unique(first.edge_block_ids).numel(),
            first.num_semantic_blocks,
        )

        edge_users = first.ui_edge_index[0, :first.num_interactions]
        edge_items = (
            first.ui_edge_index[1, :first.num_interactions]
            - first.n_users
        )
        expected_keys = (
            edge_users * first.cf_block_num_prototypes
            + first.item_prototype_ids[edge_items]
        )
        for block_id in range(first.num_semantic_blocks):
            member_edges = torch.nonzero(
                first.edge_block_ids == block_id, as_tuple=False
            ).flatten()
            self.assertEqual(
                torch.unique(expected_keys[member_edges]).numel(), 1
            )

        zero_features = PGL_MASKED_CF._semantic_item_features(
            torch.zeros(4, 3), torch.zeros(4, 2), 0.5
        )
        assignments = PGL_MASKED_CF._spherical_kmeans(
            zero_features, 2, 7, 3, 1e-4
        )
        self.assertTrue(torch.isfinite(zero_features).all())
        self.assertEqual(tuple(assignments.shape), (4,))

    def test_semantic_block_sampling_queries_and_forced_graph(self):
        model = self.block_model()
        model.pre_epoch_processing()
        users = torch.tensor([0, 1, 2])
        dummy = torch.zeros_like(users)
        block_ids, sampled_users, positives, negatives = (
            model._sample_counterfactual_candidates(
                users, dummy, dummy
            )
        )
        self.assertEqual(block_ids.numel(), 2)
        self.assertEqual(torch.unique(block_ids).numel(), 2)
        self.assertEqual(tuple(positives.shape), (2, 3))
        self.assertEqual(tuple(negatives.shape), (2, 3))

        forward_items = (
            model.ui_edge_index[1, :model.num_interactions]
            - model.n_users
        )
        for block_id, user, positive_row, negative_row in zip(
            block_ids, sampled_users, positives, negatives
        ):
            start = int(model.block_edge_ptr[block_id].item())
            stop = int(model.block_edge_ptr[block_id + 1].item())
            block_edges = model.block_edge_ids[start:stop]
            block_items = forward_items[block_edges]
            history_start = int(model.user_edge_ptr[user].item())
            history_stop = int(model.user_edge_ptr[user + 1].item())
            history_edges = model.user_edge_ids[
                history_start:history_stop
            ]
            history_items = forward_items[history_edges]
            self.assertTrue(torch.isin(positive_row, history_items).all())
            self.assertFalse(torch.isin(negative_row, history_items).any())

            excluded_edges = model._block_query_excluded_edge_ids(
                block_id, positive_row
            )
            for excluded_edge in excluded_edges[excluded_edges >= 0]:
                model.hard_train_indices = torch.tensor([0, 2, 4])
                plus_loo = model._forced_block_interaction_indices(
                    block_id,
                    keep=True,
                    excluded_edge_id=excluded_edge,
                )
                minus_loo = model._forced_block_interaction_indices(
                    block_id,
                    keep=False,
                    excluded_edge_id=excluded_edge,
                )
                was_selected = torch.isin(
                    excluded_edge, model.hard_train_indices
                )
                self.assertEqual(
                    bool(torch.isin(excluded_edge, plus_loo)),
                    bool(was_selected),
                )
                self.assertEqual(
                    bool(torch.isin(excluded_edge, minus_loo)),
                    bool(was_selected),
                )

            model.hard_train_indices = torch.tensor([0, 2, 4])
            plus = model._forced_block_interaction_indices(
                block_id, keep=True
            )
            minus = model._forced_block_interaction_indices(
                block_id, keep=False
            )
            self.assertTrue(torch.isin(block_edges, plus).all())
            self.assertFalse(torch.isin(block_edges, minus).any())
            outside_base = model.hard_train_indices[
                ~torch.isin(model.hard_train_indices, block_edges)
            ]
            self.assertTrue(torch.isin(outside_base, plus).all())
            self.assertTrue(torch.isin(outside_base, minus).all())

    def test_semantic_block_features_scores_and_weighted_target(self):
        model = self.block_model()
        edge_features = model.causal_edge_features()
        block_features = model.causal_block_features()
        for block_id in range(model.num_semantic_blocks):
            members = torch.nonzero(
                model.edge_block_ids == block_id, as_tuple=False
            ).flatten()
            torch.testing.assert_close(
                block_features[block_id], edge_features[members].mean(0)
            )

        with torch.no_grad():
            model.causal_scorer[2].weight.fill_(0.25)
        block_scores = model.block_causal_score()
        edge_scores = model.causal_score()
        torch.testing.assert_close(
            edge_scores, block_scores[model.edge_block_ids]
        )

        full = torch.tensor([-1.0, 0.0, 1.0])
        plus = torch.tensor([0.3, 0.2, 0.1])
        minus = torch.tensor([0.0, 0.0, 0.0])
        actual = model._weighted_block_effect(full, plus, minus, 1.0)
        weights = torch.sigmoid(-full)
        expected = (
            weights * (F.logsigmoid(plus) - F.logsigmoid(minus))
        ).sum() / weights.sum()
        torch.testing.assert_close(actual, expected)

        base = torch.tensor([0.1, 0.1, 0.05])
        total, add, remove = model._weighted_block_effect_components(
            full, plus, base, minus, 1.0
        )
        expected_add = (
            weights * (F.logsigmoid(plus) - F.logsigmoid(base))
        ).sum() / weights.sum()
        expected_remove = (
            weights * (F.logsigmoid(base) - F.logsigmoid(minus))
        ).sum() / weights.sum()
        torch.testing.assert_close(add, expected_add)
        torch.testing.assert_close(remove, expected_remove)
        torch.testing.assert_close(total, add + remove)
        torch.testing.assert_close(total, actual)

    def test_semantic_block_training_artifacts_and_checkpoint(self):
        model = self.block_model()
        model.train()
        model.pre_epoch_processing()
        loss = model.calculate_loss(self.interaction())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreaterEqual(model._last_cf_propagation_count, 4)
        self.assertLessEqual(model._last_cf_propagation_count, 12)
        self.assertEqual(model._last_cf_propagation_count % 2, 0)
        self.assertEqual(
            model.observed_block_target_count.sum().item(), 2
        )
        self.assertEqual(
            model.observed_block_query_count.sum().item(), 6
        )
        torch.testing.assert_close(
            model.observed_block_target_sum,
            model.observed_block_add_effect_sum
            + model.observed_block_remove_effect_sum,
        )
        self.assertIsNotNone(model.causal_scorer[2].weight.grad)

        model.post_epoch_processing()
        artifacts = model.get_analysis_artifacts()
        blocks = artifacts['counterfactual_blocks']
        self.assertEqual(
            blocks['block_features'].shape,
            (model.num_semantic_blocks, 6),
        )
        self.assertIn('observed_target_mean', blocks)
        self.assertIn('observed_add_effect_mean', blocks)
        self.assertIn('observed_remove_effect_mean', blocks)
        self.assertIn('observed_query_count', blocks)
        self.assertEqual(
            artifacts['metadata']['cf_intervention_mode'],
            'semantic_block',
        )

        restored = self.block_model()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            restored.item_prototype_ids, model.item_prototype_ids
        )
        torch.testing.assert_close(
            restored.edge_block_ids, model.edge_block_ids
        )
        torch.testing.assert_close(
            restored.observed_block_target_sum,
            model.observed_block_target_sum,
        )
        torch.testing.assert_close(
            restored.observed_block_add_effect_sum,
            model.observed_block_add_effect_sum,
        )
        torch.testing.assert_close(
            restored.observed_block_remove_effect_sum,
            model.observed_block_remove_effect_sum,
        )

    def test_semantic_block_bpr_path_detaches_block_scorer(self):
        model = self.block_model(warmup=100, lambda_cf=0.0)
        model.train()
        model.pre_epoch_processing()
        loss = model.calculate_loss(self.interaction())
        loss.backward()
        for parameter in model.causal_scorer.parameters():
            self.assertIsNone(parameter.grad)
        self.assertIsNotNone(model.mask_logits.grad)

    def test_semantic_block_rejects_synergy_target(self):
        with self.assertRaisesRegex(
            ValueError, 'requires cf_target_mode'
        ):
            self.model(
                target_mode='synergy',
                intervention_mode='semantic_block',
            )

    def test_active_training_artifacts_and_checkpoint_round_trip(self):
        model = self.model(warmup=0)
        model.train()
        model.pre_epoch_processing()
        loss = model.calculate_loss(self.interaction())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(model._last_cf_propagation_count, 4)
        self.assertEqual(model.latest_loss_components['cf_samples'].item(), 2)

        model.post_epoch_processing()
        artifacts = model.get_analysis_artifacts()
        mask_artifact = artifacts['masks']['masked_branch']
        torch.testing.assert_close(
            torch.sigmoid(mask_artifact['logits']),
            mask_artifact['probabilities'],
        )
        self.assertEqual(
            mask_artifact['selected_at_keep_ratio'].sum().item(),
            model.hard_keep_count,
        )
        self.assertEqual(
            mask_artifact['causal_features'].shape,
            (model.num_interactions, 6),
        )
        self.assertIn('observed_tau_syn_mean', mask_artifact)
        self.assertIn('observed_tau_syn_count', mask_artifact)
        self.assertIn('observed_cf_target_mean', mask_artifact)
        self.assertIn('observed_cf_target_count', mask_artifact)
        self.assertEqual(mask_artifact['cf_target_mode'], 'synergy')

        restored = self.model(warmup=0)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(
            restored.effective_mask_logits(), model.effective_mask_logits()
        )
        torch.testing.assert_close(
            restored.full_uncertainty, model.full_uncertainty
        )
        torch.testing.assert_close(
            restored.full_mask_disagreement,
            model.full_mask_disagreement,
        )
        torch.testing.assert_close(
            restored.observed_tau_syn_sum,
            model.observed_tau_syn_sum,
        )
        torch.testing.assert_close(
            restored.observed_tau_syn_count,
            model.observed_tau_syn_count,
        )


if __name__ == '__main__':
    unittest.main()
