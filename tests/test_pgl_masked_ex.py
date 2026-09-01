import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.pgl_masked_ex import PGL_MASKED_EX  # noqa: E402


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


class TestablePGLMaskedEx(PGL_MASKED_EX):
    def _build_or_load_mm_graph(self, config):
        indices = torch.arange(self.n_items).repeat(2, 1)
        values = torch.ones(self.n_items)
        adjacency = torch.sparse_coo_tensor(
            indices, values, (self.n_items, self.n_items)
        ).coalesce()
        self.register_buffer('mm_adj', adjacency)


def make_score_model(
    residual=(0.1, -0.4, 0.2),
    visual=(0.2, 0.8, -0.1),
    textual=(0.6, -0.2, 0.3),
    variant='prior_residual',
    lambda_sem=2.0,
    lambda_r=0.25,
    residual_temperature=1.0,
    keep_ratio=0.3,
):
    """Build the mask-specific part of the model without dataset I/O."""
    model = PGL_MASKED_EX.__new__(PGL_MASKED_EX)
    nn.Module.__init__(model)
    model.semantic_mask_variant = variant
    model.lambda_sem = lambda_sem
    model.lambda_r = lambda_r
    model.residual_temperature = residual_temperature
    model.semantic_gamma = nn.Parameter(torch.zeros(2))
    model.mask_logits = nn.Parameter(torch.tensor(residual, dtype=torch.float32))
    model.register_buffer(
        'semantic_visual_affinity',
        torch.tensor(visual, dtype=torch.float32),
    )
    model.register_buffer(
        'semantic_textual_affinity',
        torch.tensor(textual, dtype=torch.float32),
    )
    model.num_interactions = len(residual)
    model.mask_keep_ratio = keep_ratio
    model.hard_mask_temperature = 1.0
    model.mask_graph_mode = 'hard'
    model.register_buffer(
        'hard_train_indices',
        torch.empty(0, dtype=torch.long),
        persistent=False,
    )
    model.register_buffer(
        'hard_eval_indices',
        torch.empty(0, dtype=torch.long),
        persistent=False,
    )
    if variant == 'semantic_only':
        model.mask_logits.requires_grad_(False)
    return model


class LeaveOneOutAffinityTest(unittest.TestCase):
    def test_manual_affinities_degree_one_and_zero_vectors(self):
        features = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        users = torch.tensor([0, 0, 0, 1, 2, 2])
        items = torch.tensor([0, 1, 2, 0, 3, 0])

        affinity = PGL_MASKED_EX._leave_one_out_affinity(
            features, users, items, num_users=3, chunk_size=2
        )
        expected = torch.tensor([
            2.0 ** -0.5,
            2.0 ** -0.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ])

        torch.testing.assert_close(affinity, expected, atol=1e-6, rtol=0)
        self.assertTrue(torch.isfinite(affinity).all())


class SemanticScoreTest(unittest.TestCase):
    def test_prior_residual_formula_weights_and_gradients(self):
        model = make_score_model()

        alpha = model.semantic_weights()
        prior = model.semantic_prior()
        scores = model.effective_mask_logits()

        torch.testing.assert_close(alpha, torch.tensor([0.5, 0.5]))
        torch.testing.assert_close(alpha.sum(), torch.tensor(1.0))
        torch.testing.assert_close(prior, torch.tensor([0.4, 0.3, 0.1]))
        torch.testing.assert_close(scores, torch.tensor([0.9, 0.2, 0.4]))

        scores.sum().backward()
        torch.testing.assert_close(model.mask_logits.grad, torch.ones(3))
        self.assertIsNotNone(model.semantic_gamma.grad)
        self.assertFalse(model.semantic_visual_affinity.requires_grad)
        self.assertFalse(model.semantic_textual_affinity.requires_grad)

    def test_semantic_only_ignores_and_freezes_residual(self):
        model = make_score_model(variant='semantic_only')
        scores = model.effective_mask_logits()

        torch.testing.assert_close(scores, torch.tensor([0.8, 0.6, 0.2]))
        self.assertFalse(model.mask_logits.requires_grad)
        scores.sum().backward()
        self.assertIsNone(model.mask_logits.grad)
        self.assertIsNotNone(model.semantic_gamma.grad)

    def test_zero_lambda_matches_residual_mask(self):
        model = make_score_model(lambda_sem=0.0)
        torch.testing.assert_close(
            model.effective_mask_logits(), model.mask_logits
        )

    def test_bounded_residual_formula_bound_and_gradients(self):
        model = make_score_model(
            residual=(0.0, 0.25, -0.25),
            variant='bounded_residual',
            lambda_r=0.25,
            residual_temperature=0.5,
        )
        prior = model.semantic_prior()
        correction = model.effective_mask_logits() - prior
        expected_correction = 0.25 * torch.tanh(
            torch.tensor([0.0, 0.5, -0.5])
        )

        torch.testing.assert_close(correction, expected_correction)
        self.assertLessEqual(correction.abs().max().item(), 0.25)
        model.effective_mask_logits().sum().backward()
        self.assertIsNotNone(model.mask_logits.grad)
        self.assertTrue(torch.isfinite(model.mask_logits.grad).all())
        self.assertIsNotNone(model.semantic_gamma.grad)

    def test_hard_eval_uses_effective_score_and_floor_keep_count(self):
        model = make_score_model(
            residual=(0.9, 0.1, 0.0, -0.1, -0.2),
            visual=(0.0, 1.0, 0.0, 0.0, 0.0),
            textual=(0.0, 1.0, 0.0, 0.0, 0.0),
            lambda_sem=2.0,
            keep_ratio=0.34,
        )
        model.eval()
        model.post_epoch_processing()

        self.assertEqual(model.hard_keep_count, 1)
        self.assertEqual(model.hard_eval_indices.numel(), 1)
        self.assertEqual(model.hard_eval_indices.item(), 1)

    def test_soft_adjacency_uses_sigmoid_of_effective_score(self):
        model = make_score_model(
            residual=(0.0, 0.0),
            visual=(1.0, -1.0),
            textual=(1.0, -1.0),
            lambda_sem=1.0,
            keep_ratio=0.5,
        )
        model.mask_graph_mode = 'soft'
        model.mask_degree_mode = 'full'
        model.n_nodes = 4
        model.ui_edge_index = torch.tensor([
            [0, 1, 2, 3],
            [2, 3, 0, 1],
        ])
        model.full_norm_edge_weights = torch.ones(4)

        adjacency, interaction_mask = model._masked_ui_adjacency()
        expected_mask = torch.sigmoid(torch.tensor([1.0, -1.0]))
        dense = adjacency.to_dense()

        torch.testing.assert_close(interaction_mask, expected_mask)
        torch.testing.assert_close(dense[0, 2], expected_mask[0])
        torch.testing.assert_close(dense[2, 0], expected_mask[0])
        torch.testing.assert_close(dense[1, 3], expected_mask[1])
        torch.testing.assert_close(dense[3, 1], expected_mask[1])

    def test_semantic_parameters_and_buffers_round_trip_in_state_dict(self):
        source = make_score_model()
        target = make_score_model(
            residual=(0.0, 0.0, 0.0),
            visual=(0.0, 0.0, 0.0),
            textual=(0.0, 0.0, 0.0),
        )
        with torch.no_grad():
            source.semantic_gamma.copy_(torch.tensor([0.7, -0.3]))

        target.load_state_dict(source.state_dict())

        torch.testing.assert_close(
            target.effective_mask_logits(), source.effective_mask_logits()
        )
        torch.testing.assert_close(
            target.semantic_visual_affinity,
            source.semantic_visual_affinity,
        )
        torch.testing.assert_close(
            target.semantic_textual_affinity,
            source.semantic_textual_affinity,
        )


class ModelIntegrationTest(unittest.TestCase):
    def make_config(self, root, graph_mode='hard', variant='prior_residual'):
        return NullableConfig({
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'NEG_PREFIX': 'neg_',
            'train_batch_size': 2,
            'device': torch.device('cpu'),
            'end2end': False,
            'is_multimodal_model': True,
            'data_path': str(root) + os.sep,
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
            'mask_graph_mode': graph_mode,
            'user_embedding_mode': 'separate',
            'ui_branch_mode': 'dual',
            'semantic_mask_variant': variant,
            'lambda_sem': 1.0,
            'lambda_r': 0.25,
            'residual_temperature': 1.0,
            'cl_weight': 0.05,
            'cl_temperature': 0.2,
            'dropout': 0.0,
            'mask_weight': 0.0,
        })

    def write_features(self, root):
        dataset_dir = Path(root) / 'toy'
        dataset_dir.mkdir()
        image_features = np.array([
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        text_features = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.2, 0.8],
        ], dtype=np.float32)
        np.save(dataset_dir / 'image_feat.npy', image_features)
        np.save(dataset_dir / 'text_feat.npy', text_features)

    def test_hard_smoke_loss_artifacts_and_checkpoint_restore(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            self.write_features(temporary_root)
            config = self.make_config(temporary_root)
            train_data = FakeTrainData()
            model = TestablePGLMaskedEx(config, train_data)

            model.train()
            model.pre_epoch_processing()
            interaction = (
                torch.tensor([0, 2]),
                torch.tensor([0, 1]),
                torch.tensor([2, 3]),
            )
            loss = model.calculate_loss(interaction)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.mask_logits.grad)
            self.assertIsNotNone(model.semantic_gamma.grad)

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
            self.assertIn('residual_mask_logits', mask_artifact)
            self.assertIn('semantic_prior', mask_artifact)
            self.assertIn('bounded_residual_correction', mask_artifact)
            self.assertEqual(mask_artifact['lambda_r'], 0.25)
            self.assertEqual(mask_artifact['residual_temperature'], 1.0)

            restored = TestablePGLMaskedEx(config, train_data)
            restored.load_state_dict(model.state_dict())
            restored.post_epoch_processing()
            torch.testing.assert_close(
                restored.effective_mask_logits(),
                model.effective_mask_logits(),
            )
            self.assertEqual(
                restored.hard_eval_indices.numel(),
                restored.hard_keep_count,
            )

    def test_soft_forward_and_semantic_only_freezes_residual(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            self.write_features(temporary_root)
            config = self.make_config(
                temporary_root,
                graph_mode='soft',
                variant='semantic_only',
            )
            model = TestablePGLMaskedEx(config, FakeTrainData())
            users, items = model.forward()

            self.assertEqual(tuple(users.shape), (3, 4))
            self.assertEqual(tuple(items.shape), (4, 4))
            self.assertFalse(model.mask_logits.requires_grad)
            self.assertTrue(torch.isfinite(users).all())
            self.assertTrue(torch.isfinite(items).all())

    def test_bounded_residual_starts_from_semantic_prior(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            self.write_features(temporary_root)
            config = self.make_config(
                temporary_root,
                graph_mode='hard',
                variant='bounded_residual',
            )
            model = TestablePGLMaskedEx(config, FakeTrainData())

            torch.testing.assert_close(
                model.mask_logits,
                torch.zeros_like(model.mask_logits),
            )
            torch.testing.assert_close(
                model.effective_mask_logits(), model.semantic_prior()
            )


if __name__ == '__main__':
    unittest.main()
