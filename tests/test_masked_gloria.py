import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.masked_gloria import Base_gcn, GCN, MASKED_GLORIA  # noqa: E402


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
                np.ones(6, dtype=np.float32),
                (
                    np.array([0, 0, 1, 2, 2, 2]),
                    np.array([0, 1, 2, 1, 3, 3]),
                ),
            ),
            shape=(3, 4),
        )

    def inter_matrix(self, form='coo'):
        return self._interactions.asformat(form)


class MaskedGloriaTest(unittest.TestCase):
    @staticmethod
    def write_features(root):
        dataset_dir = Path(root) / 'toy'
        dataset_dir.mkdir()
        np.save(
            dataset_dir / 'image_feat.npy',
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.8, 0.2, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        np.save(
            dataset_dir / 'text_feat.npy',
            np.array(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.2, 0.8],
                ],
                dtype=np.float32,
            ),
        )

    @staticmethod
    def make_config(
        root,
        image_weight=0.25,
        cl_weight=0.0,
        dropout=0.2,
        item_embedding_mode='id',
        mm_propagation_mode='sequential',
        fusion='concat',
    ):
        return NullableConfig(
            {
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
                'mm_image_weight': image_weight,
                'aggr_mode': 'add',
                'fusion': fusion,
                'mask_keep_ratio': 0.3,
                'mask_weight': 0.1,
                'mask_binary_weight': 0.1,
                'cl_weight': cl_weight,
                'cl_temperature': 0.2,
                'dropout': dropout,
                'item_embedding_mode': item_embedding_mode,
                'mm_propagation_mode': mm_propagation_mode,
            }
        )

    def make_model(
        self,
        root,
        image_weight=0.25,
        cl_weight=0.0,
        dropout=0.2,
        item_embedding_mode='id',
        mm_propagation_mode='sequential',
        fusion='concat',
    ):
        return MASKED_GLORIA(
            self.make_config(
                root,
                image_weight,
                cl_weight,
                dropout,
                item_embedding_mode,
                mm_propagation_mode,
                fusion,
            ),
            FakeTrainData(),
        )

    @staticmethod
    def training_interaction():
        return torch.tensor([[0, 2], [0, 3], [2, 0]], dtype=torch.long)

    @staticmethod
    def load_cache(file_path):
        try:
            return torch.load(file_path, map_location='cpu', weights_only=True)
        except TypeError:
            return torch.load(file_path, map_location='cpu')

    def test_forward_loss_gradient_and_input_immutability(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root)
            interaction = self.training_interaction()
            original_interaction = interaction.clone()

            positive_scores, negative_scores = model.forward(interaction)
            self.assertEqual(tuple(positive_scores.shape), (2,))
            self.assertEqual(tuple(negative_scores.shape), (2,))
            torch.testing.assert_close(interaction, original_interaction)

            representations = model._encode()
            self.assertEqual(tuple(representations['users'].shape), (3, 4))
            self.assertEqual(tuple(representations['items'].shape), (4, 4))

            loss = model.calculate_loss(interaction)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.mask_logits.grad)
            self.assertTrue(torch.isfinite(model.mask_logits.grad).all())
            self.assertGreater(model.mask_logits.grad.abs().sum().item(), 0.0)
            self.assertIsNotNone(model.id_embedding_full.weight.grad)
            torch.testing.assert_close(interaction, original_interaction)

    def test_cl_weight_zero_skips_infonce(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root, cl_weight=0.0)
            with mock.patch.object(
                model,
                'InfoNCE',
                side_effect=AssertionError('InfoNCE should be disabled'),
            ):
                loss = model.calculate_loss(self.training_interaction())

            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(
                model.latest_loss_components['contrastive'].item(), 0.0
            )

    def test_positive_cl_weight_adds_original_pgl_infonce(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root, cl_weight=0.5, dropout=0.0)
            loss = model.calculate_loss(self.training_interaction())
            components = model.latest_loss_components

            self.assertTrue(torch.isfinite(components['contrastive']))
            self.assertGreater(components['contrastive'].item(), 0.0)
            expected = (
                components['bpr']
                + 0.5 * components['contrastive']
                + 0.1 * components['mask']
            )
            torch.testing.assert_close(loss.detach(), expected)
            loss.backward()
            self.assertIsNotNone(model.mask_logits.grad)

    def test_infonce_matches_original_pgl_formula(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root)
            first = torch.tensor([[1.0, 0.0], [0.2, 0.8]])
            second = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
            temperature = 0.2

            normalized_first = torch.nn.functional.normalize(first, dim=1)
            normalized_second = torch.nn.functional.normalize(second, dim=1)
            positive_scores = torch.exp(
                (normalized_first * normalized_second).sum(dim=1)
                / temperature
            )
            total_scores = torch.exp(
                normalized_first @ normalized_second.transpose(0, 1)
                / temperature
            ).sum(dim=1)
            expected = -torch.log(positive_scores / total_scores).mean()
            actual = model.InfoNCE(first, second, temperature)
            torch.testing.assert_close(actual, expected)

    def test_three_hop_gcn_matches_dense_reference(self):
        features = torch.tensor([[0.6, 0.8], [0.8, -0.6]])
        model = GCN(
            datasets=None,
            batch_size=2,
            num_user=2,
            num_item=2,
            dim_id=2,
            aggr_mode='add',
            num_layer=3,
            has_feature=False,
            dropout=0.0,
            dim_latent=2,
            device=torch.device('cpu'),
            features=features,
        )
        with torch.no_grad():
            model.preference.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            )

        edge_index = torch.tensor(
            [[0, 0, 1, 2, 3], [2, 3, 2, 0, 1]], dtype=torch.long
        )
        edge_norm = torch.tensor([0.5, 0.25, 0.75, 0.5, 0.25])
        edge_mask = torch.tensor([0.7, 0.4, 0.9, 0.7, 0.4])
        actual, _ = model(
            edge_index,
            features,
            edge_mask=edge_mask,
            edge_norm=edge_norm,
        )

        initial = torch.cat((model.preference, features), dim=0)
        initial = torch.nn.functional.normalize(initial, dim=-1)
        source, target = edge_index
        adjacency = torch.zeros(4, 4)
        adjacency.index_put_(
            (target, source), edge_norm * edge_mask, accumulate=True
        )
        first = adjacency @ initial
        second = adjacency @ first
        third = adjacency @ second
        expected = initial + first + second + third
        torch.testing.assert_close(actual, expected)

    def test_edge_propagation_handles_large_sparse_node_space(self):
        num_nodes = 100_000
        features = torch.zeros(num_nodes, 2)
        features[0] = torch.tensor([1.0, 2.0])
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long
        )
        convolution = Base_gcn(2, 2, aggr='add')
        output = convolution(features, edge_index)
        self.assertEqual(tuple(output.shape), (num_nodes, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_multimodal_graph_is_weighted_image_text_sum(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root, image_weight=0.25)
            _, image_adj = model.get_knn_adj_mat(model.v_feat)
            _, text_adj = model.get_knn_adj_mat(model.t_feat)
            expected = 0.25 * image_adj.to_dense() + 0.75 * text_adj.to_dense()
            torch.testing.assert_close(model.mm_adj.to_dense(), expected)

    def test_multimodal_sequential_and_parallel_routing(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            parallel = self.make_model(
                root,
                item_embedding_mode='multimodal',
                mm_propagation_mode='parallel',
            )
            sequential = self.make_model(
                root,
                item_embedding_mode='multimodal',
                mm_propagation_mode='sequential',
            )
            sequential.load_state_dict(parallel.state_dict())

            parallel_rep = parallel._encode()
            sequential_rep = sequential._encode()

            self.assertIsNone(parallel.id_embedding_full)
            self.assertIsNone(parallel.id_embedding_masked)
            self.assertIsNone(parallel.full_gcn.preference)
            self.assertIsNone(parallel.mask_gcn.preference)
            self.assertEqual(tuple(parallel.user_image.weight.shape), (3, 2))
            self.assertEqual(tuple(parallel.user_text.weight.shape), (3, 2))
            self.assertEqual(
                tuple(parallel_rep['multimodal_items'].shape), (4, 4)
            )
            self.assertEqual(tuple(parallel_rep['users'].shape), (3, 8))
            self.assertEqual(tuple(parallel_rep['items'].shape), (4, 8))

            # Both modes have identical U-I representations; only the I-I
            # routing after those representations is changed.
            torch.testing.assert_close(
                parallel_rep['full_items'], sequential_rep['full_items']
            )
            torch.testing.assert_close(
                parallel_rep['masked_items'], sequential_rep['masked_items']
            )

            parallel_mm = parallel._propagate_item_graph(
                parallel_rep['multimodal_items']
            )
            expected_parallel = torch.cat(
                (
                    parallel_rep['full_items'] + parallel_mm,
                    parallel_rep['masked_items'] + parallel_mm,
                ),
                dim=1,
            )
            torch.testing.assert_close(
                parallel_rep['mm_items'], parallel_mm
            )
            torch.testing.assert_close(
                parallel_rep['items'], expected_parallel
            )

            full_sequential_mm = sequential._propagate_item_graph(
                sequential_rep['full_items']
            )
            masked_sequential_mm = sequential._propagate_item_graph(
                sequential_rep['masked_items']
            )
            expected_sequential = torch.cat(
                (
                    sequential_rep['full_items'] + full_sequential_mm,
                    sequential_rep['masked_items'] + masked_sequential_mm,
                ),
                dim=1,
            )
            self.assertIsNone(sequential_rep['mm_items'])
            torch.testing.assert_close(
                sequential_rep['full_mm_items'], full_sequential_mm
            )
            torch.testing.assert_close(
                sequential_rep['masked_mm_items'], masked_sequential_mm
            )
            torch.testing.assert_close(
                sequential_rep['items'], expected_sequential
            )

            users_before = parallel_rep['full_users'].detach().clone()
            with torch.no_grad():
                parallel.image_embedding.weight[0].mul_(-1.0)
            users_after = parallel._encode()['full_users']
            self.assertFalse(torch.allclose(users_before, users_after))

            loss = parallel.calculate_loss(self.training_interaction())
            loss.backward()
            for parameter in (
                parallel.image_embedding.weight,
                parallel.text_embedding.weight,
                parallel.image_trs.weight,
                parallel.text_trs.weight,
                parallel.user_image.weight,
                parallel.user_text.weight,
            ):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

            artifacts = parallel.get_analysis_artifacts()
            self.assertEqual(
                artifacts['metadata']['item_embedding_mode'], 'multimodal'
            )
            self.assertEqual(
                artifacts['metadata']['mm_propagation_mode'], 'parallel'
            )
            self.assertIn(
                'image_embedding.weight', artifacts['embedding_tables']
            )
            self.assertIn(
                'text_embedding.weight', artifacts['embedding_tables']
            )
            self.assertIn('user_image.weight', artifacts['embedding_tables'])
            self.assertIn('user_text.weight', artifacts['embedding_tables'])

            expected = parallel._encode()
            restored = self.make_model(
                root,
                item_embedding_mode='multimodal',
                mm_propagation_mode='parallel',
            )
            restored.load_state_dict(parallel.state_dict())
            actual = restored._encode()
            torch.testing.assert_close(actual['users'], expected['users'])
            torch.testing.assert_close(actual['items'], expected['items'])

    def test_sequential_gated_sum_fuses_users_and_post_ii_items(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(
                root,
                item_embedding_mode='multimodal',
                mm_propagation_mode='sequential',
                fusion='gated_sum',
            )
            representations = model._encode()

            # d=2 per modality, so each branch and both final tables are 2d=4.
            self.assertEqual(model.branch_embedding_dim, 4)
            self.assertEqual(model.final_embedding_dim, 4)
            self.assertEqual(tuple(representations['users'].shape), (3, 4))
            self.assertEqual(tuple(representations['items'].shape), (4, 4))

            expected_full_items = (
                representations['full_items']
                + model._propagate_item_graph(
                    representations['full_items']
                )
            )
            expected_masked_items = (
                representations['masked_items']
                + model._propagate_item_graph(
                    representations['masked_items']
                )
            )
            item_gate = torch.sigmoid(
                model.fusion_gate(
                    torch.cat(
                        (expected_full_items, expected_masked_items), dim=1
                    )
                )
            )
            expected_items = (
                item_gate * expected_full_items
                + (1.0 - item_gate) * expected_masked_items
            )
            torch.testing.assert_close(
                representations['item_fusion_gate'], item_gate
            )
            torch.testing.assert_close(
                representations['items'], expected_items
            )

            user_gate = torch.sigmoid(
                model.fusion_gate(
                    torch.cat(
                        (
                            representations['full_users'],
                            representations['masked_users'],
                        ),
                        dim=1,
                    )
                )
            )
            expected_users = (
                user_gate * representations['full_users']
                + (1.0 - user_gate) * representations['masked_users']
            )
            torch.testing.assert_close(
                representations['user_fusion_gate'], user_gate
            )
            torch.testing.assert_close(
                representations['users'], expected_users
            )

            loss = model.calculate_loss(self.training_interaction())
            loss.backward()
            self.assertIsNotNone(model.fusion_gate.weight.grad)
            self.assertTrue(
                torch.isfinite(model.fusion_gate.weight.grad).all()
            )
            scores = model.full_sort_predict((torch.tensor([0, 1]),))
            self.assertEqual(tuple(scores.shape), (2, 4))

            artifacts = model.get_analysis_artifacts()
            self.assertEqual(
                artifacts['metadata']['ui_fusion_mode'], 'gated_sum'
            )
            self.assertIn(
                'user_fusion_gate', artifacts['representations']
            )
            self.assertIn(
                'item_fusion_gate', artifacts['representations']
            )

            restored = self.make_model(
                root,
                item_embedding_mode='multimodal',
                mm_propagation_mode='sequential',
                fusion='gated_sum',
            )
            restored.load_state_dict(model.state_dict())
            restored_rep = restored._encode()
            expected_rep = model._encode()
            torch.testing.assert_close(
                restored_rep['users'], expected_rep['users']
            )
            torch.testing.assert_close(
                restored_rep['items'], expected_rep['items']
            )

    def test_valid_cache_is_reused_and_stale_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root)
            cache_file = model.mm_cache_file

            with mock.patch.object(
                MASKED_GLORIA,
                'get_knn_adj_mat',
                side_effect=AssertionError('valid cache was not reused'),
            ):
                restored_from_cache = self.make_model(root)
            torch.testing.assert_close(
                restored_from_cache.mm_adj.to_dense(), model.mm_adj.to_dense()
            )

            payload = self.load_cache(cache_file)
            payload['metadata']['version'] = -1
            torch.save(payload, cache_file)
            rebuilt = self.make_model(root)
            rebuilt_payload = self.load_cache(rebuilt.mm_cache_file)
            self.assertEqual(
                rebuilt_payload['metadata']['version'],
                MASKED_GLORIA.MM_CACHE_VERSION,
            )

    def test_full_sort_recomputes_embeddings(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root)
            evaluation_interaction = (torch.tensor([0, 1]),)
            before = model.full_sort_predict(evaluation_interaction)

            with torch.no_grad():
                model.full_gcn.preference.mul_(-1.0)
            after = model.full_sort_predict(evaluation_interaction)

            self.assertEqual(tuple(after.shape), (2, 4))
            self.assertFalse(torch.allclose(before, after))

    def test_artifacts_and_state_dict_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root)
            artifacts = model.get_analysis_artifacts()

            self.assertEqual(artifacts['metadata']['model'], 'MASKED_GLORIA')
            self.assertEqual(
                artifacts['ui_edges']['user_ids'].numel(),
                model.num_interactions,
            )
            mask_artifact = artifacts['masks']['masked_branch']
            self.assertEqual(mask_artifact['logits'].numel(), model.num_interactions)
            self.assertEqual(
                int(mask_artifact['selected_at_keep_ratio'].sum()),
                max(1, round(model.num_interactions * model.mask_keep_ratio)),
            )
            self.assertIn('users', artifacts['representations'])
            self.assertIn(
                'id_embedding_full.weight', artifacts['embedding_tables']
            )

            checkpoint_file = Path(root) / 'masked_gloria-checkpoint.pt'
            torch.save(
                {'model_state_dict': model.state_dict()}, checkpoint_file
            )
            checkpoint = self.load_cache(checkpoint_file)
            restored = self.make_model(root)
            restored.load_state_dict(checkpoint['model_state_dict'])
            interaction = self.training_interaction()
            actual = model.forward(interaction)
            expected = restored.forward(interaction)
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])

    def test_feature_count_mismatch_fails_clearly(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            feature_file = Path(root) / 'toy' / 'image_feat.npy'
            np.save(feature_file, np.ones((3, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'Visual feature count'):
                self.make_model(root)


if __name__ == '__main__':
    unittest.main()
