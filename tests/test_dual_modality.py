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

from models.dual_modality import DUAL_MODALITY  # noqa: E402


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
        self.interactions = sp.coo_matrix(
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
        return self.interactions.asformat(form)


class DualModalityTest(unittest.TestCase):
    @staticmethod
    def write_features(root):
        dataset_dir = Path(root) / 'toy'
        dataset_dir.mkdir()
        np.save(
            dataset_dir / 'image.npy',
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
            dataset_dir / 'text.npy',
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
        cl_weight=0.0,
        mask_graph_mode='hard',
        mask_sharing_mode='separate',
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
                'vision_feature_file': 'image.npy',
                'text_feature_file': 'text.npy',
                'embedding_size': 2,
                'feat_embed_dim': 2,
                'knn_k': 2,
                'n_mm_layers': 1,
                'n_ui_layers': 2,
                'mm_image_weight': 0.25,
                'mask_sharing_mode': mask_sharing_mode,
                'mask_graph_mode': mask_graph_mode,
                'mask_degree_mode': 'full',
                'mask_keep_ratio': 0.3,
                'hard_mask_temperature': 1.0,
                'mask_weight': 0.1,
                'mask_binary_weight': 0.1,
                'cl_weight': cl_weight,
                'cl_temperature': 0.2,
                'dropout': 0.2,
            }
        )

    def make_model(
        self,
        root,
        cl_weight=0.0,
        mask_graph_mode='hard',
        mask_sharing_mode='separate',
    ):
        return DUAL_MODALITY(
            self.make_config(
                root,
                cl_weight,
                mask_graph_mode,
                mask_sharing_mode,
            ),
            FakeTrainData(),
        )

    @staticmethod
    def interaction():
        return torch.tensor([[0, 2], [0, 3], [2, 0]], dtype=torch.long)

    def test_four_branches_use_pgl_layer_mean_and_parallel_ii(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root)
            model.eval()
            users, items = model.forward(model.norm_adj)
            representations = model.latest_representations

            self.assertEqual(tuple(users.shape), (3, 4))
            self.assertEqual(tuple(items.shape), (4, 4))
            self.assertEqual(model.final_embedding_dim, 4)
            for table in (
                model.user_image,
                model.masked_user_image,
                model.user_text,
                model.masked_user_text,
            ):
                self.assertEqual(tuple(table.weight.shape), (3, 2))

            image_features = torch.nn.functional.normalize(
                model.image_trs(model.image_embedding.weight), dim=-1
            )
            initial = torch.cat(
                (model.user_image.weight, image_features), dim=0
            )
            first = torch.sparse.mm(model.norm_adj, initial)
            second = torch.sparse.mm(model.norm_adj, first)
            expected = (initial + first + second) / 3.0
            actual = torch.cat(
                (
                    representations['image_full_users'],
                    representations['image_full_items'],
                ),
                dim=0,
            )
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                items,
                representations['ui_items'] + representations['mm_items'],
            )

    def test_image_and_text_masks_are_independent(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root)
            with torch.no_grad():
                model.image_mask_logits.copy_(
                    torch.arange(model.num_interactions, dtype=torch.float32)
                )
                model.text_mask_logits.copy_(
                    torch.arange(
                        model.num_interactions - 1,
                        -1,
                        -1,
                        dtype=torch.float32,
                    )
                )
            model.eval()
            model.post_epoch_processing()

            image_kept = set(model.image_hard_eval_indices.tolist())
            text_kept = set(model.text_hard_eval_indices.tolist())
            self.assertEqual(len(image_kept), model.hard_keep_count)
            self.assertEqual(len(text_kept), model.hard_keep_count)
            self.assertNotEqual(image_kept, text_kept)

            image_adj, _ = model._masked_ui_adjacency(
                'image', model.image_mask_logits
            )
            text_adj, _ = model._masked_ui_adjacency(
                'text', model.text_mask_logits
            )
            self.assertEqual(
                image_adj._nnz(), 2 * model.hard_keep_count
            )
            self.assertEqual(text_adj._nnz(), 2 * model.hard_keep_count)

    def test_shared_mask_uses_one_parameter_and_one_edge_selection(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root, mask_sharing_mode='shared')

            mask_parameters = {
                name: parameter
                for name, parameter in model.named_parameters()
                if 'mask_logits' in name
            }
            self.assertEqual(set(mask_parameters), {'shared_mask_logits'})
            self.assertIsNone(model.image_mask_logits)
            self.assertIsNone(model.text_mask_logits)
            self.assertIs(
                model._get_mask_logits('image'),
                model._get_mask_logits('text'),
            )

            model.pre_epoch_processing()
            image_adj, image_mask = model._masked_ui_adjacency(
                'image', model._get_mask_logits('image')
            )
            text_adj, text_mask = model._masked_ui_adjacency(
                'text', model._get_mask_logits('text')
            )
            self.assertEqual(
                model.shared_hard_train_indices.numel(),
                model.hard_keep_count,
            )
            torch.testing.assert_close(
                image_adj.coalesce().indices(),
                text_adj.coalesce().indices(),
            )
            torch.testing.assert_close(
                image_adj.coalesce().values(),
                text_adj.coalesce().values(),
            )
            torch.testing.assert_close(image_mask, text_mask)

            loss = model.calculate_loss(self.interaction())
            loss.backward()
            self.assertIsNotNone(model.shared_mask_logits.grad)
            self.assertTrue(
                torch.isfinite(model.shared_mask_logits.grad).all()
            )

            artifacts = model.get_analysis_artifacts()
            self.assertEqual(
                artifacts['metadata']['mask_sharing_mode'], 'shared'
            )
            self.assertEqual(set(artifacts['masks']), {'shared'})

            soft_model = self.make_model(
                root,
                mask_graph_mode='soft',
                mask_sharing_mode='shared',
            )
            image_adj, image_mask = soft_model._masked_ui_adjacency(
                'image', soft_model._get_mask_logits('image')
            )
            text_adj, text_mask = soft_model._masked_ui_adjacency(
                'text', soft_model._get_mask_logits('text')
            )
            torch.testing.assert_close(image_mask, text_mask)
            torch.testing.assert_close(
                image_adj.coalesce().indices(),
                text_adj.coalesce().indices(),
            )
            torch.testing.assert_close(
                image_adj.coalesce().values(),
                text_adj.coalesce().values(),
            )

    def test_loss_gradients_cl_switch_and_input_immutability(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root, cl_weight=0.0)
            interaction = self.interaction()
            original = interaction.clone()
            model.pre_epoch_processing()
            with mock.patch.object(
                model,
                'InfoNCE',
                side_effect=AssertionError('CL should be disabled'),
            ):
                loss = model.calculate_loss(interaction)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

            for parameter in (
                model.user_image.weight,
                model.masked_user_image.weight,
                model.user_text.weight,
                model.masked_user_text.weight,
                model.image_trs.weight,
                model.text_trs.weight,
                model.image_fusion_gate.weight,
                model.text_fusion_gate.weight,
                model.image_mask_logits,
                model.text_mask_logits,
            ):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            torch.testing.assert_close(interaction, original)
            self.assertEqual(
                model.latest_loss_components['contrastive'].item(), 0.0
            )

    def test_graph_mix_full_sort_artifacts_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root)
            _, image_adj = model.get_knn_adj_mat(model.v_feat)
            _, text_adj = model.get_knn_adj_mat(model.t_feat)
            expected_adj = (
                0.25 * image_adj.to_dense() + 0.75 * text_adj.to_dense()
            )
            torch.testing.assert_close(model.mm_adj.to_dense(), expected_adj)

            model.eval()
            before = model.full_sort_predict((torch.tensor([0, 1]),))
            with torch.no_grad():
                model.user_image.weight.mul_(-1.0)
            after = model.full_sort_predict((torch.tensor([0, 1]),))
            self.assertEqual(tuple(after.shape), (2, 4))
            self.assertFalse(torch.allclose(before, after))

            artifacts = model.get_analysis_artifacts()
            self.assertEqual(artifacts['metadata']['model'], 'DUAL_MODALITY')
            self.assertEqual(set(artifacts['masks']), {'image', 'text'})
            self.assertIn(
                'image_full_users', artifacts['representations']
            )
            self.assertIn(
                'text_masked_items', artifacts['representations']
            )

            restored = self.make_model(root)
            restored.load_state_dict(model.state_dict())
            restored.eval()
            actual_users, actual_items = restored.forward(restored.norm_adj)
            expected_users, expected_items = model.forward(model.norm_adj)
            torch.testing.assert_close(actual_users, expected_users)
            torch.testing.assert_close(actual_items, expected_items)


if __name__ == '__main__':
    unittest.main()
