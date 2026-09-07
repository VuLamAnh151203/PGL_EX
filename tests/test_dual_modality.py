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
from mask_analysis.dual_modality_diagnostics import (  # noqa: E402
    summarize_rankings,
)


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
        fusion_gate_mode='separate',
        cl_mode='pgl_dropout',
        aux_bpr_mode='none',
        aux_bpr_weight=0.0,
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
                'fusion_gate_mode': fusion_gate_mode,
                'mask_graph_mode': mask_graph_mode,
                'mask_degree_mode': 'full',
                'mask_keep_ratio': 0.3,
                'hard_mask_temperature': 1.0,
                'mask_weight': 0.1,
                'mask_binary_weight': 0.1,
                'cl_weight': cl_weight,
                'cl_mode': cl_mode,
                'aux_bpr_mode': aux_bpr_mode,
                'aux_bpr_weight': aux_bpr_weight,
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
        fusion_gate_mode='separate',
        cl_mode='pgl_dropout',
        aux_bpr_mode='none',
        aux_bpr_weight=0.0,
    ):
        return DUAL_MODALITY(
            self.make_config(
                root,
                cl_weight,
                mask_graph_mode,
                mask_sharing_mode,
                fusion_gate_mode,
                cl_mode,
                aux_bpr_mode,
                aux_bpr_weight,
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

    def test_shared_fusion_gate_matches_joint_full_masked_formula(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(
                root,
                mask_sharing_mode='separate',
                fusion_gate_mode='shared',
            )
            self.assertIsNone(model.image_fusion_gate)
            self.assertIsNone(model.text_fusion_gate)
            self.assertEqual(
                tuple(model.shared_fusion_gate.weight.shape), (4, 8)
            )
            self.assertIsNot(
                model.image_mask_logits, model.text_mask_logits
            )

            model.eval()
            users, _ = model.forward(model.norm_adj)
            representations = model.latest_representations
            full_users = torch.cat(
                (
                    representations['image_full_users'],
                    representations['text_full_users'],
                ),
                dim=1,
            )
            masked_users = torch.cat(
                (
                    representations['image_masked_users'],
                    representations['text_masked_users'],
                ),
                dim=1,
            )
            expected_user_gate = torch.sigmoid(
                model.shared_fusion_gate(
                    torch.cat((full_users, masked_users), dim=1)
                )
            )
            expected_users = (
                expected_user_gate * full_users
                + (1.0 - expected_user_gate) * masked_users
            )
            torch.testing.assert_close(users, expected_users)
            torch.testing.assert_close(
                representations['shared_user_gate'], expected_user_gate
            )

            full_items = torch.cat(
                (
                    representations['image_full_items'],
                    representations['text_full_items'],
                ),
                dim=1,
            )
            masked_items = torch.cat(
                (
                    representations['image_masked_items'],
                    representations['text_masked_items'],
                ),
                dim=1,
            )
            expected_item_gate = torch.sigmoid(
                model.shared_fusion_gate(
                    torch.cat((full_items, masked_items), dim=1)
                )
            )
            expected_ui_items = (
                expected_item_gate * full_items
                + (1.0 - expected_item_gate) * masked_items
            )
            torch.testing.assert_close(
                representations['ui_items'], expected_ui_items
            )

            model.train()
            loss = model.calculate_loss(self.interaction())
            loss.backward()
            for parameter in (
                model.shared_fusion_gate.weight,
                model.image_mask_logits,
                model.text_mask_logits,
            ):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

            artifacts = model.get_analysis_artifacts()
            self.assertEqual(
                artifacts['metadata']['fusion_gate_mode'], 'shared'
            )
            self.assertEqual(set(artifacts['masks']), {'image', 'text'})

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

    def test_full_masked_concat_cl_uses_concatenated_branch_views(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(
                root,
                cl_weight=0.1,
                mask_sharing_mode='separate',
                fusion_gate_mode='shared',
                cl_mode='full_masked_concat',
            )
            interaction = self.interaction()
            original_info_nce = model.InfoNCE
            with mock.patch.object(
                model.dropoutf,
                'forward',
                side_effect=AssertionError(
                    'Branch-view CL must not create dropout views.'
                ),
            ), mock.patch.object(
                model, 'InfoNCE', wraps=original_info_nce
            ) as info_nce:
                loss = model.calculate_loss(interaction)

            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(info_nce.call_count, 2)
            representations = model.latest_representations
            user_call = info_nce.call_args_list[0].args
            item_call = info_nce.call_args_list[1].args
            torch.testing.assert_close(
                user_call[0],
                representations['full_users'][interaction[0]],
            )
            torch.testing.assert_close(
                user_call[1],
                representations['masked_users'][interaction[0]],
            )
            torch.testing.assert_close(
                item_call[0],
                representations['full_items'][interaction[1]],
            )
            torch.testing.assert_close(
                item_call[1],
                representations['masked_items'][interaction[1]],
            )
            self.assertEqual(user_call[0].shape[1], 4)
            self.assertEqual(item_call[0].shape[1], 4)

            loss.backward()
            self.assertTrue(
                torch.isfinite(model.image_mask_logits.grad).all()
            )
            self.assertTrue(
                torch.isfinite(model.text_mask_logits.grad).all()
            )
            artifacts = model.get_analysis_artifacts()
            self.assertEqual(
                artifacts['metadata']['cl_mode'], 'full_masked_concat'
            )

    def test_modality_aux_bpr_and_score_diagnostics(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(
                root,
                cl_weight=0.0,
                mask_sharing_mode='separate',
                fusion_gate_mode='shared',
                aux_bpr_mode='modality',
                aux_bpr_weight=0.4,
            )
            interaction = self.interaction()
            loss = model.calculate_loss(interaction)
            representations = model.latest_representations

            image_loss = model.bpr_loss(
                representations['image_users'][interaction[0]],
                representations['image_items'][interaction[1]],
                representations['image_items'][interaction[2]],
            )
            text_loss = model.bpr_loss(
                representations['text_users'][interaction[0]],
                representations['text_items'][interaction[1]],
                representations['text_items'][interaction[2]],
            )
            torch.testing.assert_close(
                model.latest_loss_components['image_bpr'],
                image_loss.detach(),
            )
            torch.testing.assert_close(
                model.latest_loss_components['text_bpr'],
                text_loss.detach(),
            )
            torch.testing.assert_close(
                model.latest_loss_components['aux_bpr'],
                (0.5 * (image_loss + text_loss)).detach(),
            )
            expected_total = (
                model.latest_loss_components['bpr']
                + 0.4 * model.latest_loss_components['aux_bpr']
                + model.mask_weight
                * model.latest_loss_components['mask']
            )
            torch.testing.assert_close(loss.detach(), expected_total)

            scores = model.full_sort_predict_modalities(
                (interaction[0],)
            )
            torch.testing.assert_close(
                scores['joint'], scores['image'] + scores['text']
            )
            torch.testing.assert_close(
                scores['joint'],
                model.full_sort_predict((interaction[0],)),
            )

            margins = model.modality_triplet_margins(interaction)
            batch_rows = torch.arange(interaction.shape[1])
            for modality in ('image', 'text', 'joint'):
                expected_margin = (
                    scores[modality][batch_rows, interaction[1]]
                    - scores[modality][batch_rows, interaction[2]]
                )
                torch.testing.assert_close(
                    margins[modality], expected_margin
                )

            loss.backward()
            self.assertTrue(torch.isfinite(model.image_trs.weight.grad).all())
            self.assertTrue(torch.isfinite(model.text_trs.weight.grad).all())
            artifacts = model.get_analysis_artifacts()
            self.assertEqual(
                artifacts['metadata']['aux_bpr_mode'], 'modality'
            )
            self.assertEqual(
                artifacts['metadata']['aux_bpr_weight'], 0.4
            )

    def test_post_training_ranking_summary_reports_rescue_and_harm(self):
        positive_items = [np.array([1]), np.array([2])]
        rankings = {
            'image': np.array([[1, 0], [0, 2]]),
            'text': np.array([[0, 1], [2, 0]]),
            'joint': np.array([[1, 0], [0, 2]]),
        }
        result = summarize_rankings(rankings, positive_items, [1, 2])
        self.assertEqual(result['ranking_metrics']['joint']['Recall@1'], 0.5)
        top1 = result['complementarity']['Top1']
        self.assertEqual(top1['image_rescue_count'], 1)
        self.assertEqual(top1['text_miss_count'], 1)
        self.assertEqual(top1['image_rescue_rate_among_text_misses'], 1.0)
        self.assertEqual(top1['image_harm_count'], 1)
        self.assertEqual(top1['text_hit_count'], 1)
        self.assertEqual(top1['image_harm_rate_among_text_hits'], 1.0)

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
