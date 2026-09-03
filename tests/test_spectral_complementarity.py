import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mask_analysis.spectral_complementarity import (  # noqa: E402
    analyze_spectral_complementarity,
    analyze_user_representations,
    extract_graph_views,
    extract_pre_propagation_user_representations,
    extract_user_representations,
    main,
    validate_k_values,
)


def make_artifact():
    probabilities = torch.tensor([0.2, 0.4, 0.7, 0.8, 0.6, 0.3])
    return {
        "metadata": {
            "num_users": 4,
            "num_items": 5,
            "num_interactions": 6,
        },
        "ui_edges": {
            "user_ids": torch.tensor([0, 0, 1, 2, 2, 3]),
            "item_ids": torch.tensor([0, 1, 1, 2, 3, 4]),
        },
        "masks": {
            "masked_branch": {
                "logits": torch.logit(probabilities),
                "probabilities": probabilities,
                "selected_at_keep_ratio": torch.tensor(
                    [False, False, True, True, True, False]
                ),
            }
        },
        "representations": {
            "full_users": torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]),
            "masked_users": torch.tensor([
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.1, 0.9],
                [0.8, 1.0, 0.9],
            ]),
        },
        "embedding_tables": {
            "user_text.weight": torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]),
            "second_user_text.weight": torch.tensor([
                [0.8, 0.2, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.1, 0.9],
                [0.9, 0.8, 1.0],
            ]),
            "user_image.weight": torch.tensor([
                [0.5, 0.2, 0.1],
                [0.1, 0.6, 0.2],
                [0.2, 0.1, 0.7],
                [0.8, 0.7, 0.9],
            ]),
            "second_user_image.weight": torch.tensor([
                [0.6, 0.1, 0.2],
                [0.2, 0.5, 0.1],
                [0.1, 0.2, 0.8],
                [0.7, 0.9, 0.8],
            ]),
        },
    }


class SpectralComplementarityTest(unittest.TestCase):
    def setUp(self):
        self.original_dense = np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        self.weighted_dense = self.original_dense * np.array(
            [
                [0.2, 0.8, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.9, 0.4],
                [0.6, 0.0, 0.0, 0.25, 0.0],
            ]
        )

    def test_sparse_energy_matches_full_svd(self):
        result = analyze_spectral_complementarity(
            sp.csr_matrix(self.original_dense),
            sp.csr_matrix(self.weighted_dense),
            k_values=(1, 2, 3),
            seed=7,
            tolerance=1e-12,
        )
        original_singular = np.linalg.svd(
            self.original_dense, compute_uv=False
        )
        weighted_singular = np.linalg.svd(
            self.weighted_dense, compute_uv=False
        )
        for metric in result["metrics"]:
            k = metric["k"]
            expected_original = np.square(original_singular[:k]).sum()
            expected_original /= np.square(self.original_dense).sum()
            expected_weighted = np.square(weighted_singular[:k]).sum()
            expected_weighted /= np.square(self.weighted_dense).sum()
            self.assertAlmostEqual(
                metric["original_spectral_energy"], expected_original, places=10
            )
            self.assertAlmostEqual(
                metric["weighted_spectral_energy"], expected_weighted, places=10
            )
        original_global_bands = [
            band["original_global_energy_contribution"]
            for band in result["spectral_bands"]
        ]
        weighted_global_bands = [
            band["weighted_global_energy_contribution"]
            for band in result["spectral_bands"]
        ]
        self.assertAlmostEqual(
            sum(original_global_bands),
            result["metrics"][-1]["original_spectral_energy"],
            places=10,
        )
        self.assertAlmostEqual(
            sum(weighted_global_bands),
            result["metrics"][-1]["weighted_spectral_energy"],
            places=10,
        )
        self.assertAlmostEqual(
            sum(
                band["original_within_top_3_share"]
                for band in result["spectral_bands"]
            ),
            1.0,
            places=10,
        )
        self.assertAlmostEqual(
            sum(
                band["weighted_within_top_3_share"]
                for band in result["spectral_bands"]
            ),
            1.0,
            places=10,
        )

    def test_uniform_scaling_has_identical_spectrum_and_subspaces(self):
        original = sp.csr_matrix(self.original_dense)
        result = analyze_spectral_complementarity(
            original,
            0.35 * original,
            k_values=(1, 2, 3),
            seed=3,
            tolerance=1e-12,
        )
        for metric in result["metrics"]:
            self.assertAlmostEqual(
                metric["original_spectral_energy"],
                metric["weighted_spectral_energy"],
                places=10,
            )
            self.assertAlmostEqual(metric["complementarity"], 0.0, places=10)

    def test_nonuniform_result_metrics_are_bounded(self):
        result = analyze_spectral_complementarity(
            sp.csr_matrix(self.original_dense),
            sp.csr_matrix(self.weighted_dense),
            k_values=(1, 2, 3),
            seed=11,
        )
        bounded_fields = (
            "original_spectral_energy",
            "weighted_spectral_energy",
            "user_subspace_overlap",
            "item_subspace_overlap",
            "complementarity",
        )
        for metric in result["metrics"]:
            for field in bounded_fields:
                self.assertGreaterEqual(metric[field], 0.0)
                self.assertLessEqual(metric[field], 1.0)

    def test_extract_graph_views_builds_expected_matrices(self):
        views = extract_graph_views(make_artifact())
        self.assertEqual(views.original.shape, (4, 5))
        self.assertEqual(views.original.nnz, 6)
        np.testing.assert_allclose(views.original.data, np.ones(6))
        np.testing.assert_allclose(
            np.sort(views.weighted.data),
            np.sort(np.array([0.2, 0.4, 0.7, 0.8, 0.6, 0.3])),
            rtol=1e-6,
        )
        self.assertEqual(views.hard_masked.nnz, 3)
        np.testing.assert_allclose(views.hard_masked.data, np.ones(3))

    def test_extract_and_analyze_user_representations(self):
        full_users, masked_users = extract_user_representations(make_artifact())
        self.assertEqual(full_users.shape, (4, 3))
        self.assertEqual(masked_users.shape, (4, 3))

        result = analyze_user_representations(
            full_users, masked_users, k_values=(1, 2)
        )
        self.assertEqual(result["matrix_shape"], [4, 3])
        self.assertEqual(len(result["metrics"]), 2)
        self.assertEqual(len(result["spectral_bands"]), 2)
        self.assertEqual(
            result["paired_user_similarity"]["valid_user_count"], 4
        )
        for metric in result["metrics"]:
            for field in (
                "full_spectral_energy",
                "masked_spectral_energy",
                "user_subspace_overlap",
                "feature_subspace_overlap",
                "complementarity",
            ):
                self.assertGreaterEqual(metric[field], 0.0)
                self.assertLessEqual(metric[field], 1.0)

        pre_propagation = extract_pre_propagation_user_representations(
            make_artifact()
        )
        self.assertEqual(set(pre_propagation), {"text", "image"})
        self.assertEqual(pre_propagation["text"][0].shape, (4, 3))
        self.assertEqual(pre_propagation["image"][1].shape, (4, 3))

        full_rank = analyze_user_representations(
            pre_propagation["text"][0],
            pre_propagation["text"][1],
            k_values=(1, 2, 3),
        )
        self.assertAlmostEqual(
            full_rank["metrics"][-1]["full_spectral_energy"], 1.0
        )
        self.assertAlmostEqual(
            full_rank["metrics"][-1]["masked_spectral_energy"], 1.0
        )

    def test_identical_user_representations_have_no_complementarity(self):
        users = np.random.RandomState(17).standard_normal((12, 5))
        result = analyze_user_representations(
            users, users.copy(), k_values=(1, 2, 4)
        )
        for metric in result["metrics"]:
            self.assertAlmostEqual(metric["spectral_energy_delta"], 0.0, places=12)
            self.assertAlmostEqual(metric["complementarity"], 0.0, places=12)
        self.assertAlmostEqual(
            result["paired_user_similarity"]["cosine_similarity"]["mean"],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            result["global_similarity"]["centered_linear_cka"],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            result["global_similarity"][
                "centered_orthogonal_procrustes_similarity"
            ],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            result["global_similarity"][
                "centered_scaled_procrustes_relative_error"
            ],
            0.0,
            places=12,
        )

    def test_cka_and_scaled_procrustes_ignore_rotation_and_scale(self):
        random_generator = np.random.RandomState(29)
        full_users = random_generator.standard_normal((20, 5))
        rotation, _ = np.linalg.qr(random_generator.standard_normal((5, 5)))
        masked_users = 0.3 * full_users @ rotation
        result = analyze_user_representations(
            full_users, masked_users, k_values=(1, 2, 4)
        )
        self.assertAlmostEqual(
            result["global_similarity"]["centered_linear_cka"],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            result["global_similarity"][
                "centered_orthogonal_procrustes_similarity"
            ],
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            result["global_similarity"][
                "centered_scaled_procrustes_optimal_scale"
            ],
            0.3,
            places=12,
        )
        self.assertAlmostEqual(
            result["global_similarity"][
                "centered_scaled_procrustes_relative_error"
            ],
            0.0,
            places=12,
        )

    def test_user_representation_validation_rejects_missing_or_invalid(self):
        missing = make_artifact()
        del missing["representations"]["masked_users"]
        with self.assertRaisesRegex(ValueError, "masked_users"):
            extract_user_representations(missing)

        invalid = make_artifact()
        invalid["representations"]["masked_users"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            extract_user_representations(invalid)

        missing_table = make_artifact()
        del missing_table["embedding_tables"]["second_user_image.weight"]
        with self.assertRaisesRegex(ValueError, "second_user_image"):
            extract_pre_propagation_user_representations(missing_table)

    def test_artifact_validation_rejects_invalid_values(self):
        cases = []

        missing_branch = make_artifact()
        missing_branch["masks"] = {}
        cases.append((missing_branch, "was not found"))

        wrong_length = make_artifact()
        wrong_length["masks"]["masked_branch"]["probabilities"] = torch.ones(5)
        cases.append((wrong_length, "does not match the edge count"))

        invalid_id = make_artifact()
        invalid_id["ui_edges"]["item_ids"][0] = 5
        cases.append((invalid_id, "outside"))

        nonfinite = make_artifact()
        nonfinite["masks"]["masked_branch"]["probabilities"][0] = float("nan")
        cases.append((nonfinite, "finite"))

        wrong_hard_length = make_artifact()
        wrong_hard_length["masks"]["masked_branch"][
            "selected_at_keep_ratio"
        ] = torch.ones(5, dtype=torch.bool)
        cases.append((wrong_hard_length, "hard-selection count"))

        for artifact, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    extract_graph_views(artifact)

    def test_invalid_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "smaller than"):
            validate_k_values((1, 4), (4, 5))
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_k_values((0, 1), (4, 5))

    def test_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "analysis.pt"
            output_path = Path(temporary_directory) / "spectral.json"
            heatmap_path = Path(temporary_directory) / "transitions.png"
            torch.save(make_artifact(), artifact_path)
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main([
                    "--analysis-file",
                    str(artifact_path),
                    "--k-values",
                    "1",
                    "2",
                    "--include-hard",
                    "--random-baseline-runs",
                    "3",
                    "--random-baseline-seed",
                    "13",
                    "--cluster-transition-heatmap",
                    str(heatmap_path),
                    "--num-clusters",
                    "2",
                    "--analyze-user-embeddings",
                    "--output-json",
                    str(output_path),
                ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(heatmap_path.is_file())
            self.assertGreater(heatmap_path.stat().st_size, 0)
            with output_path.open("r", encoding="utf-8") as result_file:
                result = json.load(result_file)
            self.assertEqual(result["k_values"], [1, 2])
            self.assertEqual(result["branch"], "masked_branch")
            self.assertEqual(result["weighted_graph_definition"],
                             "binary_adjacency * sigmoid(mask_logit)")
            self.assertEqual(
                result["hard_masked_analysis"]["num_selected_edges"], 3
            )
            self.assertEqual(
                result["hard_masked_analysis"]["weighted_graph_definition"],
                "binary_adjacency * selected_at_keep_ratio",
            )
            baseline = result["random_hard_baseline"]
            self.assertEqual(baseline["sampling"], "uniform_without_replacement")
            self.assertEqual(baseline["num_runs"], 3)
            self.assertEqual(baseline["random_seed"], 13)
            self.assertEqual(baseline["num_selected_edges_per_run"], 3)
            self.assertEqual(len(baseline["metrics"]), 2)
            self.assertEqual(len(baseline["spectral_bands"]), 2)
            self.assertEqual(result["cluster_transition"]["num_clusters"], 2)
            self.assertEqual(
                len(result["cluster_transition"]["user"]["row_percentages"]),
                2,
            )
            embedding_result = result["user_embedding_analysis"]
            self.assertEqual(embedding_result["matrix_shape"], [4, 3])
            self.assertEqual(embedding_result["k_values"], [1, 2])
            self.assertEqual(len(embedding_result["metrics"]), 2)
            pre_propagation = result[
                "pre_propagation_user_embedding_analysis"
            ]
            self.assertEqual(pre_propagation["text"]["matrix_shape"], [4, 3])
            self.assertEqual(pre_propagation["image"]["k_values"], [1, 2])
            for metric in baseline["metrics"]:
                self.assertGreaterEqual(metric["one_sided_empirical_p"], 0.0)
                self.assertLessEqual(metric["one_sided_empirical_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
