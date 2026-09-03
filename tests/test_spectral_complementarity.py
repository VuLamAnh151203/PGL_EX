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
    extract_graph_views,
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
                    "--output-json",
                    str(output_path),
                ])
            self.assertEqual(exit_code, 0)
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
            for metric in baseline["metrics"]:
                self.assertGreaterEqual(metric["one_sided_empirical_p"], 0.0)
                self.assertLessEqual(metric["one_sided_empirical_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
