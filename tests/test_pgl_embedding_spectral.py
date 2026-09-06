import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mask_analysis.pgl_embedding_spectral import (  # noqa: E402
    analyze_embedding_spectrum,
    build_normalized_ui_adjacency,
    load_embedding_tensor,
    main,
    reconstruct_post_propagation_users,
)


class PGLEmbeddingSpectralTest(unittest.TestCase):
    def test_exact_top_k_and_band_contributions(self):
        matrix = np.diag([4.0, 3.0, 2.0, 1.0])
        result = analyze_embedding_spectrum(matrix, k_values=[1, 2, 4])

        self.assertEqual(result["matrix_shape"], [4, 4])
        np.testing.assert_allclose(
            result["svd"]["singular_values"], [4.0, 3.0, 2.0, 1.0]
        )
        contributions = [
            row["spectral_energy_contribution"]
            for row in result["top_k_spectral_energy"]
        ]
        np.testing.assert_allclose(contributions, [16 / 30, 25 / 30, 1.0])
        global_bands = [
            row["global_energy_contribution"]
            for row in result["spectral_bands"]
        ]
        np.testing.assert_allclose(global_bands, [16 / 30, 9 / 30, 5 / 30])
        within_bands = [
            row["within_top_4_share"] for row in result["spectral_bands"]
        ]
        np.testing.assert_allclose(within_bands, global_bands)

    def test_reconstruction_matches_manual_pgl_propagation(self):
        state = {
            "user_image.weight": torch.tensor([[1.0], [2.0]]),
            "user_text.weight": torch.tensor([[3.0], [4.0]]),
            "image_embedding.weight": torch.tensor([[1.0], [2.0]]),
            "text_embedding.weight": torch.tensor([[3.0], [4.0]]),
            "image_trs.weight": torch.tensor([[1.0]]),
            "image_trs.bias": torch.tensor([0.0]),
            "text_trs.weight": torch.tensor([[1.0]]),
            "text_trs.bias": torch.tensor([0.0]),
        }
        users = np.array([0, 1], dtype=np.int64)
        items = np.array([0, 1], dtype=np.int64)
        actual = reconstruct_post_propagation_users(
            state, users, items, n_ui_layers=1, device=torch.device("cpu")
        )

        adjacency = build_normalized_ui_adjacency(
            users, items, 2, 2, torch.device("cpu")
        )
        initial_users = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        initial_items = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        initial = torch.cat((initial_users, initial_items), dim=0)
        expected = (initial + torch.sparse.mm(adjacency, initial)) / 2.0
        torch.testing.assert_close(actual, expected[:2])

    def test_embedding_file_cli_writes_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            embedding_file = directory / "users.pt"
            output_file = directory / "spectrum.json"
            torch.save(torch.eye(4), embedding_file)

            with contextlib.redirect_stdout(io.StringIO()):
                return_code = main([
                    "--embedding-file",
                    str(embedding_file),
                    "--k-values",
                    "1",
                    "2",
                    "4",
                    "--output-json",
                    str(output_file),
                ])

            self.assertEqual(return_code, 0)
            with output_file.open("r", encoding="utf-8") as input_file:
                result = json.load(input_file)
            self.assertEqual(result["matrix_shape"], [4, 4])
            self.assertEqual(len(result["spectral_bands"]), 3)

    def test_load_nested_embedding_tensor(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "artifact.pt"
            expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)
            torch.save({"representations": {"full_users": expected}}, path)
            actual = load_embedding_tensor(path, "representations.full_users")
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
