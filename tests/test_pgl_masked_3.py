import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.pgl_masked_3 import PGL_MASKED_3  # noqa: E402


class PGLMasked3PropagationTest(unittest.TestCase):
    @staticmethod
    def make_model(n_layers=2):
        model = PGL_MASKED_3.__new__(PGL_MASKED_3)
        nn.Module.__init__(model)
        model.n_ui_layers = n_layers
        return model

    @staticmethod
    def propagate_native(adjacency, initial_embeddings, n_layers):
        layers = [initial_embeddings]
        current = initial_embeddings
        for _ in range(n_layers):
            current = torch.sparse.mm(adjacency, current)
            layers.append(current)
        return torch.stack(layers, dim=1).mean(dim=1)

    def test_custom_backward_matches_native_sparse_mm(self):
        indices = torch.tensor(
            [[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]], dtype=torch.long
        )
        initial_data = torch.tensor(
            [
                [0.2, -0.1],
                [0.5, 0.3],
                [-0.4, 0.7],
                [0.8, -0.2],
            ]
        )
        value_data = torch.tensor([0.4, 0.7, 0.2, 0.9, 0.6])

        custom_initial = initial_data.clone().requires_grad_()
        custom_values = value_data.clone().requires_grad_()
        custom_adjacency = torch.sparse_coo_tensor(
            indices, custom_values, (4, 4)
        ).coalesce()
        model = self.make_model()
        actual = model._propagate_ui_graph(
            custom_adjacency, custom_initial
        )
        actual_loss = actual.square().sum()
        actual_grad_values, actual_grad_initial = torch.autograd.grad(
            actual_loss, (custom_values, custom_initial)
        )

        native_initial = initial_data.clone().requires_grad_()
        native_values = value_data.clone().requires_grad_()
        native_adjacency = torch.sparse_coo_tensor(
            indices, native_values, (4, 4)
        ).coalesce()
        expected = self.propagate_native(
            native_adjacency, native_initial, model.n_ui_layers
        )
        expected_loss = expected.square().sum()
        expected_grad_values, expected_grad_initial = torch.autograd.grad(
            expected_loss, (native_values, native_initial)
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual_grad_values, expected_grad_values
        )
        torch.testing.assert_close(
            actual_grad_initial, expected_grad_initial
        )

    def test_static_adjacency_uses_original_propagation_result(self):
        indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
        values = torch.tensor([0.5, 0.25, 0.75])
        adjacency = torch.sparse_coo_tensor(
            indices, values, (3, 3)
        ).coalesce()
        initial = torch.randn(3, 4, requires_grad=True)

        model = self.make_model(n_layers=1)
        actual = model._propagate_ui_graph(adjacency, initial)
        expected = self.propagate_native(adjacency, initial, 1)

        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertIsNotNone(initial.grad)


if __name__ == '__main__':
    unittest.main()
