import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.pgl_masked_2 import PGL_MASKED_2  # noqa: E402


class PGLMasked2PropagationTest(unittest.TestCase):
    @staticmethod
    def make_model(n_layers=2):
        model = PGL_MASKED_2.__new__(PGL_MASKED_2)
        nn.Module.__init__(model)
        model.n_ui_layers = n_layers
        return model

    @staticmethod
    def make_dense_adjacency(indices, values, size):
        adjacency = values.new_zeros(size)
        return adjacency.index_put(
            (indices[0], indices[1]), values, accumulate=True
        )

    def test_differentiable_sparse_propagation_matches_dense_reference(self):
        indices = torch.tensor(
            [[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]], dtype=torch.long
        )
        initial = torch.tensor(
            [
                [0.2, -0.1],
                [0.5, 0.3],
                [-0.4, 0.7],
                [0.8, -0.2],
            ],
            requires_grad=True,
        )
        edge_values = torch.tensor(
            [0.4, 0.7, 0.2, 0.9, 0.6], requires_grad=True
        )
        adjacency = torch.sparse_coo_tensor(
            indices, edge_values, (4, 4)
        ).coalesce()

        model = self.make_model()
        actual = model._propagate_ui_graph(adjacency, initial)
        actual_loss = actual.square().sum()
        actual_grad_values, actual_grad_initial = torch.autograd.grad(
            actual_loss, (edge_values, initial)
        )

        reference_values = edge_values.detach().clone().requires_grad_()
        reference_initial = initial.detach().clone().requires_grad_()
        dense_adjacency = self.make_dense_adjacency(
            indices, reference_values, (4, 4)
        )
        layers = [reference_initial]
        current = reference_initial
        for _ in range(model.n_ui_layers):
            current = dense_adjacency @ current
            layers.append(current)
        expected = torch.stack(layers, dim=1).mean(dim=1)
        expected_loss = expected.square().sum()
        expected_grad_values, expected_grad_initial = torch.autograd.grad(
            expected_loss, (reference_values, reference_initial)
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual_grad_values, expected_grad_values
        )
        torch.testing.assert_close(
            actual_grad_initial, expected_grad_initial
        )

    def test_static_adjacency_still_propagates_without_edge_gradients(self):
        indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
        values = torch.tensor([0.5, 0.25, 0.75])
        adjacency = torch.sparse_coo_tensor(
            indices, values, (3, 3)
        ).coalesce()
        initial = torch.randn(3, 4, requires_grad=True)

        output = self.make_model(n_layers=1)._propagate_ui_graph(
            adjacency, initial
        )
        expected = torch.stack(
            (initial, adjacency.to_dense() @ initial), dim=1
        ).mean(dim=1)

        torch.testing.assert_close(output, expected)
        output.sum().backward()
        self.assertIsNotNone(initial.grad)


if __name__ == '__main__':
    unittest.main()
