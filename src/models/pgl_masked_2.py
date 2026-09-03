"""Memory-safe variant of :mod:`models.pgl_masked`.

``torch.sparse.mm`` can materialize an ``n_nodes x n_nodes`` intermediate
while differentiating with respect to sparse adjacency values.  The masked
branch has learnable edge values, so large datasets can run out of memory in
backward even though only observed user-item edges are parameterized.

This variant preserves the original model and replaces only that
differentiable sparse matrix multiplication with edge-wise message passing.
Static adjacencies continue to use ``torch.sparse.mm``.
"""

import torch

from models.pgl_masked import PGL_MASKED


class PGL_MASKED_2(PGL_MASKED):
    """PGL_MASKED with memory-safe gradients for learnable graph edges."""

    @staticmethod
    def _edgewise_sparse_mm(adjacency, embeddings):
        """Compute ``adjacency @ embeddings`` using only stored COO edges.

        Unlike the backward implementation used by ``torch.sparse.mm`` for a
        differentiable sparse operand, this expression keeps both activation
        memory and edge-weight gradients proportional to the number of stored
        edges instead of ``n_nodes ** 2``.
        """
        adjacency = adjacency.coalesce()
        row, col = adjacency.indices()
        messages = (
            adjacency.values().unsqueeze(-1)
            * embeddings.index_select(0, col)
        )
        output = embeddings.new_zeros(
            (adjacency.size(0), embeddings.size(1))
        )
        return output.index_add(0, row, messages)

    def _propagate_ui_graph(self, adjacency, initial_embeddings):
        """Propagate without dense adjacency gradients on masked branches."""
        adjacency = adjacency.coalesce()
        differentiable_adjacency = adjacency.requires_grad

        embeddings = [initial_embeddings]
        current_embeddings = initial_embeddings
        for _ in range(self.n_ui_layers):
            if differentiable_adjacency:
                current_embeddings = self._edgewise_sparse_mm(
                    adjacency, current_embeddings
                )
            else:
                current_embeddings = torch.sparse.mm(
                    adjacency, current_embeddings
                )
            embeddings.append(current_embeddings)

        return torch.stack(embeddings, dim=1).mean(dim=1)
