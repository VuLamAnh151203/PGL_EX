"""PGL_MASKED with the original sparse forward and a memory-safe backward.

The native backward of ``torch.sparse.mm`` can materialize a dense gradient
with respect to a differentiable sparse adjacency.  For a user-item graph,
that intermediate has shape ``(n_users + n_items) ** 2`` even though mask
parameters exist only on observed edges.

This module keeps the original ``torch.sparse.mm`` forward.  Its custom
backward evaluates adjacency gradients only at stored COO edges and computes
embedding gradients with another sparse matrix multiplication.  Consequently
the mathematical forward and gradients stay unchanged while the dense
``n_nodes x n_nodes`` allocation is avoided.
"""

import torch

from models.pgl_masked import PGL_MASKED


class _ObservedEdgeSparseMM(torch.autograd.Function):
    """Sparse matrix multiplication with edge-only adjacency gradients."""

    @staticmethod
    def forward(ctx, indices, values, size, embeddings):
        adjacency = torch.sparse_coo_tensor(
            indices,
            values,
            size,
            dtype=values.dtype,
            device=values.device,
        ).coalesce()

        ctx.adjacency_size = tuple(size)
        ctx.save_for_backward(
            adjacency.indices(), adjacency.values(), embeddings
        )
        return torch.sparse.mm(adjacency, embeddings)

    @staticmethod
    def backward(ctx, grad_output):
        indices, values, embeddings = ctx.saved_tensors
        row, col = indices

        grad_values = None
        if ctx.needs_input_grad[1]:
            output_gradients = grad_output.index_select(0, row)
            source_embeddings = embeddings.index_select(0, col)
            grad_values = (output_gradients * source_embeddings).sum(dim=1)

        grad_embeddings = None
        if ctx.needs_input_grad[3]:
            transpose_indices = torch.stack((col, row), dim=0)
            transpose_size = (
                ctx.adjacency_size[1], ctx.adjacency_size[0]
            )
            transpose_adjacency = torch.sparse_coo_tensor(
                transpose_indices,
                values,
                transpose_size,
                dtype=values.dtype,
                device=values.device,
            ).coalesce()
            grad_embeddings = torch.sparse.mm(
                transpose_adjacency, grad_output
            )

        return None, grad_values, None, grad_embeddings


class PGL_MASKED_3(PGL_MASKED):
    """Closest-to-original PGL_MASKED variant without dense mask gradients."""

    @staticmethod
    def _memory_safe_sparse_mm(adjacency, embeddings):
        adjacency = adjacency.coalesce()
        return _ObservedEdgeSparseMM.apply(
            adjacency.indices(),
            adjacency.values(),
            tuple(adjacency.shape),
            embeddings,
        )

    def _propagate_ui_graph(self, adjacency, initial_embeddings):
        """Use custom backward only when adjacency values need gradients."""
        adjacency = adjacency.coalesce()
        differentiable_adjacency = adjacency.requires_grad

        embeddings = [initial_embeddings]
        current_embeddings = initial_embeddings
        for _ in range(self.n_ui_layers):
            if differentiable_adjacency:
                current_embeddings = self._memory_safe_sparse_mm(
                    adjacency, current_embeddings
                )
            else:
                current_embeddings = torch.sparse.mm(
                    adjacency, current_embeddings
                )
            embeddings.append(current_embeddings)

        return torch.stack(embeddings, dim=1).mean(dim=1)
