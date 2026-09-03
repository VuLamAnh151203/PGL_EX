"""Measure spectral concentration and complementarity of learned U-I weights.

The original view is the binary user-item interaction matrix. The weighted
view has the same support and uses ``sigmoid(mask_logit)`` as each edge value.
This script intentionally analyzes the rectangular user-item matrices rather
than the symmetric propagation adjacency, and it does not apply degree
normalization.

Example:
    python src/mask_analysis/spectral_complementarity.py \
        --analysis-file src/saved/PGL_MASKED-baby-...-analysis.pt \
        --analyze-user-embeddings \
        --output-json src/saved/PGL_MASKED-baby-spectral.json
"""

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence, svds


DEFAULT_K_VALUES = (8, 16, 32, 64)


@dataclass(frozen=True)
class GraphViews:
    """Validated graph information extracted from an analysis artifact."""

    original: sp.csr_matrix
    weighted: sp.csr_matrix
    hard_masked: Optional[sp.csr_matrix]
    probabilities: np.ndarray
    hard_selection: Optional[np.ndarray]
    metadata: Dict[str, Any]
    branch: str


@dataclass(frozen=True)
class _SpectralDecomposition:
    left: np.ndarray
    singular_values: np.ndarray
    right_transposed: np.ndarray
    frobenius_energy: float


def _as_numpy_vector(value: Any, field_name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim != 1:
        raise ValueError("{} must be a one-dimensional vector.".format(field_name))
    return value


def _as_numpy_matrix(value: Any, field_name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim != 2:
        raise ValueError("{} must be a two-dimensional matrix.".format(field_name))
    return value.astype(np.float64, copy=False)


def _positive_metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("metadata.{} must be a positive integer.".format(key))
    value = int(value)
    if value <= 0:
        raise ValueError("metadata.{} must be a positive integer.".format(key))
    return value


def _validate_edge_ids(ids: np.ndarray, upper_bound: int, name: str) -> np.ndarray:
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("{} must contain integer IDs.".format(name))
    ids = ids.astype(np.int64, copy=False)
    if ids.size and (ids.min() < 0 or ids.max() >= upper_bound):
        raise ValueError(
            "{} contains an ID outside [0, {}).".format(name, upper_bound)
        )
    return ids


def extract_graph_views(
    artifact: Mapping[str, Any], branch: str = "masked_branch"
) -> GraphViews:
    """Validate an artifact and construct binary and probability-weighted views."""
    if not isinstance(artifact, Mapping):
        raise ValueError("The analysis artifact must contain a mapping.")

    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("The analysis artifact is missing metadata.")
    num_users = _positive_metadata_int(metadata, "num_users")
    num_items = _positive_metadata_int(metadata, "num_items")

    ui_edges = artifact.get("ui_edges")
    if not isinstance(ui_edges, Mapping):
        raise ValueError("The analysis artifact is missing ui_edges.")
    if "user_ids" not in ui_edges or "item_ids" not in ui_edges:
        raise ValueError("ui_edges must contain user_ids and item_ids.")
    user_ids = _validate_edge_ids(
        _as_numpy_vector(ui_edges["user_ids"], "ui_edges.user_ids"),
        num_users,
        "ui_edges.user_ids",
    )
    item_ids = _validate_edge_ids(
        _as_numpy_vector(ui_edges["item_ids"], "ui_edges.item_ids"),
        num_items,
        "ui_edges.item_ids",
    )
    if user_ids.size != item_ids.size:
        raise ValueError("user_ids and item_ids must have the same length.")
    if user_ids.size == 0:
        raise ValueError("The analysis artifact contains no user-item edges.")

    masks = artifact.get("masks")
    if not isinstance(masks, Mapping) or branch not in masks:
        available = sorted(masks) if isinstance(masks, Mapping) else []
        raise ValueError(
            "Mask branch {!r} was not found. Available branches: {}.".format(
                branch, available
            )
        )
    branch_data = masks[branch]
    if not isinstance(branch_data, Mapping) or "probabilities" not in branch_data:
        raise ValueError(
            "Mask branch {!r} does not contain probabilities.".format(branch)
        )
    probabilities = _as_numpy_vector(
        branch_data["probabilities"],
        "masks.{}.probabilities".format(branch),
    ).astype(np.float64, copy=False)
    if probabilities.size != user_ids.size:
        raise ValueError(
            "The probability count ({}) does not match the edge count ({}).".format(
                probabilities.size, user_ids.size
            )
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Mask probabilities must all be finite.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Mask probabilities must lie in [0, 1].")
    if not np.any(probabilities > 0.0):
        raise ValueError("The weighted graph must have positive Frobenius energy.")

    if "logits" in branch_data:
        logits = _as_numpy_vector(
            branch_data["logits"], "masks.{}.logits".format(branch)
        ).astype(np.float64, copy=False)
        if logits.size != probabilities.size:
            raise ValueError(
                "The logit count ({}) does not match the probability count ({}).".format(
                    logits.size, probabilities.size
                )
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError("Mask logits must all be finite.")
        expected_probabilities = np.empty_like(logits)
        nonnegative = logits >= 0.0
        expected_probabilities[nonnegative] = 1.0 / (
            1.0 + np.exp(-logits[nonnegative])
        )
        exp_logits = np.exp(logits[~nonnegative])
        expected_probabilities[~nonnegative] = exp_logits / (1.0 + exp_logits)
        if not np.allclose(
            probabilities, expected_probabilities, rtol=1e-5, atol=1e-7
        ):
            raise ValueError(
                "Stored probabilities do not match sigmoid(logits) for branch {!r}.".format(
                    branch
                )
            )

    hard_selection = None
    if "selected_at_keep_ratio" in branch_data:
        hard_selection = _as_numpy_vector(
            branch_data["selected_at_keep_ratio"],
            "masks.{}.selected_at_keep_ratio".format(branch),
        )
        if hard_selection.size != user_ids.size:
            raise ValueError(
                "The hard-selection count ({}) does not match the edge count ({}).".format(
                    hard_selection.size, user_ids.size
                )
            )
        if not (
            np.issubdtype(hard_selection.dtype, np.bool_)
            or np.issubdtype(hard_selection.dtype, np.integer)
        ):
            raise ValueError("Hard-selection values must be boolean or binary integers.")
        if not np.all((hard_selection == 0) | (hard_selection == 1)):
            raise ValueError("Hard-selection values must all be either zero or one.")
        hard_selection = hard_selection.astype(bool, copy=False)
        if not np.any(hard_selection):
            raise ValueError("The hard-masked graph must retain at least one edge.")

    declared_edges = metadata.get("num_interactions")
    if declared_edges is not None:
        if isinstance(declared_edges, bool) or not isinstance(
            declared_edges, (int, np.integer)
        ):
            raise ValueError("metadata.num_interactions must be an integer.")
        if int(declared_edges) != user_ids.size:
            raise ValueError(
                "metadata.num_interactions ({}) does not match the edge count ({}).".format(
                    declared_edges, user_ids.size
                )
            )

    linear_edge_ids = user_ids * np.int64(num_items) + item_ids
    if np.unique(linear_edge_ids).size != linear_edge_ids.size:
        raise ValueError("ui_edges contains duplicate user-item pairs.")

    shape = (num_users, num_items)
    coordinates = (user_ids, item_ids)
    original = sp.coo_matrix(
        (np.ones(user_ids.size, dtype=np.float64), coordinates), shape=shape
    ).tocsr()
    weighted = sp.coo_matrix(
        (probabilities, coordinates), shape=shape
    ).tocsr()
    hard_masked = None
    if hard_selection is not None:
        hard_coordinates = (user_ids[hard_selection], item_ids[hard_selection])
        hard_masked = sp.coo_matrix(
            (
                np.ones(int(hard_selection.sum()), dtype=np.float64),
                hard_coordinates,
            ),
            shape=shape,
        ).tocsr()
    return GraphViews(
        original=original,
        weighted=weighted,
        hard_masked=hard_masked,
        probabilities=probabilities,
        hard_selection=hard_selection,
        metadata=dict(metadata),
        branch=branch,
    )


def _load_analysis_artifact(analysis_file: Path) -> Mapping[str, Any]:
    """Load and minimally validate a PyTorch analysis artifact."""
    analysis_file = Path(analysis_file)
    if not analysis_file.is_file():
        raise ValueError("Analysis file does not exist: {}".format(analysis_file))
    try:
        artifact = torch.load(
            str(analysis_file), map_location="cpu", weights_only=True
        )
    except TypeError:
        # Compatibility with the older PyTorch versions supported by the repo.
        artifact = torch.load(str(analysis_file), map_location="cpu")
    if not isinstance(artifact, Mapping):
        raise ValueError("The analysis artifact must contain a mapping.")
    return artifact


def load_graph_views(
    analysis_file: Path, branch: str = "masked_branch"
) -> GraphViews:
    """Load a PyTorch analysis artifact and return its graph views."""
    artifact = _load_analysis_artifact(analysis_file)
    return extract_graph_views(artifact, branch)


def extract_user_representations(
    artifact: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract aligned full/masked user representations from an artifact."""
    if not isinstance(artifact, Mapping):
        raise ValueError("The analysis artifact must contain a mapping.")
    representations = artifact.get("representations")
    if not isinstance(representations, Mapping):
        raise ValueError("The analysis artifact is missing representations.")

    required_keys = ("full_users", "masked_users")
    missing = [key for key in required_keys if key not in representations]
    if missing:
        raise ValueError(
            "representations is missing: {}.".format(", ".join(missing))
        )
    full_users = _as_numpy_matrix(
        representations["full_users"], "representations.full_users"
    )
    masked_users = _as_numpy_matrix(
        representations["masked_users"], "representations.masked_users"
    )
    if full_users.shape != masked_users.shape:
        raise ValueError(
            "representations.full_users and representations.masked_users "
            "must have the same shape."
        )
    if min(full_users.shape) <= 1:
        raise ValueError("User representation matrices must be non-trivial.")
    if not np.all(np.isfinite(full_users)):
        raise ValueError("representations.full_users must contain finite values.")
    if not np.all(np.isfinite(masked_users)):
        raise ValueError("representations.masked_users must contain finite values.")
    if np.linalg.norm(full_users) <= 0.0:
        raise ValueError("representations.full_users must have positive energy.")
    if np.linalg.norm(masked_users) <= 0.0:
        raise ValueError("representations.masked_users must have positive energy.")

    metadata = artifact.get("metadata")
    if isinstance(metadata, Mapping) and "num_users" in metadata:
        num_users = _positive_metadata_int(metadata, "num_users")
        if full_users.shape[0] != num_users:
            raise ValueError(
                "The number of representation rows ({}) does not match "
                "metadata.num_users ({}).".format(full_users.shape[0], num_users)
            )
    return full_users, masked_users


def load_user_representations(
    analysis_file: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load full_users and masked_users without retaining the whole artifact."""
    artifact = _load_analysis_artifact(analysis_file)
    return extract_user_representations(artifact)


def extract_pre_propagation_user_representations(
    artifact: Mapping[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract learned full/masked user tables before graph propagation."""
    if not isinstance(artifact, Mapping):
        raise ValueError("The analysis artifact must contain a mapping.")
    embedding_tables = artifact.get("embedding_tables")
    if not isinstance(embedding_tables, Mapping):
        raise ValueError("The analysis artifact is missing embedding_tables.")

    pairs = {}
    for modality in ("text", "image"):
        full_key = "user_{}.weight".format(modality)
        masked_key = "second_user_{}.weight".format(modality)
        missing = [
            key for key in (full_key, masked_key) if key not in embedding_tables
        ]
        if missing:
            raise ValueError(
                "embedding_tables is missing: {}.".format(", ".join(missing))
            )
        full_users = _as_numpy_matrix(
            embedding_tables[full_key],
            "embedding_tables.{}".format(full_key),
        )
        masked_users = _as_numpy_matrix(
            embedding_tables[masked_key],
            "embedding_tables.{}".format(masked_key),
        )
        if full_users.shape != masked_users.shape:
            raise ValueError(
                "embedding_tables.{} and embedding_tables.{} must have the "
                "same shape.".format(full_key, masked_key)
            )
        if min(full_users.shape) <= 1:
            raise ValueError(
                "Pre-propagation {} user embedding matrices must be "
                "non-trivial.".format(modality)
            )
        if not np.all(np.isfinite(full_users)) or not np.all(
            np.isfinite(masked_users)
        ):
            raise ValueError(
                "Pre-propagation {} user embeddings must contain only "
                "finite values.".format(modality)
            )
        if (
            np.linalg.norm(full_users) <= 0.0
            or np.linalg.norm(masked_users) <= 0.0
        ):
            raise ValueError(
                "Pre-propagation {} user embeddings must have positive "
                "energy.".format(modality)
            )
        pairs[modality] = (full_users, masked_users)

    metadata = artifact.get("metadata")
    if isinstance(metadata, Mapping) and "num_users" in metadata:
        num_users = _positive_metadata_int(metadata, "num_users")
        for modality, (full_users, _) in pairs.items():
            if full_users.shape[0] != num_users:
                raise ValueError(
                    "The {} embedding row count ({}) does not match "
                    "metadata.num_users ({}).".format(
                        modality, full_users.shape[0], num_users
                    )
                )
    return pairs


def validate_k_values(k_values: Iterable[int], shape: Tuple[int, int]) -> Tuple[int, ...]:
    """Return sorted unique ranks accepted by scipy.sparse.linalg.svds."""
    try:
        values = tuple(k_values)
    except TypeError as error:
        raise ValueError("k_values must be an iterable of integers.") from error
    if not values:
        raise ValueError("At least one k value is required.")
    if any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in values):
        raise ValueError("Every k value must be an integer.")
    values = tuple(sorted(set(int(k) for k in values)))
    if values[0] <= 0:
        raise ValueError("Every k value must be positive.")
    dimension_limit = min(shape)
    if values[-1] >= dimension_limit:
        raise ValueError(
            "Every k value must be smaller than min(matrix.shape)={}.".format(
                dimension_limit
            )
        )
    return values


def _validate_dense_k_values(
    k_values: Iterable[int], shape: Tuple[int, int]
) -> Tuple[int, ...]:
    """Validate ranks for an exact dense SVD, including the full rank."""
    try:
        values = tuple(k_values)
    except TypeError as error:
        raise ValueError("k_values must be an iterable of integers.") from error
    if not values:
        raise ValueError("At least one k value is required.")
    if any(
        isinstance(k, bool) or not isinstance(k, (int, np.integer))
        for k in values
    ):
        raise ValueError("Every k value must be an integer.")
    values = tuple(sorted(set(int(k) for k in values)))
    if values[0] <= 0:
        raise ValueError("Every k value must be positive.")
    dimension_limit = min(shape)
    if values[-1] > dimension_limit:
        raise ValueError(
            "Every k value must be at most min(matrix.shape)={} for an "
            "exact dense SVD.".format(dimension_limit)
        )
    return values


def _truncated_svd(
    matrix: sp.spmatrix,
    rank: int,
    initial_vector: np.ndarray,
    tolerance: float,
    max_iterations: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        left, singular_values, right_transposed = svds(
            matrix,
            k=rank,
            which="LM",
            tol=tolerance,
            maxiter=max_iterations,
            v0=initial_vector,
            return_singular_vectors=True,
        )
    except ArpackNoConvergence as error:
        raise RuntimeError(
            "Truncated SVD did not converge at rank {}. Increase --maxiter or "
            "relax --tol.".format(rank)
        ) from error
    order = np.argsort(singular_values)[::-1]
    return (
        left[:, order],
        singular_values[order],
        right_transposed[order, :],
    )


def _frobenius_energy(matrix: sp.spmatrix) -> float:
    data = matrix.data.astype(np.float64, copy=False)
    return float(np.dot(data, data))


def _prepare_sparse_matrix(matrix: sp.spmatrix, name: str) -> sp.csr_matrix:
    if not sp.issparse(matrix):
        raise ValueError("{} must be a SciPy sparse matrix.".format(name))
    matrix = matrix.astype(np.float64).tocsr()
    matrix.sum_duplicates()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("{} must contain only finite values.".format(name))
    if _frobenius_energy(matrix) <= 0.0:
        raise ValueError("{} must have positive Frobenius energy.".format(name))
    return matrix


def _decompose_matrix(
    matrix: sp.csr_matrix,
    rank: int,
    initial_vector: np.ndarray,
    tolerance: float,
    max_iterations: Optional[int],
) -> _SpectralDecomposition:
    left, singular_values, right_transposed = _truncated_svd(
        matrix, rank, initial_vector, tolerance, max_iterations
    )
    return _SpectralDecomposition(
        left=left,
        singular_values=singular_values,
        right_transposed=right_transposed,
        frobenius_energy=_frobenius_energy(matrix),
    )


def _compare_decompositions(
    original: _SpectralDecomposition,
    candidate: _SpectralDecomposition,
    matrix_shape: Tuple[int, int],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    metrics = []
    for k in k_values:
        original_concentration = float(
            np.dot(
                original.singular_values[:k], original.singular_values[:k]
            )
            / original.frobenius_energy
        )
        candidate_concentration = float(
            np.dot(
                candidate.singular_values[:k], candidate.singular_values[:k]
            )
            / candidate.frobenius_energy
        )
        original_concentration = float(
            np.clip(original_concentration, 0.0, 1.0)
        )
        candidate_concentration = float(
            np.clip(candidate_concentration, 0.0, 1.0)
        )

        user_cross = np.matmul(
            original.left[:, :k].T, candidate.left[:, :k]
        )
        item_cross = np.matmul(
            original.right_transposed[:k, :],
            candidate.right_transposed[:k, :].T,
        )
        user_overlap = float(np.square(user_cross).sum() / k)
        item_overlap = float(np.square(item_cross).sum() / k)
        user_overlap = float(np.clip(user_overlap, 0.0, 1.0))
        item_overlap = float(np.clip(item_overlap, 0.0, 1.0))
        complementarity = float(
            np.clip(1.0 - 0.5 * (user_overlap + item_overlap), 0.0, 1.0)
        )

        metrics.append({
            "k": int(k),
            "original_spectral_energy": original_concentration,
            "weighted_spectral_energy": candidate_concentration,
            "spectral_energy_delta": (
                candidate_concentration - original_concentration
            ),
            "user_subspace_overlap": user_overlap,
            "item_subspace_overlap": item_overlap,
            "complementarity": complementarity,
        })

    maximum_k = k_values[-1]
    original_top_energy = float(
        np.dot(
            original.singular_values[:maximum_k],
            original.singular_values[:maximum_k],
        )
    )
    candidate_top_energy = float(
        np.dot(
            candidate.singular_values[:maximum_k],
            candidate.singular_values[:maximum_k],
        )
    )
    spectral_bands = []
    previous_end = 0
    for end_rank in k_values:
        start_index = previous_end
        end_index = int(end_rank)
        original_band_energy = float(
            np.dot(
                original.singular_values[start_index:end_index],
                original.singular_values[start_index:end_index],
            )
        )
        candidate_band_energy = float(
            np.dot(
                candidate.singular_values[start_index:end_index],
                candidate.singular_values[start_index:end_index],
            )
        )
        original_global_contribution = (
            original_band_energy / original.frobenius_energy
        )
        candidate_global_contribution = (
            candidate_band_energy / candidate.frobenius_energy
        )
        original_within_top_share = original_band_energy / original_top_energy
        candidate_within_top_share = candidate_band_energy / candidate_top_energy
        spectral_bands.append({
            "label": "{}-{}".format(start_index + 1, end_index),
            "start_rank": start_index + 1,
            "end_rank": end_index,
            "original_global_energy_contribution": float(
                original_global_contribution
            ),
            "weighted_global_energy_contribution": float(
                candidate_global_contribution
            ),
            "global_energy_contribution_delta": float(
                candidate_global_contribution - original_global_contribution
            ),
            "original_within_top_{}_share".format(maximum_k): float(
                original_within_top_share
            ),
            "weighted_within_top_{}_share".format(maximum_k): float(
                candidate_within_top_share
            ),
            "within_top_{}_share_delta".format(maximum_k): float(
                candidate_within_top_share - original_within_top_share
            ),
        })
        previous_end = end_index

    return {
        "matrix_shape": [int(matrix_shape[0]), int(matrix_shape[1])],
        "original_frobenius_energy": original.frobenius_energy,
        "weighted_frobenius_energy": candidate.frobenius_energy,
        "k_values": list(k_values),
        "singular_values": {
            "original": original.singular_values.tolist(),
            "weighted": candidate.singular_values.tolist(),
        },
        "metrics": metrics,
        "spectral_bands": spectral_bands,
    }


def _analyze_spectral_views(
    original: sp.spmatrix,
    candidates: Mapping[str, sp.spmatrix],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    seed: int = 0,
    tolerance: float = 1e-7,
    max_iterations: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare several candidate views while decomposing the original once."""
    if not candidates:
        raise ValueError("At least one candidate graph view is required.")
    original = _prepare_sparse_matrix(original, "original")
    if len(original.shape) != 2 or min(original.shape) <= 1:
        raise ValueError("The graph matrices must be two-dimensional and non-trivial.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative number.")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be positive when provided.")

    checked_k_values = validate_k_values(k_values, original.shape)
    maximum_k = checked_k_values[-1]
    random_generator = np.random.RandomState(int(seed))
    initial_vector = random_generator.standard_normal(min(original.shape))
    initial_vector /= np.linalg.norm(initial_vector)

    original_decomposition = _decompose_matrix(
        original, maximum_k, initial_vector, tolerance, max_iterations
    )
    results = {}
    for name, candidate in candidates.items():
        candidate = _prepare_sparse_matrix(candidate, name)
        if original.shape != candidate.shape:
            raise ValueError(
                "original and {} matrices must have the same shape.".format(name)
            )
        candidate_decomposition = _decompose_matrix(
            candidate, maximum_k, initial_vector, tolerance, max_iterations
        )
        results[name] = _compare_decompositions(
            original_decomposition,
            candidate_decomposition,
            original.shape,
            checked_k_values,
        )
    return results


def analyze_spectral_complementarity(
    original: sp.spmatrix,
    weighted: sp.spmatrix,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    seed: int = 0,
    tolerance: float = 1e-7,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute top-k spectral energy and user/item subspace complementarity."""
    return _analyze_spectral_views(
        original,
        {"weighted": weighted},
        k_values=k_values,
        seed=seed,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )["weighted"]


def _dense_decomposition(matrix: np.ndarray) -> _SpectralDecomposition:
    """Compute an exact, descending SVD for a dense embedding matrix."""
    left, singular_values, right_transposed = np.linalg.svd(
        matrix, full_matrices=False
    )
    order = np.argsort(singular_values)[::-1]
    return _SpectralDecomposition(
        left=left[:, order],
        singular_values=singular_values[order],
        right_transposed=right_transposed[order, :],
        frobenius_energy=float(np.square(matrix).sum()),
    )


def _distribution_summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Cannot summarize an empty value distribution.")
    return {
        "minimum": float(values.min()),
        "quantile_05": float(np.quantile(values, 0.05)),
        "quantile_25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "quantile_75": float(np.quantile(values, 0.75)),
        "quantile_95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
    }


def analyze_user_representations(
    full_users: np.ndarray,
    masked_users: np.ndarray,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    full_source_key: str = "representations.full_users",
    masked_source_key: str = "representations.masked_users",
) -> Dict[str, Any]:
    """Compare aligned full-branch and masked-branch user representations.

    Spectral metrics use the uncentered ``num_users x embedding_dim`` matrices
    to stay analogous to the graph analysis. Linear CKA and orthogonal
    Procrustes metrics use centered matrices and are insensitive to a shared
    translation; both are also invariant to isotropic scaling and orthogonal
    changes of basis.
    """
    full_users = _as_numpy_matrix(full_users, "full_users")
    masked_users = _as_numpy_matrix(masked_users, "masked_users")
    if full_users.shape != masked_users.shape:
        raise ValueError(
            "full_users and masked_users must have the same shape."
        )
    if min(full_users.shape) <= 1:
        raise ValueError("User representation matrices must be non-trivial.")
    if not np.all(np.isfinite(full_users)):
        raise ValueError("full_users must contain only finite values.")
    if not np.all(np.isfinite(masked_users)):
        raise ValueError("masked_users must contain only finite values.")

    full_energy = float(np.square(full_users).sum())
    masked_energy = float(np.square(masked_users).sum())
    if full_energy <= 0.0 or masked_energy <= 0.0:
        raise ValueError("Both user representation matrices need positive energy.")
    checked_k_values = _validate_dense_k_values(k_values, full_users.shape)

    full_decomposition = _dense_decomposition(full_users)
    masked_decomposition = _dense_decomposition(masked_users)
    spectral = _compare_decompositions(
        full_decomposition,
        masked_decomposition,
        full_users.shape,
        checked_k_values,
    )

    full_norms = np.linalg.norm(full_users, axis=1)
    masked_norms = np.linalg.norm(masked_users, axis=1)
    epsilon = np.finfo(np.float64).eps
    valid_cosine = (full_norms > epsilon) & (masked_norms > epsilon)
    if not np.any(valid_cosine):
        raise ValueError(
            "No user has non-zero representations in both branches."
        )
    paired_cosines = np.einsum(
        "ij,ij->i", full_users[valid_cosine], masked_users[valid_cosine]
    )
    paired_cosines /= full_norms[valid_cosine] * masked_norms[valid_cosine]
    paired_cosines = np.clip(paired_cosines, -1.0, 1.0)

    valid_full_norm = full_norms > epsilon
    norm_ratios = masked_norms[valid_full_norm] / full_norms[valid_full_norm]

    centered_full = full_users - full_users.mean(axis=0, keepdims=True)
    centered_masked = masked_users - masked_users.mean(axis=0, keepdims=True)
    centered_full_energy = float(np.square(centered_full).sum())
    centered_masked_energy = float(np.square(centered_masked).sum())
    if centered_full_energy <= epsilon or centered_masked_energy <= epsilon:
        raise ValueError(
            "Centered user representations must have positive energy."
        )

    cross_covariance = centered_full.T @ centered_masked
    full_covariance = centered_full.T @ centered_full
    masked_covariance = centered_masked.T @ centered_masked
    cka_denominator = float(
        np.linalg.norm(full_covariance, ord="fro")
        * np.linalg.norm(masked_covariance, ord="fro")
    )
    linear_cka = float(
        np.square(cross_covariance).sum() / cka_denominator
    )
    linear_cka = float(np.clip(linear_cka, 0.0, 1.0))

    procrustes_left, procrustes_values, procrustes_right_t = np.linalg.svd(
        cross_covariance, full_matrices=False
    )
    rotation = procrustes_left @ procrustes_right_t
    aligned_full = centered_full @ rotation
    procrustes_similarity = float(
        procrustes_values.sum()
        / np.sqrt(centered_full_energy * centered_masked_energy)
    )
    procrustes_similarity = float(
        np.clip(procrustes_similarity, 0.0, 1.0)
    )
    procrustes_relative_error = float(
        np.linalg.norm(aligned_full - centered_masked)
        / np.sqrt(centered_masked_energy)
    )
    optimal_procrustes_scale = float(
        procrustes_values.sum() / centered_full_energy
    )
    scaled_procrustes_relative_error = float(
        np.linalg.norm(
            optimal_procrustes_scale * aligned_full - centered_masked
        )
        / np.sqrt(centered_masked_energy)
    )

    metrics = []
    for metric in spectral["metrics"]:
        metrics.append({
            "k": metric["k"],
            "full_spectral_energy": metric["original_spectral_energy"],
            "masked_spectral_energy": metric["weighted_spectral_energy"],
            "spectral_energy_delta": metric["spectral_energy_delta"],
            "user_subspace_overlap": metric["user_subspace_overlap"],
            "feature_subspace_overlap": metric["item_subspace_overlap"],
            "complementarity": metric["complementarity"],
        })

    maximum_k = checked_k_values[-1]
    full_share_key = "original_within_top_{}_share".format(maximum_k)
    masked_share_key = "weighted_within_top_{}_share".format(maximum_k)
    bands = []
    for band in spectral["spectral_bands"]:
        bands.append({
            "label": band["label"],
            "start_rank": band["start_rank"],
            "end_rank": band["end_rank"],
            "full_global_energy_contribution": band[
                "original_global_energy_contribution"
            ],
            "masked_global_energy_contribution": band[
                "weighted_global_energy_contribution"
            ],
            "global_energy_contribution_delta": band[
                "global_energy_contribution_delta"
            ],
            "full_within_top_{}_share".format(maximum_k): band[
                full_share_key
            ],
            "masked_within_top_{}_share".format(maximum_k): band[
                masked_share_key
            ],
            "within_top_{}_share_delta".format(maximum_k): band[
                "within_top_{}_share_delta".format(maximum_k)
            ],
        })

    return {
        "source_keys": {
            "full": str(full_source_key),
            "masked": str(masked_source_key),
        },
        "matrix_shape": [int(full_users.shape[0]), int(full_users.shape[1])],
        "spectral_centered": False,
        "k_values": list(checked_k_values),
        "frobenius_energy": {
            "full": full_energy,
            "masked": masked_energy,
        },
        "singular_values": {
            "full": spectral["singular_values"]["original"],
            "masked": spectral["singular_values"]["weighted"],
        },
        "metrics": metrics,
        "spectral_bands": bands,
        "paired_user_similarity": {
            "valid_user_count": int(valid_cosine.sum()),
            "excluded_zero_norm_user_count": int((~valid_cosine).sum()),
            "cosine_similarity": _distribution_summary(paired_cosines),
            "masked_to_full_norm_ratio": _distribution_summary(norm_ratios),
            "raw_relative_frobenius_difference": float(
                np.linalg.norm(masked_users - full_users)
                / np.sqrt(full_energy)
            ),
            "masked_to_full_frobenius_norm_ratio": float(
                np.sqrt(masked_energy / full_energy)
            ),
        },
        "global_similarity": {
            "centered_linear_cka": linear_cka,
            "centered_orthogonal_procrustes_similarity": (
                procrustes_similarity
            ),
            "centered_orthogonal_procrustes_relative_error": (
                procrustes_relative_error
            ),
            "centered_scaled_procrustes_optimal_scale": (
                optimal_procrustes_scale
            ),
            "centered_scaled_procrustes_relative_error": (
                scaled_procrustes_relative_error
            ),
        },
    }


def _random_hard_graphs(
    original: sp.csr_matrix,
    selected_count: int,
    num_runs: int,
    random_seed: int,
) -> Dict[str, sp.csr_matrix]:
    original_coo = original.tocoo()
    random_generator = np.random.RandomState(random_seed)
    candidates = {}
    for run_index in range(num_runs):
        selected = random_generator.choice(
            original_coo.nnz, size=selected_count, replace=False
        )
        candidates["random_{:03d}".format(run_index)] = sp.coo_matrix(
            (
                np.ones(selected_count, dtype=np.float64),
                (original_coo.row[selected], original_coo.col[selected]),
            ),
            shape=original.shape,
        ).tocsr()
    return candidates


def _summarize_random_baseline(
    learned_hard: Mapping[str, Any],
    random_results: Sequence[Mapping[str, Any]],
    random_seed: int,
    selected_count: int,
    total_edges: int,
) -> Dict[str, Any]:
    learned_by_k = {metric["k"]: metric for metric in learned_hard["metrics"]}
    random_by_k = {
        k: [
            next(metric for metric in result["metrics"] if metric["k"] == k)
            for result in random_results
        ]
        for k in learned_by_k
    }
    summaries = []
    for k, learned_metric in learned_by_k.items():
        complementarity_values = np.array(
            [metric["complementarity"] for metric in random_by_k[k]],
            dtype=np.float64,
        )
        energy_delta_values = np.array(
            [metric["spectral_energy_delta"] for metric in random_by_k[k]],
            dtype=np.float64,
        )
        random_mean = float(complementarity_values.mean())
        random_std = float(
            complementarity_values.std(ddof=1)
            if complementarity_values.size > 1
            else 0.0
        )
        learned_value = float(learned_metric["complementarity"])
        empirical_p = float(
            (1 + np.count_nonzero(complementarity_values >= learned_value))
            / (complementarity_values.size + 1)
        )
        summaries.append({
            "k": int(k),
            "learned_hard_complementarity": learned_value,
            "random_complementarity_mean": random_mean,
            "random_complementarity_std": random_std,
            "random_complementarity_interval_95": [
                float(np.quantile(complementarity_values, 0.025)),
                float(np.quantile(complementarity_values, 0.975)),
            ],
            "excess_complementarity": learned_value - random_mean,
            "one_sided_empirical_p": empirical_p,
            "learned_hard_spectral_energy_delta": float(
                learned_metric["spectral_energy_delta"]
            ),
            "random_spectral_energy_delta_mean": float(
                energy_delta_values.mean()
            ),
            "random_spectral_energy_delta_std": float(
                energy_delta_values.std(ddof=1)
                if energy_delta_values.size > 1
                else 0.0
            ),
        })
    learned_bands = {
        band["label"]: band for band in learned_hard["spectral_bands"]
    }
    random_band_summaries = []
    for label, learned_band in learned_bands.items():
        random_values = np.array(
            [
                next(
                    band
                    for band in result["spectral_bands"]
                    if band["label"] == label
                )["weighted_global_energy_contribution"]
                for result in random_results
            ],
            dtype=np.float64,
        )
        learned_value = float(
            learned_band["weighted_global_energy_contribution"]
        )
        random_mean = float(random_values.mean())
        random_band_summaries.append({
            "label": label,
            "start_rank": learned_band["start_rank"],
            "end_rank": learned_band["end_rank"],
            "original_global_energy_contribution": learned_band[
                "original_global_energy_contribution"
            ],
            "learned_hard_global_energy_contribution": learned_value,
            "random_global_energy_contribution_mean": random_mean,
            "random_global_energy_contribution_std": float(
                random_values.std(ddof=1) if random_values.size > 1 else 0.0
            ),
            "learned_minus_random_global_contribution": (
                learned_value - random_mean
            ),
        })
    return {
        "sampling": "uniform_without_replacement",
        "num_runs": len(random_results),
        "random_seed": int(random_seed),
        "num_selected_edges_per_run": int(selected_count),
        "selection_ratio": selected_count / total_edges,
        "metrics": summaries,
        "spectral_bands": random_band_summaries,
    }


def _spectral_node_embeddings(
    original: sp.csr_matrix,
    candidate: sp.csr_matrix,
    rank: int,
    seed: int,
    tolerance: float,
    max_iterations: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return U sqrt(S) and V sqrt(S) coordinates for two graph views."""
    checked_rank = validate_k_values((rank,), original.shape)[0]
    random_generator = np.random.RandomState(int(seed))
    initial_vector = random_generator.standard_normal(min(original.shape))
    initial_vector /= np.linalg.norm(initial_vector)
    original_left, original_values, original_right_t = _truncated_svd(
        original,
        checked_rank,
        initial_vector,
        tolerance,
        max_iterations,
    )
    candidate_left, candidate_values, candidate_right_t = _truncated_svd(
        candidate,
        checked_rank,
        initial_vector,
        tolerance,
        max_iterations,
    )
    original_scale = np.sqrt(np.maximum(original_values, 0.0))
    candidate_scale = np.sqrt(np.maximum(candidate_values, 0.0))
    return (
        original_left * original_scale,
        candidate_left * candidate_scale,
        original_right_t.T * original_scale,
        candidate_right_t.T * candidate_scale,
    )


def _normalize_embedding_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms > 1e-12,
    )


def _cluster_active_nodes(
    embeddings: np.ndarray,
    active: np.ndarray,
    num_clusters: int,
    seed: int,
) -> np.ndarray:
    active_count = int(active.sum())
    if active_count < num_clusters:
        raise ValueError(
            "Cannot create {} clusters from only {} active nodes.".format(
                num_clusters, active_count
            )
        )
    normalized = _normalize_embedding_rows(embeddings[active])
    _, active_labels = kmeans2(
        normalized,
        num_clusters,
        iter=100,
        thresh=1e-6,
        minit="++",
        missing="raise",
        check_finite=True,
        seed=np.random.RandomState(int(seed)),
    )
    labels = np.full(embeddings.shape[0], -1, dtype=np.int64)
    labels[active] = active_labels
    return labels


def _aligned_transition_matrix(
    original_labels: np.ndarray,
    candidate_labels: np.ndarray,
    original_active: np.ndarray,
    candidate_active: np.ndarray,
    num_clusters: int,
) -> Dict[str, Any]:
    active_in_both = original_active & candidate_active
    matching_counts = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    np.add.at(
        matching_counts,
        (
            original_labels[active_in_both],
            candidate_labels[active_in_both],
        ),
        1,
    )
    matched_original, matched_candidate = linear_sum_assignment(
        -matching_counts
    )
    candidate_to_aligned = np.empty(num_clusters, dtype=np.int64)
    candidate_to_aligned[matched_candidate] = matched_original

    aligned_candidate_labels = np.full_like(candidate_labels, -1)
    aligned_candidate_labels[candidate_active] = candidate_to_aligned[
        candidate_labels[candidate_active]
    ]
    counts = np.zeros((num_clusters, num_clusters + 1), dtype=np.int64)
    np.add.at(
        counts,
        (
            original_labels[active_in_both],
            aligned_candidate_labels[active_in_both],
        ),
        1,
    )
    isolated_after_mask = original_active & ~candidate_active
    np.add.at(
        counts,
        (original_labels[isolated_after_mask], np.full(isolated_after_mask.sum(), num_clusters)),
        1,
    )
    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals > 0,
    )
    total_active = int(original_active.sum())
    retained_count = int(np.trace(counts[:, :num_clusters]))
    isolated_count = int(isolated_after_mask.sum())
    return {
        "counts": counts.tolist(),
        "row_percentages": percentages.tolist(),
        "original_active_nodes": total_active,
        "candidate_active_nodes": int(candidate_active.sum()),
        "isolated_after_mask": isolated_count,
        "isolated_ratio": isolated_count / total_active,
        "aligned_cluster_retention_ratio": retained_count / total_active,
        "aligned_cluster_transition_ratio": 1.0 - retained_count / total_active,
        "candidate_cluster_alignment": candidate_to_aligned.tolist(),
    }


def generate_cluster_transition_heatmap(
    analysis_file: Path,
    output_file: Path,
    branch: str = "masked_branch",
    rank: int = 64,
    num_clusters: int = 8,
    seed: int = 0,
    tolerance: float = 1e-7,
    max_iterations: Optional[int] = None,
) -> Dict[str, Any]:
    """Cluster spectral node embeddings and plot original-to-hard transitions."""
    if isinstance(num_clusters, bool) or not isinstance(
        num_clusters, (int, np.integer)
    ):
        raise ValueError("num_clusters must be an integer.")
    if num_clusters < 2:
        raise ValueError("num_clusters must be at least 2.")
    graph_views = load_graph_views(analysis_file, branch)
    if graph_views.hard_masked is None:
        raise ValueError(
            "Mask branch {!r} does not contain selected_at_keep_ratio.".format(
                branch
            )
        )

    original = graph_views.original
    hard_masked = graph_views.hard_masked
    original_users, hard_users, original_items, hard_items = (
        _spectral_node_embeddings(
            original,
            hard_masked,
            rank=rank,
            seed=seed,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    )
    original_user_active = np.asarray(original.getnnz(axis=1) > 0).ravel()
    hard_user_active = np.asarray(hard_masked.getnnz(axis=1) > 0).ravel()
    original_item_active = np.asarray(original.getnnz(axis=0) > 0).ravel()
    hard_item_active = np.asarray(hard_masked.getnnz(axis=0) > 0).ravel()

    original_user_labels = _cluster_active_nodes(
        original_users, original_user_active, num_clusters, seed
    )
    hard_user_labels = _cluster_active_nodes(
        hard_users, hard_user_active, num_clusters, seed + 1
    )
    original_item_labels = _cluster_active_nodes(
        original_items, original_item_active, num_clusters, seed + 2
    )
    hard_item_labels = _cluster_active_nodes(
        hard_items, hard_item_active, num_clusters, seed + 3
    )

    user_transition = _aligned_transition_matrix(
        original_user_labels,
        hard_user_labels,
        original_user_active,
        hard_user_active,
        num_clusters,
    )
    item_transition = _aligned_transition_matrix(
        original_item_labels,
        hard_item_labels,
        original_item_active,
        hard_item_active,
        num_clusters,
    )

    try:
        matplotlib_config = Path(tempfile.gettempdir()) / "pgl_matplotlib_cache"
        matplotlib_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required to generate the cluster-transition heatmap."
        ) from error

    figure, axes = plt.subplots(
        1, 2, figsize=(18, 7), constrained_layout=True
    )
    column_labels = [
        "Hard C{}".format(index + 1) for index in range(num_clusters)
    ] + ["Isolated"]
    row_labels = [
        "Original C{}".format(index + 1) for index in range(num_clusters)
    ]
    image = None
    for axis, title, transition in (
        (axes[0], "User cluster transitions", user_transition),
        (axes[1], "Item cluster transitions", item_transition),
    ):
        percentages = np.asarray(transition["row_percentages"]) * 100.0
        image = axis.imshow(
            percentages, cmap="Blues", vmin=0.0, vmax=100.0, aspect="auto"
        )
        axis.set_xticks(np.arange(num_clusters + 1), labels=column_labels)
        axis.set_yticks(np.arange(num_clusters), labels=row_labels)
        axis.tick_params(axis="x", labelrotation=45)
        axis.set_xlabel("Learned hard-mask clusters")
        axis.set_ylabel("Original-graph clusters")
        axis.set_title(
            "{}\n{} isolated after mask ({:.1%})".format(
                title,
                transition["isolated_after_mask"],
                transition["isolated_ratio"],
            )
        )
        annotation_threshold = max(10.0, float(percentages.max()) * 0.45)
        for row_index in range(num_clusters):
            for column_index in range(num_clusters + 1):
                value = percentages[row_index, column_index]
                if value < 0.5:
                    continue
                axis.text(
                    column_index,
                    row_index,
                    "{:.1f}".format(value),
                    ha="center",
                    va="center",
                    color="white" if value >= annotation_threshold else "black",
                    fontsize=8,
                )
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Percentage of each original cluster",
        shrink=0.9,
    )
    figure.suptitle(
        "Spectral cluster transitions: original graph to learned hard mask "
        "(rank={}, clusters={})".format(rank, num_clusters),
        fontsize=14,
    )
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_file), dpi=220, bbox_inches="tight")
    plt.close(figure)

    return {
        "output_file": str(output_file),
        "rank": int(rank),
        "num_clusters": int(num_clusters),
        "cluster_embedding": "row_l2_normalized_U_or_V_times_sqrt_sigma",
        "cluster_label_alignment": "hungarian_maximum_overlap",
        "user": user_transition,
        "item": item_transition,
    }


def analyze_artifact(
    analysis_file: Path,
    branch: str = "masked_branch",
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    seed: int = 0,
    tolerance: float = 1e-7,
    max_iterations: Optional[int] = None,
    include_hard: bool = False,
    random_baseline_runs: int = 0,
    random_baseline_seed: int = 0,
    include_user_embeddings: bool = False,
) -> Dict[str, Any]:
    """Load an artifact and produce a JSON-serializable analysis result."""
    graph_views = load_graph_views(analysis_file, branch)
    if (
        isinstance(random_baseline_runs, bool)
        or not isinstance(random_baseline_runs, (int, np.integer))
        or random_baseline_runs < 0
    ):
        raise ValueError("random_baseline_runs must be a non-negative integer.")
    if isinstance(random_baseline_seed, bool) or not isinstance(
        random_baseline_seed, (int, np.integer)
    ):
        raise ValueError("random_baseline_seed must be an integer.")
    include_hard = include_hard or random_baseline_runs > 0
    if include_hard and (
        graph_views.hard_masked is None or graph_views.hard_selection is None
    ):
        raise ValueError(
            "Mask branch {!r} does not contain selected_at_keep_ratio.".format(
                branch
            )
        )

    candidates = {"probability": graph_views.weighted}
    if include_hard:
        candidates["hard"] = graph_views.hard_masked
    selected_count = (
        int(graph_views.hard_selection.sum()) if include_hard else 0
    )
    random_names = []
    if random_baseline_runs > 0:
        random_candidates = _random_hard_graphs(
            graph_views.original,
            selected_count=selected_count,
            num_runs=int(random_baseline_runs),
            random_seed=int(random_baseline_seed),
        )
        candidates.update(random_candidates)
        random_names = list(random_candidates)

    analyses = _analyze_spectral_views(
        graph_views.original,
        candidates,
        k_values=k_values,
        seed=seed,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    result = analyses["probability"]
    probabilities = graph_views.probabilities
    result.update({
        "analysis_file": str(Path(analysis_file).resolve()),
        "branch": branch,
        "weighted_graph_definition": "binary_adjacency * sigmoid(mask_logit)",
        "degree_normalized": False,
        "metadata": graph_views.metadata,
        "num_edges": int(probabilities.size),
        "probability_statistics": {
            "minimum": float(probabilities.min()),
            "maximum": float(probabilities.max()),
            "mean": float(probabilities.mean()),
            "standard_deviation": float(probabilities.std()),
        },
        "svd_parameters": {
            "seed": int(seed),
            "tolerance": float(tolerance),
            "max_iterations": max_iterations,
        },
    })
    if include_hard:
        hard_result = analyses["hard"]
        hard_result.update({
            "weighted_graph_definition": (
                "binary_adjacency * selected_at_keep_ratio"
            ),
            "num_selected_edges": selected_count,
            "selection_ratio": selected_count / int(probabilities.size),
        })
        result["hard_masked_analysis"] = hard_result
        if random_baseline_runs > 0:
            result["random_hard_baseline"] = _summarize_random_baseline(
                hard_result,
                [analyses[name] for name in random_names],
                random_seed=int(random_baseline_seed),
                selected_count=selected_count,
                total_edges=int(probabilities.size),
            )
    if include_user_embeddings:
        embedding_artifact = _load_analysis_artifact(analysis_file)
        full_users, masked_users = extract_user_representations(
            embedding_artifact
        )
        pre_propagation_pairs = extract_pre_propagation_user_representations(
            embedding_artifact
        )
        del embedding_artifact
        result["user_embedding_analysis"] = analyze_user_representations(
            full_users,
            masked_users,
            k_values=k_values,
        )
        result["pre_propagation_user_embedding_analysis"] = {
            "stage": "learned_embedding_tables_before_graph_propagation",
            "text": analyze_user_representations(
                pre_propagation_pairs["text"][0],
                pre_propagation_pairs["text"][1],
                k_values=k_values,
                full_source_key="embedding_tables.user_text.weight",
                masked_source_key=(
                    "embedding_tables.second_user_text.weight"
                ),
            ),
            "image": analyze_user_representations(
                pre_propagation_pairs["image"][0],
                pre_propagation_pairs["image"][1],
                k_values=k_values,
                full_source_key="embedding_tables.user_image.weight",
                masked_source_key=(
                    "embedding_tables.second_user_image.weight"
                ),
            ),
        }
    return result


def _print_metric_table(metrics: Sequence[Mapping[str, Any]]) -> None:
    print(
        "{:>4} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
            "k", "C_original", "C_weighted", "delta_C", "overlap_U",
            "overlap_I", "comp_k"
        )
    )
    for metric in metrics:
        print(
            "{k:4d} {original_spectral_energy:12.6f} "
            "{weighted_spectral_energy:12.6f} {spectral_energy_delta:12.6f} "
            "{user_subspace_overlap:12.6f} {item_subspace_overlap:12.6f} "
            "{complementarity:12.6f}".format(**metric)
        )


def _print_spectral_band_table(analysis: Mapping[str, Any]) -> None:
    maximum_k = analysis["k_values"][-1]
    original_share_key = "original_within_top_{}_share".format(maximum_k)
    weighted_share_key = "weighted_within_top_{}_share".format(maximum_k)
    print("Spectral-band contribution (non-overlapping ranks)")
    print(
        "{:>9} {:>13} {:>13} {:>13} {:>15} {:>15}".format(
            "band",
            "global_orig",
            "global_view",
            "global_delta",
            "within_orig",
            "within_view",
        )
    )
    for band in analysis["spectral_bands"]:
        print(
            "{label:>9} {original_global_energy_contribution:13.6f} "
            "{weighted_global_energy_contribution:13.6f} "
            "{global_energy_contribution_delta:13.6f} "
            "{original_within_share:15.6f} "
            "{weighted_within_share:15.6f}".format(
                original_within_share=band[original_share_key],
                weighted_within_share=band[weighted_share_key],
                **band
            )
        )


def _print_user_embedding_analysis(analysis: Mapping[str, Any]) -> None:
    rows, columns = analysis["matrix_shape"]
    source_keys = analysis["source_keys"]
    paired = analysis["paired_user_similarity"]
    cosine = paired["cosine_similarity"]
    norm_ratio = paired["masked_to_full_norm_ratio"]
    global_similarity = analysis["global_similarity"]
    print(
        "User representations: {} vs {}".format(
            source_keys["full"], source_keys["masked"]
        )
    )
    print("Matrix: {} users x {} embedding dimensions".format(rows, columns))
    print(
        "Paired cosine: mean={:.6f}, median={:.6f}, p05={:.6f}, "
        "p95={:.6f}".format(
            cosine["mean"],
            cosine["median"],
            cosine["quantile_05"],
            cosine["quantile_95"],
        )
    )
    print(
        "Masked/full user-norm ratio: mean={:.6f}, median={:.6f}; "
        "global norm ratio={:.6f}".format(
            norm_ratio["mean"],
            norm_ratio["median"],
            paired["masked_to_full_frobenius_norm_ratio"],
        )
    )
    print(
        "Centered linear CKA={:.6f}; Procrustes similarity={:.6f}; "
        "scale-aligned error={:.6f} (optimal scale={:.6f})".format(
            global_similarity["centered_linear_cka"],
            global_similarity[
                "centered_orthogonal_procrustes_similarity"
            ],
            global_similarity[
                "centered_scaled_procrustes_relative_error"
            ],
            global_similarity[
                "centered_scaled_procrustes_optimal_scale"
            ],
        )
    )
    print(
        "{:>4} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
            "k",
            "C_full",
            "C_masked",
            "delta_C",
            "overlap_U",
            "overlap_feat",
            "comp_k",
        )
    )
    for metric in analysis["metrics"]:
        print(
            "{k:4d} {full_spectral_energy:12.6f} "
            "{masked_spectral_energy:12.6f} {spectral_energy_delta:12.6f} "
            "{user_subspace_overlap:12.6f} "
            "{feature_subspace_overlap:12.6f} "
            "{complementarity:12.6f}".format(**metric)
        )

    maximum_k = analysis["k_values"][-1]
    full_share_key = "full_within_top_{}_share".format(maximum_k)
    masked_share_key = "masked_within_top_{}_share".format(maximum_k)
    print("Embedding spectral-band contribution")
    print(
        "{:>9} {:>13} {:>13} {:>13} {:>15} {:>15}".format(
            "band",
            "global_full",
            "global_mask",
            "global_delta",
            "within_full",
            "within_mask",
        )
    )
    for band in analysis["spectral_bands"]:
        print(
            "{label:>9} {full_global_energy_contribution:13.6f} "
            "{masked_global_energy_contribution:13.6f} "
            "{global_energy_contribution_delta:13.6f} "
            "{full_within_share:15.6f} "
            "{masked_within_share:15.6f}".format(
                full_within_share=band[full_share_key],
                masked_within_share=band[masked_share_key],
                **band
            )
        )


def print_report(result: Mapping[str, Any]) -> None:
    """Print a compact human-readable report."""
    rows, columns = result["matrix_shape"]
    stats = result["probability_statistics"]
    print("Spectral complementarity of learned U-I graph views")
    print("Analysis file: {}".format(result["analysis_file"]))
    print("Branch: {}".format(result["branch"]))
    print("Matrix: {} users x {} items; {} edges".format(
        rows, columns, result["num_edges"]
    ))
    print(
        "Mask probability: min={:.6f}, mean={:.6f}, std={:.6f}, max={:.6f}".format(
            stats["minimum"],
            stats["mean"],
            stats["standard_deviation"],
            stats["maximum"],
        )
    )
    print()
    print("Probability-weighted view: R * sigmoid(mask_logit)")
    _print_metric_table(result["metrics"])
    print()
    _print_spectral_band_table(result)

    hard_result = result.get("hard_masked_analysis")
    if hard_result is not None:
        print()
        print(
            "Hard-masked view: {} selected edges ({:.2%})".format(
                hard_result["num_selected_edges"],
                hard_result["selection_ratio"],
            )
        )
        _print_metric_table(hard_result["metrics"])
        print()
        _print_spectral_band_table(hard_result)

    random_baseline = result.get("random_hard_baseline")
    if random_baseline is not None:
        print()
        print(
            "Uniform random hard-mask baseline: {} runs, {} edges/run ({:.2%})".format(
                random_baseline["num_runs"],
                random_baseline["num_selected_edges_per_run"],
                random_baseline["selection_ratio"],
            )
        )
        print(
            "{:>4} {:>13} {:>13} {:>13} {:>13} {:>12}".format(
                "k",
                "comp_learned",
                "comp_random",
                "random_std",
                "learned-rand",
                "p_one_sided",
            )
        )
        for metric in random_baseline["metrics"]:
            print(
                "{k:4d} {learned_hard_complementarity:13.6f} "
                "{random_complementarity_mean:13.6f} "
                "{random_complementarity_std:13.6f} "
                "{excess_complementarity:13.6f} "
                "{one_sided_empirical_p:12.6f}".format(**metric)
            )
        print()
        print("Random baseline: global spectral-band contribution")
        print(
            "{:>9} {:>13} {:>13} {:>13} {:>13}".format(
                "band", "original", "learned", "random_mean", "learned-rand"
            )
        )
        for band in random_baseline["spectral_bands"]:
            print(
                "{label:>9} {original_global_energy_contribution:13.6f} "
                "{learned_hard_global_energy_contribution:13.6f} "
                "{random_global_energy_contribution_mean:13.6f} "
                "{learned_minus_random_global_contribution:13.6f}".format(
                    **band
                )
            )
    user_embedding_analysis = result.get("user_embedding_analysis")
    if user_embedding_analysis is not None:
        print()
        print("Post-propagation user representations")
        _print_user_embedding_analysis(user_embedding_analysis)
    pre_propagation_analysis = result.get(
        "pre_propagation_user_embedding_analysis"
    )
    if pre_propagation_analysis is not None:
        for modality in ("text", "image"):
            print()
            print(
                "Pre-propagation {} user embedding tables".format(modality)
            )
            _print_user_embedding_analysis(
                pre_propagation_analysis[modality]
            )
    print()
    print("delta_C < 0: the weighted spectrum is less concentrated.")
    print("Larger comp_k: less overlap between dominant user/item subspaces.")
    print("These are structural metrics, not evidence of recommendation utility.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a binary user-item graph with the same graph weighted by "
            "sigmoid(mask logits)."
        )
    )
    parser.add_argument(
        "--analysis-file",
        type=Path,
        required=True,
        help="Path to a *-analysis.pt artifact.",
    )
    parser.add_argument(
        "--branch",
        default="masked_branch",
        help="Mask branch inside the artifact (default: masked_branch).",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        metavar="K",
        help="Spectral ranks to report (default: 8 16 32 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to initialize the iterative SVD solver (default: 0).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="ARPACK convergence tolerance (default: 1e-7).",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Maximum ARPACK iterations (default: SciPy's internal value).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the full JSON result.",
    )
    parser.add_argument(
        "--include-hard",
        action="store_true",
        help=(
            "Also compare the original graph with the binary graph selected "
            "by selected_at_keep_ratio."
        ),
    )
    parser.add_argument(
        "--random-baseline-runs",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Run N uniform random hard masks with the same edge count as the "
            "learned hard mask (default: 0). This also enables --include-hard."
        ),
    )
    parser.add_argument(
        "--random-baseline-seed",
        type=int,
        default=0,
        help="Seed used to sample random hard masks (default: 0).",
    )
    parser.add_argument(
        "--cluster-transition-heatmap",
        type=Path,
        default=None,
        metavar="PNG",
        help=(
            "Optional output path for user/item cluster-transition heatmaps "
            "between the original and learned hard graph."
        ),
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
        help="Number of spectral user/item clusters in the heatmap (default: 8).",
    )
    parser.add_argument(
        "--analyze-user-embeddings",
        action="store_true",
        help=(
            "Compare representations.full_users with masked_users and also "
            "compare the pre-propagation user_text/second_user_text and "
            "user_image/second_user_image tables using paired cosine, "
            "centered CKA, Procrustes, and exact dense spectral metrics."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        result = analyze_artifact(
            arguments.analysis_file,
            branch=arguments.branch,
            k_values=arguments.k_values,
            seed=arguments.seed,
            tolerance=arguments.tol,
            max_iterations=arguments.maxiter,
            include_hard=(
                arguments.include_hard
                or arguments.cluster_transition_heatmap is not None
            ),
            random_baseline_runs=arguments.random_baseline_runs,
            random_baseline_seed=arguments.random_baseline_seed,
            include_user_embeddings=arguments.analyze_user_embeddings,
        )
        cluster_transition = None
        if arguments.cluster_transition_heatmap is not None:
            cluster_transition = generate_cluster_transition_heatmap(
                arguments.analysis_file,
                arguments.cluster_transition_heatmap,
                branch=arguments.branch,
                rank=max(arguments.k_values),
                num_clusters=arguments.num_clusters,
                seed=arguments.seed,
                tolerance=arguments.tol,
                max_iterations=arguments.maxiter,
            )
            result["cluster_transition"] = cluster_transition
        print_report(result)
        if cluster_transition is not None:
            print(
                "Saved cluster-transition heatmap to {}".format(
                    cluster_transition["output_file"]
                )
            )
            print(
                "User: transition={:.2%}, isolated={:.2%}; "
                "Item: transition={:.2%}, isolated={:.2%}".format(
                    cluster_transition["user"][
                        "aligned_cluster_transition_ratio"
                    ],
                    cluster_transition["user"]["isolated_ratio"],
                    cluster_transition["item"][
                        "aligned_cluster_transition_ratio"
                    ],
                    cluster_transition["item"]["isolated_ratio"],
                )
            )
        if arguments.output_json is not None:
            output_path = arguments.output_json.resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(result, output_file, indent=2, ensure_ascii=False, allow_nan=False)
                output_file.write("\n")
            print("Saved JSON result to {}".format(output_path))
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
