"""Measure spectral concentration and complementarity of learned U-I weights.

The original view is the binary user-item interaction matrix. The weighted
view has the same support and uses ``sigmoid(mask_logit)`` as each edge value.
This script intentionally analyzes the rectangular user-item matrices rather
than the symmetric propagation adjacency, and it does not apply degree
normalization.

Example:
    python src/mask_analysis/spectral_complementarity.py \
        --analysis-file src/saved/PGL_MASKED-baby-...-analysis.pt \
        --output-json src/saved/PGL_MASKED-baby-spectral.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
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


def load_graph_views(
    analysis_file: Path, branch: str = "masked_branch"
) -> GraphViews:
    """Load a PyTorch analysis artifact and return its graph views."""
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
    return extract_graph_views(artifact, branch)


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
            include_hard=arguments.include_hard,
            random_baseline_runs=arguments.random_baseline_runs,
            random_baseline_seed=arguments.random_baseline_seed,
        )
        print_report(result)
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
