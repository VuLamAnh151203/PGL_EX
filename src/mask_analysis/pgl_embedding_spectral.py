"""Analyze the spectrum of one PGL user-representation matrix.

This script is intended for the original PGL model, which has only one
evaluation view.  It reconstructs the post-propagation user representations
from a PGL checkpoint and the training interaction graph, computes an exact
SVD, and reports:

* cumulative top-k spectral energy C(k); and
* spectral-band contribution for ranks 1-k1, k1+1-k2, ...

The analysis is deliberately *single-view*.  It measures spectral
concentration, not spectral complementarity between two model branches.

Run from ``PGL/src`` (paths may also be absolute):

    python mask_analysis/pgl_embedding_spectral.py \
        --checkpoint saved/PGL-clothing-seed999-....pth \
        --k-values 8 16 32 64 \
        --output-json saved/PGL-clothing-user-spectral.json

For an old checkpoint without saved configuration, provide the dataset and,
if necessary, the interaction file explicitly:

    python mask_analysis/pgl_embedding_spectral.py \
        --checkpoint saved/old-PGL-clothing.pth \
        --dataset clothing \
        --interaction-file ../data/clothing/clothing.inter \
        --n-ui-layers 2

An already exported ``num_users x user_dim`` tensor is also supported:

    python mask_analysis/pgl_embedding_spectral.py \
        --embedding-file saved/pgl_u_g_embeddings.pt \
        --k-values 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import yaml


DEFAULT_K_VALUES = (8, 16, 32, 64)
SRC_DIRECTORY = Path(__file__).resolve().parents[1]


def _torch_load(path: Path) -> Any:
    """Load tensor-only project artifacts on old and new PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as input_file:
        value = yaml.safe_load(input_file) or {}
    if not isinstance(value, dict):
        raise ValueError("Configuration file must contain a mapping: {}".format(path))
    return value


def _checkpoint_parts(payload: Any) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Return a state-like mapping and serialized config from a PGL artifact."""
    if not isinstance(payload, Mapping):
        raise ValueError("PGL checkpoint must contain a mapping, not a bare tensor.")

    config = payload.get("config", {})
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint['config'] must be a mapping when present.")

    if isinstance(payload.get("model_state_dict"), Mapping):
        state = payload["model_state_dict"]
    elif isinstance(payload.get("learnable_parameters"), Mapping):
        # PGL analysis artifacts made by Trainer use this fallback because the
        # original PGL class has no get_analysis_artifacts method.
        state = payload["learnable_parameters"]
    else:
        state = payload

    if not isinstance(state, Mapping):
        raise ValueError("Could not find model_state_dict or learnable_parameters.")
    return state, dict(config)


def _strip_common_prefix(
    state: Mapping[str, Any], prefix: str
) -> Dict[str, Any]:
    keys = [key for key in state if isinstance(key, str)]
    if keys and all(key.startswith(prefix) for key in keys):
        return {key[len(prefix):]: value for key, value in state.items()}
    return dict(state)


def _normalize_state_keys(state: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(state)
    for prefix in ("module.", "_orig_mod."):
        normalized = _strip_common_prefix(normalized, prefix)
    return normalized


def _load_merged_config(
    checkpoint_config: Mapping[str, Any], dataset_override: Optional[str]
) -> Dict[str, Any]:
    """Merge repository defaults with checkpoint values (checkpoint wins)."""
    dataset_name = dataset_override or checkpoint_config.get("dataset")
    config: Dict[str, Any] = {}
    config.update(_load_yaml(SRC_DIRECTORY / "configs" / "overall.yaml"))
    if dataset_name:
        config.update(
            _load_yaml(
                SRC_DIRECTORY / "configs" / "dataset" / "{}.yaml".format(
                    dataset_name
                )
            )
        )
    config.update(_load_yaml(SRC_DIRECTORY / "configs" / "model" / "PGL.yaml"))
    config.update(dict(checkpoint_config))
    if dataset_override:
        config["dataset"] = dataset_override
    return config


def _state_tensor(state: Mapping[str, Any], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError("Checkpoint is missing tensor {!r}.".format(key))
    return value.detach()


def _resolve_interaction_file(
    config: Mapping[str, Any], explicit_file: Optional[Path]
) -> Path:
    if explicit_file is not None:
        path = explicit_file.expanduser().resolve()
    else:
        required = ("data_path", "dataset", "inter_file_name")
        missing = [name for name in required if not config.get(name)]
        if missing:
            raise ValueError(
                "Cannot locate interactions; missing config keys {}. Supply "
                "--dataset and/or --interaction-file.".format(", ".join(missing))
            )
        data_root = Path(str(config["data_path"])).expanduser()
        if not data_root.is_absolute():
            # PGL training is normally launched from PGL/src, so saved relative
            # data_path values are interpreted against that directory.
            data_root = SRC_DIRECTORY / data_root
        path = data_root / str(config["dataset"]) / str(config["inter_file_name"])
        path = path.resolve()
    if not path.is_file():
        raise ValueError("Interaction file does not exist: {}".format(path))
    return path


def load_training_edges(
    interaction_file: Path,
    config: Mapping[str, Any],
    num_users: int,
    num_items: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and de-duplicate the training user-item pairs used by PGL."""
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "pandas is required only when reconstructing embeddings from a "
            "checkpoint. Install the normal PGL dependencies first."
        ) from error

    user_field = config.get("USER_ID_FIELD")
    item_field = config.get("ITEM_ID_FIELD")
    split_field = config.get("inter_splitting_label")
    separator = config.get("field_separator", "\t")
    if not user_field or not item_field:
        raise ValueError(
            "USER_ID_FIELD and ITEM_ID_FIELD are required in checkpoint/config."
        )

    requested_columns = [str(user_field), str(item_field)]
    if split_field:
        requested_columns.append(str(split_field))
    frame = pd.read_csv(
        interaction_file,
        sep=str(separator),
        usecols=requested_columns,
    )
    if split_field:
        frame = frame[frame[str(split_field)] == 0]
    frame = frame[[str(user_field), str(item_field)]].drop_duplicates()
    if frame.empty:
        raise ValueError("No training interactions (split label 0) were found.")

    users = frame[str(user_field)].to_numpy(dtype=np.int64, copy=True)
    items = frame[str(item_field)].to_numpy(dtype=np.int64, copy=True)
    if users.min() < 0 or users.max() >= num_users:
        raise ValueError(
            "Training user ids must be in [0, {}), observed [{}, {}].".format(
                num_users, int(users.min()), int(users.max())
            )
        )
    if items.min() < 0 or items.max() >= num_items:
        raise ValueError(
            "Training item ids must be in [0, {}), observed [{}, {}].".format(
                num_items, int(items.min()), int(items.max())
            )
        )
    return users, items


def build_normalized_ui_adjacency(
    users: np.ndarray,
    items: np.ndarray,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    """Reproduce PGL.get_norm_adj_mat for a binary symmetric U-I graph."""
    rows = np.concatenate((users, items + num_users))
    columns = np.concatenate((items + num_users, users))
    adjacency = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float64), (rows, columns)),
        shape=(num_users + num_items, num_users + num_items),
    ).tocsr()
    adjacency.data.fill(1.0)
    degrees = np.asarray((adjacency > 0).sum(axis=1)).reshape(-1) + 1e-7
    inverse_sqrt = np.power(degrees, -0.5)
    normalized = sp.diags(inverse_sqrt) @ adjacency @ sp.diags(inverse_sqrt)
    normalized = normalized.tocoo()
    indices = torch.from_numpy(
        np.vstack((normalized.row, normalized.col)).astype(np.int64)
    )
    values = torch.from_numpy(normalized.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=normalized.shape,
        device=device,
    ).coalesce()


def _linear_from_state(
    inputs: torch.Tensor,
    state: Mapping[str, Any],
    prefix: str,
    device: torch.device,
) -> torch.Tensor:
    weight = _state_tensor(state, prefix + ".weight").to(device)
    bias_value = state.get(prefix + ".bias")
    if bias_value is not None and not isinstance(bias_value, torch.Tensor):
        raise ValueError("{} must be a tensor or None.".format(prefix + ".bias"))
    bias = None if bias_value is None else bias_value.detach().to(device)
    return F.linear(inputs, weight, bias)


def reconstruct_post_propagation_users(
    state: Mapping[str, Any],
    users: np.ndarray,
    items: np.ndarray,
    n_ui_layers: int,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct PGL.forward(norm_adj)[0] without loading raw feature files."""
    state = _normalize_state_keys(state)
    user_image = _state_tensor(state, "user_image.weight")
    user_text = _state_tensor(state, "user_text.weight")
    image_features = _state_tensor(state, "image_embedding.weight")
    text_features = _state_tensor(state, "text_embedding.weight")

    num_users = int(user_image.shape[0])
    num_items = int(image_features.shape[0])
    if user_text.shape[0] != num_users:
        raise ValueError("user_image and user_text have different user counts.")
    if text_features.shape[0] != num_items:
        raise ValueError("image and text embeddings have different item counts.")
    if n_ui_layers < 0:
        raise ValueError("n_ui_layers must be non-negative.")

    adjacency = build_normalized_ui_adjacency(
        users, items, num_users, num_items, device
    )
    with torch.no_grad():
        image_projected = _linear_from_state(
            image_features.to(device), state, "image_trs", device
        )
        text_projected = _linear_from_state(
            text_features.to(device), state, "text_trs", device
        )
        image_projected = F.normalize(image_projected, dim=-1)
        text_projected = F.normalize(text_projected, dim=-1)
        initial_users = torch.cat(
            (user_image.to(device), user_text.to(device)), dim=1
        )
        initial_items = torch.cat((image_projected, text_projected), dim=1)
        if initial_users.shape[1] != initial_items.shape[1]:
            raise ValueError(
                "User and item representation widths differ: {} versus {}."
                .format(initial_users.shape[1], initial_items.shape[1])
            )

        propagated = torch.cat((initial_users, initial_items), dim=0)
        layers = [propagated]
        for _ in range(n_ui_layers):
            propagated = torch.sparse.mm(adjacency, propagated)
            layers.append(propagated)
        averaged = torch.stack(layers, dim=0).mean(dim=0)
        result = averaged[:num_users].cpu()
    return result


def _validate_k_values(
    k_values: Iterable[int], matrix_shape: Tuple[int, int]
) -> Tuple[int, ...]:
    try:
        values = tuple(k_values)
    except TypeError as error:
        raise ValueError("k_values must be an iterable of integers.") from error
    if not values:
        raise ValueError("At least one k value is required.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError("Every k value must be an integer.")
    checked = tuple(sorted(set(int(value) for value in values)))
    maximum_rank = min(matrix_shape)
    if checked[0] <= 0 or checked[-1] > maximum_rank:
        raise ValueError(
            "k values must be in [1, min(matrix.shape)={}].".format(maximum_rank)
        )
    return checked


def analyze_embedding_spectrum(
    user_embeddings: Any,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    centered: bool = False,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute exact singular values, top-k energy, and band contribution."""
    if isinstance(user_embeddings, torch.Tensor):
        matrix = user_embeddings.detach().cpu().numpy()
    else:
        matrix = np.asarray(user_embeddings)
    if matrix.ndim != 2:
        raise ValueError("user_embeddings must be a two-dimensional matrix.")
    if min(matrix.shape) <= 1:
        raise ValueError("user_embeddings must have at least two rows and columns.")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("user_embeddings must contain only finite values.")
    if centered:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)

    frobenius_energy = float(np.square(matrix).sum())
    if frobenius_energy <= 0.0:
        raise ValueError("user_embeddings must have positive Frobenius energy.")
    checked_k_values = _validate_k_values(k_values, matrix.shape)

    # compute_uv=False avoids allocating the very large num_users x user_dim U
    # matrix.  All requested metrics depend only on the singular values.
    singular_values = np.linalg.svd(
        matrix, full_matrices=False, compute_uv=False
    )
    singular_values = np.sort(singular_values)[::-1]
    squared = np.square(singular_values)

    top_k = []
    for rank in checked_k_values:
        energy = float(squared[:rank].sum())
        top_k.append({
            "k": int(rank),
            "energy": energy,
            "spectral_energy_contribution": float(
                np.clip(energy / frobenius_energy, 0.0, 1.0)
            ),
        })

    maximum_k = checked_k_values[-1]
    maximum_top_energy = float(squared[:maximum_k].sum())
    bands = []
    previous_end = 0
    for end_rank in checked_k_values:
        band_energy = float(squared[previous_end:end_rank].sum())
        bands.append({
            "label": "{}-{}".format(previous_end + 1, end_rank),
            "start_rank": int(previous_end + 1),
            "end_rank": int(end_rank),
            "energy": band_energy,
            "global_energy_contribution": float(band_energy / frobenius_energy),
            "within_top_{}_share".format(maximum_k): float(
                band_energy / maximum_top_energy
            ),
        })
        previous_end = end_rank

    result: Dict[str, Any] = {
        "analysis_type": "single_view_pgl_user_representation_spectrum",
        "source": source,
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "spectral_centered": bool(centered),
        "svd": {
            "method": "numpy.linalg.svd(compute_uv=False)",
            "exact": True,
            "singular_values": singular_values.tolist(),
        },
        "frobenius_energy": frobenius_energy,
        "svd_energy": float(squared.sum()),
        "k_values": list(checked_k_values),
        "top_k_spectral_energy": top_k,
        "spectral_bands": bands,
        "tail_after_top_{}_global_energy_contribution".format(maximum_k): float(
            max(0.0, 1.0 - maximum_top_energy / frobenius_energy)
        ),
        "note": (
            "Single-view spectral concentration only; complementarity requires "
            "a second aligned representation view."
        ),
    }
    return result


def _nested_value(payload: Any, dotted_key: str) -> Any:
    current = payload
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError("Tensor key {!r} was not found.".format(dotted_key))
        current = current[part]
    return current


def load_embedding_tensor(path: Path, tensor_key: Optional[str]) -> torch.Tensor:
    payload = _torch_load(path)
    if tensor_key:
        value = _nested_value(payload, tensor_key)
    elif isinstance(payload, torch.Tensor):
        value = payload
    else:
        candidates = (
            "user_embeddings",
            "u_g_embeddings",
            "representations.full_users",
            "representations.fused_users",
        )
        found = []
        for candidate in candidates:
            try:
                candidate_value = _nested_value(payload, candidate)
            except ValueError:
                continue
            if isinstance(candidate_value, torch.Tensor):
                found.append((candidate, candidate_value))
        if len(found) != 1:
            raise ValueError(
                "Could not select one embedding tensor automatically; supply "
                "--tensor-key (for example representations.full_users)."
            )
        _, value = found[0]
    if not isinstance(value, torch.Tensor):
        raise ValueError("Selected embedding value is not a torch.Tensor.")
    return value.detach().cpu()


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return device


def _scalar_int(value: Any, name: str) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "{} is not scalar in config; supply --{} explicitly.".format(
                    name, name.replace("_", "-")
                )
            )
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("{} must be an integer.".format(name))
    return int(value)


def _print_report(result: Mapping[str, Any]) -> None:
    rows, columns = result["matrix_shape"]
    print("PGL post-propagation user representation spectrum")
    print("Source: {}".format(result.get("source")))
    print("Matrix: {} users x {} dimensions".format(rows, columns))
    print("Centered before SVD: {}".format(result["spectral_centered"]))
    print("Frobenius energy: {:.6f}".format(result["frobenius_energy"]))
    print()
    print("Spectral top-k")
    print("{:>6} {:>18} {:>18}".format("k", "energy", "C(k)"))
    for row in result["top_k_spectral_energy"]:
        print(
            "{k:6d} {energy:18.6f} {spectral_energy_contribution:18.6f}".format(
                **row
            )
        )

    maximum_k = result["k_values"][-1]
    within_key = "within_top_{}_share".format(maximum_k)
    print()
    print("Spectral-band contribution")
    print(
        "{:>9} {:>18} {:>18} {:>18}".format(
            "band", "energy", "global", "within_top_{}".format(maximum_k)
        )
    )
    for row in result["spectral_bands"]:
        print(
            "{label:>9} {energy:18.6f} "
            "{global_energy_contribution:18.6f} {within_share:18.6f}".format(
                within_share=row[within_key], **row
            )
        )
    tail_key = "tail_after_top_{}_global_energy_contribution".format(maximum_k)
    print(
        "Tail after top-{} (global contribution): {:.6f}".format(
            maximum_k, result[tail_key]
        )
    )
    print(
        "Note: this is single-view spectral concentration, not two-view "
        "spectral complementarity."
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact SVD, spectral top-k energy, and spectral-band "
            "contribution for PGL user representations."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="PGL .pth checkpoint or PGL fallback *-analysis.pt artifact.",
    )
    source.add_argument(
        "--embedding-file",
        type=Path,
        help="A .pt file containing an already computed user matrix.",
    )
    parser.add_argument(
        "--tensor-key",
        default=None,
        help="Dotted tensor key used with --embedding-file.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset override/fallback for a checkpoint without config.",
    )
    parser.add_argument(
        "--interaction-file",
        type=Path,
        default=None,
        help="Explicit interaction file used to reconstruct the train graph.",
    )
    parser.add_argument(
        "--n-ui-layers",
        type=int,
        default=None,
        help="Override UI propagation layers (normally read from checkpoint).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for propagation: auto, cpu, cuda, or cuda:N (default: auto).",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        metavar="K",
        help="Cumulative ranks and band boundaries (default: 8 16 32 64).",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Subtract the mean user vector before SVD (default: uncentered).",
    )
    parser.add_argument(
        "--save-user-embeddings",
        type=Path,
        default=None,
        help="Optional .pt output for reconstructed post-propagation users.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for all singular values and metrics.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.embedding_file is not None:
            embedding_path = arguments.embedding_file.expanduser().resolve()
            if not embedding_path.is_file():
                raise ValueError(
                    "Embedding file does not exist: {}".format(embedding_path)
                )
            user_embeddings = load_embedding_tensor(
                embedding_path, arguments.tensor_key
            )
            source_description = str(embedding_path)
        else:
            checkpoint_path = arguments.checkpoint.expanduser().resolve()
            if not checkpoint_path.is_file():
                raise ValueError(
                    "Checkpoint does not exist: {}".format(checkpoint_path)
                )
            payload = _torch_load(checkpoint_path)
            state, checkpoint_config = _checkpoint_parts(payload)
            config = _load_merged_config(checkpoint_config, arguments.dataset)
            normalized_state = _normalize_state_keys(state)
            num_users = int(
                _state_tensor(normalized_state, "user_image.weight").shape[0]
            )
            num_items = int(
                _state_tensor(normalized_state, "image_embedding.weight").shape[0]
            )
            interaction_file = _resolve_interaction_file(
                config, arguments.interaction_file
            )
            train_users, train_items = load_training_edges(
                interaction_file, config, num_users, num_items
            )
            layer_value = (
                arguments.n_ui_layers
                if arguments.n_ui_layers is not None
                else config.get("n_ui_layers")
            )
            if layer_value is None:
                raise ValueError(
                    "n_ui_layers is absent; supply --n-ui-layers."
                )
            n_ui_layers = _scalar_int(layer_value, "n_ui_layers")
            device = _choose_device(arguments.device)
            user_embeddings = reconstruct_post_propagation_users(
                normalized_state,
                train_users,
                train_items,
                n_ui_layers,
                device,
            )
            source_description = (
                "{} -> PGL.forward(full_norm_adj)[0], n_ui_layers={}, edges={}"
                .format(checkpoint_path, n_ui_layers, train_users.size)
            )

        if arguments.save_user_embeddings is not None:
            embedding_output = arguments.save_user_embeddings.expanduser().resolve()
            embedding_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(user_embeddings, embedding_output)
            print("Saved user representations to {}".format(embedding_output))

        result = analyze_embedding_spectrum(
            user_embeddings,
            k_values=arguments.k_values,
            centered=arguments.center,
            source=source_description,
        )
        _print_report(result)
        if arguments.output_json is not None:
            output_path = arguments.output_json.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(
                    result,
                    output_file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                output_file.write("\n")
            print("Saved JSON result to {}".format(output_path))
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
