"""Reusable checkpoint analyses for DUAL_MODALITY edge masks."""

import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_raw_features(config, train_dataset):
    """Load fixed pre-training features instead of learned feature tables."""
    dataset_path = Path(train_dataset.dataset_path)
    image_path = dataset_path / str(config['vision_feature_file'])
    text_path = dataset_path / str(config['text_feature_file'])
    if not image_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(
            'Could not find raw image/text features in {}'.format(
                dataset_path
            )
        )
    return (
        np.load(image_path).astype(np.float32, copy=False),
        np.load(text_path).astype(np.float32, copy=False),
    )


@torch.no_grad()
def collect_mask_snapshot(model):
    """Collect probabilities and exact hard selections (or soft top-k proxy)."""
    if model.edge_weights_are_shared:
        raise ValueError('Mask analysis requires separate image/text masks.')
    model.eval()
    model.set_mask_assignment_mode('normal')
    model.forward(model.norm_adj)
    representations = model.latest_representations

    if model.uses_user_normalized_weights:
        probabilities = {
            modality: representations[
                '{}_edge_distribution'.format(modality)
            ].detach().cpu()
            for modality in ('image', 'text')
        }
        relative_weights = {
            modality: representations[
                '{}_relative_edge_weights'.format(modality)
            ].detach().cpu()
            for modality in ('image', 'text')
        }
        weight_semantics = 'per_user_distribution_q'
    else:
        probabilities = {
            modality: torch.sigmoid(
                representations['{}_mask_logits'.format(modality)]
            ).detach().cpu()
            for modality in ('image', 'text')
        }
        relative_weights = None
        weight_semantics = 'sigmoid_probability'
    selections = {}
    if model.mask_graph_mode == 'hard':
        for modality in ('image', 'text'):
            selected_indices = getattr(
                model, '{}_hard_eval_indices'.format(modality)
            )
            selected = torch.zeros(
                model.num_interactions, dtype=torch.bool
            )
            selected[selected_indices.detach().cpu()] = True
            selections[modality] = selected
        selection_kind = 'exact_hard_mask'
    else:
        for modality in ('image', 'text'):
            selected_indices = torch.topk(
                probabilities[modality],
                model.hard_keep_count,
                sorted=False,
            ).indices
            selected = torch.zeros(
                model.num_interactions, dtype=torch.bool
            )
            selected[selected_indices] = True
            selections[modality] = selected
        selection_kind = 'global_topk_probability_proxy_for_soft_mask'

    return {
        'edge_users': model.edge_indices[0].detach().cpu(),
        'edge_items': model.edge_indices[1].detach().cpu(),
        'probabilities': probabilities,
        'relative_weights': relative_weights,
        'weight_semantics': weight_semantics,
        'selected': selections,
        'selection_kind': selection_kind,
    }


def probability_summary(probabilities, low=0.05, high=0.95):
    values = np.asarray(probabilities, dtype=np.float64)
    quantile_levels = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    quantiles = np.quantile(values, quantile_levels)
    result = {
        'mean': float(values.mean()),
        'std': float(values.std()),
        'near_zero_rate': float((values <= low).mean()),
        'near_one_rate': float((values >= high).mean()),
    }
    result.update({
        'q{:g}'.format(level): float(value)
        for level, value in zip(quantile_levels, quantiles)
    })
    return result


def _edge_positions_by_user(edge_users, num_users):
    counts = np.bincount(edge_users, minlength=num_users)
    order = np.argsort(edge_users, kind='stable')
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    return [
        order[boundaries[user]:boundaries[user + 1]]
        for user in range(num_users)
    ]


def _per_user_js(
    user_positions,
    image_probabilities,
    text_probabilities,
    num_users,
    eps,
):
    result = np.zeros(num_users, dtype=np.float64)
    for user in range(num_users):
        selected = user_positions[user]
        if selected.size == 0:
            continue
        image_mass = image_probabilities[selected] + eps
        text_mass = text_probabilities[selected] + eps
        image_distribution = image_mass / image_mass.sum()
        text_distribution = text_mass / text_mass.sum()
        mixture = 0.5 * (image_distribution + text_distribution)
        result[user] = 0.5 * (
            np.sum(image_distribution * np.log(image_distribution / mixture))
            + np.sum(
                text_distribution * np.log(text_distribution / mixture)
            )
        )
    return result


def _rank_discrepancy(
    user_positions, image_probabilities, text_probabilities, num_users
):
    normalized_rank_gap = np.full(num_users, np.nan, dtype=np.float64)
    rank_correlation = np.full(num_users, np.nan, dtype=np.float64)
    for user in range(num_users):
        selected = user_positions[user]
        degree = selected.size
        if degree < 2:
            continue
        image_order = np.argsort(-image_probabilities[selected])
        text_order = np.argsort(-text_probabilities[selected])
        image_ranks = np.empty(degree, dtype=np.float64)
        text_ranks = np.empty(degree, dtype=np.float64)
        image_ranks[image_order] = np.arange(degree)
        text_ranks[text_order] = np.arange(degree)
        normalized_rank_gap[user] = np.mean(
            np.abs(image_ranks - text_ranks)
        ) / (degree - 1)
        if np.std(image_ranks) > 0.0 and np.std(text_ranks) > 0.0:
            rank_correlation[user] = np.corrcoef(
                image_ranks, text_ranks
            )[0, 1]
    return normalized_rank_gap, rank_correlation


def _per_user_probability_statistics(
    user_positions, probabilities, num_users
):
    result = {
        statistic: np.full(num_users, np.nan, dtype=np.float64)
        for statistic in ('mean', 'std', 'q25', 'q50', 'q75')
    }
    for user, positions in enumerate(user_positions):
        if positions.size == 0:
            continue
        values = probabilities[positions]
        result['mean'][user] = values.mean()
        result['std'][user] = values.std()
        result['q25'][user], result['q50'][user], result['q75'][user] = (
            np.quantile(values, [0.25, 0.5, 0.75])
        )
    return result


def per_user_mask_statistics(snapshot, num_users, eps=1e-8):
    edge_users = snapshot['edge_users'].numpy()
    image_probabilities = snapshot['probabilities']['image'].numpy()
    text_probabilities = snapshot['probabilities']['text'].numpy()
    image_selected = snapshot['selected']['image'].numpy()
    text_selected = snapshot['selected']['text'].numpy()

    degree = np.bincount(edge_users, minlength=num_users)
    user_positions = _edge_positions_by_user(edge_users, num_users)
    image_kept = np.bincount(
        edge_users, weights=image_selected.astype(np.float64),
        minlength=num_users,
    ).astype(np.int64)
    text_kept = np.bincount(
        edge_users, weights=text_selected.astype(np.float64),
        minlength=num_users,
    ).astype(np.int64)
    both = image_selected & text_selected
    image_only = image_selected & ~text_selected
    text_only = ~image_selected & text_selected
    neither = ~image_selected & ~text_selected
    group_masks = {
        'both': both,
        'image_only': image_only,
        'text_only': text_only,
        'neither': neither,
    }
    group_counts = {
        group: np.bincount(
            edge_users,
            weights=mask.astype(np.float64),
            minlength=num_users,
        ).astype(np.int64)
        for group, mask in group_masks.items()
    }
    union_count = image_kept + text_kept - group_counts['both']
    jaccard = np.full(num_users, np.nan, dtype=np.float64)
    nonempty_union = union_count > 0
    jaccard[nonempty_union] = (
        group_counts['both'][nonempty_union]
        / union_count[nonempty_union]
    )
    valid_degree = degree > 0
    image_keep_rate = np.zeros(num_users, dtype=np.float64)
    text_keep_rate = np.zeros(num_users, dtype=np.float64)
    image_keep_rate[valid_degree] = (
        image_kept[valid_degree] / degree[valid_degree]
    )
    text_keep_rate[valid_degree] = (
        text_kept[valid_degree] / degree[valid_degree]
    )
    normalized_rank_gap, rank_correlation = _rank_discrepancy(
        user_positions, image_probabilities, text_probabilities, num_users
    )
    per_user_js = _per_user_js(
        user_positions,
        image_probabilities,
        text_probabilities,
        num_users,
        eps,
    )
    image_probability_statistics = _per_user_probability_statistics(
        user_positions, image_probabilities, num_users
    )
    text_probability_statistics = _per_user_probability_statistics(
        user_positions, text_probabilities, num_users
    )

    if np.std(image_probabilities) == 0.0 or np.std(text_probabilities) == 0.0:
        global_probability_correlation = np.nan
    else:
        global_probability_correlation = float(
            np.corrcoef(image_probabilities, text_probabilities)[0, 1]
        )

    return {
        'degree': degree,
        'image_kept': image_kept,
        'text_kept': text_kept,
        'image_keep_rate': image_keep_rate,
        'text_keep_rate': text_keep_rate,
        'both_empty': union_count == 0,
        'jaccard': jaccard,
        'per_user_js': per_user_js,
        'normalized_rank_gap': normalized_rank_gap,
        'rank_correlation': rank_correlation,
        'image_probability_statistics': image_probability_statistics,
        'text_probability_statistics': text_probability_statistics,
        'global_probability_correlation': global_probability_correlation,
        'group_masks': group_masks,
        'group_counts': group_counts,
        'global_group_rates': {
            group: float(mask.mean()) for group, mask in group_masks.items()
        },
        'image_zero_edge_user_rate': float(
            (image_kept[valid_degree] == 0).mean()
        ),
        'text_zero_edge_user_rate': float(
            (text_kept[valid_degree] == 0).mean()
        ),
    }


def random_mask_baseline(snapshot, user_statistics, repeats=100, seed=2024):
    """Random masks preserving image/text kept-edge counts for every user."""
    if repeats <= 0:
        raise ValueError('repeats must be positive.')
    num_edges = snapshot['edge_users'].numel()
    degree = user_statistics['degree']
    image_kept = user_statistics['image_kept']
    text_kept = user_statistics['text_kept']
    users_with_history = degree > 0
    rng = np.random.default_rng(seed)
    results = []
    for repeat in range(repeats):
        intersection = np.zeros_like(degree)
        intersection[users_with_history] = rng.hypergeometric(
            image_kept[users_with_history],
            degree[users_with_history] - image_kept[users_with_history],
            text_kept[users_with_history],
        )
        image_only = image_kept - intersection
        text_only = text_kept - intersection
        union = image_kept + text_kept - intersection
        neither = degree - union
        valid = union > 0
        results.append({
            'repeat': repeat,
            'both': float(intersection.sum() / num_edges),
            'image_only': float(image_only.sum() / num_edges),
            'text_only': float(text_only.sum() / num_edges),
            'neither': float(neither.sum() / num_edges),
            'mean_user_jaccard': float(
                np.mean(intersection[valid] / union[valid])
            ) if np.any(valid) else np.nan,
            'both_empty_user_rate': float(
                (union[users_with_history] == 0).mean()
            ),
        })
    return results


def edge_group_characteristics(snapshot, user_statistics, num_items):
    edge_users = snapshot['edge_users'].numpy()
    edge_items = snapshot['edge_items'].numpy()
    item_popularity = np.bincount(edge_items, minlength=num_items)
    labels = np.full(edge_users.size, 'neither', dtype=object)
    for group in ('both', 'image_only', 'text_only'):
        labels[user_statistics['group_masks'][group]] = group
    return {
        'edge_users': edge_users,
        'edge_items': edge_items,
        'group': labels,
        'item_popularity': item_popularity[edge_items],
        'user_activity': user_statistics['degree'][edge_users],
    }


def leave_one_out_content_similarity(
    edge_users, edge_items, item_features
):
    """Cosine(item, mean(other history items)) without materializing E x D."""
    features = np.asarray(item_features, dtype=np.float32)
    feature_norm = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(feature_norm, 1e-12)
    result = np.full(edge_users.size, np.nan, dtype=np.float32)
    num_users = int(edge_users.max()) + 1 if edge_users.size else 0
    user_positions = _edge_positions_by_user(edge_users, num_users)
    for positions in user_positions:
        if positions.size < 2:
            continue
        history_features = features[edge_items[positions]]
        other_sum = history_features.sum(axis=0, keepdims=True) - history_features
        other_norm = np.linalg.norm(other_sum, axis=1)
        numerator = np.sum(history_features * other_sum, axis=1)
        valid = other_norm > 1e-12
        similarities = np.full(positions.size, np.nan, dtype=np.float32)
        similarities[valid] = numerator[valid] / other_norm[valid]
        result[positions] = similarities
    return result


def _representation_change(full_users, alternative_users):
    direction_change = 1.0 - F.cosine_similarity(
        full_users, alternative_users, dim=1, eps=1e-12
    )
    norm_ratio = alternative_users.norm(dim=1) / full_users.norm(
        dim=1
    ).clamp_min(1e-12)
    return {
        'cosine_distance': direction_change.detach().cpu().numpy(),
        'norm_ratio': norm_ratio.detach().cpu().numpy(),
    }


@torch.no_grad()
def controlled_propagation_effects(model):
    """Compare full/masked propagation while holding H0 fixed per modality."""
    model.eval()
    model.set_mask_assignment_mode('normal')
    model.forward(model.norm_adj)
    representations = model.latest_representations
    image_features, text_features = model._project_item_features()
    results = {}
    for modality, item_features, user_table in (
        ('image', image_features, model.masked_user_image.weight),
        ('text', text_features, model.masked_user_text.weight),
    ):
        initial = torch.cat((user_table, item_features), dim=0)
        full_output = model._propagate_ui_graph(
            model.norm_adj, initial, model.n_ui_layers
        )
        mask_logits = representations['{}_mask_logits'.format(modality)]
        masked_adjacency, _ = model._masked_ui_adjacency(
            modality, mask_logits
        )
        masked_output = model._propagate_ui_graph(
            masked_adjacency,
            initial,
            model.n_ui_layers,
            propagation_scale=model._propagation_gamma(modality),
        )
        results[modality] = {
            'masked': _representation_change(
                full_output[:model.n_users],
                masked_output[:model.n_users],
            )
        }

        if (
            model.mask_graph_mode == 'soft'
            and (
                model.mask_degree_mode == 'full'
                or model.uses_user_normalized_weights
            )
        ):
            if model.uses_user_normalized_weights:
                constant_logits = torch.zeros_like(mask_logits)
                constant_probability = None
            else:
                probability = torch.sigmoid(mask_logits).mean().clamp(
                    1e-6, 1.0 - 1e-6
                )
                constant_logit = torch.log(
                    probability / (1.0 - probability)
                )
                constant_logits = torch.full_like(
                    mask_logits, float(constant_logit.item())
                )
                constant_probability = float(probability.item())
            constant_adjacency, _ = model._masked_ui_adjacency(
                modality, constant_logits
            )
            constant_output = model._propagate_ui_graph(
                constant_adjacency,
                initial,
                model.n_ui_layers,
                propagation_scale=model._propagation_gamma(modality),
            )
            results[modality]['constant_mask'] = _representation_change(
                full_output[:model.n_users],
                constant_output[:model.n_users],
            )
            results[modality][
                'constant_probability'
            ] = constant_probability
            if model.uses_user_normalized_weights:
                results[modality]['constant_relative_weight'] = 1.0
    return results


def _within_user_permutation(edge_users, seed):
    edge_users_cpu = edge_users.detach().cpu().numpy()
    permutation = np.arange(edge_users_cpu.size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    num_users = int(edge_users_cpu.max()) + 1 if edge_users_cpu.size else 0
    for positions in _edge_positions_by_user(edge_users_cpu, num_users):
        permutation[positions] = rng.permutation(positions)
    return torch.from_numpy(permutation).to(edge_users.device)


@contextmanager
def mask_intervention(model, intervention='original', seed=2024):
    """Temporarily alter only mask routing/edge assignment for evaluation."""
    valid = {
        'original',
        'constant_mask',
        'permute_image',
        'permute_text',
        'permute_both',
        'swapped',
        'full_adjacency',
    }
    if intervention not in valid:
        raise ValueError('Unknown mask intervention: {}'.format(intervention))
    if model.edge_weights_are_shared:
        raise ValueError('Mask interventions require separate masks.')
    if intervention == 'constant_mask' and model.mask_graph_mode != 'soft':
        raise ValueError(
            'constant_mask intervention requires mask_graph_mode=soft.'
        )

    previous_assignment = model.mask_assignment_mode
    had_assigned_override = '_assigned_mask_logits' in model.__dict__
    previous_assigned = model.__dict__.get('_assigned_mask_logits')
    had_adjacency_override = '_masked_ui_adjacency' in model.__dict__
    previous_adjacency = model.__dict__.get('_masked_ui_adjacency')
    original_assigned = model._assigned_mask_logits
    model.set_mask_assignment_mode('normal')

    if intervention == 'constant_mask':
        def constant_logits(image_features=None, text_features=None):
            logits = dict(
                original_assigned(image_features, text_features)
            )
            result = []
            for modality in ('image', 'text'):
                probability = torch.sigmoid(logits[modality]).mean().clamp(
                    1e-6, 1.0 - 1e-6
                )
                logit = torch.log(probability / (1.0 - probability))
                result.append((
                    modality,
                    torch.full_like(logits[modality], float(logit.item())),
                ))
            return tuple(result)

        model._assigned_mask_logits = constant_logits
    elif intervention == 'swapped':
        model.set_mask_assignment_mode('swapped')
    elif intervention.startswith('permute_'):
        permutation = _within_user_permutation(model.edge_indices[0], seed)

        def permuted_logits(image_features=None, text_features=None):
            logits = dict(
                original_assigned(image_features, text_features)
            )
            if intervention in {'permute_image', 'permute_both'}:
                logits['image'] = logits['image'].index_select(
                    0, permutation
                )
            if intervention in {'permute_text', 'permute_both'}:
                logits['text'] = logits['text'].index_select(0, permutation)
            return (('image', logits['image']), ('text', logits['text']))

        model._assigned_mask_logits = permuted_logits
    elif intervention == 'full_adjacency':

        def full_masked_adjacency(modality, mask_logits):
            del modality
            return (
                model.norm_adj,
                torch.ones_like(mask_logits),
            )

        model._masked_ui_adjacency = full_masked_adjacency

    try:
        yield model
    finally:
        if had_assigned_override:
            model._assigned_mask_logits = previous_assigned
        else:
            model.__dict__.pop('_assigned_mask_logits', None)
        if had_adjacency_override:
            model._masked_ui_adjacency = previous_adjacency
        else:
            model.__dict__.pop('_masked_ui_adjacency', None)
        model.set_mask_assignment_mode(previous_assignment)


@torch.no_grad()
def triplet_diagnostics(model, triplets):
    users, positive_items, negative_items = triplets[:3]
    all_users, all_items = model.forward(model.norm_adj)
    representations = model.latest_representations

    def margins(user_table, item_table):
        user_embeddings = user_table[users]
        return torch.sum(
            user_embeddings
            * (item_table[positive_items] - item_table[negative_items]),
            dim=1,
        )

    margin_tensors = {
        'image_masked': margins(
            representations['image_masked_users'],
            representations['image_masked_items'],
        ),
        'text_masked': margins(
            representations['text_masked_users'],
            representations['text_masked_items'],
        ),
        'final': margins(all_users, all_items),
    }
    summary = {}
    for name, margin in margin_tensors.items():
        summary[name] = {
            'bpr': float(F.softplus(-margin).mean().item()),
            'mean_margin': float(margin.mean().item()),
            'positive_margin_rate': float((margin > 0).float().mean().item()),
        }
    return {
        'summary': summary,
        'per_triplet': {
            name: values.detach().cpu()
            for name, values in margin_tensors.items()
        },
    }


def evaluate_intervention(
    model,
    eval_data,
    config,
    triplets,
    intervention='original',
    seed=2024,
):
    from mask_analysis.dual_modality_mask_swap import evaluate_topk

    with mask_intervention(model, intervention, seed):
        metrics, rankings = evaluate_topk(model, eval_data, config)
        triplet_result = triplet_diagnostics(model, triplets)
        representations = model.latest_representations
        if model.uses_user_normalized_weights:
            mask_probability_means = {
                modality: float(
                    representations[
                        '{}_edge_distribution'.format(modality)
                    ].mean().item()
                )
                for modality in ('image', 'text')
            }
            relative_weight_means = {
                modality: float(
                    representations[
                        '{}_relative_edge_weights'.format(modality)
                    ].mean().item()
                )
                for modality in ('image', 'text')
            }
        else:
            mask_probability_means = {
                modality: float(torch.sigmoid(
                    representations['{}_mask_logits'.format(modality)]
                ).mean().item())
                for modality in ('image', 'text')
            }
            relative_weight_means = None
    return {
        'intervention': intervention,
        'seed': int(seed),
        'metrics': metrics,
        'rankings': rankings,
        'triplets': triplet_result,
        'mask_probability_means': mask_probability_means,
        'relative_weight_means': relative_weight_means,
    }


@torch.no_grad()
def collect_gate_snapshot(model):
    model.eval()
    model.set_mask_assignment_mode('normal')
    model.forward(model.norm_adj)
    representations = model.latest_representations
    result = {}
    for entity in ('user', 'item'):
        for modality in ('image', 'text'):
            gate = representations[
                '{}_{}_gate'.format(modality, entity)
            ]
            result['{}_{}'.format(modality, entity)] = (
                gate.mean(dim=1).detach().cpu().numpy()
            )
    return result
