"""Post-training modality diagnostics for DUAL_MODALITY checkpoints."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.metrics import ndcg_, recall_  # noqa: E402


def _parse_topk(value):
    topk = sorted({int(item.strip()) for item in value.split(',')})
    if not topk or topk[0] <= 0:
        raise argparse.ArgumentTypeError('topk must contain positive integers.')
    return topk


def _hit_matrix(topk_indices, positive_items):
    return np.asarray(
        [
            [item in set(positives) for item in recommendations]
            for positives, recommendations in zip(
                positive_items, topk_indices
            )
        ],
        dtype=bool,
    )


def summarize_rankings(topk_by_view, positive_items, requested_topk):
    """Compute branch metrics and joint rescue/harm counts."""
    if set(topk_by_view) != {'image', 'text', 'joint'}:
        raise ValueError('Expected image, text, and joint rankings.')
    if not positive_items:
        raise ValueError('At least one evaluated user is required.')

    max_available_k = min(
        rankings.shape[1] for rankings in topk_by_view.values()
    )
    topk = [k for k in requested_topk if k <= max_available_k]
    if not topk:
        raise ValueError('No requested K fits the available ranking width.')

    positive_lengths = np.asarray(
        [len(items) for items in positive_items], dtype=np.int64
    )
    hit_matrices = {
        view: _hit_matrix(rankings[:, :max_available_k], positive_items)
        for view, rankings in topk_by_view.items()
    }
    metrics = {}
    for view, hits in hit_matrices.items():
        recall_values = recall_(hits, positive_lengths)
        ndcg_values = ndcg_(hits, positive_lengths)
        metrics[view] = {
            'Recall@{}'.format(k): float(recall_values[k - 1])
            for k in topk
        }
        metrics[view].update({
            'NDCG@{}'.format(k): float(ndcg_values[k - 1])
            for k in topk
        })

    complementarity = {}
    for k in topk:
        text_miss_count = 0
        text_hit_count = 0
        image_rescue_count = 0
        image_harm_count = 0
        for positives, text_ranking, joint_ranking in zip(
            positive_items,
            topk_by_view['text'][:, :k],
            topk_by_view['joint'][:, :k],
        ):
            text_set = set(text_ranking.tolist())
            joint_set = set(joint_ranking.tolist())
            for positive_item in positives:
                text_hit = positive_item in text_set
                joint_hit = positive_item in joint_set
                if text_hit:
                    text_hit_count += 1
                    image_harm_count += int(not joint_hit)
                else:
                    text_miss_count += 1
                    image_rescue_count += int(joint_hit)

        complementarity['Top{}'.format(k)] = {
            'image_rescue_count': image_rescue_count,
            'text_miss_count': text_miss_count,
            'image_rescue_rate_among_text_misses': (
                image_rescue_count / text_miss_count
                if text_miss_count
                else 0.0
            ),
            'image_harm_count': image_harm_count,
            'text_hit_count': text_hit_count,
            'image_harm_rate_among_text_hits': (
                image_harm_count / text_hit_count
                if text_hit_count
                else 0.0
            ),
            'joint_recall_gain_over_best_branch': (
                metrics['joint']['Recall@{}'.format(k)]
                - max(
                    metrics['image']['Recall@{}'.format(k)],
                    metrics['text']['Recall@{}'.format(k)],
                )
            ),
            'joint_ndcg_gain_over_best_branch': (
                metrics['joint']['NDCG@{}'.format(k)]
                - max(
                    metrics['image']['NDCG@{}'.format(k)],
                    metrics['text']['NDCG@{}'.format(k)],
                )
            ),
        }
    return {'ranking_metrics': metrics, 'complementarity': complementarity}


def _fixed_validation_triplets(
    users,
    positive_items,
    history_by_user,
    num_items,
    seed,
):
    rng = np.random.default_rng(seed)
    triplet_users = []
    triplet_positives = []
    triplet_negatives = []
    for user, positives in zip(users, positive_items):
        user = int(user)
        positives = [int(item) for item in positives]
        forbidden = set(history_by_user.get(user, ()))
        forbidden.update(positives)
        if len(forbidden) >= num_items:
            continue
        for positive_item in positives:
            negative_item = int(rng.integers(num_items))
            while negative_item in forbidden:
                negative_item = int(rng.integers(num_items))
            triplet_users.append(user)
            triplet_positives.append(positive_item)
            triplet_negatives.append(negative_item)
    if not triplet_users:
        raise ValueError('Could not construct any validation triplets.')
    return torch.tensor(
        [triplet_users, triplet_positives, triplet_negatives],
        dtype=torch.long,
    )


@torch.no_grad()
def evaluate_modality_diagnostics(
    model,
    eval_data,
    train_dataset,
    topk=(10, 20),
    triplet_seed=2024,
):
    """Evaluate ranking, margin, rescue, and harm diagnostics."""
    model.eval()
    requested_topk = sorted(set(int(k) for k in topk))
    max_k = min(max(requested_topk), model.n_items)
    ranking_batches = {'image': [], 'text': [], 'joint': []}

    for batched_data in eval_data:
        score_views = model.full_sort_predict_modalities(batched_data)
        masked_items = batched_data[1]
        for view, scores in score_views.items():
            scores = scores.clone()
            scores[masked_items[0], masked_items[1]] = -1e10
            ranking_batches[view].append(
                torch.topk(scores, max_k, dim=-1).indices.cpu()
            )

    topk_by_view = {
        view: torch.cat(batches, dim=0).numpy()
        for view, batches in ranking_batches.items()
    }
    positive_items = eval_data.get_eval_items()
    result = summarize_rankings(
        topk_by_view, positive_items, requested_topk
    )

    uid_field = train_dataset.uid_field
    iid_field = train_dataset.iid_field
    history_by_user = {
        int(user): set(int(item) for item in group[iid_field].values)
        for user, group in train_dataset.df.groupby(uid_field)
    }
    triplets = _fixed_validation_triplets(
        eval_data.get_eval_users().numpy(),
        positive_items,
        history_by_user,
        model.n_items,
        triplet_seed,
    ).to(model.device)
    margins = model.modality_triplet_margins(triplets)
    result['triplet_diagnostics'] = {
        'seed': int(triplet_seed),
        'num_triplets': int(triplets.size(1)),
        'image_positive_margin_rate': float(
            (margins['image'] > 0).float().mean().item()
        ),
        'text_positive_margin_rate': float(
            (margins['text'] > 0).float().mean().item()
        ),
        'joint_positive_margin_rate': float(
            (margins['joint'] > 0).float().mean().item()
        ),
        'image_mean_margin': float(margins['image'].mean().item()),
        'text_mean_margin': float(margins['text'].mean().item()),
        'joint_mean_margin': float(margins['joint'].mean().item()),
    }
    return result


def _load_checkpoint(checkpoint_path, force_cpu):
    from utils.configurator import Config
    from utils.dataloader import TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import get_model

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
        raise ValueError('Not a complete PGL trainer checkpoint.')
    saved_config = dict(checkpoint['config'])
    model_name = saved_config.get('model')
    dataset_name = saved_config.get('dataset')
    if str(model_name).upper() != 'DUAL_MODALITY':
        raise ValueError('Checkpoint model must be DUAL_MODALITY.')
    if force_cpu:
        saved_config['use_gpu'] = False

    config = Config(model_name, dataset_name, saved_config)
    complete_dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = complete_dataset.split()
    # Dataloader initialization expects these statistics to be populated.
    str(train_dataset)
    str(valid_dataset)
    str(test_dataset)
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=False,
    )
    model = get_model(model_name)(config, train_data).to(config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return config, model, train_dataset, valid_dataset, test_dataset


def main():
    from utils.dataloader import EvalDataLoader

    parser = argparse.ArgumentParser(
        description='Analyze image/text/joint DUAL_MODALITY rankings.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=('valid', 'test'), default='valid')
    parser.add_argument('--topk', type=_parse_topk, default=[10, 20])
    parser.add_argument('--triplet-seed', type=int, default=2024)
    parser.add_argument('--output')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    config, model, train_dataset, valid_dataset, test_dataset = (
        _load_checkpoint(args.checkpoint, args.cpu)
    )
    evaluation_dataset = (
        valid_dataset if args.split == 'valid' else test_dataset
    )
    eval_data = EvalDataLoader(
        config,
        evaluation_dataset,
        additional_dataset=train_dataset,
        batch_size=config['eval_batch_size'],
    )
    result = evaluate_modality_diagnostics(
        model,
        eval_data,
        train_dataset,
        topk=args.topk,
        triplet_seed=args.triplet_seed,
    )
    result['checkpoint'] = os.path.abspath(args.checkpoint)
    result['split'] = args.split

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.checkpoint).with_suffix(''))
        output_path += '-modality-diagnostics.json'
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    print('Saved diagnostics to {}'.format(output_path))


if __name__ == '__main__':
    main()
