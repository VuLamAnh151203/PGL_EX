"""Evaluate a DUAL_MODALITY checkpoint before and after swapping masks.

The ablation keeps every checkpoint parameter fixed. It only routes the
learned text edge-mask vector to the image masked branch and the learned image
edge-mask vector to the text masked branch during the second evaluation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mask_analysis.dual_modality_diagnostics import (  # noqa: E402
    _load_checkpoint,
)


def _parse_topk(value):
    topk = sorted({int(item.strip()) for item in value.split(',')})
    if not topk or topk[0] <= 0:
        raise argparse.ArgumentTypeError(
            'topk must contain positive integers.'
        )
    return topk


@torch.no_grad()
def evaluate_topk(model, eval_data, config):
    """Run the same full-sort ranking protocol used by PGL Trainer."""
    from utils.topk_evaluator import TopKEvaluator

    model.eval()
    evaluator = TopKEvaluator(config)
    ranking_batches = []
    for batched_data in eval_data:
        scores = model.full_sort_predict(batched_data)
        masked_items = batched_data[1]
        scores[masked_items[0], masked_items[1]] = -1e10
        ranking_batches.append(
            torch.topk(
                scores, max(config['topk']), dim=-1
            ).indices.cpu()
        )
    metrics = evaluator.evaluate(ranking_batches, eval_data)
    rankings = torch.cat(ranking_batches, dim=0)
    return metrics, rankings


def compare_rankings(normal_rankings, swapped_rankings, topk):
    """Measure whether swapping masks changes recommendation lists."""
    if normal_rankings.shape != swapped_rankings.shape:
        raise ValueError('Normal and swapped ranking shapes do not match.')
    comparisons = {}
    num_users = normal_rankings.size(0)
    for k in topk:
        normal = normal_rankings[:, :k]
        swapped = swapped_rankings[:, :k]
        changed_order = torch.any(normal != swapped, dim=1)
        changed_set = torch.any(
            torch.sort(normal, dim=1).values
            != torch.sort(swapped, dim=1).values,
            dim=1,
        )
        item_overlap = (
            (normal.unsqueeze(2) == swapped.unsqueeze(1))
            .any(dim=2)
            .float()
            .mean(dim=1)
        )
        comparisons['Top{}'.format(k)] = {
            'num_users': int(num_users),
            'users_with_changed_order': int(changed_order.sum().item()),
            'changed_order_rate': float(changed_order.float().mean().item()),
            'users_with_changed_item_set': int(changed_set.sum().item()),
            'changed_item_set_rate': float(changed_set.float().mean().item()),
            'mean_item_overlap_rate': float(item_overlap.mean().item()),
        }
    return comparisons


def metric_deltas(normal_metrics, swapped_metrics):
    if set(normal_metrics) != set(swapped_metrics):
        raise ValueError('Normal and swapped metric keys do not match.')
    return {
        metric: round(
            float(swapped_metrics[metric]) - float(normal_metrics[metric]),
            8,
        )
        for metric in normal_metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Compare DUAL_MODALITY performance before and after routing '
            'the image/text masks to the opposite modality branches.'
        )
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=('valid', 'test'), default='test')
    parser.add_argument(
        '--topk',
        type=_parse_topk,
        help='Optional comma-separated K values; defaults to checkpoint topk.',
    )
    parser.add_argument('--output')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    from utils.dataloader import EvalDataLoader

    config, model, train_dataset, valid_dataset, test_dataset = (
        _load_checkpoint(args.checkpoint, args.cpu)
    )
    if model.mask_sharing_mode != 'separate':
        raise ValueError(
            'Mask-swap evaluation requires mask_sharing_mode=separate; '
            'a shared image/text mask cannot be swapped.'
        )
    if args.topk is not None:
        config['topk'] = args.topk
    if max(config['topk']) > model.n_items:
        raise ValueError('Requested topk exceeds the number of items.')

    evaluation_dataset = (
        valid_dataset if args.split == 'valid' else test_dataset
    )
    eval_data = EvalDataLoader(
        config,
        evaluation_dataset,
        additional_dataset=train_dataset,
        batch_size=config['eval_batch_size'],
    )

    model.set_mask_assignment_mode('normal')
    normal_metrics, normal_rankings = evaluate_topk(
        model, eval_data, config
    )
    model.set_mask_assignment_mode('swapped')
    swapped_metrics, swapped_rankings = evaluate_topk(
        model, eval_data, config
    )
    model.set_mask_assignment_mode('normal')

    deltas = metric_deltas(normal_metrics, swapped_metrics)
    result = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'split': args.split,
        'topk': [int(k) for k in config['topk']],
        'mask_generation_mode': model.mask_generation_mode,
        'mask_graph_mode': model.mask_graph_mode,
        'normal': normal_metrics,
        'swapped': swapped_metrics,
        'delta_swapped_minus_normal': deltas,
        'performance_changed_at_reported_precision': any(
            difference != 0.0 for difference in deltas.values()
        ),
        'ranking_changes': compare_rankings(
            normal_rankings, swapped_rankings, config['topk']
        ),
    }

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.checkpoint).with_suffix(''))
        output_path += '-mask-swap-evaluation.json'
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)

    print(json.dumps(result, indent=2, sort_keys=True))
    print('Saved mask-swap evaluation to {}'.format(output_path))


if __name__ == '__main__':
    main()
