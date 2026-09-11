# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import yaml
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='PGL', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='optional YAML file whose values override dataset/model config',
    )

    args, _ = parser.parse_known_args()

    config_dict = {
        'gpu_id': 0,
    }
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as config_file:
            overrides = yaml.safe_load(config_file) or {}
        if not isinstance(overrides, dict):
            raise ValueError('--config must point to a YAML mapping.')
        config_dict.update(overrides)

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True, mg=args.mg)


