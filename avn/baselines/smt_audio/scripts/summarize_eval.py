"""Summarize eval results from per-episode stats JSON files.

Usage:
    python scripts/summarize_eval.py data/models/av_nav_tta_single_source
    python scripts/summarize_eval.py data/models/*_tta_*           # glob multiple model dirs

For each model dir, looks for `tb/<split>_stats_<seed>.json` files,
prints aggregated mean for each found split.
"""
import glob
import json
import os
import sys
from collections import OrderedDict


METRIC_ORDER = [
    'success',
    'spl',
    'softspl',
    'sna',
    'sws',
    'na',
    'distance_to_goal',
    'normalized_distance_to_goal',
    'reward',
    'geodesic_distance',
    'euclidean_distance',
]


def aggregate(stats_path: str) -> OrderedDict:
    with open(stats_path) as f:
        stats = json.load(f)
    n = len(stats)
    if n == 0:
        return OrderedDict([('episodes', 0)])
    keys = list(next(iter(stats.values())).keys())
    means = OrderedDict([('episodes', n)])
    # ordered metrics first
    for k in METRIC_ORDER:
        if k in keys:
            means[k] = sum(v[k] for v in stats.values()) / n
    # any extras
    for k in keys:
        if k not in means:
            means[k] = sum(v[k] for v in stats.values()) / n
    return means


def summarize_dir(model_dir: str):
    tb_dir = os.path.join(model_dir, 'tb')
    if not os.path.isdir(tb_dir):
        print(f'[skip] no tb/ in {model_dir}')
        return
    json_files = sorted(glob.glob(os.path.join(tb_dir, '*_stats_*.json')))
    if not json_files:
        print(f'[skip] no *_stats_*.json in {tb_dir}')
        return
    print(f'\n=== {model_dir} ===')
    for jf in json_files:
        means = aggregate(jf)
        tag = os.path.basename(jf).replace('.json', '')
        print(f'  [{tag}]  n={means["episodes"]}')
        for k, v in means.items():
            if k == 'episodes':
                continue
            print(f'    {k:30s} = {v:.4f}')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    targets = []
    for arg in sys.argv[1:]:
        targets.extend(sorted(glob.glob(arg)))
    if not targets:
        print('no model dirs matched')
        sys.exit(1)
    for d in targets:
        if os.path.isdir(d):
            summarize_dir(d)


if __name__ == '__main__':
    main()
