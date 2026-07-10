#!/usr/bin/env python3
"""Compute AP/F/FT/BT metrics from continual-evaluation JSON files."""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


HIGHER_BETTER = {
    "success", "spl", "success_weighted_by_num_action",
    "reward", "success_when_silent", "normalized_distance_to_goal",
}
LOWER_BETTER = {"distance_to_goal", "num_action"}

METRIC_ALIASES = {
    "success_weighted_by_num_action": "sna",
    "normalized_distance_to_goal": "ndtg",
    "distance_to_goal": "dtg",
    "num_action": "na",
    "success_when_silent": "sws",
}

REPORT_METRICS = [
    "success", "spl", "success_weighted_by_num_action",
    "distance_to_goal", "normalized_distance_to_goal",
    "num_action", "success_when_silent",
]


def load_eval_json(path: str):
    with open(path) as f:
        d = json.load(f)
    domain_order: List[int] = d["domain_order"]
    pm_raw: Dict = d["performance_matrix"]
    return domain_order, pm_raw


def load_base_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def build_matrix(domain_order: List[int], pm_raw: Dict, metric: str) -> np.ndarray:
    N = len(domain_order)
    mat = np.full((N, N), np.nan)
    for i, train_d in enumerate(domain_order):
        row = pm_raw.get(str(train_d), {})
        for j, eval_d in enumerate(domain_order):
            cell = row.get(str(eval_d), {})
            if isinstance(cell, dict) and metric in cell:
                mat[i, j] = cell[metric]
    return mat


def compute_ap(mat: np.ndarray) -> float:
    last_row = mat[-1, :]
    valid = last_row[~np.isnan(last_row)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def compute_forgetting(mat: np.ndarray, higher_better: bool) -> float:
    """Compute forgetting with metric direction awareness."""
    vals = []
    for j in range(mat.shape[0] - 1):
        col = mat[:-1, j]
        last = mat[-1, j]
        if np.isnan(last) or np.all(np.isnan(col)):
            continue
        if higher_better:
            vals.append(float(np.nanmax(col)) - last)
        else:
            vals.append(last - float(np.nanmin(col)))
    return float(np.mean(vals)) if vals else np.nan


def compute_forward_transfer(
    mat: np.ndarray,
    domain_order: List[int],
    base_perf: Dict,
    metric: str,
) -> float:
    """FT = mean_j( mat[j, j+1] - base[domain_order[j+1]] )"""
    vals = []
    for j in range(mat.shape[0] - 1):
        next_d = domain_order[j + 1]
        perf_before = mat[j, j + 1]
        base_val = base_perf.get(str(next_d), {}).get(metric, None)
        if np.isnan(perf_before) or base_val is None:
            continue
        vals.append(perf_before - base_val)
    return float(np.mean(vals)) if vals else np.nan


def compute_backward_transfer(mat: np.ndarray) -> float:
    """BT = mean_j( mat[-1,j] - mat[j,j] ) over previously learned tasks."""
    vals = []
    for j in range(mat.shape[0] - 1):
        diag = mat[j, j]
        last = mat[-1, j]
        if np.isnan(diag) or np.isnan(last):
            continue
        vals.append(last - diag)
    return float(np.mean(vals)) if vals else np.nan


def fmt_signed(v) -> str:
    if v is None or np.isnan(v):
        return "   N/A  "
    return f"{v:+.4f}"


def run(eval_json: str, base_json: str, output: Optional[str]):
    domain_order, pm_raw = load_eval_json(eval_json)
    base_perf = load_base_json(base_json)
    N = len(domain_order)

    lines = []
    lines.append("=" * 72)
    lines.append("Continual learning metric summary")
    lines.append(f"  eval : {eval_json}")
    lines.append(f"  base : {base_json}")
    lines.append(f"  domains: {N}   order: {domain_order}")
    lines.append("=" * 72)
    lines.append(f"{'Metric':<36} {'AP':>8} {'F':>9} {'FT':>9} {'BT':>9}")
    lines.append("-" * 72)

    for metric in REPORT_METRICS:
        mat = build_matrix(domain_order, pm_raw, metric)
        if np.all(np.isnan(mat)):
            continue
        hb = metric in HIGHER_BETTER
        ap = compute_ap(mat)
        f  = compute_forgetting(mat, hb)
        ft = compute_forward_transfer(mat, domain_order, base_perf, metric)
        bt = compute_backward_transfer(mat)

        alias = METRIC_ALIASES.get(metric, metric)
        ap_s = f"{ap:.4f}" if not np.isnan(ap) else "  N/A  "
        lines.append(
            f"{alias:<36} {ap_s:>8} {fmt_signed(f):>9} {fmt_signed(ft):>9} {fmt_signed(bt):>9}"
        )

    lines.append("=" * 72)
    text = "\n".join(lines) + "\n"
    print(text)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text)
        print(f"Saved summary to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute AP/F/FT/BT metrics from cl_eval_state.json."
    )
    parser.add_argument("--eval-json", required=True,
                        help="Path to cl_eval_state.json")
    parser.add_argument("--base-json", required=True,
                        help="Path to pretrained_performance_single/multi_source.json")
    parser.add_argument("--output", default=None,
                        help="Optional output path")
    args = parser.parse_args()
    run(args.eval_json, args.base_json, args.output)


if __name__ == "__main__":
    main()
