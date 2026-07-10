#!/usr/bin/env python3
"""Compute AP/F/FT/BT metrics from CSV performance matrices."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


AP_METRICS = {
    "success",
    "spl",
    "distance_to_goal",
    "normalized_distance_to_goal",
    "num_action",
    "success_weighted_by_num_action",
}

HIGHER_BETTER_FORGETTING = {
    "success": "F_sr",
    "spl": "F_spl",
    "success_weighted_by_num_action": "F_sna",
}

LOWER_BETTER_FORGETTING = {
    "distance_to_goal": "F_dtg",
    "num_action": "F_na",
}

TRANSFER_METRICS = {
    "success": ("FT_sr", "BT_sr"),
    "spl": ("FT_spl", "BT_spl"),
    "success_weighted_by_num_action": ("FT_sna", "BT_sna"),
}

BASE_FILE_MAPPING = {
    "success_weighted_by_num_action": "sna",
    "num_action": "num_action",
}


def _valid_float(value) -> Optional[float]:
    if pd.isna(value) or value == "-":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_values(values) -> List[float]:
    return [v for v in (_valid_float(value) for value in values) if v is not None]


def load_performance_matrix(csv_file: Path) -> pd.DataFrame:
    return pd.read_csv(csv_file, index_col=0)


def load_base_performance(base_csv: Path) -> Dict[str, float]:
    if not base_csv.exists():
        print(f"  Warning: base file not found: {base_csv}")
        return {}

    df = pd.read_csv(base_csv)
    if "Domain" not in df.columns or "Base_Performance" not in df.columns:
        print(f"  Warning: invalid base file format: {base_csv}")
        return {}

    base_perf = {}
    for _, row in df.iterrows():
        value = _valid_float(row["Base_Performance"])
        if value is not None:
            base_perf[str(row["Domain"])] = value
    return base_perf


def compute_average_performance(df: pd.DataFrame, metric_name: str) -> Optional[float]:
    if metric_name not in AP_METRICS:
        return None
    values = _numeric_values(df.iloc[-1])
    return float(np.mean(values)) if values else None


def compute_forgetting(df: pd.DataFrame, higher_better: bool) -> Optional[float]:
    last_row = df.iloc[-1]
    values = []

    for col in df.columns[:-1]:
        column_values = _numeric_values(df.iloc[:-1][col])
        last_value = _valid_float(last_row[col])
        if not column_values or last_value is None:
            continue
        reference = max(column_values) if higher_better else min(column_values)
        values.append(reference - last_value if higher_better else last_value - reference)

    return float(np.mean(values)) if values else None


def compute_forward_transfer(df: pd.DataFrame, base_perf: Dict[str, float]) -> Optional[float]:
    if not base_perf:
        return None

    domains = list(df.columns)
    values = []
    for idx in range(len(domains) - 1):
        train_domain = domains[idx]
        next_domain = domains[idx + 1]
        if train_domain not in df.index or next_domain not in df.columns:
            continue

        perf_before = _valid_float(df.loc[train_domain, next_domain])
        base_value = base_perf.get(next_domain)
        if perf_before is not None and base_value is not None:
            values.append(perf_before - base_value)

    return float(np.mean(values)) if values else None


def compute_backward_transfer(df: pd.DataFrame) -> Optional[float]:
    last_row = df.iloc[-1]
    values = []

    for domain in df.columns[:-1]:
        if domain not in df.index:
            continue
        diagonal_value = _valid_float(df.loc[domain, domain])
        final_value = _valid_float(last_row[domain])
        if diagonal_value is not None and final_value is not None:
            values.append(final_value - diagonal_value)

    return float(np.mean(values)) if values else None


def process_metric_files(results_dir: Path, base_dir: Path) -> Dict[str, Dict[str, float]]:
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        return {}
    if not base_dir.exists():
        print(f"Error: base directory not found: {base_dir}")
        return {}

    metric_files = sorted(results_dir.glob("*.csv"))
    if not metric_files:
        print(f"Error: no CSV files found in {results_dir}")
        return {}

    results = {}
    for metric_file in metric_files:
        metric_name = metric_file.stem
        df = load_performance_matrix(metric_file)
        metric_results = {}

        ap = compute_average_performance(df, metric_name)
        if ap is not None:
            metric_results["average_performance"] = ap

        if metric_name in HIGHER_BETTER_FORGETTING:
            value = compute_forgetting(df, higher_better=True)
            if value is not None:
                metric_results[HIGHER_BETTER_FORGETTING[metric_name]] = value

        if metric_name in LOWER_BETTER_FORGETTING:
            value = compute_forgetting(df, higher_better=False)
            if value is not None:
                metric_results[LOWER_BETTER_FORGETTING[metric_name]] = value

        if metric_name in TRANSFER_METRICS:
            base_name = BASE_FILE_MAPPING.get(metric_name, metric_name)
            base_perf = load_base_performance(base_dir / f"{base_name}.csv")
            ft_name, bt_name = TRANSFER_METRICS[metric_name]

            ft = compute_forward_transfer(df, base_perf)
            if ft is not None:
                metric_results[ft_name] = ft

            bt = compute_backward_transfer(df)
            if bt is not None:
                metric_results[bt_name] = bt

        if metric_results:
            results[metric_name] = metric_results
            print(f"Processed {metric_name}: {metric_results}")

    return results


def save_results(
    results: Dict[str, Dict[str, float]],
    output_file: Path,
    results_dir: Path,
    base_dir: Path,
) -> None:
    lines = [
        "=" * 80,
        "Continual learning metric summary",
        "=" * 80,
        f"Results directory: {results_dir}",
        f"Base directory: {base_dir}",
        "",
    ]

    for section_name, prefix in [
        ("Average Performance", "average_performance"),
        ("Forgetting", "F_"),
        ("Forward Transfer", "FT_"),
        ("Backward Transfer", "BT_"),
    ]:
        lines.append(section_name)
        lines.append("-" * 40)
        for metric_name in sorted(results):
            for key, value in sorted(results[metric_name].items()):
                if key == prefix or key.startswith(prefix):
                    lines.append(f"{metric_name:35s} {key:30s} {value:8.4f}")
        lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved results to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute AP/F/FT/BT metrics from CSV performance matrices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/compute_cl_metrics.py \\
      --results-dir data/results/performance_matrix/spark_avn_seed_1 \\
      --base-dir data/results/base_results_single \\
      --output summary_single.txt
""",
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("cl_metrics_summary.txt"))
    args = parser.parse_args()

    results = process_metric_files(args.results_dir, args.base_dir)
    if not results:
        print("Error: no valid metrics were computed.")
        return

    save_results(results, args.output, args.results_dir, args.base_dir)


if __name__ == "__main__":
    main()
