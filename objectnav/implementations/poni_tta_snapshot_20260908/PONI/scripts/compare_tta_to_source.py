#!/usr/bin/env python3
"""Compare one completed TTA val_part with the matching Source PONI result."""

import argparse
import json
from pathlib import Path
import re


METRICS = (
    ("success", True),
    ("spl", True),
    ("softspl", True),
    ("distance_to_goal", False),
    ("reward", True),
)


def safe_json(path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def latest_run(experiment_root, run_id=None):
    root = Path(experiment_root).resolve()
    if run_id:
        direct = root / "runs" / run_id
        if (direct / "run_manifest.json").is_file():
            return direct
        raise SystemExit("Run ID not found under {}: {}".format(root, run_id))
    candidates = list((root / "runs").glob("*/run_manifest.json"))
    candidates.extend(root.glob("*/runs/*/run_manifest.json"))
    if not candidates and (root / "run_manifest.json").is_file():
        return root
    if not candidates:
        raise SystemExit("No run_manifest.json found under {}".format(root))

    def key(path):
        payload = safe_json(path) or {}
        return float(payload.get("start_time_epoch", 0.0)), path.stat().st_mtime

    return max(candidates, key=key).parent


def runtime_root(run_root, manifest):
    stats_glob = list(run_root.glob("tb_seed_100_val_part_*/stats.json"))
    if stats_glob:
        return run_root
    poni_root = Path(manifest["poni_root"]).resolve()
    try:
        relative = run_root.relative_to(poni_root)
    except ValueError:
        return run_root
    legacy = poni_root / "hlab" / relative
    return legacy if legacy.is_dir() else run_root


def aggregate(path):
    payload = safe_json(path)
    if not isinstance(payload, dict) or not payload:
        raise SystemExit("Missing or empty stats: {}".format(path))
    rows = list(payload.values())
    means = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key, _ in METRICS
    }
    means["goal_distance"] = sum(
        float(row["goal_distance"]) for row in rows
    ) / len(rows)
    return len(rows), means


def step_count(log_path):
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    matches = re.findall(r"# steps: (\d+)", text)
    return int(matches[-1]) if matches else None


def format_metric(name, value):
    if name in ("success", "spl", "softspl"):
        return "{:.2f}%".format(value * 100.0)
    return "{:.4f}".format(value)


def format_delta(name, value):
    if name in ("success", "spl", "softspl"):
        return "{:+.2f}pp".format(value * 100.0)
    return "{:+.4f}".format(value)


def main():
    script_dir = Path(__file__).resolve().parent
    poni_root = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--source-root",
        default=str(poni_root / "experiments/mp3d_poni_seed_123"),
    )
    args = parser.parse_args()

    run_root = latest_run(args.experiment_root, args.run_id)
    manifest = safe_json(run_root / "run_manifest.json")
    if not isinstance(manifest, dict):
        raise SystemExit("Invalid run manifest: {}".format(run_root))
    output_root = runtime_root(run_root, manifest)
    part = args.part
    tta_stats = output_root / "tb_seed_100_val_part_{}".format(part) / "stats.json"
    source_root = Path(args.source_root).resolve()
    source_stats = source_root / "tb_seed_100_val_part_{}".format(part) / "stats.json"
    tta_count, tta = aggregate(tta_stats)
    source_count, source = aggregate(source_stats)
    if tta_count != source_count:
        raise SystemExit(
            "Episode count mismatch: TTA={} Source={}".format(
                tta_count, source_count
            )
        )
    if abs(tta["goal_distance"] - source["goal_distance"]) > 1e-6:
        raise SystemExit("Episode stream mismatch: mean goal_distance differs")

    tta_steps = step_count(
        output_root / "logs_seed_100_val_part_{}.txt".format(part)
    )
    source_steps = step_count(
        source_root / "logs_seed_100_val_part_{}.txt".format(part)
    )
    print("Method: {}".format(manifest.get("method", "unknown")))
    print("Run: {}".format(manifest.get("run_id", run_root.name)))
    print("Part: val_part_{} ({} episodes)".format(part, tta_count))
    print("Run directory: {}".format(output_root))
    print()
    print("Metric                 Source          TTA        Delta    Better?")
    print("--------------------  -----------  -----------  -----------  -------")
    for name, higher_is_better in METRICS:
        delta = tta[name] - source[name]
        improved = delta > 0 if higher_is_better else delta < 0
        print(
            "{:<20}  {:>11}  {:>11}  {:>11}  {}".format(
                name,
                format_metric(name, source[name]),
                format_metric(name, tta[name]),
                format_delta(name, delta),
                "yes" if improved else "no",
            )
        )
    if tta_steps is not None and source_steps is not None:
        source_average = source_steps / source_count
        tta_average = tta_steps / tta_count
        print(
            "{:<20}  {:>11.1f}  {:>11.1f}  {:>+11.1f}  {}".format(
                "steps/episode",
                source_average,
                tta_average,
                tta_average - source_average,
                "yes" if tta_average < source_average else "no",
            )
        )


if __name__ == "__main__":
    main()
