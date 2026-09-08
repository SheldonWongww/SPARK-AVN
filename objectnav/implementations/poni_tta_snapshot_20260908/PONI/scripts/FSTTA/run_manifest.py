#!/usr/bin/env python3
"""Create and finalize the machine-readable FSTTA run manifest."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_completed(save_root, part):
    path = save_root / "tb_seed_100_val_part_{}".format(part) / "stats.json"
    if not path.is_file():
        return 0
    try:
        with path.open() as handle:
            payload = json.load(handle)
        return len(payload) if isinstance(payload, dict) else 0
    except (OSError, ValueError):
        return 0


def parse_tta(values):
    output = {}
    for value in values:
        if "=" not in value:
            raise ValueError("TTA value must use key=value syntax: {}".format(value))
        key, raw = value.split("=", 1)
        output[key] = raw
    return output


def start(args):
    path = Path(args.path).resolve()
    save_root = Path(args.save_root).resolve()
    parts = [int(value) for value in args.parts.split()]
    baseline = {str(part): read_completed(save_root, part) for part in parts}
    payload = {
        "schema_version": 1,
        "method": "fstta",
        "status": "running",
        "started_at": now_iso(),
        "start_time_epoch": time.time(),
        "launcher_pid": int(args.launcher_pid),
        "run_id": args.run_id,
        "poni_root": str(Path(args.poni_root).resolve()),
        "navtta_root": str(Path(args.navtta_root).resolve()),
        "save_root": str(save_root),
        "model_path": str(Path(args.model_path).resolve()),
        "gpu_id": str(args.gpu_id),
        "max_jobs": int(args.max_jobs),
        "parts": parts,
        "test_episode_count": int(args.test_episode_count),
        "baseline_completed_episodes": baseline,
        "tta": parse_tta(args.tta),
    }
    atomic_write(path, payload)


def finish(args):
    path = Path(args.path).resolve()
    try:
        with path.open() as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        payload = {"schema_version": 1, "method": "fstta"}
    ended = time.time()
    payload.update(
        {
            "status": args.status,
            "ended_at": now_iso(),
            "end_time_epoch": ended,
            "elapsed_seconds": max(
                0.0, ended - float(payload.get("start_time_epoch", ended))
            ),
        }
    )
    atomic_write(path, payload)


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--path", required=True)
    start_parser.add_argument("--save-root", required=True)
    start_parser.add_argument("--poni-root", required=True)
    start_parser.add_argument("--navtta-root", required=True)
    start_parser.add_argument("--model-path", required=True)
    start_parser.add_argument("--run-id", required=True)
    start_parser.add_argument("--launcher-pid", required=True)
    start_parser.add_argument("--gpu-id", required=True)
    start_parser.add_argument("--max-jobs", required=True)
    start_parser.add_argument("--parts", required=True)
    start_parser.add_argument("--test-episode-count", required=True)
    start_parser.add_argument("--tta", action="append", default=[])
    start_parser.set_defaults(handler=start)
    finish_parser = subparsers.add_parser("finish")
    finish_parser.add_argument("--path", required=True)
    finish_parser.add_argument(
        "--status", choices=("complete", "failed", "interrupted"), required=True
    )
    finish_parser.set_defaults(handler=finish)
    return parser


def main():
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
