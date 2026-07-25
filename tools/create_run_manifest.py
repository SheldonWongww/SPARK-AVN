#!/usr/bin/env python3
"""Create the immutable pre-run portion of a NavTTA experiment manifest."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True, choices=("avn", "vln", "objectnav"))
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--source-setting", default="")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aux-checkpoint", action="append", default=[])
    parser.add_argument("--dataset", default="")
    parser.add_argument("--dataset-version", default="")
    parser.add_argument("--stream-order-sha256", default="")
    parser.add_argument("--stream-content-sha256", default="")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoint = os.path.abspath(args.checkpoint)
    auxiliary_checkpoints = []
    for item in args.aux_checkpoint:
        if "=" not in item:
            parser.error("--aux-checkpoint must use NAME=PATH")
        name, path = item.split("=", 1)
        path = os.path.abspath(path)
        if not name or not os.path.isfile(path):
            parser.error("invalid auxiliary checkpoint: {}".format(item))
        auxiliary_checkpoints.append(
            {"name": name, "path": path, "sha256": sha256(path)}
        )

    dataset = None
    if args.dataset:
        dataset_path = os.path.abspath(args.dataset)
        if not os.path.isfile(dataset_path):
            parser.error("dataset index does not exist: {}".format(dataset_path))
        dataset = {
            "path": dataset_path,
            "version": args.dataset_version,
            "index_sha256": sha256(dataset_path),
            "stream_order_sha256": args.stream_order_sha256,
            "stream_content_sha256": args.stream_content_sha256,
        }

    manifest = {
        "run_id": args.run_id,
        "task": args.task,
        "benchmark": args.benchmark,
        "model": args.model,
        "method": args.method,
        "run_tag": args.run_tag,
        "source_setting": args.source_setting,
        "seed": args.seed,
        "git_commit": git_commit(repo_root),
        "config": args.config,
        "config_overrides": args.extra,
        "checkpoint": {"path": checkpoint, "sha256": sha256(checkpoint)},
        "auxiliary_checkpoints": auxiliary_checkpoints,
        "dataset": dataset,
        "hardware": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "status": "running",
        "exit_code": None,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
