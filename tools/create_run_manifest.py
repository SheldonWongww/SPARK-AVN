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
    parser.add_argument("--source-setting", default="")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoint = os.path.abspath(args.checkpoint)
    manifest = {
        "run_id": args.run_id,
        "task": args.task,
        "benchmark": args.benchmark,
        "model": args.model,
        "method": args.method,
        "source_setting": args.source_setting,
        "seed": args.seed,
        "git_commit": git_commit(repo_root),
        "config": args.config,
        "config_overrides": args.extra,
        "checkpoint": {"path": checkpoint, "sha256": sha256(checkpoint)},
        "hardware": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
