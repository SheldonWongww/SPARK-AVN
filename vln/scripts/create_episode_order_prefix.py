#!/usr/bin/env python3
"""Create one immutable, self-validating canonical episode-order prefix."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core"))

from navtta_core.experiment.episode_order import (  # noqa: E402
    load_episode_order_manifest,
    prefix_episode_order_manifest,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", required=True, type=int)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()

    parent_path = Path(args.parent).resolve()
    output_path = Path(args.output).resolve()
    parent = load_episode_order_manifest(str(parent_path))
    prefix = prefix_episode_order_manifest(parent, args.episodes)
    prefix["canonical_parent"] = {
        "path": str(parent_path),
        "sha256": _sha256(parent_path),
        "episode_count": parent["episode_count"],
        "order_sha256": parent["order_sha256"],
    }
    prefix["audit_prefix"] = {
        "protocol": args.protocol,
        "canonical_prefix": True,
        "episode_count": args.episodes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if _canonical(existing) != _canonical(prefix):
            parser.error("immutable prefix manifest already exists with new content")
        return
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(prefix, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    os.replace(str(temporary), str(output_path))


if __name__ == "__main__":
    main()
