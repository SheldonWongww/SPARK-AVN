#!/usr/bin/env python3
"""Validate the immutable identity and successful completion of a run."""

import argparse
import json
import os
import sys


def nested_value(document, path):
    value = document
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source-setting", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--stream-order-sha256", required=True)
    parser.add_argument("--stream-content-sha256", required=True)
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        print("invalid run manifest '{}': {}".format(manifest_path, error), file=sys.stderr)
        return 1

    expected = {
        "task": "avn",
        "benchmark": "mp3d",
        "run_tag": args.run_tag,
        "model": args.model,
        "method": args.method,
        "source_setting": args.source_setting,
        "seed": args.seed,
        "git_commit": args.git_commit,
        "status": "completed",
        "exit_code": 0,
        "checkpoint.sha256": args.checkpoint_sha256,
        "dataset.stream_order_sha256": args.stream_order_sha256,
        "dataset.stream_content_sha256": args.stream_content_sha256,
    }

    mismatches = []
    for path, expected_value in expected.items():
        actual_value = nested_value(manifest, path)
        if actual_value != expected_value:
            mismatches.append(
                "{} expected={!r} actual={!r}".format(
                    path, expected_value, actual_value
                )
            )

    if mismatches:
        print(
            "run manifest identity mismatch '{}': {}".format(
                manifest_path, "; ".join(mismatches)
            ),
            file=sys.stderr,
        )
        return 1

    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
