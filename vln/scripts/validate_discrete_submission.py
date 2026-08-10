#!/usr/bin/env python3
"""Validate R2R/REVERIE MatterSim predictions against an order manifest."""

import argparse
import hashlib
import json
import math
import os
import sys


def fail(message):
    raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True, choices=("r2r", "reverie"))
    parser.add_argument("--expected-benchmark", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with open(args.submission, "r", encoding="utf-8") as stream:
        predictions = json.load(stream)

    if manifest.get("schema") != "navtta.episode_order.v1":
        fail("unsupported episode-order manifest")
    if manifest.get("split") != "test":
        fail("submission validation requires a test manifest")
    if manifest.get("benchmark") != args.expected_benchmark:
        fail(
            "manifest benchmark mismatch: expected {!r}, got {!r}".format(
                args.expected_benchmark, manifest.get("benchmark")
            )
        )
    dataset_sha256 = sha256_file(args.dataset)
    expected_dataset_sha256 = manifest.get("dataset", {}).get("sha256")
    if dataset_sha256 != expected_dataset_sha256:
        fail(
            "dataset SHA256 mismatch: expected {}, got {}".format(
                expected_dataset_sha256, dataset_sha256
            )
        )
    if not isinstance(predictions, list):
        fail("submission must be a JSON list")
    if any(not isinstance(record, dict) for record in predictions):
        fail("every prediction must be a JSON object")

    expected_ids = [str(record["episode_id"]) for record in manifest["episodes"]]
    actual_ids = [str(record.get("instr_id")) for record in predictions]
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        fail(
            "episode order/identity mismatch (missing={}, extra={})".format(
                missing[:10], extra[:10]
            )
        )

    step_count = 0
    for record in predictions:
        episode_id = str(record["instr_id"])
        trajectory = record.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            fail("episode {} has an empty or invalid trajectory".format(episode_id))
        for index, step in enumerate(trajectory):
            if not isinstance(step, (list, tuple)) or len(step) != 3:
                fail("episode {} step {} is malformed".format(episode_id, index))
            if not isinstance(step[0], str) or not step[0]:
                fail("episode {} step {} has no viewpoint ID".format(episode_id, index))
            for angle in step[1:3]:
                if (
                    isinstance(angle, bool)
                    or not isinstance(angle, (int, float))
                    or not math.isfinite(angle)
                ):
                    fail("episode {} step {} has an invalid angle".format(episode_id, index))
        if args.task == "reverie":
            if "pred_objid" in record or "predObjId" not in record:
                fail(
                    "REVERIE episode {} must use canonical predObjId".format(
                        episode_id
                    )
                )
            object_id = record["predObjId"]
            if object_id is not None and (
                isinstance(object_id, bool) or not isinstance(object_id, int)
            ):
                fail(
                    "REVERIE episode {} predObjId must be an integer or null".format(
                        episode_id
                    )
                )
        step_count += len(trajectory)

    print(
        "valid {} submission: {} episodes, {} steps ({})".format(
            args.task.upper(),
            len(predictions),
            step_count,
            os.path.abspath(args.submission),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("invalid discrete submission: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
