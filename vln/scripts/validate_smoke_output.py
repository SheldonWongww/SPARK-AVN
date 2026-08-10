#!/usr/bin/env python3
"""Validate non-formal single/prefix-episode VLN GPU smoke outputs."""

import argparse
import glob
import json
import math
import os
import re
import sys


DISCRETE_SETTINGS = {
    "duet-r2r",
    "duet-reverie",
    "hamt-r2r",
    "hamt-reverie",
    "goat-r2r",
    "goat-reverie",
}
CE_SETTINGS = {"etpnav-r2r-ce", "bevbert-r2r-ce"}


def fail(message):
    raise ValueError(message)


def reject_non_finite(value, label):
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            fail("{} contains a non-finite number".format(label))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            reject_non_finite(item, "{}.{}".format(label, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_non_finite(item, "{}[{}]".format(label, index))


def expected_ids(manifest_path, count):
    with open(manifest_path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("split") != "val_seen":
        fail("GPU smoke requires a val_seen order manifest")
    records = manifest.get("episodes")
    if not isinstance(records, list) or count <= 0 or count > len(records):
        fail("invalid smoke episode count")
    return [str(item["episode_id"]) for item in records[:count]]


def one_match(pattern, label):
    matches = sorted(glob.glob(pattern, recursive=True))
    if len(matches) != 1:
        fail("expected one {}, found {}: {}".format(label, len(matches), matches))
    return matches[0]


def validate_discrete(setting, result_root, ids):
    prediction_path = one_match(
        os.path.join(result_root, "**", "submit_val_seen*.json"),
        "discrete smoke prediction file",
    )
    with open(prediction_path, "r", encoding="utf-8") as stream:
        predictions = json.load(stream)
    if not isinstance(predictions, list):
        fail("discrete smoke predictions must be a list")
    actual_ids = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            fail("prediction {} is not an object".format(index))
        identifier = prediction.get(
            "instr_id", prediction.get("episode_id", prediction.get("instruction_id"))
        )
        actual_ids.append(str(identifier))
        trajectory = prediction.get("trajectory", prediction.get("path"))
        if not isinstance(trajectory, list) or not trajectory:
            fail("prediction {} has no trajectory".format(index))
        if setting.endswith("reverie") and "predObjId" not in prediction:
            fail("REVERIE smoke prediction lacks predObjId")
        reject_non_finite(prediction, "prediction[{}]".format(index))
    if actual_ids != ids:
        fail("discrete smoke IDs mismatch: expected {}, got {}".format(ids, actual_ids))

    metric_path = one_match(
        os.path.join(result_root, "**", "valid.txt"),
        "discrete validation metric file",
    )
    with open(metric_path, "r", encoding="utf-8") as stream:
        metric_lines = [line for line in stream if "Env name: val_seen" in line]
    if len(metric_lines) != 1:
        fail("expected one val_seen metric summary")
    if re.search(r"(?i)(?:^|[^a-z])(nan|[+-]?inf)(?:$|[^a-z])", metric_lines[0]):
        fail("discrete metric summary contains a non-finite value")
    values = re.findall(
        r":\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        metric_lines[0],
    )
    if not values or any(not math.isfinite(float(value)) for value in values):
        fail("discrete metric summary has no finite values")


def validate_ce(result_root, ids):
    stats_path = one_match(
        os.path.join(result_root, "**", "stats_ep_ckpt_*_val_seen_r0_w1.json"),
        "continuous per-episode stats file",
    )
    with open(stats_path, "r", encoding="utf-8") as stream:
        stats = json.load(stream)
    if not isinstance(stats, dict) or list(map(str, stats.keys())) != ids:
        fail("continuous smoke IDs mismatch: expected {}, got {}".format(ids, list(stats)))
    for identifier, metrics in stats.items():
        if not isinstance(metrics, dict) or not metrics:
            fail("continuous smoke episode {} has no metrics".format(identifier))
    reject_non_finite(stats, "continuous_stats")


def validate_stream(result_root, ids):
    result_path = os.path.join(result_root, "result.json")
    if not os.path.isfile(result_path):
        fail("StreamVLN smoke result is missing")
    documents = []
    with open(result_path, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                documents.append(json.loads(line))
    episodes = [item for item in documents if "episode_id" in item]
    actual_ids = [str(item["episode_id"]) for item in episodes]
    if actual_ids != ids:
        fail("StreamVLN smoke IDs mismatch: expected {}, got {}".format(ids, actual_ids))
    aggregates = [item for item in documents if "length" in item]
    if len(aggregates) != 1 or int(aggregates[0]["length"]) != len(ids):
        fail("StreamVLN aggregate episode count is inconsistent")
    reject_non_finite(documents, "streamvln_results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    args = parser.parse_args()

    ids = expected_ids(args.manifest, args.expected_count)
    if args.setting in DISCRETE_SETTINGS:
        validate_discrete(args.setting, args.result_root, ids)
    elif args.setting in CE_SETTINGS:
        validate_ce(args.result_root, ids)
    elif args.setting == "streamvln-r2r-ce":
        validate_stream(args.result_root, ids)
    else:
        fail("unsupported smoke setting: {}".format(args.setting))
    print(
        "GPU smoke output passed: {} episodes={} first_id={}".format(
            args.setting, len(ids), ids[0]
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("smoke validation failed: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
