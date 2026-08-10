#!/usr/bin/env python3
"""Validate an R2R-CE trajectory file against a canonical split manifest."""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys


MAX_FORWARD_STEP_METERS = 0.25
POSITION_TOLERANCE_METERS = 1e-4


def fail(message):
    raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_start_positions(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(episodes, list):
        fail("dataset must contain an episodes list")
    starts = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            fail("dataset contains a non-object episode")
        episode_id = str(episode.get("episode_id"))
        position = episode.get("start_position")
        if episode_id in starts or not isinstance(position, list) or len(position) != 3:
            fail("dataset contains invalid or duplicate episode starts")
        starts[episode_id] = position
    return starts


def validate_state(state, episode_id, index):
    if set(state) != {"position", "heading", "stop"}:
        fail(
            "episode {} state {} must contain position, heading, and stop".format(
                episode_id, index
            )
        )
    position = state["position"]
    if not isinstance(position, list) or len(position) != 3:
        fail("episode {} state {} has an invalid position".format(episode_id, index))
    values = position + [state["heading"]]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        fail("episode {} state {} contains a non-finite number".format(episode_id, index))
    if not isinstance(state["stop"], bool):
        fail("episode {} state {} has a non-boolean stop".format(episode_id, index))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-benchmark", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    with open(args.submission, "r", encoding="utf-8") as stream:
        submission = json.load(stream)

    if manifest.get("schema") != "navtta.episode_order.v1":
        fail("unsupported episode-order manifest")
    if manifest.get("split") != "test":
        fail("R2R-CE submission validation requires a test manifest")
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
    start_positions = load_start_positions(args.dataset)
    if not isinstance(submission, dict):
        fail("submission must be a JSON object keyed by episode ID")

    expected_ids = [str(record["episode_id"]) for record in manifest["episodes"]]
    actual_ids = list(submission)
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        fail(
            "episode order/identity mismatch (missing={}, extra={})".format(
                missing[:10], extra[:10]
            )
        )
    if set(start_positions) != set(expected_ids):
        fail("dataset episode IDs do not match the manifest")

    state_count = 0
    for episode_id in expected_ids:
        path = submission[episode_id]
        if not isinstance(path, list) or not path:
            fail("episode {} has an empty or invalid path".format(episode_id))
        if len(path) > 500:
            fail("episode {} contains more than 500 states".format(episode_id))
        for index, state in enumerate(path):
            if not isinstance(state, dict):
                fail("episode {} state {} is not an object".format(episode_id, index))
            validate_state(state, episode_id, index)
            if index < len(path) - 1 and state["stop"]:
                fail(
                    "episode {} has a non-final state marked stopped".format(
                        episode_id
                    )
                )
            if index:
                previous_position = path[index - 1]["position"]
                distance = math.sqrt(
                    sum(
                        (current - previous) ** 2
                        for current, previous in zip(
                            state["position"], previous_position
                        )
                    )
                )
                if distance > (
                    MAX_FORWARD_STEP_METERS + POSITION_TOLERANCE_METERS
                ):
                    fail(
                        "episode {} moves {:.6f}m between states {} and {}; "
                        "maximum is {:.2f}m".format(
                            episode_id,
                            distance,
                            index - 1,
                            index,
                            MAX_FORWARD_STEP_METERS,
                        )
                    )
        if not path[-1]["stop"]:
            fail("episode {} final state is not marked stopped".format(episode_id))
        if any(
            not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-4)
            for actual, expected in zip(
                path[0]["position"], start_positions[episode_id]
            )
        ):
            fail("episode {} does not begin at its annotated start".format(episode_id))
        state_count += len(path)

    print(
        "valid R2R-CE submission: {} episodes, {} states ({})".format(
            len(expected_ids), state_count, os.path.abspath(args.submission)
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print("invalid R2R-CE submission: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
