#!/usr/bin/env python3
"""Convert graph-agent trajectories to the official R2R/REVERIE schema."""

import argparse
import json
import math
import os
import sys
import tempfile


ANGLE_INCREMENT = math.radians(30.0)


def fail(message):
    raise ValueError(message)


def is_finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def load_episode_metadata(path, task):
    with open(path, "r", encoding="utf-8") as stream:
        dataset = json.load(stream)
    if not isinstance(dataset, list):
        fail("encoded dataset must be a JSON list")

    episodes = {}
    for item_index, item in enumerate(dataset):
        if not isinstance(item, dict):
            fail("dataset item {} is not an object".format(item_index))
        instructions = item.get("instructions")
        path_points = item.get("path")
        scan = item.get("scan")
        heading = item.get("heading")
        if not isinstance(instructions, list) or not instructions:
            fail("dataset item {} has no instructions".format(item_index))
        if (
            not isinstance(path_points, list)
            or not path_points
            or not isinstance(path_points[0], str)
            or not path_points[0]
        ):
            fail("dataset item {} has no start viewpoint".format(item_index))
        if not isinstance(scan, str) or not scan:
            fail("dataset item {} has no scan".format(item_index))
        if not is_finite_number(heading):
            fail("dataset item {} has an invalid start heading".format(item_index))

        if task == "r2r":
            if "path_id" not in item:
                fail("R2R dataset item {} has no path_id".format(item_index))
            identifier_prefix = str(item["path_id"])
        else:
            if item.get("objId") is not None:
                if "path_id" not in item:
                    fail(
                        "REVERIE dataset item {} has objId but no path_id".format(
                            item_index
                        )
                    )
                identifier_prefix = "{}_{}".format(
                    item["path_id"], item["objId"]
                )
            elif "id" in item:
                identifier_prefix = str(item["id"])
            elif "path_id" in item:
                identifier_prefix = str(item["path_id"])
            else:
                fail("REVERIE dataset item {} has no id".format(item_index))

        for instruction_index in range(len(instructions)):
            instruction_id = "{}_{}".format(
                identifier_prefix, instruction_index
            )
            if instruction_id in episodes:
                fail("duplicate dataset instruction ID {}".format(instruction_id))
            episodes[instruction_id] = {
                "scan": scan,
                "start_viewpoint": path_points[0],
                "start_heading": float(heading),
            }
    return episodes


def load_candidate_map(path):
    with open(path, "r", encoding="utf-8") as stream:
        candidates = json.load(stream)
    if not isinstance(candidates, dict):
        fail("viewpoint candidate map must be a JSON object")
    return candidates


def candidate_view_index(candidates, scan, source, target):
    source_key = "{}_{}".format(scan, source)
    try:
        candidate = candidates[source_key][target]
    except (KeyError, TypeError):
        fail(
            "no navigable edge for scan {} from {} to {}".format(
                scan, source, target
            )
        )
    if isinstance(candidate, list):
        if not candidate:
            fail("empty candidate metadata for {} to {}".format(source, target))
        candidate = candidate[0]
    if (
        isinstance(candidate, bool)
        or not isinstance(candidate, (int, float))
        or int(candidate) != candidate
    ):
        fail("invalid view index for {} to {}".format(source, target))
    view_index = int(candidate)
    if view_index < 0 or view_index >= 36:
        fail("view index out of range for {} to {}".format(source, target))
    return view_index


def is_canonical_step(step):
    return (
        isinstance(step, (list, tuple))
        and len(step) == 3
        and isinstance(step[0], str)
        and bool(step[0])
        and is_finite_number(step[1])
        and is_finite_number(step[2])
    )


def validate_start(points, metadata, instruction_id):
    if points[0][0] != metadata["start_viewpoint"]:
        fail(
            "episode {} starts at {}, expected {}".format(
                instruction_id,
                points[0][0],
                metadata["start_viewpoint"],
            )
        )


def validate_edges(points, metadata, candidates):
    scan = metadata["scan"]
    for previous, current in zip(points, points[1:]):
        if previous[0] != current[0]:
            candidate_view_index(candidates, scan, previous[0], current[0])


def canonicalize_trajectory(
    trajectory, metadata, candidates, instruction_id
):
    if not isinstance(trajectory, list) or not trajectory:
        fail("episode {} has an empty or invalid trajectory".format(instruction_id))

    canonical_flags = [is_canonical_step(step) for step in trajectory]
    if all(canonical_flags):
        canonical = [
            [step[0], float(step[1]), float(step[2])] for step in trajectory
        ]
        validate_start(canonical, metadata, instruction_id)
        validate_edges(canonical, metadata, candidates)
        return canonical
    if any(canonical_flags):
        fail("episode {} mixes canonical steps and graph paths".format(instruction_id))

    viewpoints = []
    for segment_index, segment in enumerate(trajectory):
        if (
            not isinstance(segment, (list, tuple))
            or not segment
            or any(
                not isinstance(viewpoint, str) or not viewpoint
                for viewpoint in segment
            )
        ):
            fail(
                "episode {} graph segment {} is malformed".format(
                    instruction_id, segment_index
                )
            )
        viewpoints.extend(segment)

    if viewpoints[0] != metadata["start_viewpoint"]:
        fail(
            "episode {} starts at {}, expected {}".format(
                instruction_id,
                viewpoints[0],
                metadata["start_viewpoint"],
            )
        )

    canonical = [
        [viewpoints[0], float(metadata["start_heading"]), 0.0]
    ]
    scan = metadata["scan"]
    for viewpoint in viewpoints[1:]:
        previous = canonical[-1]
        if viewpoint == previous[0]:
            heading, elevation = previous[1], previous[2]
        else:
            view_index = candidate_view_index(
                candidates, scan, previous[0], viewpoint
            )
            heading = (view_index % 12) * ANGLE_INCREMENT
            elevation = (view_index // 12 - 1) * ANGLE_INCREMENT
        canonical.append([viewpoint, heading, elevation])
    return canonical


def canonicalize_submission(predictions, episodes, candidates):
    if not isinstance(predictions, list):
        fail("submission must be a JSON list")
    seen = set()
    output = []
    for record_index, record in enumerate(predictions):
        if not isinstance(record, dict):
            fail("prediction {} is not an object".format(record_index))
        instruction_id = str(record.get("instr_id"))
        if instruction_id in seen:
            fail("duplicate prediction instruction ID {}".format(instruction_id))
        if instruction_id not in episodes:
            fail("unknown prediction instruction ID {}".format(instruction_id))
        seen.add(instruction_id)
        converted = dict(record)
        converted["instr_id"] = instruction_id
        converted["trajectory"] = canonicalize_trajectory(
            record.get("trajectory"),
            episodes[instruction_id],
            candidates,
            instruction_id,
        )
        output.append(converted)
    return output


def write_json_atomic(path, payload):
    output = os.path.abspath(path)
    output_directory = os.path.dirname(output)
    os.makedirs(output_directory, exist_ok=True)
    try:
        output_mode = os.stat(output).st_mode & 0o777
    except FileNotFoundError:
        output_mode = 0o644
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".{}-".format(os.path.basename(output)),
        suffix=".tmp",
        dir=output_directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.chmod(temporary_path, output_mode)
        os.replace(temporary_path, output)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scanvp-candidates", required=True)
    parser.add_argument("--task", required=True, choices=("r2r", "reverie"))
    args = parser.parse_args()

    episodes = load_episode_metadata(args.dataset, args.task)
    candidates = load_candidate_map(args.scanvp_candidates)
    with open(args.submission, "r", encoding="utf-8") as stream:
        predictions = json.load(stream)
    converted = canonicalize_submission(predictions, episodes, candidates)
    point_count = sum(len(record["trajectory"]) for record in converted)
    write_json_atomic(args.output, converted)
    print(
        "canonicalized {} submission: {} episodes, {} points ({})".format(
            args.task.upper(),
            len(converted),
            point_count,
            os.path.abspath(args.output),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            "invalid discrete submission normalization: {}".format(error),
            file=sys.stderr,
        )
        raise SystemExit(1)
