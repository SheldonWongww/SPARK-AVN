#!/usr/bin/env python3
"""Build a canonical VLN episode-order manifest from an annotation file.

Run this from an environment where ``navtta-core`` is installed editable, as
required by the workspace setup.  The input may be JSON or gzip-compressed
JSON.  Only annotation metadata is read; scene assets and features are not.
"""

import argparse
import gzip
import json
import os
from typing import Any, Dict, Iterable, List

from navtta_core.experiment import build_episode_order_manifest, sha256_file


def _read_json(path: str) -> Any:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _root_episodes(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        payload = payload.get("episodes")
    if not isinstance(payload, list):
        raise ValueError("Annotation must be a list or contain an episodes list")
    return payload


def _expanded_records(
    payload: Any, annotation_format: str
) -> Iterable[Dict[str, str]]:
    items = _root_episodes(payload)
    if annotation_format in ("habitat", "expanded"):
        return items

    records = []
    for item in items:
        instructions = item.get("instructions")
        if not isinstance(instructions, list):
            raise ValueError(
                "{} input requires an instructions list".format(annotation_format)
            )
        scene_id = item.get("scan", item.get("scene_id"))
        if scene_id is None:
            raise ValueError("Annotation item is missing scan/scene_id")
        for instruction_index in range(len(instructions)):
            if annotation_format == "r2r":
                base_id = item["path_id"]
                identifier = "{}_{}".format(base_id, instruction_index)
            elif "objId" in item:
                identifier = "{}_{}_{}".format(
                    item["path_id"], item["objId"], instruction_index
                )
            else:
                base_id = item.get("id", item.get("path_id"))
                if base_id is None:
                    raise ValueError("REVERIE item is missing id/path_id")
                identifier = "{}_{}".format(base_id, instruction_index)
            records.append(
                {"episode_id": str(identifier), "scene_id": str(scene_id)}
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="loaded annotation JSON[.gz]")
    parser.add_argument("--output", required=True, help="manifest JSON to create")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--split", required=True, choices=("val_seen", "val_unseen", "test")
    )
    parser.add_argument(
        "--format",
        required=True,
        choices=("habitat", "r2r", "reverie", "expanded"),
        dest="annotation_format",
        help=(
            "habitat: root episodes with episode_id/scene_id; r2r/reverie: "
            "expand each instructions list; expanded: instr_id/scan records"
        ),
    )
    parser.add_argument(
        "--source-id-field",
        default=None,
        help="defaults to episode_id for habitat and instr_id otherwise",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="portable path recorded in the manifest (defaults to --input)",
    )
    parser.add_argument("--expected-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _read_json(args.input)
    episodes = _expanded_records(payload, args.annotation_format)
    source_id_field = args.source_id_field
    if source_id_field is None:
        source_id_field = (
            "episode_id" if args.annotation_format == "habitat" else "instr_id"
        )
    manifest = build_episode_order_manifest(
        episodes,
        benchmark=args.benchmark,
        split=args.split,
        dataset_path=args.dataset_path or args.input,
        dataset_sha256=sha256_file(args.input),
        source_id_field=source_id_field,
    )
    if (
        args.expected_count is not None
        and manifest["episode_count"] != args.expected_count
    ):
        raise ValueError(
            "Expected {} episodes, found {}".format(
                args.expected_count, manifest["episode_count"]
            )
        )
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(
        "wrote {} episodes ({}) to {}".format(
            manifest["episode_count"], manifest["order_sha256"], args.output
        )
    )


if __name__ == "__main__":
    main()
