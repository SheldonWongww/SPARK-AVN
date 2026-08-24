#!/usr/bin/env python3
"""Build R2R-CE v1.3 annotations with the legacy BERT token IDs.

ETPNav and BEVBert consume the BERT-indexed v1.2 files, while StreamVLN
consumes the public v1.3 files.  The episode identities and instructions are
the same, but every episode has a different start rotation.  This utility
keeps every v1.3 episode field and replaces only the instruction vocabulary
and token IDs with the pinned v1.2 BERT encoding.  Evaluation splits remain
the default; ``--include-train`` additionally builds the source-training split
needed for offline IDEA source-statistics collection.
"""

import argparse
import copy
import gzip
import hashlib
import json
import os


EVALUATION_SPLITS = ("val_seen", "val_unseen", "test")


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparable_episode(episode):
    value = copy.deepcopy(episode)
    value.pop("start_rotation", None)
    instruction = value.get("instruction", {})
    instruction.pop("instruction_tokens", None)
    return value


def write_deterministic_gzip_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    temporary = path + ".tmp"
    with open(temporary, "wb") as raw_stream:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_stream,
            mtime=0,
        ) as gzip_stream:
            gzip_stream.write(encoded)
    os.replace(temporary, path)


def build_split(v13_path, v12_bert_path, output_path):
    v13 = read_gzip_json(v13_path)
    v12 = read_gzip_json(v12_bert_path)
    v13_episodes = v13.get("episodes")
    v12_episodes = v12.get("episodes")
    if not isinstance(v13_episodes, list) or not isinstance(v12_episodes, list):
        raise ValueError("both inputs must contain an episodes list")

    v12_by_id = {str(item["episode_id"]): item for item in v12_episodes}
    v13_ids = [str(item["episode_id"]) for item in v13_episodes]
    if len(v12_by_id) != len(v12_episodes) or set(v13_ids) != set(v12_by_id):
        raise ValueError("v1.2 and v1.3 episode IDs do not match exactly")

    output = copy.deepcopy(v13)
    output["instruction_vocab"] = copy.deepcopy(v12["instruction_vocab"])
    for episode in output["episodes"]:
        episode_id = str(episode["episode_id"])
        token_source = v12_by_id[episode_id]
        if comparable_episode(episode) != comparable_episode(token_source):
            raise ValueError(
                "episode {} differs beyond start_rotation/token IDs".format(
                    episode_id
                )
            )
        episode["instruction"]["instruction_tokens"] = copy.deepcopy(
            token_source["instruction"]["instruction_tokens"]
        )

    write_deterministic_gzip_json(output_path, output)
    rebuilt = read_gzip_json(output_path)
    if rebuilt != output:
        raise RuntimeError("deterministic output failed round-trip validation")
    return {
        "episodes": len(output["episodes"]),
        "sha256": sha256_file(output_path),
        "path": os.path.abspath(output_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v13-root", required=True)
    parser.add_argument("--v12-bert-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="also build train/train_bertidx.json.gz for IDEA source statistics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = {}
    splits = (
        ("train",) + EVALUATION_SPLITS
        if args.include_train else EVALUATION_SPLITS
    )
    for split in splits:
        summary[split] = build_split(
            os.path.join(args.v13_root, split, split + ".json.gz"),
            os.path.join(
                args.v12_bert_root,
                split,
                split + "_bertidx.json.gz",
            ),
            os.path.join(
                args.output_root,
                split,
                split + "_bertidx.json.gz",
            ),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
