#!/usr/bin/env python3
"""Fingerprint the exact ordered episode stream used by AVN TTA evaluation."""

import argparse
import gzip
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def scene_key(episode):
    return os.path.splitext(os.path.basename(str(episode["scene_id"])))[0]


def episode_key(episode):
    episode_id = str(episode["episode_id"])
    try:
        return 0, int(episode_id), episode_id
    except ValueError:
        return 1, episode_id, episode_id


def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def build_stream(dataset_file, episodes_per_scene, expected_scenes, seed):
    dataset_file = Path(dataset_file).resolve()
    dataset_dir = dataset_file.parent
    root = load_json_gz(dataset_file)
    episodes = list(root.get("episodes", []))

    content_template = root.get(
        "content_scenes_path", "{data_path}/content/{scene}.json.gz"
    )
    prefix, suffix = content_template.split("{scene}", 1)
    content_dir = Path(prefix.format(data_path=str(dataset_dir)))
    content_suffix = suffix
    for path in sorted(content_dir.iterdir()):
        if path.name.endswith(content_suffix):
            episodes.extend(load_json_gz(path).get("episodes", []))

    grouped = defaultdict(list)
    for episode in episodes:
        grouped[scene_key(episode)].append(episode)

    if len(grouped) != expected_scenes:
        raise ValueError(
            "expected {} scenes, found {}: {}".format(
                expected_scenes, len(grouped), ", ".join(sorted(grouped))
            )
        )

    selected = []
    for scene in sorted(grouped):
        scene_episodes = sorted(grouped[scene], key=episode_key)
        if len(scene_episodes) < episodes_per_scene:
            raise ValueError(
                "scene '{}' has only {} episodes; {} required".format(
                    scene, len(scene_episodes), episodes_per_scene
                )
            )
        selected.extend(scene_episodes[:episodes_per_scene])

    random.Random(seed).shuffle(selected)
    return selected


def fingerprints(stream):
    order_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    for episode in stream:
        order_line = "{}\t{}\n".format(
            scene_key(episode), str(episode["episode_id"])
        )
        order_digest.update(order_line.encode("utf-8"))
        canonical = json.dumps(
            episode, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        content_digest.update(canonical.encode("utf-8"))
        content_digest.update(b"\n")
    return order_digest.hexdigest(), content_digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes-per-scene", type=int, default=100)
    parser.add_argument("--expected-scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episode-count",
        type=int,
        help="fingerprint only this prefix of the constructed stream",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    stream = build_stream(
        args.dataset,
        episodes_per_scene=args.episodes_per_scene,
        expected_scenes=args.expected_scenes,
        seed=args.seed,
    )
    if args.episode_count is not None:
        if args.episode_count < 1 or args.episode_count > len(stream):
            parser.error(
                "--episode-count must be in [1, {}]".format(len(stream))
            )
        stream = stream[: args.episode_count]
    order_sha256, content_sha256 = fingerprints(stream)
    result = {
        "dataset": str(Path(args.dataset).resolve()),
        "seed": args.seed,
        "expected_scenes": args.expected_scenes,
        "episodes_per_scene": args.episodes_per_scene,
        "episode_count": len(stream),
        "order_sha256": order_sha256,
        "content_sha256": content_sha256,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(order_sha256, content_sha256)


if __name__ == "__main__":
    main()
