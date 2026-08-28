#!/usr/bin/env python3
"""Build a digest-pinned 128-episode AVN source subset for IDEA."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "navtta.avn.idea_source_selection.v2"
DOMAIN = "navtta.avn.idea.source128.sample.v2"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scene_id(value):
    return os.path.splitext(str(value).replace("\\", "/").rstrip("/").split("/")[-1])[0]


def _load(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _portable(path):
    path = Path(path).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _dataset_files(index_path):
    index_path = Path(index_path).resolve()
    root = _load(index_path)
    template = root.get(
        "content_scenes_path", "{data_path}/content/{scene}.json.gz"
    )
    prefix, suffix = template.split("{scene}", 1)
    content_dir = Path(prefix.format(data_path=str(index_path.parent)))
    content = sorted(
        path for path in content_dir.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    )
    return root, [index_path] + content


def build_manifest(dataset, checkpoint, model, source_setting, seed=0, count=128):
    if count != 128:
        raise ValueError("canonical IDEA source collection requires exactly 128 episodes")
    if model not in ("smt_audio", "enmus"):
        raise ValueError("unsupported AVN source model: {}".format(model))
    if source_setting not in ("single_source", "multi_source"):
        raise ValueError(
            "unsupported AVN source setting: {}".format(source_setting)
        )
    dataset = Path(dataset).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not dataset.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("dataset index and checkpoint must exist")
    dataset_parts = tuple(part.lower() for part in dataset.parts)
    if (
        dataset.name.lower() != "train.json.gz"
        or dataset.parent.name.lower() != "train"
        or "v1" not in dataset_parts
        or source_setting not in dataset_parts
    ):
        raise ValueError(
            "canonical AVN IDEA source data must be the matching "
            "<single_source|multi_source>/mp3d/v1/train/train.json.gz bundle"
        )

    root, files = _dataset_files(dataset)
    common = Path(os.path.commonpath([str(path) for path in files]))
    if common.is_file():
        common = common.parent
    file_records = [
        {"path": str(path.relative_to(common)), "sha256": _sha256(path)}
        for path in files
    ]
    dataset_sha256 = _canonical_sha(file_records)

    ranked = []
    seen = set()
    for path in files:
        payload = root if path == dataset else _load(path)
        for episode in payload.get("episodes", []):
            scene = _scene_id(episode["scene_id"])
            episode_id = str(episode["episode_id"])
            key = (scene, episode_id)
            if key in seen:
                raise ValueError("duplicate source episode: {}/{}".format(*key))
            seen.add(key)
            rank_payload = "\0".join(
                (DOMAIN, str(int(seed)), source_setting, dataset_sha256, scene, episode_id)
            ).encode("utf-8")
            ranked.append((hashlib.sha256(rank_payload).hexdigest(), scene, episode_id))
    if len(ranked) < count:
        raise ValueError("source dataset has only {} unique episodes".format(len(ranked)))
    ranked.sort()
    episodes = [
        {
            "scene_id": scene,
            "episode_id": episode_id,
            "trajectory_id": "{}/{}".format(scene, episode_id),
        }
        for _, scene, episode_id in ranked[:count]
    ]
    protocol = {
        "action_selection": "sample",
        "algorithm": "domain_separated_sha256_rank_v1",
        "count": count,
        "domain_separator": DOMAIN,
        "seed": int(seed),
    }
    return {
        "schema": SCHEMA,
        "task": "avn",
        "benchmark": "mp3d",
        "model": model,
        "source_setting": source_setting,
        "dataset": {
            "name": "avn_mp3d_{}".format(source_setting),
            "version": "v1",
            "split": "train",
            "index_path": _portable(dataset),
            "bundle_root": _portable(common),
            "bundle_sha256": dataset_sha256,
            "files": file_records,
        },
        "checkpoint": {
            "path": _portable(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "selection": protocol,
        "protocol_sha256": _canonical_sha(protocol),
        "episode_count": len(episodes),
        "episode_order_sha256": _canonical_sha(episodes),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("smt_audio", "enmus"))
    parser.add_argument(
        "--source-setting", required=True,
        choices=("single_source", "multi_source"),
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = build_manifest(
        args.dataset, args.checkpoint, args.model, args.source_setting,
        seed=args.seed, count=args.count,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(_sha256(output))


if __name__ == "__main__":
    main()
