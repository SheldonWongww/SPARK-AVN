#!/usr/bin/env python3
"""Build one ordered 11-scene MP3D split for global-continual TTA."""

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    script_dir = Path(__file__).resolve().parent
    poni_root = script_dir.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=str(
            poni_root / "data/datasets/objectnav/mp3d/v1/val_parts"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(poni_root / "experiments/global_continual_dataset"),
    )
    parser.add_argument("--split", default="val_tta_all")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    split_dir = Path(args.output_root).resolve() / args.split
    content_dir = split_dir / "content"
    content_dir.mkdir(parents=True, exist_ok=True)

    root_payload = None
    scenes = []
    total_episodes = 0
    expected_outputs = set()
    for part in range(11):
        part_name = "val_part_{}".format(part)
        part_dir = source_root / part_name
        with gzip.open(part_dir / (part_name + ".json.gz"), "rt") as handle:
            metadata = json.load(handle)
        if root_payload is None:
            root_payload = metadata
        else:
            for key in (
                "category_to_task_category_id",
                "category_to_mp3d_category_id",
            ):
                if metadata.get(key) != root_payload.get(key):
                    raise RuntimeError("category mapping differs in " + part_name)
        files = sorted((part_dir / "content").glob("*.json.gz"))
        if len(files) != 1:
            raise RuntimeError(
                "Expected one scene file in {}, found {}".format(
                    part_dir, len(files)
                )
            )
        source = files[0]
        ordered_name = "{:02d}__{}".format(part, source.name)
        destination = content_dir / ordered_name
        expected_outputs.add(destination.name)
        if not destination.exists():
            try:
                os.link(str(source), str(destination))
            except OSError:
                shutil.copy2(str(source), str(destination))
        with gzip.open(source, "rt") as handle:
            scene_payload = json.load(handle)
        count = len(scene_payload.get("episodes", []))
        total_episodes += count
        scenes.append({
            "order": part,
            "part": part_name,
            "source_scene_file": source.name,
            "ordered_scene_file": ordered_name,
            "episodes": count,
        })

    for path in content_dir.glob("*.json.gz"):
        if path.name not in expected_outputs:
            path.unlink()
    split_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(split_dir / (args.split + ".json.gz"), "wt") as handle:
        json.dump(root_payload, handle)
    manifest = {
        "protocol": "global_continual",
        "split": args.split,
        "iterator_options": {
            "shuffle": False,
            "group_by_scene": True,
            "max_scene_repeat_episodes": -1,
            "max_scene_repeat_steps": -1,
        },
        "scene_order": scenes,
        "total_episodes": total_episodes,
        "dataset_path_template": str(
            Path(args.output_root).resolve() / "{split}" / "{split}.json.gz"
        ),
    }
    atomic_json(split_dir / "STREAM_MANIFEST.json", manifest)
    if total_episodes != 2195:
        raise RuntimeError(
            "Expected 2195 episodes, built {}".format(total_episodes)
        )
    print(split_dir / (args.split + ".json.gz"))


if __name__ == "__main__":
    main()
