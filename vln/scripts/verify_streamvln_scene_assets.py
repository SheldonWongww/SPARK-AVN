#!/usr/bin/env python3
"""Check a StreamVLN Habitat MP3D scene before native simulator loading."""

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from mp3d_scene_assets import (  # noqa: E402
    DEFAULT_SCENE,
    SceneAssetError,
    validate_scene_directory,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the required GLB, navmesh, house, and semantic PLY for "
            "one or more StreamVLN MP3D scenes."
        )
    )
    parser.add_argument("scenes", nargs="*", default=[DEFAULT_SCENE])
    parser.add_argument(
        "--scene-root",
        default=str(REPO_ROOT / "vln/data/scene_datasets/mp3d"),
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args()

    reports = []
    failures = []
    scene_root = Path(args.scene_root)
    for scene in args.scenes:
        try:
            reports.append(validate_scene_directory(scene_root / scene, scene=scene))
        except (OSError, SceneAssetError) as error:
            failures.append({"scene": scene, "error": str(error)})

    document = {"ok": not failures, "reports": reports, "failures": failures}
    if args.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        for report in reports:
            semantic = report["semantic_ply"]
            print(
                "PASS {} glb={} semantic={} vertices={} faces={}".format(
                    report["scene"],
                    report["glb"]["size"],
                    semantic["size"],
                    semantic["vertices"],
                    semantic["faces"],
                )
            )
        for failure in failures:
            print(
                "FAIL {}: {}".format(failure["scene"], failure["error"]),
                file=sys.stderr,
            )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
