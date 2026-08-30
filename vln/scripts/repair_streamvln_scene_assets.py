#!/usr/bin/env python3
"""Restore one StreamVLN MP3D scene from the manifest-verified archive."""

import argparse
import json
import sys
import zipfile
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from mp3d_scene_assets import (  # noqa: E402
    DEFAULT_SCENE,
    SceneAssetError,
    load_archive_spec,
    repair_scene,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify the complete official MP3D archive against eval_assets.json, "
            "extract one scene into a staging directory, validate it, and rotate "
            "the current scene into a retained timestamped backup. Stop StreamVLN "
            "before running a non-dry repair."
        )
    )
    parser.add_argument("scene", nargs="?", default=DEFAULT_SCENE)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--scene-root")
    parser.add_argument("--manifest")
    parser.add_argument("--archive")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify, extract, and validate staging without replacing the runtime scene",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace a scene even when the current copy passes integrity checks",
    )
    args = parser.parse_args()

    try:
        archive_spec = load_archive_spec(
            args.repo_root,
            manifest_path=args.manifest,
            archive_path=args.archive,
        )
        print(
            "Full SHA256 scan of {} bytes: {}".format(
                archive_spec["expected_size"], archive_spec["archive_path"]
            ),
            file=sys.stderr,
            flush=True,
        )
        last_bucket = [0]

        def report_hash_progress(completed, total):
            bucket = min(10, completed * 10 // total)
            if bucket > last_bucket[0]:
                last_bucket[0] = bucket
                print(
                    "archive SHA256: {}% ({}/{})".format(
                        bucket * 10, completed, total
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        report = repair_scene(
            args.repo_root,
            scene=args.scene,
            scene_root=args.scene_root,
            manifest_path=args.manifest,
            archive_path=args.archive,
            dry_run=args.dry_run,
            force=args.force,
            progress=report_hash_progress,
        )
    except (OSError, SceneAssetError, zipfile.BadZipFile) as error:
        print("repair failed: {}".format(error), file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
