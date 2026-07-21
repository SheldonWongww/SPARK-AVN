#!/usr/bin/env python3
"""Expose canonical AVN data to the legacy baseline-relative paths."""

import os
from pathlib import Path


AVN_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATA = AVN_ROOT / "data"
BASELINES = (
    AVN_ROOT / "baselines" / "smt_audio",
    AVN_ROOT / "baselines" / "enmus",
)
LINKS = {
    "scene_datasets": CANONICAL_DATA / "scene_datasets",
    "scene_observations": CANONICAL_DATA / "scene_observations",
    "metadata": CANONICAL_DATA / "metadata",
    "sounds": CANONICAL_DATA / "sounds",
    "binaural_rirs": CANONICAL_DATA / "binaural_rirs",
    "datasets": CANONICAL_DATA / "datasets",
}


def ensure_link(link: Path, target: Path) -> None:
    if not target.exists():
        raise FileNotFoundError("Canonical AVN data path is missing: {}".format(target))

    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = Path(os.path.relpath(str(target), str(link.parent)))
    if link.is_symlink():
        if Path(os.readlink(str(link))) == relative_target:
            print("ok     {} -> {}".format(link, relative_target))
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            "Refusing to replace non-symlink baseline data path: {}".format(link)
        )

    link.symlink_to(relative_target, target_is_directory=True)
    print("linked {} -> {}".format(link, relative_target))


def main() -> None:
    for baseline in BASELINES:
        for name, target in LINKS.items():
            ensure_link(baseline / "data" / name, target)


if __name__ == "__main__":
    main()
