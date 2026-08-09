#!/usr/bin/env python3
"""Fetch the pinned, read-only VLN upstream repositories.

Datasets and checkpoints are deliberately not downloaded by this script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "vln" / "manifests" / "upstream_repositories.json"


class FetchError(RuntimeError):
    pass


def run(command: Sequence[str], cwd: Optional[Path] = None) -> str:
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    completed = subprocess.run(
        list(command),
        cwd=str(cwd or REPO_ROOT),
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise FetchError("command failed: {}\n{}".format(" ".join(command), detail))
    return completed.stdout.strip()


def load_manifest(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FetchError("cannot read {}: {}".format(path, error)) from error
    if document.get("schema_version") != 1:
        raise FetchError("unsupported manifest schema")
    if not isinstance(document.get("repositories"), list):
        raise FetchError("manifest repositories must be a list")
    return document


def fetch_one(source_root: Path, record: dict) -> None:
    name = str(record["name"])
    url = str(record["url"])
    commit = str(record["commit"])
    destination = source_root / name

    if destination.exists():
        if not (destination / ".git").is_dir():
            raise FetchError("{} exists but is not a Git repository".format(destination))
        actual = run(("git", "rev-parse", "HEAD"), destination)
        if actual != commit:
            raise FetchError(
                "{} is at {}, expected {}; move it aside before fetching".format(
                    destination, actual, commit
                )
            )
        print("verified {} {}".format(name, commit))
        return

    source_root.mkdir(parents=True, exist_ok=True)
    run(
        (
            "git",
            "clone",
            "--filter=blob:none",
            "--depth",
            "1",
            "--no-checkout",
            url,
            str(destination),
        )
    )
    try:
        if run(("git", "rev-parse", "HEAD"), destination) != commit:
            run(("git", "fetch", "--depth", "1", "origin", commit), destination)
        run(("git", "checkout", "--detach", commit), destination)
    except BaseException:
        raise FetchError(
            "fetch of {} was incomplete; inspect or remove {} before retrying".format(
                name, destination
            )
        )
    print("fetched {} {}".format(name, commit))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    document = load_manifest(args.manifest.resolve())
    source_root = REPO_ROOT / str(document["source_root"])
    for record in document["repositories"]:
        fetch_one(source_root, record)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FetchError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
