#!/usr/bin/env python3
"""Verify task isolation and required NavTTA repository structure."""

import json
import os
import sys


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    required = [
        "core/navtta_core/tta/tta_core.py",
        "avn/baselines/smt_audio/UPSTREAM.md",
        "avn/baselines/enmus/UPSTREAM.md",
        "avn/data/manifests/datasets.yaml",
        "avn/checkpoints/manifests/imported_pretrained.yaml",
        "vln/STATUS.md",
        "objectnav/STATUS.md",
        "references/catalog.json",
    ]
    errors = []
    for relative in required:
        if not os.path.exists(os.path.join(repo_root, relative)):
            errors.append("missing {}".format(relative))

    with open(os.path.join(repo_root, "references", "catalog.json"), encoding="utf-8") as handle:
        catalog = json.load(handle)
    if len(catalog.get("repositories", [])) != 15:
        errors.append("reference catalog must contain 15 repositories")

    for current, directories, _ in os.walk(repo_root):
        relative = os.path.relpath(current, repo_root)
        if relative.startswith(os.path.join("references", "repos")):
            directories[:] = []
            continue
        if ".git" in directories:
            if relative != ".":
                errors.append("nested .git outside references: {}".format(relative))
            directories.remove(".git")

    if errors:
        for error in errors:
            print("ERROR: " + error)
        return 1
    print("NavTTA layout verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
