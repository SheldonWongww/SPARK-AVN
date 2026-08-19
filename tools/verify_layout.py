#!/usr/bin/env python3
"""Verify task isolation and required NavTTA repository structure."""

import json
import os
import posixpath
import re
import sys


REFERENCE_FIELDS = ("name", "category", "url", "commit", "path")
REQUIRED_REFERENCE_NAMES = frozenset(
    {
        "VLN-DUET",
        "VLN-HAMT",
        "sound-spaces",
        "scene-memory-transformer",
        "ENMuS",
        "PONI",
        "Stubborn",
        "Tent",
        "ICML2024-FSTTA",
        "NeurIPS25-ATENA",
        "PEA-TTA",
        "GOLD",
        "TTRV",
        "phd-skills",
        "academic-research-skills-codex",
    }
)
GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def validate_reference_catalog(catalog):
    """Return content errors without imposing a fixed repository count."""
    if not isinstance(catalog, dict):
        return ["reference catalog must be a JSON object"]

    repositories = catalog.get("repositories")
    if not isinstance(repositories, list):
        return ["reference catalog repositories must be a list"]

    errors = []
    names = set()
    paths = set()
    for index, entry in enumerate(repositories):
        label = "reference catalog repositories[{}]".format(index)
        if not isinstance(entry, dict):
            errors.append("{} must be an object".format(label))
            continue

        missing = [field for field in REFERENCE_FIELDS if field not in entry]
        if missing:
            errors.append(
                "{} missing required fields: {}".format(label, ", ".join(missing))
            )

        for field in REFERENCE_FIELDS:
            if field in entry and (
                not isinstance(entry[field], str) or not entry[field].strip()
            ):
                errors.append("{} field {} must be a non-empty string".format(label, field))

        name = entry.get("name")
        if isinstance(name, str) and name.strip():
            if name in names:
                errors.append("duplicate reference repository name: {}".format(name))
            names.add(name)

        path = entry.get("path")
        if isinstance(path, str) and path.strip():
            if path in paths:
                errors.append("duplicate reference repository path: {}".format(path))
            paths.add(path)
            if (
                posixpath.isabs(path)
                or posixpath.normpath(path) != path
                or not path.startswith("references/repos/")
            ):
                errors.append(
                    "{} path must be normalized under references/repos/: {}".format(
                        label, path
                    )
                )

        commit = entry.get("commit")
        if (
            isinstance(commit, str)
            and commit.strip()
            and not GIT_COMMIT_PATTERN.fullmatch(commit)
        ):
            errors.append("{} commit must be a 40-character lowercase SHA-1".format(label))

    missing_required = sorted(REQUIRED_REFERENCE_NAMES - names)
    if missing_required:
        errors.append(
            "reference catalog missing required repositories: {}".format(
                ", ".join(missing_required)
            )
        )
    return errors


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    required = [
        "core/navtta_core/tta/tta_core.py",
        "avn/baselines/smt_audio/UPSTREAM.md",
        "avn/baselines/enmus/UPSTREAM.md",
        "avn/data/manifests/datasets.yaml",
        "avn/checkpoints/manifests/imported_pretrained.yaml",
        "avn/results/analysis/hparam_search/README.md",
        "avn/results/analysis/hparam_search/TENT_HYPERPARAMETER_SEARCH_REPORT.md",
        "avn/results/analysis/hparam_search/FSTTA_HYPERPARAMETER_SEARCH_REPORT.md",
        "avn/results/analysis/hparam_search/EAM_HYPERPARAMETER_SEARCH_REPORT.md",
        "avn/results/analysis/hparam_search/FEEDTTA_HYPERPARAMETER_SEARCH_REPORT.md",
        "vln/STATUS.md",
        "objectnav/STATUS.md",
        "references/catalog.json",
    ]
    errors = []
    for relative in required:
        if not os.path.exists(os.path.join(repo_root, relative)):
            errors.append("missing {}".format(relative))

    catalog_path = os.path.join(repo_root, "references", "catalog.json")
    if os.path.isfile(catalog_path):
        try:
            with open(catalog_path, encoding="utf-8") as handle:
                catalog = json.load(handle)
        except (OSError, ValueError) as error:
            errors.append("invalid reference catalog: {}".format(error))
        else:
            errors.extend(validate_reference_catalog(catalog))

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
