#!/usr/bin/env python3
"""Verify the installed VLN evaluation assets without loading model tensors."""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORE_SOURCE_ROOT = os.path.join(REPO_ROOT, "core")
if CORE_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, CORE_SOURCE_ROOT)

from navtta_core.experiment.episode_order import (  # noqa: E402
    canonical_episode_records,
    sha256_file,
    validate_episode_order_manifest,
    verify_manifest_dataset,
)


SMALL_FILE_LIMIT = 64 * 1024 * 1024
SPLITS = ("val_seen", "val_unseen", "test")


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def scene_name(scene_id):
    return os.path.splitext(str(scene_id).replace("\\", "/").split("/")[-1])[0]


def load_annotation_episodes(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        payload = payload.get("episodes")
    if not isinstance(payload, list):
        raise ValueError(
            "annotation must be a list or contain a root episodes list"
        )
    return payload


def check_source_train_assets(repo_root, errors):
    """Verify the separately tracked inputs used to build IDEA source anchors."""
    manifest_path = os.path.join(
        repo_root,
        "vln",
        "manifests",
        "assets",
        "source_train_assets.json",
    )
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError) as error:
        errors.append("cannot read source-train asset manifest: {}".format(error))
        return 0

    checked = 0
    for asset in manifest.get("assets", []):
        paths = asset.get("paths")
        if paths is None:
            paths = [asset.get("path")]
        if not paths or any(not item for item in paths):
            errors.append("source-train asset {} has no path".format(asset.get("id")))
            continue
        for relative in paths:
            path = resolve_path(repo_root, relative)
            checked += 1
            if not os.path.isfile(path):
                errors.append("missing source-train asset {}: {}".format(
                    asset.get("id"), path
                ))
                continue
            size = os.path.getsize(path)
            if size != int(asset["size"]):
                errors.append(
                    "source-train size mismatch for {}: expected {}, got {}"
                    .format(asset.get("id"), asset["size"], size)
                )
                continue
            if sha256_file(path) != asset["sha256"]:
                errors.append(
                    "source-train SHA256 mismatch for {}".format(asset.get("id"))
                )
                continue
            try:
                count = len(load_annotation_episodes(path))
            except (OSError, TypeError, ValueError) as error:
                errors.append(
                    "source-train annotation {} is invalid: {}".format(
                        asset.get("id"), error
                    )
                )
                continue
            expected_count = asset.get("episodes", asset.get("records"))
            if count != int(expected_count):
                errors.append(
                    "source-train record count mismatch for {}: expected {}, got {}"
                    .format(asset.get("id"), expected_count, count)
                )

    scene_root = os.path.join(
        repo_root, "vln", "data", "scene_datasets", "mp3d"
    )
    for scene in manifest.get("train_only_mp3d_scenes_extracted", []):
        for suffix in (".glb", ".navmesh"):
            path = os.path.join(scene_root, scene, scene + suffix)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                errors.append("missing train-only scene asset: {}".format(path))
    return checked


def canonical_annotation_records(path, manifest):
    episodes = load_annotation_episodes(path)
    source_id_field = manifest["source_id_field"]
    source_field_present = [
        isinstance(episode, dict) and source_id_field in episode
        for episode in episodes
    ]

    if all(source_field_present):
        records = []
        for episode in episodes:
            scene_id = episode.get("scene_id", episode.get("scan"))
            if scene_id is None:
                raise ValueError("annotation item is missing scan/scene_id")
            records.append(
                {
                    "episode_id": episode[source_id_field],
                    "scene_id": scene_id,
                }
            )
        return canonical_episode_records(records)

    if any(source_field_present):
        raise ValueError(
            "annotation source ID field {!r} is present inconsistently".format(
                source_id_field
            )
        )
    if source_id_field != "instr_id":
        raise ValueError(
            "annotation items are missing source ID field {!r}".format(
                source_id_field
            )
        )

    records = []
    is_reverie = str(manifest["benchmark"]).startswith("reverie_")
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("annotation episodes must be objects")
        instructions = episode.get("instructions")
        if not isinstance(instructions, list):
            raise ValueError("instruction annotation requires an instructions list")
        scene_id = episode.get("scan", episode.get("scene_id"))
        if scene_id is None:
            raise ValueError("annotation item is missing scan/scene_id")
        for instruction_index in range(len(instructions)):
            if is_reverie and "objId" in episode:
                identifier = "{}_{}_{}".format(
                    episode["path_id"], episode["objId"], instruction_index
                )
            elif is_reverie:
                base_id = episode.get("id", episode.get("path_id"))
                if base_id is None:
                    raise ValueError("REVERIE item is missing id/path_id")
                identifier = "{}_{}".format(base_id, instruction_index)
            else:
                identifier = "{}_{}".format(
                    episode["path_id"], instruction_index
                )
            records.append(
                {"episode_id": identifier, "scene_id": scene_id}
            )
    return canonical_episode_records(records)


def episode_record_mismatch(expected, actual):
    if len(expected) != len(actual):
        return "expected {} records, found {}".format(len(expected), len(actual))
    for index, (expected_record, actual_record) in enumerate(zip(expected, actual)):
        if expected_record != actual_record:
            return "record {} expected {}, found {}".format(
                index, expected_record, actual_record
            )
    return "records differ"


def check_episode_manifest(repo_root, path, errors):
    relative = os.path.relpath(path, repo_root)
    try:
        document = load_json(path)
    except (OSError, ValueError) as error:
        errors.append("cannot read {}: {}".format(relative, error))
        return

    try:
        validate_episode_order_manifest(document)
    except (TypeError, ValueError) as error:
        errors.append("{} is invalid: {}".format(relative, error))
        return

    dataset_path = resolve_path(repo_root, document["dataset"]["path"])
    if not os.path.isfile(dataset_path):
        errors.append(
            "{} dataset is missing: {}".format(relative, dataset_path)
        )
        return

    try:
        verify_manifest_dataset(document, dataset_path)
    except (OSError, TypeError, ValueError) as error:
        errors.append("{} dataset verification failed: {}".format(relative, error))

    try:
        actual_records = canonical_annotation_records(dataset_path, document)
    except (OSError, TypeError, ValueError) as error:
        errors.append("{} dataset episodes are invalid: {}".format(relative, error))
        return

    expected_records = document["episodes"]
    if actual_records != expected_records:
        errors.append(
            "{} dataset episode records mismatch: {}".format(
                relative,
                episode_record_mismatch(expected_records, actual_records),
            )
        )


def check_episode_manifests(repo_root, errors):
    root = os.path.join(repo_root, "vln", "manifests", "episode_order")
    files = []
    manifest_directories = []
    for directory, _, names in os.walk(root):
        json_names = {name for name in names if name.endswith(".json")}
        if json_names:
            manifest_directories.append(directory)
            expected_names = {split + ".json" for split in SPLITS}
            if json_names != expected_names:
                errors.append(
                    "{} must contain exactly {}".format(
                        os.path.relpath(directory, repo_root),
                        sorted(expected_names),
                    )
                )
        for name in names:
            if name.endswith(".json"):
                files.append(os.path.join(directory, name))
    for path in sorted(files):
        check_episode_manifest(repo_root, path, errors)
    return len(files), len(manifest_directories)


def collect_ce_scenes(repo_root):
    scenes = set()
    for split in SPLITS:
        path = os.path.join(
            repo_root,
            "vln",
            "data",
            "datasets",
            "r2r",
            split,
            split + ".json.gz",
        )
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
        scenes.update(scene_name(episode["scene_id"]) for episode in document["episodes"])
    return scenes


def check_scene_subset(repo_root, expected, errors):
    scenes_root = os.path.join(repo_root, "vln", "data", "scene_datasets", "mp3d")
    actual = {
        name
        for name in os.listdir(scenes_root)
        if os.path.isdir(os.path.join(scenes_root, name))
    }
    source_scenes = collect_ce_scenes(repo_root)
    if not source_scenes.issubset(actual):
        errors.append(
            "CE scene directory misses val/test scenes: {}".format(
                sorted(source_scenes - actual)
            )
        )
    if len(source_scenes) != expected:
        errors.append(
            "expected {} val/test CE scenes, found {}".format(
                expected, len(source_scenes)
            )
        )
    # Extra scene directories are valid: IDEA source-statistics collection
    # additionally needs the eight train-only MP3D scenes.
    for scene in sorted(source_scenes):
        directory = os.path.join(scenes_root, scene)
        for suffix in (".glb", ".navmesh"):
            path = os.path.join(directory, scene + suffix)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                errors.append("missing scene asset: {}".format(path))


def resolve_path(repo_root, path):
    return path if os.path.isabs(path) else os.path.join(repo_root, path)


def tree_digest(root, include_top_level=None):
    digest = hashlib.sha256()
    relative_paths = []
    for directory, _, names in os.walk(root):
        for name in names:
            relative = os.path.relpath(
                os.path.join(directory, name), root
            ).replace(os.sep, "/")
            if (
                include_top_level is None
                or relative.split("/", 1)[0] in include_top_level
            ):
                relative_paths.append(relative)
    for relative in sorted(relative_paths):
        path = os.path.join(root, *relative.split("/"))
        digest.update(
            "{}  ./{}\n".format(sha256_file(path), relative).encode("utf-8")
        )
    return digest.hexdigest()


def environment_versions(python, packages):
    program = (
        "import json,sys\n"
        "try:\n from importlib.metadata import version\n"
        "except ImportError:\n from importlib_metadata import version\n"
        "out={}\n"
        "for name in json.loads(sys.argv[1]):\n"
        " try: out[name]=version(name)\n"
        " except Exception as error: out[name]='ERROR:'+str(error)\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    completed = subprocess.run(
        [python, "-c", program, json.dumps(sorted(packages))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "package query failed")
    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hash",
        choices=("none", "small", "all"),
        default="small",
        help=(
            "bulk-hash no assets, assets up to 64 MiB, or all assets; "
            "manifest-linked annotations and derived/native integrity checks "
            "always run"
        ),
    )
    parser.add_argument("--allow-non-autodl", action="store_true")
    args = parser.parse_args()

    repo_root = REPO_ROOT
    if not args.allow_non_autodl and not repo_root.startswith("/root/autodl-tmp/"):
        print("refusing to verify runtime assets outside /root/autodl-tmp", file=sys.stderr)
        return 2

    asset_manifest = load_json(
        os.path.join(repo_root, "vln", "manifests", "assets", "eval_assets.json")
    )
    environment_manifest = load_json(
        os.path.join(
            repo_root,
            "vln",
            "manifests",
            "environments",
            "eval_environments.json",
        )
    )
    errors = []
    hashed = 0

    for asset in asset_manifest["assets"]:
        path = resolve_path(repo_root, asset["path"])
        if not os.path.isfile(path):
            errors.append("missing asset {}: {}".format(asset["id"], path))
            continue
        size = os.path.getsize(path)
        if size != asset["size"]:
            errors.append(
                "size mismatch for {}: expected {}, got {}".format(
                    asset["id"], asset["size"], size
                )
            )
            continue
        should_hash = args.hash == "all" or (
            args.hash == "small" and size <= SMALL_FILE_LIMIT
        )
        if should_hash:
            actual = sha256_file(path)
            hashed += 1
            if actual != asset["sha256"]:
                errors.append("SHA256 mismatch for {}".format(asset["id"]))

    for link in asset_manifest.get("runtime_links", []):
        path = resolve_path(repo_root, link["path"])
        target = link["target"]
        if os.path.isabs(target):
            expected = os.path.realpath(target)
        else:
            expected = os.path.realpath(os.path.join(os.path.dirname(path), target))
        if not os.path.islink(path):
            errors.append("runtime link is missing: {}".format(path))
        elif os.path.realpath(path) != expected:
            errors.append(
                "runtime link target mismatch for {}: expected {}, got {}".format(
                    path, expected, os.path.realpath(path)
                )
            )
        elif not os.path.exists(expected):
            errors.append("runtime link target is missing: {}".format(expected))

    for name, environment in environment_manifest["environments"].items():
        prefix = environment["prefix"]
        python = os.path.join(prefix, "bin", "python")
        if not os.path.isfile(python):
            errors.append("missing environment {}: {}".format(name, prefix))
            continue
        if any("pending" in str(value).lower() for value in environment.values()):
            errors.append("environment {} still contains a pending field".format(name))
        expected_versions = environment.get("verify_packages", {})
        try:
            actual_versions = environment_versions(python, expected_versions)
        except (OSError, RuntimeError, ValueError) as error:
            errors.append("cannot inspect environment {}: {}".format(name, error))
            continue
        for package, expected_version in expected_versions.items():
            if actual_versions.get(package) != expected_version:
                errors.append(
                    "environment {} package {} expected {}, got {}".format(
                        name,
                        package,
                        expected_version,
                        actual_versions.get(package),
                    )
                )

    for dependency_name, dependency in environment_manifest.get(
        "native_dependencies", {}
    ).items():
        for field in ("runtime_path", "git_metadata_backup"):
            path = dependency.get(field)
            if path and not os.path.exists(path):
                errors.append(
                    "missing native dependency {} {}: {}".format(
                        dependency_name, field, path
                    )
                )

        path = dependency.get("path")
        if path:
            path = resolve_path(repo_root, path)
            if not os.path.isfile(path):
                errors.append(
                    "missing native dependency file {}: {}".format(
                        dependency_name, path
                    )
                )
            else:
                expected_size = dependency.get("size")
                if expected_size is not None and os.path.getsize(path) != expected_size:
                    errors.append(
                        "native dependency size mismatch for {}: expected {}, got {}".format(
                            dependency_name, expected_size, os.path.getsize(path)
                        )
                    )
                expected_sha256 = dependency.get("sha256")
                if expected_sha256 and sha256_file(path) != expected_sha256:
                    errors.append(
                        "native dependency SHA256 mismatch: {}".format(
                            dependency_name
                        )
                    )

        binary = dependency.get("python_module")
        expected_sha256 = dependency.get("sha256")
        if binary and expected_sha256:
            if not os.path.isfile(binary):
                errors.append("missing native module: {}".format(binary))
            elif sha256_file(binary) != expected_sha256:
                errors.append("native module SHA256 mismatch: {}".format(binary))

    source_train_asset_count = check_source_train_assets(repo_root, errors)
    manifest_count, manifest_directory_count = check_episode_manifests(
        repo_root, errors
    )
    scene_check = asset_manifest["derived_checks"]["ce_scene_set"]
    check_scene_subset(repo_root, scene_check["scene_count"], errors)
    scene_path = resolve_path(repo_root, scene_check["path"])
    if os.path.isdir(scene_path):
        if tree_digest(
            scene_path, include_top_level=collect_ce_scenes(repo_root)
        ) != scene_check["tree_digest_sha256"]:
            errors.append("derived tree digest mismatch: ce_scene_set")

    for check_name in ("discrete_connectivity", "goat_connectivity_mirror"):
        check = asset_manifest["derived_checks"][check_name]
        path = resolve_path(repo_root, check["path"])
        if not os.path.isdir(path):
            errors.append("missing derived tree {}: {}".format(check_name, path))
        elif tree_digest(path) != check["tree_digest_sha256"]:
            errors.append("derived tree digest mismatch: {}".format(check_name))

    if errors:
        for error in errors:
            print("ERROR: " + error)
        return 1
    print(
        "VLN preflight passed: {} eval assets ({} bulk-hashed), {} source-train paths, {} links, {} order manifests + source datasets verified in {} families, {} environments".format(
            len(asset_manifest["assets"]),
            hashed,
            source_train_asset_count,
            len(asset_manifest.get("runtime_links", [])),
            manifest_count,
            manifest_directory_count,
            len(environment_manifest["environments"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
