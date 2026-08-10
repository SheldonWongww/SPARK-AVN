#!/usr/bin/env python3
"""Validate the immutable identity and successful completion of a run."""

import argparse
import hashlib
import json
import os
import re
import sys

try:
    from .run_manifest_identity import (
        IMMUTABLE_IDENTITY_SHA256_FIELD,
        immutable_identity_sha256,
    )
except ImportError:
    from run_manifest_identity import (
        IMMUTABLE_IDENTITY_SHA256_FIELD,
        immutable_identity_sha256,
    )


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_value(document, path):
    value = document
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_file_reference(
    label, metadata, hash_field, manifest_directory, mismatches
):
    if not isinstance(metadata, dict):
        mismatches.append("{} metadata is not an object".format(label))
        return
    path = metadata.get("path")
    if not isinstance(path, str) or not path:
        mismatches.append("{} path is missing".format(label))
        return
    resolved_path = path
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.abspath(
            os.path.join(manifest_directory, resolved_path)
        )
    if not os.path.isfile(resolved_path):
        mismatches.append("{} file is missing: {}".format(label, path))
        return

    if "size" in metadata:
        expected_size = metadata.get("size")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            mismatches.append("{} size metadata is invalid".format(label))
        elif os.path.getsize(resolved_path) != expected_size:
            mismatches.append("{} size changed: {}".format(label, path))

    if hash_field in metadata:
        expected_sha256 = metadata.get(hash_field)
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ):
            mismatches.append("{} SHA256 metadata is invalid".format(label))
        elif sha256(resolved_path) != expected_sha256:
            mismatches.append("{} SHA256 changed: {}".format(label, path))


def validate_referenced_files(manifest, manifest_directory, mismatches):
    if "checkpoint" in manifest:
        validate_file_reference(
            "checkpoint",
            manifest.get("checkpoint"),
            "sha256",
            manifest_directory,
            mismatches,
        )

    if "auxiliary_checkpoints" in manifest:
        auxiliary = manifest.get("auxiliary_checkpoints")
        if not isinstance(auxiliary, list):
            mismatches.append("auxiliary_checkpoints must be a list")
        else:
            names = set()
            for index, item in enumerate(auxiliary):
                name = item.get("name") if isinstance(item, dict) else None
                if not isinstance(name, str) or not name or name in names:
                    mismatches.append(
                        "auxiliary checkpoint name is missing or duplicated at index {}".format(
                            index
                        )
                    )
                    label = "auxiliary checkpoint {}".format(index)
                else:
                    names.add(name)
                    label = "auxiliary checkpoint {}".format(name)
                validate_file_reference(
                    label, item, "sha256", manifest_directory, mismatches
                )

    dataset = manifest.get("dataset")
    if dataset is not None:
        validate_file_reference(
            "dataset",
            dataset,
            "index_sha256",
            manifest_directory,
            mismatches,
        )

    if "pinned_manifests" in manifest:
        pinned = manifest.get("pinned_manifests")
        if not isinstance(pinned, dict):
            mismatches.append("pinned_manifests must be an object")
        else:
            for name, metadata in sorted(pinned.items()):
                validate_file_reference(
                    "pinned manifest {}".format(name),
                    metadata,
                    "sha256",
                    manifest_directory,
                    mismatches,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", choices=("avn", "vln", "objectnav"), default="avn")
    parser.add_argument("--benchmark", default="mp3d")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source-setting", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--stream-order-sha256", required=True)
    parser.add_argument("--stream-content-sha256", required=True)
    parser.add_argument("--require-immutable-identity", action="store_true")
    parser.add_argument("--require-result-artifacts", action="store_true")
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        print("invalid run manifest '{}': {}".format(manifest_path, error), file=sys.stderr)
        return 1

    expected = {
        "task": args.task,
        "benchmark": args.benchmark,
        "run_tag": args.run_tag,
        "model": args.model,
        "method": args.method,
        "source_setting": args.source_setting,
        "seed": args.seed,
        "git_commit": args.git_commit,
        "status": "completed",
        "exit_code": 0,
        "checkpoint.sha256": args.checkpoint_sha256,
        "dataset.stream_order_sha256": args.stream_order_sha256,
        "dataset.stream_content_sha256": args.stream_content_sha256,
    }

    mismatches = []
    if IMMUTABLE_IDENTITY_SHA256_FIELD not in manifest:
        if args.require_immutable_identity:
            mismatches.append("immutable identity SHA256 is required")
    else:
        recorded_identity = manifest.get(IMMUTABLE_IDENTITY_SHA256_FIELD)
        if not isinstance(recorded_identity, str) or not re.fullmatch(
            r"[0-9a-f]{64}", recorded_identity
        ):
            mismatches.append("immutable identity SHA256 is invalid")
        else:
            try:
                actual_identity = immutable_identity_sha256(manifest)
            except (TypeError, ValueError) as error:
                mismatches.append(
                    "immutable identity cannot be recomputed: {}".format(error)
                )
            else:
                if actual_identity != recorded_identity:
                    mismatches.append("immutable identity SHA256 changed")

    for path, expected_value in expected.items():
        actual_value = nested_value(manifest, path)
        if actual_value != expected_value:
            mismatches.append(
                "{} expected={!r} actual={!r}".format(
                    path, expected_value, actual_value
                )
            )

    validate_referenced_files(
        manifest, os.path.dirname(manifest_path), mismatches
    )

    artifacts = manifest.get("result_artifacts")
    if args.require_result_artifacts and (
        not isinstance(artifacts, list) or not artifacts
    ):
        mismatches.append("result_artifacts must be a non-empty list")
    elif isinstance(artifacts, list):
        names = set()
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                mismatches.append("result artifact is not an object")
                continue
            name = artifact.get("name")
            path = artifact.get("path")
            if not isinstance(name, str) or not name or name in names:
                mismatches.append("result artifact name is missing or duplicated")
                continue
            names.add(name)
            if not isinstance(path, str) or not os.path.isfile(path):
                mismatches.append("result artifact is missing: {}".format(path))
                continue
            if os.path.getsize(path) != artifact.get("size"):
                mismatches.append("result artifact size changed: {}".format(name))
            elif sha256(path) != artifact.get("sha256"):
                mismatches.append("result artifact SHA256 changed: {}".format(name))

    if mismatches:
        print(
            "run manifest identity mismatch '{}': {}".format(
                manifest_path, "; ".join(mismatches)
            ),
            file=sys.stderr,
        )
        return 1

    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
