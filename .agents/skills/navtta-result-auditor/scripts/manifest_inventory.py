#!/usr/bin/env python3
"""Read-only inventory and integrity check for NavTTA run manifests."""

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


IDENTITY_FIELD = "immutable_identity_sha256"
IDENTITY_FIELDS = (
    "run_id", "task", "benchmark", "model", "method", "run_tag",
    "source_setting", "seed", "git_commit", "config", "config_overrides",
    "checkpoint", "auxiliary_checkpoints", "dataset", "pinned_manifests",
    "hardware", "started_at",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
METRIC_NAME = re.compile(r"(?:metric|summary|valid)", re.IGNORECASE)
VALIDATION_NAME = re.compile(r"(?:^|[/_.-])valid(?:ation)?(?:[/_.-]|$)", re.IGNORECASE)


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_IDENTITY_HELPERS = {}


def immutable_digest(document, repo_root):
    """Use the repository's canonical identity helper when it is available."""
    helper_path = (repo_root / "tools/run_manifest_identity.py").resolve()
    helper = _IDENTITY_HELPERS.get(helper_path)
    if helper is None and helper_path.is_file():
        spec = importlib.util.spec_from_file_location(
            "navtta_run_manifest_identity", helper_path
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            helper = module.immutable_identity_sha256
            _IDENTITY_HELPERS[helper_path] = helper
    if helper is not None:
        return helper(document), str(helper_path)

    payload = {field: document.get(field) for field in IDENTITY_FIELDS}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return digest_bytes(encoded), "bundled-fallback"


def parse_path_maps(values):
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError("--path-map must use REMOTE=LOCAL")
        old, new = value.split("=", 1)
        if not old or not new:
            raise ValueError("--path-map must use non-empty REMOTE=LOCAL")
        mappings.append((old.rstrip("/"), Path(new).resolve()))
    return mappings


def resolve_reference(raw, manifest_path, repo_root, mappings):
    if not isinstance(raw, str) or not raw:
        return None
    for old, new in mappings:
        if raw == old or raw.startswith(old + "/"):
            suffix = raw[len(old):].lstrip("/")
            return new / suffix
    path = Path(raw)
    if path.is_absolute():
        return path
    by_manifest = manifest_path.parent / path
    if by_manifest.exists():
        return by_manifest
    return repo_root / path


def valid_metadata(item, hash_key="sha256"):
    return (
        isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and bool(item.get("path"))
        and isinstance(item.get("size"), int)
        and not isinstance(item.get("size"), bool)
        and item.get("size") >= 0
        and isinstance(item.get(hash_key), str)
        and bool(SHA256.fullmatch(item.get(hash_key)))
    )


def file_issue(label, item, hash_key, manifest_path, repo_root, mappings):
    if not valid_metadata(item, hash_key):
        return "{} metadata invalid".format(label)
    path = resolve_reference(item["path"], manifest_path, repo_root, mappings)
    if path is None or not path.is_file():
        return "{} missing".format(label)
    if path.stat().st_size != item["size"]:
        return "{} size mismatch".format(label)
    if digest_file(path) != item[hash_key]:
        return "{} SHA256 mismatch".format(label)
    return None


def canonical_path(path, task):
    parts = path.resolve().parts
    needle = (task, "results", "runs")
    return any(tuple(parts[index:index + 3]) == needle for index in range(len(parts) - 2))


def nested_value(document, dotted):
    value = document
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def parse_time(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def audit_manifest(path, verify, repo_root, mappings, expectations):
    record = {
        "path": str(path.resolve()), "manifest_sha256": None,
        "run_id": None, "task": None, "model": None, "method": None,
        "status": None, "identity": "fail", "verification": verify,
        "evidence_level": "process", "issues": [],
    }
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        record["issues"].append("invalid manifest: {}".format(error))
        return record
    if not isinstance(document, dict):
        record["issues"].append("manifest root is not an object")
        return record

    record["manifest_sha256"] = digest_bytes(raw)
    for key in ("run_id", "task", "model", "method", "status"):
        record[key] = document.get(key)

    required_strings = (
        "run_id", "task", "benchmark", "model", "method", "run_tag",
        "source_setting", "git_commit", "config", "started_at", "completed_at",
    )
    for key in required_strings:
        if not isinstance(document.get(key), str) or not document.get(key):
            record["issues"].append("{} missing".format(key))
    if document.get("task") not in {"avn", "vln", "objectnav"}:
        record["issues"].append("task invalid")
    if not isinstance(document.get("seed"), int) or isinstance(document.get("seed"), bool):
        record["issues"].append("seed invalid")
    if not isinstance(document.get("git_commit"), str) or not COMMIT.fullmatch(document.get("git_commit", "")):
        record["issues"].append("top-level git commit invalid")
    if document.get("status") != "completed" or document.get("exit_code") != 0:
        record["issues"].append("run not successfully completed")
    started = parse_time(document.get("started_at"))
    completed = parse_time(document.get("completed_at"))
    try:
        timestamps_valid = started is not None and completed is not None and completed >= started
    except TypeError:
        timestamps_valid = False
    if not timestamps_valid:
        record["issues"].append("run timestamps invalid")
    if not isinstance(document.get("config_overrides"), list):
        record["issues"].append("config_overrides invalid")
    if not isinstance(document.get("hardware"), dict) or not document.get("hardware"):
        record["issues"].append("hardware missing")
    if not valid_metadata(document.get("checkpoint")):
        record["issues"].append("checkpoint metadata invalid")
    if not valid_metadata(document.get("dataset"), "index_sha256") or not document.get("dataset", {}).get("version"):
        record["issues"].append("dataset metadata/version invalid")
    dataset = document.get("dataset") if isinstance(document.get("dataset"), dict) else {}
    for key in ("stream_order_sha256", "stream_content_sha256"):
        if not isinstance(dataset.get(key), str) or not SHA256.fullmatch(dataset.get(key, "")):
            record["issues"].append("dataset.{} invalid".format(key))
    auxiliary = document.get("auxiliary_checkpoints")
    if not isinstance(auxiliary, list):
        record["issues"].append("auxiliary_checkpoints invalid")
        auxiliary = []
    else:
        auxiliary_names = set()
        for index, item in enumerate(auxiliary):
            name = item.get("name") if isinstance(item, dict) else None
            if (
                not isinstance(name, str) or not name or name in auxiliary_names
                or not valid_metadata(item)
            ):
                record["issues"].append(
                    "auxiliary checkpoint metadata/name invalid at {}".format(index)
                )
            else:
                auxiliary_names.add(name)
    pinned = document.get("pinned_manifests")
    if not isinstance(pinned, dict):
        record["issues"].append("pinned_manifests invalid")
        pinned = {}
    else:
        for name, item in pinned.items():
            if not isinstance(name, str) or not name or not valid_metadata(item):
                record["issues"].append(
                    "pinned manifest metadata invalid for {}".format(name)
                )

    recorded = document.get(IDENTITY_FIELD)
    try:
        actual, identity_source = immutable_digest(document, repo_root)
        record["identity_schema_source"] = identity_source
    except (TypeError, ValueError) as error:
        actual = None
        record["identity_schema_source"] = None
        record["issues"].append("immutable identity cannot be computed: {}".format(error))
    if not isinstance(recorded, str) or not SHA256.fullmatch(recorded):
        record["issues"].append("immutable identity missing/invalid")
    elif actual != recorded:
        record["issues"].append("immutable identity mismatch")
    else:
        record["identity"] = "pass"

    artifacts = document.get("result_artifacts")
    artifact_metadata_ok = isinstance(artifacts, list) and bool(artifacts)
    names = set()
    metrics_present = False
    validation_present = False
    if not artifact_metadata_ok:
        record["issues"].append("result_artifacts missing/empty")
        artifacts = []
    for index, item in enumerate(artifacts):
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name or name in names:
            artifact_metadata_ok = False
            record["issues"].append("result artifact name missing/duplicated at {}".format(index))
        else:
            names.add(name)
            metrics_present = metrics_present or bool(METRIC_NAME.search(name))
            validation_present = validation_present or bool(
                VALIDATION_NAME.search(name)
            )
        if not valid_metadata(item):
            artifact_metadata_ok = False
            record["issues"].append("result artifact metadata invalid at {}".format(index))
    record["metrics_declared"] = metrics_present
    record["runner_validation_declared"] = validation_present
    if not metrics_present:
        record["issues"].append("metric or validation artifact is not declared")
    if not validation_present:
        record["issues"].append("runner validation artifact is not declared")

    for key, expected in expectations.items():
        if nested_value(document, key) != expected:
            record["issues"].append("expected {}={!r}".format(key, expected))

    task = document.get("task") if isinstance(document.get("task"), str) else ""
    if "legacy" in path.parts:
        record["issues"].append("legacy path is not formal")
    elif task and not canonical_path(path, task):
        record["issues"].append("manifest path is not canonical task/results/runs")

    base_ok = not record["issues"] and artifact_metadata_ok
    selected = []
    if verify == "all":
        selected.append(("checkpoint", document.get("checkpoint"), "sha256"))
        for index, item in enumerate(auxiliary):
            selected.append(("auxiliary checkpoint {}".format(index), item, "sha256"))
        selected.append(("dataset", document.get("dataset"), "index_sha256"))
        for name, item in sorted(pinned.items()):
            selected.append(("pinned manifest {}".format(name), item, "sha256"))
    if verify in {"results", "all"}:
        for index, item in enumerate(artifacts):
            name = item.get("name", index) if isinstance(item, dict) else index
            selected.append(("result artifact {}".format(name), item, "sha256"))
    file_issues = []
    for label, item, hash_key in selected:
        issue = file_issue(label, item, hash_key, path, repo_root, mappings)
        if issue:
            file_issues.append(issue)
    record["issues"].extend(file_issues)

    if metrics_present:
        record["evidence_level"] = "metrics"
    if verify in {"results", "all"} and base_ok and not record["issues"]:
        record["evidence_level"] = "validated"
    if verify == "all" and not record["issues"]:
        record["evidence_level"] = "formal"
    return record


def discover(inputs, include_legacy):
    found = set()
    for value in inputs:
        path = Path(value)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("manifest.json")
        else:
            continue
        for candidate in candidates:
            if not include_legacy and "legacy" in candidate.parts:
                continue
            found.add(candidate.resolve())
    return sorted(found)


def parse_expectations(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--expect must use FIELD=JSON_VALUE")
        key, raw = value.split("=", 1)
        if not key:
            raise ValueError("--expect field must be non-empty")
        try:
            result[key] = json.loads(raw)
        except ValueError:
            result[key] = raw
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="manifest file(s) or directories")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--verify", choices=("none", "results", "all"), default="none")
    parser.add_argument("--path-map", action="append", default=[], metavar="REMOTE=LOCAL")
    parser.add_argument("--expect", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit nonzero unless every manifest is formal")
    args = parser.parse_args()
    try:
        mappings = parse_path_maps(args.path_map)
        expectations = parse_expectations(args.expect)
    except ValueError as error:
        parser.error(str(error))

    paths = discover(args.paths, args.include_legacy)
    records = [
        audit_manifest(path, args.verify, args.repo_root.resolve(), mappings, expectations)
        for path in paths
    ]
    by_run_id = {}
    for record in records:
        run_id = record.get("run_id")
        if run_id:
            by_run_id.setdefault(run_id, []).append(record)
    for run_id, duplicates in by_run_id.items():
        identities = {item.get("manifest_sha256") for item in duplicates}
        if len(duplicates) > 1:
            message = "duplicate run_id {} with {} manifest(s)".format(run_id, len(duplicates))
            if len(identities) > 1:
                message += " and differing bytes"
            for item in duplicates:
                item["issues"].append(message)
                if item["evidence_level"] == "formal":
                    item["evidence_level"] = "validated"

    if args.json:
        json.dump({"manifests": records, "count": len(records)}, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print("level\tstatus\tidentity\ttask\tmodel\tmethod\trun_id\tpath\tissues")
        for item in records:
            print("\t".join(
                str(value).replace("\t", " ").replace("\n", " ")
                for value in (
                    item["evidence_level"], item.get("status") or "-", item["identity"],
                    item.get("task") or "-", item.get("model") or "-", item.get("method") or "-",
                    item.get("run_id") or "-", item["path"], "; ".join(item["issues"]) or "-",
                )
            ))
    if args.strict and (not records or any(item["evidence_level"] != "formal" for item in records)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
