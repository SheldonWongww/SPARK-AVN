#!/usr/bin/env python3
"""Build the two-split R2R final comparison from downloaded evidence.

This is an offline, read-only evidence consumer: it never launches an
experiment and never changes a batch, tuning output, run manifest, registry,
or Source ledger.  Its only writes are the explicitly requested final JSON
and Markdown report.

The val_seen registry is the sole hyperparameter-selection source.  The
val_unseen batch is accepted only when all fifteen jobs use those parameters
verbatim and their metrics are linked through downloaded ``metrics.json``
records to digest-pinned formal manifests.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


RESULT_SCHEMA = "navtta.vln_r2r_benchmark_final_results.v1"
REGISTRY_SCHEMA = "navtta.vln_r2r_final_registry.v1"
SPEC_SCHEMA = "navtta.vln_r2r_val_unseen_frozen_eval.v1"
SOURCE_SCHEMA = "navtta.vln_r2r_val_unseen_reused_source_controls.v1"
BATCH_SCHEMA = "navtta.vln_r2r_val_unseen_frozen_batch.v1"
SUMMARY_SCHEMA = "navtta.vln_r2r_val_unseen_frozen_summary.v1"
JOB_SCHEMA = "navtta.vln_r2r_val_unseen_frozen_job.v1"

DEFAULT_REGISTRY = REPO_ROOT / "vln/results/final/r2r/registry.json"
DEFAULT_SPEC = REPO_ROOT / "vln/experiments/r2r_val_unseen_frozen_eval_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "vln/results/final/r2r/benchmark_results.json"
DEFAULT_REPORT = REPO_ROOT / "vln/results/final/r2r/BENCHMARK_RESULTS.md"

SETTINGS = ("duet-r2r", "hamt-r2r", "goat-r2r")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
ALL_METHODS = ("source",) + METHODS
MODEL_FOR_SETTING = {
    "duet-r2r": "duet",
    "hamt-r2r": "hamt",
    "goat-r2r": "goat",
}
BENCHMARK_FOR_SETTING = {
    "duet-r2r": "r2r_discrete_duet_hamt",
    "hamt-r2r": "r2r_discrete_duet_hamt",
    "goat-r2r": "r2r_discrete_goat",
}
SUPERVISION_FOR_METHOD = {
    "source": "source_no_adaptation",
    "tent": "unsupervised_tta",
    "fstta": "unsupervised_tta",
    "eam": "unsupervised_tta",
    "feedtta": "binary_episode_feedback_tta",
    "atena": "binary_episode_feedback_tta",
}
FEEDBACK_ENDPOINT = {
    "feedtta": "r2r_submitted_trajectory_evaluator_success_every_episode",
    "atena": "r2r_submitted_trajectory_evaluator_success_lazy_query",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
METRIC_RE = re.compile(r"\b(sr|spl):\s*(-?[0-9]+(?:\.[0-9]+)?)")


class ArchiveError(RuntimeError):
    """Downloaded evidence is incomplete, inconsistent, or unauthenticated."""


def _read_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArchiveError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise ArchiveError("{} must be a JSON object: {}".format(label, path))
    return value


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def _resolve(repo_root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def _display_path(repo_root, path):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _require_file(path, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise ArchiveError("missing {}: {}".format(label, path))
    return path


def _metrics(value, label):
    if not isinstance(value, dict):
        raise ArchiveError("{} metrics must be an object".format(label))
    result = {}
    for name in ("SR", "SPL"):
        number = value.get(name)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or not 0.0 <= float(number) <= 100.0
        ):
            raise ArchiveError("{} {} is invalid".format(label, name))
        result[name] = number
    return result


def _same(left, right):
    return _canonical(left) == _canonical(right)


def _delta(metrics, source):
    return {
        name: round(float(metrics[name]) - float(source[name]), 8)
        for name in ("SR", "SPL")
    }


def _formal_path(repo_root, declared, expected_run_id, label):
    relative = Path(
        "vln/results/runs/{}/manifest.json".format(expected_run_id)
    ).as_posix()
    normalized = str(declared or "").replace("\\", "/").rstrip("/")
    if normalized != relative and not normalized.endswith("/" + relative):
        raise ArchiveError("{} formal-manifest path is noncanonical".format(label))
    return _require_file(Path(repo_root) / relative, "{} formal manifest".format(label))


def _artifact_index(manifest, label):
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ArchiveError("{} formal manifest has no result artifacts".format(label))
    by_sha = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ArchiveError("{} has a malformed result artifact".format(label))
        name = artifact.get("name")
        size = artifact.get("size")
        digest = artifact.get("sha256")
        path = str(artifact.get("path", "")).replace("\\", "/")
        if not isinstance(name, str) or not name:
            raise ArchiveError("{} result artifact lacks a name".format(label))
        if path != name and not path.endswith("/" + name):
            raise ArchiveError("{} artifact path/name mismatch".format(label))
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ArchiveError("{} artifact size is invalid".format(label))
        if not _is_sha256(digest) or digest in by_sha:
            raise ArchiveError("{} artifact digest is invalid/duplicate".format(label))
        by_sha[digest] = artifact
    return by_sha


def _validate_manifest(
    repo_root,
    declared_path,
    declared_sha256,
    setting,
    method,
    split,
    run_tag,
    checkpoint_sha256,
    order_sha256,
    dataset_sha256=None,
    immutable_sha256=None,
    git_commit=None,
):
    label = "{} {} {}".format(setting, method, split)
    run_id = "{}-{}-{}-native".format(run_tag, setting, split)
    path = _formal_path(repo_root, declared_path, run_id, label)
    if not _is_sha256(declared_sha256) or _sha256(path) != declared_sha256:
        raise ArchiveError("{} formal-manifest SHA256 mismatch".format(label))
    manifest = _read_json(path, "{} formal manifest".format(label))
    expected = {
        "run_id": run_id,
        "task": "vln",
        "benchmark": BENCHMARK_FOR_SETTING[setting],
        "model": MODEL_FOR_SETTING[setting],
        "method": method,
        "run_tag": run_tag,
        "source_setting": "{}:{}:native{}".format(
            setting,
            split,
            "" if method == "source" and split == "val_unseen" else ":" + method,
        ),
        "seed": 0,
        "status": "completed",
        "exit_code": 0,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ArchiveError("{} manifest {} mismatch".format(label, key))
    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or HEX_COMMIT.fullmatch(commit) is None:
        raise ArchiveError("{} manifest git commit is invalid".format(label))
    if git_commit is not None and commit != git_commit:
        raise ArchiveError("{} manifest git commit changed".format(label))
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ArchiveError("{} checkpoint digest mismatch".format(label))
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get(
        "stream_order_sha256"
    ) != order_sha256:
        raise ArchiveError("{} episode-order digest mismatch".format(label))
    if dataset_sha256 is not None and dataset.get(
        "stream_content_sha256"
    ) != dataset_sha256:
        raise ArchiveError("{} dataset digest mismatch".format(label))
    identity = manifest.get("immutable_identity_sha256")
    if not _is_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise ArchiveError("{} immutable identity mismatch".format(label))
    if immutable_sha256 is not None and identity != immutable_sha256:
        raise ArchiveError("{} pinned immutable identity mismatch".format(label))
    artifacts = _artifact_index(manifest, label)
    provenance = {
        "run_tag": run_tag,
        "git_commit": commit,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset.get("stream_content_sha256"),
        "episode_order_sha256": dataset.get("stream_order_sha256"),
        "formal_manifest_path": _display_path(repo_root, path),
        "formal_manifest_sha256": declared_sha256,
        "formal_immutable_identity_sha256": identity,
    }
    return manifest, artifacts, provenance


def _load_registry(repo_root, registry_path):
    registry_path = _require_file(registry_path, "R2R val_seen registry")
    registry = _read_json(registry_path, "R2R val_seen registry")
    expected = {
        "schema": REGISTRY_SCHEMA,
        "registry_status": "complete",
        "benchmark": "r2r",
        "split": "val_seen",
    }
    for key, value in expected.items():
        if registry.get(key) != value:
            raise ArchiveError("val_seen registry {} mismatch".format(key))
    protocol = registry.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("episode_count") != 1021
        or protocol.get("order_seed") != 0
        or not _is_sha256(protocol.get("episode_order_sha256"))
    ):
        raise ArchiveError("val_seen registry protocol is invalid")
    records = registry.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise ArchiveError("val_seen registry setting matrix is incomplete")

    archived = {}
    for setting in SETTINGS:
        methods = records[setting]
        if not isinstance(methods, dict) or set(methods) != set(ALL_METHODS):
            raise ArchiveError("{} val_seen method matrix is incomplete".format(setting))
        source_metrics = _metrics(methods["source"].get("metrics"), setting + " source")
        archived[setting] = {}
        for method in ALL_METHODS:
            label = "{} {}".format(setting, method)
            record = methods[method]
            if not isinstance(record, dict):
                raise ArchiveError("{} val_seen record is malformed".format(label))
            if record.get("method") != method or record.get(
                "supervision_category"
            ) != SUPERVISION_FOR_METHOD[method]:
                raise ArchiveError("{} val_seen method/supervision mismatch".format(label))
            parameters = record.get("parameters")
            if not isinstance(parameters, dict) or not parameters:
                raise ArchiveError("{} frozen parameters are invalid".format(label))
            metrics = _metrics(record.get("metrics"), label)
            if record.get("delta_vs_source_pp") != _delta(metrics, source_metrics):
                raise ArchiveError("{} val_seen Source delta mismatch".format(label))
            checkpoint = record.get("checkpoint_sha256")
            if not _is_sha256(checkpoint):
                raise ArchiveError("{} checkpoint digest is invalid".format(label))
            _, _, provenance = _validate_manifest(
                repo_root,
                record.get("formal_manifest_path"),
                record.get("formal_manifest_sha256"),
                setting,
                method,
                "val_seen",
                record.get("run_tag"),
                checkpoint,
                protocol["episode_order_sha256"],
                immutable_sha256=record.get("formal_immutable_identity_sha256"),
            )
            archived[setting][method] = {
                "parameters": parameters,
                "metrics": metrics,
                "delta_vs_source_pp": record["delta_vs_source_pp"],
                "provenance": provenance,
            }
    return registry_path, registry, archived


def _load_spec(repo_root, spec_path, registry_path, registry):
    spec_path = _require_file(spec_path, "R2R val_unseen specification")
    spec = _read_json(spec_path, "R2R val_unseen specification")
    if spec.get("schema") != SPEC_SCHEMA:
        raise ArchiveError("unsupported R2R val_unseen specification")
    protocol = spec.get("protocol")
    expected_protocol = {
        "benchmark": "r2r",
        "split": "val_unseen",
        "episode_count": 2349,
        "canonical_order_seed": 0,
        "full_split_only": True,
        "selection_on_val_unseen": False,
        "report_all_cells": True,
    }
    if not isinstance(protocol, dict):
        raise ArchiveError("val_unseen protocol is missing")
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise ArchiveError("val_unseen protocol {} mismatch".format(key))
    if not _is_sha256(protocol.get("episode_order_sha256")):
        raise ArchiveError("val_unseen order digest is invalid")
    order_manifests = protocol.get("order_manifests")
    if not isinstance(order_manifests, dict) or set(order_manifests) != {
        "duet_hamt", "goat"
    }:
        raise ArchiveError("val_unseen order-manifest bindings are incomplete")
    for name, binding in order_manifests.items():
        if not isinstance(binding, dict):
            raise ArchiveError("{} order binding is malformed".format(name))
        path = _require_file(
            _resolve(repo_root, binding.get("path", "")),
            "{} episode-order manifest".format(name),
        )
        if not _is_sha256(binding.get("sha256")) or _sha256(path) != binding["sha256"]:
            raise ArchiveError("{} episode-order manifest digest mismatch".format(name))
        document = _read_json(path, "{} episode-order manifest".format(name))
        if (
            document.get("split") != "val_unseen"
            or document.get("episode_count") != 2349
            or document.get("order_sha256") != protocol["episode_order_sha256"]
        ):
            raise ArchiveError("{} episode-order manifest identity mismatch".format(name))

    dependency = spec.get("registry_dependency")
    if (
        not isinstance(dependency, dict)
        or dependency.get("schema") != REGISTRY_SCHEMA
        or _resolve(repo_root, dependency.get("path", "")) != registry_path
        or dependency.get("sha256") != _sha256(registry_path)
    ):
        raise ArchiveError("val_unseen specification registry binding mismatch")
    matrix = spec.get("matrix")
    if (
        not isinstance(matrix, dict)
        or tuple(matrix.get("setting_order", ())) != SETTINGS
        or tuple(matrix.get("method_order", ())) != METHODS
        or matrix.get("strict_model_barrier") is not True
    ):
        raise ArchiveError("val_unseen frozen matrix identity mismatch")
    jobs = matrix.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 15:
        raise ArchiveError("val_unseen specification must contain 15 TTA jobs")
    pairs = [(job.get("setting"), job.get("method")) for job in jobs]
    expected_pairs = [(setting, method) for setting in SETTINGS for method in METHODS]
    if pairs != expected_pairs:
        raise ArchiveError("val_unseen specification is not the model-major 3x5 matrix")
    for job in jobs:
        setting = job["setting"]
        method = job["method"]
        selected = registry["records"][setting][method]
        if job.get("model") != MODEL_FOR_SETTING[setting]:
            raise ArchiveError("val_unseen specification model binding mismatch")
        if (
            job.get("selected_run_tag") != selected["run_tag"]
            or job.get("selected_formal_manifest_sha256")
            != selected["formal_manifest_sha256"]
        ):
            raise ArchiveError(
                "val_unseen specification selection binding mismatch for {} {}"
                .format(setting, method)
            )
    return spec_path, spec


def _load_unseen_sources(repo_root, source_path, spec, registry, seen_archive):
    source_path = _require_file(source_path, "R2R val_unseen Source ledger")
    ledger = _read_json(source_path, "R2R val_unseen Source ledger")
    expected = {
        "schema": SOURCE_SCHEMA,
        "benchmark": "r2r",
        "split": "val_unseen",
        "source_protocol": "standard_argmax",
        "episode_count": 2349,
        "canonical_order_seed": 0,
        "episode_order_sha256": spec["protocol"]["episode_order_sha256"],
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise ArchiveError("val_unseen Source ledger {} mismatch".format(key))
    policy = ledger.get("source_execution_policy", {})
    if policy.get("execution") != "reuse_only" or policy.get(
        "rerun_forbidden"
    ) is not True:
        raise ArchiveError("val_unseen Source ledger is not reuse-only")
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise ArchiveError("val_unseen Source ledger setting matrix is incomplete")
    archived = {}
    for setting in SETTINGS:
        label = setting + " source"
        record = records[setting]
        if record.get("evidence_status") != "ready":
            raise ArchiveError("{} evidence is not ready".format(label))
        if record.get("model") != MODEL_FOR_SETTING[setting]:
            raise ArchiveError("{} model mismatch".format(label))
        if not _same(record.get("parameters"), seen_archive[setting]["source"]["parameters"]):
            raise ArchiveError("{} protocol changed across splits".format(label))
        checkpoint = record.get("checkpoint_sha256")
        if checkpoint != registry["records"][setting]["source"].get(
            "checkpoint_sha256"
        ):
            raise ArchiveError("{} checkpoint changed across splits".format(label))
        metrics = _metrics(record.get("metrics"), label)
        _, artifacts, provenance = _validate_manifest(
            repo_root,
            record.get("formal_manifest_path"),
            record.get("formal_manifest_sha256"),
            setting,
            "source",
            "val_unseen",
            record.get("run_tag"),
            checkpoint,
            spec["protocol"]["episode_order_sha256"],
            dataset_sha256=record.get("dataset_sha256"),
            immutable_sha256=record.get("immutable_identity_sha256"),
        )
        metric_digest = record.get("metrics_artifact_sha256")
        if not _is_sha256(metric_digest) or metric_digest not in artifacts:
            raise ArchiveError("{} metrics are not formal-manifest authenticated".format(label))
        archived[setting] = {
            "metrics": metrics,
            "delta_vs_source_pp": {"SR": 0.0, "SPL": 0.0},
            "provenance": provenance,
        }
    return source_path, ledger, archived


def _load_batch(batch_root, spec_path, spec, registry_path, source_path):
    batch_root = Path(batch_root).resolve()
    batch_path = _require_file(batch_root / "BATCH.json", "downloaded batch identity")
    summary_path = _require_file(batch_root / "SUMMARY.json", "downloaded batch summary")
    batch = _read_json(batch_path, "downloaded batch identity")
    summary = _read_json(summary_path, "downloaded batch summary")
    expected_batch = {
        "schema": BATCH_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "spec_sha256": _sha256(spec_path),
        "source_execution_jobs": 0,
        "tta_jobs": 15,
    }
    for key, value in expected_batch.items():
        if batch.get(key) != value:
            raise ArchiveError("downloaded batch {} mismatch".format(key))
    if batch.get("registry") != spec["registry_dependency"]:
        raise ArchiveError("downloaded batch registry binding mismatch")
    if batch.get("source_ledger") != {
        "path": spec["source_control"]["manifest"],
        "sha256": spec["source_control"]["sha256"],
    }:
        raise ArchiveError("downloaded batch Source-ledger binding mismatch")
    if _sha256(registry_path) != batch["registry"]["sha256"]:
        raise ArchiveError("downloaded batch registry digest mismatch")
    if _sha256(source_path) != batch["source_ledger"]["sha256"]:
        raise ArchiveError("downloaded batch Source-ledger digest mismatch")

    expected_summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "source_execution_jobs": 0,
        "total_jobs": 15,
        "complete": True,
        "source_evidence_blockers": [],
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ArchiveError("downloaded summary {} mismatch".format(key))
    states = summary.get("states")
    if not isinstance(states, dict) or states.get("completed") != 15 or any(
        value != 0 for key, value in states.items() if key != "completed"
    ):
        raise ArchiveError("downloaded summary is not a clean 15/15 completion")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 15:
        raise ArchiveError("downloaded summary must contain exactly 15 results")
    pairs = [(row.get("setting"), row.get("method")) for row in results]
    expected_pairs = [(setting, method) for setting in SETTINGS for method in METHODS]
    if pairs != expected_pairs:
        raise ArchiveError("downloaded summary is not the model-major 3x5 matrix")

    metric_rows = {}
    for path in sorted(batch_root.rglob("metrics.json")):
        value = _read_json(path, "downloaded attempt metrics")
        if value.get("schema") != JOB_SCHEMA:
            continue
        tag = value.get("run_tag")
        if not isinstance(tag, str) or not tag or tag in metric_rows:
            raise ArchiveError("duplicate/invalid downloaded attempt run tag")
        metric_rows[tag] = (path, value)
    if set(metric_rows) != {row.get("run_tag") for row in results}:
        raise ArchiveError("downloaded attempt metrics do not exactly cover SUMMARY.json")
    return batch_root, batch_path, batch, summary_path, summary, metric_rows


def _validate_job_config(path, row, spec, registry_record):
    config = _read_json(path, "downloaded frozen job configuration")
    expected = {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": row["batch_id"],
        "stage": "frozen_val_unseen",
        "setting": row["setting"],
        "method": row["method"],
        "search_method": row["method"],
        "episodes": -1,
        "parameters": registry_record["parameters"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ArchiveError("{} configuration {} mismatch".format(row["run_tag"], key))
    if "order_seed" in config:
        raise ArchiveError("{} configuration illegally overrides order_seed".format(row["run_tag"]))
    provenance = config.get("frozen_evaluation_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("selection_split") != "val_seen"
        or provenance.get("evaluation_split") != "val_unseen"
        or provenance.get("selection_on_val_unseen") is not False
        or provenance.get("canonical_order_seed") != 0
        or provenance.get("registry") != spec["registry_dependency"]
        or provenance.get("selected_anchor") != registry_record
    ):
        raise ArchiveError("{} frozen-selection provenance mismatch".format(row["run_tag"]))
    return config


def _archive_unseen_tta(
    repo_root,
    spec,
    registry,
    seen_archive,
    unseen_sources,
    batch,
    summary,
    metric_rows,
):
    archived = {setting: {} for setting in SETTINGS}
    for summary_row in summary["results"]:
        setting = summary_row["setting"]
        method = summary_row["method"]
        label = "{} {}".format(setting, method)
        registry_record = registry["records"][setting][method]
        if not _same(summary_row.get("parameters"), registry_record["parameters"]):
            raise ArchiveError("{} val_unseen parameters drifted from val_seen".format(label))
        metrics = _metrics(summary_row.get("metrics"), label)
        metric_path, row = metric_rows[summary_row["run_tag"]]
        expected_job = {
            "schema": JOB_SCHEMA,
            "batch_id": batch["batch_id"],
            "spec_sha256": batch["spec_sha256"],
            "setting": setting,
            "model": MODEL_FOR_SETTING[setting],
            "method": method,
            "parameters": registry_record["parameters"],
            "selected_anchor": registry_record,
            "episode_count": 2349,
            "canonical_order_seed": 0,
            "expected_checkpoint_sha256": registry["records"][setting]["source"][
                "checkpoint_sha256"
            ],
            "expected_dataset_sha256": unseen_sources[setting]["provenance"][
                "dataset_sha256"
            ],
            "expected_episode_order_sha256": spec["protocol"][
                "episode_order_sha256"
            ],
            "run_tag": summary_row["run_tag"],
            "metrics": metrics,
            "formal_manifest_sha256": summary_row.get("formal_manifest_sha256"),
        }
        for key, value in expected_job.items():
            if not _same(row.get(key), value):
                raise ArchiveError("{} downloaded metrics {} mismatch".format(label, key))
        if row.get("formal_manifest") != summary_row.get("formal_manifest"):
            raise ArchiveError("{} formal-manifest path differs between outputs".format(label))
        _validate_job_config(metric_path.parent / "parameters.json", row, spec, registry_record)
        manifest, artifacts, provenance = _validate_manifest(
            repo_root,
            row.get("formal_manifest"),
            row.get("formal_manifest_sha256"),
            setting,
            method,
            "val_unseen",
            row["run_tag"],
            row["expected_checkpoint_sha256"],
            row["expected_episode_order_sha256"],
            dataset_sha256=row["expected_dataset_sha256"],
            immutable_sha256=row.get("formal_immutable_identity_sha256"),
            git_commit=row.get("git_commit"),
        )
        for field in ("metric_artifact_sha256", "diagnostics_sha256"):
            digest = row.get(field)
            if not _is_sha256(digest) or digest not in artifacts:
                raise ArchiveError("{} {} is not manifest-authenticated".format(label, field))
        parsed = {
            key.upper(): float(value)
            for key, value in METRIC_RE.findall(str(row.get("metric_line", "")))
        }
        if set(parsed) != {"SR", "SPL"} or any(
            not math.isclose(parsed[name], float(metrics[name]), abs_tol=1e-9)
            for name in ("SR", "SPL")
        ) or "Env name: val_unseen" not in row.get("metric_line", ""):
            raise ArchiveError("{} metric line does not match structured metrics".format(label))
        expected_endpoint = FEEDBACK_ENDPOINT.get(method)
        if row.get("feedback_endpoint") != expected_endpoint:
            raise ArchiveError("{} supervision endpoint mismatch".format(label))
        if manifest.get("git_commit") != provenance["git_commit"]:
            raise ArchiveError("{} provenance commit mismatch".format(label))
        archived[setting][method] = {
            "metrics": metrics,
            "delta_vs_source_pp": _delta(metrics, unseen_sources[setting]["metrics"]),
            "provenance": provenance,
        }
    return archived


def build_results(
    batch_root,
    repo_root=REPO_ROOT,
    registry_path=None,
    spec_path=None,
):
    """Return a deterministic final-results document without writing inputs."""
    repo_root = Path(repo_root).resolve()
    registry_path = _resolve(
        repo_root,
        registry_path or "vln/results/final/r2r/registry.json",
    )
    spec_path = _resolve(
        repo_root,
        spec_path or "vln/experiments/r2r_val_unseen_frozen_eval_v1.json",
    )
    registry_path, registry, seen_archive = _load_registry(
        repo_root, registry_path
    )
    spec_path, spec = _load_spec(
        repo_root, spec_path, registry_path, registry
    )
    source_binding = spec.get("source_control")
    if not isinstance(source_binding, dict):
        raise ArchiveError("val_unseen specification lacks a Source binding")
    source_path = _resolve(repo_root, source_binding.get("manifest", ""))
    if source_binding.get("sha256") != _sha256(_require_file(source_path, "Source ledger")):
        raise ArchiveError("val_unseen specification Source-ledger digest mismatch")
    source_path, source_ledger, unseen_sources = _load_unseen_sources(
        repo_root, source_path, spec, registry, seen_archive
    )
    (
        batch_root,
        batch_path,
        batch,
        summary_path,
        summary,
        metric_rows,
    ) = _load_batch(batch_root, spec_path, spec, registry_path, source_path)
    unseen_tta = _archive_unseen_tta(
        repo_root,
        spec,
        registry,
        seen_archive,
        unseen_sources,
        batch,
        summary,
        metric_rows,
    )

    records = {}
    for setting in SETTINGS:
        methods = {}
        for method in ALL_METHODS:
            unseen = (
                unseen_sources[setting]
                if method == "source"
                else unseen_tta[setting][method]
            )
            methods[method] = {
                "method": method,
                "supervision_category": SUPERVISION_FOR_METHOD[method],
                "frozen_parameters": seen_archive[setting][method]["parameters"],
                "val_seen": {
                    key: seen_archive[setting][method][key]
                    for key in ("metrics", "delta_vs_source_pp", "provenance")
                },
                "val_unseen": unseen,
            }
        records[setting] = {
            "model": MODEL_FOR_SETTING[setting],
            "methods": methods,
        }

    result = {
        "schema": RESULT_SCHEMA,
        "result_status": "complete_val_seen_selection_and_val_unseen_frozen_evaluation",
        "benchmark": "r2r",
        "metric_unit": "percentage_points",
        "selection_policy": {
            "selection_split": "val_seen",
            "evaluation_split": "val_unseen",
            "selection_on_val_unseen": False,
            "order_seed": 0,
            "qualification": (
                "val_seen selected each model-method configuration; val_unseen "
                "evaluated the frozen configuration once and did not reselect it."
            ),
        },
        "protocols": {
            "val_seen": registry["protocol"],
            "val_unseen": {
                "episode_count": spec["protocol"]["episode_count"],
                "order_seed": spec["protocol"]["canonical_order_seed"],
                "episode_order_sha256": spec["protocol"]["episode_order_sha256"],
                "source_protocol": source_ledger["source_protocol"],
                "full_split_only": True,
            },
        },
        "inputs": {
            "val_seen_registry": {
                "path": _display_path(repo_root, registry_path),
                "sha256": _sha256(registry_path),
            },
            "val_unseen_specification": {
                "path": _display_path(repo_root, spec_path),
                "sha256": _sha256(spec_path),
            },
            "val_unseen_source_ledger": {
                "path": _display_path(repo_root, source_path),
                "sha256": _sha256(source_path),
            },
            "val_unseen_batch": {
                "batch_id": batch["batch_id"],
                "batch_path": _display_path(repo_root, batch_path),
                "batch_sha256": _sha256(batch_path),
                "summary_path": _display_path(repo_root, summary_path),
                "summary_sha256": _sha256(summary_path),
            },
        },
        "supervision_categories": registry["supervision_categories"],
        "records": records,
    }
    validate_results_document(result)
    return result


def validate_results_document(document):
    if not isinstance(document, dict) or document.get("schema") != RESULT_SCHEMA:
        raise ArchiveError("unsupported R2R benchmark-results schema")
    if document.get("benchmark") != "r2r" or document.get(
        "result_status"
    ) != "complete_val_seen_selection_and_val_unseen_frozen_evaluation":
        raise ArchiveError("R2R benchmark-results status is incomplete")
    policy = document.get("selection_policy", {})
    if (
        policy.get("selection_split") != "val_seen"
        or policy.get("evaluation_split") != "val_unseen"
        or policy.get("selection_on_val_unseen") is not False
        or policy.get("order_seed") != 0
    ):
        raise ArchiveError("R2R benchmark-results selection policy is invalid")
    records = document.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise ArchiveError("R2R benchmark-results setting matrix is incomplete")
    for setting in SETTINGS:
        methods = records[setting].get("methods")
        if not isinstance(methods, dict) or set(methods) != set(ALL_METHODS):
            raise ArchiveError("{} benchmark-results method matrix is incomplete".format(setting))
        for split in ("val_seen", "val_unseen"):
            source = _metrics(methods["source"][split].get("metrics"), setting + " source")
            for method in ALL_METHODS:
                cell = methods[method]
                if cell.get("method") != method or cell.get(
                    "supervision_category"
                ) != SUPERVISION_FOR_METHOD[method]:
                    raise ArchiveError("{} {} method metadata mismatch".format(setting, method))
                if not isinstance(cell.get("frozen_parameters"), dict):
                    raise ArchiveError("{} {} lacks frozen parameters".format(setting, method))
                result = cell.get(split)
                metrics = _metrics(result.get("metrics"), "{} {} {}".format(setting, method, split))
                if result.get("delta_vs_source_pp") != _delta(metrics, source):
                    raise ArchiveError("{} {} {} Source delta mismatch".format(setting, method, split))
                provenance = result.get("provenance")
                required = {
                    "run_tag", "git_commit", "checkpoint_sha256",
                    "dataset_sha256", "episode_order_sha256",
                    "formal_manifest_path", "formal_manifest_sha256",
                    "formal_immutable_identity_sha256",
                }
                if not isinstance(provenance, dict) or set(provenance) != required:
                    raise ArchiveError("{} {} {} provenance is incomplete".format(setting, method, split))
    return document


def _fmt_metrics(value):
    return "{:.2f} / {:.2f}".format(float(value["SR"]), float(value["SPL"]))


def _fmt_delta(value):
    return "{:+.2f} / {:+.2f}".format(float(value["SR"]), float(value["SPL"]))


def render_report(document):
    validate_results_document(document)
    lines = [
        "# R2R final benchmark results",
        "",
        "These results use order seed 0. Hyperparameters were selected on "
        "`val_seen`; `val_unseen` evaluates the frozen configuration once and "
        "was not used for reselection.",
        "",
        "FeedTTA and ATENA consume binary episode-success feedback. Tent, "
        "FSTTA, and EAM are unsupervised TTA.",
        "",
        "## Main comparison",
        "",
        "Values and deltas are `SR / SPL` in percentage points.",
        "",
        "| Model | Method | Supervision | val_seen | val_unseen | val_unseen delta vs Source |",
        "|---|---|---|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        model = document["records"][setting]["model"].upper()
        for method in ALL_METHODS:
            cell = document["records"][setting]["methods"][method]
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                model,
                method,
                cell["supervision_category"],
                _fmt_metrics(cell["val_seen"]["metrics"]),
                _fmt_metrics(cell["val_unseen"]["metrics"]),
                _fmt_delta(cell["val_unseen"]["delta_vs_source_pp"]),
            ))

    lines.extend([
        "",
        "## Frozen parameters",
        "",
        "Source uses the standard deterministic argmax protocol. TTA parameter "
        "objects below are copied verbatim from the `val_seen` registry.",
        "",
        "| Model | Method | Parameters |",
        "|---|---|---|",
    ])
    for setting in SETTINGS:
        model = document["records"][setting]["model"].upper()
        for method in ALL_METHODS:
            parameters = document["records"][setting]["methods"][method][
                "frozen_parameters"
            ]
            lines.append("| {} | {} | `{}` |".format(
                model, method, _canonical(parameters)
            ))

    lines.extend([
        "",
        "## Provenance",
        "",
        "The machine-readable companion file contains the complete input "
        "digests, dataset/order/checkpoint identities, and immutable formal-run "
        "identities. The table below identifies every formal run.",
        "",
        "| Model | Method | Split | Run tag | Git commit | Formal manifest SHA256 |",
        "|---|---|---|---|---|---|",
    ])
    for setting in SETTINGS:
        model = document["records"][setting]["model"].upper()
        for method in ALL_METHODS:
            cell = document["records"][setting]["methods"][method]
            for split in ("val_seen", "val_unseen"):
                provenance = cell[split]["provenance"]
                lines.append("| {} | {} | {} | `{}` | `{}` | `{}` |".format(
                    model,
                    method,
                    split,
                    provenance["run_tag"],
                    provenance["git_commit"],
                    provenance["formal_manifest_sha256"],
                ))
    lines.append("")
    return "\n".join(lines)


def _atomic_text(path, content):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_results(document, output_path=DEFAULT_OUTPUT, report_path=DEFAULT_REPORT):
    validate_results_document(document)
    json_text = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    report = render_report(document)
    _atomic_text(output_path, json_text)
    _atomic_text(report_path, report)
    return Path(output_path).resolve(), Path(report_path).resolve()


def validate_outputs(document, output_path=DEFAULT_OUTPUT, report_path=DEFAULT_REPORT):
    output_path = _require_file(output_path, "R2R final benchmark JSON")
    report_path = _require_file(report_path, "R2R final benchmark report")
    actual = _read_json(output_path, "R2R final benchmark JSON")
    validate_results_document(actual)
    if not _same(actual, document):
        raise ArchiveError("final benchmark JSON differs from downloaded evidence")
    if report_path.read_text(encoding="utf-8") != render_report(document):
        raise ArchiveError("final benchmark report differs from downloaded evidence")
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    document = build_results(
        args.batch_root,
        repo_root=args.repo_root,
        registry_path=args.registry,
        spec_path=args.spec,
    )
    if args.check_only:
        print("validated complete R2R Source+15 TTA two-split evidence")
        return 0
    if args.validate:
        validate_outputs(document, args.output, args.report)
        print("validated R2R final benchmark outputs")
        return 0
    output, report = write_results(document, args.output, args.report)
    print("wrote {}".format(output))
    print("wrote {}".format(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArchiveError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
