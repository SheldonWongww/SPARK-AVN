#!/usr/bin/env python3
"""Run the frozen 3-model x 5-method R2R val_unseen evaluation matrix.

The fifteen TTA configurations are read verbatim from the completed R2R
val_seen registry.  This runner never executes Source and never selects or
promotes a configuration on val_unseen.  Models are strict sequential phases;
the five methods belonging to one model may run concurrently.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if value not in sys.path:
        sys.path.insert(0, value)

import build_r2r_final_registry as registry_builder  # noqa: E402
from joint_campaign_contract import (  # noqa: E402
    process_identity,
    process_identity_alive,
)
from tools.run_manifest_identity import (  # noqa: E402
    immutable_identity_sha256,
)


SCHEMA = "navtta.vln_r2r_val_unseen_frozen_eval.v1"
SOURCE_SCHEMA = "navtta.vln_r2r_val_unseen_reused_source_controls.v1"
REGISTRY_SCHEMA = "navtta.vln_r2r_final_registry.v1"
JOB_SCHEMA = "navtta.vln_r2r_val_unseen_frozen_job.v1"
SUMMARY_SCHEMA = "navtta.vln_r2r_val_unseen_frozen_summary.v1"
BATCH_SCHEMA = "navtta.vln_r2r_val_unseen_frozen_batch.v1"

DEFAULT_SPEC = (
    REPO_ROOT / "vln/experiments/r2r_val_unseen_frozen_eval_v1.json"
)
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
LOG_ROOT = REPO_ROOT / "vln/results/logs/r2r/frozen_val_unseen"
TUNING_ROOT = REPO_ROOT / "vln/results/tuning"
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"

SETTINGS = ("duet-r2r", "hamt-r2r", "goat-r2r")
MODELS = ("duet", "hamt", "goat")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
MODEL_FOR_SETTING = dict(zip(SETTINGS, MODELS))
EXPECTED_BENCHMARK = {
    "duet-r2r": "r2r_discrete_duet_hamt",
    "hamt-r2r": "r2r_discrete_duet_hamt",
    "goat-r2r": "r2r_discrete_goat",
}
EXPECTED_FEEDBACK_ENDPOINT = {
    "feedtta": "r2r_submitted_trajectory_evaluator_success_every_episode",
    "atena": "r2r_submitted_trajectory_evaluator_success_lazy_query",
}
METRIC_RE = re.compile(r"\b(sr|spl):\s*(-?[0-9]+(?:\.[0-9]+)?)")


class UserError(RuntimeError):
    pass


def _read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("cannot read JSON {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise UserError("JSON document must be an object: {}".format(path))
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value):
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None


def _resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_file(value, label, required_root=None):
    path = _resolve_repo_path(value).resolve()
    root = (required_root or REPO_ROOT).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise UserError("{} is outside {}: {}".format(label, root, path))
    if not path.is_file():
        raise UserError("missing {}: {}".format(label, path))
    return path


def _git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _tracked_worktree_dirty():
    return bool(subprocess.check_output(
        [
            "git", "-C", str(REPO_ROOT), "status", "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip())


def _validate_order_manifest(binding, expected_settings, protocol):
    if set(binding) != {"path", "sha256", "settings"}:
        raise UserError("episode-order binding has unexpected fields")
    if tuple(binding["settings"]) != tuple(expected_settings):
        raise UserError("episode-order setting binding mismatch")
    path = _repo_file(binding["path"], "episode-order manifest")
    if not _valid_sha256(binding["sha256"]) or _sha256(path) != binding[
        "sha256"
    ]:
        raise UserError("episode-order manifest SHA256 mismatch")
    document = _read_json(path)
    expected_benchmark = (
        "r2r_discrete_goat"
        if tuple(expected_settings) == ("goat-r2r",)
        else "r2r_discrete_duet_hamt"
    )
    expected = {
        "schema": "navtta.episode_order.v1",
        "benchmark": expected_benchmark,
        "split": "val_unseen",
        "episode_count": protocol["episode_count"],
        "order_sha256": protocol["episode_order_sha256"],
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise UserError(
                "episode-order manifest {} mismatch".format(key)
            )


def _validate_spec_structure(spec):
    if spec.get("schema") != SCHEMA:
        raise UserError("unsupported R2R val_unseen frozen-eval schema")
    if not re.fullmatch(
        r"[A-Za-z0-9._-]+", str(spec.get("experiment_id", ""))
    ):
        raise UserError("invalid experiment_id")

    protocol = spec.get("protocol")
    if not isinstance(protocol, dict):
        raise UserError("missing protocol")
    expected_protocol = {
        "benchmark": "r2r",
        "split": "val_unseen",
        "episode_count": 2349,
        "canonical_order_seed": 0,
        "episode_order_sha256": (
            "bfaa25c07a8755e67585b1cde3c99833377005ea70017107457761e8ad82a390"
        ),
        "full_split_only": True,
        "selection_on_val_unseen": False,
        "order_seed_cli_forbidden": True,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise UserError("protocol {} mismatch".format(key))
    order_manifests = protocol.get("order_manifests")
    if not isinstance(order_manifests, dict) or set(order_manifests) != {
        "duet_hamt", "goat"
    }:
        raise UserError("protocol must pin both R2R order manifests")
    _validate_order_manifest(
        order_manifests["duet_hamt"], ("duet-r2r", "hamt-r2r"), protocol
    )
    _validate_order_manifest(
        order_manifests["goat"], ("goat-r2r",), protocol
    )

    dependency = spec.get("registry_dependency")
    if not isinstance(dependency, dict):
        raise UserError("missing R2R final-registry dependency")
    if dependency.get("schema") != REGISTRY_SCHEMA:
        raise UserError("registry schema binding mismatch")
    registry_path = _repo_file(
        dependency.get("path", ""), "R2R final registry"
    )
    if not _valid_sha256(dependency.get("sha256")) or _sha256(
        registry_path
    ) != dependency.get("sha256"):
        raise UserError("R2R final registry SHA256 mismatch")

    source = spec.get("source_control")
    if not isinstance(source, dict):
        raise UserError("missing val_unseen Source control binding")
    if source.get("execution") != "reuse_only" or source.get(
        "rerun_forbidden"
    ) is not True:
        raise UserError("Source must be reuse-only and rerun-forbidden")
    source_path = _repo_file(
        source.get("manifest", ""), "val_unseen Source ledger"
    )
    if not _valid_sha256(source.get("sha256")) or _sha256(
        source_path
    ) != source.get("sha256"):
        raise UserError("val_unseen Source ledger SHA256 mismatch")

    matrix = spec.get("matrix")
    if not isinstance(matrix, dict):
        raise UserError("missing frozen matrix")
    if tuple(matrix.get("setting_order", ())) != SETTINGS:
        raise UserError("frozen matrix setting order mismatch")
    if tuple(matrix.get("model_order", ())) != MODELS:
        raise UserError("frozen matrix model order mismatch")
    if tuple(matrix.get("method_order", ())) != METHODS:
        raise UserError("frozen matrix method order mismatch")
    if matrix.get("strict_model_barrier") is not True:
        raise UserError("model barrier must be strict")
    if matrix.get("parallel_methods_within_model") is not True:
        raise UserError("methods must be parallel within each model")
    jobs = matrix.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 15:
        raise UserError("frozen matrix must contain exactly 15 TTA jobs")
    expected_pairs = {
        (setting, method) for setting in SETTINGS for method in METHODS
    }
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise UserError("frozen job must be an object")
        allowed = {
            "setting", "model", "method", "selected_run_tag",
            "selected_formal_manifest_sha256",
        }
        if set(job) != allowed:
            raise UserError("frozen job has unexpected or missing fields")
        pair = (job.get("setting"), job.get("method"))
        if pair not in expected_pairs or pair in seen:
            raise UserError("invalid or duplicate frozen matrix pair")
        if job.get("model") != MODEL_FOR_SETTING[pair[0]]:
            raise UserError("frozen job model/setting mismatch")
        if not re.fullmatch(
            r"[A-Za-z0-9._-]+", str(job.get("selected_run_tag", ""))
        ):
            raise UserError("frozen job selected run tag is invalid")
        if not _valid_sha256(job.get("selected_formal_manifest_sha256")):
            raise UserError("frozen job selected manifest digest is invalid")
        seen.add(pair)
    if seen != expected_pairs:
        raise UserError("frozen matrix is incomplete")

    execution = spec.get("execution")
    if not isinstance(execution, dict):
        raise UserError("missing execution policy")
    if execution.get("formal_requires_clean_tracked_tree") is not True:
        raise UserError("formal execution must require a clean tracked tree")
    caps = execution.get("max_workers_by_model")
    if caps != {model: 5 for model in MODELS}:
        raise UserError("each model phase must permit its five methods")
    if execution.get("launch_stagger_seconds") != 0:
        raise UserError("five methods must launch without serial staggering")

    budget = spec.get("budget")
    if budget != {
        "source_execution_jobs": 0,
        "tta_jobs": 15,
        "total_executed_jobs": 15,
    }:
        raise UserError("frozen-eval budget must be exactly 15 TTA + 0 Source")
    return registry_path, source_path


def load_registry(spec):
    dependency = spec["registry_dependency"]
    path = _repo_file(dependency["path"], "R2R final registry")
    if _sha256(path) != dependency["sha256"]:
        raise UserError("R2R final registry changed after spec validation")
    try:
        registry = registry_builder.validate_registry(path)
    except registry_builder.RegistryError as error:
        raise UserError("R2R final registry is invalid: {}".format(error))
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get(
        "registry_status"
    ) != "complete":
        raise UserError("R2R final registry is not complete")
    return path, registry


def _registry_entry(registry, setting, method):
    try:
        entry = registry["records"][setting][method]
    except (KeyError, TypeError):
        raise UserError("R2R registry lacks {} {}".format(setting, method))
    if not isinstance(entry, dict):
        raise UserError("R2R registry entry is malformed")
    return entry


def _validate_matrix_against_registry(spec, registry):
    for declared in spec["matrix"]["jobs"]:
        entry = _registry_entry(
            registry, declared["setting"], declared["method"]
        )
        if entry.get("run_tag") != declared["selected_run_tag"]:
            raise UserError(
                "selected run mismatch for {} {}".format(
                    declared["setting"], declared["method"]
                )
            )
        if entry.get("formal_manifest_sha256") != declared[
            "selected_formal_manifest_sha256"
        ]:
            raise UserError(
                "selected manifest mismatch for {} {}".format(
                    declared["setting"], declared["method"]
                )
            )
        if not isinstance(entry.get("parameters"), dict):
            raise UserError("selected registry entry has no parameter object")
    return registry


def _validate_source_manifest(record, setting):
    path = _repo_file(
        record["formal_manifest_path"],
        "{} Source formal manifest".format(setting),
        FORMAL_ROOT,
    )
    if _sha256(path) != record.get("formal_manifest_sha256"):
        raise UserError("{} Source formal manifest SHA256 mismatch".format(setting))
    manifest = _read_json(path)
    expected = {
        "task": "vln",
        "benchmark": EXPECTED_BENCHMARK[setting],
        "model": MODEL_FOR_SETTING[setting],
        "method": "source",
        "run_tag": record["run_tag"],
        "source_setting": "{}:val_unseen:native".format(setting),
        "seed": 0,
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise UserError(
                "{} Source formal manifest {} mismatch".format(setting, key)
            )
    if manifest.get("checkpoint", {}).get("sha256") != record[
        "checkpoint_sha256"
    ]:
        raise UserError("{} Source checkpoint mismatch".format(setting))
    dataset = manifest.get("dataset", {})
    if dataset.get("stream_content_sha256") != record["dataset_sha256"]:
        raise UserError("{} Source dataset mismatch".format(setting))
    if dataset.get("stream_order_sha256") != record[
        "episode_order_sha256"
    ]:
        raise UserError("{} Source episode order mismatch".format(setting))
    identity = manifest.get("immutable_identity_sha256")
    if identity != record.get("immutable_identity_sha256") or (
        immutable_identity_sha256(manifest) != identity
    ):
        raise UserError("{} Source immutable identity mismatch".format(setting))
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or record.get(
        "metrics_artifact_sha256"
    ) not in {
        artifact.get("sha256")
        for artifact in artifacts
        if isinstance(artifact, dict)
    }:
        raise UserError(
            "{} Source metrics are not authenticated by its formal manifest"
            .format(setting)
        )
    return path, manifest


def _validate_source_metrics_artifact(record, setting):
    artifact = _repo_file(
        record["metrics_artifact_path"],
        "{} Source metrics artifact".format(setting),
    )
    if _sha256(artifact) != record["metrics_artifact_sha256"]:
        raise UserError(
            "{} Source metrics artifact SHA256 mismatch".format(setting)
        )
    matches = []
    for line in artifact.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        values = {
            key.upper(): float(value) for key, value in METRIC_RE.findall(line)
        }
        if set(values) == {"SR", "SPL"} and "Env name: val_unseen" in line:
            matches.append(values)
    if len(matches) != 1 or any(
        not math.isclose(
            float(matches[0][key]), float(record["metrics"][key]), abs_tol=1e-9
        )
        for key in ("SR", "SPL")
    ):
        raise UserError(
            "{} Source metrics artifact does not match the ledger".format(setting)
        )


def validate_source_ledger(spec, require_ready=False, require_metrics=False):
    binding = spec["source_control"]
    path = _repo_file(binding["manifest"], "val_unseen Source ledger")
    if _sha256(path) != binding["sha256"]:
        raise UserError("val_unseen Source ledger changed after spec validation")
    ledger = _read_json(path)
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
            raise UserError("Source ledger {} mismatch".format(key))
    if ledger.get("source_execution_policy") != {
        "execution": "reuse_only",
        "rerun_forbidden": True,
        "recovery_policy": (
            "Recover an existing formal manifest before considering any "
            "Source execution."
        ),
    }:
        raise UserError("Source ledger reuse-only policy mismatch")
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise UserError("Source ledger must contain exactly three R2R models")

    _, registry = load_registry(spec)
    blockers = []
    for setting in SETTINGS:
        record = records[setting]
        if not isinstance(record, dict):
            raise UserError("{} Source record is malformed".format(setting))
        if record.get("model") != MODEL_FOR_SETTING[setting]:
            raise UserError("{} Source model mismatch".format(setting))
        if record.get("parameters") != {
            "action_selection": "argmax", "action_seed": 0
        }:
            raise UserError("{} Source protocol parameters mismatch".format(setting))
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or not all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics.get(key), bool)
            and 0.0 <= float(metrics[key]) <= 100.0
            for key in ("SR", "SPL")
        ):
            raise UserError("{} Source metrics are invalid".format(setting))
        registry_source = _registry_entry(registry, setting, "source")
        if record.get("checkpoint_sha256") != registry_source.get(
            "checkpoint_sha256"
        ):
            raise UserError(
                "{} val_unseen Source checkpoint differs from frozen registry"
                .format(setting)
            )
        if record.get("episode_order_sha256") != ledger[
            "episode_order_sha256"
        ]:
            raise UserError("{} Source order digest mismatch".format(setting))
        if not _valid_sha256(record.get("dataset_sha256")):
            raise UserError("{} Source dataset digest is invalid".format(setting))

        status = record.get("evidence_status")
        if status == "ready":
            required = (
                "formal_manifest_path", "formal_manifest_sha256",
                "immutable_identity_sha256", "metrics_artifact_path",
                "metrics_artifact_sha256",
            )
            if any(not record.get(key) for key in required):
                raise UserError("{} ready Source record is incomplete".format(setting))
            if not all(
                _valid_sha256(record[key])
                for key in (
                    "formal_manifest_sha256", "immutable_identity_sha256",
                    "metrics_artifact_sha256",
                )
            ):
                raise UserError("{} Source evidence digest is invalid".format(setting))
            _validate_source_manifest(record, setting)
            if require_metrics:
                _validate_source_metrics_artifact(record, setting)
        elif status == "missing_formal_manifest":
            if not record.get("blocking_reason"):
                raise UserError("{} Source blocker lacks a reason".format(setting))
            if (
                not record.get("metrics_artifact_path")
                or not _valid_sha256(record.get("metrics_artifact_sha256"))
            ):
                raise UserError(
                    "{} blocked Source record lacks raw metrics evidence"
                    .format(setting)
                )
            if require_metrics:
                _validate_source_metrics_artifact(record, setting)
            expected_path = _resolve_repo_path(
                record.get("expected_formal_manifest_path", "")
            ).resolve()
            try:
                expected_path.relative_to(FORMAL_ROOT.resolve())
            except ValueError:
                raise UserError("{} expected formal path is noncanonical".format(setting))
            blockers.append({
                "setting": setting,
                "reason": record["blocking_reason"],
                "expected_formal_manifest_path": record[
                    "expected_formal_manifest_path"
                ],
                "manifest_present_but_unregistered": expected_path.is_file(),
            })
        else:
            raise UserError("{} Source evidence status is invalid".format(setting))

    if require_ready and blockers:
        details = "; ".join(
            "{}: {} ({})".format(
                item["setting"], item["reason"],
                item["expected_formal_manifest_path"],
            )
            for item in blockers
        )
        raise UserError(
            "formal execution blocked by incomplete val_unseen Source evidence; "
            "recover the existing manifest without rerunning Source: {}".format(details)
        )
    return path, ledger, blockers


def load_spec(path=DEFAULT_SPEC):
    path = Path(path).resolve()
    spec = _read_json(path)
    _validate_spec_structure(spec)
    _, registry = load_registry(spec)
    _validate_matrix_against_registry(spec, registry)
    validate_source_ledger(spec, require_ready=False, require_metrics=False)
    return spec


def _job_digest(setting, method, entry):
    identity = {
        "setting": setting,
        "method": method,
        "selected_run_tag": entry["run_tag"],
        "parameters": entry["parameters"],
    }
    return hashlib.sha256(
        _canonical(identity).encode("utf-8")
    ).hexdigest()[:10]


def expand_jobs(spec, batch_id, gpu=0):
    _, registry = load_registry(spec)
    _validate_matrix_against_registry(spec, registry)
    declared = {
        (job["setting"], job["method"]): job
        for job in spec["matrix"]["jobs"]
    }
    jobs = []
    ordinal = 0
    for model_index, setting in enumerate(SETTINGS):
        for method_index, method in enumerate(METHODS):
            binding = declared[(setting, method)]
            entry = _registry_entry(registry, setting, method)
            run_tag = (
                "{}-frozen-{:02d}-{}-{:02d}-{}-{}".format(
                    batch_id, model_index, MODEL_FOR_SETTING[setting],
                    method_index, method, _job_digest(setting, method, entry),
                )
            )
            jobs.append({
                "setting": setting,
                "model": MODEL_FOR_SETTING[setting],
                "method": method,
                "model_index": model_index,
                "method_index": method_index,
                "ordinal": ordinal,
                "base_run_tag": run_tag,
                "parameters": json.loads(_canonical(entry["parameters"])),
                "selected_anchor": json.loads(_canonical(entry)),
                "selected_binding": binding,
                "gpu": int(gpu),
            })
            ordinal += 1
    return jobs


def model_phases(jobs):
    phases = []
    for model_index, (setting, model) in enumerate(zip(SETTINGS, MODELS)):
        phase = [job for job in jobs if job["model_index"] == model_index]
        if [job["method"] for job in phase] != list(METHODS):
            raise UserError("{} phase is not the exact five-method matrix".format(model))
        if any(job["setting"] != setting for job in phase):
            raise UserError("{} phase contains another model".format(model))
        phases.append(phase)
    return phases


def _attempt_tag(base_tag, attempt):
    return base_tag if attempt == 0 else "{}-retry{}".format(base_tag, attempt)


def _attempt_dir(batch_root, job, attempt):
    return (
        Path(batch_root)
        / "models"
        / "{:02d}-{}".format(job["model_index"], job["model"])
        / "jobs"
        / job["base_run_tag"]
        / "attempt-{:02d}".format(attempt)
    )


def materialize_attempt(spec, spec_path, batch_id, batch_root, job, attempt):
    run_tag = _attempt_tag(job["base_run_tag"], attempt)
    attempt_dir = _attempt_dir(batch_root, job, attempt)
    result_root = TUNING_ROOT / run_tag / job["setting"] / "val_unseen"
    formal_manifest = (
        FORMAL_ROOT
        / "{}-{}-val_unseen-native".format(run_tag, job["setting"])
        / "manifest.json"
    )
    source_path, source, _ = validate_source_ledger(
        spec, require_ready=False, require_metrics=False
    )
    source_record = source["records"][job["setting"]]
    config = {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": batch_id,
        "stage": "frozen_val_unseen",
        "setting": job["setting"],
        "method": job["method"],
        "search_method": job["method"],
        "episodes": -1,
        "parameters": job["parameters"],
        "frozen_evaluation_provenance": {
            "selection_benchmark": "r2r",
            "selection_split": "val_seen",
            "evaluation_split": "val_unseen",
            "selection_on_val_unseen": False,
            "canonical_order_seed": 0,
            "registry": spec["registry_dependency"],
            "selected_anchor": job["selected_anchor"],
            "source_ledger": {
                "path": spec["source_control"]["manifest"],
                "sha256": _sha256(source_path),
            },
        },
    }
    config_path = attempt_dir / "parameters.json"
    command = [
        str(RUNNER), job["setting"], "val_unseen", str(job["gpu"]),
        "--run-tag", run_tag,
        "--tta-config", str(config_path),
    ]
    metadata = {
        "schema": JOB_SCHEMA,
        "batch_id": batch_id,
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "attempt": attempt,
        "run_tag": run_tag,
        "base_run_tag": job["base_run_tag"],
        "model_index": job["model_index"],
        "method_index": job["method_index"],
        "ordinal": job["ordinal"],
        "setting": job["setting"],
        "model": job["model"],
        "method": job["method"],
        "parameters": job["parameters"],
        "selected_anchor": job["selected_anchor"],
        "episode_count": spec["protocol"]["episode_count"],
        "canonical_order_seed": 0,
        "git_commit": _git_commit(),
        "expected_benchmark": EXPECTED_BENCHMARK[job["setting"]],
        "expected_checkpoint_sha256": source_record["checkpoint_sha256"],
        "expected_dataset_sha256": source_record["dataset_sha256"],
        "expected_episode_order_sha256": spec["protocol"][
            "episode_order_sha256"
        ],
        "source_ledger_path": str(source_path),
        "source_ledger_sha256": _sha256(source_path),
        "result_root": str(result_root),
        "formal_manifest": str(formal_manifest),
        "command": command,
    }
    _atomic_json(config_path, config)
    _atomic_json(attempt_dir / "job.json", metadata)
    return attempt_dir, metadata


def _existing_attempts(batch_root, job):
    root = _attempt_dir(batch_root, job, 0).parent
    if not root.is_dir():
        return []
    values = []
    for path in root.glob("attempt-[0-9][0-9]"):
        try:
            values.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            continue
    return sorted(values)


def _pid_alive(path):
    if not Path(path).is_file():
        return False
    try:
        identity_path = Path(path).with_name("process_identity.json")
        if identity_path.is_file():
            return process_identity_alive(_read_json(identity_path))
        os.kill(int(Path(path).read_text(encoding="utf-8").strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def _state_for_attempt(attempt_dir):
    attempt_dir = Path(attempt_dir)
    if (attempt_dir / "metrics.json").is_file():
        return "completed"
    exit_path = attempt_dir / "exitcode"
    if exit_path.is_file():
        try:
            return "finished" if int(exit_path.read_text().strip()) == 0 else "failed"
        except ValueError:
            return "invalid"
    if _pid_alive(attempt_dir / "pid"):
        return "running"
    if (attempt_dir / "job.json").is_file():
        return "orphaned"
    return "pending"


def _latest_attempt(batch_root, job):
    attempts = _existing_attempts(batch_root, job)
    return attempts[-1] if attempts else None


def _parse_metrics(result_root):
    matches = []
    for path in sorted(Path(result_root).rglob("valid.txt")):
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            values = {
                key.upper(): float(value) for key, value in METRIC_RE.findall(line)
            }
            if set(values) == {"SR", "SPL"} and "Env name: val_unseen" in line:
                matches.append((path, values, line))
    if len(matches) != 1:
        raise UserError(
            "expected one R2R val_unseen metric line under {}, found {}"
            .format(result_root, len(matches))
        )
    path, values, line = matches[0]
    if any(not 0.0 <= value <= 100.0 for value in values.values()):
        raise UserError("R2R val_unseen metric is outside [0, 100]")
    return path, values, line


def _validate_formal_manifest(path, metadata, required_artifacts=()):
    path = Path(path).resolve()
    try:
        path.relative_to(FORMAL_ROOT.resolve())
    except ValueError:
        raise UserError("formal manifest is outside vln/results/runs")
    expected_run_id = "{}-{}-val_unseen-native".format(
        metadata["run_tag"], metadata["setting"]
    )
    if path.name != "manifest.json" or path.parent.name != expected_run_id:
        raise UserError("formal manifest path is noncanonical")
    if not path.is_file():
        raise UserError("missing formal manifest: {}".format(path))
    manifest = _read_json(path)
    expected = {
        "run_id": expected_run_id,
        "task": "vln",
        "benchmark": metadata["expected_benchmark"],
        "model": metadata["model"],
        "method": metadata["method"],
        "run_tag": metadata["run_tag"],
        "source_setting": "{}:val_unseen:native:{}".format(
            metadata["setting"], metadata["method"]
        ),
        "seed": 0,
        "git_commit": metadata["git_commit"],
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise UserError("formal manifest {} mismatch".format(key))
    if manifest.get("checkpoint", {}).get("sha256") != metadata[
        "expected_checkpoint_sha256"
    ]:
        raise UserError("formal manifest checkpoint SHA256 mismatch")
    dataset = manifest.get("dataset", {})
    if dataset.get("stream_content_sha256") != metadata[
        "expected_dataset_sha256"
    ]:
        raise UserError("formal manifest dataset SHA256 mismatch")
    if dataset.get("stream_order_sha256") != metadata[
        "expected_episode_order_sha256"
    ]:
        raise UserError("formal manifest episode-order SHA256 mismatch")
    identity = manifest.get("immutable_identity_sha256")
    if not _valid_sha256(identity) or immutable_identity_sha256(
        manifest
    ) != identity:
        raise UserError("formal manifest immutable identity mismatch")
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise UserError("formal manifest has no result artifacts")
    authenticated = set()
    result_root = Path(metadata["result_root"]).resolve()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise UserError("formal manifest artifact is malformed")
        artifact_path = Path(str(artifact.get("path", ""))).resolve()
        try:
            artifact_path.relative_to(result_root)
        except ValueError:
            raise UserError("formal artifact escapes result root")
        if not artifact_path.is_file():
            raise UserError("formal result artifact is missing")
        if artifact.get("size") != artifact_path.stat().st_size or _sha256(
            artifact_path
        ) != artifact.get("sha256"):
            raise UserError("formal result artifact digest mismatch")
        authenticated.add(artifact_path)
    for required in required_artifacts:
        if Path(required).resolve() not in authenticated:
            raise UserError("required artifact is absent from formal manifest")
    return manifest


def _validate_diagnostics(metadata, diagnostics, metrics):
    expected_episodes = metadata["episode_count"]
    method = metadata["method"]
    if diagnostics.get("method") != method:
        raise UserError("TTA diagnostics method mismatch")
    if diagnostics.get("episode_count") != expected_episodes:
        raise UserError("TTA diagnostics episode count mismatch")
    adapter = diagnostics.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("episodes") != expected_episodes:
        raise UserError("adapter episode count mismatch")

    if method in EXPECTED_FEEDBACK_ENDPOINT:
        if diagnostics.get("supervision") != "binary_navigation_success_feedback":
            raise UserError("binary-feedback supervision label mismatch")
        endpoint = diagnostics.get("binary_feedback_endpoint")
        if endpoint != EXPECTED_FEEDBACK_ENDPOINT[method] or "distance" in endpoint:
            raise UserError("invalid R2R binary-feedback endpoint")
        if method == "feedtta":
            successes = adapter.get("successful_feedback_episodes")
            failures = adapter.get("failed_feedback_episodes")
            if (
                adapter.get("feedback_episodes") != expected_episodes
                or not isinstance(successes, int)
                or not isinstance(failures, int)
                or successes + failures != expected_episodes
            ):
                raise UserError("FeedTTA feedback accounting mismatch")
            if successes != int(round(metrics["SR"] * expected_episodes / 100.0)):
                raise UserError("FeedTTA feedback does not match evaluator SR")
            if adapter.get("action_selection_protocol") != "target_native_argmax":
                raise UserError("FeedTTA did not use target-native argmax")
        else:
            queries = adapter.get("queries")
            self_labels = adapter.get("self_label_episodes")
            observed = adapter.get("feedback_observed_episodes")
            if (
                not isinstance(queries, int)
                or not isinstance(self_labels, int)
                or queries + self_labels != expected_episodes
                or observed != queries
                or adapter.get("query_gate_evaluations") != expected_episodes
            ):
                raise UserError("ATENA lazy feedback accounting mismatch")
            if (
                float(metadata["parameters"].get("query_threshold", math.nan)) == 0.0
                and queries != expected_episodes
            ):
                raise UserError("zero-threshold ATENA must query every episode")
    else:
        if diagnostics.get("supervision") != "unsupervised":
            raise UserError("unsupervised method has wrong supervision label")
        if diagnostics.get("binary_feedback_endpoint") is not None:
            raise UserError("unsupervised method unexpectedly reports feedback")
        if method == "fstta" and adapter.get(
            "variance_history_lifetime"
        ) != "test_stream":
            raise UserError("FSTTA did not preserve stream variance history")
    return adapter


def validate_attempt(attempt_dir):
    attempt_dir = Path(attempt_dir)
    metadata = _read_json(attempt_dir / "job.json")
    try:
        exit_code = int((attempt_dir / "exitcode").read_text().strip())
    except (OSError, ValueError) as error:
        raise UserError("invalid worker exit status: {}".format(error))
    if exit_code != 0:
        raise UserError("worker exited with status {}".format(exit_code))
    metric_path, metrics, metric_line = _parse_metrics(metadata["result_root"])
    diagnostics_path = Path(metadata["result_root"]) / "tta_diagnostics.json"
    if not diagnostics_path.is_file():
        raise UserError("missing TTA diagnostics")
    diagnostics = _read_json(diagnostics_path)
    adapter = _validate_diagnostics(metadata, diagnostics, metrics)
    manifest = _validate_formal_manifest(
        metadata["formal_manifest"], metadata,
        required_artifacts=(metric_path, diagnostics_path),
    )
    result = {
        **metadata,
        "metrics": metrics,
        "metric_artifact": str(metric_path),
        "metric_artifact_sha256": _sha256(metric_path),
        "metric_line": metric_line,
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "feedback_endpoint": diagnostics.get("binary_feedback_endpoint"),
        "adapter_diagnostics": adapter,
        "formal_manifest_sha256": _sha256(metadata["formal_manifest"]),
        "formal_immutable_identity_sha256": manifest.get(
            "immutable_identity_sha256"
        ),
    }
    _atomic_json(attempt_dir / "metrics.json", result)
    return result


def _write_worker(attempt_dir, command):
    exit_path = Path(attempt_dir) / "exitcode"
    script = "\n".join((
        "#!/usr/bin/env bash",
        "set +e",
        shlex.join(command),
        "status=$?",
        "printf '%s\\n' \"$status\" > {}".format(
            shlex.quote(str(exit_path) + ".tmp")
        ),
        "mv {} {}".format(
            shlex.quote(str(exit_path) + ".tmp"), shlex.quote(str(exit_path))
        ),
        "exit \"$status\"",
        "",
    ))
    path = Path(attempt_dir) / "worker.sh"
    _atomic_text(path, script)
    path.chmod(0o755)
    return path


def launch_attempt(attempt_dir, metadata):
    worker = _write_worker(attempt_dir, metadata["command"])
    launcher_log = (Path(attempt_dir) / "launcher.log").open("ab", buffering=0)
    process = subprocess.Popen(
        ["bash", str(worker)],
        cwd=str(REPO_ROOT),
        stdout=launcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _atomic_text(Path(attempt_dir) / "pid", "{}\n".format(process.pid))
    identity = process_identity(process.pid)
    if identity is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        launcher_log.close()
        raise UserError("cannot bind worker process identity")
    _atomic_json(Path(attempt_dir) / "process_identity.json", identity)
    return process, launcher_log


def collect_states(batch_root, jobs):
    states = {key: 0 for key in (
        "completed", "finished", "failed", "invalid", "running",
        "orphaned", "pending",
    )}
    rows = []
    for job in jobs:
        latest = _latest_attempt(batch_root, job)
        if latest is None:
            states["pending"] += 1
            continue
        _, attempt_dir = latest
        state = _state_for_attempt(attempt_dir)
        states[state] += 1
        if state == "completed":
            rows.append(_read_json(attempt_dir / "metrics.json"))
    return states, rows


def write_summary(batch_root, spec, jobs):
    states, rows = collect_states(batch_root, jobs)
    _, _, blockers = validate_source_ledger(spec, require_ready=False)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "source_execution_jobs": 0,
        "total_jobs": 15,
        "complete": states["completed"] == 15,
        "states": states,
        "source_evidence_blockers": blockers,
        "results": [{
            "setting": row["setting"],
            "method": row["method"],
            "run_tag": row["run_tag"],
            "parameters": row["parameters"],
            "metrics": row["metrics"],
            "feedback_endpoint": row["feedback_endpoint"],
            "formal_manifest": row["formal_manifest"],
            "formal_manifest_sha256": row["formal_manifest_sha256"],
        } for row in sorted(rows, key=lambda item: item["ordinal"])],
    }
    _atomic_json(Path(batch_root) / "SUMMARY.json", payload)
    with (Path(batch_root) / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("setting", "method", "SR", "SPL", "run_tag"),
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["ordinal"]):
            writer.writerow({
                "setting": row["setting"],
                "method": row["method"],
                "SR": row["metrics"]["SR"],
                "SPL": row["metrics"]["SPL"],
                "run_tag": row["run_tag"],
            })
    return payload


def _prepare_batch(spec_path, spec, batch_id, batch_root, gpu, resume):
    snapshot = {
        "schema": BATCH_SCHEMA,
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "registry": spec["registry_dependency"],
        "source_ledger": {
            "path": spec["source_control"]["manifest"],
            "sha256": spec["source_control"]["sha256"],
        },
        "gpu": int(gpu),
        "source_execution_jobs": 0,
        "tta_jobs": 15,
    }
    path = Path(batch_root) / "BATCH.json"
    if path.is_file():
        if not resume:
            raise UserError("batch already exists; use --resume")
        if _canonical(_read_json(path)) != _canonical(snapshot):
            raise UserError("batch ID is bound to a different immutable plan")
    elif Path(batch_root).exists() and any(Path(batch_root).iterdir()):
        raise UserError("nonempty batch root lacks BATCH.json")
    _atomic_json(path, snapshot)


def run_model_phase(
    spec, spec_path, batch_id, batch_root, phase_jobs, retry_failed=False
):
    model = phase_jobs[0]["model"]
    cap = spec["execution"]["max_workers_by_model"][model]
    poll_seconds = spec["execution"]["poll_seconds"]
    active = {}
    try:
        while True:
            completed = 0
            running = 0
            launchable = []
            for job in phase_jobs:
                latest = _latest_attempt(batch_root, job)
                if latest is None:
                    launchable.append((job, 0))
                    continue
                attempt, attempt_dir = latest
                state = _state_for_attempt(attempt_dir)
                if state == "completed":
                    completed += 1
                elif state == "finished":
                    try:
                        validate_attempt(attempt_dir)
                    except Exception as error:
                        _atomic_json(
                            attempt_dir / "validation_error.json",
                            {"error": str(error)},
                        )
                        raise
                    completed += 1
                elif state == "running":
                    running += 1
                elif state in ("failed", "invalid", "orphaned"):
                    if not retry_failed:
                        raise UserError(
                            "{} is {}; use --resume --retry-failed".format(
                                job["base_run_tag"], state
                            )
                        )
                    launchable.append((job, attempt + 1))
                else:
                    launchable.append((job, attempt))

            if completed == len(phase_jobs):
                return

            # ``running`` already includes workers launched by this process,
            # because their persisted process identities are authoritative.
            slots = max(0, cap - running)
            for job, attempt in launchable[:slots]:
                attempt_dir, metadata = materialize_attempt(
                    spec, spec_path, batch_id, batch_root, job, attempt
                )
                process, launcher_log = launch_attempt(attempt_dir, metadata)
                active[job["base_run_tag"]] = (
                    process, launcher_log, attempt_dir, metadata
                )
                print(
                    "launched {} pid={}".format(
                        metadata["run_tag"], process.pid
                    )
                )

            finished = []
            for tag, (
                process, launcher_log, attempt_dir, metadata
            ) in active.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                launcher_log.close()
                finished.append(tag)
                if return_code != 0:
                    raise UserError(
                        "{} failed with {}; inspect {}".format(
                            metadata["run_tag"], return_code,
                            attempt_dir / "launcher.log",
                        )
                    )
                result = validate_attempt(attempt_dir)
                print(
                    "completed {} SR/SPL={:.2f}/{:.2f}".format(
                        result["run_tag"], result["metrics"]["SR"],
                        result["metrics"]["SPL"],
                    )
                )
            for tag in finished:
                active.pop(tag)
            if not finished:
                time.sleep(poll_seconds)
    except Exception:
        # A failed method must not leave sibling workers detached from a dead
        # model-phase scheduler. Their atomic exit files make the interrupted
        # attempts explicit and retryable.
        for process, launcher_log, _, _ in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            launcher_log.close()
        raise


def print_plan(spec, jobs, batch_id, gpu, print_commands=False):
    _, _, blockers = validate_source_ledger(spec, require_ready=False)
    print(
        "batch_id={} gpu={} tta_jobs=15 source_jobs=0 split=val_unseen "
        "episodes=2349 canonical_order_seed=0".format(batch_id, gpu)
    )
    for phase in model_phases(jobs):
        print(
            "model {}: methods={} max_workers={} strict_barrier_after=true"
            .format(
                phase[0]["model"], ",".join(job["method"] for job in phase),
                spec["execution"]["max_workers_by_model"][phase[0]["model"]],
            )
        )
        if print_commands:
            for job in phase:
                print(
                    shlex.join([
                        str(RUNNER), job["setting"], "val_unseen", str(gpu),
                        "--run-tag", job["base_run_tag"],
                        "--tta-config", "<ATTEMPT_DIR>/parameters.json",
                    ])
                )
    if blockers:
        print("formal execution blocked:")
        for blocker in blockers:
            print(
                "  {}: {} ({})".format(
                    blocker["setting"], blocker["reason"],
                    blocker["expected_formal_manifest_path"],
                )
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--batch-id", default="vln-r2r-val-unseen-frozen-eval-v1-seed0"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.batch_id):
        parser.error("invalid --batch-id")

    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    jobs = expand_jobs(spec, args.batch_id, args.gpu)
    batch_root = LOG_ROOT / args.batch_id
    if args.plan_only:
        print_plan(
            spec, jobs, args.batch_id, args.gpu,
            print_commands=args.print_commands,
        )
        return 0
    if args.status:
        states, rows = collect_states(batch_root, jobs)
        _, _, blockers = validate_source_ledger(spec, require_ready=False)
        print(json.dumps({
            "batch_id": args.batch_id,
            "started": (batch_root / "BATCH.json").is_file(),
            "states": states,
            "completed_results": len(rows),
            "source_evidence_blockers": blockers,
        }, indent=2, sort_keys=True))
        return 0
    if not args.confirm_reviewed:
        raise UserError("formal execution requires --confirm-reviewed")
    validate_source_ledger(spec, require_ready=True, require_metrics=True)
    if _tracked_worktree_dirty():
        raise UserError("tracked worktree must be clean for formal execution")
    _prepare_batch(
        spec_path, spec, args.batch_id, batch_root, args.gpu, args.resume
    )
    for phase in model_phases(jobs):
        run_model_phase(
            spec, spec_path, args.batch_id, batch_root, phase,
            retry_failed=args.retry_failed,
        )
        write_summary(batch_root, spec, jobs)
    summary = write_summary(batch_root, spec, jobs)
    if not summary["complete"]:
        raise UserError("frozen val_unseen campaign ended before 15/15 jobs")
    print("all 15 R2R val_unseen frozen TTA jobs completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (UserError, registry_builder.RegistryError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
