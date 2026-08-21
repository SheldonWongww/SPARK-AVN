#!/usr/bin/env python3
"""Run a model-major R2R local-refinement campaign.

The campaign plan is intentionally different from the legacy Cartesian
search: candidates are model-specific and execution is a strict sequence of
single-model, single-method phases.  Each phase delegates worker lifecycle,
resource checks, retry handling, and durable progress to
``run_tta_hparam_search.run_batch``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tta_hparam_search as staged  # noqa: E402
import tta_config_cli as config_cli  # noqa: E402


DEFAULT_SPEC = (
    REPO_ROOT / "vln/experiments/r2r_five_method_local_refinement_v1.json"
)
LOG_ROOT = REPO_ROOT / "vln/results/logs/r2r/hparam_search"
TUNING_ROOT = REPO_ROOT / "vln/results/tuning/r2r/hparam_search"
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
RESULT_LAYOUT = "r2r_benchmark_model_method_local_refinement_v1"
SPEC_SCHEMA_V1 = "navtta.vln_r2r_five_method_local_refinement_plan.v1"
SPEC_SCHEMA_V2 = "navtta.vln_r2r_two_method_postfix_search_plan.v1"
SPEC_SCHEMAS = (SPEC_SCHEMA_V1, SPEC_SCHEMA_V2)
PLAN_SCHEMA = "navtta.vln_r2r_local_refinement_campaign_plan.v1"
PHASE_SCHEMA = "navtta.vln_r2r_local_refinement_phase.v1"
CALIBRATION_SCHEMA = "navtta.vln_r2r_local_refinement_calibration.v1"
CALIBRATION_PLAN_SCHEMA = "navtta.vln_r2r_local_refinement_calibration_plan.v1"
REUSED_SOURCE_SCHEMA = "navtta.vln_r2r_reused_source_controls.v1"
STAGE = "local_refinement"

SETTINGS = ("duet-r2r", "hamt-r2r", "goat-r2r")
SETTING_MODEL = {
    "duet-r2r": "duet",
    "hamt-r2r": "hamt",
    "goat-r2r": "goat",
}
SUPPORTED_TTA_METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
SUPPORTED_PHASE_ORDER = (
    "source",
    "tent",
    "fstta",
    "eam",
    "feedtta_control",
    "feedtta",
    "atena",
)
CONTROL_METHODS = ("source", "feedtta_control")
REUSED_SOURCE_PROVENANCE_FIELDS = (
    "checkpoint_sha256",
    "formal_manifest_path",
    "formal_manifest_sha256",
    "job_json_path",
    "job_json_sha256",
    "metrics_json_path",
    "metrics_json_sha256",
    "parameters_json_path",
    "parameters_json_sha256",
)
REUSED_SOURCE_BATCH_PROVENANCE_FIELDS = (
    "grid_path",
    "grid_sha256",
    "summary_path",
    "summary_sha256",
)

DEFAULT_MAX_MEMORY_GIB = 75.0
DEFAULT_LAUNCH_STAGGER_SECONDS = 15.0
DEFAULT_RESOURCE_WAIT_TIMEOUT_SECONDS = 900.0
CALIBRATION_RESOURCE_FIELDS = (
    "observed_at", "elapsed_seconds", "phase_id", "setting", "method",
    "launched_count", "active_count", "completed_count", "gpu_memory_mib",
    "gpu_utilization_pct", "cgroup_memory_gib", "action",
)


class UserError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args], text=True
    ).strip()


def safe_component(label, value):
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in (".", "..")
        or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
    ):
        raise UserError(f"unsafe {label} path component: {value!r}")
    return value


def _mapping(value, label):
    if not isinstance(value, dict):
        raise UserError(f"{label} must be an object")
    return value


def _exact_keys(value, expected, label):
    actual = set(_mapping(value, label))
    if actual != set(expected):
        raise UserError(
            f"{label} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UserError(f"{label} must be a positive integer")
    return value


def _nonempty_string(value, label):
    if not isinstance(value, str) or not value:
        raise UserError(f"{label} must be a nonempty string")
    return value


def _sha256_string(value, label):
    value = _nonempty_string(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise UserError(f"{label} must be a lowercase SHA256 digest")
    return value


def _git_commit_string(value, label):
    value = _nonempty_string(value, label)
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise UserError(f"{label} must be a lowercase 40-hex Git commit")
    return value


def _safe_relative_path(value, label):
    value = _nonempty_string(value, label)
    path = Path(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in path.parts)
        or re.fullmatch(r"[A-Za-z0-9._/-]+", value) is None
    ):
        raise UserError(f"{label} must be a safe repository-relative path")
    return value


def _validate_reused_source_document(document, label):
    document = _mapping(document, label)
    if document.get("schema") != REUSED_SOURCE_SCHEMA:
        raise UserError("unsupported reused Source controls schema")
    if document.get("benchmark") != "r2r" or document.get("split") != "val_seen":
        raise UserError("reused Source controls must be R2R val_seen")
    if document.get("source_protocol") != "standard_argmax":
        raise UserError("reused Source controls must use standard_argmax")
    if document.get("episode_count") != 1021 or document.get("order_seed") != 0:
        raise UserError(
            "reused Source controls must use 1,021 episodes and order seed 0"
        )
    _sha256_string(
        document.get("episode_order_sha256"), f"{label}.episode_order_sha256"
    )
    _nonempty_string(document.get("dataset_version"), f"{label}.dataset_version")
    _nonempty_string(document.get("hardware"), f"{label}.hardware")

    source_batch = _mapping(document.get("source_batch"), f"{label}.source_batch")
    _nonempty_string(
        source_batch.get("batch_id"), f"{label}.source_batch.batch_id"
    )
    _git_commit_string(
        source_batch.get("git_commit"), f"{label}.source_batch.git_commit"
    )
    _sha256_string(
        source_batch.get("spec_sha256"), f"{label}.source_batch.spec_sha256"
    )
    for key in REUSED_SOURCE_BATCH_PROVENANCE_FIELDS:
        value = source_batch.get(key)
        validator = _safe_relative_path if key.endswith("_path") else _sha256_string
        validator(value, f"{label}.source_batch.{key}")
    for key, value in source_batch.items():
        _nonempty_string(value, f"{label}.source_batch.{key}")
        if key.endswith("_path"):
            _safe_relative_path(value, f"{label}.source_batch.{key}")
        elif key.endswith("_sha256"):
            _sha256_string(value, f"{label}.source_batch.{key}")

    records = _mapping(document.get("records"), f"{label}.records")
    _exact_keys(records, SETTINGS, f"{label}.records")
    for setting in SETTINGS:
        record_label = f"{label}.records.{setting}"
        record = _mapping(records[setting], record_label)
        run_tag = _nonempty_string(record.get("run_tag"), f"{record_label}.run_tag")
        batch_id = _nonempty_string(
            record.get("batch_id"), f"{record_label}.batch_id"
        )
        safe_component("reused Source run_tag", run_tag)
        safe_component("reused Source batch_id", batch_id)
        if batch_id != source_batch["batch_id"]:
            raise UserError(f"{record_label}.batch_id does not match source_batch")
        if record.get("model") != SETTING_MODEL[setting]:
            raise UserError(f"{record_label}.model mismatch")

        parameters = _mapping(record.get("parameters"), f"{record_label}.parameters")
        _exact_keys(
            parameters,
            ("action_selection", "action_seed"),
            f"{record_label}.parameters",
        )
        if (
            parameters.get("action_selection") != "argmax"
            or parameters.get("action_seed") != 0
        ):
            raise UserError(
                f"{record_label}.parameters must select argmax with seed 0"
            )

        metrics = _mapping(record.get("metrics"), f"{record_label}.metrics")
        for metric in ("SR", "SPL"):
            value = metrics.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise UserError(f"{record_label}.metrics.{metric} must be finite")
        for key in REUSED_SOURCE_PROVENANCE_FIELDS:
            value = record.get(key)
            validator = _safe_relative_path if key.endswith("_path") else _sha256_string
            validator(value, f"{record_label}.{key}")
        for key, value in record.items():
            if key.endswith("_path"):
                _safe_relative_path(value, f"{record_label}.{key}")
            elif key.endswith("_sha256"):
                _sha256_string(value, f"{record_label}.{key}")

    for key, value in document.items():
        if key.endswith("_path"):
            _safe_relative_path(value, f"{label}.{key}")
        elif key.endswith("_sha256"):
            _sha256_string(value, f"{label}.{key}")

    try:
        canonical(document)
    except (TypeError, ValueError) as error:
        raise UserError(f"invalid reused Source controls manifest: {error}") from error
    return document


def _load_reused_source_manifest(spec):
    controls = _mapping(spec.get("controls"), "controls")
    reference = _nonempty_string(
        controls.get("standard_argmax_source_manifest"),
        "controls.standard_argmax_source_manifest",
    )
    _safe_relative_path(reference, "controls.standard_argmax_source_manifest")
    candidate = REPO_ROOT / reference
    path = candidate.resolve()
    try:
        relative = path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise UserError(
            "controls.standard_argmax_source_manifest must be inside the repository"
        ) from error
    try:
        git("ls-files", "--error-unmatch", "--", relative.as_posix())
    except subprocess.CalledProcessError as error:
        raise UserError(
            "controls.standard_argmax_source_manifest must be tracked by Git"
        ) from error
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"invalid reused Source controls manifest {path}: {error}") from error
    _validate_reused_source_document(document, str(relative))
    return {
        "reference": reference,
        "path": str(path),
        "relative_path": relative.as_posix(),
        "sha256": sha256(path),
        "document": document,
    }


def _validate_reused_source_artifacts(spec):
    """Authenticate every reused Source artifact before a formal launch."""
    binding = _reused_source_manifest(spec)
    document = binding["document"]

    def verify(relative_path, expected_digest, label):
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise UserError(f"missing reused Source {label}: {relative_path}")
        actual = sha256(path)
        if actual != expected_digest:
            raise UserError(
                f"reused Source {label} SHA256 mismatch: "
                f"{actual} != {expected_digest}"
            )

    source_batch = document["source_batch"]
    verify(
        source_batch["grid_path"], source_batch["grid_sha256"],
        "source GRID.json",
    )
    verify(
        source_batch["summary_path"], source_batch["summary_sha256"],
        "source SUMMARY.json",
    )
    for setting, record in document["records"].items():
        for stem in ("job_json", "metrics_json", "parameters_json"):
            verify(
                record[f"{stem}_path"], record[f"{stem}_sha256"],
                f"{setting} {stem}",
            )
        formal_candidates = [record["formal_manifest_path"]]
        evidence_copy = record.get("formal_manifest_evidence_copy_path")
        if evidence_copy is not None:
            formal_candidates.append(evidence_copy)
        existing = [
            relative for relative in formal_candidates
            if (REPO_ROOT / relative).is_file()
        ]
        if not existing:
            raise UserError(
                f"missing reused Source {setting} formal manifest; checked "
                + ", ".join(formal_candidates)
            )
        if not any(
            sha256(REPO_ROOT / relative) == record["formal_manifest_sha256"]
            for relative in existing
        ):
            raise UserError(
                f"reused Source {setting} formal manifest SHA256 mismatch"
            )
        job = _read_json_object(
            REPO_ROOT / record["job_json_path"], f"{setting} Source job"
        )
        parameters = _read_json_object(
            REPO_ROOT / record["parameters_json_path"],
            f"{setting} Source parameters",
        )
        metrics = _read_json_object(
            REPO_ROOT / record["metrics_json_path"],
            f"{setting} Source metrics",
        )
        formal_path = next(
            REPO_ROOT / relative for relative in existing
            if sha256(REPO_ROOT / relative) == record["formal_manifest_sha256"]
        )
        formal = _read_json_object(formal_path, f"{setting} Source formal manifest")
        job_checks = {
            "batch_id": source_batch["batch_id"],
            "run_tag": record["run_tag"],
            "setting": setting,
            "model": record["model"],
            "search_method": "source",
            "config_method": "source",
            "parameters": record["parameters"],
        }
        if any(job.get(key) != value for key, value in job_checks.items()):
            raise UserError(f"reused Source {setting} job/record semantic mismatch")
        parameter_checks = {
            "batch_id": source_batch["batch_id"],
            "run_tag": record["run_tag"],
            "setting": setting,
            "method": "source",
            "search_method": "source",
            "parameters": record["parameters"],
        }
        if any(
            parameters.get(key) != value
            for key, value in parameter_checks.items()
        ):
            raise UserError(
                f"reused Source {setting} parameters/record semantic mismatch"
            )
        metric_checks = dict(job_checks, expected_episodes=document["episode_count"])
        if metrics.get("metrics") != record["metrics"] or any(
            metrics.get(key) != value for key, value in metric_checks.items()
        ):
            raise UserError(
                f"reused Source {setting} metrics/record semantic mismatch"
            )
        formal_checks = {
            "task": "vln",
            "model": record["model"],
            "method": "source",
            "run_tag": record["run_tag"],
            "source_setting": f"{setting}:val_seen:native:source",
            "seed": document["order_seed"],
            "git_commit": source_batch["git_commit"],
            "status": "completed",
            "exit_code": 0,
        }
        if any(formal.get(key) != value for key, value in formal_checks.items()):
            raise UserError(
                f"reused Source {setting} formal/record semantic mismatch"
            )
        if (
            (formal.get("checkpoint") or {}).get("sha256")
            != record["checkpoint_sha256"]
            or (formal.get("dataset") or {}).get("stream_order_sha256")
            != document["episode_order_sha256"]
            or (formal.get("hardware") or {}).get("gpu_name")
            != document["hardware"]
        ):
            raise UserError(
                f"reused Source {setting} formal provenance mismatch"
            )
    return binding


def _validate_search_prior_artifacts(spec):
    if spec.get("schema") != SPEC_SCHEMA_V2:
        return
    for method, settings in spec["search_priors"].items():
        for setting, prior in settings.items():
            path = REPO_ROOT / prior["prior_job_path"]
            if not path.is_file():
                raise UserError(
                    f"missing search-prior job for {method}/{setting}: "
                    f"{prior['prior_job_path']}"
                )
            if sha256(path) != prior["prior_job_sha256"]:
                raise UserError(
                    f"search-prior job SHA256 mismatch for {method}/{setting}"
                )
            job = _read_json_object(path, f"{method}/{setting} search-prior job")
            checks = {
                "run_tag": prior["prior_run_tag"],
                "setting": setting,
                "search_method": method,
                "parameters": prior["historical_parameters"],
            }
            if any(job.get(key) != value for key, value in checks.items()):
                raise UserError(
                    f"search-prior job semantic mismatch for {method}/{setting}"
                )


def _reused_source_manifest(spec):
    binding = spec.get("_reused_source_manifest")
    reference = spec.get("controls", {}).get("standard_argmax_source_manifest")
    if isinstance(binding, dict) and binding.get("reference") == reference:
        return binding
    return _load_reused_source_manifest(spec)


def _reused_source_plan_binding(spec):
    binding = _reused_source_manifest(spec)
    source_batch = binding["document"]["source_batch"]
    return {
        "reference": binding["reference"],
        "path": binding["relative_path"],
        "sha256": binding["sha256"],
        "source_batch": {
            "batch_id": source_batch["batch_id"],
            "git_commit": source_batch["git_commit"],
            "spec_sha256": source_batch["spec_sha256"],
        },
    }


def enabled_phases_by_setting(spec):
    execution = _mapping(spec.get("execution"), "execution")
    declared = execution.get("enabled_methods_by_setting")
    if declared is None:
        raise UserError("execution.enabled_methods_by_setting is required")
    _exact_keys(declared, SETTINGS, "execution.enabled_methods_by_setting")
    output = {}
    global_order = execution.get("method_order_within_model")
    if not isinstance(global_order, list) or len(global_order) != len(set(global_order)):
        raise UserError("execution.method_order_within_model must be a unique array")
    if any(method not in SUPPORTED_PHASE_ORDER for method in global_order):
        raise UserError("execution.method_order_within_model has an unsupported phase")
    global_positions = {method: index for index, method in enumerate(global_order)}
    for setting in SETTINGS:
        phases = declared[setting]
        if not isinstance(phases, list) or not phases:
            raise UserError(f"enabled phases for {setting} must be a nonempty array")
        if len(phases) != len(set(phases)):
            raise UserError(f"enabled phases for {setting} contain duplicates")
        if any(method not in global_positions for method in phases):
            raise UserError(f"enabled phases for {setting} are outside the global order")
        positions = [global_positions[method] for method in phases]
        if positions != sorted(positions):
            raise UserError(f"enabled phases for {setting} violate method order")
        sampled_control_required = bool(
            spec.get("controls", {}).get(
                "feedtta_sampled_no_update_per_setting", False
            )
        )
        if sampled_control_required and (
            ("feedtta" in phases) != ("feedtta_control" in phases)
        ):
            raise UserError(
                f"{setting} must pair FeedTTA with its sampled no-update control"
            )
        if not sampled_control_required and "feedtta_control" in phases:
            raise UserError(
                f"{setting} enables an obsolete sampled FeedTTA control"
            )
        output[setting] = list(phases)
    return output


def active_tta_methods(spec):
    enabled = enabled_phases_by_setting(spec)
    active = {
        method for phases in enabled.values() for method in phases
        if method in SUPPORTED_TTA_METHODS
    }
    return tuple(method for method in SUPPORTED_TTA_METHODS if method in active)


def enabled_settings_for_method(spec, method):
    enabled = enabled_phases_by_setting(spec)
    return tuple(setting for setting in SETTINGS if method in enabled[setting])


def _merge_disjoint(base, extra, label):
    overlap = set(base).intersection(extra)
    if overlap:
        raise UserError(f"{label} redefines parameters: {sorted(overlap)}")
    output = dict(base)
    output.update(extra)
    return output


def _validate_runtime_parameters(method, parameters, label):
    executed_method = _config_method(method)
    allowed = config_cli.COMMON | config_cli.METHOD_KEYS[executed_method]
    unknown = set(parameters).difference(allowed)
    if unknown:
        raise UserError(
            f"{label} has unknown {executed_method} parameters: {sorted(unknown)}"
        )
    for key in ("action_seed", "sgr_seed"):
        if key in parameters and type(parameters[key]) is not int:
            raise UserError(f"{label}.{key} must be an exact integer")


def _expand_setting(method, setting, spec):
    method_spec = spec["methods"][method]
    setting_spec = method_spec["settings"][setting]
    method_fixed = _mapping(method_spec.get("fixed", {}), f"{method}.fixed")
    expanded = []

    grids = setting_spec.get("grids", [])
    if not isinstance(grids, list):
        raise UserError(f"{method}/{setting}.grids must be an array")
    for grid_index, grid_value in enumerate(grids):
        grid = _mapping(grid_value, f"{method}/{setting}.grids[{grid_index}]")
        grid_fixed = _mapping(
            grid.get("fixed", {}),
            f"{method}/{setting}.grids[{grid_index}].fixed",
        )
        axes = [key for key in grid if key != "fixed"]
        if not axes:
            raise UserError(f"{method}/{setting}.grids[{grid_index}] has no axes")
        for axis in axes:
            values = grid[axis]
            if not isinstance(values, list) or not values:
                raise UserError(
                    f"{method}/{setting}.grids[{grid_index}].{axis} "
                    "must be a nonempty array"
                )
        for values in itertools.product(*(grid[axis] for axis in axes)):
            candidate = dict(zip(axes, values))
            candidate = _merge_disjoint(
                candidate,
                grid_fixed,
                f"{method}/{setting}.grids[{grid_index}].fixed",
            )
            parameters = _merge_disjoint(
                method_fixed, candidate, f"{method}/{setting} candidate"
            )
            _validate_runtime_parameters(
                method, parameters, f"{method}/{setting} candidate"
            )
            expanded.append({
                "candidate": candidate,
                "parameters": parameters,
                "role": None,
            })

    points = setting_spec.get("points", [])
    if not isinstance(points, list):
        raise UserError(f"{method}/{setting}.points must be an array")
    for point_index, point_value in enumerate(points):
        raw = dict(_mapping(
            point_value, f"{method}/{setting}.points[{point_index}]"
        ))
        role = raw.pop("role", None)
        if role is not None and (not isinstance(role, str) or not role):
            raise UserError(f"{method}/{setting}.points[{point_index}].role is invalid")
        if not raw:
            raise UserError(f"{method}/{setting}.points[{point_index}] is empty")
        parameters = _merge_disjoint(
            method_fixed, raw, f"{method}/{setting}.points[{point_index}]"
        )
        _validate_runtime_parameters(
            method, parameters, f"{method}/{setting}.points[{point_index}]"
        )
        expanded.append({
            "candidate": raw,
            "parameters": parameters,
            "role": role,
        })

    keys = [canonical(item["parameters"]) for item in expanded]
    if len(keys) != len(set(keys)):
        raise UserError(f"{method}/{setting} contains duplicate candidates")
    expected = _positive_int(
        setting_spec.get("candidate_count"),
        f"{method}/{setting}.candidate_count",
    )
    if len(expanded) != expected:
        raise UserError(
            f"{method}/{setting} candidate count mismatch: "
            f"expected {expected}, expanded {len(expanded)}"
        )
    # This also rejects NaN/Infinity anywhere in the parameter objects.
    for item in expanded:
        try:
            canonical(item["parameters"])
        except (TypeError, ValueError) as error:
            raise UserError(f"invalid {method}/{setting} parameters: {error}") from error
    return expanded


def expand_candidates(method, setting, spec):
    if method not in active_tta_methods(spec):
        raise UserError(f"cannot expand control method {method}")
    if setting not in enabled_settings_for_method(spec, method):
        raise UserError(f"{method} is disabled for {setting}")
    return _expand_setting(method, setting, spec)


def control_candidates(method):
    if method == "source":
        return [{
            "candidate": {"action_selection": "argmax", "action_seed": 0},
            "parameters": {"action_selection": "argmax", "action_seed": 0},
            "role": "standard_argmax_source",
        }]
    if method == "feedtta_control":
        return [{
            "candidate": {"action_selection": "sample", "action_seed": 0},
            "parameters": {"action_selection": "sample", "action_seed": 0},
            "role": "feedtta_sampled_no_update",
        }]
    raise UserError(f"unknown control method {method}")


def _validate_spec(document):
    if document.get("schema") not in SPEC_SCHEMAS:
        raise UserError("unsupported local-refinement schema")
    protocol = _mapping(document.get("protocol"), "protocol")
    if protocol.get("benchmark") != "r2r" or protocol.get("split") != "val_seen":
        raise UserError("local refinement must be R2R val_seen")
    if protocol.get("episode_count") != 1021 or protocol.get("order_seed") != 0:
        raise UserError("local refinement must use 1,021 episodes and order seed 0")
    if protocol.get("settings") != list(SETTINGS):
        raise UserError("local refinement setting order mismatch")
    if protocol.get("full_split_only") is not True:
        raise UserError("local refinement must require the full split")
    selection = protocol.get("selection")
    if not isinstance(selection, list) or selection[:2] != ["higher_SR", "higher_SPL"]:
        raise UserError("local refinement must select by SR then SPL")
    if (
        document.get("schema") == SPEC_SCHEMA_V2
        and protocol.get("prefix_screening_forbidden") is not True
    ):
        raise UserError("post-fix search must forbid prefix screening")

    execution = _mapping(document.get("execution"), "execution")
    if not isinstance(execution.get("launchable"), bool):
        raise UserError("execution.launchable must be boolean")
    if (
        execution.get("launchable") is True
        and execution.get("barrier_implementation_status")
        != "implemented_validated"
    ):
        raise UserError(
            "launchable refinement requires implemented_validated barriers"
        )
    if execution.get("model_order") != list(SETTINGS):
        raise UserError("local refinement model order mismatch")
    enabled = enabled_phases_by_setting(document)
    reused_source_manifest = _load_reused_source_manifest(document)
    if any("source" in phases for phases in enabled.values()):
        raise UserError(
            "Source phases must be disabled when a reused Source manifest is configured"
        )
    for key in (
        "strict_model_barrier_required",
        "strict_method_barrier_required",
        "cross_model_overlap_forbidden",
        "cross_method_overlap_forbidden",
    ):
        if execution.get(key) is not True:
            raise UserError(f"execution.{key} must be true")

    active_methods = active_tta_methods(document)
    if not active_methods:
        raise UserError("local refinement enables no TTA methods")
    if (
        document.get("schema") == SPEC_SCHEMA_V2
        and active_methods != ("fstta", "feedtta")
    ):
        raise UserError("post-fix search must contain only FSTTA and FeedTTA")
    _exact_keys(document.get("methods"), active_methods, "methods")
    prior_key = (
        "search_priors"
        if document.get("schema") == SPEC_SCHEMA_V2
        else "parent_anchors"
    )
    _exact_keys(document.get(prior_key), active_methods, prior_key)
    budget = _mapping(document.get("budget"), "budget")
    search_by_method = {}
    search_by_setting = {setting: 0 for setting in SETTINGS}
    for method in active_methods:
        method_settings = enabled_settings_for_method(document, method)
        method_spec = document["methods"][method]
        _exact_keys(method_spec.get("settings"), method_settings, f"{method}.settings")
        _exact_keys(
            document[prior_key][method],
            method_settings,
            f"{prior_key}.{method}",
        )
        method_total = 0
        for setting in method_settings:
            points = _expand_setting(method, setting, document)
            method_total += len(points)
            search_by_setting[setting] += len(points)
            anchor = _mapping(
                document[prior_key][method][setting],
                f"{prior_key}.{method}.{setting}",
            )
            if document.get("schema") == SPEC_SCHEMA_V2:
                run_tag = anchor.get("prior_run_tag")
                candidate_key = "transformed_candidate"
                _mapping(
                    anchor.get("historical_parameters"),
                    f"{prior_key}.{method}.{setting}.historical_parameters",
                )
                _safe_relative_path(
                    anchor.get("prior_job_path"),
                    f"{prior_key}.{method}.{setting}.prior_job_path",
                )
                _sha256_string(
                    anchor.get("prior_job_sha256"),
                    f"{prior_key}.{method}.{setting}.prior_job_sha256",
                )
                _nonempty_string(
                    anchor.get("transfer_note"),
                    f"{prior_key}.{method}.{setting}.transfer_note",
                )
            else:
                run_tag = anchor.get("parent_run_tag")
                candidate_key = "candidate"
            safe_component(f"{prior_key} run_tag", run_tag)
            candidate = _mapping(
                anchor.get(candidate_key),
                f"{prior_key}.{method}.{setting}.{candidate_key}",
            )
            occurrences = sum(
                canonical(item["candidate"]) == canonical(candidate)
                for item in points
            )
            if occurrences != 1:
                raise UserError(
                    f"search prior {method}/{setting} occurs {occurrences} times"
                )
        declared = _positive_int(
            method_spec.get("candidate_count"), f"{method}.candidate_count"
        )
        if method_total != declared:
            raise UserError(
                f"{method} total mismatch: declared {declared}, expanded {method_total}"
            )
        search_by_method[method] = method_total

    controls = _mapping(document.get("controls"), "controls")
    if controls.get("standard_argmax_source_per_setting") is not True:
        raise UserError("standard argmax Source controls are required")
    if controls.get("standard_argmax_source_execution") != "reuse_completed_manifest":
        raise UserError(
            "standard argmax Source execution must reuse the completed manifest"
        )
    sampled_control_required = document.get("schema") == SPEC_SCHEMA_V1
    if controls.get("feedtta_sampled_no_update_per_setting") is not (
        sampled_control_required
    ):
        raise UserError(
            "FeedTTA sampled-control policy does not match the plan schema"
        )
    if document.get("schema") == SPEC_SCHEMA_V2 and controls.get(
        "feedtta_action_protocol"
    ) != "target_native_argmax":
        raise UserError(
            "post-fix FeedTTA search must use target_native_argmax"
        )
    if document.get("schema") == SPEC_SCHEMA_V2:
        supervision = _mapping(
            protocol.get("supervision_groups"), "protocol.supervision_groups"
        )
        if supervision != {
            "unsupervised": ["fstta"],
            "binary_feedback_supervised": ["feedtta"],
        }:
            raise UserError("post-fix supervision groups are invalid")
        fstta_fixed = _mapping(
            document["methods"]["fstta"].get("fixed"), "fstta.fixed"
        )
        required_fstta = {
            "episodic": False,
            "norm_scope": "last_k_ln",
            "last_k_ln": 4,
            "fast_grad_mode": "concordant",
            "use_fast_lr_scaler": True,
            "use_slow": True,
        }
        if any(fstta_fixed.get(key) != value for key, value in required_fstta.items()):
            raise UserError("post-fix FSTTA fixed protocol is invalid")
        feedtta_fixed = _mapping(
            document["methods"]["feedtta"].get("fixed"), "feedtta.fixed"
        )
        required_feedtta = {
            "action_selection": "argmax",
            "sgr_seed": 0,
            "episodic": False,
        }
        if any(
            feedtta_fixed.get(key) != value
            for key, value in required_feedtta.items()
        ):
            raise UserError("post-fix FeedTTA fixed protocol is invalid")
        allowed_scopes = {"paper_full", "last_crossmodal", "action_head"}
        for setting in SETTINGS:
            candidates = expand_candidates("feedtta", setting, document)
            if any(
                item["parameters"].get("action_selection") != "argmax"
                or item["parameters"].get("scope_profile") not in allowed_scopes
                for item in candidates
            ):
                raise UserError(
                    f"post-fix FeedTTA candidate protocol is invalid for {setting}"
                )
        published_fstta_anchor = {
            "lr_fast": 0.0006, "lr_slow": 0.001, "m": 3, "n": 4,
        }
        duet_candidates = expand_candidates("fstta", "duet-r2r", document)
        if sum(
            all(item["parameters"].get(key) == value for key, value in (
                published_fstta_anchor.items()
            ))
            for item in duet_candidates
        ) != 1:
            raise UserError("post-fix FSTTA search lacks the published DUET anchor")
    control_count = sum(
        method in CONTROL_METHODS
        for phases in enabled.values()
        for method in phases
    )
    if controls.get("candidate_count") != control_count:
        raise UserError("control count does not match enabled control phases")

    activation = _mapping(document.get("activation"), "activation")
    if activation.get("automatic_launch_forbidden") is not True:
        raise UserError("activation must forbid automatic launch")
    if activation.get("manual_review_required") is not True:
        raise UserError("activation must require manual review")

    if budget.get("search_candidates") != search_by_method:
        raise UserError("budget.search_candidates mismatch")
    if budget.get("by_setting_before_controls") != search_by_setting:
        raise UserError("budget.by_setting_before_controls mismatch")
    search_total = sum(search_by_method.values())
    if budget.get("search_total") != search_total:
        raise UserError("budget.search_total mismatch")
    if budget.get("controls") != controls["candidate_count"]:
        raise UserError("budget.controls mismatch")
    if budget.get("total_before_confirmation") != (
        search_total + controls["candidate_count"]
    ):
        raise UserError("budget.total_before_confirmation mismatch")

    control_execution = _mapping(
        execution.get("control_execution"), "execution.control_execution"
    )
    enabled_control_methods = tuple(
        method for method in CONTROL_METHODS
        if any(method in phases for phases in enabled.values())
    )
    _exact_keys(
        control_execution,
        enabled_control_methods,
        "execution.control_execution",
    )
    for method in enabled_control_methods:
        value = _mapping(
            control_execution[method], f"execution.control_execution.{method}"
        )
        if value.get("max_workers") != 1:
            raise UserError(f"{method} control must use exactly one worker")

    gpu_safety = _mapping(execution.get("gpu_safety"), "execution.gpu_safety")
    planned_mib = _positive_int(
        gpu_safety.get("production_planned_steady_used_mib_max"),
        "execution.gpu_safety.production_planned_steady_used_mib_max",
    )
    emergency_mib = _positive_int(
        gpu_safety.get("emergency_abort_observed_used_mib"),
        "execution.gpu_safety.emergency_abort_observed_used_mib",
    )
    if planned_mib >= emergency_mib:
        raise UserError("planned GPU memory must be below the emergency abort line")
    if gpu_safety.get("raise_concurrency_one_worker_at_a_time") is not False:
        raise UserError("grouped calibration policy must disable one-step ramps")
    if gpu_safety.get("calibration_initial_group_workers") != 5:
        raise UserError("calibration must start from a five-worker group")
    if gpu_safety.get("calibration_episode_limit") != 100:
        raise UserError("calibration must use the 100-episode prefix")
    if gpu_safety.get("grouped_jump_after_measured_baseline") is not True:
        raise UserError("grouped calibration jumps must require a baseline")
    if (
        gpu_safety.get("grouped_jump_requires_independent_steady_validation")
        is not True
    ):
        raise UserError("each grouped calibration jump must be revalidated")
    projection_mib = _positive_int(
        gpu_safety.get("grouped_jump_projection_max_mib"),
        "execution.gpu_safety.grouped_jump_projection_max_mib",
    )
    if projection_mib >= planned_mib:
        raise UserError("grouped jump projection must stay below the plan line")
    projection_margin = gpu_safety.get(
        "grouped_jump_per_worker_margin_factor"
    )
    if (
        isinstance(projection_margin, bool)
        or not isinstance(projection_margin, (int, float))
        or not 1.0 <= float(projection_margin) <= 1.5
    ):
        raise UserError(
            "execution.gpu_safety grouped jump margin must be in [1, 1.5]"
        )
    gate_range = gpu_safety.get("scheduler_prelaunch_gate_mib_range")
    if (
        not isinstance(gate_range, list)
        or len(gate_range) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in gate_range
        )
        or gate_range[0] > gate_range[1]
        or gate_range[1] >= planned_mib
    ):
        raise UserError("execution.gpu_safety prelaunch gate range is invalid")

    calibration = _mapping(
        execution.get("concurrency_calibration"),
        "execution.concurrency_calibration",
    )
    _exact_keys(calibration, active_methods, "execution.concurrency_calibration")
    for method in active_methods:
        method_settings = enabled_settings_for_method(document, method)
        method_calibration = dict(calibration[method])
        method_calibration.pop("seven_workers_forbidden", None)
        _exact_keys(
            method_calibration,
            method_settings,
            f"execution.concurrency_calibration.{method}",
        )
        for setting in method_settings:
            value = _mapping(
                method_calibration[setting],
                f"execution.concurrency_calibration.{method}.{setting}",
            )
            limit = value.get(
                "production_cap",
                value.get("default_production_cap_before_pure_test",
                          value.get("calibration_baseline")),
            )
            _positive_int(limit, f"{method}/{setting} safe worker limit")
            _positive_int(value.get("prelaunch_gate_mib"),
                          f"{method}/{setting} prelaunch gate")
            _positive_int(value.get("estimated_mib_per_job"),
                          f"{method}/{setting} estimated MiB per job")
            steps = value.get("conditional_test_steps", [])
            if not isinstance(steps, list):
                raise UserError(
                    f"{method}/{setting}.conditional_test_steps must be an array"
                )
            for step in steps:
                _positive_int(step, f"{method}/{setting} conditional worker limit")
    if document.get("schema") == SPEC_SCHEMA_V2:
        exceptions = gpu_safety.get("approved_projection_exceptions")
        if exceptions != ["fstta/goat-r2r", "feedtta/goat-r2r"]:
            raise UserError("post-fix GPU projection exceptions are invalid")
        exception_set = set(exceptions)
        for method in active_methods:
            for setting in enabled_settings_for_method(document, method):
                value = calibration[method][setting]
                key = f"{method}/{setting}"
                if key in exception_set:
                    _positive_int(
                        value.get("projected_peak_mib"),
                        f"{key}.projected_peak_mib",
                    )
                    if "projection" not in str(value.get("cap_basis", "")):
                        raise UserError(f"{key} must declare a projection cap basis")
                else:
                    _positive_int(
                        value.get("observed_peak_mib"),
                        f"{key}.observed_peak_mib",
                    )
    return reused_source_manifest


def load_spec(path=DEFAULT_SPEC):
    path = Path(path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"invalid local-refinement spec {path}: {error}") from error
    reused_source_manifest = _validate_spec(document)
    document["setting_episode_counts"] = {
        setting: int(document["protocol"]["episode_count"])
        for setting in SETTINGS
    }
    document["_path"] = str(path)
    document["_sha256"] = sha256(path)
    document["_reused_source_manifest"] = reused_source_manifest
    return document


def phase_sequence(spec):
    phases = []
    global_ordinal = 0
    enabled = enabled_phases_by_setting(spec)
    for setting in spec["execution"]["model_order"]:
        for method in enabled[setting]:
            count = (
                1 if method in CONTROL_METHODS
                else len(expand_candidates(method, setting, spec))
            )
            index = len(phases)
            phase_id = f"{index:02d}-{setting}-{method}"
            phases.append({
                "index": index,
                "phase_id": phase_id,
                "setting": setting,
                "model": SETTING_MODEL[setting],
                "method": method,
                "job_count": count,
                "global_ordinal_start": global_ordinal,
            })
            global_ordinal += count
    expected = int(spec["budget"]["total_before_confirmation"])
    if global_ordinal != expected:
        raise UserError(
            f"phase job total mismatch: expected {expected}, got {global_ordinal}"
        )
    return phases


def campaign_root(batch_id):
    safe_component("batch_id", batch_id)
    return LOG_ROOT / batch_id


def phase_root(batch_id, phase):
    return campaign_root(batch_id) / "phases" / phase["phase_id"]


def tuning_job_parent(batch_id, model, method):
    for label, value in (("batch_id", batch_id), ("model", model), ("method", method)):
        safe_component(label, value)
    return TUNING_ROOT / batch_id / model / method / "jobs"


def _config_method(method):
    return "source" if method in CONTROL_METHODS else method


def _point_digest(config_method, parameters):
    payload = {"method": config_method, "parameters": parameters}
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()[:10]


def _phase_candidates(phase, spec):
    method = phase["method"]
    return (
        control_candidates(method)
        if method in CONTROL_METHODS
        else expand_candidates(method, phase["setting"], spec)
    )


def build_phase_jobs(phase, batch_id, spec, gpu=0):
    candidates = _phase_candidates(phase, spec)
    if len(candidates) != phase["job_count"]:
        raise UserError(f"{phase['phase_id']} candidate count changed")
    root = phase_root(batch_id, phase)
    jobs = []
    prior = None
    if phase["method"] not in CONTROL_METHODS:
        prior_key = (
            "search_priors"
            if spec.get("schema") == SPEC_SCHEMA_V2
            else "parent_anchors"
        )
        prior = spec[prior_key][phase["method"]][phase["setting"]]
    for point_index, item in enumerate(candidates):
        method = phase["method"]
        parameters = dict(item["parameters"])
        executed_method = _config_method(method)
        digest = _point_digest(executed_method, parameters)
        run_tag = (
            f"{batch_id}-refine-{phase['index']:02d}-{method}-{point_index:04d}-"
            f"{phase['setting']}-{digest}"
        )
        safe_component("run_tag", run_tag)
        job_dir = root / "jobs" / run_tag
        config_path = job_dir / "parameters.json"
        result_parent = tuning_job_parent(
            batch_id, phase["model"], method
        )
        result_root = result_parent / run_tag / "val_seen"
        command = [
            str(RUNNER), phase["setting"], "val_seen", str(gpu),
            "--run-tag", run_tag,
            "--tta-config", str(config_path),
            "--result-root", str(result_root),
        ]
        parent_run_tags = []
        search_prior = None
        if prior is not None:
            prior_candidate_key = (
                "transformed_candidate"
                if spec.get("schema") == SPEC_SCHEMA_V2 else "candidate"
            )
            if canonical(item["candidate"]) == canonical(
                prior[prior_candidate_key]
            ):
                if spec.get("schema") == SPEC_SCHEMA_V2:
                    search_prior = {
                        "run_tag": prior["prior_run_tag"],
                        "job_path": prior["prior_job_path"],
                        "job_sha256": prior["prior_job_sha256"],
                        "historical_parameters": prior["historical_parameters"],
                        "transfer_note": prior["transfer_note"],
                    }
                else:
                    parent_run_tags.append(prior["parent_run_tag"])
        jobs.append({
            "batch_id": batch_id,
            "ordinal": point_index,
            "campaign_ordinal": phase["global_ordinal_start"] + point_index,
            "point_index": point_index,
            "phase_index": phase["index"],
            "phase_id": phase["phase_id"],
            "base_run_tag": run_tag,
            "run_tag": run_tag,
            "attempt": 0,
            "setting": phase["setting"],
            "model": phase["model"],
            "family": "discrete",
            "benchmark": "r2r",
            "search_method": method,
            "config_method": executed_method,
            "result_layout": RESULT_LAYOUT,
            "result_namespace": method,
            "stage": STAGE,
            "config_stage": None,
            "episodes": -1,
            "order_seed": None,
            "parameters": parameters,
            "candidate_role": item.get("role"),
            "parent_run_tags": parent_run_tags,
            "search_prior": search_prior,
            "config_path": str(config_path),
            "job_dir": str(job_dir),
            "result_root": str(result_root),
            "retry_result_root_parent": str(result_parent),
            "command": command,
        })
    return jobs


def _safe_worker_limit(phase, spec):
    method = phase["method"]
    if method in CONTROL_METHODS:
        value = spec["execution"]["control_execution"][method]
        return _positive_int(value.get("max_workers"), f"{method}.max_workers")
    value = spec["execution"]["concurrency_calibration"][method][phase["setting"]]
    limit = value.get(
        "production_cap",
        value.get("default_production_cap_before_pure_test",
                  value.get("calibration_baseline")),
    )
    return _positive_int(limit, f"{method}/{phase['setting']} safe worker limit")


def _declared_worker_limits(phase, spec):
    method = phase["method"]
    if method in CONTROL_METHODS:
        return {_safe_worker_limit(phase, spec)}
    value = spec["execution"]["concurrency_calibration"][method][phase["setting"]]
    limits = set()
    for key in (
        "calibration_baseline", "default_production_cap_before_pure_test",
        "production_cap",
    ):
        if value.get(key) is not None:
            limits.add(_positive_int(value[key], f"{method}/{phase['setting']}.{key}"))
    steps = value.get("conditional_test_steps", [])
    if not isinstance(steps, list):
        raise UserError(f"{method}/{phase['setting']}.conditional_test_steps is invalid")
    limits.update(
        _positive_int(item, f"{method}/{phase['setting']} conditional worker limit")
        for item in steps
    )
    return limits


def _configured_worker_limit(cli, phase, spec):
    limit = _safe_worker_limit(phase, spec)
    overrides = getattr(cli, "phase_max_workers", {}) or {}
    if phase["phase_id"] in overrides:
        requested = overrides[phase["phase_id"]]
        allowed = _declared_worker_limits(phase, spec)
        if requested not in allowed:
            raise UserError(
                f"phase worker override {phase['phase_id']}={requested} is not "
                f"declared by the spec; allowed {sorted(allowed)}"
            )
        limit = requested
    global_cap = getattr(cli, "max_workers", None)
    if global_cap is not None:
        limit = min(limit, global_cap)
    return limit


def _prelaunch_gate(phase, spec):
    if phase["method"] in CONTROL_METHODS:
        return min(spec["execution"]["gpu_safety"]["scheduler_prelaunch_gate_mib_range"])
    value = spec["execution"]["concurrency_calibration"][phase["method"]][
        phase["setting"]
    ]
    return int(value["prelaunch_gate_mib"])


def _configured_prelaunch_gate(cli, phase, spec):
    gate = _prelaunch_gate(phase, spec)
    global_cap = getattr(cli, "max_gpu_memory_mib", None)
    return min(gate, global_cap) if global_cap is not None else gate


def _calibration_policy(phase, spec):
    """Persist the complete resource-sizing policy used by the helper."""
    if phase["method"] in CONTROL_METHODS:
        return None
    safety = spec["execution"]["gpu_safety"]
    method_policy = spec["execution"]["concurrency_calibration"][
        phase["method"]
    ][phase["setting"]]
    return {
        "mode": "measured_grouped_jump_v1",
        "initial_group_workers": int(
            safety["calibration_initial_group_workers"]
        ),
        "episode_limit": int(safety["calibration_episode_limit"]),
        "projection_max_mib": int(
            safety["grouped_jump_projection_max_mib"]
        ),
        "projection_safety_factor": float(
            safety["grouped_jump_per_worker_margin_factor"]
        ),
        "estimated_mib_per_job": int(method_policy["estimated_mib_per_job"]),
        "allowed_worker_counts": sorted(_declared_worker_limits(phase, spec)),
        "independent_steady_validation": bool(
            safety["grouped_jump_requires_independent_steady_validation"]
        ),
        "steady_used_mib_stop": int(
            safety["production_planned_steady_used_mib_max"]
        ),
        "observed_used_mib_abort": int(
            safety["emergency_abort_observed_used_mib"]
        ),
    }


def _is_relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _read_json_object(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise UserError(f"{label} must be a JSON object: {path}")
    return value


def _calibration_config_identity(document):
    return {
        key: value for key, value in document.items()
        if key not in ("batch_id", "run_tag")
    }


def _validated_worker_calibration(path, phase, jobs, requested, spec, cli):
    """Bind a raised formal worker limit to a completed grouped calibration."""
    path = Path(path).expanduser().resolve()
    if path.name != "CALIBRATION.json" or not _is_relative_to(path, LOG_ROOT):
        raise UserError(
            "phase calibration must be a CALIBRATION.json below the R2R log root"
        )
    summary = _read_json_object(path, "phase calibration summary")
    plan_path = path.parent / "CALIBRATION_PLAN.json"
    calibration_plan = _read_json_object(
        plan_path, "phase calibration plan"
    )
    if summary.get("schema") != CALIBRATION_SCHEMA:
        raise UserError("phase calibration summary schema is unsupported")
    if calibration_plan.get("schema") != CALIBRATION_PLAN_SCHEMA:
        raise UserError("phase calibration plan schema is unsupported")
    resource_path = (path.parent / str(summary.get("resource_csv", ""))).resolve()
    if (
        summary.get("resource_csv") != "resource.csv"
        or resource_path.parent != path.parent
        or not resource_path.is_file()
        or sha256(resource_path) != summary.get("resource_csv_sha256")
    ):
        raise UserError("phase calibration resource CSV digest mismatch")
    try:
        with resource_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CALIBRATION_RESOURCE_FIELDS:
                raise UserError("phase calibration resource CSV header mismatch")
            resource_rows = list(reader)
        resource_peak_mib = max(
            float(row["gpu_memory_mib"]) for row in resource_rows
        )
        resource_max_active = max(
            int(row["active_count"]) for row in resource_rows
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise UserError("phase calibration resource CSV is invalid") from error
    if not resource_rows or any(
        row.get("phase_id") != phase["phase_id"]
        or row.get("setting") != phase["setting"]
        or row.get("method") != phase["method"]
        for row in resource_rows
    ):
        raise UserError("phase calibration resource CSV identity mismatch")
    identity = {
        "phase_id": phase["phase_id"],
        "setting": phase["setting"],
        "model": phase["model"],
        "method": phase["method"],
    }
    for key, expected in identity.items():
        if summary.get(key) != expected or calibration_plan.get(key) != expected:
            raise UserError(f"phase calibration identity mismatch for {key}")
    if summary.get("calibration_id") != calibration_plan.get("calibration_id"):
        raise UserError("phase calibration id does not match its plan")

    policy = _calibration_policy(phase, spec)
    expected_commit = git("rev-parse", "HEAD")
    for document, label in ((summary, "summary"), (calibration_plan, "plan")):
        if document.get("source_git_commit") != expected_commit:
            raise UserError(f"phase calibration {label} Git commit mismatch")
        if document.get("source_spec_sha256") != spec["_sha256"]:
            raise UserError(f"phase calibration {label} spec digest mismatch")
        if document.get("gpu") != cli.gpu:
            raise UserError(f"phase calibration {label} GPU mismatch")
        if document.get("calibration_policy") != policy:
            raise UserError(f"phase calibration {label} policy mismatch")
        if document.get("episode_limit") != policy["episode_limit"] or (
            document.get("episode_budget_per_worker")
            != policy["episode_limit"]
        ):
            raise UserError(f"phase calibration {label} episode budget mismatch")
        if document.get("target_workers") != requested or (
            document.get("initial_workers") != requested
        ):
            raise UserError(f"phase calibration {label} worker count mismatch")

    if (
        summary.get("status") != "completed"
        or summary.get("stop_reason") != "target_completed"
        or summary.get("recommended_cap") != requested
        or summary.get("launched_workers") != requested
        or summary.get("successful_workers") != requested
        or summary.get("failed_workers") != 0
        or summary.get("unlaunched_workers") != 0
    ):
        raise UserError("phase calibration did not completely validate the limit")
    try:
        peak_mib = float(summary["peak_observed_gpu_memory_mib"])
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("phase calibration lacks valid peak VRAM") from error
    if (
        not math.isfinite(peak_mib)
        or not 0 < peak_mib < 29_000
        or resource_peak_mib != peak_mib
        or resource_max_active < requested
    ):
        raise UserError("phase calibration peak VRAM is outside the safe line")
    initial_group = summary.get("initial_group")
    if not isinstance(initial_group, dict) or initial_group.get(
        "steady_confirmed"
    ) is not True:
        raise UserError("phase calibration group was not steady-confirmed")

    levels = [
        value for value in summary.get("levels", [])
        if isinstance(value, dict) and value.get("active_count") == requested
    ]
    if len(levels) != 1 or levels[0].get("steady_confirmed") is not True:
        raise UserError("phase calibration lacks one confirmed target level")
    baseline = int(policy["initial_group_workers"])
    expected_mode = "fresh_bootstrap" if requested == baseline else "sizing_jump"
    expected_origin = (
        "current_bootstrap_group"
        if requested == baseline else "current_sizing_jump_group"
    )
    if summary.get("initial_group_mode") != expected_mode or (
        levels[0].get("evidence_origin") != expected_origin
    ):
        raise UserError("phase calibration is not independent grouped evidence")
    if requested > baseline:
        parent = summary.get("sizing_parent")
        projection = summary.get("sizing_projection")
        if not isinstance(parent, dict) or not isinstance(projection, dict):
            raise UserError("raised phase calibration lacks sizing evidence")
        if projection.get("accepted") is not True:
            raise UserError("raised phase calibration projection was not accepted")
        for parent_key, digest_key in (
            ("path", "sha256"), ("plan_path", "plan_sha256")
        ):
            parent_path = Path(str(parent.get(parent_key, ""))).resolve()
            if not _is_relative_to(parent_path, LOG_ROOT) or not parent_path.is_file():
                raise UserError("phase calibration sizing parent path is invalid")
            if sha256(parent_path) != parent.get(digest_key):
                raise UserError("phase calibration sizing parent digest mismatch")

    planned_jobs = calibration_plan.get("jobs")
    summary_jobs = summary.get("jobs")
    if (
        not isinstance(planned_jobs, list)
        or len(planned_jobs) != requested
        or not isinstance(summary_jobs, list)
        or len(summary_jobs) != requested
        or len(jobs) < requested
    ):
        raise UserError("phase calibration job count mismatch")
    ordered_summary_jobs = sorted(
        summary_jobs, key=lambda item: item.get("worker_index", -1)
    )
    if [item.get("worker_index") for item in ordered_summary_jobs] != list(
        range(1, requested + 1)
    ) or any(
        item.get("exit_code") != 0
        or item.get("episode_budget") != policy["episode_limit"]
        for item in ordered_summary_jobs
    ):
        raise UserError("phase calibration contains an incomplete worker")
    if [item.get("source_run_tag") for item in ordered_summary_jobs] != [
        item.get("source_run_tag") for item in planned_jobs
    ]:
        raise UserError("phase calibration summary/plan job prefix mismatch")

    formal_identities = []
    for index in range(requested):
        formal_config = staged._job_config(jobs[index])
        if formal_config.get("episodes") != -1:
            raise UserError("formal refinement config must retain episodes=-1")
        formal_config["episodes"] = policy["episode_limit"]
        formal_identity = _calibration_config_identity(formal_config)
        formal_identities.append(formal_identity)

        planned_job = planned_jobs[index]
        planned_job_dir = Path(str(planned_job.get("job_dir", ""))).resolve()
        if not _is_relative_to(planned_job_dir, path.parent / "jobs"):
            raise UserError("phase calibration job directory escapes its artifact")
        try:
            exit_code = int(
                (planned_job_dir / "exitcode")
                .read_text(encoding="utf-8")
                .strip()
            )
        except (OSError, ValueError) as error:
            raise UserError("phase calibration job exitcode is invalid") from error
        if exit_code != 0:
            raise UserError("phase calibration job did not exit successfully")
        command = planned_job.get("command")
        if not isinstance(command, list) or command.count("--episode-limit") != 1:
            raise UserError("phase calibration command lacks one episode limit")
        option_index = command.index("--episode-limit")
        if (
            option_index + 1 >= len(command)
            or command[option_index + 1] != str(policy["episode_limit"])
        ):
            raise UserError("phase calibration command episode limit mismatch")
        calibration_config = _read_json_object(
            planned_job.get("config_path"), "phase calibration config"
        )
        if calibration_config.get("episodes") != policy["episode_limit"]:
            raise UserError("phase calibration config episode limit mismatch")
        calibration_identity = _calibration_config_identity(calibration_config)
        if calibration_identity != formal_identity:
            raise UserError(
                f"phase calibration candidate {index + 1} config mismatch"
            )
    identity_sha256 = hashlib.sha256(
        canonical(formal_identities).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "plan_path": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "calibration_id": summary["calibration_id"],
        "target_workers": requested,
        "peak_observed_gpu_memory_mib": peak_mib,
        "source_job_config_identity_sha256": identity_sha256,
    }


def _validate_worker_calibration_overrides(cli, phases, spec):
    phase_by_id = {phase["phase_id"]: phase for phase in phases}
    overrides = getattr(cli, "phase_max_workers", {}) or {}
    evidence_paths = getattr(cli, "phase_calibrations", {}) or {}
    unknown = set(evidence_paths).difference(phase_by_id)
    if unknown:
        raise UserError(f"unknown phase calibration evidence: {sorted(unknown)}")
    extraneous = set(evidence_paths).difference(overrides)
    if extraneous:
        raise UserError(
            "phase calibration evidence requires matching worker overrides: "
            f"{sorted(extraneous)}"
        )
    bindings = {}
    for phase_id, requested in overrides.items():
        phase = phase_by_id[phase_id]
        safe_default = _safe_worker_limit(phase, spec)
        evidence_path = evidence_paths.get(phase_id)
        if requested > safe_default and evidence_path is None:
            raise UserError(
                f"raised worker limit {phase_id}={requested} requires "
                "--phase-calibration"
            )
        if evidence_path is not None:
            jobs = build_phase_jobs(phase, cli.batch_id, spec, gpu=cli.gpu)
            bindings[phase_id] = _validated_worker_calibration(
                evidence_path, phase, jobs, requested, spec, cli
            )
    return bindings


def phase_manifest(phase, batch_id, jobs, spec, cli):
    return {
        "schema": PHASE_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        **phase,
        "benchmark": "r2r",
        "split": "val_seen",
        "episode_count": 1021,
        "order_seed": 0,
        "gpu": cli.gpu,
        "git_commit": git("rev-parse", "HEAD"),
        "spec_path": spec["_path"],
        "spec_sha256": spec["_sha256"],
        "reused_source_manifest": _reused_source_plan_binding(spec),
        "result_layout": RESULT_LAYOUT,
        "safe_worker_limit": _safe_worker_limit(phase, spec),
        "effective_worker_limit": _configured_worker_limit(cli, phase, spec),
        "declared_worker_limits": sorted(_declared_worker_limits(phase, spec)),
        "calibration_policy": _calibration_policy(phase, spec),
        "worker_calibration": (
            getattr(cli, "worker_calibration_bindings", {}) or {}
        ).get(phase["phase_id"]),
        "prelaunch_gate_mib": _prelaunch_gate(phase, spec),
        "effective_prelaunch_gate_mib": _configured_prelaunch_gate(
            cli, phase, spec
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def campaign_manifest(batch_id, phases, spec, cli):
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "benchmark": "r2r",
        "split": "val_seen",
        "episode_count": 1021,
        "order_seed": 0,
        "gpu": cli.gpu,
        "git_commit": git("rev-parse", "HEAD"),
        "spec_path": spec["_path"],
        "spec_sha256": spec["_sha256"],
        "reused_source_manifest": _reused_source_plan_binding(spec),
        "result_layout": RESULT_LAYOUT,
        "strict_model_barrier": True,
        "strict_method_barrier": True,
        "job_count": sum(phase["job_count"] for phase in phases),
        "phases": phases,
        "runtime": {
            "phase_max_workers": dict(sorted(
                (getattr(cli, "phase_max_workers", {}) or {}).items()
            )),
            "phase_calibrations": dict(sorted(
                (getattr(cli, "worker_calibration_bindings", {}) or {}).items()
            )),
            "global_max_workers": cli.max_workers,
            "global_max_gpu_memory_mib": cli.max_gpu_memory_mib,
            "max_memory_gib": cli.max_memory_gib,
            "launch_stagger_seconds": cli.launch_stagger,
            "resource_wait_timeout_seconds": cli.resource_wait_timeout,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _validate_persisted_jobs(phase, batch_id, jobs, spec, gpu):
    expected = build_phase_jobs(phase, batch_id, spec, gpu=gpu)
    if len(jobs) != len(expected):
        raise UserError(f"persisted {phase['phase_id']} job count mismatch")
    immutable = (
        "batch_id", "ordinal", "campaign_ordinal", "point_index",
        "phase_index", "phase_id", "base_run_tag", "setting", "model",
        "family", "benchmark", "search_method", "config_method",
        "result_layout", "result_namespace", "stage", "config_stage",
        "episodes", "order_seed", "parameters", "candidate_role",
        "parent_run_tags", "search_prior", "retry_result_root_parent",
    )
    for actual, planned in zip(jobs, expected):
        for key in immutable:
            if actual.get(key) != planned.get(key):
                raise UserError(
                    f"persisted {phase['phase_id']} job "
                    f"{actual.get('ordinal')} mismatch for {key}"
                )
        for key in ("job_dir", "config_path", "retry_result_root_parent"):
            if Path(actual.get(key, "")) != Path(planned[key]):
                raise UserError(f"persisted {phase['phase_id']} {key} mismatch")
        attempt = actual.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise UserError("persisted refinement attempt is invalid")
        expected_run_tag = planned["base_run_tag"]
        if attempt:
            expected_run_tag += f"-retry{attempt}"
        if actual.get("run_tag") != expected_run_tag:
            raise UserError("persisted refinement retry run_tag mismatch")
        expected_result_root = (
            Path(planned["result_root"])
            if attempt == 0 else
            Path(planned["retry_result_root_parent"]) / expected_run_tag / "val_seen"
        )
        if Path(actual.get("result_root", "")) != expected_result_root:
            raise UserError("persisted refinement result_root mismatch")
        expected_command = list(planned["command"])
        expected_command[expected_command.index("--run-tag") + 1] = expected_run_tag
        expected_command[expected_command.index("--result-root") + 1] = str(
            expected_result_root
        )
        if actual.get("command") != expected_command:
            raise UserError("persisted refinement command mismatch")
        try:
            runtime_config = json.loads(
                Path(actual["config_path"]).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise UserError(f"invalid persisted refinement config: {error}") from error
        if runtime_config != staged._job_config(actual):
            raise UserError("persisted refinement runtime config mismatch")
        for prior_attempt in range(attempt):
            archived = Path(actual["job_dir"]) / "attempts" / (
                f"attempt-{prior_attempt:02d}"
            ) / "job.json"
            if not archived.is_file():
                raise UserError("persisted refinement retry archive mismatch")


def _manifest_matches(actual, expected, label):
    for key, value in expected.items():
        if key == "created_at":
            continue
        if actual.get(key) != value:
            raise UserError(f"persisted {label} mismatch for {key}")


def ensure_phase_plan(phase, batch_id, spec, cli, resume):
    root = phase_root(batch_id, phase)
    manifest_path = root / "PHASE.json"
    expected_jobs = build_phase_jobs(phase, batch_id, spec, gpu=cli.gpu)
    expected_manifest = phase_manifest(phase, batch_id, expected_jobs, spec, cli)
    if manifest_path.is_file():
        if not resume:
            raise UserError(f"phase exists; use --resume: {phase['phase_id']}")
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserError(f"invalid {manifest_path}: {error}") from error
        _manifest_matches(actual, expected_manifest, phase["phase_id"])
        jobs = staged.load_jobs(root)
        _validate_persisted_jobs(phase, batch_id, jobs, spec, cli.gpu)
        return root, jobs
    if root.exists() and any(root.iterdir()):
        raise UserError(f"nonempty phase without PHASE.json: {root}")
    root.mkdir(parents=True, exist_ok=True)
    staged.write_plan(root, expected_jobs)
    staged.atomic_json(manifest_path, expected_manifest)
    staged.atomic_json(root / "progress.json", {
        "planned": len(expected_jobs),
        "pending": len(expected_jobs),
        "running": 0,
        "succeeded": 0,
        "failed": 0,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "running_jobs": [],
    })
    return root, expected_jobs


def ensure_campaign_plan(cli, spec):
    phases = phase_sequence(spec)
    known_phases = {phase["phase_id"] for phase in phases}
    unknown_overrides = set(
        (getattr(cli, "phase_max_workers", {}) or {})
    ).difference(known_phases)
    if unknown_overrides:
        raise UserError(
            f"unknown phase worker overrides: {sorted(unknown_overrides)}"
        )
    cli.worker_calibration_bindings = _validate_worker_calibration_overrides(
        cli, phases, spec
    )
    root = campaign_root(cli.batch_id)
    manifest_path = root / "PLAN.json"
    expected = campaign_manifest(cli.batch_id, phases, spec, cli)
    if manifest_path.is_file():
        if not cli.resume:
            raise UserError("campaign exists; use --resume")
        try:
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserError(f"invalid campaign PLAN.json: {error}") from error
        _manifest_matches(actual, expected, "campaign PLAN.json")
    else:
        if root.exists() and any(root.iterdir()):
            raise UserError("nonempty campaign root without PLAN.json")
        root.mkdir(parents=True, exist_ok=True)
        staged.atomic_json(manifest_path, expected)
    planned = []
    for phase in phases:
        phase_path, jobs = ensure_phase_plan(
            phase, cli.batch_id, spec, cli, resume=manifest_path.is_file()
        )
        planned.append((phase, phase_path, jobs))
    return root, planned


def _load_validated_results(root, spec, phase=None):
    jobs = staged.load_jobs(root)
    results, errors = staged.write_summary(root, jobs, spec)
    if len(results) + len(errors) != len(jobs) or errors:
        raise UserError(f"refinement phase is incomplete: {root}")
    if phase is not None:
        for result in results:
            _validate_postfix_result_contract(phase, result, spec)
    return results


def _phase_complete(root, spec, phase=None):
    try:
        _load_validated_results(root, spec, phase=phase)
        return True
    except (UserError, staged.UserError):
        return False


def _load_phase_results(root, spec, phase):
    if spec.get("schema") == SPEC_SCHEMA_V2:
        return _load_validated_results(root, spec, phase)
    return _load_validated_results(root, spec)


def _phase_complete_for_spec(root, spec, phase):
    if spec.get("schema") == SPEC_SCHEMA_V2:
        return _phase_complete(root, spec, phase)
    return _phase_complete(root, spec)


def _result_score(result):
    metrics = result.get("metrics", {})
    if "SR" not in metrics or "SPL" not in metrics:
        raise UserError(f"candidate lacks SR/SPL: {result.get('run_tag')}")
    adapter = result.get("adapter_diagnostics") or {}
    drift = float(adapter.get("relative_param_drift", math.inf))
    updates = float(adapter.get("updates", math.inf))
    return (
        float(metrics["SR"]),
        float(metrics["SPL"]),
        -drift,
        -updates,
        canonical(result["parameters"]),
    )


def _validate_postfix_result_contract(phase, result, spec):
    if spec.get("schema") != SPEC_SCHEMA_V2:
        return
    adapter = result.get("adapter_diagnostics") or {}
    method = phase["method"]
    if method == "fstta":
        if adapter.get("variance_history_lifetime") != "test_stream":
            raise UserError(
                f"{result['run_tag']} did not preserve FSTTA stream variance"
            )
        return
    if method != "feedtta":
        return

    parameters = result.get("parameters") or {}
    if parameters.get("action_selection") != "argmax":
        raise UserError(f"{result['run_tag']} is not target-native argmax")
    if adapter.get("action_selection_protocol") != "target_native_argmax":
        raise UserError(
            f"{result['run_tag']} adapter action protocol is not argmax"
        )
    expected_episodes = int(spec["protocol"]["episode_count"])
    feedback_episodes = int(adapter.get("feedback_episodes", -1))
    successes = int(adapter.get("successful_feedback_episodes", -1))
    failures = int(adapter.get("failed_feedback_episodes", -1))
    if feedback_episodes != expected_episodes or successes + failures != (
        expected_episodes
    ):
        raise UserError(f"{result['run_tag']} has incomplete FeedTTA feedback")
    metric_successes = int(round(
        float(result["metrics"]["SR"]) * expected_episodes / 100.0
    ))
    if successes != metric_successes:
        raise UserError(
            f"{result['run_tag']} feedback/evaluator endpoint mismatch: "
            f"{successes} != {metric_successes}"
        )

    diagnostics_path = Path(result["diagnostics_path"])
    diagnostics = _read_json_object(
        diagnostics_path, f"{result['run_tag']} diagnostics"
    )
    if diagnostics.get("action_selection") != "target_native_argmax":
        raise UserError(f"{result['run_tag']} top-level action protocol mismatch")
    if diagnostics.get("feedtta_scope_profile") != parameters.get(
        "scope_profile"
    ):
        raise UserError(f"{result['run_tag']} FeedTTA scope profile mismatch")
    prefixes = diagnostics.get("trainable_prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise UserError(f"{result['run_tag']} has no FeedTTA trainable prefixes")
    if any("pooler" in prefix or "local_his" in prefix for prefix in prefixes):
        raise UserError(
            f"{result['run_tag']} includes post-logit zero-gradient modules"
        )
    _validate_feedtta_scope_prefixes(
        phase["setting"], parameters.get("scope_profile"), prefixes,
        result["run_tag"],
    )


def _validate_feedtta_scope_prefixes(setting, profile, prefixes, run_tag):
    heads = {
        "duet-r2r": (
            "vln_bert.global_sap_head", "vln_bert.local_sap_head",
            "vln_bert.sap_fuse_linear",
        ),
        "goat-r2r": (
            "vln_bert.global_sap_head", "vln_bert.local_sap_head",
            "vln_bert.sap_fuse_linear",
        ),
        "hamt-r2r": ("vln_bert.next_action",),
    }[setting]
    if setting == "duet-r2r":
        stacks = (
            "vln_bert.global_encoder.encoder.x_layers",
            "vln_bert.local_encoder.encoder.x_layers",
        )
    elif setting == "goat-r2r":
        stacks = (
            "vln_bert.global_encoder.encoder.crossattention",
            "vln_bert.local_encoder.encoder.crossattention",
        )
    else:
        stacks = ("vln_bert.encoder.x_layers",)

    actual = tuple(prefixes)
    if profile == "action_head":
        valid = actual == heads
    elif profile == "paper_full":
        valid = actual == stacks + heads
    elif profile == "last_crossmodal":
        if len(actual) != len(stacks) + len(heads) or actual[-len(heads):] != heads:
            valid = False
        else:
            last_layers = actual[:len(stacks)]
            valid = all(
                re.fullmatch(re.escape(stack) + r"\.\d+", layer) is not None
                for stack, layer in zip(stacks, last_layers)
            )
    else:
        valid = False
    if not valid:
        raise UserError(
            f"{run_tag} has invalid {setting}/{profile} FeedTTA prefixes: "
            f"{list(actual)}"
        )


def _validate_or_mark_postfix_results(phase, root, jobs, spec, results):
    """Turn a post-run scientific-contract failure into a retryable job."""
    if spec.get("schema") != SPEC_SCHEMA_V2:
        return results
    by_tag = {result.get("run_tag"): result for result in results}
    failures = []
    for job in jobs:
        result = by_tag.get(job["run_tag"])
        if result is None:
            continue
        try:
            _validate_postfix_result_contract(phase, result, spec)
        except UserError as error:
            failures.append((job, str(error)))
    if not failures:
        return results
    for job, message in failures:
        job_root = Path(job["job_dir"])
        staged.atomic_json(job_root / "POSTFIX_CONTRACT_ERROR.json", {
            "schema": "navtta.vln_r2r_postfix_contract_error.v1",
            "run_tag": job["run_tag"],
            "phase_id": phase["phase_id"],
            "error": message,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        staged.atomic_text(job_root / "exitcode", "86\n")
    staged.write_summary(root, jobs, spec)
    raise UserError(
        f"{phase['phase_id']} has {len(failures)} post-fix contract failure(s); "
        "rerun with --resume --retry-failed after correcting the cause"
    )


def _feedback_budget(result):
    adapter = result.get("adapter_diagnostics") or {}
    keys = (
        "feedback_observed_episodes", "feedback_observation_rate",
        "feedback_episodes", "successful_feedback_episodes",
        "failed_feedback_episodes",
        "queries", "query_rate", "queried_feedback_successes",
        "self_label_episodes", "self_feedback_successes",
    )
    value = {key: adapter[key] for key in keys if key in adapter}
    return value or None


def _find_phase(spec, setting, method):
    for phase in phase_sequence(spec):
        if phase["setting"] == setting and phase["method"] == method:
            return phase
    raise UserError(f"missing phase {setting}/{method}")


def _control_result(batch_id, spec, setting, method):
    if method == "source":
        if setting not in SETTINGS:
            raise UserError(f"unknown reused Source setting {setting}")
        record = _reused_source_manifest(spec)["document"]["records"][setting]
        # Return a detached JSON value so callers cannot mutate the pinned manifest.
        return json.loads(canonical(record))
    phase = _find_phase(spec, setting, method)
    results = _load_phase_results(phase_root(batch_id, phase), spec, phase)
    if len(results) != 1:
        raise UserError(f"{setting}/{method} must contain exactly one control")
    return results[0]


def _top5_fields():
    return [
        "setting", "method", "rank", "run_tag", "sr", "spl",
        "source_sr", "source_spl", "delta_sr_pp", "delta_spl_pp",
        "sampled_control_sr", "sampled_control_spl",
        "delta_sampled_control_sr_pp", "delta_sampled_control_spl_pp",
        "relative_param_drift", "updates", "feedback_queries",
        "feedback_query_rate", "feedback_observed_episodes", "parameters",
    ]


def summarize_phase(phase, batch_id, spec, results=None):
    root = phase_root(batch_id, phase)
    results = (
        results
        if results is not None
        else _load_phase_results(root, spec, phase)
    )
    method = phase["method"]
    if len(results) != phase["job_count"]:
        raise UserError(f"{phase['phase_id']} result count mismatch")
    if method in CONTROL_METHODS:
        result = results[0]
        document = {
            "schema": "navtta.vln_r2r_local_refinement_control.v1",
            "experiment_id": spec["experiment_id"],
            "batch_id": batch_id,
            "phase_id": phase["phase_id"],
            "setting": phase["setting"],
            "control": method,
            "run_tag": result["run_tag"],
            "parameters": result["parameters"],
            "metrics": result["metrics"],
            "git_commit": git("rev-parse", "HEAD"),
            "spec_sha256": spec["_sha256"],
        }
        staged.atomic_json(root / "CONTROL.json", document)
        return document

    for result in results:
        _validate_postfix_result_contract(phase, result, spec)

    source = _control_result(batch_id, spec, phase["setting"], "source")
    sampled = None
    if (
        method == "feedtta"
        and "feedtta_control" in enabled_phases_by_setting(spec)[phase["setting"]]
    ):
        sampled = _control_result(
            batch_id, spec, phase["setting"], "feedtta_control"
        )
    ranked = sorted(results, key=_result_score, reverse=True)
    winner = ranked[0]
    winner_record = {
        "winner_run_tag": winner["run_tag"],
        "winner_parameters": winner["parameters"],
        "winner_metrics": winner["metrics"],
        "winner_feedback_budget": _feedback_budget(winner),
        "source_run_tag": source["run_tag"],
        "source_metrics": source["metrics"],
        "delta_sr_pp": winner["metrics"]["SR"] - source["metrics"]["SR"],
        "delta_spl_pp": winner["metrics"]["SPL"] - source["metrics"]["SPL"],
    }
    if sampled is not None:
        winner_record.update({
            "sampled_control_run_tag": sampled["run_tag"],
            "sampled_control_metrics": sampled["metrics"],
            "delta_sampled_control_sr_pp": (
                winner["metrics"]["SR"] - sampled["metrics"]["SR"]
            ),
            "delta_sampled_control_spl_pp": (
                winner["metrics"]["SPL"] - sampled["metrics"]["SPL"]
            ),
        })
    if method == "atena" and spec["methods"]["atena"].get(
        "secondary_feedback_efficiency_report", {}
    ).get("enabled"):
        near = [
            item for item in ranked
            if item["metrics"]["SR"] >= winner["metrics"]["SR"] - 0.1
            and item["metrics"]["SPL"] >= winner["metrics"]["SPL"] - 0.2
            and _feedback_budget(item) is not None
            and "query_rate" in _feedback_budget(item)
        ]
        if near:
            efficient = min(
                near,
                key=lambda item: (
                    float(_feedback_budget(item)["query_rate"]),
                    -float(item["metrics"]["SR"]),
                    -float(item["metrics"]["SPL"]),
                ),
            )
            winner_record["feedback_efficiency_point"] = {
                "run_tag": efficient["run_tag"],
                "parameters": efficient["parameters"],
                "metrics": efficient["metrics"],
                "feedback_budget": _feedback_budget(efficient),
            }

    document = {
        "schema": "navtta.vln_r2r_local_refinement_winner.v1",
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "phase_id": phase["phase_id"],
        "setting": phase["setting"],
        "method": method,
        "selection": ["SR", "SPL", "lower_parameter_drift", "fewer_updates"],
        "source_protocol": "standard_argmax",
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": spec["_sha256"],
        **winner_record,
    }
    staged.atomic_json(root / "WINNER.json", document)

    rows = []
    for rank, candidate in enumerate(ranked[:5], start=1):
        adapter = candidate.get("adapter_diagnostics") or {}
        row = {
            "setting": phase["setting"],
            "method": method,
            "rank": rank,
            "run_tag": candidate["run_tag"],
            "sr": candidate["metrics"]["SR"],
            "spl": candidate["metrics"]["SPL"],
            "source_sr": source["metrics"]["SR"],
            "source_spl": source["metrics"]["SPL"],
            "delta_sr_pp": candidate["metrics"]["SR"] - source["metrics"]["SR"],
            "delta_spl_pp": candidate["metrics"]["SPL"] - source["metrics"]["SPL"],
            "sampled_control_sr": sampled["metrics"]["SR"] if sampled else None,
            "sampled_control_spl": sampled["metrics"]["SPL"] if sampled else None,
            "delta_sampled_control_sr_pp": (
                candidate["metrics"]["SR"] - sampled["metrics"]["SR"]
                if sampled else None
            ),
            "delta_sampled_control_spl_pp": (
                candidate["metrics"]["SPL"] - sampled["metrics"]["SPL"]
                if sampled else None
            ),
            "relative_param_drift": adapter.get("relative_param_drift"),
            "updates": adapter.get("updates"),
            "feedback_queries": adapter.get("queries"),
            "feedback_query_rate": adapter.get("query_rate"),
            "feedback_observed_episodes": adapter.get("feedback_observed_episodes"),
            "parameters": canonical(candidate["parameters"]),
        }
        rows.append(row)
    with (root / "TOP5.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_top5_fields())
        writer.writeheader()
        writer.writerows(rows)
    return document


def aggregate_campaign(batch_id, spec):
    active_methods = active_tta_methods(spec)
    enabled = enabled_phases_by_setting(spec)
    winners = {method: {} for method in active_methods}
    controls = {
        setting: {
            "source": _control_result(batch_id, spec, setting, "source")
        }
        for setting in SETTINGS
    }
    top_rows = []
    for phase in phase_sequence(spec):
        root = phase_root(batch_id, phase)
        if phase["method"] in CONTROL_METHODS:
            path = root / "CONTROL.json"
            if not path.is_file():
                summarize_phase(phase, batch_id, spec)
            controls[phase["setting"]][phase["method"]] = json.loads(
                path.read_text(encoding="utf-8")
            )
            continue
        path = root / "WINNER.json"
        if not path.is_file():
            summarize_phase(phase, batch_id, spec)
        winner = json.loads(path.read_text(encoding="utf-8"))
        winners[phase["method"]][phase["setting"]] = winner
        with (root / "TOP5.csv").open("r", encoding="utf-8", newline="") as stream:
            top_rows.extend(csv.DictReader(stream))

    root = campaign_root(batch_id)
    document = {
        "schema": "navtta.vln_r2r_local_refinement_campaign_winners.v1",
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "benchmark": "r2r",
        "split": "val_seen",
        "source_protocol": "standard_argmax",
        "selection": ["SR", "SPL", "lower_parameter_drift", "fewer_updates"],
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": spec["_sha256"],
        "controls": controls,
        "methods": winners,
    }
    staged.atomic_json(root / "WINNERS.json", document)
    frozen = {
        "schema": "navtta.vln_r2r_local_refinement_frozen_hparams.v1",
        "generated_from": "WINNERS.json",
        "batch_id": batch_id,
        "git_commit": document["git_commit"],
        "spec_sha256": spec["_sha256"],
        "settings": {
            setting: {
                method: winners[method][setting]["winner_parameters"]
                for method in active_methods
                if method in enabled[setting]
            }
            for setting in SETTINGS
        },
    }
    staged.atomic_json(root / "FROZEN_HPARAMETERS.json", frozen)
    with (root / "TOP5.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_top5_fields())
        writer.writeheader()
        writer.writerows(top_rows)
    return document


def runtime_args(cli, phase, spec):
    max_workers = _configured_worker_limit(cli, phase, spec)
    gate = _configured_prelaunch_gate(cli, phase, spec)
    return argparse.Namespace(
        method=phase["method"],
        batch_id=cli.batch_id,
        settings=[phase["setting"]],
        max_workers=max_workers,
        max_per_model=max_workers,
        max_discrete_workers=max_workers,
        max_continuous_workers=1,
        max_gpu_memory_mib=gate,
        max_memory_gib=cli.max_memory_gib,
        launch_stagger=cli.launch_stagger,
        resource_wait_timeout=cli.resource_wait_timeout,
        resume=cli.resume,
        retry_failed=cli.retry_failed,
        fail_fast=True,
    )


def run_phase_batch(args, phase, phase_path, jobs, spec):
    """Run one phase with the plan's observed-memory emergency stop line."""
    emergency_mib = int(
        spec["execution"]["gpu_safety"]["emergency_abort_observed_used_mib"]
    )
    stop = threading.Event()
    observed = {"gpu_memory_mib": None}

    def monitor():
        # Give run_batch time to install its forwarding signal handlers.
        while not stop.wait(5.0):
            gpu_memory, _ = staged.gpu_stats()
            if gpu_memory >= emergency_mib:
                observed["gpu_memory_mib"] = gpu_memory
                staged.atomic_json(Path(phase_path) / "EMERGENCY_ABORT.json", {
                    "schema": "navtta.vln_r2r_local_refinement_emergency_abort.v1",
                    "phase_id": phase["phase_id"],
                    "threshold_mib": emergency_mib,
                    "observed_gpu_memory_mib": gpu_memory,
                    "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                })
                os.kill(os.getpid(), signal.SIGTERM)
                return

    thread = threading.Thread(
        target=monitor,
        name=f"gpu-emergency-{phase['phase_id']}",
        daemon=True,
    )
    thread.start()
    try:
        results = staged.run_batch(args, phase_path, jobs, spec)
        return _validate_or_mark_postfix_results(
            phase, phase_path, jobs, spec, results
        )
    except staged.UserError as error:
        if observed["gpu_memory_mib"] is not None:
            raise UserError(
                f"{phase['phase_id']} crossed the {emergency_mib} MiB emergency "
                f"line (observed {observed['gpu_memory_mib']} MiB)"
            ) from error
        raise
    finally:
        stop.set()
        thread.join(timeout=1.0)


def _phase_started(root):
    try:
        jobs = staged.load_jobs(root)
    except staged.UserError:
        return False
    for job in jobs:
        job_root = Path(job["job_dir"])
        if (job_root / "exitcode").exists() or (job_root / "worker_state.json").exists():
            return True
    return False


def _phase_has_failure(root):
    try:
        jobs = staged.load_jobs(root)
    except staged.UserError:
        return False
    for job in jobs:
        exit_path = Path(job["job_dir"]) / "exitcode"
        if not exit_path.is_file():
            continue
        try:
            if int(exit_path.read_text().strip()) != 0:
                return True
        except ValueError:
            return True
    return False


def _assert_no_live_other_phase(planned, active_phase):
    for phase, root, jobs in planned:
        if phase["phase_id"] == active_phase["phase_id"]:
            continue
        for job in jobs:
            pid = staged._worker_pid(job)
            if pid is not None and staged.process_alive(pid):
                raise UserError(
                    f"live worker exists outside active phase: {phase['phase_id']}"
                )


@contextmanager
def campaign_lock(root):
    path = Path(root) / ".scheduler.lock"
    payload = json.dumps({"pid": os.getpid(), "created_at": time.time()}) + "\n"
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
            break
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                pid = int(existing["pid"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pid = None
            if pid is not None and staged.process_alive(pid):
                raise UserError(f"another refinement scheduler is active: pid {pid}")
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    else:
        raise UserError("could not acquire refinement scheduler lock")
    try:
        yield
    finally:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if int(existing.get("pid", -1)) == os.getpid():
                path.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            pass


def assert_launchable(cli, spec):
    execution = spec["execution"]
    if execution.get("launchable") is not True:
        raise UserError("spec is explicitly not launchable")
    if execution.get("barrier_implementation_status") != "implemented_validated":
        raise UserError("spec barriers are not implemented_validated")
    activation = spec.get("activation", {})
    if (
        activation.get("manual_review_required")
        or activation.get("automatic_launch_forbidden")
    ) and not cli.confirm_reviewed:
        raise UserError("launch requires --confirm-reviewed")
    _validate_reused_source_artifacts(spec)
    _validate_search_prior_artifacts(spec)


def execute_campaign(cli, spec):
    if not cli.plan_only:
        assert_launchable(cli, spec)
    root, planned = ensure_campaign_plan(cli, spec)
    if cli.plan_only:
        print(f"campaign={cli.batch_id} phases={len(planned)} jobs="
              f"{sum(len(jobs) for _, _, jobs in planned)} root={root}")
        for phase, _, jobs in planned:
            print(
                f"phase={phase['phase_id']} jobs={len(jobs)} "
                f"workers={_configured_worker_limit(cli, phase, spec)} "
                f"prelaunch_gpu_mib={_configured_prelaunch_gate(cli, phase, spec)}"
            )
        if cli.print_commands:
            for phase, _, jobs in planned:
                for job in jobs:
                    print(f"phase={phase['phase_id']} " + subprocess.list2cmdline(job["command"]))
        return
    with campaign_lock(root):
        completed = []
        for index, (phase, phase_path, jobs) in enumerate(planned):
            if any(
                not _phase_complete_for_spec(path, spec, prior_phase)
                for prior_phase, path, _ in completed
            ):
                raise UserError("upstream phase is incomplete; barrier refused advance")
            if _phase_complete_for_spec(phase_path, spec, phase):
                results = _load_phase_results(phase_path, spec, phase)
                summarize_phase(phase, cli.batch_id, spec, results=results)
                completed.append((phase, phase_path, jobs))
                continue
            downstream = [
                later[0]["phase_id"] for later in planned[index + 1:]
                if _phase_started(later[1])
            ]
            if downstream:
                action = "retry" if (
                    cli.retry_failed and _phase_has_failure(phase_path)
                ) else "execute"
                raise UserError(
                    f"cannot {action} an upstream phase after downstream execution: "
                    + ", ".join(downstream)
                )
            _assert_no_live_other_phase(planned, phase)
            args = runtime_args(cli, phase, spec)
            results = run_phase_batch(args, phase, phase_path, jobs, spec)
            summarize_phase(phase, cli.batch_id, spec, results=results)
            completed.append((phase, phase_path, jobs))
    aggregate_campaign(cli.batch_id, spec)


def _read_progress(root, planned, spec):
    """Build a read-only status from evidence, not a possibly stale progress file."""
    succeeded = failed = invalid = orphaned = running = 0
    running_jobs = []
    try:
        jobs = staged.load_jobs(root)
    except staged.UserError:
        jobs = []
    for job in jobs:
        job_root = Path(job["job_dir"])
        exit_path = job_root / "exitcode"
        if exit_path.is_file():
            try:
                code = int(exit_path.read_text().strip())
            except ValueError:
                code = 255
            if code == 0:
                try:
                    staged.parse_metrics(job, spec)
                    succeeded += 1
                except Exception:
                    failed += 1
                    invalid += 1
            else:
                failed += 1
            continue
        pid = staged._worker_pid(job)
        if pid is not None and staged.process_alive(pid):
            running += 1
            running_jobs.append(job["run_tag"])
        elif (job_root / "worker_state.json").exists():
            failed += 1
            orphaned += 1
    return {
        "planned": planned,
        "pending": max(0, planned - succeeded - failed - running),
        "running": running,
        "succeeded": succeeded,
        "failed": failed,
        "invalid": invalid,
        "orphaned": orphaned,
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "running_jobs": running_jobs,
    }


def status_snapshot(batch_id):
    root = campaign_root(batch_id)
    manifest_path = root / "PLAN.json"
    if not manifest_path.is_file():
        raise UserError(f"campaign is not planned: {batch_id}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"invalid campaign PLAN.json: {error}") from error
    status_spec = {
        "setting_episode_counts": {
            setting: int(manifest.get("episode_count", 1021))
            for setting in SETTINGS
        }
    }
    phases = []
    totals = {key: 0 for key in (
        "planned", "pending", "running", "succeeded", "failed",
        "invalid", "orphaned",
    )}
    active_phase = None
    upstream_complete = True
    for phase in manifest.get("phases", []):
        progress = _read_progress(
            phase_root(batch_id, phase), phase["job_count"], status_spec
        )
        complete = (
            int(progress.get("succeeded", 0)) == int(progress.get("planned", 0))
            and int(progress.get("failed", 0)) == 0
        )
        if active_phase is None and not complete:
            active_phase = phase["phase_id"]
        value = {
            "phase_id": phase["phase_id"],
            "setting": phase["setting"],
            "method": phase["method"],
            "barrier_released": upstream_complete,
            "complete": complete,
            "progress": progress,
        }
        phases.append(value)
        upstream_complete = upstream_complete and complete
        for key in totals:
            totals[key] += int(progress.get(key, 0))
    complete = (
        totals["planned"] == int(manifest.get("job_count", -1))
        and totals["succeeded"] == totals["planned"]
        and totals["failed"] == 0
    )
    needs_attention = totals["failed"] > 0 and totals["running"] == 0
    return {
        "batch_id": batch_id,
        "active_phase": active_phase,
        "phases": phases,
        "totals": totals,
        "complete": complete,
        "needs_attention": needs_attention,
        "terminal": complete or needs_attention,
    }


def show_status(batch_id, watch=False, interval=10.0):
    while True:
        snapshot = status_snapshot(batch_id)
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        if not watch or snapshot["terminal"]:
            return
        time.sleep(interval)


def _assert_clean_tracked_spec(spec):
    for label, candidate in (
        ("spec", Path(spec["_path"])),
        ("runner", Path(__file__)),
    ):
        try:
            relative = candidate.resolve().relative_to(REPO_ROOT)
        except ValueError as error:
            raise UserError(f"{label} must be inside the repository") from error
        try:
            git("ls-files", "--error-unmatch", "--", relative.as_posix())
        except subprocess.CalledProcessError as error:
            raise UserError(
                f"{label} must be tracked before planning or launch"
            ) from error
    if git("status", "--porcelain", "--untracked-files=no"):
        raise UserError("worktree must be clean before planning or launch")


def _parse_phase_worker_override(value):
    phase_id, separator, raw_limit = value.rpartition("=")
    if not separator:
        raise argparse.ArgumentTypeError("expected PHASE_ID=WORKERS")
    try:
        safe_component("phase_id", phase_id)
        limit = int(raw_limit)
    except (UserError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if limit < 1:
        raise argparse.ArgumentTypeError("phase workers must be positive")
    return phase_id, limit


def _parse_phase_calibration_override(value):
    phase_id, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError("expected PHASE_ID=CALIBRATION.json")
    try:
        safe_component("phase_id", phase_id)
    except UserError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return phase_id, str(Path(raw_path).expanduser().resolve())


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--max-workers", type=int,
        help="global cap only; cannot raise a phase above its spec-approved safe limit",
    )
    parser.add_argument(
        "--phase-max-workers", action="append", default=[],
        type=_parse_phase_worker_override, metavar="PHASE_ID=WORKERS",
        help=(
            "set one phase to a worker count explicitly declared by its "
            "calibration baseline/steps/production cap; repeatable"
        ),
    )
    parser.add_argument(
        "--phase-calibration", action="append", default=[],
        type=_parse_phase_calibration_override,
        metavar="PHASE_ID=CALIBRATION.json",
        help=(
            "bind a phase worker override to completed grouped calibration "
            "evidence; required when raising the spec default"
        ),
    )
    parser.add_argument(
        "--max-gpu-memory-mib", type=int,
        help="global prelaunch cap only; cannot raise a phase gate from the spec",
    )
    parser.add_argument("--max-memory-gib", type=float, default=DEFAULT_MAX_MEMORY_GIB)
    parser.add_argument(
        "--launch-stagger", type=float, default=DEFAULT_LAUNCH_STAGGER_SECONDS
    )
    parser.add_argument(
        "--resource-wait-timeout", type=float,
        default=DEFAULT_RESOURCE_WAIT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=float, default=10.0)
    args = parser.parse_args(argv)
    safe_component("batch_id", args.batch_id)
    if args.gpu < 0:
        parser.error("gpu must be non-negative")
    for label in ("max_workers", "max_gpu_memory_mib"):
        value = getattr(args, label)
        if value is not None and value < 1:
            parser.error(f"{label.replace('_', '-')} must be positive")
    if args.max_memory_gib <= 0:
        parser.error("max-memory-gib must be positive")
    if args.launch_stagger < 0:
        parser.error("launch-stagger must be non-negative")
    if args.resource_wait_timeout <= 0:
        parser.error("resource-wait-timeout must be positive")
    if args.watch_interval <= 0:
        parser.error("watch-interval must be positive")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if args.print_commands and not args.plan_only:
        parser.error("--print-commands requires --plan-only")
    phase_limits = {}
    for phase_id, limit in args.phase_max_workers:
        if phase_id in phase_limits:
            parser.error(f"duplicate --phase-max-workers for {phase_id}")
        phase_limits[phase_id] = limit
    args.phase_max_workers = phase_limits
    phase_calibrations = {}
    for phase_id, path in args.phase_calibration:
        if phase_id in phase_calibrations:
            parser.error(f"duplicate --phase-calibration for {phase_id}")
        phase_calibrations[phase_id] = path
    args.phase_calibrations = phase_calibrations
    if args.watch:
        args.status = True
    return args


def main(argv=None):
    cli = parse_args(argv)
    if cli.status:
        show_status(cli.batch_id, watch=cli.watch, interval=cli.watch_interval)
        return
    spec = load_spec(cli.spec)
    _assert_clean_tracked_spec(spec)
    execute_campaign(cli, spec)


if __name__ == "__main__":
    try:
        main()
    except (UserError, staged.UserError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
