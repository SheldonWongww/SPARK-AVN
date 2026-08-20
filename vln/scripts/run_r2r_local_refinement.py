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
SPEC_SCHEMA = "navtta.vln_r2r_five_method_local_refinement_plan.v1"
PLAN_SCHEMA = "navtta.vln_r2r_local_refinement_campaign_plan.v1"
PHASE_SCHEMA = "navtta.vln_r2r_local_refinement_phase.v1"
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

DEFAULT_MAX_MEMORY_GIB = 75.0
DEFAULT_LAUNCH_STAGGER_SECONDS = 15.0
DEFAULT_RESOURCE_WAIT_TIMEOUT_SECONDS = 900.0


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
        if phases[0] != "source":
            raise UserError(f"enabled phases for {setting} must start with source")
        if ("feedtta" in phases) != ("feedtta_control" in phases):
            raise UserError(
                f"{setting} must pair FeedTTA with its sampled no-update control"
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
    if document.get("schema") != SPEC_SCHEMA:
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
    _exact_keys(document.get("methods"), active_methods, "methods")
    _exact_keys(document.get("parent_anchors"), active_methods, "parent_anchors")
    budget = _mapping(document.get("budget"), "budget")
    search_by_method = {}
    search_by_setting = {setting: 0 for setting in SETTINGS}
    for method in active_methods:
        method_settings = enabled_settings_for_method(document, method)
        method_spec = document["methods"][method]
        _exact_keys(method_spec.get("settings"), method_settings, f"{method}.settings")
        _exact_keys(
            document["parent_anchors"][method],
            method_settings,
            f"parent_anchors.{method}",
        )
        method_total = 0
        for setting in method_settings:
            points = _expand_setting(method, setting, document)
            method_total += len(points)
            search_by_setting[setting] += len(points)
            anchor = _mapping(
                document["parent_anchors"][method][setting],
                f"parent_anchors.{method}.{setting}",
            )
            run_tag = anchor.get("parent_run_tag")
            safe_component("parent_run_tag", run_tag)
            candidate = _mapping(
                anchor.get("candidate"),
                f"parent_anchors.{method}.{setting}.candidate",
            )
            occurrences = sum(
                canonical(item["candidate"]) == canonical(candidate)
                for item in points
            )
            if occurrences != 1:
                raise UserError(
                    f"parent anchor {method}/{setting} occurs {occurrences} times"
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
    if controls.get("feedtta_sampled_no_update_per_setting") is not True:
        raise UserError("FeedTTA sampled no-update controls are required")
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
    _exact_keys(
        control_execution, CONTROL_METHODS, "execution.control_execution"
    )
    for method in CONTROL_METHODS:
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
            steps = value.get("conditional_test_steps", [])
            if not isinstance(steps, list):
                raise UserError(
                    f"{method}/{setting}.conditional_test_steps must be an array"
                )
            for step in steps:
                _positive_int(step, f"{method}/{setting} conditional worker limit")


def load_spec(path=DEFAULT_SPEC):
    path = Path(path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"invalid local-refinement spec {path}: {error}") from error
    _validate_spec(document)
    document["setting_episode_counts"] = {
        setting: int(document["protocol"]["episode_count"])
        for setting in SETTINGS
    }
    document["_path"] = str(path)
    document["_sha256"] = sha256(path)
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
    parent_anchor = None
    if phase["method"] not in CONTROL_METHODS:
        parent_anchor = spec["parent_anchors"][phase["method"]][phase["setting"]]
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
        if parent_anchor is not None and canonical(item["candidate"]) == canonical(
            parent_anchor["candidate"]
        ):
            parent_run_tags.append(parent_anchor["parent_run_tag"])
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
        "result_layout": RESULT_LAYOUT,
        "safe_worker_limit": _safe_worker_limit(phase, spec),
        "effective_worker_limit": _configured_worker_limit(cli, phase, spec),
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
        "result_layout": RESULT_LAYOUT,
        "strict_model_barrier": True,
        "strict_method_barrier": True,
        "job_count": sum(phase["job_count"] for phase in phases),
        "phases": phases,
        "runtime": {
            "phase_max_workers": dict(sorted(
                (getattr(cli, "phase_max_workers", {}) or {}).items()
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
        "parent_run_tags", "retry_result_root_parent",
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


def _load_validated_results(root, spec):
    jobs = staged.load_jobs(root)
    results, errors = staged.write_summary(root, jobs, spec)
    if len(results) + len(errors) != len(jobs) or errors:
        raise UserError(f"refinement phase is incomplete: {root}")
    return results


def _phase_complete(root, spec):
    try:
        _load_validated_results(root, spec)
        return True
    except (UserError, staged.UserError):
        return False


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


def _feedback_budget(result):
    adapter = result.get("adapter_diagnostics") or {}
    keys = (
        "feedback_observed_episodes", "feedback_observation_rate",
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
    phase = _find_phase(spec, setting, method)
    results = _load_validated_results(phase_root(batch_id, phase), spec)
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
    results = results if results is not None else _load_validated_results(root, spec)
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

    source = _control_result(batch_id, spec, phase["setting"], "source")
    sampled = None
    if method == "feedtta":
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
    controls = {setting: {} for setting in SETTINGS}
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
        return staged.run_batch(args, phase_path, jobs, spec)
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
            if any(not _phase_complete(path, spec) for _, path, _ in completed):
                raise UserError("upstream phase is incomplete; barrier refused advance")
            if _phase_complete(phase_path, spec):
                results = _load_validated_results(phase_path, spec)
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
