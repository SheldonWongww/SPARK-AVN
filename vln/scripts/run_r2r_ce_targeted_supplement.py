#!/usr/bin/env python3
"""Run the minimal, model-barriered R2R-CE targeted supplement.

The shared staged scheduler remains unchanged.  This campaign builds its own
immutable four-phase plan while reusing its worker entrypoint, retry helper,
resource probes, metric parser, and formal-manifest authenticator.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tta_hparam_search as staged  # noqa: E402
import tta_config_cli as config_cli  # noqa: E402
from shared_gpu_launch_guard import (  # noqa: E402
    ReservationLedgerError,
    release_shared_gpu_reservation,
    shared_gpu_launch_guard,
)


DEFAULT_SPEC = (
    REPO_ROOT / "vln/experiments/r2r_ce_targeted_supplement_v1.json"
)
LOG_ROOT = REPO_ROOT / "vln/results/logs/r2r-ce/hparam_search"
TUNING_ROOT = REPO_ROOT / "vln/results/tuning/r2r-ce/hparam_search"
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
SPEC_SCHEMA = "navtta.vln_r2r_ce_targeted_supplement.v1"
PLAN_SCHEMA = "navtta.vln_r2r_ce_targeted_supplement_plan.v1"
PHASE_SCHEMA = "navtta.vln_r2r_ce_targeted_supplement_phase.v1"
RESULT_LAYOUT = "r2r_ce_targeted_supplement_model_method_v1"
SETTINGS = ("etpnav-r2r-ce", "bevbert-r2r-ce")
SETTING_MODEL = {
    "etpnav-r2r-ce": "etpnav",
    "bevbert-r2r-ce": "bevbert",
}
METHOD_ORDER = ("tent", "fstta", "feedtta")
SCREENING_STAGE = "r2r_ce_targeted_screening"
FULL_STAGE = "r2r_ce_targeted_full"
RESERVATION_ROLE = "r2r_ce_targeted_supplement"
PROCESS_TOKEN_ENV = "NAVTTA_R2R_CE_TARGETED_PROCESS_TOKEN"


class UserError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def safe_component(label, value):
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in (".", "..")
        or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
    ):
        raise UserError("unsafe {} path component: {!r}".format(label, value))
    return value


def _repo_reference(reference, label):
    if not isinstance(reference, dict):
        raise UserError("{} must be an object".format(label))
    relative = reference.get("path")
    if not isinstance(relative, str) or not relative:
        raise UserError("{} has no path".format(label))
    path = (REPO_ROOT / relative).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise UserError("{} escapes the repository".format(label)) from error
    if not path.is_file():
        raise UserError("missing {}: {}".format(label, path))
    expected = reference.get("sha256")
    if not isinstance(expected, str) or sha256(path) != expected:
        raise UserError("{} SHA256 mismatch".format(label))
    return path


def _exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise UserError(
            "{} keys mismatch: expected {}, got {}".format(
                label, sorted(expected), actual
            )
        )


def enabled_methods(setting, spec):
    return tuple(
        method
        for method in METHOD_ORDER
        if setting in spec["methods"][method]["settings"]
    )


def expand_candidates(method, setting, spec):
    try:
        cell = spec["methods"][method]["settings"][setting]
    except KeyError as error:
        raise UserError("disabled cell {} / {}".format(setting, method)) from error
    fixed = cell.get("fixed")
    candidates = cell.get("candidates")
    if not isinstance(fixed, dict) or not isinstance(candidates, list):
        raise UserError("malformed cell {} / {}".format(setting, method))
    output = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate).intersection(fixed):
            raise UserError(
                "candidate/fixed overlap in {} / {}".format(setting, method)
            )
        point = dict(fixed)
        point.update(candidate)
        output.append(point)
    if not output or len({canonical(item) for item in output}) != len(output):
        raise UserError("empty or duplicate candidates in {} / {}".format(
            setting, method
        ))
    return output


def _validate_canonical_order(spec):
    order_spec = spec.get("canonical_order", {})
    if (
        order_spec.get("seed") != 0
        or spec.get("primary_order_seed") != 0
        or order_spec.get("screening_prefix_episodes") != 100
        or order_spec.get("full_episodes") != 778
    ):
        raise UserError("campaign must use canonical seed-0 100/778 streams")
    path = _repo_reference(order_spec.get("manifest"), "canonical order manifest")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("invalid canonical order manifest: {}".format(error))
    expected = {
        "schema": "navtta.episode_order.v1",
        "benchmark": spec.get("benchmark"),
        "split": "val_seen",
        "episode_count": 778,
        "order_sha256": order_spec["manifest"].get("order_sha256"),
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise UserError("canonical order mismatch for {}".format(key))
    episodes = document.get("episodes")
    identifiers = [str(item.get("episode_id")) for item in episodes or []]
    if len(identifiers) != 778 or len(set(identifiers)) != 778:
        raise UserError("canonical order has invalid episode coverage")
    return path, document


def _validate_search_grid(spec):
    _exact_keys(spec.get("methods"), METHOD_ORDER, "methods")
    expected = {
        "tent": {"etpnav-r2r-ce"},
        "fstta": set(SETTINGS),
        "feedtta": set(SETTINGS),
    }
    screening_total = 0
    for method in METHOD_ORDER:
        settings = spec["methods"][method].get("settings")
        _exact_keys(settings, expected[method], "{}.settings".format(method))
        for setting in settings:
            points = expand_candidates(method, setting, spec)
            if len(points) != 3:
                raise UserError("each enabled cell must contain three candidates")
            screening_total += len(points)
            if method == "fstta":
                maximum = int(spec["selection"]["fstta"]["max_fast_window_m"])
                if maximum > 15 or any(
                    isinstance(point.get("m"), bool)
                    or not isinstance(point.get("m"), int)
                    or point["m"] > maximum
                    or point["m"] < 1
                    for point in points
                ):
                    raise UserError("FSTTA M exceeds the fail-closed episode bound")
    budget = spec.get("budget")
    if budget != {
        "screening_tta_jobs": 15,
        "full_val_seen_tta_jobs_max": 5,
        "source_execution_jobs": 0,
        "total_executed_jobs_max": 20,
    } or screening_total != 15:
        raise UserError("targeted supplement must remain an exact 15 + <=5 plan")


def _validate_execution(spec):
    execution = spec.get("execution", {})
    if (
        execution.get("strict_model_barrier") is not True
        or execution.get("model_order") != list(SETTINGS)
        or execution.get("phase_order_within_model")
        != ["screening", "full_confirmation"]
        or execution.get("round_robin_methods_within_screening") is not True
        or execution.get("parallel_peer_campaign")
        != "reverie_frozen_evaluation"
        or execution.get("shared_gpu_launch_guard_required") is not True
        or execution.get("shared_active_reservation_required") is not True
    ):
        raise UserError(
            "execution must use reviewed model barriers and shared-GPU coordination"
        )
    expected = {
        "etpnav-r2r-ce": ((9, 3), (3, 3)),
        "bevbert-r2r-ce": ((6, 3), (2, 2)),
    }
    phases = execution.get("phases", {})
    _exact_keys(phases, SETTINGS, "execution.phases")
    for setting, values in expected.items():
        screen, full = values
        if phases[setting].get("screening") != {
            "planned_jobs": screen[0], "max_workers": screen[1]
        }:
            raise UserError("invalid screening concurrency for {}".format(setting))
        if phases[setting].get("full_confirmation") != {
            "max_jobs": full[0], "max_workers": full[1]
        }:
            raise UserError("invalid full concurrency for {}".format(setting))
    limits = execution.get("resource_limits", {})
    if (
        int(limits.get("max_gpu_memory_mib_before_launch", -1))
        + int(limits.get("estimated_job_gpu_memory_mib", -1))
        > int(limits.get("max_aggregate_gpu_memory_mib", -1))
        or float(limits.get("max_cgroup_memory_gib_before_launch", -1))
        + float(limits.get("estimated_job_memory_gib", -1))
        > float(limits.get("max_aggregate_cgroup_memory_gib", -1))
    ):
        raise UserError("resource projection exceeds aggregate cap")


def load_spec(path=DEFAULT_SPEC):
    path = Path(path).resolve()
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("invalid targeted supplement spec: {}".format(error))
    if spec.get("schema") != SPEC_SCHEMA:
        raise UserError("unsupported targeted supplement schema")
    if spec.get("split") != "val_seen" or spec.get("settings") != list(SETTINGS):
        raise UserError("targeted supplement scope must be two-model val_seen")
    if spec.get("setting_episode_counts") != {item: 778 for item in SETTINGS}:
        raise UserError("both settings must contain 778 full episodes")
    _repo_reference(spec.get("parent_evidence", {}).get("analysis"), "parent analysis")
    _repo_reference(
        spec.get("parent_evidence", {}).get("search_spec"), "parent search spec"
    )
    _repo_reference(spec.get("reused_source_controls"), "reused Source controls")
    _validate_canonical_order(spec)
    _validate_search_grid(spec)
    _validate_execution(spec)
    protocol = spec.get("protocol", {})
    if (
        protocol.get("source_control_execution")
        != "reuse_completed_formal_manifest"
        or protocol.get("source_execution_jobs") != 0
        or protocol.get("reset_from_source_between_jobs") is not True
        or protocol.get("require_formal_final_manifest") is not True
    ):
        raise UserError("Source reuse/restart/formal-manifest contract is invalid")
    # Reuse the shared evidence authenticator.  This is deliberately done at
    # load time so a missing or mutated Source artifact prevents any planning.
    for setting in SETTINGS:
        staged.reused_source_result(setting, 100, spec)
        staged.reused_source_result(setting, -1, spec)
    spec["_path"] = str(path)
    spec["_sha256"] = sha256(path)
    return spec


def phase_sequence(spec):
    phases = []
    for setting in spec["execution"]["model_order"]:
        phase_spec = spec["execution"]["phases"][setting]
        for kind in ("screening", "full_confirmation"):
            index = len(phases)
            values = phase_spec[kind]
            phases.append({
                "index": index,
                "phase_id": "{:02d}-{}-{}".format(index, setting, kind),
                "setting": setting,
                "model": SETTING_MODEL[setting],
                "kind": kind,
                "stage": SCREENING_STAGE if kind == "screening" else FULL_STAGE,
                "planned_jobs": values.get("planned_jobs"),
                "max_jobs": values.get("max_jobs"),
                "max_workers": values["max_workers"],
            })
    return phases


def campaign_root(batch_id):
    safe_component("batch_id", batch_id)
    return LOG_ROOT / batch_id


def phase_root(batch_id, phase):
    return campaign_root(batch_id) / "phases" / phase["phase_id"]


def _point_digest(method, parameters):
    value = {"method": method, "parameters": parameters}
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:10]


def _make_job(
    phase, batch_id, method, parameters, point_index, ordinal, spec, gpu,
    parent_run_tag=None,
):
    safe_component("method", method)
    digest = _point_digest(method, parameters)
    kind = "screen" if phase["kind"] == "screening" else "full"
    run_tag = (
        "{}-r2rce-supp-{}-{}-{}-{:02d}-{}".format(
            batch_id, phase["model"], kind, method, point_index, digest
        )
    )
    safe_component("run_tag", run_tag)
    root = phase_root(batch_id, phase)
    job_dir = root / "jobs" / run_tag
    config_path = job_dir / "parameters.json"
    result_parent = (
        TUNING_ROOT / batch_id / phase["model"] / method / kind / "jobs"
    )
    result_root = result_parent / run_tag / "val_seen"
    episodes = 100 if phase["kind"] == "screening" else -1
    command = [
        str(RUNNER), phase["setting"], "val_seen", str(gpu),
        "--run-tag", run_tag,
        "--tta-config", str(config_path),
        "--result-root", str(result_root),
    ]
    if episodes > 0:
        command.extend(["--episode-limit", str(episodes)])
    return {
        "batch_id": batch_id,
        "ordinal": ordinal,
        "point_index": point_index,
        "phase_index": phase["index"],
        "phase_id": phase["phase_id"],
        "base_run_tag": run_tag,
        "run_tag": run_tag,
        "attempt": 0,
        "setting": phase["setting"],
        "model": phase["model"],
        "family": "continuous",
        "gpu": int(gpu),
        "benchmark": "r2r-ce",
        "search_method": method,
        "config_method": method,
        "result_layout": RESULT_LAYOUT,
        "result_namespace": method,
        "stage": phase["stage"],
        "config_stage": phase["stage"],
        "episodes": episodes,
        "order_seed": None,
        "canonical_order_seed": 0,
        "canonical_order_sha256": spec["canonical_order"]["manifest"][
            "order_sha256"
        ],
        "parameters": parameters,
        "parent_run_tags": [parent_run_tag] if parent_run_tag else [],
        "restart_from_source_checkpoint": True,
        "config_path": str(config_path),
        "job_dir": str(job_dir),
        "result_root": str(result_root),
        "retry_result_root_parent": str(result_parent),
        "command": command,
    }


def build_screening_jobs(phase, batch_id, spec, gpu=0):
    if phase["kind"] != "screening":
        raise UserError("screening builder received a full phase")
    methods = enabled_methods(phase["setting"], spec)
    expanded = {
        method: expand_candidates(method, phase["setting"], spec)
        for method in methods
    }
    jobs = []
    # Candidate-index-major ordering makes the first wave span distinct
    # methods whenever the model has at least three enabled cells.
    for point_index in range(3):
        for method in methods:
            jobs.append(_make_job(
                phase, batch_id, method, expanded[method][point_index],
                point_index, len(jobs), spec, gpu,
            ))
    expected = int(phase["planned_jobs"])
    if len(jobs) != expected:
        raise UserError("{} planned {} jobs, built {}".format(
            phase["phase_id"], expected, len(jobs)
        ))
    return jobs


def build_full_jobs(phase, batch_id, promotions, spec, gpu=0):
    if phase["kind"] != "full_confirmation":
        raise UserError("full builder received a screening phase")
    if promotions.get("setting") != phase["setting"]:
        raise UserError("promotion setting does not match full phase")
    jobs = []
    for method in enabled_methods(phase["setting"], spec):
        selected = promotions["cells"][method].get("selected")
        if selected is None:
            continue
        jobs.append(_make_job(
            phase,
            batch_id,
            method,
            dict(selected["parameters"]),
            int(selected["point_index"]),
            len(jobs),
            spec,
            gpu,
            parent_run_tag=selected["run_tag"],
        ))
    if len(jobs) > int(phase["max_jobs"]):
        raise UserError("full phase exceeds reviewed finalist cap")
    return jobs


def _job_identity(job):
    return {
        key: job.get(key)
        for key in (
            "batch_id", "ordinal", "point_index", "phase_index", "phase_id",
            "base_run_tag", "setting", "model", "family", "benchmark",
            "gpu",
            "search_method", "config_method", "result_layout",
            "result_namespace", "stage", "config_stage", "episodes",
            "order_seed", "canonical_order_seed", "canonical_order_sha256",
            "parameters", "parent_run_tags", "restart_from_source_checkpoint",
            "config_path", "job_dir", "retry_result_root_parent",
        )
    }


def _validate_persisted_jobs(root, expected):
    actual = staged.load_jobs(root)
    if len(actual) != len(expected):
        raise UserError("persisted phase job count changed")
    by_base = {job["base_run_tag"]: job for job in actual}
    if set(by_base) != {job["base_run_tag"] for job in expected}:
        raise UserError("persisted phase base run tags changed")
    output = []
    for planned in expected:
        job = by_base[planned["base_run_tag"]]
        if canonical(_job_identity(job)) != canonical(_job_identity(planned)):
            raise UserError("persisted job identity changed: {}".format(
                planned["base_run_tag"]
            ))
        attempt = job.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise UserError("persisted job has invalid retry attempt")
        expected_tag = planned["base_run_tag"] + (
            "-retry{}".format(attempt) if attempt else ""
        )
        if job.get("run_tag") != expected_tag:
            raise UserError("persisted retry run tag is inconsistent")
        command = list(job.get("command", []))
        expected_command = list(planned["command"])
        expected_command[expected_command.index("--run-tag") + 1] = job["run_tag"]
        expected_command[expected_command.index("--result-root") + 1] = job[
            "result_root"
        ]
        if command != expected_command:
            raise UserError("persisted worker command changed")
        expected_limits = ["100"] if job["episodes"] == 100 else []
        actual_limits = [
            command[index + 1]
            for index, token in enumerate(command[:-1])
            if token == "--episode-limit"
        ]
        if actual_limits != expected_limits or "--order-seed" in command:
            raise UserError("persisted canonical-order command changed")
        result_root = Path(job["result_root"])
        expected_result_root = (
            Path(job["retry_result_root_parent"]) / job["run_tag"] / "val_seen"
        )
        if result_root != expected_result_root:
            raise UserError("persisted result root does not match retry tag")
        config = json.loads(Path(job["config_path"]).read_text(encoding="utf-8"))
        if canonical(config) != canonical(staged._job_config(job)):
            raise UserError("persisted runtime config changed")
        output.append(job)
    output.sort(key=lambda item: int(item["ordinal"]))
    if [item["ordinal"] for item in output] != list(range(len(output))):
        raise UserError("persisted phase ordinals are not contiguous")
    return output


def _phase_manifest(phase, batch_id, jobs, spec):
    return {
        "schema": PHASE_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "phase": phase,
        "job_count": len(jobs),
        "job_count_cap": (
            phase["planned_jobs"]
            if phase["kind"] == "screening" else phase["max_jobs"]
        ),
        "git_commit": git("rev-parse", "HEAD"),
        "spec_path": spec["_path"],
        "spec_sha256": spec["_sha256"],
        "canonical_order_seed": 0,
        "canonical_order_sha256": spec["canonical_order"]["manifest"][
            "order_sha256"
        ],
        "source_execution_jobs": 0,
        "restart_from_source_checkpoint": True,
        "shared_gpu_coordination": {
            "peer_campaign": spec["execution"]["parallel_peer_campaign"],
            "launch_guard_required": True,
            "active_reservation_required": True,
            "reservation_role": RESERVATION_ROLE,
        },
        "resource_limits": runtime_limits(phase, spec),
    }


def _immutable_json(path, document):
    path = Path(path)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserError("invalid immutable document {}: {}".format(path, error))
        if canonical(existing) != canonical(document):
            raise UserError("immutable document changed: {}".format(path))
        return existing
    staged.atomic_json(path, document)
    return document


def ensure_phase_plan(phase, batch_id, jobs, spec, resume):
    assert_campaign_head(campaign_root(batch_id))
    root = phase_root(batch_id, phase)
    manifest_path = root / "PHASE.json"
    expected_manifest = _phase_manifest(phase, batch_id, jobs, spec)
    if manifest_path.is_file():
        if not resume:
            raise UserError("phase exists; use --resume: {}".format(root))
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if canonical(existing) != canonical(expected_manifest):
            raise UserError("persisted phase manifest changed: {}".format(root))
        if not jobs:
            if list((root / "jobs").glob("**/job.json")):
                raise UserError("empty phase unexpectedly contains persisted jobs")
            return root, []
        return root, _validate_persisted_jobs(root, jobs)
    if root.exists() and any(root.iterdir()):
        raise UserError("phase directory is nonempty without PHASE.json")
    root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        staged._write_job(job)
    staged.atomic_json(manifest_path, expected_manifest)
    return root, jobs


def _campaign_plan(batch_id, spec):
    phases = phase_sequence(spec)
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "benchmark": spec["benchmark"],
        "split": "val_seen",
        "git_commit": git("rev-parse", "HEAD"),
        "spec_path": spec["_path"],
        "spec_sha256": spec["_sha256"],
        "canonical_order_seed": 0,
        "canonical_order_sha256": spec["canonical_order"]["manifest"][
            "order_sha256"
        ],
        "source_execution_jobs": 0,
        "screening_jobs": 15,
        "full_jobs_max": 5,
        "total_jobs_max": 20,
        "strict_model_barrier": True,
        "shared_gpu_coordination": {
            "peer_campaign": spec["execution"]["parallel_peer_campaign"],
            "launch_guard_required": True,
            "active_reservation_required": True,
            "reservation_role": RESERVATION_ROLE,
        },
        "phases": phases,
    }


def assert_campaign_head(root):
    """Refuse planning, validation, or launch after the campaign HEAD moves."""
    path = Path(root) / "PLAN.json"
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("invalid campaign PLAN.json: {}".format(error))
    if plan.get("schema") != PLAN_SCHEMA:
        raise UserError("campaign PLAN.json has an unsupported schema")
    current = git("rev-parse", "HEAD")
    if plan.get("git_commit") != current:
        raise UserError(
            "campaign Git commit changed: planned {}, current {}".format(
                plan.get("git_commit"), current
            )
        )
    spec_path = Path(str(plan.get("spec_path", ""))).resolve()
    try:
        spec_path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise UserError("campaign spec path escapes the repository") from error
    if not spec_path.is_file() or sha256(spec_path) != plan.get("spec_sha256"):
        raise UserError("campaign spec changed after PLAN.json was frozen")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise UserError("tracked worktree changed after campaign planning")
    return plan


def ensure_campaign_plan(cli, spec):
    root = campaign_root(cli.batch_id)
    root.mkdir(parents=True, exist_ok=True)
    plan = _campaign_plan(cli.batch_id, spec)
    plan_path = root / "PLAN.json"
    if plan_path.is_file() and not cli.resume:
        raise UserError("campaign exists; use --resume")
    _immutable_json(plan_path, plan)
    assert_campaign_head(root)
    planned = []
    for phase in phase_sequence(spec):
        if phase["kind"] != "screening":
            continue
        jobs = build_screening_jobs(phase, cli.batch_id, spec, cli.gpu)
        phase_path, persisted = ensure_phase_plan(
            phase, cli.batch_id, jobs, spec, cli.resume
        )
        planned.append((phase, phase_path, persisted))
    return root, planned


def _canonical_episode_ids(spec, count):
    _, document = _validate_canonical_order(spec)
    return [str(item["episode_id"]) for item in document["episodes"][:count]]


def _source_episode_records(setting, spec, count=100):
    controls_path = _repo_reference(
        spec["reused_source_controls"], "reused Source controls"
    )
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    reference = controls["settings"][setting]["per_episode_artifact"]
    path = _repo_reference(reference, "{} Source episode records".format(setting))
    records = json.loads(path.read_text(encoding="utf-8"))
    identifiers = _canonical_episode_ids(spec, count)
    if any(identifier not in records for identifier in identifiers):
        raise UserError("Source episode records do not cover canonical prefix")
    return identifiers, records, path


def count_navigation_record_changes(candidate, source, identifiers, fields):
    changes = 0
    for identifier in identifiers:
        if identifier not in candidate or identifier not in source:
            raise UserError("episode record coverage mismatch for {}".format(identifier))
        candidate_record = candidate[identifier]
        source_record = source[identifier]
        signatures = []
        for record, label in ((candidate_record, "candidate"), (source_record, "Source")):
            if not isinstance(record, dict):
                raise UserError("{} episode record is not an object".format(label))
            signature = {}
            for field in fields:
                value = record.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise UserError("{} record lacks numeric {}".format(label, field))
                if not math.isfinite(float(value)):
                    raise UserError("{} record contains nonfinite {}".format(label, field))
                signature[field] = value
            signatures.append(signature)
        if canonical(signatures[0]) != canonical(signatures[1]):
            changes += 1
    return changes


def enrich_screening_result(job, result, spec):
    paths = sorted(Path(job["result_root"]).glob(
        "metrics/source_val_seen/stats_ep*.json"
    ))
    if len(paths) != 1:
        raise UserError("screening job must produce exactly one episode-record file")
    try:
        candidate = json.loads(paths[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("invalid screening episode records: {}".format(error))
    identifiers, source, source_path = _source_episode_records(
        job["setting"], spec, 100
    )
    if list(candidate) != identifiers or len(candidate) != 100:
        raise UserError(
            "screening episode records are not the ordered canonical prefix"
        )
    fields = spec["selection"]["feedtta"]["navigation_record_fields"]
    output = dict(result)
    output["navigation_record_change_count"] = count_navigation_record_changes(
        candidate, source, identifiers, fields
    )
    output["navigation_record_artifact"] = str(paths[0])
    output["navigation_record_artifact_sha256"] = sha256(paths[0])
    output["matched_source_record_artifact"] = str(source_path)
    output["matched_source_record_artifact_sha256"] = sha256(source_path)
    return output


def _metrics_tied(left, right, tolerance):
    return all(
        math.isclose(
            float(left["metrics"][name]),
            float(right["metrics"][name]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for name in ("SPL", "SR")
    )


def select_finalist(method, setting, results, source, spec):
    tolerance = float(spec["selection"]["metric_tie_absolute_tolerance"])
    floor = float(source["metrics"]["SR"]) - float(
        spec["selection"]["source_sr_floor_tolerance_percentage_points"]
    )
    reviewed = []
    for result in results:
        reasons = []
        if result.get("setting") != setting or result.get("config_method") != method:
            reasons.append("cell_identity_mismatch")
        metrics = result.get("metrics", {})
        for name in ("SPL", "SR"):
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                reasons.append("missing_or_invalid_{}".format(name.lower()))
            elif not math.isfinite(float(value)):
                reasons.append("nonfinite_{}".format(name.lower()))
        if isinstance(metrics.get("SR"), (int, float)) and not isinstance(
            metrics.get("SR"), bool
        ) and float(metrics["SR"]) < floor:
            reasons.append("below_matched_source_sr_floor")
        adapter = result.get("adapter_diagnostics") or {}
        updates = adapter.get("updates")
        drift = adapter.get("relative_param_drift")
        if method in ("fstta", "feedtta") and (
            isinstance(updates, bool)
            or not isinstance(updates, (int, float))
            or not math.isfinite(float(updates))
            or float(updates) <= 0
        ):
            reasons.append("no_effective_updates")
        if method == "fstta":
            maximum = int(spec["selection"]["fstta"]["max_fast_window_m"])
            m_value = result.get("parameters", {}).get("m")
            if (
                isinstance(m_value, bool)
                or not isinstance(m_value, int)
                or m_value < 1
                or m_value > maximum
            ):
                reasons.append("fast_window_exceeds_episode_bound")
        change_count = result.get("navigation_record_change_count")
        if method == "feedtta" and (
            isinstance(change_count, bool)
            or not isinstance(change_count, int)
            or change_count < 0
        ):
            reasons.append("missing_navigation_record_change_count")
        reviewed.append({
            "result": result,
            "eligible": not reasons,
            "reasons": reasons,
            "score": [
                float(metrics.get("SPL", -1e300)),
                float(metrics.get("SR", -1e300)),
                -float(drift) if isinstance(drift, (int, float)) else -1e300,
                -float(updates) if isinstance(updates, (int, float)) else -1e300,
            ],
        })
    eligible = [item for item in reviewed if item["eligible"]]
    eligible.sort(key=lambda item: tuple(item["score"]), reverse=True)
    selected = None
    decision = "no_eligible_candidate"
    if eligible:
        if method != "feedtta":
            selected = eligible[0]["result"]
            decision = "highest_safe_spl_sr_then_stability"
        else:
            best = eligible[0]["result"]
            tied = [
                item["result"] for item in eligible
                if _metrics_tied(item["result"], best, tolerance)
            ]
            changed = [
                item for item in tied
                if int(item["navigation_record_change_count"]) > 0
            ]
            source_tie = _metrics_tied(best, source, tolerance)
            if changed:
                selected = max(
                    changed,
                    key=lambda item: (
                        float(item["parameters"]["lr"]),
                        int(item["navigation_record_change_count"]),
                    ),
                )
                decision = "metric_tie_highest_safe_nonzero_behavior_lr"
            elif not source_tie:
                # A genuine metric change cannot be a complete per-episode
                # record no-op; reaching this branch signals inconsistent
                # evidence and therefore remains fail-closed.
                decision = "inconsistent_metric_change_without_record_change"
            else:
                decision = "all_top_candidates_are_navigation_noops"
    record = {
        "method": method,
        "setting": setting,
        "source_sr_floor": floor,
        "decision": decision,
        "selected_run_tag": selected.get("run_tag") if selected else None,
        "ranked": [
            {
                "run_tag": item["result"].get("run_tag"),
                "eligible": item["eligible"],
                "reasons": item["reasons"],
                "score": item["score"],
                "navigation_record_change_count": item["result"].get(
                    "navigation_record_change_count"
                ),
            }
            for item in reviewed
        ],
    }
    return selected, record


def summarize_screening(phase, root, jobs, results, batch_id, spec):
    assert_campaign_head(campaign_root(batch_id))
    enriched = []
    by_tag = {item["run_tag"]: item for item in results}
    for job in jobs:
        try:
            result = by_tag[job["run_tag"]]
        except KeyError as error:
            raise UserError("screening result is missing {}".format(job["run_tag"])) from error
        value = enrich_screening_result(job, result, spec)
        staged.atomic_json(Path(job["job_dir"]) / "metrics.json", value)
        enriched.append(value)
    if len(enriched) != int(phase["planned_jobs"]):
        raise UserError("screening phase is incomplete")
    source = staged.reused_source_result(phase["setting"], 100, spec)
    cells = {}
    for method in enabled_methods(phase["setting"], spec):
        values = [item for item in enriched if item["config_method"] == method]
        if len(values) != 3:
            raise UserError("{} screening cell is incomplete".format(method))
        selected, record = select_finalist(
            method, phase["setting"], values, source, spec
        )
        cells[method] = {
            "selected": (
                {
                    "run_tag": selected["run_tag"],
                    "point_index": selected["point_index"],
                    "parameters": selected["parameters"],
                    "metrics": selected["metrics"],
                    "adapter_diagnostics": selected["adapter_diagnostics"],
                    "navigation_record_change_count": selected[
                        "navigation_record_change_count"
                    ],
                }
                if selected is not None else None
            ),
            "selection": record,
        }
    promotion = {
        "schema": "navtta.vln_r2r_ce_targeted_supplement_promotion.v1",
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "setting": phase["setting"],
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": spec["_sha256"],
        "canonical_prefix_episodes": 100,
        "canonical_order_seed": 0,
        "source": {
            "run_tag": source["run_tag"],
            "metrics": source["metrics"],
            "formal_manifest": source["reused_formal_manifest"],
            "formal_manifest_sha256": source[
                "reused_formal_manifest_sha256"
            ],
        },
        "cells": cells,
    }
    _immutable_json(Path(root) / "PROMOTION.json", promotion)
    assert_campaign_head(campaign_root(batch_id))
    return promotion


def validate_full_result_contract(job, result):
    """Bind the formal identity to this config and authenticated aggregates."""
    manifest_path = Path(str(result.get("formal_manifest_path", "")))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("invalid authenticated full manifest: {}".format(error))
    if manifest.get("config") != job["config_path"]:
        raise UserError("formal manifest config path does not match the job")
    method, tokens = config_cli.translate(
        job["setting"], job["config_path"],
        str(Path(job["result_root"]) / "tta_diagnostics.json"),
    )
    overrides = manifest.get("config_overrides")
    if (
        method != job["config_method"]
        or not isinstance(overrides, list)
        or overrides[-len(tokens):] != tokens
    ):
        raise UserError("formal manifest does not authenticate translated TTA config")
    artifacts = manifest.get("result_artifacts") or []
    aggregates = [
        Path(item["path"])
        for item in artifacts
        if isinstance(item, dict)
        and str(item.get("name", "")).startswith("metrics/")
        and "/stats_ckpt_" in "/" + str(item.get("name", ""))
        and "/stats_ep_" not in "/" + str(item.get("name", ""))
    ]
    if len(aggregates) != 1:
        raise UserError("formal manifest must authenticate one aggregate artifact")
    raw = json.loads(aggregates[0].read_text(encoding="utf-8"))
    mapping = {
        "steps_taken": "STEPS_TAKEN",
        "distance_to_goal": "DISTANCE_TO_GOAL",
        "success": "SR",
        "oracle_success": "OSR",
        "path_length": "PATH_LENGTH",
        "collisions": "COLLISIONS",
        "spl": "SPL",
        "ndtw": "NDTW",
        "sdtw": "SDTW",
        "ghost_cnt": "GHOST_CNT",
    }
    metrics = {}
    for raw_name, metric_name in mapping.items():
        value = raw.get(raw_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UserError("authenticated aggregate lacks {}".format(raw_name))
        value = float(value)
        if not math.isfinite(value):
            raise UserError("authenticated aggregate contains nonfinite metrics")
        metrics[metric_name] = (
            100.0 * value
            if metric_name in ("SR", "OSR", "SPL", "NDTW", "SDTW")
            else value
        )
    for name, value in result.get("metrics", {}).items():
        if name in metrics and not math.isclose(
            float(value), metrics[name], rel_tol=0.0, abs_tol=1e-3
        ):
            raise UserError(
                "console metric {} disagrees with authenticated aggregate".format(name)
            )
    hardware = manifest.get("hardware")
    if not isinstance(hardware, dict) or not hardware:
        raise UserError("formal manifest has no hardware provenance")
    return {
        "metrics": metrics,
        "job_config_sha256": sha256(job["config_path"]),
        "aggregate_artifact_path": str(aggregates[0]),
        "aggregate_artifact_sha256": sha256(aggregates[0]),
    }


def summarize_full(phase, root, jobs, results, batch_id, spec):
    assert_campaign_head(campaign_root(batch_id))
    by_tag = {item["run_tag"]: item for item in results}
    cells = {}
    for job in jobs:
        result = by_tag.get(job["run_tag"])
        if result is None:
            raise UserError("full result is missing {}".format(job["run_tag"]))
        if not result.get("formal_manifest_path"):
            raise UserError("full result lacks authenticated formal manifest")
        contract = validate_full_result_contract(job, result)
        cells[job["config_method"]] = {
            "run_tag": result["run_tag"],
            "parent_screening_run_tag": job["parent_run_tags"][0],
            "parameters": result["parameters"],
            "metrics": contract["metrics"],
            "adapter_diagnostics": result["adapter_diagnostics"],
            "job_config_path": job["config_path"],
            "job_config_sha256": contract["job_config_sha256"],
            "aggregate_artifact_path": contract["aggregate_artifact_path"],
            "aggregate_artifact_sha256": contract[
                "aggregate_artifact_sha256"
            ],
            "formal_manifest_path": result["formal_manifest_path"],
            "formal_manifest_sha256": result["formal_manifest_sha256"],
            "formal_immutable_identity_sha256": result[
                "formal_immutable_identity_sha256"
            ],
        }
    source = staged.reused_source_result(phase["setting"], -1, spec)
    document = {
        "schema": "navtta.vln_r2r_ce_targeted_supplement_confirmation.v1",
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "setting": phase["setting"],
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": spec["_sha256"],
        "canonical_full_episodes": 778,
        "canonical_order_seed": 0,
        "restart_from_source_checkpoint": True,
        "source": {
            "run_tag": source["run_tag"],
            "metrics": source["metrics"],
            "formal_manifest": source["reused_formal_manifest"],
            "formal_manifest_sha256": source[
                "reused_formal_manifest_sha256"
            ],
        },
        "cells": cells,
    }
    _immutable_json(Path(root) / "CONFIRMATION.json", document)
    assert_campaign_head(campaign_root(batch_id))
    return document


def runtime_limits(phase, spec):
    limits = spec["execution"]["resource_limits"]
    return {
        "max_workers": int(phase["max_workers"]),
        "max_per_model": int(phase["max_workers"]),
        "max_discrete_workers": 1,
        "max_continuous_workers": int(phase["max_workers"]),
        "max_gpu_memory_mib": int(limits["max_gpu_memory_mib_before_launch"]),
        "estimated_job_gpu_memory_mib": int(
            limits["estimated_job_gpu_memory_mib"]
        ),
        "max_aggregate_gpu_memory_mib": int(
            limits["max_aggregate_gpu_memory_mib"]
        ),
        "max_memory_gib": float(
            limits["max_cgroup_memory_gib_before_launch"]
        ),
        "estimated_job_memory_gib": float(limits["estimated_job_memory_gib"]),
        "max_aggregate_memory_gib": float(
            limits["max_aggregate_cgroup_memory_gib"]
        ),
        "launch_stagger": float(limits["launch_stagger_seconds"]),
        "resource_wait_timeout": float(limits["resource_wait_timeout_seconds"]),
    }


def runtime_args(cli, phase, spec):
    values = runtime_limits(phase, spec)
    return argparse.Namespace(
        method="r2r_ce_targeted_supplement",
        batch_id=cli.batch_id,
        settings=[phase["setting"]],
        gpu=cli.gpu,
        resume=cli.resume,
        retry_failed=cli.retry_failed,
        fail_fast=phase["kind"] == "full_confirmation",
        **values,
    )


def _reservation_token(job):
    return "r2r_ce_targeted:{}:{}".format(job["batch_id"], job["run_tag"])


def _reservation_metadata(args, job):
    return {
        "role": RESERVATION_ROLE,
        "batch_id": args.batch_id,
        "run_tag": job["run_tag"],
        "setting": job["setting"],
        "method": job["config_method"],
        "phase_id": job["phase_id"],
    }


def _job_process_token(job):
    try:
        payload = {
            "schema": "navtta.vln_r2r_ce_targeted_process_group.v1",
            "batch_id": job["batch_id"],
            "run_tag": job["run_tag"],
            "job_dir": str(Path(job["job_dir"]).resolve()),
            "setting": job["setting"],
            "method": job["config_method"],
            "phase_id": job["phase_id"],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("targeted job lacks process-token identity") from error
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def _proc_token_processes(token, proc_root=Path("/proc")):
    """Return live Linux processes authenticated by an inherited job token."""
    if (
        not Path(proc_root).is_dir()
        or re.fullmatch(r"[0-9a-f]{64}", str(token)) is None
    ):
        return []
    marker = "{}={}".format(PROCESS_TOKEN_ENV, token).encode("ascii")
    records = []
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw_stat = (entry / "stat").read_text(encoding="utf-8")
            tail = raw_stat[raw_stat.rfind(")") + 2:].split()
            state, pgid, sid = tail[0], int(tail[2]), int(tail[3])
            if state == "Z":
                continue
            environment = (entry / "environ").read_bytes().split(b"\0")
        except (IndexError, OSError, ValueError):
            continue
        if marker not in environment:
            continue
        identity = staged.process_identity(int(entry.name))
        if identity is not None and staged.process_alive(
            identity.get("pid"), identity
        ):
            records.append({"identity": identity, "pgid": pgid, "sid": sid})
    return sorted(records, key=lambda item: item["identity"]["pid"])


def _recorded_live_processes(job):
    """Fallback for local non-Linux tests; formal recovery uses procfs tokens."""
    path = Path(job["job_dir"]) / "worker_state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = []
    for pid_key, identity_key in (
        ("worker_pid", "worker_process"),
        ("runner_pid", "runner_process"),
    ):
        identity = state.get(identity_key)
        try:
            pid = int(state[pid_key])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            isinstance(identity, dict)
            and identity.get("pid") == pid
            and staged.process_alive(pid, identity)
        ):
            try:
                pgid, sid = os.getpgid(pid), os.getsid(pid)
            except OSError:
                continue
            stat_path = Path("/proc") / str(pid) / "stat"
            if stat_path.is_file():
                try:
                    raw_stat = stat_path.read_text(encoding="utf-8")
                    state = raw_stat[raw_stat.rfind(")") + 2:].split()[0]
                except (IndexError, OSError):
                    continue
                if state == "Z":
                    continue
            records.append({"identity": identity, "pgid": pgid, "sid": sid})
    return records


def _live_job_processes(job):
    records = _proc_token_processes(_job_process_token(job))
    seen = {
        (
            item["identity"].get("pid"),
            item["identity"].get("start_token"),
            item["identity"].get("cmdline_sha256"),
        )
        for item in records
    }
    for item in _recorded_live_processes(job):
        key = (
            item["identity"].get("pid"),
            item["identity"].get("start_token"),
            item["identity"].get("cmdline_sha256"),
        )
        if key not in seen:
            records.append(item)
            seen.add(key)
    return sorted(records, key=lambda item: item["identity"]["pid"])


def _live_job_identities(job):
    return [item["identity"] for item in _live_job_processes(job)]


def _release_reservation(args, job):
    if job.get("batch_id") != args.batch_id or job.get("gpu") != args.gpu:
        raise UserError("targeted reservation job differs from scheduler identity")
    release_shared_gpu_reservation(job["gpu"], _reservation_token(job))


def _claim_running_reservation(args, job, identities):
    if not identities:
        raise UserError("cannot reserve resources for a non-live targeted worker")
    if job.get("batch_id") != args.batch_id or job.get("gpu") != args.gpu:
        raise UserError("targeted reservation job differs from scheduler identity")
    token = _reservation_token(job)
    expected_metadata = _reservation_metadata(args, job)
    with shared_gpu_launch_guard(args.gpu) as ledger:
        record = ledger.document["reservations"].get(token)
        if record is None:
            gpu_memory, _ = staged.gpu_stats(args.gpu)
            memory = staged.cgroup_memory_gib()
            owner = staged.process_identity()
            if owner is None:
                raise UserError("cannot bind targeted scheduler process identity")
            ledger.reserve(
                token,
                gpu_memory_mib=args.estimated_job_gpu_memory_mib,
                cgroup_memory_gib=args.estimated_job_memory_gib,
                observed_gpu_memory_mib=gpu_memory,
                observed_cgroup_memory_gib=memory,
                owner=owner,
                metadata=expected_metadata,
            )
            record = ledger.document["reservations"][token]
        elif (
            record.get("gpu_memory_mib") != args.estimated_job_gpu_memory_mib
            or not math.isclose(
                float(record.get("cgroup_memory_gib", math.nan)),
                float(args.estimated_job_memory_gib),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or record.get("metadata") != expected_metadata
        ):
            raise UserError("targeted worker reservation binding mismatch")
        owner_keys = ("pid", "start_token", "cmdline_sha256")
        for identity in identities:
            if not any(
                all(owner.get(key) == identity.get(key) for key in owner_keys)
                for owner in record.get("owners", [])
            ):
                ledger.add_owner(token, identity)
    return token


def _reserved_resources_ok(args, ledger):
    gpu_memory, _ = staged.gpu_stats(args.gpu)
    memory = staged.cgroup_memory_gib()
    snapshot = ledger.snapshot(gpu_memory, memory)
    effective_gpu = snapshot["effective_gpu_memory_mib"]
    effective_memory = snapshot["effective_cgroup_memory_gib"]
    okay = (
        effective_gpu <= args.max_gpu_memory_mib
        and effective_memory <= args.max_memory_gib
        and effective_gpu + args.estimated_job_gpu_memory_mib
        <= args.max_aggregate_gpu_memory_mib
        and effective_memory + args.estimated_job_memory_gib
        <= args.max_aggregate_memory_gib
    )
    return okay, gpu_memory, memory, snapshot


def _cleanup_process_groups(job, process=None):
    records = _live_job_processes(job)
    pgids = {item["pgid"] for item in records}
    if process is not None and process.poll() is None:
        # This direct child was created by this scheduler with
        # start_new_session=True.  Include it even before procfs exposes the
        # inherited token or worker_state.json is written.
        pgids.add(process.pid)
    return records, pgids


def _terminate_launched_process(process, job, timeout_seconds=10.0):
    records, pgids = _cleanup_process_groups(job, process)
    if not records and not pgids:
        return True
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        records, pgids = _cleanup_process_groups(job, process)
        if not records and not pgids:
            return True
        time.sleep(0.1)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        records, pgids = _cleanup_process_groups(job, process)
        if not records and not pgids:
            return True
        time.sleep(0.1)
    return False


def _rollback_failed_launch(ledger, token, process, job):
    if not _terminate_launched_process(process, job):
        raise UserError(
            "failed launch still has authenticated descendants; "
            "reservation retained"
        )
    ledger.release(token)


def run_phase_batch(args, phase, root, jobs, spec):
    """Run one phase under the GPU lock shared with REVERIE campaigns."""
    assert_campaign_head(campaign_root(args.batch_id))
    running = {}
    pending = []
    terminal_failure = False
    for job in jobs:
        exit_path = Path(job["job_dir"]) / "exitcode"
        processes = _live_job_processes(job)
        if processes:
            running[job["run_tag"]] = {
                "process": None,
                "job": job,
                "processes": processes,
            }
            continue
        if exit_path.is_file():
            try:
                code = int(exit_path.read_text().strip())
            except ValueError:
                code = 255
            if code == 0:
                try:
                    parsed = staged.parse_metrics(job, spec)
                    if phase["kind"] == "screening":
                        enrich_screening_result(job, parsed, spec)
                    _release_reservation(args, job)
                    continue
                except Exception:
                    code = 255
            _release_reservation(args, job)
            if args.retry_failed:
                staged._bump_attempt(job)
                pending.append(job)
            else:
                terminal_failure = True
            continue
        if staged._worker_pid(job) is not None:
            _release_reservation(args, job)
            if args.retry_failed:
                staged._bump_attempt(job)
                pending.append(job)
            else:
                terminal_failure = True
        else:
            pending.append(job)

    stop_requested = False
    commit_error = None
    prior_handlers = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }

    def request_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    scheduler_log = Path(root) / "scheduler.log"
    resource_log = Path(root) / "resource.csv"
    last_observation = 0.0
    resource_blocked_since = None
    launch_blocked = terminal_failure and args.fail_fast
    try:
        with scheduler_log.open("a", encoding="utf-8", buffering=1) as log:
            def record(message):
                log.write("{} {}\n".format(
                    time.strftime("%Y-%m-%dT%H:%M:%S%z"), message
                ))

            record(
                "scheduler_start jobs={} workers={} resume={}".format(
                    len(jobs), args.max_workers, int(args.resume)
                )
            )
            while pending or running:
                for tag, active in list(running.items()):
                    exit_path = Path(active["job"]["job_dir"]) / "exitcode"
                    process = active["process"]
                    processes = _live_job_processes(active["job"])
                    if (
                        not processes
                        and process is not None
                        and process.poll() is None
                    ):
                        processes = [
                            item for item in active.get("processes", [])
                            if staged.process_alive(
                                item["identity"].get("pid"), item["identity"]
                            )
                        ]
                    if processes:
                        active["processes"] = processes
                        _claim_running_reservation(
                            args, active["job"], [
                                item["identity"] for item in processes
                            ]
                        )
                        continue
                    if exit_path.is_file():
                        try:
                            code = int(exit_path.read_text().strip())
                        except ValueError:
                            code = 255
                        if process is not None:
                            process.poll()
                        record("finish run_tag={} exit={}".format(
                            active["job"]["run_tag"], code
                        ))
                        if code != 0:
                            terminal_failure = True
                            launch_blocked = launch_blocked or args.fail_fast
                        _release_reservation(args, active["job"])
                        del running[tag]
                    else:
                        record("orphaned_worker run_tag={}".format(
                            active["job"]["run_tag"]
                        ))
                        terminal_failure = True
                        launch_blocked = launch_blocked or args.fail_fast
                        _release_reservation(args, active["job"])
                        del running[tag]

                now = time.time()
                if now - last_observation >= 5.0:
                    try:
                        assert_campaign_head(campaign_root(args.batch_id))
                    except UserError as error:
                        commit_error = error
                        stop_requested = True
                        launch_blocked = True
                    staged.append_resource(resource_log, running, args.gpu)
                    staged._progress(root, jobs, running, pending)
                    last_observation = now

                if stop_requested:
                    launch_blocked = True
                    terminal_failure = True
                    for active in running.values():
                        pgids = {
                            item["pgid"] for item in active.get("processes", [])
                        }
                        for pgid in pgids:
                            try:
                                os.killpg(pgid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass

                launched = False
                if (
                    len(running) < args.max_workers
                    and pending
                    and not launch_blocked
                    and not stop_requested
                ):
                    with shared_gpu_launch_guard(args.gpu) as ledger:
                        resources_ok, gpu_memory, memory, snapshot = (
                            _reserved_resources_ok(args, ledger)
                        )
                        if resources_ok:
                            resource_blocked_since = None
                            job = pending.pop(0)
                            assert_campaign_head(campaign_root(args.batch_id))
                            token = _reservation_token(job)
                            owner = staged.process_identity()
                            if owner is None:
                                raise UserError(
                                    "cannot bind targeted scheduler process identity"
                                )
                            ledger.reserve(
                                token,
                                gpu_memory_mib=args.estimated_job_gpu_memory_mib,
                                cgroup_memory_gib=args.estimated_job_memory_gib,
                                observed_gpu_memory_mib=gpu_memory,
                                observed_cgroup_memory_gib=memory,
                                owner=owner,
                                metadata=_reservation_metadata(args, job),
                            )
                            process = None
                            try:
                                process = subprocess.Popen(
                                    [
                                        sys.executable,
                                        str(Path(staged.__file__).resolve()),
                                        "--worker-job",
                                        str(Path(job["job_dir"]) / "job.json"),
                                    ],
                                    cwd=str(REPO_ROOT),
                                    start_new_session=True,
                                    env=dict(
                                        os.environ,
                                        **{
                                            PROCESS_TOKEN_ENV:
                                            _job_process_token(job)
                                        }
                                    ),
                                )
                                identity = staged.process_identity(process.pid)
                                if identity is None:
                                    raise UserError(
                                        "cannot bind worker process identity"
                                    )
                                ledger.add_owner(token, identity)
                            except Exception:
                                _rollback_failed_launch(
                                    ledger, token, process, job
                                )
                                raise
                            running[job["run_tag"]] = {
                                "process": process,
                                "job": job,
                                "processes": [{
                                    "identity": identity,
                                    "pgid": process.pid,
                                    "sid": process.pid,
                                }],
                            }
                            record("launch worker_pid={} run_tag={}".format(
                                process.pid, job["run_tag"]
                            ))
                            launched = True
                            # Keep the cross-campaign launch lock until CUDA
                            # allocation becomes visible to the next scheduler.
                            time.sleep(args.launch_stagger)
                            descendants = _live_job_processes(job)
                            if descendants:
                                running[job["run_tag"]]["processes"] = descendants
                                for item in descendants:
                                    descendant = item["identity"]
                                    if descendant.get("pid") != identity.get("pid"):
                                        ledger.add_owner(token, descendant)
                        else:
                            if resource_blocked_since is None:
                                resource_blocked_since = now
                                record(
                                    "resource_wait gpu_mib={} projected_gpu_mib={} "
                                    "memory_gib={:.3f} projected_memory_gib={:.3f}"
                                    .format(
                                        snapshot["effective_gpu_memory_mib"],
                                        snapshot["effective_gpu_memory_mib"]
                                        + args.estimated_job_gpu_memory_mib,
                                        snapshot["effective_cgroup_memory_gib"],
                                        snapshot["effective_cgroup_memory_gib"]
                                        + args.estimated_job_memory_gib,
                                    )
                                )
                            if (
                                not running
                                and now - resource_blocked_since
                                >= args.resource_wait_timeout
                            ):
                                raise UserError(
                                    "resource thresholds blocked all launches for "
                                    "{} seconds".format(args.resource_wait_timeout)
                                )
                if launch_blocked and not running:
                    break
                if not launched:
                    time.sleep(1.0)
            record("scheduler_finish failed={} pending={}".format(
                int(terminal_failure), len(pending)
            ))
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)

    staged._progress(root, jobs, running, pending)
    results, errors = staged.write_summary(root, jobs, spec)
    if commit_error is not None:
        raise commit_error
    assert_campaign_head(campaign_root(args.batch_id))
    terminal = len(results) + len(errors) == len(jobs)
    if stop_requested or not terminal or terminal_failure or errors:
        raise UserError("phase completed with failed, invalid, or pending jobs")
    if phase["kind"] == "screening":
        for job, result in zip(jobs, results):
            enrich_screening_result(job, result, spec)
    return results


def _phase_started(root):
    try:
        jobs = staged.load_jobs(root)
    except staged.UserError:
        return False
    return any(
        (Path(job["job_dir"]) / name).exists()
        for job in jobs
        for name in ("worker_state.json", "exitcode")
    )


def _load_valid_results(root, jobs, spec):
    if jobs:
        assert_campaign_head(campaign_root(jobs[0]["batch_id"]))
    results, errors = staged.write_summary(root, jobs, spec)
    if errors or len(results) != len(jobs):
        raise UserError("phase is incomplete or contains invalid results: {}".format(root))
    return results


def _phase_complete(root, jobs, spec):
    if not jobs:
        return (Path(root) / "EMPTY_COMPLETE.json").is_file()
    try:
        results = _load_valid_results(root, jobs, spec)
        if jobs and jobs[0]["stage"] == SCREENING_STAGE:
            by_tag = {item["run_tag"]: item for item in results}
            for job in jobs:
                enrich_screening_result(job, by_tag[job["run_tag"]], spec)
        return True
    except (UserError, staged.UserError):
        return False


def _mark_empty_phase(root, phase, batch_id, spec):
    assert_campaign_head(campaign_root(batch_id))
    _immutable_json(Path(root) / "EMPTY_COMPLETE.json", {
        "schema": "navtta.vln_r2r_ce_targeted_supplement_empty_phase.v1",
        "phase_id": phase["phase_id"],
        "reason": "no_candidate_passed_the_predeclared_promotion_gate",
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": spec["_sha256"],
    })


def _assert_no_downstream_started(all_phases, active_index, batch_id):
    for phase in all_phases[active_index + 1:]:
        root = phase_root(batch_id, phase)
        if root.is_dir() and _phase_started(root):
            raise UserError(
                "downstream phase started before barrier: {}".format(
                    phase["phase_id"]
                )
            )


@contextmanager
def campaign_lock(root):
    path = Path(root) / ".scheduler.lock"
    payload = {"pid": os.getpid(), "created_at_unix": time.time()}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            pid = int(existing["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pid = None
        if pid is not None and staged.process_alive(pid):
            raise UserError("another targeted scheduler is active: pid {}".format(pid))
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
    try:
        yield
    finally:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if int(existing.get("pid", -1)) == os.getpid():
                path.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            pass


def _assert_launchable(cli, spec):
    activation = spec.get("activation", {})
    if (
        activation.get("manual_review_required") is True
        or activation.get("automatic_launch_forbidden") is True
    ) and not cli.confirm_reviewed:
        raise UserError("formal launch requires --confirm-reviewed")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise UserError("formal launch requires a clean tracked worktree")
    required = (
        Path(__file__).resolve(),
        Path(spec["_path"]).resolve(),
        REPO_ROOT / "vln/experiments/R2R_CE_TARGETED_SUPPLEMENT_V1_PLAN.md",
    )
    for path in required:
        try:
            relative = path.relative_to(REPO_ROOT.resolve()).as_posix()
            git("ls-files", "--error-unmatch", "--", relative)
        except (ValueError, subprocess.CalledProcessError) as error:
            raise UserError(
                "formal launch requires tracked implementation file: {}".format(path)
            ) from error
    untracked = git(
        "ls-files", "--others", "--exclude-standard", "--",
        "core", "tools", "vln/baselines", "vln/navtta_vln", "vln/scripts",
        "vln/experiments", "vln/manifests",
    )
    if untracked:
        raise UserError(
            "formal launch refuses untracked execution files: {}".format(
                ", ".join(untracked.splitlines())
            )
        )
    load_spec(spec["_path"])


def aggregate_campaign(batch_id, spec, confirmations):
    assert_campaign_head(campaign_root(batch_id))
    document = {
        "schema": "navtta.vln_r2r_ce_targeted_supplement_results.v1",
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "benchmark": spec["benchmark"],
        "split": "val_seen",
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": spec["_sha256"],
        "canonical_order_seed": 0,
        "source_execution_jobs": 0,
        "supervision_groups": spec["protocol"]["supervision_groups"],
        "settings": {
            item["setting"]: item for item in confirmations
        },
    }
    _immutable_json(campaign_root(batch_id) / "RESULTS.json", document)
    assert_campaign_head(campaign_root(batch_id))
    return document


def execute_campaign(cli, spec):
    if not cli.plan_only:
        _assert_launchable(cli, spec)
    root, initial = ensure_campaign_plan(cli, spec)
    screening_by_setting = {
        phase["setting"]: (phase, phase_path, jobs)
        for phase, phase_path, jobs in initial
    }
    phases = phase_sequence(spec)
    if cli.plan_only:
        print(
            "campaign={} screening_jobs=15 full_jobs_max=5 root={}".format(
                cli.batch_id, root
            )
        )
        for phase in phases:
            jobs = phase.get("planned_jobs") or phase.get("max_jobs")
            print(
                "phase={} jobs={}{} workers={} setting={}".format(
                    phase["phase_id"],
                    "max:" if phase["kind"] == "full_confirmation" else "",
                    jobs,
                    phase["max_workers"],
                    phase["setting"],
                )
            )
        if cli.print_commands:
            for _, _, jobs in initial:
                for job in jobs:
                    print(subprocess.list2cmdline(job["command"]))
        return

    confirmations = []
    completed_phases = []
    with campaign_lock(root):
        for setting in spec["execution"]["model_order"]:
            screening_phase, screening_root, screening_jobs = (
                screening_by_setting[setting]
            )
            screening_complete = _phase_complete(
                screening_root, screening_jobs, spec
            )
            if not screening_complete:
                _assert_no_downstream_started(
                    phases, screening_phase["index"], cli.batch_id
                )
                results = run_phase_batch(
                    runtime_args(cli, screening_phase, spec),
                    screening_phase,
                    screening_root,
                    screening_jobs,
                    spec,
                )
            else:
                results = _load_valid_results(
                    screening_root, screening_jobs, spec
                )
            promotion = summarize_screening(
                screening_phase, screening_root, screening_jobs, results,
                cli.batch_id, spec,
            )
            completed_phases.append((screening_root, screening_jobs))

            full_phase = phases[screening_phase["index"] + 1]
            full_jobs = build_full_jobs(
                full_phase, cli.batch_id, promotion, spec, cli.gpu
            )
            full_root, full_jobs = ensure_phase_plan(
                full_phase, cli.batch_id, full_jobs, spec, cli.resume
            )
            if any(
                not _phase_complete(path, phase_jobs, spec)
                for path, phase_jobs in completed_phases
            ):
                raise UserError("upstream phase is incomplete; barrier refused advance")
            if not full_jobs:
                if not _phase_complete(full_root, full_jobs, spec):
                    _assert_no_downstream_started(
                        phases, full_phase["index"], cli.batch_id
                    )
                _mark_empty_phase(
                    full_root, full_phase, cli.batch_id, spec
                )
                full_results = []
            else:
                full_complete = _phase_complete(full_root, full_jobs, spec)
                if not full_complete:
                    _assert_no_downstream_started(
                        phases, full_phase["index"], cli.batch_id
                    )
                    full_results = run_phase_batch(
                        runtime_args(cli, full_phase, spec),
                        full_phase,
                        full_root,
                        full_jobs,
                        spec,
                    )
                else:
                    full_results = _load_valid_results(
                        full_root, full_jobs, spec
                    )
            confirmations.append(summarize_full(
                full_phase, full_root, full_jobs, full_results,
                cli.batch_id, spec,
            ))
            completed_phases.append((full_root, full_jobs))
        aggregate_campaign(cli.batch_id, spec, confirmations)


def status_snapshot(batch_id):
    root = campaign_root(batch_id)
    try:
        plan = json.loads((root / "PLAN.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("campaign is not planned: {}".format(error))
    values = []
    for phase in plan.get("phases", []):
        phase_path = phase_root(batch_id, phase)
        try:
            jobs = staged.load_jobs(phase_path)
        except staged.UserError:
            jobs = []
        succeeded = failed = running = 0
        for job in jobs:
            exit_path = Path(job["job_dir"]) / "exitcode"
            if exit_path.is_file():
                try:
                    code = int(exit_path.read_text().strip())
                except ValueError:
                    code = 255
                succeeded += int(code == 0)
                failed += int(code != 0)
                continue
            if _live_job_identities(job):
                running += 1
        values.append({
            "phase_id": phase["phase_id"],
            "setting": phase["setting"],
            "kind": phase["kind"],
            "planned": len(jobs) if jobs else phase.get("max_jobs", 0),
            "materialized": len(jobs),
            "succeeded": succeeded,
            "failed": failed,
            "running": running,
            "pending": max(0, len(jobs) - succeeded - failed - running),
        })
    return {"batch_id": batch_id, "phases": values}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if not args.spec.is_absolute():
        args.spec = (REPO_ROOT / args.spec).resolve()
    else:
        args.spec = args.spec.resolve()
    try:
        safe_component("batch_id", args.batch_id)
    except UserError as error:
        parser.error(str(error))
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if args.print_commands and not args.plan_only:
        parser.error("--print-commands requires --plan-only")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        print(json.dumps(status_snapshot(args.batch_id), indent=2, sort_keys=True))
        return
    spec = load_spec(args.spec)
    execute_campaign(args, spec)


if __name__ == "__main__":
    try:
        main()
    except (UserError, staged.UserError, ReservationLedgerError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
