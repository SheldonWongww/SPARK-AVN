#!/usr/bin/env python3
"""Run the frozen 2-model x 5-method R2R-CE val_unseen matrix.

The runner executes ten TTA jobs and zero Source jobs.  Hyperparameters come
verbatim from a completed val_seen registry and cannot be selected or changed
using val_unseen outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

import build_r2r_ce_final_registry as registry_builder  # noqa: E402
import run_tta_hparam_search as resources  # noqa: E402
import tta_config_cli as config_cli  # noqa: E402
import joint_campaign_contract as joint_contract  # noqa: E402
import shared_gpu_launch_guard as gpu_guard  # noqa: E402
from joint_campaign_contract import (  # noqa: E402
    JointLaunchError,
    campaign_lifetime_lock,
    process_identity,
    process_identity_alive,
)
from shared_gpu_launch_guard import (  # noqa: E402
    ReservationLedgerError,
    release_shared_gpu_reservation,
    shared_gpu_launch_guard,
)
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


SCHEMA = "navtta.vln_r2r_ce_val_unseen_frozen_eval.v1"
JOB_SCHEMA = "navtta.vln_r2r_ce_val_unseen_frozen_job.v1"
BATCH_SCHEMA = "navtta.vln_r2r_ce_val_unseen_frozen_batch.v1"
SUMMARY_SCHEMA = "navtta.vln_r2r_ce_val_unseen_frozen_summary.v1"
DEFAULT_SPEC = REPO_ROOT / "vln/experiments/r2r_ce_val_unseen_frozen_eval_v1.json"
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
LOG_ROOT = REPO_ROOT / "vln/results/logs/r2r-ce/frozen_val_unseen"
TUNING_ROOT = REPO_ROOT / "vln/results/tuning"
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"
SETTINGS = registry_builder.SETTINGS
MODELS = registry_builder.MODELS
METHODS = registry_builder.METHODS
MODEL_FOR_SETTING = registry_builder.MODEL_FOR_SETTING
CANONICAL_BATCH_ID = "vln-r2r-ce-val-unseen-frozen-eval-v1-seed0"
MAX_ATTEMPTS = 2
PROCESS_GROUP_ENV = "NAVTTA_R2R_CE_PROCESS_GROUP_TOKEN"
METRIC_MAP = {
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


class UserError(RuntimeError):
    pass


def _read_json(path, label="JSON"):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("invalid {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise UserError("{} must be an object".format(label))
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        "{}.{}.{}.tmp".format(path.name, os.getpid(), time.time_ns())
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        "{}.{}.{}.tmp".format(path.name, os.getpid(), time.time_ns())
    )
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _expected_process_group_token(metadata, attempt_dir):
    return hashlib.sha256(_canonical({
        "schema": "navtta.vln_r2r_ce_process_group.v1",
        "batch_id": metadata["batch_id"],
        "run_tag": metadata["run_tag"],
        "attempt_dir": str(Path(attempt_dir).resolve()),
        "spec_sha256": metadata["spec_sha256"],
        "git_commit": metadata["git_commit"],
    }).encode("utf-8")).hexdigest()


def _valid_sha256(value):
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_file(value, label, required_root=None):
    path = _resolve(value).resolve()
    root = (required_root or REPO_ROOT).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise UserError("{} is outside {}".format(label, root)) from error
    if not path.is_file():
        raise UserError("missing {}: {}".format(label, path))
    return path


def _git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _tracked_worktree_dirty():
    return bool(subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip())


def _untracked_execution_files():
    result = subprocess.run([
        "git", "-C", str(REPO_ROOT), "ls-files", "--others",
        "--exclude-standard", "--", "core", "tools", "vln/baselines",
        "vln/navtta_vln", "vln/scripts", "vln/experiments", "vln/manifests",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        raise UserError("cannot inspect untracked execution files")
    return [line for line in result.stdout.splitlines() if line]


def _require_tracked_inputs(spec_path, spec, registry):
    paths = {
        Path(__file__).resolve(),
        Path(registry_builder.__file__).resolve(),
        Path(registry_builder.targeted_runner.__file__).resolve(),
        Path(registry_builder.staged_runner.__file__).resolve(),
        Path(config_cli.__file__).resolve(),
        Path(resources.__file__).resolve(),
        Path(joint_contract.__file__).resolve(),
        Path(gpu_guard.__file__).resolve(),
        RUNNER.resolve(),
        Path(spec_path).resolve(),
        _resolve(spec["registry_dependency"]["path"]).resolve(),
        _resolve(spec["source_control"]["manifest"]).resolve(),
        _resolve(spec["protocol"]["order_manifest"]["path"]).resolve(),
        _resolve(registry["selected_winners"]["path"]).resolve(),
        _resolve(registry["source_ledger"]["path"]).resolve(),
    }
    for setting in SETTINGS:
        for method in ("source",) + METHODS:
            paths.add(_resolve(
                registry["records"][setting][method]["formal_manifest_path"]
            ).resolve())
    source_ledger = _read_json(
        _resolve(spec["source_control"]["manifest"]),
        "val_unseen Source ledger",
    )
    for setting in SETTINGS:
        paths.add(_resolve(
            source_ledger["records"][setting]["formal_manifest_path"]
        ).resolve())
    for path in paths:
        try:
            relative = path.relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError as error:
            raise UserError("formal input escapes repository: {}".format(path)) from error
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", relative],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            raise UserError("formal input must be committed before launch: {}".format(relative))
    untracked = _untracked_execution_files()
    if untracked:
        raise UserError("untracked execution files: {}".format(", ".join(untracked)))


def _validate_spec_structure(spec):
    if spec.get("schema") != SCHEMA:
        raise UserError("unsupported R2R-CE val_unseen frozen-eval schema")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(spec.get("experiment_id", ""))):
        raise UserError("invalid experiment_id")
    if spec.get("canonical_batch_id") != CANONICAL_BATCH_ID:
        raise UserError("frozen evaluation canonical batch ID mismatch")
    protocol = spec.get("protocol", {})
    expected_protocol = {
        "benchmark": "r2r_ce_v1_3_unified_etpnav_bevbert",
        "split": "val_unseen",
        "episode_count": 1839,
        "canonical_order_seed": 0,
        "episode_order_sha256": "e5a86bf609770d0525efc0be6527d8b88c02f932a0a35e1c4162b06172307a8a",
        "order_seed_cli_forbidden": True,
        "full_split_only": True,
        "selection_on_val_unseen": False,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise UserError("protocol {} mismatch".format(key))
    order = protocol.get("order_manifest", {})
    if set(order) != {"path", "sha256"}:
        raise UserError("order-manifest binding is malformed")
    order_path = _repo_file(order["path"], "episode-order manifest")
    if _sha256(order_path) != order["sha256"]:
        raise UserError("episode-order manifest SHA256 mismatch")
    order_document = _read_json(order_path, "episode-order manifest")
    if any(order_document.get(key) != value for key, value in {
        "schema": "navtta.episode_order.v1",
        "benchmark": protocol["benchmark"],
        "split": "val_unseen",
        "episode_count": 1839,
        "order_sha256": protocol["episode_order_sha256"],
    }.items()):
        raise UserError("episode-order manifest protocol mismatch")

    dependency = spec.get("registry_dependency", {})
    if dependency.get("schema") != registry_builder.REGISTRY_SCHEMA or (
        dependency.get("required_status") != "complete"
    ):
        raise UserError("registry dependency contract mismatch")
    registry_path = _repo_file(dependency.get("path", ""), "R2R-CE registry")
    if not _valid_sha256(dependency.get("sha256")) or _sha256(
        registry_path
    ) != dependency["sha256"]:
        raise UserError("R2R-CE registry SHA256 mismatch")

    source = spec.get("source_control", {})
    if source.get("execution") != "reuse_only" or source.get(
        "rerun_forbidden"
    ) is not True:
        raise UserError("Source must be reuse-only and rerun-forbidden")
    source_path = _repo_file(source.get("manifest", ""), "val_unseen Source ledger")
    if not _valid_sha256(source.get("sha256")) or _sha256(source_path) != source[
        "sha256"
    ]:
        raise UserError("val_unseen Source ledger SHA256 mismatch")

    matrix = spec.get("matrix", {})
    if tuple(matrix.get("setting_order", ())) != SETTINGS or tuple(
        matrix.get("model_order", ())
    ) != MODELS or tuple(matrix.get("method_order", ())) != METHODS:
        raise UserError("frozen matrix ordering mismatch")
    if matrix.get("strict_model_barrier") is not True or matrix.get(
        "max_workers"
    ) != 3:
        raise UserError("frozen matrix must use strict model barriers and 3 workers")
    jobs = matrix.get("jobs")
    expected_pairs = [(setting, method) for setting in SETTINGS for method in METHODS]
    if not isinstance(jobs, list) or len(jobs) != 10 or [
        (item.get("setting"), item.get("method")) for item in jobs
    ] != expected_pairs:
        raise UserError("frozen matrix must be model-major 2x5")
    for item in jobs:
        if set(item) != {
            "setting", "model", "method", "selected_run_tag",
            "selected_formal_manifest_sha256",
        }:
            raise UserError("frozen job fields mismatch")
        if item["model"] != MODEL_FOR_SETTING[item["setting"]]:
            raise UserError("frozen job model mismatch")
        if not _valid_sha256(item["selected_formal_manifest_sha256"]):
            raise UserError("frozen job manifest digest is invalid")
    if spec.get("budget") != {
        "source_execution_jobs": 0,
        "tta_jobs": 10,
        "total_executed_jobs": 10,
    }:
        raise UserError("budget must be exactly ten TTA and zero Source jobs")
    execution = spec.get("execution", {})
    if execution.get("max_workers") != 3 or execution.get(
        "strict_model_barrier"
    ) is not True:
        raise UserError("execution concurrency/barrier mismatch")
    if tuple(execution.get("model_order", ())) != MODELS:
        raise UserError("execution model order mismatch")
    if execution.get("formal_requires_clean_tracked_tree") is not True or (
        execution.get("formal_requires_review_confirmation") is not True
    ):
        raise UserError("formal launch safeguards are disabled")
    if execution.get("campaign_lifetime_lock_required") is not True or (
        execution.get("shared_gpu_reservation_required") is not True
    ):
        raise UserError("scheduler lock/reservation safeguards are disabled")
    if execution.get("retry_assigns_new_run_tag") is not True or execution.get(
        "max_attempts_per_job"
    ) != MAX_ATTEMPTS:
        raise UserError("retry policy must allow one fresh-run-tag retry")
    limits = execution.get("resource_limits", {})
    expected_limits = {
        "max_gpu_memory_mib_before_launch": 22000,
        "estimated_job_gpu_memory_mib": 8000,
        "max_aggregate_gpu_memory_mib": 30000,
        "max_cgroup_memory_gib_before_launch": 55.0,
        "estimated_job_memory_gib": 20.0,
        "max_aggregate_cgroup_memory_gib": 75.0,
        "resource_wait_timeout_seconds": 3600,
    }
    if limits != expected_limits or execution.get("launch_stagger_seconds") != 15:
        raise UserError("resource policy differs from the reviewed 3-worker profile")
    return registry_path, source_path


def load_registry(spec):
    path = _repo_file(spec["registry_dependency"]["path"], "R2R-CE registry")
    if _sha256(path) != spec["registry_dependency"]["sha256"]:
        raise UserError("R2R-CE registry changed after spec freeze")
    try:
        registry = registry_builder.validate_registry(path)
    except registry_builder.RegistryError as error:
        raise UserError("invalid R2R-CE registry: {}".format(error))
    return path, registry


def _entry(registry, setting, method):
    try:
        item = registry["records"][setting][method]
    except (KeyError, TypeError) as error:
        raise UserError("registry lacks {} {}".format(setting, method)) from error
    if not isinstance(item, dict):
        raise UserError("registry entry is malformed")
    return item


def _validate_matrix(spec, registry):
    for declared in spec["matrix"]["jobs"]:
        entry = _entry(registry, declared["setting"], declared["method"])
        if entry.get("run_tag") != declared["selected_run_tag"]:
            raise UserError("selected run mismatch for {} {}".format(
                declared["setting"], declared["method"]
            ))
        if entry.get("formal_manifest_sha256") != declared[
            "selected_formal_manifest_sha256"
        ]:
            raise UserError("selected manifest mismatch for {} {}".format(
                declared["setting"], declared["method"]
            ))


def validate_source_ledger(spec, require_artifacts=True):
    try:
        _, ledger = registry_builder.validate_val_unseen_source_ledger(
            {
                "protocol": {
                    "benchmark": spec["protocol"]["benchmark"],
                    "evaluation_order_sha256": spec["protocol"]["episode_order_sha256"],
                },
                "val_unseen_source_control": {
                    "path": spec["source_control"]["manifest"],
                    "sha256": spec["source_control"]["sha256"],
                },
            },
            REPO_ROOT,
            require_artifacts=require_artifacts,
        )
    except registry_builder.RegistryError as error:
        raise UserError(str(error))
    _, registry = load_registry(spec)
    for setting in SETTINGS:
        if ledger["records"][setting]["checkpoint_sha256"] != _entry(
            registry, setting, "source"
        )["checkpoint_sha256"]:
            raise UserError("{} Source checkpoint differs across splits".format(setting))
    return ledger


def load_spec(path=DEFAULT_SPEC):
    path = Path(path).resolve()
    spec = _read_json(path, "R2R-CE frozen-eval spec")
    _validate_spec_structure(spec)
    _, registry = load_registry(spec)
    _validate_matrix(spec, registry)
    validate_source_ledger(spec, require_artifacts=False)
    return spec


def _job_digest(setting, method, entry):
    value = {
        "setting": setting,
        "method": method,
        "selected_run_tag": entry["run_tag"],
        "parameters": entry["parameters"],
    }
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:10]


def expand_jobs(spec, batch_id, gpu=0):
    _, registry = load_registry(spec)
    declared = {
        (item["setting"], item["method"]): item for item in spec["matrix"]["jobs"]
    }
    jobs = []
    ordinal = 0
    for model_index, setting in enumerate(SETTINGS):
        for method_index, method in enumerate(METHODS):
            entry = _entry(registry, setting, method)
            jobs.append({
                "setting": setting,
                "model": MODEL_FOR_SETTING[setting],
                "method": method,
                "model_index": model_index,
                "method_index": method_index,
                "ordinal": ordinal,
                "base_run_tag": "{}-frozen-{:02d}-{}-{:02d}-{}-{}".format(
                    batch_id, model_index, MODEL_FOR_SETTING[setting], method_index,
                    method, _job_digest(setting, method, entry),
                ),
                "parameters": json.loads(_canonical(entry["parameters"])),
                "selected_anchor": json.loads(_canonical(entry)),
                "selected_binding": declared[(setting, method)],
                "gpu": int(gpu),
            })
            ordinal += 1
    return jobs


def model_phases(jobs):
    phases = []
    for index, setting in enumerate(SETTINGS):
        phase = [job for job in jobs if job["model_index"] == index]
        if [job["method"] for job in phase] != list(METHODS) or any(
            job["setting"] != setting for job in phase
        ):
            raise UserError("{} phase is not the exact five-method matrix".format(setting))
        phases.append(phase)
    return phases


def _attempt_tag(base_tag, attempt):
    return base_tag if attempt == 0 else "{}-retry{}".format(base_tag, attempt)


def _attempt_dir(batch_root, job, attempt):
    return (
        Path(batch_root) / "models" /
        "{:02d}-{}".format(job["model_index"], job["model"]) /
        "jobs" / job["base_run_tag"] / "attempt-{:02d}".format(attempt)
    )


def materialize_attempt(spec, spec_path, batch_id, batch_root, job, attempt):
    if type(attempt) is not int or not 0 <= attempt < MAX_ATTEMPTS:
        raise UserError("attempt number is outside the frozen retry budget")
    existing = _existing_attempts(batch_root, job)
    if [index for index, _ in existing] != list(range(attempt)):
        raise UserError("attempt history is not a contiguous prefix")
    run_tag = _attempt_tag(job["base_run_tag"], attempt)
    attempt_dir = _attempt_dir(batch_root, job, attempt)
    result_root = TUNING_ROOT / run_tag / job["setting"] / "val_unseen"
    formal_manifest = (
        FORMAL_ROOT /
        "{}-{}-val_unseen-v1.3-unified".format(run_tag, job["setting"]) /
        "manifest.json"
    )
    if attempt_dir.exists():
        if any(attempt_dir.iterdir()):
            raise UserError("refusing to overwrite existing attempt: {}".format(attempt_dir))
        attempt_dir.rmdir()
    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = attempt_dir.with_name(
        ".{}.materializing.{}.{}".format(
            attempt_dir.name, os.getpid(), time.time_ns()
        )
    )
    ledger = validate_source_ledger(spec, require_artifacts=False)
    source = ledger["records"][job["setting"]]
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
            "selection_benchmark": "r2r-ce",
            "selection_split": "val_seen",
            "evaluation_split": "val_unseen",
            "selection_on_val_unseen": False,
            "canonical_order_seed": 0,
            "registry": spec["registry_dependency"],
            "selected_anchor": job["selected_anchor"],
            "source_ledger": spec["source_control"],
        },
    }
    config_path = attempt_dir / "parameters.json"
    command = [
        str(RUNNER), job["setting"], "val_unseen", str(job["gpu"]),
        "--run-tag", run_tag, "--tta-config", str(config_path),
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
        "gpu": job["gpu"],
        "setting": job["setting"],
        "model": job["model"],
        "method": job["method"],
        "parameters": job["parameters"],
        "selected_anchor": job["selected_anchor"],
        "selected_binding": job["selected_binding"],
        "episode_count": 1839,
        "canonical_order_seed": 0,
        "git_commit": _git_commit(),
        "expected_checkpoint_sha256": source["checkpoint_sha256"],
        "expected_dataset_sha256": source["dataset_sha256"],
        "expected_episode_order_sha256": spec["protocol"]["episode_order_sha256"],
        "episode_order_manifest": str(_resolve(
            spec["protocol"]["order_manifest"]["path"]
        ).resolve()),
        "episode_order_manifest_sha256": spec["protocol"]["order_manifest"][
            "sha256"
        ],
        "source_ledger_path": str(_resolve(
            spec["source_control"]["manifest"]
        ).resolve()),
        "source_ledger_sha256": spec["source_control"]["sha256"],
        "result_root": str(result_root),
        "formal_manifest": str(formal_manifest),
        "config_path": str(config_path),
        "command": command,
    }
    metadata["parameters_path"] = str(config_path)
    staging_dir.mkdir(parents=False, exist_ok=False)
    _atomic_json(staging_dir / "parameters.json", config)
    metadata["parameters_sha256"] = _sha256(staging_dir / "parameters.json")
    metadata["process_group_token"] = _expected_process_group_token(
        metadata, attempt_dir
    )
    if attempt > 0:
        parent = _attempt_dir(batch_root, job, attempt - 1)
        validation_path = parent / "validation_error.json"
        if _pid_alive(parent):
            raise UserError("cannot retry while the prior attempt is alive")
        _validate_validation_error(parent)
        metadata["retry_of"] = str(parent)
        metadata["retry_of_validation_error_sha256"] = _sha256(validation_path)
        parent_job = parent / "job.json"
        metadata["retry_of_job_sha256"] = (
            _sha256(parent_job) if parent_job.is_file() else None
        )
    _atomic_json(staging_dir / "job.json", metadata)
    os.replace(str(staging_dir), str(attempt_dir))
    return attempt_dir, metadata


def _existing_attempts(batch_root, job):
    root = _attempt_dir(batch_root, job, 0).parent
    if not root.is_dir():
        return []
    paths = sorted(path for path in root.glob("attempt-*") if path.is_dir())
    values = []
    for path in paths:
        match = re.fullmatch(r"attempt-([0-9]{2})", path.name)
        if match is None:
            raise UserError("malformed attempt directory: {}".format(path))
        values.append((int(match.group(1)), path))
    if len(values) > MAX_ATTEMPTS or [item[0] for item in values] != list(
        range(len(values))
    ):
        raise UserError("attempt directories exceed or skip the frozen retry budget")
    return values


def _latest_attempt(batch_root, job):
    attempts = _existing_attempts(batch_root, job)
    return attempts[-1] if attempts else None


def _attempt_pgid(attempt_dir):
    path = Path(attempt_dir) / "pid"
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise UserError("attempt process-group ID is malformed") from error
    if value <= 0:
        raise UserError("attempt process-group ID is malformed")
    return value


def _identity_is_group_member(identity, pgid):
    if not process_identity_alive(identity):
        return False
    try:
        pid = int(identity["pid"])
        stat_path = Path("/proc") / str(pid) / "stat"
        if stat_path.is_file():
            raw_stat = stat_path.read_text(encoding="utf-8")
            tail = raw_stat[raw_stat.rfind(")") + 2:].split()
            return (
                tail[0] != "Z" and int(tail[2]) == pgid
                and int(tail[3]) == pgid
            )
        return os.getpgid(pid) == pgid and os.getsid(pid) == pgid
    except (IndexError, KeyError, OSError, TypeError, ValueError):
        return False


def _proc_group_member_identities(pgid, token, proc_root=Path("/proc")):
    """Authenticate live descendants after the session-leader shell dies.

    Formal execution is Linux-only.  Every command launched by ``worker.sh``
    inherits a per-attempt environment token; pairing that token with both the
    process-group and session IDs prevents an unrelated PID from being treated
    as this attempt after leader PID reuse.
    """
    if not Path(proc_root).is_dir() or not _valid_sha256(token):
        return []
    marker = "{}={}".format(PROCESS_GROUP_ENV, token).encode("ascii")
    identities = []
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
            state, member_pgid, member_sid = tail[0], int(tail[2]), int(tail[3])
            if state == "Z" or member_pgid != pgid or member_sid != pgid:
                continue
            environment = (entry / "environ").read_bytes().split(b"\0")
        except (IndexError, OSError, ValueError):
            continue
        if marker not in environment:
            continue
        identity = process_identity(int(entry.name))
        if identity is not None and process_identity_alive(identity):
            identities.append(identity)
    return sorted(identities, key=lambda item: item["pid"])


def _live_worker_identities(attempt_dir):
    attempt_dir = Path(attempt_dir)
    pgid = _attempt_pgid(attempt_dir)
    if pgid is None:
        return []
    identities = []
    identity_path = attempt_dir / "process_identity.json"
    if identity_path.is_file():
        leader = _read_json(identity_path, "process identity")
        if leader.get("pid") != pgid:
            raise UserError("worker identity differs from its process group")
        if _identity_is_group_member(leader, pgid):
            identities.append(leader)
    job_path = attempt_dir / "job.json"
    if not job_path.is_file():
        return identities
    metadata = _read_json(job_path, "job metadata")
    try:
        expected_token = _expected_process_group_token(metadata, attempt_dir)
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("attempt process-group binding is malformed") from error
    token = metadata.get("process_group_token")
    if token != expected_token:
        raise UserError("attempt process-group token is invalid")
    known = {item["pid"] for item in identities}
    identities.extend(
        item for item in _proc_group_member_identities(pgid, token)
        if item["pid"] not in known
    )
    return identities


def _pid_alive(attempt_dir):
    return bool(_live_worker_identities(attempt_dir))


def _recover_worker_identity(attempt_dir):
    attempt_dir = Path(attempt_dir)
    identity_path = attempt_dir / "process_identity.json"
    if identity_path.is_file():
        return _pid_alive(attempt_dir)
    pgid = _attempt_pgid(attempt_dir)
    if pgid is None:
        return False
    identity = process_identity(pgid)
    if identity is not None:
        worker = str((attempt_dir / "worker.sh").resolve())
        argv = [str(value) for value in identity.get("argv", [])]
        if worker not in argv and worker not in " ".join(argv):
            raise UserError("unbound live PID is not the canonical attempt worker")
        if not _identity_is_group_member(identity, pgid):
            raise UserError("attempt worker is not its recorded session leader")
        _atomic_json(identity_path, identity)
    return _pid_alive(attempt_dir)


def _state(attempt_dir):
    attempt_dir = Path(attempt_dir)
    if _pid_alive(attempt_dir):
        return "running"
    if (attempt_dir / "validation_error.json").is_file():
        return "invalid"
    if (attempt_dir / "metrics.json").is_file():
        return "completed"
    exit_path = attempt_dir / "exitcode"
    if exit_path.is_file():
        try:
            return "finished" if int(exit_path.read_text().strip()) == 0 else "failed"
        except ValueError:
            return "invalid"
    if (attempt_dir / "job.json").is_file() or any(attempt_dir.iterdir()):
        return "orphaned"
    return "pending"


def _validate_validation_error(attempt_dir):
    attempt_dir = Path(attempt_dir)
    path = attempt_dir / "validation_error.json"
    document = _read_json(path, "validation error")
    if set(document) != {
        "schema", "attempt", "run_tag", "job_sha256", "error_type",
        "error", "recorded_at_unix",
    } or document.get("schema") != "navtta.vln_r2r_ce_validation_error.v1":
        raise UserError("validation-error record is malformed")
    match = re.fullmatch(r"attempt-([0-9]{2})", attempt_dir.name)
    if match is None or document.get("attempt") != int(match.group(1)):
        raise UserError("validation-error attempt binding mismatch")
    if not isinstance(document.get("error_type"), str) or not isinstance(
        document.get("error"), str
    ) or isinstance(document.get("recorded_at_unix"), bool) or not isinstance(
        document.get("recorded_at_unix"), (int, float)
    ) or not math.isfinite(float(document["recorded_at_unix"])):
        raise UserError("validation-error payload is malformed")
    job_path = attempt_dir / "job.json"
    if job_path.is_file():
        job = _read_json(job_path, "failed attempt job")
        if document.get("run_tag") != job.get("run_tag") or document.get(
            "job_sha256"
        ) != _sha256(job_path):
            raise UserError("validation-error job binding mismatch")
    elif document.get("run_tag") is not None or document.get(
        "job_sha256"
    ) is not None:
        raise UserError("partial-attempt validation error claims a job")
    return document


def _validate_attempt_binding(attempt_dir, metadata, expected_job=None,
                              expected_batch_id=None, expected_spec_path=None):
    attempt_dir = Path(attempt_dir).resolve()
    if metadata.get("schema") != JOB_SCHEMA:
        raise UserError("attempt job schema mismatch")
    attempt = metadata.get("attempt")
    if type(attempt) is not int or not 0 <= attempt < MAX_ATTEMPTS:
        raise UserError("attempt number is invalid")
    base_fields = {
        "schema", "batch_id", "spec_path", "spec_sha256", "attempt",
        "run_tag", "base_run_tag", "model_index", "method_index", "ordinal",
        "gpu", "setting", "model", "method", "parameters", "selected_anchor",
        "selected_binding", "episode_count", "canonical_order_seed",
        "git_commit", "expected_checkpoint_sha256", "expected_dataset_sha256",
        "expected_episode_order_sha256", "episode_order_manifest",
        "episode_order_manifest_sha256",
        "source_ledger_path", "source_ledger_sha256", "result_root",
        "formal_manifest", "config_path", "command", "parameters_path",
        "parameters_sha256", "process_group_token",
    }
    retry_fields = {
        "retry_of", "retry_of_validation_error_sha256", "retry_of_job_sha256"
    } if attempt else set()
    if set(metadata) != base_fields | retry_fields:
        raise UserError("attempt job metadata fields mismatch")
    if metadata.get("episode_count") != 1839 or metadata.get(
        "canonical_order_seed"
    ) != 0:
        raise UserError("attempt evaluation protocol mismatch")
    if metadata.get("process_group_token") != _expected_process_group_token(
        metadata, attempt_dir
    ):
        raise UserError("attempt process-group token is invalid")
    config_path = attempt_dir / "parameters.json"
    if (
        Path(str(metadata.get("parameters_path", ""))).resolve() != config_path
        or Path(str(metadata.get("config_path", ""))).resolve() != config_path
        or (
        not config_path.is_file()
        or _sha256(config_path) != metadata.get("parameters_sha256")
        )
    ):
        raise UserError("attempt parameters path/digest is noncanonical")
    config = _read_json(config_path, "attempt parameters")
    expected_config = {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": metadata.get("batch_id"),
        "stage": "frozen_val_unseen",
        "setting": metadata.get("setting"),
        "method": metadata.get("method"),
        "search_method": metadata.get("method"),
        "episodes": -1,
        "parameters": metadata.get("parameters"),
    }
    if set(config) != set(expected_config) | {"frozen_evaluation_provenance"}:
        raise UserError("parameters.json has unexpected or missing fields")
    for key, value in expected_config.items():
        if _canonical(config.get(key)) != _canonical(value):
            raise UserError("parameters.json {} differs from job metadata".format(key))
    if "order_seed" in config:
        raise UserError("canonical seed-0 config must omit order_seed")
    provenance = config.get("frozen_evaluation_provenance")
    if not isinstance(provenance, dict) or _canonical(
        provenance.get("selected_anchor")
    ) != _canonical(metadata.get("selected_anchor")):
        raise UserError("parameters.json winner provenance mismatch")
    if (
        provenance.get("selection_benchmark") != "r2r-ce"
        or provenance.get("selection_split") != "val_seen"
        or provenance.get("evaluation_split") != "val_unseen"
        or provenance.get("selection_on_val_unseen") is not False
        or provenance.get("canonical_order_seed") != 0
    ):
        raise UserError("frozen val_unseen provenance mismatch")

    bound_spec = None
    if expected_spec_path is not None:
        spec_path = Path(expected_spec_path).resolve()
        if (
            Path(str(metadata.get("spec_path", ""))).resolve() != spec_path
            or not spec_path.is_file()
            or _sha256(spec_path) != metadata.get("spec_sha256")
        ):
            raise UserError("attempt spec binding mismatch")
        bound_spec = _read_json(spec_path, "bound frozen spec")
        if provenance.get("registry") != bound_spec.get("registry_dependency"):
            raise UserError("attempt registry provenance differs from spec")
        if metadata.get("git_commit") != _git_commit():
            raise UserError("attempt Git commit differs from current HEAD")
    source_path = _resolve(
        bound_spec["source_control"]["manifest"]
        if bound_spec is not None else metadata.get("source_ledger_path", "")
    ).resolve()
    if (
        Path(str(metadata.get("source_ledger_path", ""))).resolve() != source_path
        or not source_path.is_file()
        or _sha256(source_path) != metadata.get("source_ledger_sha256")
    ):
        raise UserError("attempt Source-ledger binding mismatch")
    source = _read_json(source_path, "bound Source ledger")
    try:
        source_record = source["records"][metadata["setting"]]
    except (KeyError, TypeError) as error:
        raise UserError("attempt Source record is missing") from error
    expected_order_path = _resolve(
        bound_spec["protocol"]["order_manifest"]["path"]
        if bound_spec is not None else metadata.get("episode_order_manifest", "")
    ).resolve()
    for key, value in (
        ("expected_checkpoint_sha256", source_record["checkpoint_sha256"]),
        ("expected_dataset_sha256", source_record["dataset_sha256"]),
        ("expected_episode_order_sha256", source_record["episode_order_sha256"]),
    ):
        if metadata.get(key) != value:
            raise UserError("attempt {} differs from Source evidence".format(key))
    if Path(str(metadata.get("episode_order_manifest", ""))).resolve() != expected_order_path:
        raise UserError("attempt episode-order path mismatch")
    expected_order_sha256 = (
        bound_spec["protocol"]["order_manifest"]["sha256"]
        if bound_spec is not None
        else metadata.get("episode_order_manifest_sha256")
    )
    if (
        not expected_order_path.is_file()
        or metadata.get("episode_order_manifest_sha256") != expected_order_sha256
        or _sha256(expected_order_path) != expected_order_sha256
    ):
        raise UserError("attempt episode-order manifest digest mismatch")
    source_binding = provenance.get("source_ledger", {})
    if (
        _resolve(source_binding.get("manifest", "")).resolve() != source_path
        or source_binding.get("sha256") != metadata["source_ledger_sha256"]
        or source_binding.get("execution") != "reuse_only"
        or source_binding.get("rerun_forbidden") is not True
    ):
        raise UserError("parameters.json Source provenance mismatch")
    if expected_batch_id is not None and metadata.get("batch_id") != expected_batch_id:
        raise UserError("attempt batch ID differs from scheduler batch")
    if expected_job is not None:
        for key in (
            "base_run_tag", "setting", "model", "method", "model_index",
            "method_index", "ordinal", "gpu", "parameters", "selected_anchor",
            "selected_binding",
        ):
            if _canonical(metadata.get(key)) != _canonical(expected_job.get(key)):
                raise UserError("attempt differs from expanded job: {}".format(key))
        expected_dir = _attempt_dir(attempt_dir.parents[4], expected_job, attempt).resolve()
        if attempt_dir != expected_dir:
            raise UserError("attempt directory is noncanonical")
        expected_tag = _attempt_tag(expected_job["base_run_tag"], attempt)
        if metadata.get("run_tag") != expected_tag:
            raise UserError("attempt run tag differs from retry identity")
        if attempt == 0:
            if "retry_of" in metadata:
                raise UserError("initial attempt cannot claim retry provenance")
        else:
            expected_parent = _attempt_dir(
                attempt_dir.parents[4], expected_job, attempt - 1
            ).resolve()
            if Path(str(metadata.get("retry_of", ""))).resolve() != expected_parent or (
                not expected_parent.is_dir()
            ):
                raise UserError("retry does not preserve its prior attempt")
            validation_path = expected_parent / "validation_error.json"
            if _pid_alive(expected_parent):
                raise UserError("retry parent is still running")
            _validate_validation_error(expected_parent)
            if metadata.get("retry_of_validation_error_sha256") != _sha256(
                validation_path
            ):
                raise UserError("retry validation-error provenance mismatch")
            parent_job_path = expected_parent / "job.json"
            expected_parent_sha = (
                _sha256(parent_job_path) if parent_job_path.is_file() else None
            )
            if metadata.get("retry_of_job_sha256") != expected_parent_sha:
                raise UserError("retry parent-job provenance mismatch")
            if parent_job_path.is_file():
                parent_metadata = _read_json(parent_job_path, "retry parent job")
                _validate_attempt_binding(
                    expected_parent, parent_metadata, expected_job,
                    expected_batch_id, expected_spec_path,
                )
    expected_result = (
        TUNING_ROOT / metadata["run_tag"] / metadata["setting"] / "val_unseen"
    ).resolve()
    if Path(str(metadata.get("result_root", ""))).resolve() != expected_result:
        raise UserError("attempt result root is noncanonical")
    expected_formal = (
        FORMAL_ROOT /
        "{}-{}-val_unseen-v1.3-unified".format(
            metadata["run_tag"], metadata["setting"]
        ) / "manifest.json"
    ).resolve()
    if Path(str(metadata.get("formal_manifest", ""))).resolve() != expected_formal:
        raise UserError("attempt formal-manifest path is noncanonical")
    expected_command = [
        str(RUNNER), metadata["setting"], "val_unseen", str(metadata["gpu"]),
        "--run-tag", metadata["run_tag"], "--tta-config",
        str(metadata["config_path"]),
    ]
    if metadata.get("command") != expected_command:
        raise UserError("attempt command differs from immutable metadata")
    return config


def _aggregate_metrics(path):
    raw = _read_json(path, "R2R-CE aggregate")
    metrics = {}
    for raw_name, name in METRIC_MAP.items():
        value = raw.get(raw_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (
            not math.isfinite(float(value))
        ):
            raise UserError("aggregate lacks finite {}".format(raw_name))
        value = float(value)
        metrics[name] = 100.0 * value if name in {
            "SR", "OSR", "SPL", "NDTW", "SDTW"
        } else value
    return metrics


def _validate_diagnostics(metadata, diagnostics, metrics):
    method = metadata["method"]
    episodes = metadata["episode_count"]
    if (
        diagnostics.get("method") != method
        or diagnostics.get("episode_count") != episodes
        or diagnostics.get("action_selection") != "target_native_argmax"
    ):
        raise UserError("TTA diagnostics identity/episode count mismatch")
    adapter = diagnostics.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("episodes") != episodes:
        raise UserError("adapter episode count mismatch")
    updates = adapter.get("updates")
    drift = adapter.get("relative_param_drift")
    if type(updates) is not int or updates <= 0:
        raise UserError("frozen winner made no effective updates")
    if isinstance(drift, bool) or not isinstance(drift, (int, float)) or (
        not math.isfinite(float(drift)) or float(drift) < 0.0
    ):
        raise UserError("adapter parameter drift is invalid")

    def count(key):
        value = adapter.get(key)
        if type(value) is not int or not 0 <= value <= episodes:
            raise UserError(
                "adapter.{} must be an integer in [0, {}]".format(key, episodes)
            )
        return value

    if method == "fstta" and adapter.get("variance_history_lifetime") != "test_stream":
        raise UserError("FSTTA did not preserve stream variance history")
    if method == "feedtta":
        if (
            diagnostics.get("feedback_supervision") != "binary_episode_success"
            or diagnostics.get("binary_feedback_endpoint") is not None
            or adapter.get("action_selection_protocol") != "target_native_argmax"
        ):
            raise UserError("FeedTTA did not use target-native argmax")
        if diagnostics.get("feedtta_scope_profile") != metadata[
            "parameters"
        ].get("scope_profile"):
            raise UserError("FeedTTA scope-profile mismatch")
        feedback = count("feedback_episodes")
        successes = count("successful_feedback_episodes")
        failures = count("failed_feedback_episodes")
        if (
            feedback != episodes
            or successes + failures != episodes
            or updates != episodes
            or adapter.get("feedback_type") != "binary_episode_success"
        ):
            raise UserError("FeedTTA feedback accounting mismatch")
        if successes != int(round(metrics["SR"] * episodes / 100.0)):
            raise UserError("FeedTTA feedback count disagrees with evaluator SR")
    if method == "atena":
        if diagnostics.get("feedback_supervision") != "binary_episode_success" or (
            diagnostics.get("binary_feedback_endpoint") is not None
        ):
            raise UserError("ATENA feedback-supervision mismatch")
        queries = count("queries")
        self_labels = count("self_label_episodes")
        observed = count("feedback_observed_episodes")
        gate = count("query_gate_evaluations")
        self_evaluations = count("self_prediction_evaluations")
        queried_successes = count("queried_feedback_successes")
        self_successes = count("self_feedback_successes")
        if (
            queries + self_labels != episodes
            or observed != queries
            or gate != episodes
            or self_evaluations != episodes
            or queried_successes > queries
            or self_successes > self_labels
            or updates != episodes
            or adapter.get("action_selection") != "argmax"
        ):
            raise UserError("ATENA feedback accounting mismatch")
        if float(metadata["parameters"].get("query_threshold", math.nan)) == 0.0 and (
            queries != episodes
        ):
            raise UserError("zero-threshold ATENA must query every episode")
    if method in ("tent", "fstta", "eam"):
        if diagnostics.get("feedback_supervision") != "none" or diagnostics.get(
            "binary_feedback_endpoint"
        ) is not None:
            raise UserError("unsupervised method unexpectedly reports feedback")
        forbidden = (
            "feedback_episodes", "successful_feedback_episodes",
            "failed_feedback_episodes", "queries", "self_label_episodes",
            "feedback_observed_episodes", "query_gate_evaluations",
            "self_prediction_evaluations", "queried_feedback_successes",
            "self_feedback_successes",
        )
        if any(adapter.get(key) not in (None, 0) for key in forbidden):
            raise UserError("unsupervised method consumed episode feedback")
    return adapter


def _validate_formal_manifest(metadata, aggregate, diagnostics):
    path = Path(metadata["formal_manifest"]).resolve()
    try:
        path.relative_to(FORMAL_ROOT.resolve())
    except ValueError as error:
        raise UserError("formal manifest is outside canonical root") from error
    if not path.is_file():
        raise UserError("missing formal manifest: {}".format(path))
    manifest = _read_json(path, "formal manifest")
    expected_run_id = "{}-{}-val_unseen-v1.3-unified".format(
        metadata["run_tag"], metadata["setting"]
    )
    expected = {
        "run_id": expected_run_id,
        "task": "vln",
        "benchmark": "r2r_ce_v1_3_unified_etpnav_bevbert",
        "model": metadata["model"],
        "method": metadata["method"],
        "run_tag": metadata["run_tag"],
        "source_setting": "{}:val_unseen:v1.3-unified:{}".format(
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
    if manifest.get("config") != metadata["config_path"]:
        raise UserError("formal manifest config path mismatch")
    translated_method, tokens = config_cli.translate(
        metadata["setting"], metadata["config_path"], str(diagnostics)
    )
    overrides = manifest.get("config_overrides")
    if translated_method != metadata["method"] or not isinstance(
        overrides, list
    ) or overrides[-len(tokens):] != tokens:
        raise UserError("formal manifest does not authenticate frozen parameters")

    def override_value(name):
        positions = [index for index, token in enumerate(overrides) if token == name]
        if len(positions) != 1 or positions[0] + 1 >= len(overrides):
            raise UserError("formal manifest has invalid {}".format(name))
        return str(overrides[positions[0] + 1])

    if (
        override_value("EVAL.SPLIT") != "val_unseen"
        or override_value("TASK_CONFIG.SEED") != "0"
        or override_value("EVAL.EPISODE_COUNT") != "-1"
        or Path(override_value("EVAL.EPISODE_ORDER_MANIFEST")).resolve()
        != Path(metadata["episode_order_manifest"]).resolve().parent
    ):
        raise UserError("formal manifest split/seed/order command mismatch")
    if manifest.get("checkpoint", {}).get("sha256") != metadata[
        "expected_checkpoint_sha256"
    ]:
        raise UserError("formal checkpoint mismatch")
    dataset = manifest.get("dataset", {})
    if dataset.get("stream_content_sha256") != metadata["expected_dataset_sha256"] or (
        dataset.get("index_sha256") != metadata["expected_dataset_sha256"]
    ) or (
        dataset.get("stream_order_sha256") != metadata["expected_episode_order_sha256"]
    ):
        raise UserError("formal dataset/order mismatch")
    if manifest.get("pinned_manifests", {}).get("episode_order", {}).get(
        "sha256"
    ) != metadata["episode_order_manifest_sha256"]:
        raise UserError("formal episode-order manifest mismatch")
    identity = manifest.get("immutable_identity_sha256")
    if not _valid_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise UserError("formal immutable identity mismatch")
    authenticated = {}
    result_root = Path(metadata["result_root"]).resolve()
    for artifact in manifest.get("result_artifacts") or []:
        artifact_path = Path(str(artifact.get("path", ""))).resolve()
        try:
            artifact_path.relative_to(result_root)
        except ValueError as error:
            raise UserError("formal artifact escapes result root") from error
        if not artifact_path.is_file() or artifact.get("size") != artifact_path.stat().st_size or (
            _sha256(artifact_path) != artifact.get("sha256")
        ):
            raise UserError("formal artifact is missing or digest-mismatched")
        authenticated[artifact_path] = artifact
    if Path(aggregate).resolve() not in authenticated or Path(diagnostics).resolve() not in authenticated:
        raise UserError("required aggregate/diagnostics are unauthenticated")
    return manifest


def validate_attempt(attempt_dir, expected_job=None, expected_batch_id=None,
                     expected_spec_path=None):
    attempt_dir = Path(attempt_dir)
    metadata = _read_json(attempt_dir / "job.json", "job metadata")
    _validate_attempt_binding(
        attempt_dir, metadata, expected_job, expected_batch_id,
        expected_spec_path,
    )
    try:
        exit_code = int((attempt_dir / "exitcode").read_text().strip())
    except (OSError, ValueError) as error:
        raise UserError("invalid worker exit status: {}".format(error))
    if exit_code != 0:
        raise UserError("worker exited with status {}".format(exit_code))
    result_root = Path(metadata["result_root"])
    aggregates = sorted(result_root.glob("metrics/source_val_unseen/stats_ckpt_*.json"))
    aggregates = [path for path in aggregates if "stats_ep_" not in path.name]
    if len(aggregates) != 1:
        raise UserError("expected exactly one R2R-CE aggregate")
    diagnostics_path = result_root / "tta_diagnostics.json"
    if not diagnostics_path.is_file():
        raise UserError("missing TTA diagnostics")
    metrics = _aggregate_metrics(aggregates[0])
    diagnostics = _read_json(diagnostics_path, "TTA diagnostics")
    adapter = _validate_diagnostics(metadata, diagnostics, metrics)
    manifest = _validate_formal_manifest(metadata, aggregates[0], diagnostics_path)
    result = dict(metadata)
    result.update({
        "metrics": metrics,
        "aggregate_artifact": str(aggregates[0]),
        "aggregate_artifact_sha256": _sha256(aggregates[0]),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "adapter_diagnostics": adapter,
        "formal_manifest_sha256": _sha256(metadata["formal_manifest"]),
        "formal_immutable_identity_sha256": manifest["immutable_identity_sha256"],
    })
    _atomic_json(attempt_dir / "metrics.json", result)
    return result


def _write_worker(attempt_dir, metadata):
    exit_path = Path(attempt_dir) / "exitcode"
    authorization = Path(attempt_dir) / "launch_authorized"
    pid_path = Path(attempt_dir) / "pid"
    quoted_authorization = shlex.quote(str(authorization))
    quoted_exit_tmp = shlex.quote(str(exit_path) + ".tmp")
    quoted_exit = shlex.quote(str(exit_path))
    quoted_pid_tmp = shlex.quote(str(pid_path) + ".worker.tmp")
    quoted_pid = shlex.quote(str(pid_path))
    quoted_token = shlex.quote(metadata["process_group_token"])
    script = "\n".join((
        "#!/usr/bin/env bash", "set +e",
        "export {}={}".format(PROCESS_GROUP_ENV, quoted_token),
        "printf '%s\\n' \"$$\" > {}".format(quoted_pid_tmp),
        "mv {} {}".format(quoted_pid_tmp, quoted_pid),
        "for ((gate_wait=0; gate_wait<600; gate_wait++)); do",
        "  [[ -f {} ]] && break".format(quoted_authorization),
        "  sleep 0.1",
        "done",
        "if [[ ! -f {} ]]; then".format(quoted_authorization),
        "  printf '%s\\n' '125' > {}".format(quoted_exit_tmp),
        "  mv {} {}".format(quoted_exit_tmp, quoted_exit),
        "  exit 125",
        "fi",
        shlex.join(metadata["command"]), "status=$?",
        "printf '%s\\n' \"$status\" > {}".format(quoted_exit_tmp),
        "mv {} {}".format(quoted_exit_tmp, quoted_exit),
        "exit \"$status\"", "",
    ))
    path = Path(attempt_dir) / "worker.sh"
    _atomic_text(path, script)
    path.chmod(0o755)
    return path


def _launch(attempt_dir, metadata):
    worker = _write_worker(attempt_dir, metadata)
    log = (Path(attempt_dir) / "launcher.log").open("ab", buffering=0)
    process = subprocess.Popen(
        ["bash", str(worker)], cwd=str(REPO_ROOT), stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    _atomic_text(Path(attempt_dir) / "pid", "{}\n".format(process.pid))
    identity = process_identity(process.pid)
    if identity is None:
        os.killpg(process.pid, signal.SIGTERM)
        log.close()
        raise UserError("cannot bind worker process identity")
    if not _identity_is_group_member(identity, process.pid):
        os.killpg(process.pid, signal.SIGTERM)
        log.close()
        raise UserError("worker did not become its own process-group leader")
    _atomic_json(Path(attempt_dir) / "process_identity.json", identity)
    return process, log, metadata


def _authorize_launch(attempt_dir):
    _atomic_text(Path(attempt_dir) / "launch_authorized", "authorized\n")


def _terminate_attempt_process_group(attempt_dir, timeout_seconds=10.0):
    """Terminate only a process group authenticated as belonging to an attempt."""
    attempt_dir = Path(attempt_dir)
    if not _live_worker_identities(attempt_dir):
        return True
    pgid = _attempt_pgid(attempt_dir)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    deadline = time.monotonic() + timeout_seconds
    while _pid_alive(attempt_dir) and time.monotonic() < deadline:
        time.sleep(0.1)
    if not _pid_alive(attempt_dir):
        return True
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    deadline = time.monotonic() + timeout_seconds
    while _pid_alive(attempt_dir) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not _pid_alive(attempt_dir)


def _assert_batch_unchanged(batch_root, spec_path, spec):
    batch = _read_json(Path(batch_root) / "BATCH.json", "batch manifest")
    if batch.get("schema") != BATCH_SCHEMA or batch.get("git_commit") != _git_commit():
        raise UserError("batch Git commit changed")
    if batch.get("spec_sha256") != _sha256(spec_path) or batch.get(
        "registry_sha256"
    ) != spec["registry_dependency"]["sha256"]:
        raise UserError("batch spec/registry changed")
    if _tracked_worktree_dirty():
        raise UserError("tracked worktree changed during formal campaign")
    untracked = _untracked_execution_files()
    if untracked:
        raise UserError(
            "untracked execution files appeared during campaign: {}".format(
                ", ".join(untracked)
            )
        )


def _prepare_batch(spec_path, spec, batch_id, batch_root, gpu, resume):
    path = Path(batch_root) / "BATCH.json"
    document = {
        "schema": BATCH_SCHEMA,
        "batch_id": batch_id,
        "git_commit": _git_commit(),
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "registry_sha256": spec["registry_dependency"]["sha256"],
        "source_ledger_sha256": spec["source_control"]["sha256"],
        "registry": spec["registry_dependency"],
        "gpu": gpu,
        "source_execution_jobs": 0,
        "tta_jobs": 10,
        "max_workers": 3,
    }
    if path.is_file():
        if not resume:
            raise UserError("batch exists; use --resume")
        if _canonical(_read_json(path, "batch manifest")) != _canonical(document):
            raise UserError("persisted batch manifest changed")
    else:
        if resume:
            raise UserError("cannot resume a missing batch")
        if Path(batch_root).exists() and any(Path(batch_root).iterdir()):
            raise UserError("nonempty batch directory has no BATCH.json")
        _atomic_json(path, document)
    return path


def _resource_ok(spec, gpu):
    limits = spec["execution"]["resource_limits"]
    gpu_memory, _ = resources.gpu_stats(gpu)
    memory = resources.cgroup_memory_gib()
    okay = (
        gpu_memory <= limits["max_gpu_memory_mib_before_launch"]
        and gpu_memory + limits["estimated_job_gpu_memory_mib"]
        <= limits["max_aggregate_gpu_memory_mib"]
        and memory <= limits["max_cgroup_memory_gib_before_launch"]
        and memory + limits["estimated_job_memory_gib"]
        <= limits["max_aggregate_cgroup_memory_gib"]
    )
    return okay, gpu_memory, memory


def _reserved_resource_ok(spec, ledger, gpu):
    limits = spec["execution"]["resource_limits"]
    gpu_memory, _ = resources.gpu_stats(gpu)
    memory = resources.cgroup_memory_gib()
    snapshot = ledger.snapshot(gpu_memory, memory)
    effective_gpu = snapshot["effective_gpu_memory_mib"]
    effective_memory = snapshot["effective_cgroup_memory_gib"]
    okay = (
        effective_gpu <= limits["max_gpu_memory_mib_before_launch"]
        and effective_gpu + limits["estimated_job_gpu_memory_mib"]
        <= limits["max_aggregate_gpu_memory_mib"]
        and effective_memory <= limits["max_cgroup_memory_gib_before_launch"]
        and effective_memory + limits["estimated_job_memory_gib"]
        <= limits["max_aggregate_cgroup_memory_gib"]
    )
    return okay, gpu_memory, memory, snapshot


def _reservation_token(batch_id, run_tag):
    return "r2r-ce-val-unseen:{}:{}".format(batch_id, run_tag)


def _release_reservation(batch_id, job, attempt):
    release_shared_gpu_reservation(
        job["gpu"], _reservation_token(
            batch_id, _attempt_tag(job["base_run_tag"], attempt)
        )
    )


def _claim_running_reservation(spec, batch_id, job, attempt_dir, metadata):
    identities = _live_worker_identities(attempt_dir)
    if not identities:
        raise UserError("cannot reserve resources for a non-live worker")
    token = _reservation_token(batch_id, metadata["run_tag"])
    limits = spec["execution"]["resource_limits"]
    with shared_gpu_launch_guard(job["gpu"]) as ledger:
        record = ledger.document["reservations"].get(token)
        expected_metadata = {
            "role": "r2r_ce_val_unseen",
            "batch_id": batch_id,
            "run_tag": metadata["run_tag"],
        }
        if record is None:
            gpu_memory, _ = resources.gpu_stats(job["gpu"])
            memory = resources.cgroup_memory_gib()
            ledger.reserve(
                token,
                gpu_memory_mib=limits["estimated_job_gpu_memory_mib"],
                cgroup_memory_gib=limits["estimated_job_memory_gib"],
                observed_gpu_memory_mib=gpu_memory,
                observed_cgroup_memory_gib=memory,
                owner=process_identity(),
                metadata=expected_metadata,
            )
            record = ledger.document["reservations"][token]
        elif (
            record.get("gpu_memory_mib")
            != limits["estimated_job_gpu_memory_mib"]
            or not math.isclose(
                float(record.get("cgroup_memory_gib", math.nan)),
                float(limits["estimated_job_memory_gib"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or record.get("metadata") != expected_metadata
        ):
            raise UserError("live worker reservation binding mismatch")
        owner_keys = ("pid", "start_token", "cmdline_sha256")
        for identity in identities:
            if not any(
                all(owner.get(key) == identity.get(key) for key in owner_keys)
                for owner in record.get("owners", [])
            ):
                ledger.add_owner(token, identity)
    _authorize_launch(attempt_dir)


def _record_validation_error(attempt_dir, error):
    path = Path(attempt_dir) / "validation_error.json"
    if path.is_file():
        _validate_validation_error(attempt_dir)
        return
    match = re.fullmatch(r"attempt-([0-9]{2})", Path(attempt_dir).name)
    if match is None:
        raise UserError("cannot bind validation error to attempt directory")
    job_path = Path(attempt_dir) / "job.json"
    job = _read_json(job_path, "failed attempt job") if job_path.is_file() else None
    document = {
        "schema": "navtta.vln_r2r_ce_validation_error.v1",
        "attempt": int(match.group(1)),
        "run_tag": job.get("run_tag") if job is not None else None,
        "job_sha256": _sha256(job_path) if job is not None else None,
        "error_type": type(error).__name__,
        "error": str(error),
        "recorded_at_unix": time.time(),
    }
    _atomic_json(path, document)
    _validate_validation_error(attempt_dir)


def run_model_phase(spec, spec_path, batch_id, batch_root, jobs,
                    resume=False, retry_failed=False):
    del resume
    max_attempts = int(spec["execution"]["max_attempts_per_job"])
    if not jobs or len({job["model"] for job in jobs}) != 1:
        raise UserError("one model phase must contain exactly one model")
    limits = spec["execution"]["resource_limits"]
    cap = int(spec["execution"]["max_workers"])
    active_handles = {}
    validated_complete = set()
    last_launch = 0.0
    resource_blocked_since = None
    try:
        while True:
            _assert_batch_unchanged(batch_root, spec_path, spec)
            completed = 0
            running = []
            launchable = []
            for job in jobs:
                latest = _latest_attempt(batch_root, job)
                if latest is None:
                    launchable.append((job, 0))
                    continue
                attempt, attempt_dir = latest
                _recover_worker_identity(attempt_dir)
                state = _state(attempt_dir)
                if state in ("completed", "finished", "failed", "invalid", "orphaned"):
                    launched_handle = active_handles.pop(job["base_run_tag"], None)
                    if launched_handle is not None:
                        launched_handle[0].poll()
                        launched_handle[1].close()
                if state in ("completed", "finished"):
                    if job["base_run_tag"] not in validated_complete:
                        try:
                            result = validate_attempt(
                                attempt_dir, expected_job=job,
                                expected_batch_id=batch_id,
                                expected_spec_path=spec_path,
                            )
                        except Exception as error:
                            _release_reservation(batch_id, job, attempt)
                            _record_validation_error(attempt_dir, error)
                            if retry_failed and attempt + 1 < max_attempts:
                                launchable.append((job, attempt + 1))
                                continue
                            raise UserError(
                                "{} validation failed: {}".format(
                                    job["base_run_tag"], error
                                )
                            )
                        validated_complete.add(job["base_run_tag"])
                        print("completed {} SR/SPL={:.4f}/{:.4f}".format(
                            result["run_tag"], result["metrics"]["SR"],
                            result["metrics"]["SPL"],
                        ))
                    _release_reservation(batch_id, job, attempt)
                    completed += 1
                elif state == "running":
                    metadata = _read_json(attempt_dir / "job.json", "job metadata")
                    _validate_attempt_binding(
                        attempt_dir, metadata, expected_job=job,
                        expected_batch_id=batch_id,
                        expected_spec_path=spec_path,
                    )
                    _claim_running_reservation(
                        spec, batch_id, job, attempt_dir, metadata
                    )
                    running.append(job)
                elif state in ("failed", "invalid", "orphaned"):
                    if state != "invalid":
                        _record_validation_error(
                            attempt_dir,
                            UserError("attempt terminal state is {}".format(state)),
                        )
                    else:
                        _validate_validation_error(attempt_dir)
                    _release_reservation(batch_id, job, attempt)
                    if retry_failed and attempt + 1 < max_attempts:
                        launchable.append((job, attempt + 1))
                    else:
                        raise UserError(
                            "{} is {}; use --resume --retry-failed"
                            .format(job["base_run_tag"], state)
                        )
                else:
                    launchable.append((job, attempt))
            if completed == len(jobs):
                return
            if len(running) > cap:
                raise UserError("persisted phase exceeds the three-worker cap")
            launched = False
            for job, attempt in launchable:
                if len(running) >= cap:
                    break
                if time.monotonic() - last_launch < float(
                    spec["execution"]["launch_stagger_seconds"]
                ):
                    break
                with shared_gpu_launch_guard(job["gpu"]) as ledger:
                    okay, gpu_memory, memory, _ = _reserved_resource_ok(
                        spec, ledger, job["gpu"]
                    )
                    if not okay:
                        resource_blocked_since = resource_blocked_since or time.monotonic()
                        continue
                    resource_blocked_since = None
                    _assert_batch_unchanged(batch_root, spec_path, spec)
                    attempt_dir, metadata = materialize_attempt(
                        spec, spec_path, batch_id, batch_root, job, attempt
                    )
                    token = _reservation_token(batch_id, metadata["run_tag"])
                    ledger.reserve(
                        token,
                        gpu_memory_mib=limits["estimated_job_gpu_memory_mib"],
                        cgroup_memory_gib=limits["estimated_job_memory_gib"],
                        observed_gpu_memory_mib=gpu_memory,
                        observed_cgroup_memory_gib=memory,
                        owner=process_identity(),
                        metadata={
                            "role": "r2r_ce_val_unseen",
                            "batch_id": batch_id,
                            "run_tag": metadata["run_tag"],
                        },
                    )
                    process = None
                    handle = None
                    try:
                        process, handle, metadata = _launch(attempt_dir, metadata)
                        worker_identity = _read_json(
                            attempt_dir / "process_identity.json", "process identity"
                        )
                        ledger.add_owner(token, worker_identity)
                        _authorize_launch(attempt_dir)
                    except Exception:
                        if process is not None and process.poll() is None:
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait(timeout=10)
                            except ProcessLookupError:
                                pass
                        if handle is not None:
                            handle.close()
                        ledger.release(token)
                        raise
                active_handles[job["base_run_tag"]] = (process, handle)
                running.append(job)
                last_launch = time.monotonic()
                launched = True
                print("launched {} pid={}".format(metadata["run_tag"], process.pid))
            for tag, (process, handle) in list(active_handles.items()):
                if process.poll() is not None:
                    handle.close()
                    active_handles.pop(tag)
            if (
                resource_blocked_since is not None
                and not running
                and time.monotonic() - resource_blocked_since
                >= limits["resource_wait_timeout_seconds"]
            ):
                raise UserError("resource thresholds blocked all launches")
            if not launched:
                time.sleep(min(float(spec["execution"].get("poll_seconds", 5)), 2.0))
    except Exception:
        for tag, (process, handle) in active_handles.items():
            owner_job = next(job for job in jobs if job["base_run_tag"] == tag)
            latest = _latest_attempt(batch_root, owner_job)
            if latest is not None:
                _terminate_attempt_process_group(latest[1])
            elif process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            handle.close()
            if latest is not None and not _pid_alive(latest[1]):
                _release_reservation(batch_id, owner_job, latest[0])
        raise


def collect_states(batch_root, jobs, revalidate=False, batch_id=None,
                   spec_path=None):
    counts = {name: 0 for name in (
        "completed", "finished", "failed", "invalid", "running", "orphaned", "pending"
    )}
    results = []
    for job in jobs:
        latest = _latest_attempt(batch_root, job)
        if latest is None:
            counts["pending"] += 1
            continue
        state = _state(latest[1])
        if revalidate and state in ("completed", "finished"):
            try:
                results.append(validate_attempt(
                    latest[1], expected_job=job, expected_batch_id=batch_id,
                    expected_spec_path=spec_path,
                ))
                state = "completed"
            except Exception as error:
                _record_validation_error(latest[1], error)
                state = "invalid"
        elif state == "completed":
            results.append(_read_json(latest[1] / "metrics.json", "job metrics"))
        counts[state] += 1
    return counts, results


def write_summary(batch_root, spec, jobs, revalidate=False, batch_id=None,
                  spec_path=None):
    counts, results = collect_states(
        batch_root, jobs, revalidate=revalidate, batch_id=batch_id,
        spec_path=spec_path,
    )
    by_pair = {(item["setting"], item["method"]): item for item in results}
    document = {
        "schema": SUMMARY_SCHEMA,
        "batch_id": Path(batch_root).name,
        "split": "val_unseen",
        "selection_on_val_unseen": False,
        "source_execution_jobs": 0,
        "planned_tta_jobs": 10,
        "states": counts,
        "complete": len(results) == 10,
        "results": [by_pair[pair] for pair in (
            (setting, method) for setting in SETTINGS for method in METHODS
        ) if pair in by_pair],
    }
    _atomic_json(Path(batch_root) / "SUMMARY.json", document)
    return document


def print_plan(spec, jobs, batch_id, gpu, print_commands=False):
    print(
        "batch_id={} gpu={} tta_jobs=10 source_jobs=0 split=val_unseen "
        "episodes=1839 order_seed=0 max_workers=3".format(batch_id, gpu)
    )
    for phase in model_phases(jobs):
        print("model {}: methods={} max_workers=3 strict_barrier_after=true".format(
            phase[0]["model"], ",".join(job["method"] for job in phase)
        ))
        if print_commands:
            for job in phase:
                print(shlex.join([
                    str(RUNNER), job["setting"], "val_unseen", str(gpu),
                    "--run-tag", job["base_run_tag"],
                    "--tta-config", "<ATTEMPT_DIR>/parameters.json",
                ]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--batch-id", default=CANONICAL_BATCH_ID
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0 or not re.fullmatch(r"[A-Za-z0-9._-]+", args.batch_id):
        parser.error("invalid GPU or batch id")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    if args.batch_id != spec["canonical_batch_id"]:
        raise UserError("batch ID must equal the frozen canonical batch ID")
    jobs = expand_jobs(spec, args.batch_id, args.gpu)
    batch_root = LOG_ROOT / args.batch_id
    if args.plan_only:
        print_plan(spec, jobs, args.batch_id, args.gpu, args.print_commands)
        return 0
    if args.status:
        counts, results = collect_states(
            batch_root, jobs, revalidate=False, batch_id=args.batch_id,
            spec_path=spec_path,
        )
        print(json.dumps({
            "batch_id": args.batch_id,
            "states": counts,
            "completed_results": len(results),
            "source_execution_jobs": 0,
        }, indent=2, sort_keys=True))
        return 0
    if not args.confirm_reviewed:
        raise UserError("formal execution requires --confirm-reviewed")
    validate_source_ledger(spec, require_artifacts=True)
    if _tracked_worktree_dirty():
        raise UserError("tracked worktree must be clean for formal execution")
    _, registry = load_registry(spec)
    _require_tracked_inputs(spec_path, spec, registry)
    lock_path = batch_root.parent / "{}.scheduler.lock".format(
        spec["experiment_id"]
    )
    with campaign_lifetime_lock(
        lock_path, role="r2r_ce_val_unseen", batch_id=args.batch_id
    ):
        _prepare_batch(
            spec_path, spec, args.batch_id, batch_root, args.gpu, args.resume
        )
        for phase in model_phases(jobs):
            run_model_phase(
                spec, spec_path, args.batch_id, batch_root, phase,
                resume=args.resume, retry_failed=args.retry_failed,
            )
            write_summary(
                batch_root, spec, jobs, revalidate=True,
                batch_id=args.batch_id, spec_path=spec_path,
            )
        summary = write_summary(
            batch_root, spec, jobs, revalidate=True,
            batch_id=args.batch_id, spec_path=spec_path,
        )
    if not summary["complete"]:
        raise UserError("frozen val_unseen campaign ended before 10/10 jobs")
    print("all 10 R2R-CE val_unseen frozen TTA jobs completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        UserError, registry_builder.RegistryError, JointLaunchError,
        ReservationLedgerError,
    ) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
