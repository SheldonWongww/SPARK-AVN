#!/usr/bin/env python3
"""Measure safe single-phase concurrency for an R2R refinement campaign.

This helper consumes a campaign already materialized by
``run_r2r_local_refinement.py --plan-only``.  It never executes a persisted
formal job in place.  Instead, it clones a prefix of one phase into an
isolated calibration namespace, rewrites every run/config/result path, and
then ramps up one worker at a time while sampling GPU and cgroup memory.  The
next worker is blocked until the current level rises above the previous
level's steady VRAM and remains stable for a measured window; the fixed
stagger is only a minimum delay, never loading evidence.

The two memory lines are deliberately fixed here: stop adding workers once
29,000 MiB is observed and terminate only this helper's process groups once
30,000 MiB is observed.  A calibration cannot contain jobs from more than one
phase and a campaign-wide lock prevents two calibrations from overlapping.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "vln/results/logs/r2r/hparam_search"
TUNING_ROOT = REPO_ROOT / "vln/results/tuning/r2r/hparam_search"

PLAN_SCHEMA = "navtta.vln_r2r_local_refinement_campaign_plan.v1"
PHASE_SCHEMA = "navtta.vln_r2r_local_refinement_phase.v1"
CALIBRATION_SCHEMA = "navtta.vln_r2r_local_refinement_calibration.v1"
CALIBRATION_PLAN_SCHEMA = (
    "navtta.vln_r2r_local_refinement_calibration_plan.v1"
)

PLANNED_MEMORY_MIB = 29_000
EMERGENCY_MEMORY_MIB = 30_000
DEFAULT_STAGGER_SECONDS = 15.0
DEFAULT_SAMPLE_INTERVAL_SECONDS = 3.0
DEFAULT_TERMINATE_GRACE_SECONDS = 10.0
DEFAULT_STEADY_SECONDS = 20.0
DEFAULT_LOAD_TIMEOUT_SECONDS = 300.0
DEFAULT_MIN_LOADED_MEMORY_MIB = 512
DEFAULT_STEADY_RELATIVE_TOLERANCE = 0.02
DEFAULT_STEADY_SAMPLES = 3
FULL_VAL_EPISODES = 1021


class UserError(RuntimeError):
    pass


def _timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"invalid {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise UserError(f"{label} must be a JSON object: {path}")
    return value


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "output", "")
        raise UserError(f"git {' '.join(args)} failed: {output or error}") from error


def _assert_plan_revision(plan):
    planned = plan.get("git_commit")
    if not isinstance(planned, str) or re.fullmatch(r"[0-9a-f]{40}", planned) is None:
        raise UserError("campaign PLAN.json has no valid pinned git_commit")
    current = _git("rev-parse", "--verify", "HEAD")
    if current != planned:
        raise UserError(
            f"campaign commit mismatch: PLAN={planned}, current={current}"
        )
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise UserError("tracked worktree must be clean before calibration")


def _safe_component(label, value, max_length=160):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > max_length
        or Path(value).name != value
        or value in (".", "..")
        or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
    ):
        raise UserError(f"unsafe {label} path component: {value!r}")
    return value


def _is_relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _process_alive(pid):
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _state_live_pid(path):
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if state.get("status") == "finished":
        return None
    for key in ("runner_pid", "worker_pid", "pid"):
        pid = state.get(key)
        if _process_alive(pid):
            return int(pid)
    return None


def _load_phase(campaign_root, phase_id):
    """Return a validated PLAN phase and all of its persisted jobs."""
    campaign_root = Path(campaign_root).resolve()
    plan = _read_json(campaign_root / "PLAN.json", "campaign PLAN.json")
    if plan.get("schema") != PLAN_SCHEMA:
        raise UserError("unsupported campaign PLAN.json schema")
    batch_id = _safe_component("batch_id", plan.get("batch_id"))
    matches = [
        phase for phase in plan.get("phases", [])
        if isinstance(phase, dict) and phase.get("phase_id") == phase_id
    ]
    if len(matches) != 1:
        raise UserError(
            f"phase {phase_id!r} occurs {len(matches)} times in campaign PLAN.json"
        )
    phase = matches[0]
    for key in ("phase_id", "setting", "method", "model"):
        _safe_component(key, phase.get(key))
    phase_root = campaign_root / "phases" / phase_id
    phase_manifest = _read_json(phase_root / "PHASE.json", "phase manifest")
    if phase_manifest.get("schema") != PHASE_SCHEMA:
        raise UserError(f"unsupported PHASE.json schema for {phase_id}")
    for key in ("phase_id", "setting", "method", "model"):
        if phase_manifest.get(key) != phase.get(key):
            raise UserError(f"PLAN/PHASE mismatch for {phase_id}.{key}")
    for key in ("batch_id", "git_commit", "spec_sha256", "gpu"):
        if phase_manifest.get(key) != plan.get(key):
            raise UserError(f"PLAN/PHASE mismatch for {phase_id}.{key}")

    paths = sorted((phase_root / "jobs").glob("*/job.json"))
    expected_count = phase.get("job_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
        or len(paths) != expected_count
    ):
        raise UserError(
            f"phase {phase_id} job count mismatch: "
            f"PLAN={expected_count!r}, persisted={len(paths)}"
        )

    jobs = []
    for path in paths:
        job = _read_json(path, "phase job")
        if job.get("phase_id") != phase_id:
            raise UserError(f"cross-phase job rejected: {path}")
        for key in ("setting", "method", "model"):
            job_key = "search_method" if key == "method" else key
            if job.get(job_key) != phase.get(key):
                raise UserError(f"cross-phase {key} rejected: {path}")
        declared_job_dir = Path(str(job.get("job_dir", ""))).resolve()
        if declared_job_dir != path.parent.resolve():
            raise UserError(f"job_dir does not match persisted job path: {path}")
        if not _is_relative_to(declared_job_dir, phase_root / "jobs"):
            raise UserError(f"formal job escapes its phase directory: {path}")
        config_path = Path(str(job.get("config_path", ""))).resolve()
        if not _is_relative_to(config_path, declared_job_dir):
            raise UserError(f"formal config escapes its job directory: {path}")
        if not config_path.is_file():
            raise UserError(f"missing formal job config: {config_path}")
        command = job.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(token, str) and token for token in command
        ):
            raise UserError(f"invalid formal job command: {path}")
        if (
            len(command) < 4
            or command[1] != phase["setting"]
            or command[2] != "val_seen"
            or command[3] != str(plan.get("gpu"))
        ):
            raise UserError(f"formal job command setting/split/GPU mismatch: {path}")
        jobs.append(job)
    jobs.sort(key=lambda job: int(job["ordinal"]))
    return plan, phase, phase_manifest, jobs


def _select_distinct_jobs(jobs, target_workers):
    if (
        isinstance(target_workers, bool)
        or not isinstance(target_workers, int)
        or target_workers < 1
    ):
        raise UserError("target_workers must be a positive integer")
    selected = []
    seen = set()
    for job in jobs:
        try:
            key = _canonical({
                "config_method": job.get("config_method"),
                "parameters": job.get("parameters"),
            })
        except (TypeError, ValueError) as error:
            raise UserError(f"invalid candidate parameters: {error}") from error
        if key in seen:
            continue
        seen.add(key)
        selected.append(job)
        if len(selected) == target_workers:
            break
    if len(selected) != target_workers:
        raise UserError(
            f"phase has only {len(selected)} distinct candidates; "
            f"cannot calibrate {target_workers} workers"
        )
    return selected


def _replace_option(command, option, replacement):
    indices = [index for index, token in enumerate(command) if token == option]
    if len(indices) != 1:
        raise UserError(
            f"source command must contain {option} exactly once, got {len(indices)}"
        )
    index = indices[0]
    if index + 1 >= len(command):
        raise UserError(f"source command has no value for {option}")
    output = list(command)
    output[index + 1] = str(replacement)
    return output


def _calibration_run_tag(batch_id, calibration_id, phase, index, source_job):
    digest = hashlib.sha256(
        _canonical({
            "source_run_tag": source_job.get("run_tag"),
            "parameters": source_job.get("parameters"),
        }).encode("utf-8")
    ).hexdigest()[:10]
    # Source-tag locks use the run tag as a filename, so keep ample room under
    # the usual 255-byte component limit even for verbose campaign names.
    prefix = batch_id[:48]
    cal = calibration_id[:32]
    method = str(phase["method"])[:24]
    value = f"{prefix}-cal-{cal}-{method}-w{index:02d}-{digest}"
    return _safe_component("calibration run_tag", value, max_length=180)


def _clone_jobs(
    campaign_root,
    plan,
    phase,
    selected_jobs,
    calibration_id,
    episode_limit=None,
    tuning_root=None,
):
    """Clone selected formal jobs without writing to any formal job path."""
    campaign_root = Path(campaign_root).resolve()
    calibration_root = (
        campaign_root / "calibration" / phase["phase_id"] / calibration_id
    ).resolve()
    allowed_root = (campaign_root / "calibration").resolve()
    if not _is_relative_to(calibration_root, allowed_root):
        raise UserError("calibration output must stay below campaign/calibration")
    if calibration_root.exists() and any(calibration_root.iterdir()):
        raise UserError(f"calibration already exists: {calibration_root}")

    if episode_limit is not None:
        full_count = int(plan.get("episode_count", FULL_VAL_EPISODES))
        if episode_limit < 1 or episode_limit > full_count:
            raise UserError(
                f"episode_limit must be in [1, {full_count}], got {episode_limit}"
            )
    result_parent = (
        Path(tuning_root if tuning_root is not None else TUNING_ROOT).resolve()
        / plan["batch_id"]
        / "_calibration"
        / phase["phase_id"]
        / calibration_id
    )
    formal_phase_root = campaign_root / "phases" / phase["phase_id"]
    clones = []
    for index, source in enumerate(selected_jobs, start=1):
        run_tag = _calibration_run_tag(
            plan["batch_id"], calibration_id, phase, index, source
        )
        job_dir = calibration_root / "jobs" / run_tag
        config_path = job_dir / "parameters.json"
        result_root = result_parent / run_tag / "val_seen"
        for isolated, label in (
            (job_dir, "job_dir"),
            (config_path, "config_path"),
            (result_root, "result_root"),
        ):
            if _is_relative_to(isolated, formal_phase_root):
                raise UserError(f"calibration {label} overlaps formal phase storage")

        command = list(source["command"])
        command = _replace_option(command, "--run-tag", run_tag)
        command = _replace_option(command, "--tta-config", config_path)
        command = _replace_option(command, "--result-root", result_root)
        if "--episode-limit" in command:
            raise UserError("formal local-refinement command unexpectedly has a prefix")
        if episode_limit is not None:
            command.extend(["--episode-limit", str(episode_limit)])

        source_config = _read_json(source["config_path"], "formal job config")
        config = dict(source_config)
        config["batch_id"] = f"{plan['batch_id']}-calibration"
        config["run_tag"] = run_tag
        config["episodes"] = episode_limit if episode_limit is not None else -1

        clone = dict(source)
        clone.update({
            "batch_id": f"{plan['batch_id']}-calibration",
            "run_tag": run_tag,
            "base_run_tag": run_tag,
            "episodes": episode_limit if episode_limit is not None else -1,
            "job_dir": str(job_dir),
            "config_path": str(config_path),
            "result_root": str(result_root),
            "retry_result_root_parent": str(result_parent),
            "command": command,
            "calibration_id": calibration_id,
            "calibration_worker_index": index,
            "source_run_tag": source.get("run_tag"),
            "source_job_dir": source.get("job_dir"),
        })
        job_dir.mkdir(parents=True, exist_ok=False)
        _atomic_json(config_path, config)
        _atomic_json(job_dir / "job.json", clone)
        clones.append(clone)

    calibration_plan = {
        "schema": CALIBRATION_PLAN_SCHEMA,
        "calibration_id": calibration_id,
        "batch_id": plan["batch_id"],
        "source_plan_schema": plan.get("schema"),
        "source_git_commit": plan.get("git_commit"),
        "source_spec_sha256": plan.get("spec_sha256"),
        "phase_id": phase["phase_id"],
        "setting": phase["setting"],
        "model": phase["model"],
        "method": phase["method"],
        "target_workers": len(clones),
        "episode_limit": episode_limit,
        "episode_budget_per_worker": episode_limit or int(
            plan.get("episode_count", FULL_VAL_EPISODES)
        ),
        "planned_memory_mib": PLANNED_MEMORY_MIB,
        "emergency_memory_mib": EMERGENCY_MEMORY_MIB,
        "formal_paths_are_read_only": True,
        "jobs": [{
            "run_tag": job["run_tag"],
            "source_run_tag": job["source_run_tag"],
            "job_dir": job["job_dir"],
            "config_path": job["config_path"],
            "result_root": job["result_root"],
            "command": job["command"],
        } for job in clones],
        "created_at": _timestamp(),
    }
    _atomic_json(calibration_root / "CALIBRATION_PLAN.json", calibration_plan)
    return calibration_root, clones, calibration_plan


def _gpu_stats(gpu):
    try:
        output = subprocess.check_output([
            "nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL).strip().splitlines()
        if len(output) != 1:
            raise ValueError(f"expected one GPU row, got {len(output)}")
        values = output[0].split(",")
        return int(values[0].strip()), int(values[1].strip())
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as error:
        raise UserError(f"cannot read nvidia-smi telemetry for GPU {gpu}: {error}")


def _cgroup_memory_gib():
    for candidate in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            return int(candidate.read_text(encoding="utf-8").strip()) / 1024 ** 3
        except (OSError, ValueError):
            pass
    return None


def _open_job_console(job):
    path = Path(job["job_dir"]) / "console.log"
    return path.open("ab", buffering=0)


def _launch(job):
    console = _open_job_console(job)
    try:
        process = subprocess.Popen(
            job["command"], cwd=str(REPO_ROOT), stdout=console,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        )
    except Exception:
        console.close()
        raise
    state = {
        "status": "running",
        "runner_pid": process.pid,
        "started_at": _timestamp(),
    }
    _atomic_json(Path(job["job_dir"]) / "worker_state.json", state)
    return process, console


def _signal_process_group(process, signum):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _terminate_active(active, grace_seconds):
    for record in active.values():
        _signal_process_group(record["process"], signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while active and time.monotonic() < deadline:
        if all(record["process"].poll() is not None for record in active.values()):
            break
        time.sleep(0.1)
    for record in active.values():
        if record["process"].poll() is None:
            _signal_process_group(record["process"], signal.SIGKILL)
    kill_deadline = time.monotonic() + min(2.0, max(0.2, grace_seconds))
    while active and time.monotonic() < kill_deadline:
        if all(record["process"].poll() is not None for record in active.values()):
            break
        time.sleep(0.05)


def _record_exits(active, records, now):
    """Reap exits and return whether at least one nonzero exit was seen."""
    exited = []
    active_before_reap = len(active)
    failed = False
    for pid, record in list(active.items()):
        code = record["process"].poll()
        if code is None:
            continue
        record["exit_code"] = int(code)
        record["ended_monotonic"] = now
        record["duration_seconds"] = max(
            0.0, now - record["launched_monotonic"]
        )
        record["active_count_at_exit"] = active_before_reap
        budget = int(record["episode_budget"])
        duration = record["duration_seconds"]
        record["episode_throughput_eps"] = (
            budget / duration if code == 0 and duration > 0 else None
        )
        _atomic_text(Path(record["job"]["job_dir"]) / "exitcode", f"{code}\n")
        _atomic_json(Path(record["job"]["job_dir"]) / "worker_state.json", {
            "status": "finished",
            "runner_pid": record["process"].pid,
            "exit_code": int(code),
            "finished_at": _timestamp(),
        })
        record["console"].close()
        exited.append(pid)
        failed = failed or code != 0
    for pid in exited:
        active.pop(pid, None)
    return failed


def _sample_row(
    started_monotonic, phase, launched, active, records, gpu_memory,
    gpu_utilization, cgroup_memory, action="sample",
):
    return {
        "observed_at": _timestamp(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
        "phase_id": phase["phase_id"],
        "setting": phase["setting"],
        "method": phase["method"],
        "launched_count": launched,
        "active_count": len(active),
        "completed_count": sum(
            record.get("exit_code") is not None for record in records
        ),
        "gpu_memory_mib": gpu_memory,
        "gpu_utilization_pct": gpu_utilization,
        "cgroup_memory_gib": (
            round(cgroup_memory, 6) if cgroup_memory is not None else None
        ),
        "action": action,
    }


RESOURCE_FIELDS = (
    "observed_at", "elapsed_seconds", "phase_id", "setting", "method",
    "launched_count", "active_count", "completed_count", "gpu_memory_mib",
    "gpu_utilization_pct", "cgroup_memory_gib", "action",
)


def _steady_window(
    loaded_samples, now, steady_seconds, steady_samples, relative_tolerance
):
    """Return a stable recent window, or ``None`` while a level is loading.

    ``loaded_samples`` must already be a contiguous run of samples above this
    level's previous-steady-plus-increment threshold.  Use the shortest recent
    suffix that spans ``steady_seconds`` so early cold-start growth cannot keep
    an otherwise settled worker permanently unstable.
    """
    eligible = [
        index for index, sample in enumerate(loaded_samples)
        if now - sample["monotonic"] >= steady_seconds
    ]
    if not eligible:
        return None
    window = loaded_samples[max(eligible):]
    if len(window) < steady_samples:
        return None
    span = now - window[0]["monotonic"]
    if span < steady_seconds:
        return None
    memories = [int(sample["gpu_memory_mib"]) for sample in window]
    steady = float(statistics.median(memories))
    relative_fluctuation = (max(memories) - min(memories)) / max(1.0, steady)
    if relative_fluctuation > relative_tolerance:
        return None
    return {
        "steady_gpu_memory_mib": round(steady, 3),
        "steady_window_seconds": round(span, 6),
        "steady_window_sample_count": len(window),
        "steady_relative_fluctuation": round(relative_fluctuation, 8),
    }


def _level_summaries(samples, records, gate_levels):
    output = []
    gate_by_count = {int(level["active_count"]): level for level in gate_levels}
    counts = sorted(
        {int(row["active_count"]) for row in samples}.union(gate_by_count)
    )
    for count in counts:
        rows = [row for row in samples if int(row["active_count"]) == count]
        memories = [int(row["gpu_memory_mib"]) for row in rows]
        utils = [int(row["gpu_utilization_pct"]) for row in rows]
        cgroups = [
            float(row["cgroup_memory_gib"])
            for row in rows if row.get("cgroup_memory_gib") is not None
        ]
        tail = memories[-min(3, len(memories)):] if memories else []
        gate = gate_by_count.get(count)
        exited = [
            record for record in records
            if record.get("active_count_at_exit") == count
        ]
        successful_throughputs = [
            record["episode_throughput_eps"] for record in exited
            if record.get("episode_throughput_eps") is not None
        ]
        output.append({
            "active_count": count,
            "sample_count": len(rows),
            "peak_gpu_memory_mib": max(memories) if memories else None,
            "tail_median_gpu_memory_mib": (
                round(float(statistics.median(tail)), 3) if tail else None
            ),
            "mean_gpu_utilization_pct": (
                round(sum(utils) / len(utils), 3) if utils else None
            ),
            "peak_cgroup_memory_gib": max(cgroups) if cgroups else None,
            "steady_confirmed": bool(
                gate is not None and gate.get("steady_confirmed") is True
            ),
            "level_status": gate.get("status") if gate else "not_a_ramp_level",
            "previous_steady_gpu_memory_mib": (
                gate.get("previous_steady_gpu_memory_mib") if gate else None
            ),
            "required_loaded_gpu_memory_mib": (
                gate.get("required_loaded_gpu_memory_mib") if gate else None
            ),
            "steady_gpu_memory_mib": (
                gate.get("steady_gpu_memory_mib") if gate else None
            ),
            "loaded_memory_increment_mib": (
                gate.get("loaded_memory_increment_mib") if gate else None
            ),
            "first_loaded_elapsed_seconds": (
                gate.get("first_loaded_elapsed_seconds") if gate else None
            ),
            "load_wait_seconds": gate.get("load_wait_seconds") if gate else None,
            "steady_window_seconds": (
                gate.get("steady_window_seconds") if gate else None
            ),
            "steady_window_sample_count": (
                gate.get("steady_window_sample_count") if gate else None
            ),
            "steady_relative_fluctuation": (
                gate.get("steady_relative_fluctuation") if gate else None
            ),
            "exit_codes_observed": [record["exit_code"] for record in exited],
            "completed_job_episode_throughput_eps": (
                round(sum(successful_throughputs), 6)
                if successful_throughputs else None
            ),
        })
    return output


def _recommended_cap(levels, stop_reason):
    by_count = {level["active_count"]: level for level in levels}
    safe = 0
    for count in range(1, max(by_count, default=0) + 1):
        level = by_count.get(count)
        if level is None or level.get("steady_confirmed") is not True:
            break
        peak = level.get("peak_gpu_memory_mib")
        steady = level.get("steady_gpu_memory_mib")
        if peak is None or steady is None:
            break
        if peak >= PLANNED_MEMORY_MIB or steady >= PLANNED_MEMORY_MIB:
            break
        safe = count
    if stop_reason in ("emergency_memory_threshold", "telemetry_failure"):
        # The levels below the event remain useful, but never recommend the
        # level at which monitoring became unsafe or unavailable.
        unsafe_counts = [
            level["active_count"] for level in levels
            if level.get("peak_gpu_memory_mib") is not None
            and level["peak_gpu_memory_mib"] >= PLANNED_MEMORY_MIB
        ]
        if unsafe_counts:
            safe = min(safe, max(0, min(unsafe_counts) - 1))
    return safe


def _build_summary(
    calibration_plan, samples, records, gate_levels, started_monotonic, stop_reason,
    emergency_observed_mib=None, telemetry_error=None,
):
    levels = _level_summaries(samples, records, gate_levels)
    cap = _recommended_cap(levels, stop_reason)
    exits = [record.get("exit_code") for record in records]
    launched = len(records)
    successful = sum(code == 0 for code in exits)
    failed = sum(code not in (None, 0) for code in exits)
    pending = int(calibration_plan["target_workers"]) - launched
    max_active = max(
        (int(row["active_count"]) for row in samples), default=0
    )
    if stop_reason == "emergency_memory_threshold":
        status = "emergency_abort"
    elif stop_reason == "worker_exited_before_steady":
        status = "inconclusive"
    elif failed or stop_reason in (
        "telemetry_failure", "launch_error", "worker_failure", "load_timeout"
    ):
        status = "failed"
    elif stop_reason == "planned_memory_threshold":
        status = "safety_stop"
    elif pending or cap < int(calibration_plan["target_workers"]):
        status = "inconclusive"
    elif launched and successful == launched:
        status = "completed"
    else:
        status = "inconclusive"
    job_records = []
    for record in records:
        job_records.append({
            "worker_index": record["job"]["calibration_worker_index"],
            "run_tag": record["job"]["run_tag"],
            "source_run_tag": record["job"]["source_run_tag"],
            "pid": record["process"].pid,
            "launched_elapsed_seconds": round(
                record["launched_monotonic"] - started_monotonic, 6
            ),
            "duration_seconds": (
                round(record["duration_seconds"], 6)
                if record.get("duration_seconds") is not None else None
            ),
            "exit_code": record.get("exit_code"),
            "active_count_at_exit": record.get("active_count_at_exit"),
            "episode_budget": record["episode_budget"],
            "episode_throughput_eps": (
                round(record["episode_throughput_eps"], 6)
                if record.get("episode_throughput_eps") is not None else None
            ),
            "job_dir": record["job"]["job_dir"],
            "result_root": record["job"]["result_root"],
        })
    return {
        "schema": CALIBRATION_SCHEMA,
        "calibration_id": calibration_plan["calibration_id"],
        "batch_id": calibration_plan["batch_id"],
        "phase_id": calibration_plan["phase_id"],
        "setting": calibration_plan["setting"],
        "model": calibration_plan["model"],
        "method": calibration_plan["method"],
        "status": status,
        "stop_reason": stop_reason,
        "target_workers": calibration_plan["target_workers"],
        "launched_workers": launched,
        "unlaunched_workers": pending,
        "successful_workers": successful,
        "failed_workers": failed,
        "planned_memory_mib": PLANNED_MEMORY_MIB,
        "emergency_memory_mib": EMERGENCY_MEMORY_MIB,
        "emergency_observed_gpu_memory_mib": emergency_observed_mib,
        "telemetry_error": telemetry_error,
        "baseline_gpu_memory_mib": (
            gate_levels[0]["previous_steady_gpu_memory_mib"]
            if gate_levels else None
        ),
        "recommended_cap": cap,
        "max_observed_active_count": max_active,
        "peak_observed_gpu_memory_mib": max(
            (int(row["gpu_memory_mib"]) for row in samples), default=None
        ),
        "levels": levels,
        "jobs": job_records,
        "resource_csv": "resource.csv",
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
        "finished_at": _timestamp(),
    }


def _run_calibration(
    calibration_root,
    phase,
    jobs,
    calibration_plan,
    gpu,
    stagger_seconds=DEFAULT_STAGGER_SECONDS,
    sample_interval_seconds=DEFAULT_SAMPLE_INTERVAL_SECONDS,
    terminate_grace_seconds=DEFAULT_TERMINATE_GRACE_SECONDS,
    steady_seconds=DEFAULT_STEADY_SECONDS,
    load_timeout_seconds=DEFAULT_LOAD_TIMEOUT_SECONDS,
    min_loaded_memory_mib=DEFAULT_MIN_LOADED_MEMORY_MIB,
    steady_relative_tolerance=DEFAULT_STEADY_RELATIVE_TOLERANCE,
    steady_samples=DEFAULT_STEADY_SAMPLES,
):
    """Run a cloned single-phase adaptive ramp and emit CALIBRATION.json.

    A fixed launch delay is only a lower bound.  Worker ``k + 1`` remains
    blocked until level ``k`` has both loaded relative to level ``k - 1``'s
    confirmed steady VRAM and stayed within the configured stability window.
    """
    started = time.monotonic()
    active = {}
    records = []
    samples = []
    gate_levels = []
    current_level = None
    previous_steady_memory = None
    launched = 0
    stop_reason = None
    emergency_observed = None
    telemetry_error = None
    launch_error = None
    resource_path = Path(calibration_root) / "resource.csv"

    with resource_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESOURCE_FIELDS)
        writer.writeheader()
        stream.flush()
        try:
            while active or (launched < len(jobs) and stop_reason is None):
                now = time.monotonic()
                active_before_reap = len(active)
                worker_failed = _record_exits(active, records, now)
                worker_exited = len(active) < active_before_reap
                if worker_failed and stop_reason is None:
                    stop_reason = "worker_failure"
                    if current_level is not None:
                        current_level["status"] = stop_reason
                        current_level["load_wait_seconds"] = round(
                            now - current_level["launched_monotonic"], 6
                        )
                    _terminate_active(active, terminate_grace_seconds)
                    _record_exits(active, records, time.monotonic())
                    break
                if (
                    worker_exited
                    and current_level is not None
                    and current_level.get("steady_confirmed") is not True
                    and stop_reason is None
                ):
                    stop_reason = "worker_exited_before_steady"
                    current_level["status"] = stop_reason
                    current_level["load_wait_seconds"] = round(
                        now - current_level["launched_monotonic"], 6
                    )
                    _terminate_active(active, terminate_grace_seconds)
                    _record_exits(active, records, time.monotonic())
                    break

                try:
                    gpu_memory, gpu_utilization = _gpu_stats(gpu)
                    cgroup_memory = _cgroup_memory_gib()
                except UserError as error:
                    telemetry_error = str(error)
                    stop_reason = "telemetry_failure"
                    if current_level is not None:
                        current_level["status"] = stop_reason
                        current_level["load_wait_seconds"] = round(
                            now - current_level["launched_monotonic"], 6
                        )
                    _terminate_active(active, terminate_grace_seconds)
                    now = time.monotonic()
                    _record_exits(active, records, now)
                    break

                row = _sample_row(
                    started, phase, launched, active, records, gpu_memory,
                    gpu_utilization, cgroup_memory,
                )
                samples.append(row)
                writer.writerow(row)
                stream.flush()

                if gpu_memory >= EMERGENCY_MEMORY_MIB:
                    emergency_observed = gpu_memory
                    stop_reason = "emergency_memory_threshold"
                    if current_level is not None:
                        current_level["status"] = stop_reason
                        current_level["load_wait_seconds"] = round(
                            now - current_level["launched_monotonic"], 6
                        )
                    _terminate_active(active, terminate_grace_seconds)
                    now = time.monotonic()
                    _record_exits(active, records, now)
                    break
                if gpu_memory >= PLANNED_MEMORY_MIB and stop_reason is None:
                    stop_reason = "planned_memory_threshold"
                    if current_level is not None:
                        current_level["status"] = stop_reason
                        current_level["load_wait_seconds"] = round(
                            now - current_level["launched_monotonic"], 6
                        )

                ready_to_launch = False
                if stop_reason is None and current_level is not None:
                    current_level["observed_sample_count"] += 1
                    if gpu_memory >= current_level[
                        "required_loaded_gpu_memory_mib"
                    ]:
                        if current_level["first_loaded_elapsed_seconds"] is None:
                            current_level["first_loaded_elapsed_seconds"] = round(
                                now - started, 6
                            )
                        current_level["_loaded_samples"].append({
                            "monotonic": now,
                            "gpu_memory_mib": gpu_memory,
                        })
                    else:
                        # A transient rise is not loading evidence.  Require a
                        # new contiguous above-threshold stability window.
                        current_level["_loaded_samples"].clear()

                    stable = _steady_window(
                        current_level["_loaded_samples"], now,
                        steady_seconds, steady_samples,
                        steady_relative_tolerance,
                    )
                    minimum_delay_reached = (
                        now - current_level["launched_monotonic"]
                        >= stagger_seconds
                    )
                    if stable is not None and minimum_delay_reached:
                        current_level.update(stable)
                        current_level["steady_confirmed"] = True
                        current_level["status"] = "steady_confirmed"
                        current_level["load_wait_seconds"] = round(
                            now - current_level["launched_monotonic"], 6
                        )
                        current_level["steady_confirmed_elapsed_seconds"] = round(
                            now - started, 6
                        )
                        current_level["loaded_memory_increment_mib"] = round(
                            current_level["steady_gpu_memory_mib"]
                            - current_level["previous_steady_gpu_memory_mib"],
                            3,
                        )
                        previous_steady_memory = current_level[
                            "steady_gpu_memory_mib"
                        ]
                        current_level = None
                        ready_to_launch = launched < len(jobs)
                    elif (
                        now - current_level["launched_monotonic"]
                        >= load_timeout_seconds
                    ):
                        stop_reason = "load_timeout"
                        current_level["status"] = stop_reason
                        current_level["load_wait_seconds"] = round(
                            now - current_level["launched_monotonic"], 6
                        )
                        _terminate_active(active, terminate_grace_seconds)
                        _record_exits(active, records, time.monotonic())
                        break
                elif (
                    stop_reason is None
                    and launched == 0
                    and previous_steady_memory is None
                ):
                    previous_steady_memory = float(gpu_memory)
                    ready_to_launch = True

                if stop_reason is None and ready_to_launch:
                    job = jobs[launched]
                    try:
                        process, console = _launch(job)
                    except Exception as error:  # preserve partial evidence
                        launch_error = f"{type(error).__name__}: {error}"
                        stop_reason = "launch_error"
                        _terminate_active(active, terminate_grace_seconds)
                        now = time.monotonic()
                        _record_exits(active, records, now)
                        break
                    launched += 1
                    record = {
                        "job": job,
                        "process": process,
                        "console": console,
                        "launched_monotonic": now,
                        "ended_monotonic": None,
                        "duration_seconds": None,
                        "exit_code": None,
                        "active_count_at_exit": None,
                        "episode_budget": int(
                            calibration_plan["episode_budget_per_worker"]
                        ),
                        "episode_throughput_eps": None,
                    }
                    records.append(record)
                    active[process.pid] = record
                    current_level = {
                        "active_count": launched,
                        "status": "waiting_for_load",
                        "steady_confirmed": False,
                        "previous_steady_gpu_memory_mib": round(
                            float(previous_steady_memory), 3
                        ),
                        "required_loaded_gpu_memory_mib": round(
                            float(previous_steady_memory)
                            + int(min_loaded_memory_mib),
                            3,
                        ),
                        "loaded_memory_increment_mib": None,
                        "launched_monotonic": now,
                        "launched_elapsed_seconds": round(now - started, 6),
                        "first_loaded_elapsed_seconds": None,
                        "steady_confirmed_elapsed_seconds": None,
                        "load_wait_seconds": None,
                        "steady_gpu_memory_mib": None,
                        "steady_window_seconds": None,
                        "steady_window_sample_count": None,
                        "steady_relative_fluctuation": None,
                        "observed_sample_count": 0,
                        "_loaded_samples": [],
                    }
                    gate_levels.append(current_level)

                if active or (launched < len(jobs) and stop_reason is None):
                    time.sleep(sample_interval_seconds)
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            if current_level is not None:
                current_level["status"] = stop_reason
                current_level["load_wait_seconds"] = round(
                    time.monotonic() - current_level["launched_monotonic"], 6
                )
            _terminate_active(active, terminate_grace_seconds)
            _record_exits(active, records, time.monotonic())
        finally:
            if active:
                _terminate_active(active, terminate_grace_seconds)
                _record_exits(active, records, time.monotonic())

    if stop_reason is None:
        stop_reason = "target_completed"
    if launch_error is not None:
        telemetry_error = launch_error
    summary = _build_summary(
        calibration_plan, samples, records, gate_levels, started, stop_reason,
        emergency_observed_mib=emergency_observed,
        telemetry_error=telemetry_error,
    )
    _atomic_json(Path(calibration_root) / "CALIBRATION.json", summary)
    return summary


def _assert_no_other_work(campaign_root):
    """Reject live formal jobs or orphaned calibration subprocesses."""
    campaign_root = Path(campaign_root)
    scheduler_lock = campaign_root / ".scheduler.lock"
    if scheduler_lock.is_file():
        try:
            pid = int(_read_json(scheduler_lock, "scheduler lock").get("pid"))
        except (TypeError, ValueError, UserError):
            pid = None
        if pid is not None and _process_alive(pid):
            raise UserError(f"formal refinement scheduler is active: pid {pid}")

    for state_path in (campaign_root / "phases").glob(
        "*/jobs/*/worker_state.json"
    ):
        pid = _state_live_pid(state_path)
        if pid is not None:
            raise UserError(
                f"live formal worker prevents isolated calibration: pid {pid}"
            )
    for state_path in (campaign_root / "calibration").glob(
        "*/*/jobs/*/worker_state.json"
    ):
        pid = _state_live_pid(state_path)
        if pid is not None:
            raise UserError(
                f"live calibration worker already exists: pid {pid}"
            )


@contextmanager
def _campaign_calibration_lock(campaign_root):
    lock_root = Path(campaign_root) / "calibration"
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_root / ".calibration.lock"
    payload = {"pid": os.getpid(), "created_at": _timestamp()}
    for _ in range(2):
        try:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
                stream.write("\n")
            break
        except FileExistsError:
            try:
                prior = _read_json(path, "calibration lock")
                pid = int(prior.get("pid"))
            except (UserError, TypeError, ValueError):
                pid = None
            if pid is not None and _process_alive(pid):
                raise UserError(f"another calibration is active: pid {pid}")
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    else:
        raise UserError("could not acquire campaign calibration lock")
    try:
        yield
    finally:
        try:
            prior = _read_json(path, "calibration lock")
            if int(prior.get("pid", -1)) == os.getpid():
                path.unlink()
        except (UserError, TypeError, ValueError, FileNotFoundError):
            pass


def execute(args):
    campaign_root = (
        Path(args.campaign_root).resolve()
        if args.campaign_root is not None
        else (LOG_ROOT / args.batch_id).resolve()
    )
    _assert_no_other_work(campaign_root)
    with _campaign_calibration_lock(campaign_root):
        _assert_no_other_work(campaign_root)
        plan, phase, _, jobs = _load_phase(campaign_root, args.phase_id)
        if plan.get("batch_id") != args.batch_id:
            raise UserError(
                f"batch id mismatch: CLI={args.batch_id}, PLAN={plan.get('batch_id')}"
            )
        if plan.get("gpu") != args.gpu:
            raise UserError(
                f"GPU mismatch: CLI={args.gpu}, PLAN={plan.get('gpu')}"
            )
        _assert_plan_revision(plan)
        selected = _select_distinct_jobs(jobs, args.target_workers)
        calibration_root, clones, calibration_plan = _clone_jobs(
            campaign_root, plan, phase, selected, args.calibration_id,
            episode_limit=args.episode_limit,
        )
        calibration_plan.update({
            "gpu": args.gpu,
            "stagger_seconds": args.stagger_seconds,
            "sample_interval_seconds": args.sample_interval_seconds,
            "steady_seconds": args.steady_seconds,
            "steady_samples": args.steady_samples,
            "load_timeout_seconds": args.load_timeout_seconds,
            "min_loaded_memory_mib": args.min_loaded_memory_mib,
            "steady_relative_tolerance": args.steady_relative_tolerance,
        })
        _atomic_json(
            calibration_root / "CALIBRATION_PLAN.json", calibration_plan
        )
        return calibration_root, _run_calibration(
            calibration_root, phase, clones, calibration_plan,
            gpu=args.gpu,
            stagger_seconds=args.stagger_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            terminate_grace_seconds=args.terminate_grace_seconds,
            steady_seconds=args.steady_seconds,
            steady_samples=args.steady_samples,
            load_timeout_seconds=args.load_timeout_seconds,
            min_loaded_memory_mib=args.min_loaded_memory_mib,
            steady_relative_tolerance=args.steady_relative_tolerance,
        )


def _default_calibration_id():
    return time.strftime("%Y%m%dT%H%M%S")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Safely ramp concurrency for exactly one planned phase."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--target-workers", required=True, type=int)
    parser.add_argument("--calibration-id", default=_default_calibration_id())
    parser.add_argument(
        "--campaign-root",
        help="alternate directory containing PLAN.json (primarily for audits/tests)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--episode-limit", type=int,
        help=(
            "optional non-formal canonical prefix; omit for the complete "
            "1,021-episode val_seen run"
        ),
    )
    parser.add_argument(
        "--stagger-seconds", type=float, default=DEFAULT_STAGGER_SECONDS,
        help=(
            "minimum delay per level only; the loaded-and-steady gate always "
            "controls the next launch"
        ),
    )
    parser.add_argument(
        "--sample-interval-seconds", type=float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help="nvidia-smi/cgroup sampling period; must be between 2 and 5 seconds",
    )
    parser.add_argument(
        "--terminate-grace-seconds", type=float,
        default=DEFAULT_TERMINATE_GRACE_SECONDS,
    )
    parser.add_argument(
        "--steady-seconds", type=float, default=DEFAULT_STEADY_SECONDS,
        help="minimum duration of the recent stable VRAM window",
    )
    parser.add_argument(
        "--steady-samples", type=int, default=DEFAULT_STEADY_SAMPLES,
        help="minimum sample count in the recent stable VRAM window",
    )
    parser.add_argument(
        "--load-timeout-seconds", type=float,
        default=DEFAULT_LOAD_TIMEOUT_SECONDS,
        help="fail safely if a newly added worker does not load and settle",
    )
    parser.add_argument(
        "--min-loaded-memory-mib", type=int,
        default=DEFAULT_MIN_LOADED_MEMORY_MIB,
        help="required VRAM increase over the immediately previous steady level",
    )
    parser.add_argument(
        "--steady-relative-tolerance", type=float,
        default=DEFAULT_STEADY_RELATIVE_TOLERANCE,
        help="maximum (window max-min)/median VRAM for steady confirmation",
    )
    args = parser.parse_args(argv)
    try:
        _safe_component("batch_id", args.batch_id)
        _safe_component("phase_id", args.phase_id)
        _safe_component("calibration_id", args.calibration_id, max_length=64)
    except UserError as error:
        parser.error(str(error))
    if args.target_workers < 1:
        parser.error("--target-workers must be positive")
    if args.gpu < 0:
        parser.error("--gpu must be non-negative")
    if args.episode_limit is not None and args.episode_limit < 1:
        parser.error("--episode-limit must be positive")
    if args.stagger_seconds < 0:
        parser.error("--stagger-seconds must be non-negative")
    if not 2.0 <= args.sample_interval_seconds <= 5.0:
        parser.error("--sample-interval-seconds must be in [2, 5]")
    if args.terminate_grace_seconds < 0:
        parser.error("--terminate-grace-seconds must be non-negative")
    if args.steady_seconds <= 0:
        parser.error("--steady-seconds must be positive")
    if args.steady_samples < 2:
        parser.error("--steady-samples must be at least 2")
    if args.load_timeout_seconds <= max(
        args.steady_seconds, args.stagger_seconds
    ):
        parser.error(
            "--load-timeout-seconds must exceed both --steady-seconds and "
            "--stagger-seconds"
        )
    if args.min_loaded_memory_mib < 1:
        parser.error("--min-loaded-memory-mib must be positive")
    if not 0 < args.steady_relative_tolerance <= 1:
        parser.error("--steady-relative-tolerance must be in (0, 1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    root, summary = execute(args)
    print(json.dumps({
        "calibration_root": str(root),
        "status": summary["status"],
        "recommended_cap": summary["recommended_cap"],
        "peak_observed_gpu_memory_mib": summary[
            "peak_observed_gpu_memory_mib"
        ],
    }, indent=2, sort_keys=True))
    if summary["status"] in ("emergency_abort", "failed", "inconclusive"):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except UserError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
