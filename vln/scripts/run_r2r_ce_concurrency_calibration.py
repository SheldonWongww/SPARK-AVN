#!/usr/bin/env python3
"""Validate R2R-CE worker caps with complete 100-episode stress groups.

Each model-method cell runs in isolation.  Every worker uses the tracked
paper-anchor TTA configuration and the canonical ``val_seen`` prefix through
``run_source_eval.sh --smoke-episodes``.  A target is approved only when the
whole group is simultaneously resident, every worker completes, real TTA
updates occur, and both GPU and cgroup memory remain below the reviewed
production ceilings.  Resource failures descend one worker at a time; method
or evidence failures stop the campaign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = (
    REPO_ROOT / "vln/experiments/r2r_ce_concurrency_calibration_v1.json"
)
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
TRANSLATOR = REPO_ROOT / "vln/scripts/tta_config_cli.py"
MONITOR = REPO_ROOT / "vln/scripts/monitor_r2r_ce_resources.py"
SCHEMA = "navtta.vln_r2r_ce_concurrency_calibration.v1"
SEARCH_SCHEMA = "navtta.vln_tta_consistency_search.v2"
SUMMARY_SCHEMA = "navtta.vln_r2r_ce_concurrency_calibration_summary.v1"
APPROVED_SCHEMA = "navtta.vln_r2r_ce_approved_concurrency.v1"
JOB_SCHEMA = "navtta.vln_tta_job.v1"
SETTINGS = ("etpnav-r2r-ce", "bevbert-r2r-ce")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


class UserError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def read_json(path, label="JSON file"):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise UserError("{} must be a JSON object: {}".format(label, path))
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT)] + list(args),
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "output", "")
        raise UserError("git {} failed: {}".format(" ".join(args), output or error))


def repo_path(value, label):
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise UserError("{} must stay inside the repository".format(label))
    return path.resolve()


def validate_component(value, label):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 120
        or SAFE_COMPONENT.fullmatch(value) is None
    ):
        raise UserError("invalid {}: {!r}".format(label, value))
    return value


def _exact_keys(mapping, expected, label):
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        raise UserError("{} keys must equal {}".format(label, list(expected)))


def load_spec(path):
    path = Path(path).resolve()
    spec = read_json(path, "calibration specification")
    if spec.get("schema") != SCHEMA or spec.get("benchmark") != "r2r-ce":
        raise UserError("unsupported R2R-CE calibration specification")
    if tuple(spec.get("settings", ())) != SETTINGS:
        raise UserError("calibration setting order changed")
    if tuple(spec.get("methods", ())) != METHODS:
        raise UserError("calibration method order changed")

    protocol = spec.get("protocol", {})
    required_protocol = {
        "split": "val_seen",
        "episode_count": 100,
        "candidate_role": "paper_anchor",
        "action_selection": "argmax",
        "formal_navigation_result": False,
        "require_all_workers_complete": True,
        "require_simultaneous_gpu_residency": True,
        "require_positive_method_updates": True,
        "strict_model_method_barrier": True,
        "auto_descend_on_resource_failure": True,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise UserError("calibration protocol {} must be {!r}".format(
                key, expected
            ))
    if type(protocol.get("minimum_full_residency_samples")) is not int or (
        protocol["minimum_full_residency_samples"] < 1
    ):
        raise UserError("minimum_full_residency_samples must be positive")

    limits = spec.get("resource_limits", {})
    required_limits = {
        "production_gpu_memory_mib_exclusive": 29000,
        "emergency_gpu_memory_mib": 30000,
        "production_cgroup_memory_gib_exclusive": 80.0,
    }
    for key, expected in required_limits.items():
        if limits.get(key) != expected:
            raise UserError("resource limit {} must remain {}".format(key, expected))
    for key in (
        "poll_seconds", "termination_grace_seconds",
        "idle_gpu_memory_mib_max", "gpu_clear_timeout_seconds",
    ):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise UserError("resource limit {} must be positive".format(key))

    minimum = spec.get("minimum_concurrency")
    target = spec.get("target_concurrency")
    _exact_keys(minimum, SETTINGS, "minimum_concurrency")
    _exact_keys(target, SETTINGS, "target_concurrency")
    for setting in SETTINGS:
        _exact_keys(minimum[setting], METHODS, setting + " minimum_concurrency")
        _exact_keys(target[setting], METHODS, setting + " target_concurrency")
        for method in METHODS:
            low = minimum[setting][method]
            high = target[setting][method]
            if (
                type(low) is not int or type(high) is not int
                or low < 1 or high < low
            ):
                raise UserError("invalid concurrency range for {}/{}".format(
                    setting, method
                ))

    dependency = spec.get("base_search_spec", {})
    if not isinstance(dependency, dict) or not HEX_SHA256.fullmatch(
        str(dependency.get("sha256", ""))
    ):
        raise UserError("base_search_spec binding is invalid")
    search_path = repo_path(dependency.get("path", ""), "base_search_spec")
    if not search_path.is_file() or sha256(search_path) != dependency["sha256"]:
        raise UserError("base consistency-search spec SHA256 mismatch")
    search = read_json(search_path, "base consistency-search specification")
    if search.get("schema") != SEARCH_SCHEMA or search.get("benchmark") != "r2r-ce":
        raise UserError("base consistency-search specification is incompatible")
    if tuple(search.get("settings", ())) != SETTINGS:
        raise UserError("base search settings differ from calibration")
    _exact_keys(search.get("methods"), METHODS + ("idea",), "base search methods")
    for method in METHODS:
        anchors = [
            item for item in search["methods"][method].get("candidates", [])
            if item.get("role") == protocol["candidate_role"]
        ]
        if len(anchors) != 1 or anchors[0].get("id") != "paper_anchor":
            raise UserError("{}/paper anchor is missing or ambiguous".format(method))
        parameters = dict(search["methods"][method].get("base", {}))
        parameters.update(anchors[0].get("parameters", {}))
        if parameters.get("action_selection") != "argmax":
            raise UserError("{}/paper anchor must use argmax".format(method))
    return spec, search_path, search


def paper_anchor_parameters(search, method, worker_index):
    method_spec = search["methods"][method]
    anchor = next(
        item for item in method_spec["candidates"]
        if item.get("role") == "paper_anchor"
    )
    parameters = dict(method_spec.get("base", {}))
    parameters.update(anchor.get("parameters", {}))
    if method == "feedtta":
        parameters["sgr_seed"] = worker_index - 1
        parameters.pop("action_seed", None)
    return parameters


def build_job(batch_id, out_dir, search, setting, method, concurrency, attempt,
              worker_index, gpu, episode_count):
    attempt_tag = "{}-{}-{}-c{}-a{}".format(
        batch_id, setting, method, concurrency, attempt
    )
    run_tag = "{}-calibration-paper_anchor-w{}".format(
        attempt_tag, worker_index
    )
    validate_component(run_tag, "worker run tag")
    job_dir = (
        Path(out_dir) / setting / method / "attempt_{}_c{}".format(
            attempt, concurrency
        ) / "calibration" / "paper_anchor" / "worker_{}".format(worker_index)
    ).resolve()
    config_path = job_dir / "tta_config.json"
    result_root = (
        REPO_ROOT / "vln/results/smoke" / run_tag / setting / "val_seen"
    ).resolve()
    parameters = paper_anchor_parameters(search, method, worker_index)
    config = {
        "schema": JOB_SCHEMA,
        "namespace": "tuning",
        "stage": "resource_calibration",
        "episodes": episode_count,
        "search_method": method,
        "method": method,
        "candidate_id": "paper_anchor",
        "parameters": parameters,
        "calibration": {
            "batch_id": batch_id,
            "attempt": attempt,
            "target_concurrency": concurrency,
            "worker_index": worker_index,
            "formal_navigation_result": False,
        },
    }
    command = [
        "bash", str(RUNNER), setting, "val_seen", str(gpu),
        "--run-tag", run_tag,
        "--ce-data-version", "v1.3-unified",
        "--smoke-episodes", str(episode_count),
        "--tta-config", str(config_path),
    ]
    return {
        "run_tag": run_tag,
        "worker_index": worker_index,
        "job_dir": str(job_dir),
        "config_path": str(config_path),
        "result_root": str(result_root),
        "diagnostics_path": str(result_root / "tta_diagnostics.json"),
        "parameters": parameters,
        "config": config,
        "command": command,
    }


def validate_job_config(job, setting, method):
    completed = subprocess.run(
        [
            sys.executable, str(TRANSLATOR), "--setting", setting,
            "--config", job["config_path"],
            "--diagnostics", "/tmp/navtta-r2rce-calibration-unused.json",
            "--print-method",
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != method:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise UserError("TTA config rejected for {}/{}: {}".format(
            setting, method, detail
        ))


def prepare_jobs(batch_id, out_dir, search, setting, method, concurrency,
                 attempt, gpu, episode_count, write=True):
    jobs = [
        build_job(
            batch_id, out_dir, search, setting, method, concurrency, attempt,
            index, gpu, episode_count,
        )
        for index in range(1, concurrency + 1)
    ]
    if write:
        for job in jobs:
            job_dir = Path(job["job_dir"])
            job_dir.mkdir(parents=True, exist_ok=False)
            atomic_json(job["config_path"], job["config"])
            validate_job_config(job, setting, method)
            public_job = {
                key: value for key, value in job.items()
                if key != "config"
            }
            public_job["config_sha256"] = sha256(job["config_path"])
            atomic_json(job_dir / "job.json", public_job)
    return jobs


def gpu_state(gpu):
    command = [
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise UserError(completed.stderr.strip() or "nvidia-smi failed")
    rows = completed.stdout.strip().splitlines()
    if len(rows) != 1:
        raise UserError("nvidia-smi did not return exactly one GPU row")
    try:
        used, free, utilization = [
            float(value.strip()) for value in rows[0].split(",")[:3]
        ]
    except (ValueError, IndexError):
        raise UserError("invalid nvidia-smi GPU telemetry")
    return {
        "memory_used_mib": used,
        "memory_free_mib": free,
        "utilization_percent": utilization,
    }


def gpu_compute_pids(gpu):
    completed = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-compute-apps=pid", "--format=csv,noheader,nounits",
        ], text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise UserError(completed.stderr.strip() or "cannot query GPU processes")
    return [
        int(line.strip()) for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]


def cgroup_memory_gib():
    candidates = (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    )
    for path in candidates:
        try:
            return int(path.read_text(encoding="utf-8").strip()) / float(2 ** 30)
        except (OSError, ValueError):
            continue
    raise UserError("cannot read cgroup memory usage")


def wait_for_idle_gpu(gpu, limits):
    deadline = time.monotonic() + float(limits["gpu_clear_timeout_seconds"])
    last = None
    while time.monotonic() < deadline:
        state = gpu_state(gpu)
        pids = gpu_compute_pids(gpu)
        last = {"state": state, "pids": pids}
        if not pids and state["memory_used_mib"] <= float(
            limits["idle_gpu_memory_mib_max"]
        ):
            return state
        time.sleep(1.0)
    raise UserError("GPU did not become idle before calibration: {}".format(last))


def signal_group(process, signum):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def terminate_processes(processes, grace_seconds):
    for process in processes:
        signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + float(grace_seconds)
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process in processes):
            break
        time.sleep(0.1)
    for process in processes:
        if process.poll() is None:
            signal_group(process, signal.SIGKILL)
    for process in processes:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def tail_contains_oom(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 131072), os.SEEK_SET)
            text = stream.read().decode("utf-8", errors="replace").lower()
    except OSError:
        return False
    markers = ("out of memory", "cuda error: out of memory", "cuda oom")
    return any(marker in text for marker in markers)


def validate_diagnostics(job, setting, method, episode_count):
    path = Path(job["diagnostics_path"])
    diagnostics = read_json(path, "TTA diagnostics")
    adapter = diagnostics.get("adapter")
    errors = []
    if diagnostics.get("method") != method:
        errors.append("method mismatch")
    if diagnostics.get("baseline") not in ("etpnav", "bevbert"):
        errors.append("baseline missing")
    if diagnostics.get("episode_count") != episode_count:
        errors.append("episode_count mismatch")
    if diagnostics.get("action_selection") != "target_native_argmax":
        errors.append("action selection is not target-native argmax")
    if not isinstance(adapter, dict):
        errors.append("adapter diagnostics missing")
        updates = None
    else:
        updates = adapter.get("updates")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
            errors.append("no positive method updates")
        if adapter.get("episodes") != episode_count:
            errors.append("adapter episode count mismatch")
    expected_feedback = method in ("feedtta", "atena")
    if diagnostics.get("feedback_supervision") != (
        "binary_episode_success" if expected_feedback else "none"
    ):
        errors.append("feedback supervision mismatch")
    if method == "feedtta" and isinstance(adapter, dict):
        if adapter.get("feedback_episodes") != episode_count:
            errors.append("FeedTTA feedback episode count mismatch")
        if not isinstance(adapter.get("policy_gradient_steps"), int) or (
            adapter["policy_gradient_steps"] <= 0
        ):
            errors.append("FeedTTA policy gradients missing")
    if method == "fstta" and isinstance(adapter, dict):
        if not isinstance(adapter.get("slow_updates"), int) or (
            adapter["slow_updates"] <= 0
        ):
            errors.append("FSTTA slow updates missing")
    if method == "atena":
        if diagnostics.get("atena_exact_replay_within_declared_scope") is not True:
            errors.append("ATENA exact replay was not validated")
        if diagnostics.get("atena_optimizer_scope_matches_reachable") is not True:
            errors.append("ATENA optimizer scope was not validated")
    return {
        "valid": not errors,
        "errors": errors,
        "diagnostics_path": str(path),
        "diagnostics_sha256": sha256(path) if path.is_file() else None,
        "episode_count": diagnostics.get("episode_count"),
        "updates": updates,
    }


def start_monitor(attempt_tag, attempt_dir, gpu, limits):
    output = Path(attempt_dir) / "resource_summary.json"
    log_path = Path(attempt_dir) / "resource_monitor.log"
    log = log_path.open("ab", buffering=0)
    command = [
        sys.executable, str(MONITOR),
        "--watch-pattern", attempt_tag,
        "--output", str(output),
        "--gpu", str(gpu),
        "--poll-seconds", str(limits["poll_seconds"]),
        "--flush-seconds", "10",
        "--idle-exit-seconds", "300",
    ]
    try:
        process = subprocess.Popen(
            command, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log.close()
        raise
    return process, log, output


def stop_monitor(process, log):
    signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        signal_group(process, signal.SIGKILL)
        process.wait(timeout=3)
    log.close()


def run_attempt(batch_id, out_dir, search, setting, method, concurrency,
                attempt_number, gpu, spec):
    limits = spec["resource_limits"]
    protocol = spec["protocol"]
    wait_for_idle_gpu(gpu, limits)
    jobs = prepare_jobs(
        batch_id, out_dir, search, setting, method, concurrency,
        attempt_number, gpu, protocol["episode_count"], write=True,
    )
    attempt_dir = Path(jobs[0]["job_dir"]).parents[2]
    attempt_tag = "{}-{}-{}-c{}-a{}".format(
        batch_id, setting, method, concurrency, attempt_number
    )
    plan = {
        "schema": "navtta.vln_r2r_ce_concurrency_calibration_attempt.v1",
        "batch_id": batch_id,
        "attempt_tag": attempt_tag,
        "setting": setting,
        "method": method,
        "target_concurrency": concurrency,
        "episode_count_per_worker": protocol["episode_count"],
        "created_at": utc_now(),
        "jobs": [
            {key: value for key, value in job.items() if key != "config"}
            for job in jobs
        ],
    }
    atomic_json(attempt_dir / "ATTEMPT_PLAN.json", plan)

    monitor_process = None
    monitor_log = None
    monitor_output = None
    processes = []
    logs = []
    started = time.monotonic()
    peak_board = 0.0
    peak_cgroup = 0.0
    peak_utilization = 0.0
    abort_reason = None
    try:
        monitor_process, monitor_log, monitor_output = start_monitor(
            attempt_tag, attempt_dir, gpu, limits
        )
        for job in jobs:
            log_path = Path(job["job_dir"]) / "launcher.log"
            handle = log_path.open("ab", buffering=0)
            logs.append(handle)
            process = subprocess.Popen(
                job["command"], cwd=str(REPO_ROOT), stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
            )
            processes.append(process)

        while any(process.poll() is None for process in processes):
            if STOP_REQUESTED["value"]:
                abort_reason = "operator_signal"
                break
            state = gpu_state(gpu)
            memory = cgroup_memory_gib()
            peak_board = max(peak_board, state["memory_used_mib"])
            peak_cgroup = max(peak_cgroup, memory)
            peak_utilization = max(peak_utilization, state["utilization_percent"])
            if state["memory_used_mib"] >= limits["emergency_gpu_memory_mib"]:
                abort_reason = "emergency_gpu_memory_threshold"
                break
            if memory >= limits["production_cgroup_memory_gib_exclusive"]:
                abort_reason = "cgroup_memory_threshold"
                break
            nonzero = [
                process.returncode for process in processes
                if process.poll() not in (None, 0)
            ]
            if nonzero:
                abort_reason = "worker_failure"
                break
            time.sleep(float(limits["poll_seconds"]))
    except (OSError, UserError) as error:
        abort_reason = "telemetry_or_launch_failure"
        telemetry_error = str(error)
    else:
        telemetry_error = None
    finally:
        if abort_reason is not None:
            terminate_processes(processes, limits["termination_grace_seconds"])
        else:
            for process in processes:
                process.wait()
        for handle in logs:
            handle.close()
        if monitor_process is not None:
            stop_monitor(monitor_process, monitor_log)

    exit_codes = [process.returncode for process in processes]
    if abort_reason is None and any(code != 0 for code in exit_codes):
        abort_reason = "worker_failure"
    oom_detected = any(code in (-9, 137) for code in exit_codes) or any(
        tail_contains_oom(Path(job["job_dir"]) / "launcher.log") for job in jobs
    )
    if abort_reason == "worker_failure" and oom_detected:
        abort_reason = "worker_oom"

    resource = None
    resource_error = None
    if monitor_output is not None and monitor_output.is_file():
        try:
            resource = read_json(monitor_output, "resource summary")
        except UserError as error:
            resource_error = str(error)
    else:
        resource_error = "resource monitor produced no summary"
    phase = None if resource is None else resource.get("phases", {}).get(
        "{}/{}/calibration".format(setting, method)
    )
    if isinstance(phase, dict):
        peak_board = max(peak_board, float(phase.get(
            "peak_board_memory_used_mib", 0.0
        )))
        peak_cgroup = max(peak_cgroup, float(phase.get(
            "peak_cgroup_memory_gib", 0.0
        )))
        peak_utilization = max(peak_utilization, float(phase.get(
            "peak_gpu_utilization_percent", 0.0
        )))

    diagnostics = []
    if all(code == 0 for code in exit_codes):
        for job in jobs:
            try:
                diagnostics.append(validate_diagnostics(
                    job, setting, method, protocol["episode_count"]
                ))
            except UserError as error:
                diagnostics.append({
                    "valid": False,
                    "errors": [str(error)],
                    "diagnostics_path": job["diagnostics_path"],
                })

    histogram = {} if not isinstance(phase, dict) else phase.get(
        "concurrent_job_sample_counts", {}
    )
    full_samples = int(histogram.get(str(concurrency), 0))
    validation_errors = []
    if abort_reason is not None:
        validation_errors.append(abort_reason)
    if resource_error is not None:
        validation_errors.append(resource_error)
    if not isinstance(phase, dict):
        validation_errors.append("calibration phase was not observed")
    else:
        if phase.get("max_concurrent_jobs_observed") != concurrency:
            validation_errors.append("full simultaneous residency was not observed")
        if full_samples < protocol["minimum_full_residency_samples"]:
            validation_errors.append("insufficient full-residency samples")
        if len(phase.get("jobs", {})) != concurrency:
            validation_errors.append("resource monitor did not identify every worker")
    if len(exit_codes) != concurrency or any(code != 0 for code in exit_codes):
        validation_errors.append("not every worker exited successfully")
    if len(diagnostics) != concurrency or any(
        not item.get("valid") for item in diagnostics
    ):
        validation_errors.append("method-update diagnostics failed")
    if peak_board >= limits["production_gpu_memory_mib_exclusive"]:
        validation_errors.append("production GPU-memory ceiling exceeded")
    if peak_cgroup >= limits["production_cgroup_memory_gib_exclusive"]:
        validation_errors.append("production cgroup-memory ceiling exceeded")

    passed = not validation_errors
    capacity_failure = abort_reason in (
        "emergency_gpu_memory_threshold", "cgroup_memory_threshold", "worker_oom"
    ) or peak_board >= limits["production_gpu_memory_mib_exclusive"] or (
        peak_cgroup >= limits["production_cgroup_memory_gib_exclusive"]
    )
    result = {
        "status": "approved" if passed else "rejected",
        "batch_id": batch_id,
        "attempt_tag": attempt_tag,
        "setting": setting,
        "method": method,
        "target_concurrency": concurrency,
        "episode_count_per_worker": protocol["episode_count"],
        "started_at": plan["created_at"],
        "completed_at": utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "exit_codes": exit_codes,
        "abort_reason": abort_reason,
        "telemetry_error": telemetry_error,
        "oom_detected": oom_detected,
        "capacity_failure": capacity_failure,
        "peak_board_memory_used_mib": peak_board,
        "peak_cgroup_memory_gib": peak_cgroup,
        "peak_gpu_utilization_percent": peak_utilization,
        "full_residency_samples": full_samples,
        "resource_summary": str(monitor_output) if monitor_output else None,
        "resource_summary_sha256": (
            sha256(monitor_output)
            if monitor_output is not None and monitor_output.is_file() else None
        ),
        "diagnostics": diagnostics,
        "validation_errors": validation_errors,
    }
    atomic_json(attempt_dir / "ATTEMPT_RESULT.json", result)
    return result


def new_summary(batch_id, spec_path, spec, search_path):
    commit = git("rev-parse", "--verify", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise UserError("tracked worktree must be clean before calibration")
    return {
        "schema": SUMMARY_SCHEMA,
        "batch_id": batch_id,
        "status": "running",
        "git_commit": commit,
        "calibration_spec_path": str(Path(spec_path).resolve()),
        "calibration_spec_sha256": sha256(spec_path),
        "base_search_spec_path": str(search_path),
        "base_search_spec_sha256": sha256(search_path),
        "started_at": utc_now(),
        "completed_at": None,
        "resource_limits": spec["resource_limits"],
        "cells": {},
        "approved_concurrency": {},
    }


def load_summary(path, batch_id, spec_path, search_path, resume):
    path = Path(path)
    if not path.exists():
        return None
    if not resume:
        raise UserError("output already exists; pass --resume or use a new batch ID")
    summary = read_json(path, "calibration summary")
    expected = {
        "schema": SUMMARY_SCHEMA,
        "batch_id": batch_id,
        "git_commit": git("rev-parse", "--verify", "HEAD"),
        "calibration_spec_sha256": sha256(spec_path),
        "base_search_spec_sha256": sha256(search_path),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise UserError("resume identity mismatch for {}".format(key))
    for cell in summary.get("cells", {}).values():
        if cell.get("status") == "running":
            raise UserError("cannot resume while a recorded attempt is still running")
    summary["status"] = "running"
    summary["completed_at"] = None
    summary.setdefault("resume_times", []).append(utc_now())
    return summary


def selected(values, requested, label):
    if not requested:
        return list(values)
    unknown = sorted(set(requested).difference(values))
    if unknown:
        raise UserError("unknown {}: {}".format(label, ", ".join(unknown)))
    return [value for value in values if value in set(requested)]


def run(args):
    spec_path = Path(args.spec).resolve()
    spec, search_path, search = load_spec(spec_path)
    settings = selected(SETTINGS, args.settings, "settings")
    methods = selected(METHODS, args.methods, "methods")
    out_dir = Path(args.out_dir).resolve()

    if args.dry_run:
        if out_dir.exists() and any(out_dir.iterdir()):
            raise UserError("dry-run output directory is not empty")
        out_dir.mkdir(parents=True, exist_ok=True)
        for setting in settings:
            for method in methods:
                target = spec["target_concurrency"][setting][method]
                jobs = prepare_jobs(
                    args.batch_id, out_dir, search, setting, method, target,
                    1, args.gpu, spec["protocol"]["episode_count"], write=True,
                )
                command = jobs[0]["command"] + ["--dry-run"]
                completed = subprocess.run(
                    command, cwd=str(REPO_ROOT), text=True,
                    capture_output=True, check=False,
                )
                preflight_path = (
                    Path(jobs[0]["job_dir"]).parents[2] / "launcher_preflight.log"
                )
                preflight_path.write_text(
                    completed.stdout + completed.stderr, encoding="utf-8"
                )
                if completed.returncode != 0:
                    detail = completed.stderr.strip() or completed.stdout.strip()
                    raise UserError("launcher dry-run failed for {}/{}: {}".format(
                        setting, method, detail
                    ))
                print("# {}/{}: {} simultaneous workers".format(
                    setting, method, target
                ))
                print(" ".join(command))
        return 0

    summary_path = out_dir / "SUMMARY.json"
    summary = load_summary(
        summary_path, args.batch_id, spec_path, search_path, args.resume
    )
    if summary is None:
        out_dir.mkdir(parents=True, exist_ok=False)
        summary = new_summary(args.batch_id, spec_path, spec, search_path)
        atomic_json(summary_path, summary)

    for setting in settings:
        summary["approved_concurrency"].setdefault(setting, {})
        for method in methods:
            key = "{}/{}".format(setting, method)
            existing = summary["cells"].get(key)
            if existing and existing.get("status") == "approved":
                continue
            cell = existing or {
                "setting": setting,
                "method": method,
                "status": "running",
                "target_concurrency": spec["target_concurrency"][setting][method],
                "minimum_concurrency": spec["minimum_concurrency"][setting][method],
                "attempts": [],
            }
            cell["status"] = "running"
            summary["cells"][key] = cell
            atomic_json(summary_path, summary)
            target = cell["target_concurrency"]
            minimum = cell["minimum_concurrency"]
            approved = None
            attempted_counts = {
                item["target_concurrency"] for item in cell["attempts"]
            }
            for concurrency in range(target, minimum - 1, -1):
                if concurrency in attempted_counts:
                    continue
                attempt_number = len(cell["attempts"]) + 1
                result = run_attempt(
                    args.batch_id, out_dir, search, setting, method,
                    concurrency, attempt_number, args.gpu, spec,
                )
                cell["attempts"].append(result)
                atomic_json(summary_path, summary)
                if result["status"] == "approved":
                    approved = concurrency
                    break
                if not result["capacity_failure"]:
                    cell["status"] = "failed"
                    cell["failure"] = result["validation_errors"]
                    summary["status"] = "failed"
                    atomic_json(summary_path, summary)
                    raise UserError("non-resource calibration failure for {}".format(key))
                wait_for_idle_gpu(args.gpu, spec["resource_limits"])
            if approved is None:
                cell["status"] = "failed"
                cell["failure"] = ["no concurrency level passed"]
                summary["status"] = "failed"
                atomic_json(summary_path, summary)
                raise UserError("no approved concurrency for {}".format(key))
            cell["status"] = "approved"
            cell["approved_concurrency"] = approved
            cell["completed_at"] = utc_now()
            summary["approved_concurrency"][setting][method] = approved
            atomic_json(summary_path, summary)
            print("[approved] {}/{} concurrency={}".format(
                setting, method, approved
            ), flush=True)

    expected_cells = {"{}/{}".format(s, m) for s in settings for m in methods}
    if all(summary["cells"].get(key, {}).get("status") == "approved"
           for key in expected_cells):
        summary["status"] = "completed"
        summary["completed_at"] = utc_now()
        atomic_json(summary_path, summary)
        approved = {
            "schema": APPROVED_SCHEMA,
            "batch_id": args.batch_id,
            "git_commit": summary["git_commit"],
            "calibration_spec_sha256": summary["calibration_spec_sha256"],
            "base_search_spec_sha256": summary["base_search_spec_sha256"],
            "completed_at": summary["completed_at"],
            "settings": settings,
            "methods": methods,
            "concurrency": summary["approved_concurrency"],
            "summary_path": str(summary_path),
            "summary_sha256": sha256(summary_path),
        }
        atomic_json(out_dir / "APPROVED_CONCURRENCY.json", approved)
    return 0


STOP_REQUESTED = {"value": False}


def request_stop(_signum, _frame):
    STOP_REQUESTED["value"] = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_component(args.batch_id, "batch ID")
    except UserError as error:
        parser.error(str(error))
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    return args


def main(argv=None):
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
