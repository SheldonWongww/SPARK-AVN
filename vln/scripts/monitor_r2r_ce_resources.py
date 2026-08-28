#!/usr/bin/env python3
"""Record compact per-job resource peaks for an active R2R-CE campaign.

The monitor is deliberately external to the experiment process.  It samples
NVML through ``nvidia-smi`` and Linux ``/proc`` without modifying a formal run
or importing its Python environment.  Only maxima and job identities are
persisted; raw command lines and time-series samples are never written.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time


SCHEMA = "navtta.vln_r2r_ce_resource_calibration.v1"
SETTINGS = ("etpnav-r2r-ce", "bevbert-r2r-ce")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
STAGES = ("search", "retention", "calibration")


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def parse_number(value):
    match = re.search(r"-?[0-9]+(?:\.[0-9]+)?", str(value))
    if match is None:
        return None
    number = float(match.group(0))
    return number if math.isfinite(number) else None


def parse_compute_apps(text):
    records = []
    for line in text.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        memory = parse_number(fields[1])
        if memory is not None and memory >= 0:
            records.append({"pid": int(fields[0]), "gpu_memory_mib": memory})
    return records


def parse_gpu_state(text):
    fields = [item.strip() for item in text.strip().split(",")]
    if len(fields) < 5:
        raise ValueError("nvidia-smi GPU row has fewer than five fields")
    values = [parse_number(item) for item in fields[:5]]
    if any(item is None for item in values):
        raise ValueError("nvidia-smi GPU row contains a non-numeric field")
    return {
        "memory_total_mib": values[0],
        "memory_used_mib": values[1],
        "memory_free_mib": values[2],
        "gpu_utilization_percent": values[3],
        "memory_utilization_percent": values[4],
    }


def read_process_table(proc_root=Path("/proc")):
    table = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw_stat = (entry / "stat").read_text(encoding="utf-8")
            # The command name may contain spaces or parentheses, so split
            # only after the final closing parenthesis.
            suffix = raw_stat.rsplit(")", 1)[1].strip().split()
            ppid = int(suffix[1])
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            command = cmdline.decode("utf-8", errors="replace").strip()
            rss_kib = 0
            for line in (entry / "status").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                if line.startswith("VmRSS:"):
                    value = parse_number(line)
                    rss_kib = int(value or 0)
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        table[int(entry.name)] = {
            "ppid": ppid,
            "command": command,
            "rss_mib": rss_kib / 1024.0,
        }
    return table


def ancestry(pid, table, maximum_depth=32):
    result = []
    seen = set()
    current = int(pid)
    while current in table and current not in seen and len(result) < maximum_depth:
        seen.add(current)
        result.append(current)
        current = table[current]["ppid"]
    return result


def classify_command(command):
    setting = next((value for value in SETTINGS if value in command), None)
    method = None
    for value in METHODS:
        patterns = (
            "/{}/".format(value),
            "--methods {}".format(value),
            "-{}-".format(value),
        )
        if any(pattern in command for pattern in patterns):
            method = value
            break
    stage = next(
        (
            value
            for value in STAGES
            if "/{}/".format(value) in command or "-{}-".format(value) in command
        ),
        None,
    )
    candidate = None
    match = re.search(
        r"/(?:search|retention|calibration)/([^/]+)/(?:seed|worker)_[0-9]+",
        command,
    )
    if match:
        candidate = match.group(1)
    order_seed = None
    match = re.search(r"/seed_([0-9]+)(?:/|\s)", command)
    if match:
        order_seed = int(match.group(1))
    worker_index = None
    match = re.search(r"/worker_([0-9]+)(?:/|\s)", command)
    if match:
        worker_index = int(match.group(1))
    run_tag = None
    match = re.search(r"--run-tag\s+([^\s]+)", command)
    if match:
        run_tag = match.group(1)
    return {
        "setting": setting,
        "method": method,
        "stage": stage,
        "candidate_id": candidate,
        "order_seed": order_seed,
        "calibration_worker_index": worker_index,
        "run_tag": run_tag,
    }


def identify_job(gpu_pid, table, watch_pattern):
    chain = ancestry(gpu_pid, table)
    commands = [table[pid]["command"] for pid in chain if table[pid]["command"]]
    combined = " ".join(commands)
    identity = classify_command(combined)
    if watch_pattern not in combined:
        return None
    if not all(identity.get(key) for key in ("setting", "method", "stage", "run_tag")):
        return None
    anchor = next(
        (
            pid
            for pid in chain
            if "run_source_eval.sh" in table[pid]["command"]
            and identity["run_tag"] in table[pid]["command"]
        ),
        gpu_pid,
    )
    identity["anchor_pid"] = anchor
    identity["gpu_pid"] = gpu_pid
    return identity


def descendant_rss(anchor_pid, table):
    total = 0.0
    for pid, record in table.items():
        if anchor_pid in ancestry(pid, table):
            total += record["rss_mib"]
    return total


def maximum(record, key, value):
    if value is None:
        return
    previous = record.get(key)
    if previous is None or value > previous:
        record[key] = value


def minimum(record, key, value):
    if value is None:
        return
    previous = record.get(key)
    if previous is None or value < previous:
        record[key] = value


def running_mean(record, key, value, sample_count):
    if value is None:
        return
    previous = float(record.get(key, 0.0))
    record[key] = previous + (float(value) - previous) / float(sample_count)


def update_summary(
    summary, gpu_state, compute_apps, table, watch_pattern, cgroup_state=None
):
    grouped = {}
    for app in compute_apps:
        identity = identify_job(app["pid"], table, watch_pattern)
        if identity is None:
            continue
        phase_key = "{}/{}/{}".format(
            identity["setting"], identity["method"], identity["stage"]
        )
        grouped.setdefault(phase_key, []).append((app, identity))

    for phase_key, jobs in grouped.items():
        phase = summary["phases"].setdefault(
            phase_key,
            {
                "samples": 0,
                "max_concurrent_jobs_observed": 0,
                "jobs": {},
            },
        )
        phase["samples"] += 1
        histogram = phase.setdefault("concurrent_job_sample_counts", {})
        concurrency_key = str(len(jobs))
        histogram[concurrency_key] = histogram.get(concurrency_key, 0) + 1
        running_mean(
            phase,
            "mean_board_memory_used_mib",
            gpu_state["memory_used_mib"],
            phase["samples"],
        )
        running_mean(
            phase,
            "mean_gpu_utilization_percent",
            gpu_state["gpu_utilization_percent"],
            phase["samples"],
        )
        running_mean(
            phase,
            "mean_memory_utilization_percent",
            gpu_state["memory_utilization_percent"],
            phase["samples"],
        )
        maximum(phase, "max_concurrent_jobs_observed", len(jobs))
        maximum(
            phase,
            "peak_sum_process_gpu_memory_mib",
            sum(app["gpu_memory_mib"] for app, _ in jobs),
        )
        maximum(phase, "peak_board_memory_used_mib", gpu_state["memory_used_mib"])
        minimum(phase, "minimum_board_memory_free_mib", gpu_state["memory_free_mib"])
        maximum(
            phase, "peak_gpu_utilization_percent", gpu_state["gpu_utilization_percent"]
        )
        maximum(
            phase,
            "peak_memory_utilization_percent",
            gpu_state["memory_utilization_percent"],
        )
        if cgroup_state is not None:
            running_mean(
                phase,
                "mean_cgroup_memory_gib",
                cgroup_state.get("memory_current_gib"),
                phase["samples"],
            )
            maximum(
                phase,
                "peak_cgroup_memory_gib",
                cgroup_state.get("memory_current_gib"),
            )
        aggregate_rss = 0.0
        for app, identity in jobs:
            job = phase["jobs"].setdefault(
                identity["run_tag"],
                {
                    "candidate_id": identity["candidate_id"],
                    "order_seed": identity["order_seed"],
                    "calibration_worker_index": identity[
                        "calibration_worker_index"
                    ],
                    "samples": 0,
                },
            )
            job["samples"] += 1
            maximum(job, "peak_gpu_memory_mib", app["gpu_memory_mib"])
            rss = descendant_rss(identity["anchor_pid"], table)
            maximum(job, "peak_host_rss_mib", rss)
            aggregate_rss += rss
        maximum(phase, "peak_sum_job_host_rss_mib", aggregate_rss)
    summary["observed_at"] = utc_now()
    summary["total_samples"] += 1
    return bool(grouped)


def read_cgroup_state(current_path, maximum_path):
    current = Path(current_path).read_text(encoding="utf-8").strip()
    maximum_value = Path(maximum_path).read_text(encoding="utf-8").strip()
    current_bytes = int(current)
    maximum_bytes = None if maximum_value == "max" else int(maximum_value)
    if current_bytes < 0 or (maximum_bytes is not None and maximum_bytes <= 0):
        raise ValueError("invalid cgroup memory accounting")
    return {
        "memory_current_gib": current_bytes / float(2 ** 30),
        "memory_limit_gib": (
            None if maximum_bytes is None else maximum_bytes / float(2 ** 30)
        ),
    }


def query(command):
    completed = subprocess.run(
        command, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "command failed")
    return completed.stdout


def campaign_alive(table, watch_pattern):
    markers = (
        "run_consistency_campaign.sh",
        "run_consistency_hparam_search.py",
        "run_source_eval.sh",
    )
    return any(
        watch_pattern in record["command"]
        and any(marker in record["command"] for marker in markers)
        for record in table.values()
    )


def load_or_create_summary(args):
    output = Path(args.output)
    if output.exists():
        if not args.resume:
            raise ValueError(
                "output already exists; pass --resume to continue its evidence"
            )
        try:
            summary = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("cannot resume invalid output: {}".format(error))
        expected = {
            "schema": SCHEMA,
            "watch_pattern": args.watch_pattern,
            "gpu_index": args.gpu,
            "poll_seconds": args.poll_seconds,
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                raise ValueError("resume identity mismatch for {}".format(key))
        if not isinstance(summary.get("phases"), dict):
            raise ValueError("resume output has invalid phases")
        summary.setdefault("resume_times", []).append(utc_now())
        summary["completed_at"] = None
        return summary
    return {
        "schema": SCHEMA,
        "watch_pattern": args.watch_pattern,
        "gpu_index": args.gpu,
        "poll_seconds": args.poll_seconds,
        "started_at": utc_now(),
        "resume_times": [],
        "observed_at": None,
        "completed_at": None,
        "total_samples": 0,
        "sampling_errors": 0,
        "phases": {},
    }


def run(args):
    summary = load_or_create_summary(args)
    stopping = {"requested": False}

    def request_stop(_signum, _frame):
        stopping["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    last_flush = 0.0
    idle_since = None
    while not stopping["requested"]:
        started = time.monotonic()
        try:
            table = read_process_table(Path(args.proc_root))
            gpu_state = parse_gpu_state(
                query(
                    [
                        args.nvidia_smi,
                        "-i",
                        str(args.gpu),
                        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
                        "--format=csv,noheader,nounits",
                    ]
                )
            )
            apps = parse_compute_apps(
                query(
                    [
                        args.nvidia_smi,
                        "-i",
                        str(args.gpu),
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ]
                )
            )
            cgroup_state = read_cgroup_state(
                args.cgroup_memory_current, args.cgroup_memory_max
            )
            summary["cgroup_memory_limit_gib"] = cgroup_state[
                "memory_limit_gib"
            ]
            active = update_summary(
                summary,
                gpu_state,
                apps,
                table,
                args.watch_pattern,
                cgroup_state,
            )
            if active or campaign_alive(table, args.watch_pattern):
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= args.idle_exit_seconds:
                break
        except (OSError, RuntimeError, ValueError):
            summary["sampling_errors"] += 1
        now = time.monotonic()
        if now - last_flush >= args.flush_seconds:
            atomic_json(args.output, summary)
            last_flush = now
        if args.max_samples is not None and summary["total_samples"] >= args.max_samples:
            break
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, args.poll_seconds - elapsed))
    summary["completed_at"] = utc_now()
    atomic_json(args.output, summary)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-pattern", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--flush-seconds", type=float, default=30.0)
    parser.add_argument("--idle-exit-seconds", type=float, default=300.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--proc-root", default="/proc")
    parser.add_argument(
        "--cgroup-memory-current", default="/sys/fs/cgroup/memory.current"
    )
    parser.add_argument(
        "--cgroup-memory-max", default="/sys/fs/cgroup/memory.max"
    )
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    for name in ("poll_seconds", "flush_seconds", "idle_exit_seconds"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
