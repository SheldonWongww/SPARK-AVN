#!/usr/bin/env python3
"""Emit a read-only NavTTA batch, process, Git, and GPU status snapshot."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


CAP_KEY = re.compile(
    r"(?:concurr|workers?|jobs?_per_gpu|job_count|expected_jobs|total_jobs|"
    r"episodes?|gpu_count|gpu_memory|cgroup_memory|launch_stagger|profile)",
    re.IGNORECASE,
)
SECRET_KEY = re.compile(
    r"(?:password|passwd|secret|token|credential|private[_-]?key|cookie)",
    re.IGNORECASE,
)


def run(command):
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        return None, str(error)
    if completed.returncode:
        return None, completed.stderr.strip() or "exit {}".format(completed.returncode)
    return completed.stdout.strip(), None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def scalar(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = value if not isinstance(value, str) else value[:160]
        return text
    if isinstance(value, list) and len(value) <= 32 and all(
        isinstance(item, (str, int, float, bool)) or item is None for item in value
    ):
        return value
    return None


def json_hints(value, prefix="", inherited=False):
    hints = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = "{}.{}".format(prefix, key).strip(".")
            if SECRET_KEY.search(str(key)):
                continue
            relevant = inherited or bool(CAP_KEY.search(str(key)))
            item = scalar(child)
            if relevant and item is not None:
                hints[path] = item
            if isinstance(child, (dict, list)):
                hints.update(json_hints(child, path, relevant))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                hints.update(json_hints(child, "{}[{}]".format(prefix, index), inherited))
    return hints


def text_hints(path):
    hints = {}
    stack = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return hints
    for line in lines:
        match = re.match(r"^(\s*)([A-Za-z_][\w.-]*):\s*([^#]*)", line)
        if not match:
            continue
        indent, key, value = len(match.group(1)), match.group(2), match.group(3).strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent_relevant = any(item[2] for item in stack)
        relevant = parent_relevant or bool(CAP_KEY.search(key))
        stack.append((indent, key, relevant))
        if relevant and value and not SECRET_KEY.search(key):
            hints[".".join(item[1] for item in stack)] = value[:160]
    return hints


def spec_snapshot(path):
    if not path:
        return None
    path = path.resolve()
    result = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    result["sha256"] = sha256(path)
    document = load_json(path)
    result["format"] = "json" if document is not None else "text"
    result["runtime_hints"] = (
        json_hints(document) if document is not None else text_hints(path)
    )
    return result


def git_snapshot(repo):
    head, head_error = run(
        ["git", "--no-optional-locks", "-C", str(repo), "rev-parse", "HEAD"]
    )
    status, status_error = run(
        [
            "git", "--no-optional-locks", "-C", str(repo), "status",
            "--porcelain=v1", "--untracked-files=no",
        ]
    )
    return {
        "repo": str(repo.resolve()),
        "head": head,
        "tracked_dirty": None if status is None else bool(status),
        "tracked_change_count": None if status is None else len(status.splitlines()),
        "errors": [error for error in (head_error, status_error) if error],
    }


def csv_rows(path):
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError):
        return []


def read_marker(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()[:80]
    except OSError:
        return ""


def duration_seconds(manifest):
    from datetime import datetime

    try:
        start = datetime.fromisoformat(str(manifest["started_at"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(manifest["completed_at"]).replace("Z", "+00:00"))
        seconds = (end - start).total_seconds()
        return seconds if seconds >= 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def batch_snapshot(root, expected_override=None, eta_workers=None):
    if not root:
        return None
    root = root.resolve()
    result = {"root": str(root), "exists": root.is_dir()}
    if not root.is_dir():
        return result

    expected_sources = {}
    summary_counts = {"completed": {}, "failed": {}, "validated": {}}
    seen_summary_files = set()
    for name in ("batch.json", "SUMMARY.json", "summary.json"):
        summary_path = (root / name).resolve()
        if not summary_path.is_file():
            continue
        stat = summary_path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen_summary_files:
            continue
        seen_summary_files.add(identity)
        document = load_json(summary_path)
        if isinstance(document, dict):
            for key in ("expected_jobs", "expected", "total_jobs"):
                value = document.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    expected_sources["{}.{}".format(name, key)] = value
            for label, keys in (
                ("completed", ("completed", "succeeded", "successful")),
                ("failed", ("failed", "failures")),
                ("validated", ("validated",)),
            ):
                for key in keys:
                    value = document.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        summary_counts[label]["{}.{}".format(name, key)] = value

    plan_counts = {}
    for name in ("plan.csv", "grid.csv"):
        path = root / name
        if path.is_file():
            plan_counts[name] = len(csv_rows(path))
            expected_sources[name] = plan_counts[name]

    metrics_paths = [root / "metrics.csv"] if (root / "metrics.csv").is_file() else []
    if not metrics_paths:
        metrics_paths = list(root.rglob("metrics.csv"))
    metric_rows = {}
    metric_completed = 0
    metric_validated = 0
    metric_failed = 0
    for path in metrics_paths:
        for index, row in enumerate(csv_rows(path)):
            key = row.get("run_tag") or row.get("job_id") or "{}:{}".format(path, index)
            metric_rows[str(key)] = row
    for row in metric_rows.values():
        validation = str(row.get("validation", "")).lower()
        status = str(row.get("status", "")).lower()
        if status in {"0", "ok", "complete", "completed", "success", "succeeded"}:
            metric_completed += 1
        if validation in {"ok", "valid", "validated", "pass", "passed"}:
            metric_validated += 1
        if status not in {"", "0", "ok", "complete", "completed", "success", "succeeded"}:
            metric_failed += 1

    exit_by_job = {}
    validation_by_job = {}
    manifests = []
    durations = []
    for directory, names, files in os.walk(root):
        names[:] = [name for name in names if name not in {"raw", ".git", "__pycache__"}]
        base = Path(directory)
        if "exitcode" in files:
            exit_by_job[str(base)] = read_marker(base / "exitcode")
        elif "runner_exitcode" in files:
            exit_by_job[str(base)] = read_marker(base / "runner_exitcode")
        if "validation" in files:
            validation_by_job[str(base)] = read_marker(base / "validation").lower()
        if "manifest.json" in files:
            document = load_json(base / "manifest.json")
            if isinstance(document, dict):
                manifests.append(document)
        if "manifest.path" in files:
            pointed = Path(read_marker(base / "manifest.path"))
            if pointed.is_file():
                document = load_json(pointed)
                if isinstance(document, dict):
                    manifests.append(document)

    for manifest in manifests:
        value = duration_seconds(manifest)
        if value is not None and manifest.get("status") == "completed":
            durations.append(value)

    exit_ok = sum(value == "0" for value in exit_by_job.values())
    exit_failed = sum(value not in {"", "0"} for value in exit_by_job.values())
    marker_validated = sum(
        value in {"ok", "valid", "validated", "pass", "passed"}
        for value in validation_by_job.values()
    )
    manifest_completed = sum(
        item.get("status") == "completed" and item.get("exit_code") == 0
        for item in manifests
    )

    if expected_override is not None:
        expected = expected_override
        expected_source = "--expected-jobs"
    elif expected_sources:
        if len(set(expected_sources.values())) > 1:
            expected_source, expected = "conflicting discovered values", None
        else:
            expected_source, expected = next(iter(expected_sources.items()))
    else:
        expected_source, expected = None, None

    completed_sources = {
        "successful_exit_markers": exit_ok,
        "completed_manifests": manifest_completed,
        "completed_metric_rows": metric_completed,
        **summary_counts["completed"],
    }
    failed_sources = {
        "failed_exit_markers": exit_failed,
        "failed_metric_rows": metric_failed,
        **summary_counts["failed"],
    }
    validated_sources = {
        "validation_markers": marker_validated,
        "validated_metric_rows": metric_validated,
        **summary_counts["validated"],
    }
    completed = max(completed_sources.values(), default=0)
    failed = max(failed_sources.values(), default=0)
    validated = max(validated_sources.values(), default=0)
    remaining = None if expected is None else max(0, expected - completed)
    eta = {"remaining_jobs": remaining, "method": "unknown"}
    if remaining == 0:
        eta.update({"seconds": 0, "method": "complete"})
    elif remaining and durations:
        workers = eta_workers or 1
        median = statistics.median(durations)
        eta.update(
            {
                "seconds": int(math.ceil(remaining / workers) * median),
                "method": "median completed duration / workers",
                "median_job_seconds": median,
                "workers": workers,
                "workers_explicit": eta_workers is not None,
            }
        )

    result.update(
        {
            "expected": expected,
            "expected_source": expected_source,
            "expected_candidates": expected_sources,
            "expected_conflict": len(set(expected_sources.values())) > 1,
            "plan_counts": plan_counts,
            "completed": completed,
            "completed_sources": completed_sources,
            "failed": failed,
            "failed_sources": failed_sources,
            "metrics": len(metric_rows),
            "validated": validated,
            "validated_sources": validated_sources,
            "manifests_seen": len(manifests),
            "eta": eta,
        }
    )
    return result


def process_snapshot(match_terms):
    output, error = run(
        ["ps", "-axo", "pid=,ppid=,etime=,state=,%cpu=,%mem=,comm=,args="]
    )
    if output is None:
        return {"matches": [], "error": error}
    excluded = {os.getpid(), os.getppid()}
    matches = []
    terms = [term for term in match_terms if term]
    for line in output.splitlines():
        parts = line.strip().split(None, 7)
        if len(parts) < 8:
            continue
        pid, ppid, elapsed, state, cpu, memory, executable, arguments = parts
        try:
            numeric_pid = int(pid)
        except ValueError:
            continue
        if numeric_pid in excluded or not terms or not any(term in arguments for term in terms):
            continue
        matches.append(
            {
                "pid": numeric_pid,
                "ppid": int(ppid),
                "elapsed": elapsed,
                "state": state,
                "cpu_percent": cpu,
                "memory_percent": memory,
                "executable": Path(executable).name,
            }
        )
    return {"matches": matches, "error": None, "full_arguments_emitted": False}


def gpu_snapshot():
    fields = "index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu"
    output, error = run(
        ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"]
    )
    if output is None:
        return {"available": False, "error": error}
    gpus = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 7:
            gpus.append(dict(zip(fields.split(","), values)))
    processes, process_error = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name",
            "--format=csv,noheader,nounits",
        ]
    )
    compute = []
    if processes:
        for line in processes.splitlines():
            values = [value.strip() for value in line.split(",", 3)]
            if len(values) == 4:
                compute.append(
                    {
                        "pid": values[0],
                        "gpu_uuid": values[1],
                        "used_memory_mib": values[2],
                        "process_name": Path(values[3]).name,
                    }
                )
    return {"available": True, "gpus": gpus, "compute_processes": compute, "process_error": process_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--match", action="append", default=[])
    parser.add_argument("--expected-jobs", type=int)
    parser.add_argument("--eta-workers", type=int)
    args = parser.parse_args()
    if args.expected_jobs is not None and args.expected_jobs < 0:
        parser.error("--expected-jobs must be non-negative")
    if args.eta_workers is not None and args.eta_workers < 1:
        parser.error("--eta-workers must be positive")

    terms = list(args.match)
    if args.batch_id:
        terms.append(args.batch_id)
    elif args.batch_root:
        terms.append(args.batch_root.name)
    snapshot = {
        "read_only": True,
        "git": git_snapshot(args.repo),
        "spec": spec_snapshot(args.spec),
        "batch": batch_snapshot(args.batch_root, args.expected_jobs, args.eta_workers),
        "processes": process_snapshot(terms),
        "gpus": gpu_snapshot(),
    }
    json.dump(snapshot, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
