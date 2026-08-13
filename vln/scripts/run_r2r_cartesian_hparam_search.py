#!/usr/bin/env python3
"""Run the model-specific, full-val R2R Cartesian TTA search.

Unlike ``run_tta_hparam_search.py``, this protocol has no screening prefix,
promotion stage, or cross-model winner.  Every Cartesian point runs all 1,021
canonical-order R2R ``val_seen`` episodes and each model/method pair freezes
its own SR-first, SPL-second winner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tta_hparam_search as staged  # noqa: E402


DEFAULT_SPEC = REPO_ROOT / "vln/experiments/r2r_modelwise_cartesian_hparam_v2.json"
LOG_ROOT = REPO_ROOT / "vln/results/logs/r2r/hparam_search"
TUNING_ROOT = REPO_ROOT / "vln/results/tuning/r2r/hparam_search"
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
RESULT_LAYOUT = "r2r_benchmark_model_method_cartesian_v1"
SETTINGS = ("duet-r2r", "hamt-r2r", "goat-r2r")
SETTING_MODEL = {
    "duet-r2r": "duet",
    "hamt-r2r": "hamt",
    "goat-r2r": "goat",
}
TTA_METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
CONTROL_METHODS = ("source", "feedtta_control")
METHODS = CONTROL_METHODS + TTA_METHODS
METHOD_SEQUENCE = METHODS


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
        or not re.match(r"^[A-Za-z0-9._-]+$", value)
    ):
        raise UserError(f"unsafe {label} path component: {value!r}")
    return value


def load_spec(path=DEFAULT_SPEC):
    path = Path(path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"invalid Cartesian spec {path}: {error}") from error
    if document.get("schema") != "navtta.vln_r2r_cartesian_hparam.v1":
        raise UserError("unsupported Cartesian search schema")
    if document.get("benchmark") != "r2r" or document.get("split") != "val_seen":
        raise UserError("Cartesian search must be R2R val_seen")
    if document.get("settings") != list(SETTINGS):
        raise UserError("Cartesian search must contain exactly the three R2R settings")
    if document.get("episode_count") != 1021 or document.get("order_seed") != 0:
        raise UserError("Cartesian search must use 1,021 episodes and order seed 0")
    if set(document.get("methods", {})) != set(TTA_METHODS):
        raise UserError("Cartesian search method set mismatch")
    if document.get("selection", {}).get("primary") != "SR":
        raise UserError("Cartesian selection primary metric must be SR")
    if document.get("selection", {}).get("secondary") != "SPL":
        raise UserError("Cartesian selection secondary metric must be SPL")
    if document.get("selection", {}).get("source_protocol") != "standard_argmax":
        raise UserError("Cartesian results must use standard argmax Source")
    document["setting_episode_counts"] = {
        setting: document["episode_count"] for setting in SETTINGS
    }
    expected_total = 0
    for method in TTA_METHODS:
        count = len(expand_parameters(method, document))
        expected = int(document["methods"][method]["expected_candidates_per_setting"])
        if count != expected:
            raise UserError(
                f"{method} grid count mismatch: expected {expected}, expanded {count}"
            )
        expected_total += count * len(SETTINGS)
    if expected_total != int(document.get("expected_tta_jobs", -1)):
        raise UserError("Cartesian total TTA job count mismatch")
    document["_path"] = str(path)
    document["_sha256"] = sha256(path)
    return document


def product(mapping):
    keys = list(mapping)
    for values in itertools.product(*(mapping[key] for key in keys)):
        yield dict(zip(keys, values))


def expand_parameters(method, spec):
    method_spec = spec["methods"][method]
    grid = method_spec["grid"]
    fixed = method_spec["fixed"]
    points = []
    for raw in product(grid):
        point = dict(raw)
        if method == "eam":
            point["memory_size"], point["batch_size"] = point.pop("memory_batch")
        elif method == "feedtta":
            profile = point.pop("sgr_profile")
            point["p"] = profile["p"]
            point["alpha"] = profile["alpha"]
        elif method == "atena":
            point["lr_query"], point["lr_self"] = point.pop("lr_pair")
        point.update(fixed)
        points.append(point)
    unique = []
    seen = set()
    for point in points:
        key = canonical(point)
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return unique


def control_parameters(method):
    if method == "source":
        return [{"action_selection": "argmax", "action_seed": 0}]
    if method == "feedtta_control":
        return [{"action_selection": "sample", "action_seed": 0}]
    raise UserError(f"unknown control method {method}")


def method_parameters(method, spec):
    return control_parameters(method) if method in CONTROL_METHODS else expand_parameters(method, spec)


def point_digest(config_method, parameters):
    value = {"method": config_method, "parameters": parameters}
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:10]


def method_root(batch_id, method):
    safe_component("batch_id", batch_id)
    safe_component("method", method)
    return LOG_ROOT / batch_id / method


def tuning_job_parent(batch_id, model, method):
    for label, value in (("batch_id", batch_id), ("model", model), ("method", method)):
        safe_component(label, value)
    return TUNING_ROOT / batch_id / model / method / "jobs"


def tuning_result_root(batch_id, model, method, run_tag):
    safe_component("run_tag", run_tag)
    return tuning_job_parent(batch_id, model, method) / run_tag / "val_seen"


def config_method(method):
    return "source" if method in CONTROL_METHODS else method


def build_jobs(method, batch_id, spec, gpu=0):
    points = method_parameters(method, spec)
    expected = (
        1 if method in CONTROL_METHODS
        else int(spec["methods"][method]["expected_candidates_per_setting"])
    )
    if len(points) != expected:
        raise UserError(f"{method} expanded to {len(points)} points, expected {expected}")
    root = method_root(batch_id, method)
    jobs = []
    ordinal = 0
    for point_index, parameters in enumerate(points):
        for setting in SETTINGS:
            model = SETTING_MODEL[setting]
            executed_method = config_method(method)
            digest = point_digest(executed_method, parameters)
            run_tag = (
                f"{batch_id}-{method}-cartesian-{point_index:04d}-"
                f"{setting}-{digest}"
            )
            job_dir = root / "jobs" / setting / run_tag
            config_path = job_dir / "parameters.json"
            result_parent = tuning_job_parent(batch_id, model, method)
            result_root = result_parent / run_tag / "val_seen"
            command = [
                str(RUNNER), setting, "val_seen", str(gpu),
                "--run-tag", run_tag,
                "--tta-config", str(config_path),
                "--result-root", str(result_root),
            ]
            jobs.append({
                "batch_id": batch_id,
                "ordinal": ordinal,
                "point_index": point_index,
                "base_run_tag": run_tag,
                "run_tag": run_tag,
                "attempt": 0,
                "setting": setting,
                "model": model,
                "family": "discrete",
                "benchmark": "r2r",
                "search_method": method,
                "config_method": executed_method,
                "result_layout": RESULT_LAYOUT,
                "result_namespace": method,
                "stage": "cartesian",
                # Scheduler metadata retains a phase label, but the runtime
                # config intentionally does not claim a legacy search stage.
                "config_stage": None,
                "episodes": -1,
                "order_seed": None,
                "parameters": dict(parameters),
                "parent_run_tags": [],
                "config_path": str(config_path),
                "job_dir": str(job_dir),
                "result_root": str(result_root),
                "retry_result_root_parent": str(result_parent),
                "command": command,
            })
            ordinal += 1
    return jobs


def plan_manifest(method, batch_id, jobs, spec):
    return {
        "schema": "navtta.vln_r2r_cartesian_plan.v1",
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "method": method,
        "benchmark": "r2r",
        "split": "val_seen",
        "episode_count": 1021,
        "order_seed": 0,
        "settings": list(SETTINGS),
        "candidate_count_per_setting": len(jobs) // len(SETTINGS),
        "job_count": len(jobs),
        "git_commit": git("rev-parse", "HEAD"),
        "spec_path": spec["_path"],
        "spec_sha256": spec["_sha256"],
        "result_layout": RESULT_LAYOUT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def validate_persisted_jobs(method, batch_id, jobs, spec):
    expected = build_jobs(method, batch_id, spec, gpu=0)
    if len(jobs) != len(expected):
        raise UserError(f"persisted {method} job count mismatch")
    for actual, planned in zip(jobs, expected):
        for key in (
            "batch_id", "ordinal", "point_index", "base_run_tag", "setting",
            "model", "benchmark", "search_method", "config_method", "stage",
            "config_stage", "episodes", "parameters", "result_layout",
            "result_namespace",
        ):
            if actual.get(key) != planned.get(key):
                raise UserError(
                    f"persisted {method} job {actual.get('ordinal')} mismatch for {key}"
                )
        if Path(actual.get("job_dir", "")) != Path(planned["job_dir"]):
            raise UserError("persisted Cartesian job_dir mismatch")
        if Path(actual.get("retry_result_root_parent", "")) != Path(
            planned["retry_result_root_parent"]
        ):
            raise UserError("persisted Cartesian retry root mismatch")


def ensure_plan(method, batch_id, spec, gpu, resume):
    root = method_root(batch_id, method)
    manifest_path = root / "GRID.json"
    if manifest_path.is_file():
        if not resume:
            raise UserError(f"{method} plan exists; use --resume")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_checks = {
            "schema": "navtta.vln_r2r_cartesian_plan.v1",
            "batch_id": batch_id,
            "method": method,
            "git_commit": git("rev-parse", "HEAD"),
            "spec_sha256": spec["_sha256"],
            "result_layout": RESULT_LAYOUT,
        }
        for key, value in expected_checks.items():
            if manifest.get(key) != value:
                raise UserError(f"persisted {method} GRID.json mismatch for {key}")
        jobs = staged.load_jobs(root)
        validate_persisted_jobs(method, batch_id, jobs, spec)
        return root, jobs
    if root.exists() and any(root.iterdir()):
        raise UserError(f"nonempty Cartesian method root without GRID.json: {root}")
    root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(method, batch_id, spec, gpu=gpu)
    staged.write_plan(root, jobs)
    staged.atomic_json(manifest_path, plan_manifest(method, batch_id, jobs, spec))
    return root, jobs


def load_validated_results(root, spec):
    jobs = staged.load_jobs(root)
    results, errors = staged.write_summary(root, jobs, spec)
    if len(results) + len(errors) != len(jobs) or errors:
        raise UserError(f"Cartesian method is incomplete: {root}")
    return results


def source_by_setting(batch_id, spec):
    root = method_root(batch_id, "source")
    results = load_validated_results(root, spec)
    output = {result["setting"]: result for result in results}
    if set(output) != set(SETTINGS) or len(results) != len(SETTINGS):
        raise UserError("standard Source evidence is incomplete")
    for result in results:
        if result["parameters"].get("action_selection") != "argmax":
            raise UserError("standard Source control is not argmax")
    return output


def result_score(result):
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


def summarize_method(method, batch_id, spec):
    if method not in TTA_METHODS:
        return None
    root = method_root(batch_id, method)
    results = load_validated_results(root, spec)
    sources = source_by_setting(batch_id, spec)
    expected = int(spec["methods"][method]["expected_candidates_per_setting"])
    document = {
        "schema": "navtta.vln_r2r_cartesian_winners.v1",
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "method": method,
        "benchmark": "r2r",
        "split": "val_seen",
        "selection": ["SR", "SPL", "lower_parameter_drift", "fewer_updates"],
        "source_protocol": "standard_argmax",
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": spec["_sha256"],
        "settings": {},
    }
    top_rows = []
    for setting in SETTINGS:
        candidates = [result for result in results if result["setting"] == setting]
        if len(candidates) != expected:
            raise UserError(
                f"{method}/{setting} has {len(candidates)} candidates; expected {expected}"
            )
        candidates.sort(key=result_score, reverse=True)
        source = sources[setting]
        winner = candidates[0]
        document["settings"][setting] = {
            "winner_run_tag": winner["run_tag"],
            "winner_parameters": winner["parameters"],
            "winner_metrics": winner["metrics"],
            "source_run_tag": source["run_tag"],
            "source_metrics": source["metrics"],
            "delta_sr_pp": winner["metrics"]["SR"] - source["metrics"]["SR"],
            "delta_spl_pp": winner["metrics"]["SPL"] - source["metrics"]["SPL"],
        }
        for rank, candidate in enumerate(candidates[:5], start=1):
            adapter = candidate.get("adapter_diagnostics") or {}
            top_rows.append({
                "setting": setting,
                "rank": rank,
                "run_tag": candidate["run_tag"],
                "sr": candidate["metrics"]["SR"],
                "spl": candidate["metrics"]["SPL"],
                "source_sr": source["metrics"]["SR"],
                "source_spl": source["metrics"]["SPL"],
                "delta_sr_pp": candidate["metrics"]["SR"] - source["metrics"]["SR"],
                "delta_spl_pp": candidate["metrics"]["SPL"] - source["metrics"]["SPL"],
                "relative_param_drift": adapter.get("relative_param_drift"),
                "updates": adapter.get("updates"),
                "parameters": canonical(candidate["parameters"]),
            })
    staged.atomic_json(root / "WINNER.json", document)
    with (root / "TOP5.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(top_rows[0]))
        writer.writeheader()
        writer.writerows(top_rows)
    return document


def aggregate_campaign(batch_id, spec):
    winners = {}
    top5_rows = []
    for method in TTA_METHODS:
        root = method_root(batch_id, method)
        winner_path = root / "WINNER.json"
        if not winner_path.is_file():
            return None
        document = json.loads(winner_path.read_text(encoding="utf-8"))
        winners[method] = document["settings"]
        with (root / "TOP5.csv").open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                top5_rows.append({"method": method, **row})
    root = LOG_ROOT / batch_id
    output = {
        "schema": "navtta.vln_r2r_cartesian_campaign_winners.v1",
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "benchmark": "r2r",
        "split": "val_seen",
        "source_protocol": "standard_argmax",
        "selection": ["SR", "SPL", "lower_parameter_drift", "fewer_updates"],
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": spec["_sha256"],
        "methods": winners,
    }
    staged.atomic_json(root / "WINNERS.json", output)
    frozen = {
        "schema": "navtta.vln_r2r_cartesian_frozen_hparams.v1",
        "generated_from": "WINNERS.json",
        "batch_id": batch_id,
        "git_commit": output["git_commit"],
        "spec_sha256": output["spec_sha256"],
        "settings": {
            setting: {
                method: winners[method][setting]["winner_parameters"]
                for method in TTA_METHODS
            }
            for setting in SETTINGS
        },
    }
    staged.atomic_json(root / "FROZEN_HPARAMETERS.json", frozen)
    with (root / "TOP5.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(top5_rows[0]))
        writer.writeheader()
        writer.writerows(top5_rows)
    return output


def runtime_args(cli, method, spec):
    defaults = spec["scheduler_defaults"]
    method_defaults = defaults[method]
    max_workers = cli.max_workers or int(method_defaults["max_workers"])
    max_per_model = cli.max_per_model or int(method_defaults["max_per_model"])
    return argparse.Namespace(
        method=method,
        batch_id=cli.batch_id,
        settings=list(SETTINGS),
        max_workers=max_workers,
        max_per_model=max_per_model,
        max_discrete_workers=max_workers,
        max_continuous_workers=1,
        max_gpu_memory_mib=(
            cli.max_gpu_memory_mib
            if cli.max_gpu_memory_mib is not None
            else int(defaults["max_gpu_memory_mib_before_launch"])
        ),
        max_memory_gib=(
            cli.max_memory_gib
            if cli.max_memory_gib is not None
            else float(defaults["max_cgroup_memory_gib_before_launch"])
        ),
        launch_stagger=(
            cli.launch_stagger
            if cli.launch_stagger is not None
            else float(defaults["launch_stagger_seconds"])
        ),
        resource_wait_timeout=(
            cli.resource_wait_timeout
            if cli.resource_wait_timeout is not None
            else float(defaults["resource_wait_timeout_seconds"])
        ),
        resume=cli.resume,
        retry_failed=cli.retry_failed,
        fail_fast=cli.fail_fast,
    )


def execute_method(cli, method, spec):
    args = runtime_args(cli, method, spec)
    root, jobs = ensure_plan(method, cli.batch_id, spec, cli.gpu, cli.resume)
    if cli.plan_only:
        print(f"method={method} jobs={len(jobs)} root={root}")
        if cli.print_commands:
            for job in jobs:
                print(subprocess.list2cmdline(job["command"]))
        return
    staged.run_batch(args, root, jobs, spec)
    if method in TTA_METHODS:
        summarize_method(method, cli.batch_id, spec)


def status_snapshot(batch_id):
    root = LOG_ROOT / batch_id
    snapshot = {"batch_id": batch_id, "methods": {}, "totals": {}}
    totals = {key: 0 for key in ("planned", "pending", "running", "succeeded", "failed")}
    for method in METHOD_SEQUENCE:
        progress_path = root / method / "progress.json"
        if progress_path.is_file():
            value = json.loads(progress_path.read_text(encoding="utf-8"))
        else:
            value = {key: 0 for key in totals}
        snapshot["methods"][method] = value
        for key in totals:
            totals[key] += int(value.get(key, 0))
    snapshot["totals"] = totals
    snapshot["complete"] = totals["planned"] == 2379 and (
        totals["succeeded"] == totals["planned"] and totals["failed"] == 0
    )
    return snapshot


def show_status(batch_id, watch=False):
    while True:
        snapshot = status_snapshot(batch_id)
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        if not watch or snapshot["complete"]:
            return
        time.sleep(10)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS + ("all",))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-per-model", type=int)
    parser.add_argument("--max-gpu-memory-mib", type=int)
    parser.add_argument("--max-memory-gib", type=float)
    parser.add_argument("--launch-stagger", type=float)
    parser.add_argument("--resource-wait-timeout", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)
    safe_component("batch_id", args.batch_id)
    if args.gpu < 0:
        parser.error("gpu must be non-negative")
    for label in ("max_workers", "max_per_model", "max_gpu_memory_mib"):
        value = getattr(args, label)
        if value is not None and value < 1:
            parser.error(f"{label.replace('_', '-')} must be positive")
    if args.max_memory_gib is not None and args.max_memory_gib <= 0:
        parser.error("max-memory-gib must be positive")
    if args.launch_stagger is not None and args.launch_stagger < 0:
        parser.error("launch-stagger must be non-negative")
    if args.resource_wait_timeout is not None and args.resource_wait_timeout <= 0:
        parser.error("resource-wait-timeout must be positive")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if args.watch:
        args.status = True
    return args


def main(argv=None):
    cli = parse_args(argv)
    if cli.status:
        show_status(cli.batch_id, watch=cli.watch)
        return
    spec = load_spec(cli.spec)
    if git("status", "--porcelain", "--untracked-files=no") and not cli.plan_only:
        raise UserError("tracked worktree must be clean before launch")
    methods = METHOD_SEQUENCE if cli.method == "all" else (cli.method,)
    failures = []
    for method in methods:
        try:
            execute_method(cli, method, spec)
        except (UserError, staged.UserError) as error:
            failures.append((method, str(error)))
            if method == "source" or cli.method != "all":
                raise
            print(f"method_error method={method} error={error}", file=sys.stderr)
    if not cli.plan_only:
        aggregate_campaign(cli.batch_id, spec)
    if failures:
        raise UserError(
            "; ".join(f"{method}: {error}" for method, error in failures)
        )


if __name__ == "__main__":
    try:
        main()
    except (UserError, staged.UserError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
