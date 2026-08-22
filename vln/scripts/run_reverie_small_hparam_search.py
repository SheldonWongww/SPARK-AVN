#!/usr/bin/env python3
"""Run the bounded REVERIE val_seen follow-up search.

The campaign deliberately reuses the authenticated Source controls and the
15 completed R2R-transfer incumbents.  New points first run on the canonical
256-episode prefix.  At most one point per targeted model-method cell is then
promoted to a fresh full-split process, so no optimizer/model state can leak
from screening into the formal run.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for entry in (str(REPO_ROOT), str(SCRIPT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import run_reverie_frozen_transfer as frozen_runner  # noqa: E402
from shared_gpu_launch_guard import (  # noqa: E402
    ReservationLedgerError,
    release_shared_gpu_reservation,
    shared_gpu_launch_guard,
)
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


SCHEMA = "navtta.vln_reverie_val_seen_small_hparam_search.v1"
DEFAULT_SPEC = (
    REPO_ROOT
    / "vln/experiments/reverie_val_seen_small_hparam_search_v1.json"
)
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
LOG_ROOT = REPO_ROOT / "vln/results/logs/reverie/hparam_search"
RESULT_ROOT = REPO_ROOT / "vln/results/tuning/reverie/hparam_search"
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"
INCUMBENT_RESULT_ROOT = REPO_ROOT / "vln/results/tuning/reverie/frozen_transfer"
EXECUTION_SURFACE = (
    "core",
    "tools",
    "vln/baselines",
    "vln/navtta_vln",
    "vln/scripts",
    "vln/experiments",
    "vln/manifests",
)

SETTINGS = ("duet-reverie", "hamt-reverie", "goat-reverie")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
MODEL_FOR_SETTING = {
    "duet-reverie": "duet",
    "hamt-reverie": "hamt",
    "goat-reverie": "goat",
}
EXPECTED_FEEDBACK_ENDPOINT = {
    "feedtta": (
        "reverie_submitted_trajectory_evaluator_navigation_"
        "success_every_episode"
    ),
    "atena": (
        "reverie_submitted_trajectory_evaluator_navigation_"
        "success_lazy_query"
    ),
}


class UserError(RuntimeError):
    pass


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("cannot read JSON {}: {}".format(path, error))


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


def _repo_file(value, label, required_root=None):
    path = Path(value)
    path = path if path.is_absolute() else REPO_ROOT / path
    path = path.resolve()
    root = (required_root or REPO_ROOT).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise UserError("{} escapes {}: {}".format(label, root, path))
    if not path.is_file():
        raise UserError("missing {}: {}".format(label, path))
    return path


def _valid_sha256(value):
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None


def _parse_metric_artifact(path):
    matches = []
    for line in Path(path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        values = {
            key.upper(): float(value)
            for key, value in frozen_runner.METRIC_RE.findall(line)
        }
        if set(values) == {"SR", "SPL", "RGS", "RGSPL"}:
            matches.append(values)
    if len(matches) != 1:
        raise UserError(
            "expected exactly one authenticated REVERIE metric line in {}; "
            "found {}".format(path, len(matches))
        )
    return matches[0]


def _assert_metric_literal(label, parsed, literal):
    if set(literal or {}) != {"SR", "SPL", "RGS", "RGSPL"}:
        raise UserError("{} metric literal is incomplete".format(label))
    mismatches = [
        key for key in ("SR", "SPL", "RGS", "RGSPL")
        if float(parsed[key]) != float(literal[key])
    ]
    if mismatches:
        raise UserError(
            "{} metric literal differs from authenticated artifact: {}".format(
                label, ", ".join(mismatches)
            )
        )


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


def _untracked_execution_files():
    completed = subprocess.run(
        [
            "git", "-C", str(REPO_ROOT), "ls-files", "--others",
            "--exclude-standard", "--", *EXECUTION_SURFACE,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise UserError("cannot inspect untracked execution files: {}".format(
            completed.stderr.strip()
        ))
    return [line for line in completed.stdout.splitlines() if line]


def _require_tracked_launch_inputs(spec_path):
    relative_paths = []
    for label, path in (
        ("runner", Path(__file__).resolve()),
        ("spec", Path(spec_path).resolve()),
    ):
        try:
            relative_paths.append(str(path.relative_to(REPO_ROOT.resolve())))
        except ValueError:
            raise UserError("{} must be inside the repository: {}".format(
                label, path
            ))
    completed = subprocess.run(
        [
            "git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch",
            "--", *relative_paths,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise UserError(
            "formal launch requires runner and spec to be tracked by current "
            "HEAD: {}".format(", ".join(relative_paths))
        )


def _assert_clean_execution_tree(spec_path):
    if _tracked_worktree_dirty():
        raise UserError("tracked worktree must be clean before execution")
    _require_tracked_launch_inputs(spec_path)
    untracked = _untracked_execution_files()
    if untracked:
        raise UserError(
            "formal launch refuses untracked execution files: {}".format(
                ", ".join(untracked)
            )
        )


def _atomic_json(path, value):
    frozen_runner._atomic_json(path, value)


def _formal_manifest_path(run_tag, setting):
    return (
        FORMAL_ROOT
        / "{}-{}-val_seen-native".format(run_tag, setting)
        / "manifest.json"
    )


def _candidate_parameters(method_spec, candidate):
    fixed = method_spec.get("fixed", {})
    overlap = set(fixed).intersection(candidate)
    conflict = [key for key in overlap if fixed[key] != candidate[key]]
    if conflict:
        raise UserError(
            "candidate overrides fixed keys: {}".format(", ".join(conflict))
        )
    parameters = {
        key: value for key, value in candidate.items() if key != "role"
    }
    parameters.update(fixed)
    return parameters


def _incumbent_parameters(spec):
    dependency = spec["dependencies"]["incumbent_spec"]
    path = _repo_file(dependency["path"], "incumbent transfer spec")
    if _sha256(path) != dependency["sha256"]:
        raise UserError("incumbent transfer spec SHA256 mismatch")
    document = _read_json(path)
    jobs = document.get("jobs", [])
    values = {}
    for job in jobs:
        key = (job.get("setting"), job.get("method"))
        if key in values:
            raise UserError("duplicate incumbent parameters for {}".format(key))
        values[key] = job.get("parameters")
    expected = {(setting, method) for setting in SETTINGS for method in METHODS}
    if set(values) != expected or not all(
        isinstance(value, dict) and value for value in values.values()
    ):
        raise UserError("incumbent transfer spec is not a complete 3x5 matrix")
    return values


def _source_registry(spec):
    dependency = spec["dependencies"]["source_registry"]
    proxy = {
        "source_control": {
            "manifest": dependency["path"],
            "sha256": dependency["sha256"],
        }
    }
    try:
        _, source = frozen_runner._validate_source_registry(
            proxy, require_metrics_artifacts=True
        )
    except frozen_runner.UserError as error:
        raise UserError(str(error))
    for setting, record in source["records"].items():
        metric_path = Path(record["metrics_artifact_path"])
        if not metric_path.is_absolute():
            metric_path = REPO_ROOT / metric_path
        parsed = _parse_metric_artifact(metric_path)
        _assert_metric_literal(
            "{} Source".format(setting), parsed, record.get("metrics")
        )
        # Downstream selection consumes the authenticated parse, not the
        # duplicated registry literal.
        record["metrics"] = parsed
    return source


def _validate_incumbents(spec, source, parameters):
    expected = {(setting, method) for setting in SETTINGS for method in METHODS}
    actual = {
        (setting, method)
        for setting, methods in spec.get("incumbents", {}).items()
        for method in methods
    }
    if actual != expected:
        raise UserError("incumbents must be the complete 3x5 transfer matrix")
    expected_commit = spec["dependencies"]["incumbent_git_commit"]
    authenticated = {}
    for setting, method in sorted(expected):
        record = spec["incumbents"][setting][method]
        run_tag = record.get("run_tag")
        path = _formal_manifest_path(run_tag, setting)
        if not path.is_file():
            raise UserError("missing incumbent formal manifest: {}".format(path))
        if _sha256(path) != record.get("formal_manifest_sha256"):
            raise UserError("{} {} incumbent manifest SHA256 mismatch".format(
                setting, method
            ))
        manifest = _read_json(path)
        checks = {
            "task": "vln",
            "benchmark": (
                "reverie_discrete_goat"
                if setting == "goat-reverie"
                else "reverie_discrete_duet_hamt"
            ),
            "model": MODEL_FOR_SETTING[setting],
            "method": method,
            "run_tag": run_tag,
            "seed": 0,
            "git_commit": expected_commit,
            "status": "completed",
            "exit_code": 0,
        }
        for key, expected_value in checks.items():
            if manifest.get(key) != expected_value:
                raise UserError(
                    "{} {} incumbent {} mismatch".format(setting, method, key)
                )
        identity = manifest.get("immutable_identity_sha256")
        if not _valid_sha256(identity) or (
            immutable_identity_sha256(manifest) != identity
        ):
            raise UserError("{} {} incumbent identity mismatch".format(
                setting, method
            ))
        source_record = source["records"][setting]
        if manifest.get("checkpoint", {}).get("sha256") != source_record[
            "checkpoint_sha256"
        ]:
            raise UserError("{} {} incumbent checkpoint mismatch".format(
                setting, method
            ))
        dataset = manifest.get("dataset", {})
        if (
            dataset.get("stream_content_sha256")
            != source_record["dataset_sha256"]
            or dataset.get("stream_order_sha256")
            != source["episode_order_sha256"]
        ):
            raise UserError("{} {} incumbent dataset/order mismatch".format(
                setting, method
            ))
        metrics = record.get("metrics")
        if set(metrics or {}) != {"SR", "SPL", "RGS", "RGSPL"}:
            raise UserError("{} {} incumbent metrics are incomplete".format(
                setting, method
            ))
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 100.0
            for value in metrics.values()
        ):
            raise UserError("{} {} incumbent metrics are invalid".format(
                setting, method
            ))
        if not isinstance(parameters[(setting, method)], dict):
            raise UserError("missing incumbent parameters")
        metric_artifacts = [
            artifact for artifact in manifest.get("result_artifacts", [])
            if isinstance(artifact, dict)
            and str(artifact.get("name", "")).endswith("valid.txt")
        ]
        if len(metric_artifacts) != 1:
            raise UserError("{} {} incumbent has no unique metrics artifact".format(
                setting, method
            ))
        artifact = metric_artifacts[0]
        relative = Path(str(artifact["name"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise UserError("incumbent metrics artifact has unsafe name")
        result_root = (
            INCUMBENT_RESULT_ROOT
            / spec["dependencies"]["incumbent_batch_id"]
            / MODEL_FOR_SETTING[setting]
            / method
            / "jobs"
            / run_tag
            / "val_seen"
        ).resolve()
        metric_path = (result_root / relative).resolve()
        try:
            metric_path.relative_to(result_root)
        except ValueError:
            raise UserError("incumbent metrics artifact escapes result root")
        if not metric_path.is_file():
            raise UserError("missing incumbent metrics artifact: {}".format(
                metric_path
            ))
        if (
            metric_path.stat().st_size != artifact.get("size")
            or _sha256(metric_path) != artifact.get("sha256")
        ):
            raise UserError("{} {} incumbent metrics artifact mismatch".format(
                setting, method
            ))
        parsed = _parse_metric_artifact(metric_path)
        _assert_metric_literal(
            "{} {} incumbent".format(setting, method), parsed, metrics
        )
        authenticated[(setting, method)] = {
            **record,
            "parameters": parameters[(setting, method)],
            "metrics": parsed,
            "formal_manifest": str(path),
            "metric_artifact": str(metric_path),
            "metric_artifact_sha256": artifact["sha256"],
        }
    return authenticated


def load_spec(path=DEFAULT_SPEC, require_evidence=True):
    path = Path(path).resolve()
    spec = _read_json(path)
    if spec.get("schema") != SCHEMA:
        raise UserError("unsupported REVERIE small-search schema")
    protocol = spec.get("protocol", {})
    checks = {
        "benchmark": "reverie",
        "split": "val_seen",
        "episode_count": 1423,
        "screening_episodes": 256,
        "order_seed": 0,
        "model_seed": 0,
        "source_execution": "reuse_only",
        "source_rerun_forbidden": True,
        "screening_prefix_is_selection_only": True,
        "full_candidate_restarts_from_source_checkpoint": True,
        "one_new_full_candidate_per_target_cell": True,
        "no_val_unseen_or_test_during_search": True,
        "order_seed_cli_forbidden": True,
        "action_protocol": "target_native_argmax",
    }
    for key, expected in checks.items():
        if protocol.get(key) != expected:
            raise UserError("protocol.{} must be {!r}".format(key, expected))
    if tuple(protocol.get("settings", ())) != SETTINGS:
        raise UserError("setting order must remain DUET -> HAMT -> GOAT")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise UserError("method set/order changed")
    if protocol.get("reported_metrics") != ["SR", "SPL", "RGS", "RGSPL"]:
        raise UserError("REVERIE must report SR/SPL/RGS/RGSPL")
    if protocol.get("screening_primary_metric") != "RGSPL" or (
        protocol.get("final_primary_metric") != "RGSPL"
    ):
        raise UserError("REVERIE selection must be grounding-aware (RGSPL)")
    feedback = protocol.get("binary_feedback", {})
    if (
        feedback.get("label") != "navigation_success_only"
        or not feedback.get("grounding_feedback_forbidden")
        or not feedback.get("simulator_distance_fallback_forbidden")
        or feedback.get("feedtta_timing") != "eager_once_per_episode"
        or feedback.get("atena_timing")
        != "lazy_only_when_entropy_gate_queries"
    ):
        raise UserError("binary-feedback contract is incomplete")
    groups = protocol.get("supervision_groups", {})
    if groups != {
        "unsupervised": ["tent", "fstta", "eam"],
        "binary_feedback_supervised": ["feedtta", "atena"],
    }:
        raise UserError("supervision groups changed")

    execution = spec.get("execution", {})
    if tuple(execution.get("setting_order", ())) != SETTINGS:
        raise UserError("execution setting order changed")
    if not execution.get("strict_model_barriers_required") or not execution.get(
        "parallel_methods_within_model"
    ):
        raise UserError("model barriers and within-model method parallelism are required")
    if execution.get("model_phase_pipeline") != [
        "screening", "promotion", "full", "summary"
    ]:
        raise UserError("each model must finish screening through full before the next")
    if execution.get("max_workers_per_method") != 1:
        raise UserError("at most one candidate per method may run concurrently")
    caps = execution.get("max_workers_by_setting", {})
    if set(caps) != set(SETTINGS) or any(
        not isinstance(value, int) or value < 1 for value in caps.values()
    ):
        raise UserError("invalid per-model worker caps")

    actual_candidates = 0
    target_cells = set()
    seen_parameters = set()
    for method in METHODS:
        method_spec = spec.get("methods", {}).get(method)
        if not isinstance(method_spec, dict):
            raise UserError("missing method specification: {}".format(method))
        expected_supervision = (
            "binary_episode_navigation_feedback"
            if method == "feedtta"
            else "active_binary_episode_navigation_feedback"
            if method == "atena"
            else "unsupervised"
        )
        if method_spec.get("supervision") != expected_supervision:
            raise UserError("{} supervision label mismatch".format(method))
        by_setting = method_spec.get("candidates_by_setting", {})
        if not isinstance(by_setting, dict) or not set(by_setting).issubset(SETTINGS):
            raise UserError("{} has invalid candidate settings".format(method))
        for setting, candidates in by_setting.items():
            if not isinstance(candidates, list) or not candidates:
                raise UserError("{} {} candidate list is empty".format(
                    setting, method
                ))
            target_cells.add((setting, method))
            for candidate in candidates:
                if not isinstance(candidate, dict) or not candidate.get("role"):
                    raise UserError("every candidate requires a diagnostic role")
                parameters = _candidate_parameters(method_spec, candidate)
                key = (setting, method, _canonical(parameters))
                if key in seen_parameters:
                    raise UserError("duplicate candidate in {} {}".format(
                        setting, method
                    ))
                seen_parameters.add(key)
                actual_candidates += 1
                if method == "tent" and parameters.get("update_interval") != 1:
                    raise UserError("Tent update_interval must remain fixed at 1")
                if method in ("feedtta", "atena") and parameters.get(
                    "action_selection"
                ) != "argmax":
                    raise UserError("feedback methods must use target-native argmax")
                if method == "feedtta" and parameters.get("sgr_seed") != 0:
                    raise UserError("FeedTTA sgr_seed must remain 0")

    budget = spec.get("budget", {})
    expected_budget = {
        "screening_candidates": actual_candidates,
        "screening_episode_evaluations": (
            actual_candidates * protocol["screening_episodes"]
        ),
        "maximum_new_full_runs": len(target_cells),
        "maximum_full_episode_evaluations": (
            len(target_cells) * protocol["episode_count"]
        ),
        "maximum_new_jobs": actual_candidates + len(target_cells),
        "maximum_new_episode_evaluations": (
            actual_candidates * protocol["screening_episodes"]
            + len(target_cells) * protocol["episode_count"]
        ),
        "source_jobs": 0,
        "incumbent_reruns": 0,
        "target_cells": len(target_cells),
        "untouched_incumbent_cells": len(SETTINGS) * len(METHODS) - len(target_cells),
    }
    for key, expected in expected_budget.items():
        if budget.get(key) != expected:
            raise UserError("budget.{} must be {}".format(key, expected))
    if budget["maximum_new_jobs"] >= budget.get(
        "superseded_proposal_new_jobs", 0
    ):
        raise UserError("small search is not smaller than the superseded proposal")

    if require_evidence:
        source = _source_registry(spec)
        if source.get("episode_order_sha256") != protocol[
            "canonical_order_sha256"
        ]:
            raise UserError("Source and search canonical order SHA256 differ")
        parameters = _incumbent_parameters(spec)
        _validate_incumbents(spec, source, parameters)
    return spec


def _point_digest(setting, method, parameters):
    payload = [setting, method, parameters]
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:10]


def expand_screening_jobs(spec, batch_id, gpu=0):
    jobs = []
    ordinal = 0
    for setting_index, setting in enumerate(SETTINGS):
        for method_index, method in enumerate(METHODS):
            method_spec = spec["methods"][method]
            candidates = method_spec["candidates_by_setting"].get(setting, [])
            for candidate_index, candidate in enumerate(candidates):
                parameters = _candidate_parameters(method_spec, candidate)
                digest = _point_digest(setting, method, parameters)
                base_run_tag = (
                    "{}-screen-{:02d}-{:02d}-{}-{:02d}-{}-{}".format(
                        batch_id,
                        setting_index,
                        method_index,
                        method,
                        candidate_index,
                        setting,
                        digest,
                    )
                )
                jobs.append({
                    "stage": "screening",
                    "episodes": spec["protocol"]["screening_episodes"],
                    "setting": setting,
                    "model": MODEL_FOR_SETTING[setting],
                    "method": method,
                    "parameters": parameters,
                    "role": candidate["role"],
                    "setting_index": setting_index,
                    "method_index": method_index,
                    "candidate_index": candidate_index,
                    "ordinal": ordinal,
                    "base_run_tag": base_run_tag,
                    "gpu": int(gpu),
                })
                ordinal += 1
    return jobs


def expand_full_jobs(spec, batch_id, promotions, gpu=0, settings=None):
    selected_settings = tuple(settings or SETTINGS)
    if not selected_settings or not set(selected_settings).issubset(SETTINGS):
        raise UserError("full-job settings are invalid")
    promoted = {
        (item["setting"], item["method"]): item
        for item in promotions.get("promotions", [])
    }
    target_cells = {
        (setting, method)
        for method in METHODS
        for setting in spec["methods"][method]["candidates_by_setting"]
        if setting in selected_settings
    }
    if set(promoted) != target_cells:
        raise UserError("promotions do not cover exactly the targeted cells")
    jobs = []
    for setting, method in sorted(
        target_cells,
        key=lambda cell: (SETTINGS.index(cell[0]), METHODS.index(cell[1])),
    ):
        ordinal = SETTINGS.index(setting) * len(METHODS) + METHODS.index(method)
        item = promoted[(setting, method)]
        parameters = item["parameters"]
        digest = _point_digest(setting, method, parameters)
        jobs.append({
            "stage": "full",
            "episodes": -1,
            "setting": setting,
            "model": MODEL_FOR_SETTING[setting],
            "method": method,
            "parameters": parameters,
            "role": item["role"],
            "promoted_from_run_tag": item["screening_run_tag"],
            "ordinal": ordinal,
            "base_run_tag": "{}-full-{:02d}-{}-{}-{}".format(
                batch_id, ordinal, method, setting, digest
            ),
            "gpu": int(gpu),
        })
    return jobs


def _attempt_tag(base_run_tag, attempt):
    return base_run_tag if attempt == 0 else "{}-retry{}".format(
        base_run_tag, attempt
    )


def _attempt_dir(batch_root, job, attempt):
    return (
        Path(batch_root)
        / "stages"
        / job["stage"]
        / job["setting"]
        / "jobs"
        / job["base_run_tag"]
        / "attempt-{:02d}".format(attempt)
    )


def _existing_attempts(batch_root, job):
    parent = _attempt_dir(batch_root, job, 0).parent
    if not parent.is_dir():
        return []
    values = []
    for path in parent.glob("attempt-[0-9][0-9]"):
        try:
            values.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return sorted(values)


def _latest_attempt(batch_root, job):
    attempts = _existing_attempts(batch_root, job)
    return attempts[-1] if attempts else None


def _reservation_token(batch_id, run_tag):
    return "reverie-small-search:{}:{}".format(batch_id, run_tag)


def materialize_attempt(spec, spec_path, batch_id, batch_root, job, attempt):
    # Re-read the immutable batch binding immediately before creating every
    # attempt.  A long campaign must fail closed if HEAD changes between model
    # phases (for example, because somebody pulls on the execution server).
    _assert_execution_invariants(
        spec_path, spec, batch_id, batch_root, job["gpu"]
    )
    run_tag = _attempt_tag(job["base_run_tag"], attempt)
    attempt_dir = _attempt_dir(batch_root, job, attempt)
    result_root = (
        RESULT_ROOT
        / batch_id
        / job["model"]
        / job["method"]
        / job["stage"]
        / run_tag
        / "val_seen"
    )
    config = {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": batch_id,
        "stage": job["stage"],
        "setting": job["setting"],
        "method": job["method"],
        "search_method": job["method"],
        "episodes": job["episodes"],
        "parameters": job["parameters"],
        "selection_provenance": {
            "candidate_role": job["role"],
            "screening_run_tag": job.get("promoted_from_run_tag"),
            "fresh_source_restart": True,
        },
    }
    config_path = attempt_dir / "parameters.json"
    command = [
        str(RUNNER), job["setting"], "val_seen", str(job["gpu"]),
        "--run-tag", run_tag,
        "--tta-config", str(config_path),
        "--result-root", str(result_root),
    ]
    if job["episodes"] > 0:
        command.extend(["--episode-limit", str(job["episodes"])])
    source = _source_registry(spec)
    source_record = source["records"][job["setting"]]
    formal_manifest = (
        str(_formal_manifest_path(run_tag, job["setting"]))
        if job["stage"] == "full" else None
    )
    metadata = {
        "schema": "navtta.vln_reverie_small_search_job.v1",
        "batch_id": batch_id,
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "attempt": attempt,
        "run_tag": run_tag,
        "base_run_tag": job["base_run_tag"],
        "stage": job["stage"],
        "ordinal": job["ordinal"],
        "setting": job["setting"],
        "model": job["model"],
        "method": job["method"],
        "role": job["role"],
        "promoted_from_run_tag": job.get("promoted_from_run_tag"),
        "parameters": job["parameters"],
        "episode_count": (
            job["episodes"] if job["episodes"] > 0
            else spec["protocol"]["episode_count"]
        ),
        "git_commit": _git_commit(),
        "expected_benchmark": (
            "reverie_discrete_goat"
            if job["setting"] == "goat-reverie"
            else "reverie_discrete_duet_hamt"
        ),
        "expected_checkpoint_sha256": source_record["checkpoint_sha256"],
        "expected_dataset_sha256": source_record["dataset_sha256"],
        "expected_episode_order_sha256": source["episode_order_sha256"],
        "result_root": str(result_root),
        "formal_manifest": formal_manifest,
        "command": command,
    }
    _atomic_json(config_path, config)
    _atomic_json(attempt_dir / "job.json", metadata)
    return attempt_dir, metadata


def _count(adapter, key, expected_episodes):
    value = adapter.get(key)
    if (
        type(value) is not int
        or value < 0
        or value > expected_episodes
    ):
        raise UserError("adapter.{} must be an integer in [0, {}]".format(
            key, expected_episodes
        ))
    return value


def _validate_diagnostics(metadata, diagnostics, metrics=None):
    method = metadata["method"]
    expected_episodes = metadata["episode_count"]
    if diagnostics.get("method") != method:
        raise UserError("TTA diagnostics method mismatch")
    if diagnostics.get("episode_count") != expected_episodes:
        raise UserError("TTA diagnostics episode count mismatch")
    if diagnostics.get("action_selection") != "target_native_argmax":
        raise UserError("REVERIE search must use target-native argmax")
    adapter = diagnostics.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("episodes") != expected_episodes:
        raise UserError("adapter episode accounting mismatch")
    for key in ("updates", "relative_param_drift"):
        value = adapter.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise UserError("adapter.{} is missing or invalid".format(key))
    if method in EXPECTED_FEEDBACK_ENDPOINT:
        if diagnostics.get("supervision") != "binary_navigation_success_feedback":
            raise UserError("binary-feedback supervision label mismatch")
        endpoint = diagnostics.get("binary_feedback_endpoint")
        if endpoint != EXPECTED_FEEDBACK_ENDPOINT[method] or "distance" in endpoint:
            raise UserError("invalid REVERIE binary-feedback endpoint")
        if method == "feedtta":
            feedback = _count(adapter, "feedback_episodes", expected_episodes)
            successes = _count(
                adapter, "successful_feedback_episodes", expected_episodes
            )
            failures = _count(
                adapter, "failed_feedback_episodes", expected_episodes
            )
            if feedback != expected_episodes:
                raise UserError("FeedTTA must consume one label per episode")
            if successes + failures != expected_episodes:
                raise UserError("FeedTTA feedback accounting mismatch")
            if adapter.get("action_selection_protocol") != "target_native_argmax":
                raise UserError("FeedTTA adapter action protocol mismatch")
            if metrics is not None and successes != int(round(
                float(metrics["SR"]) * expected_episodes / 100.0
            )):
                raise UserError("FeedTTA feedback count disagrees with evaluator SR")
        else:
            queries = _count(adapter, "queries", expected_episodes)
            self_labels = _count(
                adapter, "self_label_episodes", expected_episodes
            )
            observed = _count(
                adapter, "feedback_observed_episodes", expected_episodes
            )
            gate_evaluations = _count(
                adapter, "query_gate_evaluations", expected_episodes
            )
            self_evaluations = _count(
                adapter, "self_prediction_evaluations", expected_episodes
            )
            queried_successes = _count(
                adapter, "queried_feedback_successes", expected_episodes
            )
            self_successes = _count(
                adapter, "self_feedback_successes", expected_episodes
            )
            if (
                queries + self_labels != expected_episodes
                or observed != queries
                or gate_evaluations != expected_episodes
                or self_evaluations != expected_episodes
                or queried_successes > queries
                or self_successes > self_labels
            ):
                raise UserError("ATENA query/self-label accounting mismatch")
            if (
                float(metadata["parameters"].get("query_threshold", math.nan))
                == 0.0
                and queries != expected_episodes
            ):
                raise UserError("zero-threshold ATENA must query every episode")
    else:
        if diagnostics.get("supervision") != "unsupervised":
            raise UserError("unsupervised method has wrong supervision label")
        if diagnostics.get("binary_feedback_endpoint") is not None:
            raise UserError("unsupervised method unexpectedly consumed feedback")
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
    diagnostics_path = Path(metadata["result_root"]) / "tta_diagnostics.json"
    diagnostics = _read_json(diagnostics_path)
    try:
        metric_path, metrics, metric_line = frozen_runner._parse_metrics(
            metadata["result_root"]
        )
    except frozen_runner.UserError as error:
        raise UserError(str(error))
    adapter = _validate_diagnostics(metadata, diagnostics, metrics=metrics)
    formal_sha = None
    formal_identity = None
    if metadata["stage"] == "full":
        try:
            manifest = frozen_runner._validate_formal_manifest(
                metadata["formal_manifest"], metadata,
                required_artifacts=(metric_path, diagnostics_path),
            )
        except frozen_runner.UserError as error:
            raise UserError(str(error))
        formal_sha = _sha256(metadata["formal_manifest"])
        formal_identity = manifest["immutable_identity_sha256"]
    elif metadata.get("formal_manifest") is not None:
        raise UserError("screening jobs must not claim a formal manifest")
    result = {
        **metadata,
        "metrics": metrics,
        "metric_artifact": str(metric_path),
        "metric_artifact_sha256": _sha256(metric_path),
        "metric_line": metric_line,
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "adapter_diagnostics": adapter,
        "feedback_endpoint": diagnostics.get("binary_feedback_endpoint"),
        "formal_manifest_sha256": formal_sha,
        "formal_immutable_identity_sha256": formal_identity,
    }
    _atomic_json(attempt_dir / "metrics.json", result)
    return result


def _job_state(batch_root, job):
    latest = _latest_attempt(batch_root, job)
    if latest is None:
        return "pending", None, None
    attempt, path = latest
    # A validation failure is durable evidence that the attempt cannot be
    # consumed, even if the worker exited successfully or left a stale metrics
    # cache behind.  Retry attempts naturally supersede it via _latest_attempt.
    if (path / "validation_error.json").is_file():
        return "invalid", attempt, path
    return frozen_runner._state_for_attempt(path), attempt, path


def _stage_results(batch_root, jobs, require_complete=True):
    results = []
    for job in jobs:
        state, _, path = _job_state(batch_root, job)
        validated = None
        if state in ("finished", "completed"):
            # ``metrics.json`` is only a cache.  Re-parse the result artifacts
            # and, for full jobs, re-authenticate the formal manifest and every
            # artifact before promotion/final selection can consume the run.
            validated = validate_attempt(path)
            state = "completed"
        if state != "completed":
            if require_complete:
                raise UserError("{} is {}; stage is incomplete".format(
                    job["base_run_tag"], state
                ))
            continue
        result = validated if validated is not None else validate_attempt(path)
        checks = {
            "base_run_tag": job["base_run_tag"],
            "stage": job["stage"],
            "setting": job["setting"],
            "model": job["model"],
            "method": job["method"],
            "parameters": job["parameters"],
        }
        for key, expected in checks.items():
            if _canonical(result.get(key)) != _canonical(expected):
                raise UserError(
                    "validated result does not match planned job {}: {}".format(
                        job["base_run_tag"], key
                    )
                )
        results.append(result)
    return results


def _screen_score(result):
    metrics = result["metrics"]
    adapter = result["adapter_diagnostics"]
    return (
        float(metrics["RGSPL"]),
        float(metrics["RGS"]),
        float(metrics["SPL"]),
        float(metrics["SR"]),
        -float(adapter["relative_param_drift"]),
        -float(adapter["updates"]),
    )


def promote_screening(spec_path, spec, batch_root, jobs, output_path=None):
    binding = _read_json(Path(batch_root) / "BATCH.json")
    _assert_execution_invariants(
        spec_path,
        spec,
        binding.get("batch_id"),
        batch_root,
        binding.get("gpu"),
    )
    results = _stage_results(batch_root, jobs)
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["setting"], result["method"])].append(result)
    promotions = []
    for cell in sorted(grouped, key=lambda item: (
        SETTINGS.index(item[0]), METHODS.index(item[1])
    )):
        ranked = sorted(grouped[cell], key=_screen_score, reverse=True)
        winner = ranked[0]
        promotions.append({
            "setting": winner["setting"],
            "method": winner["method"],
            "role": winner["role"],
            "parameters": winner["parameters"],
            "screening_run_tag": winner["run_tag"],
            "screening_metrics": winner["metrics"],
            "screening_adapter_diagnostics": {
                key: winner["adapter_diagnostics"].get(key)
                for key in ("updates", "relative_param_drift")
            },
            "ranked_screening_run_tags": [item["run_tag"] for item in ranked],
        })
    payload = {
        "schema": "navtta.vln_reverie_small_search_promotions.v1",
        "experiment_id": spec["experiment_id"],
        "spec_sha256": _sha256(spec_path),
        "selection_scope": "canonical_first_256_episodes_only",
        "primary_metric": "RGSPL",
        "promotions": promotions,
    }
    path = Path(output_path) if output_path is not None else (
        Path(batch_root) / "PROMOTIONS.json"
    )
    if path.is_file() and _canonical(_read_json(path)) != _canonical(payload):
        raise UserError("PROMOTIONS.json differs from recomputed screening result")
    _atomic_json(path, payload)
    return payload


def combine_model_promotions(spec_path, spec, batch_root, model_promotions):
    binding = _read_json(Path(batch_root) / "BATCH.json")
    _assert_execution_invariants(
        spec_path,
        spec,
        binding.get("batch_id"),
        batch_root,
        binding.get("gpu"),
    )
    promotions = [
        item
        for document in model_promotions
        for item in document.get("promotions", [])
    ]
    promotions.sort(key=lambda item: (
        SETTINGS.index(item["setting"]), METHODS.index(item["method"])
    ))
    expected_cells = {
        (setting, method)
        for method in METHODS
        for setting in spec["methods"][method]["candidates_by_setting"]
    }
    actual_cells = {(item["setting"], item["method"]) for item in promotions}
    if actual_cells != expected_cells or len(promotions) != len(expected_cells):
        raise UserError("model promotions do not cover exactly all target cells")
    payload = {
        "schema": "navtta.vln_reverie_small_search_promotions.v1",
        "experiment_id": spec["experiment_id"],
        "spec_sha256": _sha256(spec_path),
        "selection_scope": "canonical_first_256_episodes_only",
        "primary_metric": "RGSPL",
        "promotions": promotions,
    }
    path = Path(batch_root) / "PROMOTIONS.json"
    if path.is_file() and _canonical(_read_json(path)) != _canonical(payload):
        raise UserError("global PROMOTIONS.json differs from model promotions")
    _atomic_json(path, payload)
    return payload


def _plan_payload(spec_path, spec, batch_id, stage, jobs, gpu):
    return {
        "schema": "navtta.vln_reverie_small_search_stage.v1",
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "stage": stage,
        "git_commit": _git_commit(),
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "gpu": int(gpu),
        "job_count": len(jobs),
        "setting_order": list(SETTINGS),
        "jobs": jobs,
    }


def ensure_stage_plan(
    spec_path, spec, batch_id, batch_root, stage, jobs, gpu, output_path=None
):
    path = Path(output_path) if output_path is not None else (
        Path(batch_root) / "stages" / stage / "stage_plan.json"
    )
    payload = _plan_payload(spec_path, spec, batch_id, stage, jobs, gpu)
    if path.is_file():
        if _canonical(_read_json(path)) != _canonical(payload):
            raise UserError("existing {} plan differs from requested plan".format(stage))
    else:
        _atomic_json(path, payload)
    return jobs


def _batch_binding(spec_path, spec, batch_id, gpu):
    return {
        "schema": "navtta.vln_reverie_small_search_batch.v1",
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "git_commit": _git_commit(),
        "gpu": int(gpu),
        "source_registry_sha256": spec["dependencies"]["source_registry"]["sha256"],
        "incumbent_spec_sha256": spec["dependencies"]["incumbent_spec"]["sha256"],
    }


def _validate_batch_binding(spec_path, spec, batch_id, batch_root, gpu):
    path = Path(batch_root) / "BATCH.json"
    if not path.is_file():
        raise UserError("missing immutable BATCH.json binding")
    recorded = _read_json(path)
    expected = _batch_binding(spec_path, spec, batch_id, gpu)
    if recorded.get("git_commit") != expected["git_commit"]:
        raise UserError(
            "Git HEAD drifted from BATCH.json: expected {}, got {}".format(
                recorded.get("git_commit"), expected["git_commit"]
            )
        )
    if _canonical(recorded) != _canonical(expected):
        raise UserError("BATCH.json does not match the requested commit/spec/GPU")
    return recorded


def _prepare_batch(spec_path, spec, batch_id, batch_root, gpu, resume):
    path = Path(batch_root) / "BATCH.json"
    payload = _batch_binding(spec_path, spec, batch_id, gpu)
    if path.is_file():
        if not resume:
            raise UserError("batch already exists; use --resume")
        _validate_batch_binding(spec_path, spec, batch_id, batch_root, gpu)
    else:
        if resume:
            raise UserError("cannot resume a batch that does not exist")
        _atomic_json(path, payload)


def _assert_execution_invariants(spec_path, spec, batch_id, batch_root, gpu):
    _assert_clean_execution_tree(spec_path)
    _validate_batch_binding(spec_path, spec, batch_id, batch_root, gpu)


def _release_job_reservation(batch_id, job, attempt):
    release_shared_gpu_reservation(
        job["gpu"],
        _reservation_token(batch_id, _attempt_tag(job["base_run_tag"], attempt)),
    )


def _record_validation_failure(attempt_dir, error):
    _atomic_json(
        Path(attempt_dir) / "validation_error.json",
        {"error": str(error), "recorded_at_unix": time.time()},
    )


def _archive_invalid_attempt(attempt_dir):
    attempt_dir = Path(attempt_dir)
    archive_manifest = attempt_dir / "archived_evidence.json"
    if archive_manifest.is_file():
        return _read_json(archive_manifest)
    metadata = _read_json(attempt_dir / "job.json")
    archive_root = attempt_dir / "invalid_evidence"
    archive_root.mkdir(parents=True, exist_ok=False)
    archived = {}

    cached_metrics = attempt_dir / "metrics.json"
    if cached_metrics.exists():
        destination = archive_root / "metrics.json"
        cached_metrics.rename(destination)
        archived["metrics_cache"] = str(destination)

    result_root = Path(metadata["result_root"]).resolve()
    try:
        result_root.relative_to(RESULT_ROOT.resolve())
    except ValueError:
        raise UserError("invalid attempt result root escapes tuning results")
    if result_root.exists():
        destination = archive_root / "result_root"
        result_root.rename(destination)
        archived["result_root"] = str(destination)

    formal_manifest = metadata.get("formal_manifest")
    if formal_manifest:
        formal_dir = Path(formal_manifest).resolve().parent
        try:
            formal_dir.relative_to(FORMAL_ROOT.resolve())
        except ValueError:
            raise UserError("invalid attempt formal run escapes results/runs")
        if formal_dir.exists():
            destination = archive_root / "formal_run_manifest"
            formal_dir.rename(destination)
            archived["formal_run_manifest"] = str(destination)

    payload = {
        "schema": "navtta.vln_reverie_invalid_attempt_archive.v1",
        "run_tag": metadata["run_tag"],
        "attempt": metadata["attempt"],
        "archived": archived,
    }
    _atomic_json(archive_manifest, payload)
    return payload


def run_model_phase(
    spec, spec_path, batch_id, batch_root, jobs, retry_failed=False
):
    if not jobs:
        return
    setting = jobs[0]["setting"]
    if any(job["setting"] != setting for job in jobs):
        raise UserError("one scheduler phase may contain only one model")
    _assert_execution_invariants(
        spec_path, spec, batch_id, batch_root, jobs[0]["gpu"]
    )
    execution = spec["execution"]
    cap = execution["max_workers_by_setting"][setting]
    per_method_cap = execution["max_workers_per_method"]
    estimates = execution["estimated_gpu_memory_mib_by_method"]
    aggregate_cap = execution["max_aggregate_gpu_memory_mib"]
    minimum_free = execution["minimum_free_gpu_memory_mib"]
    emergency = execution["emergency_abort_used_gpu_memory_mib"]
    poll_seconds = execution["gpu_poll_seconds"]
    stagger = execution["launch_stagger_seconds"]
    last_launch = 0.0
    while True:
        _validate_batch_binding(
            spec_path, spec, batch_id, batch_root, jobs[0]["gpu"]
        )
        complete = 0
        running = []
        runnable = []
        for job in jobs:
            state, attempt, attempt_dir = _job_state(batch_root, job)
            if state in ("completed", "finished"):
                try:
                    validate_attempt(attempt_dir)
                except Exception as error:
                    _release_job_reservation(batch_id, job, attempt)
                    _record_validation_failure(attempt_dir, error)
                    if not retry_failed:
                        raise UserError(
                            "{} finished but validation failed: {}".format(
                                job["base_run_tag"], error
                            )
                        ) from error
                    _archive_invalid_attempt(attempt_dir)
                    runnable.append((job, attempt + 1))
                    continue
                _release_job_reservation(batch_id, job, attempt)
                complete += 1
                continue
            if state == "running":
                running.append(job)
                continue
            if state in ("failed", "invalid", "orphaned"):
                _release_job_reservation(batch_id, job, attempt)
                if not retry_failed:
                    raise UserError(
                        "{} is {}; resume with --retry-failed".format(
                            job["base_run_tag"], state
                        )
                    )
                if (
                    state == "invalid"
                    and (attempt_dir / "validation_error.json").is_file()
                ):
                    # The validation error may have been recorded by an
                    # earlier invocation that correctly stopped without
                    # --retry-failed.  Archive its evidence before creating
                    # the next attempt, just as for an in-process retry.
                    _archive_invalid_attempt(attempt_dir)
                attempt += 1
            else:
                attempt = 0
            runnable.append((job, attempt))
        if complete == len(jobs):
            return

        busy = Counter(job["method"] for job in running)
        launched = False
        for job, attempt in runnable:
            if len(running) >= cap or busy[job["method"]] >= per_method_cap:
                continue
            if time.monotonic() - last_launch < stagger:
                break
            estimate = int(estimates[job["method"]])
            with shared_gpu_launch_guard(job["gpu"]) as ledger:
                total, used, free = frozen_runner._gpu_memory_mib(job["gpu"])
                memory_gib = frozen_runner._cgroup_memory_gib()
                snapshot = ledger.snapshot(used, memory_gib)
                effective_used = snapshot["effective_gpu_memory_mib"]
                effective_free = total - effective_used
                if used >= emergency:
                    raise UserError(
                        "GPU memory reached emergency threshold: {} MiB".format(used)
                    )
                if (
                    free < minimum_free
                    or effective_free < estimate
                    or effective_used + estimate > aggregate_cap
                ):
                    continue
                attempt_dir, metadata = materialize_attempt(
                    spec, spec_path, batch_id, batch_root, job, attempt
                )
                token = _reservation_token(batch_id, metadata["run_tag"])
                ledger.reserve(
                    token,
                    gpu_memory_mib=estimate,
                    cgroup_memory_gib=0.0,
                    observed_gpu_memory_mib=used,
                    observed_cgroup_memory_gib=memory_gib,
                    owner=frozen_runner.process_identity(),
                    metadata={
                        "role": "reverie_small_search",
                        "batch_id": batch_id,
                        "run_tag": metadata["run_tag"],
                    },
                )
                try:
                    _assert_execution_invariants(
                        spec_path, spec, batch_id, batch_root, job["gpu"]
                    )
                    process, launcher_log = frozen_runner.launch_attempt(
                        attempt_dir, metadata
                    )
                    ledger.add_owner(
                        token,
                        _read_json(attempt_dir / "process_identity.json"),
                    )
                    launcher_log.close()
                except Exception:
                    ledger.release(token)
                    raise
            running.append(job)
            busy[job["method"]] += 1
            launched = True
            last_launch = time.monotonic()
            print("launched {} pid={}".format(metadata["run_tag"], process.pid))
        if not launched:
            time.sleep(min(2.0, float(poll_seconds)))


def _stage_summary(batch_root, spec, stage, jobs, output_path=None):
    binding = _read_json(Path(batch_root) / "BATCH.json")
    _assert_execution_invariants(
        binding.get("spec_path"),
        spec,
        binding.get("batch_id"),
        batch_root,
        binding.get("gpu"),
    )
    # Revalidate completed entries rather than trusting a cached metrics.json.
    rows = _stage_results(batch_root, jobs, require_complete=False)
    states = Counter()
    for job in jobs:
        state, _, _ = _job_state(batch_root, job)
        states[state] += 1
    payload = {
        "schema": "navtta.vln_reverie_small_search_summary.v1",
        "experiment_id": spec["experiment_id"],
        "stage": stage,
        "states": {key: states.get(key, 0) for key in (
            "completed", "finished", "failed", "invalid", "running",
            "orphaned", "pending",
        )},
        "total_jobs": len(jobs),
        "complete": states.get("completed", 0) == len(jobs),
        "results": [{
            "setting": row["setting"],
            "method": row["method"],
            "role": row["role"],
            "run_tag": row["run_tag"],
            "parameters": row["parameters"],
            "metrics": row["metrics"],
            "formal_manifest": row.get("formal_manifest"),
            "formal_manifest_sha256": row.get("formal_manifest_sha256"),
        } for row in sorted(rows, key=lambda item: item["ordinal"])],
    }
    path = Path(output_path) if output_path is not None else (
        Path(batch_root) / "stages" / stage / "SUMMARY.json"
    )
    _atomic_json(path, payload)
    return payload


def _final_score(record):
    metrics = record["metrics"]
    return tuple(float(metrics[key]) for key in ("RGSPL", "RGS", "SPL", "SR"))


def finalize_selection(spec_path, spec, batch_root, full_jobs):
    if not full_jobs:
        raise UserError("final selection requires promoted full jobs")
    batch_id = _read_json(Path(batch_root) / "BATCH.json").get("batch_id")
    _assert_execution_invariants(
        spec_path, spec, batch_id, batch_root, full_jobs[0]["gpu"]
    )
    full_results = {
        (item["setting"], item["method"]): item
        for item in _stage_results(batch_root, full_jobs)
    }
    expected_full_cells = {
        (job["setting"], job["method"]) for job in full_jobs
    }
    if (
        set(full_results) != expected_full_cells
        or len(full_results) != len(full_jobs)
    ):
        raise UserError(
            "full results are not exactly one authenticated run per target cell"
        )
    source = _source_registry(spec)
    incumbent_parameters = _incumbent_parameters(spec)
    # Re-authenticate the complete incumbent matrix at the point where it is
    # combined with the newly authenticated full runs.
    authenticated_incumbents = _validate_incumbents(
        spec, source, incumbent_parameters
    )
    records = {}
    frozen = {}
    tolerance = float(
        spec["protocol"]["source_sr_floor_tolerance_percentage_points"]
    )
    for setting in SETTINGS:
        records[setting] = {}
        frozen[setting] = {}
        source_metrics = source["records"][setting]["metrics"]
        for method in METHODS:
            incumbent = authenticated_incumbents[(setting, method)]
            incumbent_record = {
                "origin": "r2r_frozen_transfer_incumbent",
                "run_tag": incumbent["run_tag"],
                "parameters": incumbent["parameters"],
                "metrics": incumbent["metrics"],
                "formal_manifest": incumbent["formal_manifest"],
                "formal_manifest_sha256": incumbent["formal_manifest_sha256"],
                "metric_artifact": incumbent["metric_artifact"],
                "metric_artifact_sha256": incumbent[
                    "metric_artifact_sha256"
                ],
            }
            candidate = full_results.get((setting, method))
            eligible_candidate = None
            if candidate is not None:
                candidate_record = {
                    "origin": "new_full_candidate",
                    "run_tag": candidate["run_tag"],
                    "parameters": candidate["parameters"],
                    "metrics": candidate["metrics"],
                    "formal_manifest": candidate["formal_manifest"],
                    "formal_manifest_sha256": candidate[
                        "formal_manifest_sha256"
                    ],
                    "screening_run_tag": candidate[
                        "promoted_from_run_tag"
                    ],
                }
                if candidate["metrics"]["SR"] >= source_metrics["SR"] - tolerance:
                    eligible_candidate = candidate_record
            winner = incumbent_record
            if (
                eligible_candidate is not None
                and _final_score(eligible_candidate) > _final_score(incumbent_record)
            ):
                winner = eligible_candidate
            records[setting][method] = {
                **winner,
                "source_metrics": source_metrics,
                "delta_vs_source_pp": {
                    metric: round(
                        winner["metrics"][metric] - source_metrics[metric], 6
                    )
                    for metric in ("SR", "SPL", "RGS", "RGSPL")
                },
                "better_than_source_on_rgspl": (
                    winner["metrics"]["RGSPL"] > source_metrics["RGSPL"]
                ),
                "new_candidate_eligible": eligible_candidate is not None,
            }
            frozen[setting][method] = winner["parameters"]
    expected_cells = {(setting, method) for setting in SETTINGS for method in METHODS}
    actual_cells = {
        (setting, method)
        for setting, methods in records.items()
        for method in methods
    }
    if actual_cells != expected_cells or sum(map(len, records.values())) != 15:
        raise UserError("final selection is not a complete 3x5 matrix")
    selection = {
        "schema": "navtta.vln_reverie_val_seen_small_search_selection.v1",
        "experiment_id": spec["experiment_id"],
        "split": "val_seen",
        "git_commit": _git_commit(),
        "spec_sha256": _sha256(spec_path),
        "primary_metric": "RGSPL",
        "source_sr_floor_tolerance_percentage_points": tolerance,
        "records": records,
    }
    frozen_output = {
        "schema": "navtta.vln_reverie_val_seen_frozen_hparams.v1",
        "generated_from": "FINAL_SELECTION.json",
        "git_commit": selection["git_commit"],
        "spec_sha256": selection["spec_sha256"],
        "settings": frozen,
    }
    for path, payload in (
        (Path(batch_root) / "FINAL_SELECTION.json", selection),
        (Path(batch_root) / "FROZEN_HPARAMETERS.json", frozen_output),
    ):
        if path.is_file() and _canonical(_read_json(path)) != _canonical(payload):
            raise UserError("immutable final output differs: {}".format(path))
        _atomic_json(path, payload)
    return selection


def print_plan(spec, batch_id, gpu=0, verbose=False):
    jobs = expand_screening_jobs(spec, batch_id, gpu)
    grouped = Counter((job["setting"], job["method"]) for job in jobs)
    print("experiment={}".format(spec["experiment_id"]))
    print("source_jobs=0 incumbent_reruns=0")
    print("screening_jobs={} screening_episodes_each={}".format(
        len(jobs), spec["protocol"]["screening_episodes"]
    ))
    print("maximum_full_jobs={}".format(len(grouped)))
    print("maximum_new_jobs={}".format(len(jobs) + len(grouped)))
    print("maximum_episode_evaluations={}".format(
        spec["budget"]["maximum_new_episode_evaluations"]
    ))
    for setting in SETTINGS:
        cells = [
            "{}={}".format(method, grouped[(setting, method)])
            for method in METHODS if grouped[(setting, method)]
        ]
        print("{}: {}".format(setting, ", ".join(cells)))
    print("execution=DUET -> HAMT -> GOAT; methods parallel within one model")
    if verbose:
        for job in jobs:
            print("{} {} {} {}".format(
                job["setting"], job["method"], job["role"],
                _canonical(job["parameters"]),
            ))


def show_status(spec, batch_id, gpu=0):
    batch_root = LOG_ROOT / batch_id
    screening = expand_screening_jobs(spec, batch_id, gpu)
    snapshot = {"batch_id": batch_id, "models": {}, "stages": {}}
    for stage, jobs in (("screening", screening),):
        counts = Counter(_job_state(batch_root, job)[0] for job in jobs)
        snapshot["stages"][stage] = {
            key: counts.get(key, 0) for key in (
                "completed", "finished", "failed", "invalid", "running",
                "orphaned", "pending",
            )
        }
    full = []
    for setting in SETTINGS:
        setting_screening = [
            job for job in screening if job["setting"] == setting
        ]
        screen_counts = Counter(
            _job_state(batch_root, job)[0] for job in setting_screening
        )
        model_status = {
            "screening": {
                key: screen_counts.get(key, 0) for key in (
                    "completed", "finished", "failed", "invalid", "running",
                    "orphaned", "pending",
                )
            }
        }
        promotion_path = (
            batch_root / "stages" / "screening" / setting / "PROMOTIONS.json"
        )
        if promotion_path.is_file():
            promotion = _read_json(promotion_path)
            setting_full = expand_full_jobs(
                spec, batch_id, promotion, gpu, settings=(setting,)
            )
        else:
            full_plan_path = (
                batch_root / "stages" / "full" / setting / "stage_plan.json"
            )
            setting_full = (
                _read_json(full_plan_path).get("jobs", [])
                if full_plan_path.is_file() else []
            )
        if setting_full:
            full.extend(setting_full)
            full_counts = Counter(
                _job_state(batch_root, job)[0] for job in setting_full
            )
            model_status["full"] = {
                key: full_counts.get(key, 0) for key in (
                    "completed", "finished", "failed", "invalid", "running",
                    "orphaned", "pending",
                )
            }
        snapshot["models"][setting] = model_status
    if full:
        counts = Counter(_job_state(batch_root, job)[0] for job in full)
        snapshot["stages"]["full"] = {
            key: counts.get(key, 0) for key in (
                "completed", "finished", "failed", "invalid", "running",
                "orphaned", "pending",
            )
        }
    print(json.dumps(snapshot, indent=2, sort_keys=True))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=DEFAULT_SPEC,
    )
    parser.add_argument(
        "--batch-id",
        default="vln-reverie-val-seen-small-hparam-search-v1-seed0",
    )
    parser.add_argument(
        "--stage", choices=("plan", "screening", "full", "all", "status"),
        default="plan",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.spec.is_absolute():
        args.spec = REPO_ROOT / args.spec
    args.spec = args.spec.resolve()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.batch_id):
        parser.error("invalid batch ID")
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    return args


def main(argv=None):
    args = parse_args(argv)
    spec = load_spec(args.spec)
    if args.stage == "plan":
        print_plan(spec, args.batch_id, args.gpu, verbose=args.verbose)
        return 0
    if args.stage == "status":
        show_status(spec, args.batch_id, args.gpu)
        return 0
    if not args.confirm_reviewed:
        raise UserError("execution requires --confirm-reviewed")
    _assert_clean_execution_tree(args.spec)
    batch_root = LOG_ROOT / args.batch_id
    batch_root.mkdir(parents=True, exist_ok=True)
    _prepare_batch(
        args.spec, spec, args.batch_id, batch_root, args.gpu, args.resume
    )
    screening_jobs = expand_screening_jobs(spec, args.batch_id, args.gpu)
    ensure_stage_plan(
        args.spec, spec, args.batch_id, batch_root, "screening",
        screening_jobs, args.gpu,
    )
    model_promotions = []
    full_jobs = []
    for setting in SETTINGS:
        setting_screening = [
            job for job in screening_jobs if job["setting"] == setting
        ]
        ensure_stage_plan(
            args.spec,
            spec,
            args.batch_id,
            batch_root,
            "screening",
            setting_screening,
            args.gpu,
            output_path=(
                batch_root / "stages" / "screening" / setting
                / "stage_plan.json"
            ),
        )
        if args.stage in ("screening", "all"):
            print("starting screening model phase {}".format(setting))
            run_model_phase(
                spec, args.spec, args.batch_id, batch_root,
                setting_screening,
                retry_failed=args.retry_failed,
            )
            summary = _stage_summary(
                batch_root,
                spec,
                "screening",
                setting_screening,
                output_path=(
                    batch_root / "stages" / "screening" / setting
                    / "SUMMARY.json"
                ),
            )
            if not summary["complete"]:
                raise UserError("{} screening did not complete".format(setting))
        model_promotion = promote_screening(
            args.spec,
            spec,
            batch_root,
            setting_screening,
            output_path=(
                batch_root / "stages" / "screening" / setting
                / "PROMOTIONS.json"
            ),
        )
        model_promotions.append(model_promotion)

        setting_full = expand_full_jobs(
            spec,
            args.batch_id,
            model_promotion,
            args.gpu,
            settings=(setting,),
        )
        full_jobs.extend(setting_full)
        ensure_stage_plan(
            args.spec,
            spec,
            args.batch_id,
            batch_root,
            "full",
            setting_full,
            args.gpu,
            output_path=(
                batch_root / "stages" / "full" / setting / "stage_plan.json"
            ),
        )
        if args.stage in ("full", "all"):
            print("starting full model phase {}".format(setting))
            run_model_phase(
                spec, args.spec, args.batch_id, batch_root,
                setting_full,
                retry_failed=args.retry_failed,
            )
            summary = _stage_summary(
                batch_root,
                spec,
                "full",
                setting_full,
                output_path=(
                    batch_root / "stages" / "full" / setting / "SUMMARY.json"
                ),
            )
            if not summary["complete"]:
                raise UserError("{} full stage did not complete".format(setting))

    combine_model_promotions(args.spec, spec, batch_root, model_promotions)
    _stage_summary(batch_root, spec, "screening", screening_jobs)
    if args.stage in ("full", "all"):
        summary = _stage_summary(batch_root, spec, "full", full_jobs)
        if not summary["complete"]:
            raise UserError("full stage did not complete")
        finalize_selection(args.spec, spec, batch_root, full_jobs)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (UserError, ReservationLedgerError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
