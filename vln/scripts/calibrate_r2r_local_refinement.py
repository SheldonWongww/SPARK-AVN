#!/usr/bin/env python3
"""Measure safe single-phase concurrency for an R2R refinement campaign.

This helper consumes a campaign already materialized by
``run_r2r_local_refinement.py --plan-only``.  It never executes a persisted
formal job in place.  Instead, it clones a prefix of one phase into an
isolated calibration namespace and rewrites every run/config/result path. The
current campaign first measures a fresh five-worker/100-episode group. A
larger declared count is allowed only when that digest-bound baseline projects
below the sizing ceiling, and is then validated as a separate complete group.
Transient lower counts observed during group launch are not treated as safe
levels. Every accepted group must remain at steady VRAM for a measured window;
the fixed stagger is only a launch delay, never loading evidence.

The two memory lines are deliberately fixed here: stop adding workers once
29,000 MiB is observed and terminate only this helper's process groups once
30,000 MiB is observed.  A calibration cannot contain jobs from more than one
phase and a campaign-wide lock prevents two calibrations from overlapping.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
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
LEVEL_EVIDENCE_SCHEMA = "navtta.vln_r2r_calibration_level_evidence.v1"

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
PRIOR_RECOVERY_RELATIVE_TOLERANCE = 0.02
FULL_VAL_EPISODES = 1021
ALLOWED_CONTINUATION_DIFF_FILES = {
    "vln/scripts/calibrate_r2r_local_refinement.py",
    "vln/tests/test_calibrate_r2r_local_refinement.py",
    "vln/experiments/README.md",
}


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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "output", "")
        raise UserError(f"git {' '.join(args)} failed: {output or error}") from error


def _helper_only_commit_diff(base_commit, target_commit):
    for label, value in (("base", base_commit), ("target", target_commit)):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise UserError(f"invalid {label} commit for continuation audit")
        _git("cat-file", "-e", f"{value}^{{commit}}")
    if base_commit == target_commit:
        return []
    changed = [
        item for item in _git(
            "diff", "--name-only", base_commit, target_commit, "--"
        ).splitlines() if item
    ]
    forbidden = sorted(set(changed).difference(ALLOWED_CONTINUATION_DIFF_FILES))
    if forbidden:
        raise UserError(
            "continuation changes formal execution files outside the helper-only "
            f"allowlist: {forbidden}"
        )
    return sorted(set(changed))


def _assert_plan_revision(plan, allow_helper_only_drift=False):
    planned = plan.get("git_commit")
    if not isinstance(planned, str) or re.fullmatch(r"[0-9a-f]{40}", planned) is None:
        raise UserError("campaign PLAN.json has no valid pinned git_commit")
    current = _git("rev-parse", "--verify", "HEAD")
    allowed_diff = []
    if current != planned:
        if not allow_helper_only_drift:
            raise UserError(
                f"campaign commit mismatch: PLAN={planned}, current={current}"
            )
        allowed_diff = _helper_only_commit_diff(planned, current)
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise UserError("tracked worktree must be clean before calibration")
    return {
        "plan_git_commit": planned,
        "execution_git_commit": current,
        "allowed_helper_only_diff_files": allowed_diff,
    }


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


def _validated_grouped_policy(phase_manifest):
    """Return the immutable per-phase grouped sizing policy."""
    policy = phase_manifest.get("calibration_policy")
    if not isinstance(policy, dict):
        raise UserError("phase does not declare a grouped calibration policy")
    if policy.get("mode") != "measured_grouped_jump_v1":
        raise UserError("phase grouped calibration mode is unsupported")
    initial = policy.get("initial_group_workers")
    episode_limit = policy.get("episode_limit")
    projection_max = policy.get("projection_max_mib")
    estimate = policy.get("estimated_mib_per_job")
    factor = policy.get("projection_safety_factor")
    allowed = policy.get("allowed_worker_counts")
    for label, value in (
        ("initial_group_workers", initial),
        ("episode_limit", episode_limit),
        ("projection_max_mib", projection_max),
        ("estimated_mib_per_job", estimate),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise UserError(f"invalid grouped calibration policy {label}")
    if initial != 5 or episode_limit != 100:
        raise UserError("grouped calibration policy must start at 5 for 100 episodes")
    if projection_max >= PLANNED_MEMORY_MIB:
        raise UserError("grouped calibration projection reaches the 29,000 MiB line")
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not 1.0 <= float(factor) <= 1.5
    ):
        raise UserError("invalid grouped calibration projection safety factor")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in allowed
        )
        or allowed != sorted(set(allowed))
        or initial not in allowed
    ):
        raise UserError("invalid grouped calibration allowed worker counts")
    if policy.get("independent_steady_validation") is not True:
        raise UserError("grouped jumps must require independent steady validation")
    if policy.get("steady_used_mib_stop") != PLANNED_MEMORY_MIB:
        raise UserError("phase grouped calibration stop line changed")
    if policy.get("observed_used_mib_abort") != EMERGENCY_MEMORY_MIB:
        raise UserError("phase grouped calibration emergency line changed")
    declared = phase_manifest.get("declared_worker_limits")
    if declared != allowed:
        raise UserError("phase declared worker limits disagree with calibration policy")
    return copy.deepcopy(policy)


def _execution_config_identity(document):
    return {
        key: value for key, value in document.items()
        if key not in ("batch_id", "run_tag")
    }


def _level_evidence_sha256(level):
    return hashlib.sha256(_canonical(level).encode("utf-8")).hexdigest()


def _validated_level_chain(
    summary, summary_path, log_root, visited=None,
    expected_episode_limit=None, expected_episode_budget=None,
    enforce_episode_budget=False,
):
    """Resolve and authenticate the continuous level chain in ``summary``."""
    summary_path = Path(summary_path).resolve()
    visited = set() if visited is None else set(visited)
    if summary_path in visited:
        raise UserError("prior calibration evidence chain contains a cycle")
    visited.add(summary_path)
    if summary.get("schema") != CALIBRATION_SCHEMA:
        raise UserError("prior calibration chain has an unsupported schema")
    if enforce_episode_budget:
        chain_plan_path = summary_path.parent / "CALIBRATION_PLAN.json"
        chain_plan = _read_json(
            chain_plan_path, "prior chain CALIBRATION_PLAN.json"
        )
        if chain_plan.get("schema") != CALIBRATION_PLAN_SCHEMA:
            raise UserError(
                "prior calibration chain plan has an unsupported schema"
            )
        if chain_plan.get("episode_limit") != expected_episode_limit:
            raise UserError(
                "prior calibration chain episode_limit mismatch"
            )
        chain_budget = chain_plan.get("episode_budget_per_worker")
        if (
            expected_episode_limit is not None
            and chain_budget is None
        ):
            raise UserError(
                "prefix calibration chain lacks episode budget provenance"
            )
        if (
            chain_budget is not None
            and chain_budget != expected_episode_budget
        ):
            raise UserError(
                "prior calibration chain episode budget mismatch"
            )
        for key, expected in (
            ("episode_limit", expected_episode_limit),
            ("episode_budget_per_worker", expected_episode_budget),
        ):
            if key in summary and summary.get(key) != expected:
                raise UserError(
                    f"prior calibration chain {key} mismatch"
                )
    cap = summary.get("recommended_cap")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise UserError("prior calibration chain has an invalid recommended_cap")
    initial_workers = summary.get("initial_workers", 0) or 0
    if (
        isinstance(initial_workers, bool)
        or not isinstance(initial_workers, int)
        or initial_workers < 0
        or initial_workers > cap
    ):
        raise UserError("prior calibration chain has invalid initial_workers")
    levels = {
        level.get("active_count"): level
        for level in summary.get("levels", []) if isinstance(level, dict)
    }
    resolved = {}
    strict_level_evidence = summary.get("level_evidence_schema") == (
        LEVEL_EVIDENCE_SCHEMA
    )
    if initial_workers:
        initial_group = summary.get("initial_group")
        if not isinstance(initial_group, dict) or initial_group.get(
            "steady_confirmed"
        ) is not True:
            raise UserError("prior continuation initial group was not confirmed")
        evidence = summary.get("prior_evidence")
        if not isinstance(evidence, dict):
            raise UserError("prior continuation lacks upstream evidence")
        upstream_path = Path(str(evidence.get("path", ""))).resolve()
        if not _is_relative_to(upstream_path, log_root):
            raise UserError("upstream calibration escapes the hparam log root")
        if upstream_path in visited:
            raise UserError("prior calibration evidence chain contains a cycle")
        if _sha256(upstream_path) != evidence.get("sha256"):
            raise UserError("upstream calibration digest mismatch")
        upstream_plan_path = Path(str(evidence.get("plan_path", ""))).resolve()
        if (
            not _is_relative_to(upstream_plan_path, log_root)
            or _sha256(upstream_plan_path) != evidence.get("plan_sha256")
        ):
            raise UserError("upstream calibration plan digest mismatch")
        upstream = _read_json(upstream_path, "upstream CALIBRATION.json")
        if upstream.get("calibration_id") != evidence.get("calibration_id"):
            raise UserError("upstream calibration id mismatch")
        if upstream.get("recommended_cap") != initial_workers:
            raise UserError("upstream cap does not match continuation prefix")
        upstream_levels = _validated_level_chain(
            upstream, upstream_path, log_root, visited=visited,
            expected_episode_limit=expected_episode_limit,
            expected_episode_budget=expected_episode_budget,
            enforce_episode_budget=enforce_episode_budget,
        )
        for count in range(1, initial_workers):
            level = levels.get(count)
            source = upstream_levels.get(count)
            if source is None:
                raise UserError(f"upstream evidence is missing level {count}")
            if strict_level_evidence:
                if level is None or level.get("evidence_origin") != "prior":
                    raise UserError(
                        f"inherited level {count} lacks prior provenance"
                    )
                if (
                    level.get("evidence_calibration_sha256")
                    != evidence.get("sha256")
                    or level.get("evidence_calibration_id")
                    != evidence.get("calibration_id")
                    or level.get("evidence_level_sha256")
                    != _level_evidence_sha256(source)
                    or level.get("evidence_level_snapshot") != source
                ):
                    raise UserError(
                        f"inherited level {count} evidence binding mismatch"
                    )
                for key in (
                    "active_count", "steady_confirmed", "steady_gpu_memory_mib",
                    "peak_gpu_memory_mib",
                ):
                    if level.get(key) != source.get(key):
                        raise UserError(
                            f"inherited level {count} changed bound field {key}"
                        )
                resolved[count] = copy.deepcopy(level)
            else:
                # Continuation summaries written before level-chain support
                # contained transient replay rows for 1..N-1. Resolve those
                # rows recursively from the already digest-pinned upstream.
                inherited = copy.deepcopy(source)
                inherited.update({
                    "evidence_origin": "prior_legacy_resolved",
                    "evidence_calibration_path": str(upstream_path),
                    "evidence_calibration_id": evidence.get("calibration_id"),
                    "evidence_calibration_sha256": evidence.get("sha256"),
                    "evidence_level_sha256": _level_evidence_sha256(source),
                    "evidence_level_snapshot": copy.deepcopy(source),
                    "legacy_resolution_summary_sha256": _sha256(summary_path),
                })
                resolved[count] = inherited
        level_n = levels.get(initial_workers)
        if level_n is None or (
            strict_level_evidence
            and level_n.get("evidence_origin")
            != "current_initial_group_revalidation"
        ):
            raise UserError("continuation level N lacks current revalidation")

    start = initial_workers if initial_workers else 1
    for count in range(start, cap + 1):
        level = levels.get(count)
        if level is None or level.get("steady_confirmed") is not True:
            raise UserError(f"prior level {count} is not continuously confirmed")
        if (
            strict_level_evidence
            and count > initial_workers
            and initial_workers
            and level.get("evidence_origin") != "current"
        ):
            raise UserError(f"extension level {count} is not current evidence")
        resolved[count] = copy.deepcopy(level)
    if set(resolved) != set(range(1, cap + 1)):
        raise UserError("prior calibration level chain is not continuous")
    return resolved


def _load_prior_evidence(
    path, campaign_root, plan, phase, selected_jobs, initial_workers, gpu,
    episode_limit=None,
):
    """Validate one prior calibration before allowing a continuation ramp."""
    if (
        episode_limit is not None
        and (
            isinstance(episode_limit, bool)
            or not isinstance(episode_limit, int)
            or episode_limit < 1
        )
    ):
        raise UserError("episode_limit must be a positive integer or null")
    expected_config_episodes = (
        episode_limit if episode_limit is not None else -1
    )
    expected_episode_budget = (
        episode_limit
        if episode_limit is not None
        else int(plan.get("episode_count", FULL_VAL_EPISODES))
    )
    path = Path(path).resolve()
    campaign_parent = Path(campaign_root).resolve().parent
    if not _is_relative_to(path, campaign_parent):
        raise UserError(
            "prior calibration must stay inside the same hparam-search log root"
        )
    prior = _read_json(path, "prior CALIBRATION.json")
    if prior.get("schema") != CALIBRATION_SCHEMA:
        raise UserError("prior calibration has an unsupported schema")
    prior_plan_path = path.parent / "CALIBRATION_PLAN.json"
    prior_plan = _read_json(prior_plan_path, "prior CALIBRATION_PLAN.json")
    if prior_plan.get("schema") != CALIBRATION_PLAN_SCHEMA:
        raise UserError("prior calibration plan has an unsupported schema")

    if prior.get("batch_id") != prior_plan.get("batch_id"):
        raise UserError("prior calibration batch does not match its plan")
    current_identity = {
        "phase_id": phase["phase_id"],
        "setting": phase["setting"],
        "model": phase["model"],
        "method": phase["method"],
    }
    for key, expected in current_identity.items():
        if prior.get(key) != expected or prior_plan.get(key) != expected:
            raise UserError(f"prior calibration identity mismatch for {key}")
    prior_source_commit = prior_plan.get("source_git_commit")
    prior_spec_sha256 = prior_plan.get("source_spec_sha256")
    if prior_spec_sha256 != plan.get("spec_sha256"):
        raise UserError("prior calibration provenance mismatch for source_spec_sha256")
    if prior_plan.get("gpu") != gpu:
        raise UserError("prior calibration provenance mismatch for gpu")
    if prior_plan.get("episode_limit") != episode_limit:
        raise UserError(
            "prior calibration provenance mismatch for episode_limit"
        )
    prior_budget = prior_plan.get("episode_budget_per_worker")
    if episode_limit is not None and prior_budget is None:
        raise UserError(
            "prefix prior calibration lacks episode budget provenance"
        )
    if (
        prior_budget is not None
        and prior_budget != expected_episode_budget
    ):
        raise UserError(
            "prior calibration provenance mismatch for "
            "episode_budget_per_worker"
        )
    prior_execution_commit = prior.get(
        "execution_git_commit", prior_source_commit
    )
    current_execution_commit = _git("rev-parse", "--verify", "HEAD")
    allowed_diff = sorted(set(
        _helper_only_commit_diff(prior_source_commit, prior_execution_commit)
        + _helper_only_commit_diff(prior_execution_commit, current_execution_commit)
        + _helper_only_commit_diff(plan.get("git_commit"), current_execution_commit)
    ))
    provenance = {
        "source_git_commit": prior_source_commit,
        "source_spec_sha256": prior_spec_sha256,
        "gpu": gpu,
        "episode_limit": episode_limit,
        "episode_budget_per_worker": expected_episode_budget,
    }
    for key, expected in provenance.items():
        # New summaries carry these fields directly.  Older evidence from the
        # same helper is accepted only because its sibling immutable plan has
        # the required pinned provenance.
        if key in prior and prior.get(key) != expected:
            raise UserError(f"prior CALIBRATION.json mismatch for {key}")
    if prior.get("calibration_id") != prior_plan.get("calibration_id"):
        raise UserError("prior calibration id does not match its plan")
    if prior.get("status") not in ("completed", "inconclusive"):
        raise UserError("prior calibration must be completed or inconclusive")

    cap = prior.get("recommended_cap")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < initial_workers:
        raise UserError("prior recommended_cap is below initial_workers")
    if cap != initial_workers:
        raise UserError(
            "initial_workers must equal the prior recommended_cap; skipping "
            "or partially replaying the proven prefix is forbidden"
        )
    by_count = _validated_level_chain(
        prior, path, campaign_parent, visited=set(),
        expected_episode_limit=episode_limit,
        expected_episode_budget=expected_episode_budget,
        enforce_episode_budget=True,
    )
    for count in range(1, initial_workers + 1):
        level = by_count.get(count)
        if level is None or level.get("steady_confirmed") is not True:
            raise UserError(
                f"prior level {count} is not continuously steady-confirmed"
            )
        try:
            steady = float(level["steady_gpu_memory_mib"])
            peak = float(level["peak_gpu_memory_mib"])
        except (KeyError, TypeError, ValueError) as error:
            raise UserError(f"prior level {count} has invalid VRAM evidence") from error
        if (
            not math.isfinite(steady)
            or not math.isfinite(peak)
            or steady <= 0
            or steady >= PLANNED_MEMORY_MIB
            or peak >= PLANNED_MEMORY_MIB
        ):
            raise UserError(f"prior level {count} is outside the safe VRAM line")

    prior_plan_jobs = prior_plan.get("jobs")
    if (
        not isinstance(prior_plan_jobs, list)
        or len(prior_plan_jobs) < initial_workers
    ):
        raise UserError("prior calibration plan lacks the proven candidate prefix")
    config_identities = []
    compared_count = min(len(prior_plan_jobs), len(selected_jobs))
    for index, current_job in enumerate(selected_jobs):
        current_config = _read_json(
            current_job.get("config_path"), "current formal job config"
        )
        if current_config.get("episodes") != -1:
            raise UserError(
                "current formal source config must keep episodes=-1"
            )
        current_identity = _execution_config_identity(current_config)
        if episode_limit is not None:
            # The formal template deliberately carries episodes=-1.  A
            # calibration prefix rewrites only this execution-budget field;
            # normalize the current source identity to the exact prefix that
            # the digest-pinned prior plan and cloned config must still carry.
            current_identity["episodes"] = expected_config_episodes
        config_identities.append(current_identity)
        if index >= compared_count:
            continue
        prior_job = prior_plan_jobs[index]
        if not isinstance(prior_job, dict):
            raise UserError("prior calibration plan has an invalid job entry")
        prior_config = _read_json(
            prior_job.get("config_path"), "prior calibration job config"
        )
        if prior_config.get("episodes") != expected_config_episodes:
            raise UserError(
                f"prior calibration episodes changed at candidate {index + 1}"
            )
        if episode_limit is not None:
            prior_command = prior_job.get("command")
            if not isinstance(prior_command, list) or (
                prior_command.count("--episode-limit") != 1
            ):
                raise UserError(
                    "prefix prior calibration command must contain "
                    "--episode-limit exactly once"
                )
            option_index = prior_command.index("--episode-limit")
            if (
                option_index + 1 >= len(prior_command)
                or prior_command[option_index + 1] != str(episode_limit)
            ):
                raise UserError(
                    "prefix prior calibration command episode limit mismatch"
                )
        prior_identity = _execution_config_identity(prior_config)
        if prior_identity != current_identity:
            raise UserError(
                f"source job config identity changed at candidate {index + 1}"
            )

    prior_jobs = sorted(
        (
            job for job in prior.get("jobs", [])
            if isinstance(job, dict)
            and isinstance(job.get("worker_index"), int)
            and 1 <= job["worker_index"] <= initial_workers
        ),
        key=lambda job: job["worker_index"],
    )
    expected_tags = [
        job.get("source_run_tag")
        for job in prior_plan_jobs[:initial_workers]
    ]
    observed_tags = [job.get("source_run_tag") for job in prior_jobs]
    if len(prior_jobs) != initial_workers or observed_tags != expected_tags:
        raise UserError("prior calibration candidate prefix does not match this plan")

    level_n = by_count[initial_workers]
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "plan_path": str(prior_plan_path),
        "plan_sha256": _sha256(prior_plan_path),
        "calibration_id": prior["calibration_id"],
        "prior_batch_id": prior["batch_id"],
        "current_batch_id": plan["batch_id"],
        "status": prior["status"],
        "recommended_cap": cap,
        "initial_workers": initial_workers,
        "continuous_steady_levels_verified": True,
        "level_n_steady_gpu_memory_mib": float(
            level_n["steady_gpu_memory_mib"]
        ),
        "level_n_peak_gpu_memory_mib": float(level_n["peak_gpu_memory_mib"]),
        "source_job_config_identity_sha256": hashlib.sha256(
            _canonical(config_identities).encode("utf-8")
        ).hexdigest(),
        "source_job_configs_compared_to_prior": compared_count,
        "source_job_configs_from_current_plan": len(selected_jobs),
        "plan_git_commit": plan.get("git_commit"),
        "prior_execution_git_commit": prior_execution_commit,
        "execution_git_commit": current_execution_commit,
        "allowed_helper_only_diff_files": allowed_diff,
        "_validated_level_prefix": [
            copy.deepcopy(by_count[count])
            for count in range(1, initial_workers + 1)
        ],
        **provenance,
    }


def _load_sizing_parent_evidence(
    path, campaign_root, plan, phase, selected_jobs, gpu, policy
):
    """Authenticate the fresh five-worker measurement authorizing one jump."""
    campaign_root = Path(campaign_root).resolve()
    path = Path(path).resolve()
    phase_calibration_root = (
        campaign_root / "calibration" / phase["phase_id"]
    ).resolve()
    if path.name != "CALIBRATION.json" or not _is_relative_to(
        path, phase_calibration_root
    ):
        raise UserError(
            "sizing parent must be a CALIBRATION.json in this campaign phase"
        )
    parent = _read_json(path, "sizing parent CALIBRATION.json")
    parent_plan_path = path.parent / "CALIBRATION_PLAN.json"
    parent_plan = _read_json(
        parent_plan_path, "sizing parent CALIBRATION_PLAN.json"
    )
    if parent.get("schema") != CALIBRATION_SCHEMA:
        raise UserError("sizing parent has an unsupported schema")
    if parent_plan.get("schema") != CALIBRATION_PLAN_SCHEMA:
        raise UserError("sizing parent plan has an unsupported schema")

    resource_name = parent.get("resource_csv")
    resource_path = (path.parent / str(resource_name or "")).resolve()
    if resource_name != "resource.csv" or resource_path.parent != path.parent:
        raise UserError("sizing parent resource CSV path is invalid")
    if not resource_path.is_file() or _sha256(resource_path) != parent.get(
        "resource_csv_sha256"
    ):
        raise UserError("sizing parent resource CSV digest mismatch")
    try:
        with resource_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != RESOURCE_FIELDS:
                raise UserError("sizing parent resource CSV header mismatch")
            resource_rows = list(reader)
        resource_peak_mib = max(
            float(row["gpu_memory_mib"]) for row in resource_rows
        )
        resource_max_active = max(
            int(row["active_count"]) for row in resource_rows
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise UserError("sizing parent resource CSV is invalid") from error
    if not resource_rows or any(
        row.get("phase_id") != phase["phase_id"]
        or row.get("setting") != phase["setting"]
        or row.get("method") != phase["method"]
        for row in resource_rows
    ):
        raise UserError("sizing parent resource CSV identity mismatch")

    identity = {
        "phase_id": phase["phase_id"],
        "setting": phase["setting"],
        "model": phase["model"],
        "method": phase["method"],
    }
    for key, expected in identity.items():
        if parent.get(key) != expected or parent_plan.get(key) != expected:
            raise UserError(f"sizing parent identity mismatch for {key}")
    if parent.get("calibration_id") != parent_plan.get("calibration_id"):
        raise UserError("sizing parent id does not match its plan")
    if parent.get("batch_id") != plan.get("batch_id") or (
        parent_plan.get("batch_id") != plan.get("batch_id")
    ):
        raise UserError("sizing parent must come from the current campaign")
    for document, label in ((parent, "summary"), (parent_plan, "plan")):
        if document.get("source_spec_sha256") != plan.get("spec_sha256"):
            raise UserError(f"sizing parent {label} spec digest mismatch")
        if document.get("source_git_commit") != plan.get("git_commit"):
            raise UserError(f"sizing parent {label} source commit mismatch")
        if document.get("gpu") != gpu:
            raise UserError(f"sizing parent {label} GPU mismatch")
        if document.get("calibration_policy") != policy:
            raise UserError(f"sizing parent {label} calibration policy mismatch")

    baseline_workers = int(policy["initial_group_workers"])
    episode_limit = int(policy["episode_limit"])
    if len(selected_jobs) <= baseline_workers:
        raise UserError("a sizing jump must target more than five workers")
    target_workers = len(selected_jobs)
    if target_workers not in policy["allowed_worker_counts"]:
        raise UserError("sizing jump worker count is not declared by the phase")
    for document, label in ((parent, "summary"), (parent_plan, "plan")):
        if document.get("episode_limit") != episode_limit:
            raise UserError(f"sizing parent {label} episode_limit mismatch")
        if document.get("episode_budget_per_worker") != episode_limit:
            raise UserError(f"sizing parent {label} episode budget mismatch")
        if document.get("initial_workers") != baseline_workers:
            raise UserError(f"sizing parent {label} did not start at five")
        if document.get("target_workers") != baseline_workers:
            raise UserError(f"sizing parent {label} did not target five")
    if parent.get("initial_group_mode") != "fresh_bootstrap":
        raise UserError("sizing parent is not a fresh five-worker baseline")
    if parent.get("sizing_parent") is not None or parent_plan.get(
        "sizing_parent"
    ) is not None:
        raise UserError("a sizing jump cannot serve as the five-worker parent")
    if (
        parent.get("status") != "completed"
        or parent.get("stop_reason") != "target_completed"
        or parent.get("recommended_cap") != baseline_workers
        or parent.get("launched_workers") != baseline_workers
        or parent.get("successful_workers") != baseline_workers
        or parent.get("failed_workers") != 0
        or parent.get("unlaunched_workers") != 0
    ):
        raise UserError("sizing parent five-worker run was not completely successful")
    initial_group = parent.get("initial_group")
    if not isinstance(initial_group, dict) or initial_group.get(
        "steady_confirmed"
    ) is not True:
        raise UserError("sizing parent initial group was not steady-confirmed")

    matching_levels = [
        item for item in parent.get("levels", [])
        if isinstance(item, dict)
        and item.get("active_count") == baseline_workers
    ]
    if len(matching_levels) != 1:
        raise UserError("sizing parent lacks exactly one five-worker level")
    level = matching_levels[0]
    if (
        level.get("steady_confirmed") is not True
        or level.get("evidence_origin") != "current_bootstrap_group"
    ):
        raise UserError("sizing parent five-worker level is not fresh evidence")
    try:
        idle_mib = float(parent["baseline_gpu_memory_mib"])
        steady_mib = float(level["steady_gpu_memory_mib"])
        peak_mib = float(level["peak_gpu_memory_mib"])
        observed_peak_mib = float(parent["peak_observed_gpu_memory_mib"])
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("sizing parent has invalid VRAM evidence") from error
    if (
        not all(
            math.isfinite(value)
            for value in (idle_mib, steady_mib, peak_mib, observed_peak_mib)
        )
        or idle_mib < 0
        or steady_mib <= idle_mib
        or peak_mib < steady_mib
        or peak_mib >= PLANNED_MEMORY_MIB
        or observed_peak_mib < peak_mib
        or observed_peak_mib >= PLANNED_MEMORY_MIB
        or resource_peak_mib != observed_peak_mib
        or resource_max_active < baseline_workers
    ):
        raise UserError("sizing parent VRAM evidence is outside the safe line")

    parent_plan_jobs = parent_plan.get("jobs")
    parent_jobs = parent.get("jobs")
    if (
        not isinstance(parent_plan_jobs, list)
        or len(parent_plan_jobs) != baseline_workers
        or not isinstance(parent_jobs, list)
        or len(parent_jobs) != baseline_workers
    ):
        raise UserError("sizing parent lacks exactly five job records")
    expected_tags = [job.get("run_tag") for job in selected_jobs[:baseline_workers]]
    if [job.get("source_run_tag") for job in parent_plan_jobs] != expected_tags:
        raise UserError("sizing parent candidate prefix differs from this phase")
    ordered_parent_jobs = sorted(
        parent_jobs, key=lambda item: item.get("worker_index", -1)
    )
    if [job.get("worker_index") for job in ordered_parent_jobs] != list(
        range(1, baseline_workers + 1)
    ) or [job.get("source_run_tag") for job in ordered_parent_jobs] != expected_tags:
        raise UserError("sizing parent summary candidate prefix is invalid")
    if any(
        job.get("exit_code") != 0
        or job.get("episode_budget") != episode_limit
        for job in ordered_parent_jobs
    ):
        raise UserError("sizing parent contains an incomplete job")

    current_identities = []
    for index in range(baseline_workers):
        current_config = _read_json(
            selected_jobs[index].get("config_path"), "current formal job config"
        )
        if current_config.get("episodes") != -1:
            raise UserError("current formal source config must keep episodes=-1")
        current_identity = _execution_config_identity(current_config)
        current_identity["episodes"] = episode_limit
        current_identities.append(current_identity)

        parent_job = parent_plan_jobs[index]
        parent_job_dir = Path(str(parent_job.get("job_dir", ""))).resolve()
        if not _is_relative_to(parent_job_dir, path.parent / "jobs"):
            raise UserError("sizing parent job directory escapes its artifact")
        try:
            exit_code = int(
                (parent_job_dir / "exitcode").read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError) as error:
            raise UserError("sizing parent job exitcode is invalid") from error
        if exit_code != 0:
            raise UserError("sizing parent job did not exit successfully")
        command = parent_job.get("command")
        if not isinstance(command, list) or command.count("--episode-limit") != 1:
            raise UserError("sizing parent command must bind one episode limit")
        option_index = command.index("--episode-limit")
        if (
            option_index + 1 >= len(command)
            or command[option_index + 1] != str(episode_limit)
        ):
            raise UserError("sizing parent command episode limit mismatch")
        parent_config = _read_json(
            parent_job.get("config_path"), "sizing parent cloned config"
        )
        if parent_config.get("episodes") != episode_limit:
            raise UserError("sizing parent cloned config episode limit mismatch")
        parent_identity = _execution_config_identity(parent_config)
        if parent_identity != current_identity:
            raise UserError(
                f"sizing parent config identity changed at candidate {index + 1}"
            )

    safety_factor = float(policy["projection_safety_factor"])
    # The whole-run peak includes launch/load transients that the level's
    # steady-window row may miss. Size from the larger observed value so a
    # brief allocation spike cannot authorize an unsafe jump.
    projection_peak_mib = max(peak_mib, observed_peak_mib)
    measured_peak_per_worker = (
        projection_peak_mib - idle_mib
    ) / baseline_workers
    measured_steady_per_worker = (steady_mib - idle_mib) / baseline_workers
    effective_per_worker = max(
        float(policy["estimated_mib_per_job"]),
        measured_peak_per_worker * safety_factor,
    )
    projected_mib = idle_mib + target_workers * effective_per_worker
    ceiling_mib = float(policy["projection_max_mib"])
    if projected_mib > ceiling_mib:
        raise UserError(
            "sizing jump projection exceeds the declared ceiling: "
            f"{projected_mib:.1f} > {ceiling_mib:.1f} MiB"
        )
    identities_sha256 = hashlib.sha256(
        _canonical(current_identities).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "plan_path": str(parent_plan_path),
        "plan_sha256": _sha256(parent_plan_path),
        "calibration_id": parent["calibration_id"],
        "baseline_workers": baseline_workers,
        "target_workers": target_workers,
        "episode_limit": episode_limit,
        "gpu": gpu,
        "source_git_commit": plan.get("git_commit"),
        "source_spec_sha256": plan.get("spec_sha256"),
        "baseline_gpu_memory_mib": round(idle_mib, 3),
        "steady_gpu_memory_mib": round(steady_mib, 3),
        "peak_gpu_memory_mib": round(peak_mib, 3),
        "peak_observed_gpu_memory_mib": round(observed_peak_mib, 3),
        "projection_peak_gpu_memory_mib": round(projection_peak_mib, 3),
        "measured_peak_increment_per_worker_mib": round(
            measured_peak_per_worker, 6
        ),
        "measured_steady_increment_per_worker_mib": round(
            measured_steady_per_worker, 6
        ),
        "declared_estimated_mib_per_job": int(
            policy["estimated_mib_per_job"]
        ),
        "projection_safety_factor": safety_factor,
        "effective_projected_mib_per_worker": round(
            effective_per_worker, 6
        ),
        "parent_idle_projected_gpu_memory_mib": round(projected_mib, 3),
        "projection_max_mib": int(policy["projection_max_mib"]),
        "source_job_config_identity_sha256": identities_sha256,
    }


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
            "level_kind": gate.get("level_kind") if gate else None,
            "previous_steady_gpu_memory_mib": (
                gate.get("previous_steady_gpu_memory_mib") if gate else None
            ),
            "prior_steady_gpu_memory_mib": (
                gate.get("prior_steady_gpu_memory_mib") if gate else None
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
            "initial_group_first_launch_elapsed_seconds": (
                gate.get("initial_group_first_launch_elapsed_seconds")
                if gate else None
            ),
            "initial_group_all_launched_elapsed_seconds": (
                gate.get("initial_group_all_launched_elapsed_seconds")
                if gate else None
            ),
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


def _merge_level_evidence(
    current_levels, prior_evidence, initial_workers, sizing_evidence=None
):
    """Merge a prior continuous prefix without presenting it as a current run."""
    current = {
        level["active_count"]: copy.deepcopy(level) for level in current_levels
    }
    if not initial_workers:
        for level in current.values():
            level["evidence_origin"] = "current"
        return [current[count] for count in sorted(current)]

    if prior_evidence is None:
        # A fresh grouped start proves the whole concurrent level directly;
        # samples while that group is still launching are observations, not
        # separate proofs for every lower cardinality.
        for count, level in current.items():
            if count < initial_workers:
                level["evidence_origin"] = "current_bootstrap_transient"
            elif count == initial_workers:
                level["evidence_origin"] = (
                    "current_sizing_jump_group"
                    if sizing_evidence is not None
                    else "current_bootstrap_group"
                )
            else:
                level["evidence_origin"] = "current"
        return [current[count] for count in sorted(current)]

    prior_prefix = prior_evidence.get("_validated_level_prefix", [])
    prior_by_count = {level.get("active_count"): level for level in prior_prefix}
    for count in range(1, initial_workers):
        source = prior_by_count.get(count)
        if source is None:
            raise UserError(f"validated prior prefix is missing level {count}")
        inherited = copy.deepcopy(source)
        replay = current.get(count)
        inherited.update({
            "evidence_origin": "prior",
            "evidence_calibration_path": prior_evidence["path"],
            "evidence_calibration_id": prior_evidence["calibration_id"],
            "evidence_calibration_sha256": prior_evidence["sha256"],
            "evidence_level_sha256": _level_evidence_sha256(source),
            "evidence_level_snapshot": copy.deepcopy(source),
            "current_replay_observation": (
                {
                    "sample_count": replay.get("sample_count"),
                    "peak_gpu_memory_mib": replay.get("peak_gpu_memory_mib"),
                    "tail_median_gpu_memory_mib": replay.get(
                        "tail_median_gpu_memory_mib"
                    ),
                }
                if replay is not None else None
            ),
        })
        current[count] = inherited

    level_n = current.get(initial_workers)
    source_n = prior_by_count.get(initial_workers)
    if source_n is None:
        raise UserError("validated prior prefix lacks level N")
    if level_n is None:
        level_n = {
            "active_count": initial_workers,
            "steady_confirmed": False,
            "level_status": "initial_group_not_started",
            "peak_gpu_memory_mib": None,
            "steady_gpu_memory_mib": None,
        }
        current[initial_workers] = level_n
    level_n.update({
        "evidence_origin": "current_initial_group_revalidation",
        "prior_calibration_id": prior_evidence["calibration_id"],
        "prior_calibration_sha256": prior_evidence["sha256"],
        "prior_level_evidence_sha256": _level_evidence_sha256(source_n),
    })
    for count, level in current.items():
        if count > initial_workers:
            level["evidence_origin"] = "current"
    return [current[count] for count in sorted(current)]


def _recommended_cap(
    levels, stop_reason, initial_workers=0, initial_group_confirmed=False
):
    by_count = {level["active_count"]: level for level in levels}
    safe = 0
    start = 1
    if initial_workers:
        initial = by_count.get(initial_workers)
        if (
            initial_group_confirmed
            and initial is not None
            and initial.get("steady_confirmed") is True
            and initial.get("peak_gpu_memory_mib") is not None
            and initial.get("steady_gpu_memory_mib") is not None
            and initial["peak_gpu_memory_mib"] < PLANNED_MEMORY_MIB
            and initial["steady_gpu_memory_mib"] < PLANNED_MEMORY_MIB
        ):
            safe = initial_workers
            start = initial_workers + 1
        else:
            return 0
    for count in range(start, max(by_count, default=0) + 1):
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
    emergency_observed_mib=None, telemetry_error=None, initial_group=None,
    runtime_prior_evidence=None, runtime_sizing_evidence=None,
    runtime_sizing_projection=None, resource_path=None,
):
    initial_workers = int(calibration_plan.get("initial_workers") or 0)
    levels = _merge_level_evidence(
        _level_summaries(samples, records, gate_levels),
        runtime_prior_evidence,
        initial_workers,
        sizing_evidence=runtime_sizing_evidence,
    )
    initial_group_confirmed = bool(
        initial_group is not None
        and initial_group.get("steady_confirmed") is True
    )
    cap = _recommended_cap(
        levels, stop_reason,
        initial_workers=initial_workers,
        initial_group_confirmed=initial_group_confirmed,
    )
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
    elif stop_reason == "sizing_projection_threshold":
        status = "safety_stop"
    elif stop_reason == "planned_memory_threshold":
        status = "safety_stop"
    elif initial_workers and not initial_group_confirmed:
        status = "failed"
    elif stop_reason == "worker_exited_before_steady":
        status = "inconclusive"
    elif failed or stop_reason in (
        "telemetry_failure", "launch_error", "worker_failure", "load_timeout"
    ):
        status = "failed"
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
        "level_evidence_schema": LEVEL_EVIDENCE_SCHEMA,
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
        "source_git_commit": calibration_plan.get("source_git_commit"),
        "source_spec_sha256": calibration_plan.get("source_spec_sha256"),
        "plan_git_commit": calibration_plan.get("plan_git_commit"),
        "prior_execution_git_commit": calibration_plan.get(
            "prior_execution_git_commit"
        ),
        "execution_git_commit": calibration_plan.get("execution_git_commit"),
        "allowed_helper_only_diff_files": calibration_plan.get(
            "allowed_helper_only_diff_files", []
        ),
        "gpu": calibration_plan.get("gpu"),
        "episode_limit": calibration_plan.get("episode_limit"),
        "episode_budget_per_worker": calibration_plan.get(
            "episode_budget_per_worker"
        ),
        "calibration_policy": calibration_plan.get("calibration_policy"),
        "prior_evidence": calibration_plan.get("prior_evidence"),
        "sizing_parent": calibration_plan.get("sizing_parent"),
        "sizing_projection": runtime_sizing_projection,
        "initial_workers": initial_workers,
        "initial_group_mode": (
            "prior_revalidation"
            if initial_workers and runtime_prior_evidence is not None
            else "sizing_jump"
            if initial_workers and runtime_sizing_evidence is not None
            else "fresh_bootstrap"
            if initial_workers
            else "adaptive"
        ),
        "initial_group": (
            {
                "steady_confirmed": initial_group_confirmed,
                "status": initial_group.get("status"),
                "required_loaded_gpu_memory_mib": initial_group.get(
                    "required_loaded_gpu_memory_mib"
                ),
                "prior_steady_gpu_memory_mib": initial_group.get(
                    "prior_steady_gpu_memory_mib"
                ),
                "steady_gpu_memory_mib": initial_group.get(
                    "steady_gpu_memory_mib"
                ),
                "load_wait_seconds": initial_group.get("load_wait_seconds"),
                "steady_window_seconds": initial_group.get(
                    "steady_window_seconds"
                ),
                "steady_window_sample_count": initial_group.get(
                    "steady_window_sample_count"
                ),
                "steady_relative_fluctuation": initial_group.get(
                    "steady_relative_fluctuation"
                ),
            }
            if initial_group is not None else None
        ),
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
        "resource_csv_sha256": _sha256(resource_path),
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
    prior_evidence=None,
    sizing_evidence=None,
    initial_workers=0,
):
    """Run one isolated calibration group and emit CALIBRATION.json."""
    started = time.monotonic()
    active = {}
    records = []
    samples = []
    gate_levels = []
    current_level = None
    initial_group = None
    next_initial_launch = None
    previous_steady_memory = None
    launched = 0
    stop_reason = None
    emergency_observed = None
    telemetry_error = None
    launch_error = None
    resource_path = Path(calibration_root) / "resource.csv"
    grouped_start = initial_workers > 0
    continuation = grouped_start and prior_evidence is not None
    sizing_jump = grouped_start and sizing_evidence is not None
    if continuation and sizing_jump:
        raise UserError("prior and sizing evidence cannot be used together")
    runtime_sizing_projection = None

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
                if (
                    worker_exited
                    and grouped_start
                    and initial_group is not None
                    and initial_group.get("steady_confirmed") is not True
                    and stop_reason is None
                ):
                    stop_reason = "initial_group_worker_exit"
                    initial_group["status"] = stop_reason
                    wait_start = initial_group.get(
                        "all_launched_monotonic",
                        initial_group["first_launched_monotonic"],
                    )
                    initial_group["load_wait_seconds"] = round(
                        now - wait_start, 6
                    )
                    _terminate_active(active, terminate_grace_seconds)
                    _record_exits(active, records, time.monotonic())
                    break
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
                    if initial_group is not None and not initial_group[
                        "steady_confirmed"
                    ]:
                        initial_group["status"] = stop_reason
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
                    if initial_group is not None and not initial_group[
                        "steady_confirmed"
                    ]:
                        initial_group["status"] = stop_reason
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
                    if initial_group is not None and not initial_group[
                        "steady_confirmed"
                    ]:
                        initial_group["status"] = stop_reason
                    if current_level is not None:
                        current_level["status"] = stop_reason
                        current_level["load_wait_seconds"] = round(
                            now - current_level["launched_monotonic"], 6
                        )

                ready_to_launch = False
                launch_kind = None
                rapid_initial_launch = False
                if (
                    stop_reason is None
                    and grouped_start
                    and (
                        initial_group is None
                        or initial_group.get("steady_confirmed") is not True
                    )
                ):
                    if initial_group is None:
                        previous_steady_memory = float(gpu_memory)
                        if sizing_jump:
                            effective_per_worker = float(
                                sizing_evidence[
                                    "effective_projected_mib_per_worker"
                                ]
                            )
                            projection_ceiling = float(
                                sizing_evidence["projection_max_mib"]
                            )
                            current_projection = (
                                previous_steady_memory
                                + initial_workers * effective_per_worker
                            )
                            runtime_sizing_projection = {
                                "formula": (
                                    "current_idle_mib + target_workers * "
                                    "effective_projected_mib_per_worker"
                                ),
                                "current_idle_gpu_memory_mib": round(
                                    previous_steady_memory, 3
                                ),
                                "target_workers": initial_workers,
                                "effective_projected_mib_per_worker": round(
                                    effective_per_worker, 6
                                ),
                                "projected_gpu_memory_mib": round(
                                    current_projection, 3
                                ),
                                "projection_max_mib": round(
                                    projection_ceiling, 3
                                ),
                                "accepted": current_projection
                                <= projection_ceiling,
                            }
                            if current_projection > projection_ceiling:
                                stop_reason = "sizing_projection_threshold"
                                break
                        if continuation:
                            prior_steady = float(
                                prior_evidence[
                                    "level_n_steady_gpu_memory_mib"
                                ]
                            )
                            recovery_floor = max(
                                prior_steady * (
                                    1.0 - PRIOR_RECOVERY_RELATIVE_TOLERANCE
                                ),
                                previous_steady_memory
                                + initial_workers * int(min_loaded_memory_mib),
                            )
                            level_kind = (
                                "prior_safe_initial_group_recovery"
                            )
                        elif sizing_jump:
                            prior_steady = float(
                                sizing_evidence["steady_gpu_memory_mib"]
                            )
                            steady_per_worker = float(
                                sizing_evidence[
                                    "measured_steady_increment_per_worker_mib"
                                ]
                            )
                            recovery_floor = previous_steady_memory + (
                                initial_workers * max(
                                    float(min_loaded_memory_mib),
                                    steady_per_worker
                                    * (1.0 - PRIOR_RECOVERY_RELATIVE_TOLERANCE),
                                )
                            )
                            level_kind = "sizing_jump_initial_group"
                        else:
                            prior_steady = None
                            recovery_floor = (
                                previous_steady_memory
                                + initial_workers * int(min_loaded_memory_mib)
                            )
                            level_kind = "fresh_initial_group_bootstrap"
                        initial_group = {
                            "active_count": initial_workers,
                            "level_kind": level_kind,
                            "status": "launching_initial_group",
                            "steady_confirmed": False,
                            "previous_steady_gpu_memory_mib": round(
                                previous_steady_memory, 3
                            ),
                            "prior_steady_gpu_memory_mib": round(
                                prior_steady, 3
                            ) if prior_steady is not None else None,
                            "required_loaded_gpu_memory_mib": round(
                                recovery_floor, 3
                            ),
                            "loaded_memory_increment_mib": None,
                            "first_launched_monotonic": now,
                            "all_launched_monotonic": None,
                            "launched_monotonic": None,
                            "launched_elapsed_seconds": None,
                            "initial_group_first_launch_elapsed_seconds": round(
                                now - started, 6
                            ),
                            "initial_group_all_launched_elapsed_seconds": None,
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
                        gate_levels.append(initial_group)
                        ready_to_launch = True
                        launch_kind = "initial_group"
                    elif launched < initial_workers:
                        if now >= next_initial_launch:
                            ready_to_launch = True
                            launch_kind = "initial_group"
                    else:
                        initial_group["observed_sample_count"] += 1
                        if gpu_memory >= initial_group[
                            "required_loaded_gpu_memory_mib"
                        ]:
                            if initial_group[
                                "first_loaded_elapsed_seconds"
                            ] is None:
                                initial_group[
                                    "first_loaded_elapsed_seconds"
                                ] = round(now - started, 6)
                            initial_group["_loaded_samples"].append({
                                "monotonic": now,
                                "gpu_memory_mib": gpu_memory,
                            })
                        else:
                            initial_group["_loaded_samples"].clear()
                        stable = _steady_window(
                            initial_group["_loaded_samples"], now,
                            steady_seconds, steady_samples,
                            steady_relative_tolerance,
                        )
                        wait_start = initial_group["all_launched_monotonic"]
                        if stable is not None:
                            initial_group.update(stable)
                            initial_group["steady_confirmed"] = True
                            initial_group["status"] = "steady_confirmed"
                            initial_group["load_wait_seconds"] = round(
                                now - wait_start, 6
                            )
                            initial_group[
                                "steady_confirmed_elapsed_seconds"
                            ] = round(now - started, 6)
                            initial_group["loaded_memory_increment_mib"] = round(
                                initial_group["steady_gpu_memory_mib"]
                                - initial_group[
                                    "previous_steady_gpu_memory_mib"
                                ],
                                3,
                            )
                            previous_steady_memory = initial_group[
                                "steady_gpu_memory_mib"
                            ]
                            ready_to_launch = launched < len(jobs)
                            launch_kind = "adaptive" if ready_to_launch else None
                        elif now - wait_start >= load_timeout_seconds:
                            stop_reason = "initial_group_load_timeout"
                            initial_group["status"] = stop_reason
                            initial_group["load_wait_seconds"] = round(
                                now - wait_start, 6
                            )
                            _terminate_active(active, terminate_grace_seconds)
                            _record_exits(active, records, time.monotonic())
                            break
                elif stop_reason is None and current_level is not None:
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
                    launch_kind = "adaptive"

                if stop_reason is None and ready_to_launch:
                    job = jobs[launched]
                    try:
                        process, console = _launch(job)
                    except Exception as error:  # preserve partial evidence
                        launch_error = f"{type(error).__name__}: {error}"
                        stop_reason = "launch_error"
                        if initial_group is not None and not initial_group[
                            "steady_confirmed"
                        ]:
                            initial_group["status"] = stop_reason
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
                    if launch_kind == "initial_group":
                        next_initial_launch = now + stagger_seconds
                        rapid_initial_launch = (
                            stagger_seconds == 0
                            and launched < initial_workers
                        )
                        if launched == initial_workers:
                            initial_group["status"] = (
                                "waiting_for_initial_group_load"
                            )
                            initial_group["all_launched_monotonic"] = now
                            initial_group["launched_monotonic"] = now
                            initial_group["launched_elapsed_seconds"] = round(
                                now - started, 6
                            )
                            initial_group[
                                "initial_group_all_launched_elapsed_seconds"
                            ] = round(now - started, 6)
                    else:
                        current_level = {
                            "active_count": launched,
                            "level_kind": "adaptive_extension_level",
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
                    if not rapid_initial_launch:
                        time.sleep(sample_interval_seconds)
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            if initial_group is not None and not initial_group[
                "steady_confirmed"
            ]:
                initial_group["status"] = stop_reason
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
    if (
        initial_group is not None
        and not initial_group["steady_confirmed"]
        and initial_group.get("load_wait_seconds") is None
    ):
        wait_start = initial_group.get(
            "all_launched_monotonic",
            initial_group["first_launched_monotonic"],
        )
        initial_group["load_wait_seconds"] = round(
            time.monotonic() - wait_start, 6
        )
    if launch_error is not None:
        telemetry_error = launch_error
    summary = _build_summary(
        calibration_plan, samples, records, gate_levels, started, stop_reason,
        emergency_observed_mib=emergency_observed,
        telemetry_error=telemetry_error,
        initial_group=initial_group,
        runtime_prior_evidence=prior_evidence,
        runtime_sizing_evidence=sizing_evidence,
        runtime_sizing_projection=runtime_sizing_projection,
        resource_path=resource_path,
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
        plan, phase, phase_manifest, jobs = _load_phase(
            campaign_root, args.phase_id
        )
        if plan.get("batch_id") != args.batch_id:
            raise UserError(
                f"batch id mismatch: CLI={args.batch_id}, PLAN={plan.get('batch_id')}"
            )
        if plan.get("gpu") != args.gpu:
            raise UserError(
                f"GPU mismatch: CLI={args.gpu}, PLAN={plan.get('gpu')}"
            )
        grouped_policy = _validated_grouped_policy(phase_manifest)
        if args.prior_calibration is not None:
            raise UserError(
                "this phase uses independent grouped sizing; "
                "--prior-calibration is not permitted"
            )
        if args.episode_limit != grouped_policy["episode_limit"]:
            raise UserError(
                "grouped calibration must use the phase's 100-episode prefix"
            )
        if args.initial_workers != args.target_workers:
            raise UserError(
                "grouped calibration requires initial_workers == target_workers"
            )
        if args.target_workers not in grouped_policy["allowed_worker_counts"]:
            raise UserError(
                "grouped calibration worker count is not declared by the phase"
            )
        baseline_workers = grouped_policy["initial_group_workers"]
        if args.target_workers == baseline_workers and args.sizing_parent:
            raise UserError("the five-worker baseline must not have a sizing parent")
        if args.target_workers > baseline_workers and not args.sizing_parent:
            raise UserError(
                "worker counts above five require --sizing-parent evidence"
            )
        revision_audit = _assert_plan_revision(
            plan, allow_helper_only_drift=False
        )
        selected = _select_distinct_jobs(jobs, args.target_workers)
        prior_evidence = None
        sizing_evidence = None
        if args.sizing_parent is not None:
            sizing_evidence = _load_sizing_parent_evidence(
                args.sizing_parent, campaign_root, plan, phase, selected,
                args.gpu, grouped_policy,
            )
        public_prior_evidence = (
            {
                key: value for key, value in prior_evidence.items()
                if not key.startswith("_")
            }
            if prior_evidence is not None else None
        )
        public_sizing_evidence = (
            copy.deepcopy(sizing_evidence)
            if sizing_evidence is not None else None
        )
        calibration_root, clones, calibration_plan = _clone_jobs(
            campaign_root, plan, phase, selected, args.calibration_id,
            episode_limit=args.episode_limit,
        )
        calibration_plan.update({
            "source_git_commit": (
                prior_evidence["source_git_commit"]
                if prior_evidence is not None else plan.get("git_commit")
            ),
            "plan_git_commit": revision_audit["plan_git_commit"],
            "execution_git_commit": revision_audit["execution_git_commit"],
            "prior_execution_git_commit": (
                prior_evidence.get("prior_execution_git_commit")
                if prior_evidence is not None else None
            ),
            "allowed_helper_only_diff_files": sorted(set(
                revision_audit["allowed_helper_only_diff_files"]
                + (
                    prior_evidence["allowed_helper_only_diff_files"]
                    if prior_evidence is not None else []
                )
            )),
            "gpu": args.gpu,
            "stagger_seconds": args.stagger_seconds,
            "sample_interval_seconds": args.sample_interval_seconds,
            "steady_seconds": args.steady_seconds,
            "steady_samples": args.steady_samples,
            "load_timeout_seconds": args.load_timeout_seconds,
            "min_loaded_memory_mib": args.min_loaded_memory_mib,
            "steady_relative_tolerance": args.steady_relative_tolerance,
            "initial_workers": args.initial_workers,
            "calibration_policy": grouped_policy,
            "prior_recovery_relative_tolerance": (
                PRIOR_RECOVERY_RELATIVE_TOLERANCE
            ),
            "prior_evidence": public_prior_evidence,
            "sizing_parent": public_sizing_evidence,
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
            prior_evidence=prior_evidence,
            sizing_evidence=sizing_evidence,
            initial_workers=args.initial_workers,
        )


def _default_calibration_id():
    return time.strftime("%Y%m%dT%H%M%S")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Safely validate grouped concurrency for one planned phase."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--target-workers", required=True, type=int)
    parser.add_argument(
        "--prior-calibration",
        help=(
            "prior CALIBRATION.json proving the exact continuous safe prefix "
            "used by --initial-workers"
        ),
    )
    parser.add_argument(
        "--sizing-parent",
        help=(
            "fresh five-worker CALIBRATION.json whose measured peak VRAM "
            "authorizes one independently validated larger group"
        ),
    )
    parser.add_argument(
        "--initial-workers", type=int, default=0,
        help=(
            "launch this many workers as one measured initial group; with "
            "--prior-calibration it revalidates that proven prefix, otherwise "
            "it is a fresh bootstrap level"
        ),
    )
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
    if args.prior_calibration is not None and args.sizing_parent is not None:
        parser.error("--prior-calibration and --sizing-parent are mutually exclusive")
    if args.prior_calibration is not None and args.initial_workers == 0:
        parser.error(
            "--prior-calibration requires a positive --initial-workers"
        )
    if args.initial_workers < 0:
        parser.error("--initial-workers must be non-negative")
    if args.initial_workers > args.target_workers:
        parser.error("--initial-workers cannot exceed --target-workers")
    if (
        args.prior_calibration is not None
        and args.initial_workers >= args.target_workers
    ):
        parser.error(
            "a prior continuation requires --initial-workers below "
            "--target-workers"
        )
    if args.prior_calibration is None:
        if args.initial_workers != args.target_workers:
            parser.error(
                "grouped sizing requires --initial-workers equal to "
                "--target-workers"
            )
        if args.episode_limit != 100:
            parser.error("grouped sizing requires --episode-limit 100")
        if args.target_workers < 5:
            parser.error("grouped sizing starts at five workers, never one")
        if args.target_workers == 5 and args.sizing_parent is not None:
            parser.error("the five-worker baseline cannot have --sizing-parent")
        if args.target_workers > 5 and args.sizing_parent is None:
            parser.error("worker counts above five require --sizing-parent")
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
    if summary["status"] in (
        "emergency_abort", "failed", "inconclusive", "safety_stop"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except UserError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
