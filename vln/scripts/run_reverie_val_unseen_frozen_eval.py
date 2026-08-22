#!/usr/bin/env python3
"""Run frozen REVERIE val_unseen winners with strict model barriers.

The plan is generated after val_seen selection by
``prepare_reverie_val_unseen_frozen_eval.py``.  Source is never launched.
DUET, HAMT, then GOAT execute as sequential model phases; methods inside one
phase are scheduled concurrently under the reviewed GPU-memory cap.
"""

import argparse
from collections import Counter
import csv
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
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (str(REPO_ROOT), str(SCRIPT_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

import build_reverie_final_registry as registry_builder  # noqa: E402
import prepare_reverie_val_unseen_frozen_eval as plan_builder  # noqa: E402
from joint_campaign_contract import process_identity, process_identity_alive  # noqa: E402
from shared_gpu_launch_guard import (  # noqa: E402
    ReservationLedgerError,
    release_shared_gpu_reservation,
    shared_gpu_launch_guard,
)
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


SCHEMA = plan_builder.SCHEMA
JOB_SCHEMA = "navtta.vln_reverie_val_unseen_frozen_job.v1"
SUMMARY_SCHEMA = "navtta.vln_reverie_val_unseen_frozen_summary.v1"
BATCH_SCHEMA = "navtta.vln_reverie_val_unseen_frozen_batch.v1"
DEFAULT_SPEC = (
    REPO_ROOT / "vln/experiments/reverie_val_unseen_frozen_eval_v1.json"
)
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
LOG_ROOT = REPO_ROOT / "vln/results/logs/reverie/frozen_val_unseen"
TEST_LOG_ROOT = REPO_ROOT / "vln/results/logs/reverie/test_submissions"
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"
SETTINGS = plan_builder.SETTINGS
MODELS = plan_builder.MODELS
METHODS = plan_builder.METHODS
MODEL_FOR_SETTING = plan_builder.MODEL_FOR_SETTING
EXPECTED_BENCHMARK = registry_builder.EXPECTED_BENCHMARK
METRIC_RE = re.compile(
    r"\b(sr|spl|rgs|rgspl):\s*(-?[0-9]+(?:\.[0-9]+)?)"
)
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
EXECUTION_SURFACE = (
    "core", "tools", "vln/baselines", "vln/navtta_vln",
    "vln/scripts", "vln/experiments", "vln/manifests",
)
PROCESS_GROUP_ENV = "NAVTTA_REVERIE_PROCESS_GROUP_TOKEN"


class UserError(RuntimeError):
    pass


class LaunchCleanupError(UserError):
    """A launch failed and its process group could not be proven dead."""

    def __init__(self, message, process=None):
        super().__init__(message)
        self.process = process


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("cannot read JSON {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise UserError("JSON must be an object: {}".format(path))
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_process_group_token(metadata, attempt_dir):
    del attempt_dir
    try:
        payload = {
            "schema": "navtta.vln_reverie_process_group.v1",
            "batch_id": metadata["batch_id"],
            "run_tag": metadata["run_tag"],
            "attempt": metadata["attempt"],
            "setting": metadata["setting"],
            "method": metadata["method"],
            "spec_sha256": metadata["spec_sha256"],
            "git_commit": metadata["git_commit"],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("attempt process-group binding is malformed") from error
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _valid_sha256(value):
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _assert_clean_execution_tree(spec_path):
    dirty = subprocess.check_output([
        "git", "-C", str(REPO_ROOT), "status", "--porcelain",
        "--untracked-files=no",
    ], text=True).strip()
    if dirty:
        raise UserError("tracked worktree must be clean before formal execution")
    spec = _read_json(spec_path)
    registry_path = _resolve(spec.get("registry_dependency", {}).get("path", ""))
    registry = _read_json(registry_path)
    selected_path = _resolve(registry.get("selected_winners", {}).get("path", ""))
    dependency_paths = [
        Path(__file__).resolve(),
        Path(spec_path).resolve(),
        Path(registry_builder.__file__).resolve(),
        Path(plan_builder.__file__).resolve(),
        RUNNER.resolve(),
        registry_path.resolve(),
        selected_path.resolve(),
        _resolve(spec.get("source_control", {}).get("manifest", "")).resolve(),
        _resolve(spec.get("test_submission_policy", {}).get(
            "source", {}
        ).get("ledger", "")).resolve(),
    ]
    for section in (
        spec.get("protocol", {}).get("order_manifests", {}),
        spec.get("test_submission_policy", {}).get("order_manifests", {}),
    ):
        dependency_paths.extend(
            _resolve(binding.get("path", "")).resolve()
            for binding in section.values()
        )
    for setting in SETTINGS:
        dependency_paths.extend(
            _resolve(record.get("formal_manifest_path", "")).resolve()
            for record in registry.get("records", {}).get(setting, {}).values()
        )
    for ledger_path in dependency_paths[7:9]:
        ledger = _read_json(ledger_path)
        dependency_paths.extend(
            _resolve(record.get("formal_manifest_path", "")).resolve()
            for record in ledger.get("records", {}).values()
        )
    relative = []
    for path in dependency_paths:
        try:
            relative.append(path.relative_to(REPO_ROOT.resolve()).as_posix())
        except ValueError:
            raise UserError("runner and spec must be inside the repository")
    tracked = subprocess.run([
        "git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", "--",
        *relative,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if tracked.returncode != 0:
        raise UserError(
            "runner, spec, registry, ledgers, order manifests, and all selected/"
            "Source formal manifests must be tracked by current HEAD"
        )
    untracked = subprocess.run([
        "git", "-C", str(REPO_ROOT), "ls-files", "--others",
        "--exclude-standard", "--", *EXECUTION_SURFACE,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if untracked.returncode != 0:
        raise UserError("cannot inspect untracked execution files")
    values = [line for line in untracked.stdout.splitlines() if line]
    if values:
        raise UserError("untracked execution files: {}".format(", ".join(values)))


def _load_registry(spec):
    binding = spec.get("registry_dependency", {})
    path = _resolve(binding.get("path", "")).resolve()
    if not path.is_file() or _sha256(path) != binding.get("sha256"):
        raise UserError("REVERIE final registry digest mismatch")
    raw = _read_json(path)
    selection = _resolve(raw.get("selected_winners", {}).get("path", ""))
    try:
        registry = registry_builder.validate_registry(path, selection)
    except registry_builder.RegistryError as error:
        raise UserError("REVERIE registry provenance failed: {}".format(error))
    if (
        binding.get("schema") != registry_builder.REGISTRY_SCHEMA
        or registry.get("registry_status") != binding.get("required_status")
    ):
        raise UserError("REVERIE registry status/schema mismatch")
    return path, registry


def _validate_matrix(spec, registry):
    declared = spec["matrix"]["jobs"]
    pairs = []
    for item in declared:
        setting, method = item["setting"], item["method"]
        record = registry["records"][setting][method]
        if (
            item.get("selected_run_tag") != record.get("run_tag")
            or item.get("selected_formal_manifest_sha256")
            != record.get("formal_manifest_sha256")
        ):
            raise UserError("winner binding mismatch for {} {}".format(setting, method))
        pairs.append((setting, method))
    expected = [(setting, method) for setting in SETTINGS for method in METHODS]
    if pairs != expected:
        raise UserError("winner jobs must be model-major DUET -> HAMT -> GOAT")


def _validate_order_bindings(spec):
    bindings = spec["protocol"].get("order_manifests", {})
    expected = {
        "duet_hamt": (
            ("duet-reverie", "hamt-reverie"),
            "reverie_discrete_duet_hamt",
        ),
        "goat": (("goat-reverie",), "reverie_discrete_goat"),
    }
    if set(bindings) != set(expected):
        raise UserError("both REVERIE val_unseen order manifests are required")
    for key, (settings, benchmark) in expected.items():
        binding = bindings[key]
        if tuple(binding.get("settings", ())) != settings:
            raise UserError("{} order setting binding mismatch".format(key))
        try:
            plan_builder._validate_order_manifest(
                _resolve(binding["path"]), binding["sha256"], benchmark,
                "val_unseen", plan_builder.VAL_UNSEEN_EPISODES,
                plan_builder.VAL_UNSEEN_ORDER,
            )
        except plan_builder.PlanError as error:
            raise UserError(str(error))


def _validate_test_order_bindings(spec):
    bindings = spec["test_submission_policy"].get("order_manifests", {})
    expected = {
        "duet_hamt": (
            ("duet-reverie", "hamt-reverie"),
            "reverie_discrete_duet_hamt",
        ),
        "goat": (("goat-reverie",), "reverie_discrete_goat"),
    }
    if set(bindings) != set(expected):
        raise UserError("both REVERIE test order manifests are required")
    for key, (settings, benchmark) in expected.items():
        binding = bindings[key]
        if tuple(binding.get("settings", ())) != settings:
            raise UserError("{} test order setting binding mismatch".format(key))
        try:
            plan_builder._validate_order_manifest(
                _resolve(binding["path"]), binding["sha256"], benchmark,
                "test", plan_builder.TEST_EPISODES, plan_builder.TEST_ORDER,
            )
        except plan_builder.PlanError as error:
            raise UserError(str(error))


def load_spec(path=DEFAULT_SPEC):
    path = Path(path).resolve()
    spec = _read_json(path)
    try:
        plan_builder.validate_spec_document(spec)
    except plan_builder.PlanError as error:
        raise UserError(str(error))
    _, registry = _load_registry(spec)
    _validate_matrix(spec, registry)
    source = spec["source_control"]
    source_path = _resolve(source.get("manifest", "")).resolve()
    if (
        source.get("execution") != "reuse_only"
        or source.get("rerun_forbidden") is not True
        or not source_path.is_file()
        or _sha256(source_path) != source.get("sha256")
    ):
        raise UserError("val_unseen Source binding/reuse policy mismatch")
    try:
        _, val_unseen_source = plan_builder.validate_val_unseen_source(source_path)
    except plan_builder.PlanError as error:
        raise UserError(str(error))
    policy = spec["test_submission_policy"]
    test_source = policy["source"]
    test_path = _resolve(test_source["ledger"]).resolve()
    if not test_path.is_file() or _sha256(test_path) != test_source["sha256"]:
        raise UserError("hidden-test Source-submission ledger mismatch")
    try:
        _, test_source = plan_builder.validate_test_source(test_path)
        plan_builder.validate_cross_split_checkpoints(
            registry, val_unseen_source, test_source
        )
    except plan_builder.PlanError as error:
        raise UserError(str(error))
    _validate_order_bindings(spec)
    _validate_test_order_bindings(spec)
    return spec


def _job_digest(setting, method, record):
    identity = {
        "setting": setting, "method": method,
        "selected_run_tag": record["run_tag"],
        "parameters": record["parameters"],
    }
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:10]


def expand_jobs(spec, batch_id, gpu=0):
    _, registry = _load_registry(spec)
    _validate_matrix(spec, registry)
    jobs = []
    ordinal = 0
    for model_index, setting in enumerate(SETTINGS):
        for method_index, method in enumerate(METHODS):
            record = registry["records"][setting][method]
            base_tag = "{}-frozen-{:02d}-{}-{:02d}-{}-{}".format(
                batch_id, model_index, MODEL_FOR_SETTING[setting], method_index,
                method, _job_digest(setting, method, record),
            )
            jobs.append({
                "setting": setting,
                "model": MODEL_FOR_SETTING[setting],
                "method": method,
                "model_index": model_index,
                "method_index": method_index,
                "ordinal": ordinal,
                "base_run_tag": base_tag,
                "parameters": json.loads(_canonical(record["parameters"])),
                "selected_anchor": json.loads(_canonical(record)),
                "gpu": int(gpu),
            })
            ordinal += 1
    return jobs


def model_phases(jobs):
    phases = []
    for index, (setting, model) in enumerate(zip(SETTINGS, MODELS)):
        phase = [item for item in jobs if item["model_index"] == index]
        if (
            [item["method"] for item in phase] != list(METHODS)
            or any(item["setting"] != setting for item in phase)
            or any(item["model"] != model for item in phase)
        ):
            raise UserError("{} phase is not the exact five-method matrix".format(model))
        phases.append(phase)
    return phases


def expand_test_jobs(spec, batch_id, gpu=0):
    """Expand only methods legal without hidden-test feedback."""
    _, registry = _load_registry(spec)
    policy = spec["test_submission_policy"]
    methods = tuple(policy["eligible_tta_methods"])
    jobs = []
    ordinal = 0
    for model_index, setting in enumerate(SETTINGS):
        for method_index, method in enumerate(methods):
            try:
                plan_builder.assert_test_method_allowed(policy, method)
            except plan_builder.PlanError as error:
                raise UserError(str(error))
            record = registry["records"][setting][method]
            base_tag = "{}-submission-{:02d}-{}-{:02d}-{}-{}".format(
                batch_id, model_index, MODEL_FOR_SETTING[setting], method_index,
                method, _job_digest(setting, method, record),
            )
            jobs.append({
                "setting": setting,
                "model": MODEL_FOR_SETTING[setting],
                "method": method,
                "model_index": model_index,
                "method_index": method_index,
                "ordinal": ordinal,
                "base_run_tag": base_tag,
                "parameters": json.loads(_canonical(record["parameters"])),
                "selected_anchor": json.loads(_canonical(record)),
                "gpu": int(gpu),
            })
            ordinal += 1
    if len(jobs) != 9 or any(
        item["method"] in ("feedtta", "atena") for item in jobs
    ):
        raise UserError("hidden-test plan must be exactly nine unsupervised TTA jobs")
    return jobs


def test_model_phases(jobs):
    methods = ("tent", "fstta", "eam")
    phases = []
    for index, (setting, model) in enumerate(zip(SETTINGS, MODELS)):
        phase = [item for item in jobs if item["model_index"] == index]
        if (
            [item["method"] for item in phase] != list(methods)
            or any(item["setting"] != setting for item in phase)
        ):
            raise UserError("{} hidden-test phase is not exact".format(model))
        phases.append(phase)
    return phases


def _attempt_tag(base, attempt):
    return base if attempt == 0 else "{}-retry{}".format(base, attempt)


def _attempt_dir(batch_root, job, attempt):
    return (
        Path(batch_root) / "models"
        / "{:02d}-{}".format(job["model_index"], job["model"])
        / "jobs" / job["base_run_tag"] / "attempt-{:02d}".format(attempt)
    )


def _reservation_token(batch_id, run_tag):
    return "reverie-val-unseen:{}:{}".format(batch_id, run_tag)


def materialize_attempt(spec, spec_path, batch_id, batch_root, job, attempt):
    run_tag = _attempt_tag(job["base_run_tag"], attempt)
    attempt_dir = _attempt_dir(batch_root, job, attempt)
    # Non-val_seen jobs must use run_source_eval.sh's canonical default.
    # --result-root is intentionally restricted to val_seen tuning jobs.
    result_root = (
        REPO_ROOT / "vln/results/tuning" / run_tag / job["setting"]
        / "val_unseen"
    )
    formal_manifest = (
        FORMAL_ROOT
        / "{}-{}-val_unseen-native".format(run_tag, job["setting"])
        / "manifest.json"
    )
    source_path, source = plan_builder.validate_val_unseen_source(
        _resolve(spec["source_control"]["manifest"])
    )
    source_record = source["records"][job["setting"]]
    order_key = "goat" if job["setting"] == "goat-reverie" else "duet_hamt"
    order_binding = spec["protocol"]["order_manifests"][order_key]
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
            "selection_benchmark": "reverie",
            "selection_split": "val_seen",
            "evaluation_split": "val_unseen",
            "selection_on_val_unseen": False,
            "canonical_order_seed": 0,
            "registry": spec["registry_dependency"],
            "selected_anchor": job["selected_anchor"],
            "source_ledger": {
                "path": spec["source_control"]["manifest"],
                "sha256": _sha256(source_path),
            },
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
        "episode_count": spec["protocol"]["episode_count"],
        "canonical_order_seed": 0,
        "git_commit": _git_commit(),
        "expected_benchmark": EXPECTED_BENCHMARK[job["setting"]],
        "expected_checkpoint_sha256": source_record["checkpoint_sha256"],
        "expected_dataset_sha256": source_record["dataset_sha256"],
        "expected_episode_order_sha256": plan_builder.VAL_UNSEEN_ORDER,
        "episode_order_manifest": str(_resolve(order_binding["path"])),
        "source_ledger_path": str(source_path),
        "source_ledger_sha256": _sha256(source_path),
        "result_root": str(result_root),
        "formal_manifest": str(formal_manifest),
        "command": command,
    }
    _atomic_json(config_path, config)
    metadata["parameters_path"] = str(config_path)
    metadata["parameters_sha256"] = _sha256(config_path)
    metadata["process_group_token"] = _expected_process_group_token(
        metadata, attempt_dir
    )
    _atomic_json(attempt_dir / "job.json", metadata)
    return attempt_dir, metadata


def _test_order_binding(spec, setting):
    key = "goat" if setting == "goat-reverie" else "duet_hamt"
    return spec["test_submission_policy"]["order_manifests"][key]


def materialize_test_attempt(spec, spec_path, batch_id, batch_root, job, attempt):
    try:
        plan_builder.assert_test_method_allowed(
            spec["test_submission_policy"], job["method"]
        )
    except plan_builder.PlanError as error:
        raise UserError(str(error))
    run_tag = _attempt_tag(job["base_run_tag"], attempt)
    attempt_dir = _attempt_dir(batch_root, job, attempt)
    # Hidden test uses the same canonical runner-owned output convention.
    result_root = (
        REPO_ROOT / "vln/results/tuning" / run_tag / job["setting"] / "test"
    )
    formal_manifest = (
        FORMAL_ROOT / "{}-{}-test-native".format(run_tag, job["setting"])
        / "manifest.json"
    )
    test_ledger_path, test_ledger = plan_builder.validate_test_source(
        _resolve(spec["test_submission_policy"]["source"]["ledger"])
    )
    source_record = test_ledger["records"][job["setting"]]
    source_manifest = _read_json(_resolve(source_record["formal_manifest_path"]))
    order_binding = _test_order_binding(spec, job["setting"])
    order_document = _read_json(_resolve(order_binding["path"]))
    config = {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": batch_id,
        "stage": "frozen_test_submission",
        "setting": job["setting"],
        "method": job["method"],
        "search_method": job["method"],
        "episodes": -1,
        "parameters": job["parameters"],
        "frozen_evaluation_provenance": {
            "selection_benchmark": "reverie",
            "selection_split": "val_seen",
            "evaluation_split": "test",
            "selection_on_test": False,
            "hidden_ground_truth": True,
            "submission_generation_only": True,
            "canonical_order_seed": 0,
            "registry": spec["registry_dependency"],
            "selected_anchor": job["selected_anchor"],
            "source_submission_ledger": {
                "path": spec["test_submission_policy"]["source"]["ledger"],
                "sha256": _sha256(test_ledger_path),
            },
        },
    }
    config_path = attempt_dir / "parameters.json"
    command = [
        str(RUNNER), job["setting"], "test", str(job["gpu"]),
        "--run-tag", run_tag, "--tta-config", str(config_path),
    ]
    metadata = {
        "schema": "navtta.vln_reverie_test_submission_job.v1",
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
        "episode_count": plan_builder.TEST_EPISODES,
        "canonical_order_seed": 0,
        "hidden_ground_truth": True,
        "submission_generation_only": True,
        "git_commit": _git_commit(),
        "expected_benchmark": EXPECTED_BENCHMARK[job["setting"]],
        "expected_checkpoint_sha256": source_manifest["checkpoint"]["sha256"],
        "expected_dataset_sha256": order_document["dataset"]["sha256"],
        "expected_episode_order_sha256": plan_builder.TEST_ORDER,
        "episode_order_manifest": str(_resolve(order_binding["path"])),
        "test_source_ledger_path": str(test_ledger_path),
        "test_source_ledger_sha256": _sha256(test_ledger_path),
        "result_root": str(result_root),
        "formal_manifest": str(formal_manifest),
        "command": command,
    }
    _atomic_json(config_path, config)
    metadata["parameters_path"] = str(config_path)
    metadata["parameters_sha256"] = _sha256(config_path)
    metadata["process_group_token"] = _expected_process_group_token(
        metadata, attempt_dir
    )
    _atomic_json(attempt_dir / "job.json", metadata)
    return attempt_dir, metadata


def _existing_attempts(batch_root, job):
    root = _attempt_dir(batch_root, job, 0).parent
    if not root.is_dir():
        return []
    values = []
    for path in root.glob("attempt-[0-9][0-9]"):
        try:
            values.append((int(path.name.rsplit("-", 1)[-1]), path))
        except ValueError:
            pass
    return sorted(values)


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
    """Find live members authenticated by token, process group, and session."""
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
    job_path = attempt_dir / "job.json"
    if not job_path.is_file():
        raise UserError("attempt with a process-group ID lacks job metadata")
    metadata = _read_json(job_path)
    identities = []
    identity_path = attempt_dir / "process_identity.json"
    if identity_path.is_file():
        leader = _read_json(identity_path)
        if leader.get("pid") != pgid:
            raise UserError("worker identity differs from its process group")
        if _identity_is_group_member(leader, pgid):
            identities.append(leader)
    token = metadata.get("process_group_token")
    if (
        not _valid_sha256(token)
        or token != _expected_process_group_token(metadata, attempt_dir)
    ):
        raise UserError("attempt process-group token is invalid")
    known = {item["pid"] for item in identities}
    identities.extend(
        item for item in _proc_group_member_identities(pgid, token)
        if item["pid"] not in known
    )
    return sorted(identities, key=lambda item: item["pid"])


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


def _state_for_attempt(path):
    path = Path(path)
    if _pid_alive(path):
        return "running"
    if (path / "validation_error.json").is_file():
        return "invalid"
    if (path / "metrics.json").is_file() or (path / "submission.json").is_file():
        return "completed"
    exit_path = path / "exitcode"
    if exit_path.is_file():
        try:
            return "finished" if int(exit_path.read_text().strip()) == 0 else "failed"
        except ValueError:
            return "invalid"
    if (path / "job.json").is_file():
        return "orphaned"
    return "pending"


def _latest_attempt(batch_root, job):
    values = _existing_attempts(batch_root, job)
    return values[-1] if values else None


def _parse_metrics(result_root):
    matches = []
    for path in sorted(Path(result_root).rglob("valid.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            values = {
                key.upper(): float(value) for key, value in METRIC_RE.findall(line)
            }
            if (
                set(values) == set(plan_builder.METRICS)
                and "Env name: val_unseen" in line
            ):
                matches.append((path, values, line))
    if len(matches) != 1:
        raise UserError("expected exactly one REVERIE val_unseen metric line")
    if any(not 0.0 <= value <= 100.0 for value in matches[0][1].values()):
        raise UserError("REVERIE metric is outside [0, 100]")
    return matches[0]


def _count(adapter, key, episodes):
    value = adapter.get(key)
    if type(value) is not int or not 0 <= value <= episodes:
        raise UserError("adapter.{} must be an integer in [0, {}]".format(key, episodes))
    return value


def _validate_diagnostics(metadata, diagnostics, metrics):
    method, episodes = metadata["method"], metadata["episode_count"]
    if (
        diagnostics.get("method") != method
        or diagnostics.get("episode_count") != episodes
        or diagnostics.get("action_selection") != "target_native_argmax"
    ):
        raise UserError("TTA diagnostics method/episode/action protocol mismatch")
    adapter = diagnostics.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("episodes") != episodes:
        raise UserError("adapter episode accounting mismatch")
    for key in ("updates", "relative_param_drift"):
        value = adapter.get(key)
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0.0
        ):
            raise UserError("adapter.{} is missing or invalid".format(key))
    if method in EXPECTED_FEEDBACK_ENDPOINT:
        if diagnostics.get("supervision") != "binary_navigation_success_feedback":
            raise UserError("binary-feedback supervision mismatch")
        endpoint = diagnostics.get("binary_feedback_endpoint")
        if endpoint != EXPECTED_FEEDBACK_ENDPOINT[method] or "distance" in endpoint:
            raise UserError("invalid REVERIE binary-feedback endpoint")
        if method == "feedtta":
            feedback = _count(adapter, "feedback_episodes", episodes)
            successes = _count(adapter, "successful_feedback_episodes", episodes)
            failures = _count(adapter, "failed_feedback_episodes", episodes)
            if (
                feedback != episodes or successes + failures != episodes
                or successes != int(round(float(metrics["SR"]) * episodes / 100.0))
                or adapter.get("action_selection_protocol") != "target_native_argmax"
            ):
                raise UserError("FeedTTA feedback accounting mismatch")
        else:
            queries = _count(adapter, "queries", episodes)
            self_labels = _count(adapter, "self_label_episodes", episodes)
            observed = _count(adapter, "feedback_observed_episodes", episodes)
            gate = _count(adapter, "query_gate_evaluations", episodes)
            self_evals = _count(adapter, "self_prediction_evaluations", episodes)
            if (
                queries + self_labels != episodes or observed != queries
                or gate != episodes or self_evals != episodes
            ):
                raise UserError("ATENA query/self-label accounting mismatch")
            if (
                float(metadata["parameters"].get("query_threshold", math.nan)) == 0.0
                and queries != episodes
            ):
                raise UserError("zero-threshold ATENA must query every episode")
    else:
        if (
            diagnostics.get("supervision") != "unsupervised"
            or diagnostics.get("binary_feedback_endpoint") is not None
        ):
            raise UserError("unsupervised method unexpectedly consumed feedback")
    return adapter


def _validate_formal_manifest(path, metadata, required_artifacts,
                              split="val_unseen"):
    path = Path(path).resolve()
    try:
        path.relative_to(FORMAL_ROOT.resolve())
    except ValueError:
        raise UserError("formal manifest escapes vln/results/runs")
    expected_run_id = "{}-{}-{}-native".format(
        metadata["run_tag"], metadata["setting"], split
    )
    if path.parent.name != expected_run_id or path.name != "manifest.json" or not path.is_file():
        raise UserError("formal manifest path is missing/noncanonical")
    manifest = _read_json(path)
    expected = {
        "run_id": expected_run_id,
        "task": "vln",
        "benchmark": metadata["expected_benchmark"],
        "model": metadata["model"],
        "method": metadata["method"],
        "run_tag": metadata["run_tag"],
        "source_setting": "{}:{}:native:{}".format(
            metadata["setting"], split, metadata["method"]
        ),
        "seed": 0,
        "git_commit": metadata["git_commit"],
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise UserError("formal manifest {} mismatch".format(key))
    overrides = manifest.get("config_overrides")
    if not isinstance(overrides, list):
        raise UserError("formal manifest lacks config_overrides")
    def flag_value(flag):
        positions = [index for index, value in enumerate(overrides) if value == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(overrides):
            raise UserError("formal manifest has invalid {}".format(flag))
        return str(overrides[positions[0] + 1])
    if "--test" not in overrides:
        raise UserError("formal command is not an evaluation command")
    if flag_value("--eval_splits") != split or flag_value("--seed") != "0":
        raise UserError("formal command split/seed mismatch")
    if Path(flag_value("--output_dir")).resolve() != Path(
        metadata["result_root"]
    ).resolve():
        raise UserError("formal command output directory mismatch")
    expected_order_dir = Path(metadata["episode_order_manifest"]).resolve().parent
    if Path(flag_value("--episode_order_manifest")).resolve() != expected_order_dir:
        raise UserError("formal command episode-order manifest mismatch")
    try:
        registry_builder._assert_effective_parameters(
            manifest, metadata["setting"], metadata["method"],
            metadata["parameters"],
            "{} {} {}".format(metadata["setting"], metadata["method"], split),
        )
    except registry_builder.RegistryError as error:
        raise UserError(str(error))
    if manifest.get("checkpoint", {}).get("sha256") != metadata[
        "expected_checkpoint_sha256"
    ]:
        raise UserError("formal checkpoint mismatch")
    dataset = manifest.get("dataset", {})
    if (
        dataset.get("stream_content_sha256") != metadata["expected_dataset_sha256"]
        or dataset.get("stream_order_sha256") != metadata["expected_episode_order_sha256"]
    ):
        raise UserError("formal dataset/order mismatch")
    identity = manifest.get("immutable_identity_sha256")
    if not registry_builder._is_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise UserError("formal immutable identity mismatch")
    authenticated = set()
    result_root = Path(metadata["result_root"]).resolve()
    for artifact in manifest.get("result_artifacts", []):
        if not isinstance(artifact, dict):
            raise UserError("malformed formal artifact")
        artifact_path = Path(str(artifact.get("path", ""))).resolve()
        try:
            artifact_path.relative_to(result_root)
        except ValueError:
            raise UserError("formal artifact escapes result root")
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != artifact.get("size")
            or _sha256(artifact_path) != artifact.get("sha256")
        ):
            raise UserError("formal artifact missing or changed")
        authenticated.add(artifact_path)
    if any(Path(item).resolve() not in authenticated for item in required_artifacts):
        raise UserError("required metric/diagnostic artifact is unauthenticated")
    return manifest


def _validate_attempt_binding(attempt_dir, metadata, expected_job=None,
                              expected_batch_id=None, expected_spec_path=None,
                              split="val_unseen"):
    attempt_dir = Path(attempt_dir).resolve()
    if metadata.get("process_group_token") is not None or any(
        value is not None for value in (expected_job, expected_batch_id, expected_spec_path)
    ):
        if metadata.get("process_group_token") != _expected_process_group_token(
            metadata, attempt_dir
        ):
            raise UserError("attempt process-group token is invalid")
    bound_spec = None
    if expected_spec_path is not None:
        spec_path = Path(expected_spec_path).resolve()
        if (
            Path(str(metadata.get("spec_path", ""))).resolve() != spec_path
            or not spec_path.is_file()
            or _sha256(spec_path) != metadata.get("spec_sha256")
        ):
            raise UserError("attempt spec binding mismatch")
        bound_spec = _read_json(spec_path)
    config_path = attempt_dir / "parameters.json"
    if Path(str(metadata.get("parameters_path", ""))).resolve() != config_path:
        raise UserError("job metadata parameters path is noncanonical")
    if not config_path.is_file() or _sha256(config_path) != metadata.get(
        "parameters_sha256"
    ):
        raise UserError("parameters.json digest mismatch")
    config = _read_json(config_path)
    expected_stage = (
        "frozen_test_submission" if split == "test" else "frozen_val_unseen"
    )
    expected_config = {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": metadata.get("batch_id"),
        "stage": expected_stage,
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
        raise UserError("frozen canonical-order config must omit order_seed")
    provenance = config.get("frozen_evaluation_provenance")
    if not isinstance(provenance, dict) or _canonical(
        provenance.get("selected_anchor")
    ) != _canonical(metadata.get("selected_anchor")):
        raise UserError("parameters.json winner provenance mismatch")
    if split == "val_unseen":
        if (
            provenance.get("selection_split") != "val_seen"
            or provenance.get("evaluation_split") != "val_unseen"
            or provenance.get("selection_on_val_unseen") is not False
            or provenance.get("canonical_order_seed") != 0
        ):
            raise UserError("val_unseen frozen provenance mismatch")
        source_path = _resolve(
            bound_spec["source_control"]["manifest"]
            if bound_spec is not None
            else metadata.get("source_ledger_path", "")
        ).resolve()
        if Path(str(metadata.get("source_ledger_path", ""))).resolve() != source_path:
            raise UserError("attempt val_unseen Source-ledger path mismatch")
        source_path, source = plan_builder.validate_val_unseen_source(source_path)
        source_record = source["records"][metadata["setting"]]
        expected_source_sha = _sha256(source_path)
        expected_dataset_sha = source_record["dataset_sha256"]
        expected_checkpoint_sha = source_record["checkpoint_sha256"]
        expected_order_sha = plan_builder.VAL_UNSEEN_ORDER
        order_key = "goat" if metadata["setting"] == "goat-reverie" else "duet_hamt"
        expected_order_path = _resolve(
            bound_spec["protocol"]["order_manifests"][order_key]["path"]
            if bound_spec is not None else metadata.get("episode_order_manifest", "")
        ).resolve()
        source_binding = provenance.get("source_ledger", {})
        if (
            _resolve(source_binding.get("path", "")).resolve() != source_path
            or source_binding.get("sha256") != expected_source_sha
            or metadata.get("source_ledger_sha256") != expected_source_sha
        ):
            raise UserError("val_unseen Source-ledger provenance mismatch")
    else:
        if (
            provenance.get("selection_split") != "val_seen"
            or provenance.get("evaluation_split") != "test"
            or provenance.get("selection_on_test") is not False
            or provenance.get("hidden_ground_truth") is not True
            or provenance.get("submission_generation_only") is not True
            or provenance.get("canonical_order_seed") != 0
        ):
            raise UserError("hidden-test frozen provenance mismatch")
        source_path = _resolve(
            bound_spec["test_submission_policy"]["source"]["ledger"]
            if bound_spec is not None
            else metadata.get("test_source_ledger_path", "")
        ).resolve()
        if Path(str(metadata.get("test_source_ledger_path", ""))).resolve() != source_path:
            raise UserError("attempt test Source-ledger path mismatch")
        source_path, source = plan_builder.validate_test_source(source_path)
        source_record = source["records"][metadata["setting"]]
        source_manifest = _read_json(_resolve(source_record["formal_manifest_path"]))
        expected_source_sha = _sha256(source_path)
        expected_dataset_sha = source_record["dataset_sha256"]
        expected_checkpoint_sha = source_manifest["checkpoint"]["sha256"]
        expected_order_sha = plan_builder.TEST_ORDER
        order_key = "goat" if metadata["setting"] == "goat-reverie" else "duet_hamt"
        expected_order_path = _resolve(
            bound_spec["test_submission_policy"]["order_manifests"][order_key]["path"]
            if bound_spec is not None else metadata.get("episode_order_manifest", "")
        ).resolve()
        source_binding = provenance.get("source_submission_ledger", {})
        if (
            _resolve(source_binding.get("path", "")).resolve() != source_path
            or source_binding.get("sha256") != expected_source_sha
            or metadata.get("test_source_ledger_sha256") != expected_source_sha
        ):
            raise UserError("test Source-ledger provenance mismatch")
    for key, expected_value in (
        ("expected_dataset_sha256", expected_dataset_sha),
        ("expected_checkpoint_sha256", expected_checkpoint_sha),
        ("expected_episode_order_sha256", expected_order_sha),
    ):
        if metadata.get(key) != expected_value:
            raise UserError("attempt {} differs from Source/order evidence".format(key))
    if Path(str(metadata.get("episode_order_manifest", ""))).resolve() != expected_order_path:
        raise UserError("attempt episode-order manifest path mismatch")
    if expected_batch_id is not None and metadata.get(
        "batch_id"
    ) != expected_batch_id:
        raise UserError("attempt batch ID differs from scheduler batch")
    if expected_spec_path is not None:
        if metadata.get("git_commit") != _git_commit():
            raise UserError("attempt Git commit differs from current HEAD")
        if provenance.get("registry") != bound_spec.get("registry_dependency"):
            raise UserError("attempt registry provenance differs from spec")
    if expected_job is not None:
        for key in (
            "base_run_tag", "setting", "model", "method", "model_index",
            "method_index", "ordinal", "gpu", "parameters", "selected_anchor",
        ):
            if _canonical(metadata.get(key)) != _canonical(expected_job.get(key)):
                raise UserError("attempt differs from planned job: {}".format(key))
        expected_tag = _attempt_tag(
            expected_job["base_run_tag"], metadata.get("attempt")
        )
        if metadata.get("run_tag") != expected_tag:
            raise UserError("attempt run tag differs from retry identity")
    expected_command = [
        str(RUNNER), metadata["setting"], split,
        str(metadata.get("gpu")),
        "--run-tag", metadata["run_tag"], "--tta-config", str(config_path),
    ]
    if metadata.get("command") != expected_command:
        raise UserError("attempt command differs from immutable job metadata")
    expected_formal = (
        FORMAL_ROOT
        / "{}-{}-{}-native".format(
            metadata["run_tag"], metadata["setting"], split
        )
        / "manifest.json"
    ).resolve()
    if Path(metadata["formal_manifest"]).resolve() != expected_formal:
        raise UserError("attempt formal-manifest path is noncanonical")
    return config


def validate_attempt(attempt_dir, expected_job=None, expected_batch_id=None,
                     expected_spec_path=None):
    attempt_dir = Path(attempt_dir)
    metadata = _read_json(attempt_dir / "job.json")
    _validate_attempt_binding(
        attempt_dir, metadata, expected_job, expected_batch_id,
        expected_spec_path, split="val_unseen",
    )
    try:
        exit_code = int((attempt_dir / "exitcode").read_text().strip())
    except (OSError, ValueError) as error:
        raise UserError("invalid worker exit status: {}".format(error))
    if exit_code != 0:
        raise UserError("worker exited with {}".format(exit_code))
    metric_path, metrics, metric_line = _parse_metrics(metadata["result_root"])
    diagnostics_path = Path(metadata["result_root"]) / "tta_diagnostics.json"
    if not diagnostics_path.is_file():
        raise UserError("missing TTA diagnostics")
    diagnostics = _read_json(diagnostics_path)
    adapter = _validate_diagnostics(metadata, diagnostics, metrics)
    manifest = _validate_formal_manifest(
        metadata["formal_manifest"], metadata, (metric_path, diagnostics_path)
    )
    result = {
        **metadata,
        "metrics": metrics,
        "metric_artifact": str(metric_path),
        "metric_artifact_sha256": _sha256(metric_path),
        "metric_line": metric_line,
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "feedback_endpoint": diagnostics.get("binary_feedback_endpoint"),
        "adapter_diagnostics": adapter,
        "formal_manifest_sha256": _sha256(metadata["formal_manifest"]),
        "formal_immutable_identity_sha256": manifest["immutable_identity_sha256"],
    }
    _atomic_json(attempt_dir / "metrics.json", result)
    return result


def validate_test_attempt(attempt_dir, expected_job=None,
                          expected_batch_id=None, expected_spec_path=None):
    attempt_dir = Path(attempt_dir)
    metadata = _read_json(attempt_dir / "job.json")
    _validate_attempt_binding(
        attempt_dir, metadata, expected_job, expected_batch_id,
        expected_spec_path, split="test",
    )
    try:
        exit_code = int((attempt_dir / "exitcode").read_text().strip())
    except (OSError, ValueError) as error:
        raise UserError("invalid test worker exit status: {}".format(error))
    if exit_code != 0:
        raise UserError("test worker exited with {}".format(exit_code))
    if metadata.get("method") not in ("tent", "fstta", "eam"):
        raise UserError("hidden-test feedback method is N/A and must fail closed")
    if (
        metadata.get("hidden_ground_truth") is not True
        or metadata.get("submission_generation_only") is not True
    ):
        raise UserError("test attempt is not submission-only")
    result_root = Path(metadata["result_root"])
    submissions = sorted(result_root.rglob("submit_test*.json"))
    if len(submissions) != 1:
        raise UserError("expected exactly one REVERIE test submission artifact")
    submission_path = submissions[0]
    try:
        submission = json.loads(submission_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("invalid test submission JSON: {}".format(error))
    if not isinstance(submission, list) or len(submission) != metadata[
        "episode_count"
    ]:
        raise UserError("test submission episode count mismatch")
    order = _read_json(metadata["episode_order_manifest"])
    expected_ids = [str(item["episode_id"]) for item in order["episodes"]]
    actual_ids = []
    for index, item in enumerate(submission):
        if (
            not isinstance(item, dict)
            or not {"instr_id", "trajectory", "predObjId"}.issubset(item)
            or not isinstance(item["trajectory"], list)
        ):
            raise UserError("test submission row {} is malformed".format(index))
        actual_ids.append(str(item["instr_id"]))
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise UserError("test submission IDs/order differ from canonical seed-0 stream")
    diagnostics_path = result_root / "tta_diagnostics.json"
    if not diagnostics_path.is_file():
        raise UserError("missing test TTA diagnostics")
    diagnostics = _read_json(diagnostics_path)
    if (
        diagnostics.get("method") != metadata["method"]
        or diagnostics.get("episode_count") != metadata["episode_count"]
        or diagnostics.get("action_selection") != "target_native_argmax"
        or diagnostics.get("supervision") != "unsupervised"
        or diagnostics.get("binary_feedback_endpoint") is not None
        or diagnostics.get("adapter", {}).get("episodes")
        != metadata["episode_count"]
    ):
        raise UserError("hidden-test diagnostics violate unsupervised contract")
    for valid in result_root.rglob("valid.txt"):
        for line in valid.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Env name: test" in line and METRIC_RE.search(line):
                raise UserError("local hidden-test metrics are forbidden")
    manifest = _validate_formal_manifest(
        metadata["formal_manifest"], metadata,
        (submission_path, diagnostics_path), split="test",
    )
    result = {
        **metadata,
        "submission_path": str(submission_path),
        "submission_sha256": _sha256(submission_path),
        "submission_size": submission_path.stat().st_size,
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "formal_manifest_sha256": _sha256(metadata["formal_manifest"]),
        "formal_immutable_identity_sha256": manifest["immutable_identity_sha256"],
        "metrics": None,
    }
    _atomic_json(attempt_dir / "submission.json", result)
    return result


def _write_worker(attempt_dir, metadata):
    command = metadata["command"]
    exit_path = Path(attempt_dir) / "exitcode"
    authorization = Path(attempt_dir) / "launch_authorized"
    pid_path = Path(attempt_dir) / "pid"
    token = metadata["process_group_token"]
    if not _valid_sha256(token):
        raise UserError("worker requires a valid per-attempt process token")
    quoted_authorization = shlex.quote(str(authorization))
    quoted_exit_tmp = shlex.quote(str(exit_path) + ".tmp")
    quoted_exit = shlex.quote(str(exit_path))
    quoted_pid_tmp = shlex.quote(str(pid_path) + ".worker.tmp")
    quoted_pid = shlex.quote(str(pid_path))
    script = "\n".join((
        "#!/usr/bin/env bash", "set +e",
        "export {}={}".format(PROCESS_GROUP_ENV, shlex.quote(token)),
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
        shlex.join(command), "status=$?",
        "printf '%s\\n' \"$status\" > {}".format(quoted_exit_tmp),
        "mv {} {}".format(quoted_exit_tmp, quoted_exit),
        "exit \"$status\"", "",
    ))
    path = Path(attempt_dir) / "worker.sh"
    _atomic_text(path, script)
    path.chmod(0o755)
    return path


def launch_attempt(attempt_dir, metadata):
    worker = _write_worker(attempt_dir, metadata)
    launcher_log = (Path(attempt_dir) / "launcher.log").open("ab", buffering=0)
    process = None
    try:
        process = subprocess.Popen(
            ["bash", str(worker)], cwd=str(REPO_ROOT), stdout=launcher_log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, **{
                PROCESS_GROUP_ENV: metadata["process_group_token"],
            }),
        )
        _atomic_text(Path(attempt_dir) / "pid", "{}\n".format(process.pid))
        identity = process_identity(process.pid)
        if identity is None:
            raise UserError("cannot bind worker process identity")
        if not _identity_is_group_member(identity, process.pid):
            raise UserError("worker did not become its own process-group leader")
        _atomic_json(Path(attempt_dir) / "process_identity.json", identity)
        return process, launcher_log
    except BaseException as error:
        try:
            cleaned = _terminate_attempt_process_group(attempt_dir, process)
        except Exception:
            cleaned = False
        launcher_log.close()
        if not cleaned:
            raise LaunchCleanupError(
                "failed launch still has authenticated descendants; "
                "reservation retained", process=process,
            ) from error
        raise


def _authorize_launch(attempt_dir):
    _atomic_text(Path(attempt_dir) / "launch_authorized", "authorized\n")


def _cleanup_process_groups(attempt_dir, process=None):
    identities = _live_worker_identities(attempt_dir)
    pgids = set()
    pgid = _attempt_pgid(attempt_dir)
    if identities and pgid is not None:
        pgids.add(pgid)
    if process is not None and process.poll() is None:
        # The direct child was launched with start_new_session=True.  Include
        # it even in the narrow window before procfs/identity files settle.
        pgids.add(process.pid)
    return identities, pgids


def _terminate_attempt_process_group(attempt_dir, process=None,
                                     timeout_seconds=10.0):
    identities, pgids = _cleanup_process_groups(attempt_dir, process)
    if not identities and not pgids:
        return True
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            return False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        identities, pgids = _cleanup_process_groups(attempt_dir, process)
        if not identities and not pgids:
            return True
        time.sleep(0.1)
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            return False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        identities, pgids = _cleanup_process_groups(attempt_dir, process)
        if not identities and not pgids:
            return True
        time.sleep(0.1)
    return False


def _retain_reservation_owners(ledger, token, attempt_dir, process=None):
    try:
        identities = _live_worker_identities(attempt_dir)
    except Exception:
        identities = []
    if process is not None and process.poll() is None:
        direct = process_identity(process.pid)
        if (
            direct is not None
            and direct.get("pid") == process.pid
            and process_identity_alive(direct)
        ):
            identities.append(direct)
    owner_keys = ("pid", "start_token", "cmdline_sha256")
    unique = []
    for identity in identities:
        if not any(
            all(existing.get(key) == identity.get(key) for key in owner_keys)
            for existing in unique
        ):
            unique.append(identity)
    if not unique:
        raise LaunchCleanupError(
            "failed launch is still live but has no persistable process identity"
        )
    record = ledger.document["reservations"].get(token)
    if not isinstance(record, dict) or not isinstance(record.get("owners"), list):
        raise LaunchCleanupError("failed launch reservation disappeared")
    owners = record["owners"]
    known_owners = {
        tuple(owner.get(key) for key in owner_keys) for owner in owners
    }
    for identity in unique:
        key = tuple(identity.get(name) for name in owner_keys)
        if key not in known_owners:
            if (
                type(identity.get("pid")) is not int
                or identity["pid"] <= 0
                or not isinstance(identity.get("start_token"), str)
                or not identity["start_token"]
                or not _valid_sha256(identity.get("cmdline_sha256"))
            ):
                raise LaunchCleanupError(
                    "failed launch has an invalid live process identity"
                )
            owners.append(identity)
            known_owners.add(key)
    # ``ReservationLedger.add_owner`` mutates memory before writing.  A prior
    # write failure can therefore make the owner look present only in memory;
    # force a fresh atomic write before allowing this scheduler to exit.
    try:
        ledger._write()
    except Exception as error:
        raise LaunchCleanupError(
            "failed launch reservation owners could not be persisted"
        ) from error


def _rollback_failed_launch(ledger, token, process, attempt_dir):
    if not _terminate_attempt_process_group(attempt_dir, process):
        _retain_reservation_owners(
            ledger, token, attempt_dir, process
        )
        raise UserError(
            "failed launch still has authenticated descendants; "
            "reservation retained"
        )
    ledger.release(token)


def _gpu_memory_mib(gpu):
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits", "-i", str(gpu),
        ], text=True, stderr=subprocess.STDOUT).strip().splitlines()[0]
        total, used = (int(value.strip()) for value in output.split(","))
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as error:
        raise UserError("cannot query GPU memory: {}".format(error))
    return total, used, total - used


def _prepare_batch(spec_path, spec, batch_id, batch_root, gpu, resume,
                   mode="val_unseen"):
    if mode not in ("val_unseen", "test_submission"):
        raise UserError("unknown batch mode")
    payload = {
        "schema": BATCH_SCHEMA,
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "git_commit": _git_commit(),
        "registry": spec["registry_dependency"],
        "gpu": int(gpu),
        "mode": mode,
        "source_execution_jobs": 0,
        "tta_jobs": 15 if mode == "val_unseen" else 9,
    }
    if mode == "test_submission":
        payload["test_source_submission_ledger"] = spec[
            "test_submission_policy"
        ]["source"]
    else:
        payload["source_ledger"] = spec["source_control"]
    path = Path(batch_root) / "BATCH.json"
    if path.is_file():
        if not resume:
            raise UserError("batch already exists; use --resume")
        if _canonical(_read_json(path)) != _canonical(payload):
            raise UserError("batch ID is bound to a different commit/spec")
    elif resume:
        raise UserError("cannot resume a batch without BATCH.json")
    elif Path(batch_root).exists() and any(Path(batch_root).iterdir()):
        raise UserError("nonempty batch directory has no BATCH.json")
    _atomic_json(path, payload)


def _reservation_metadata(batch_id, run_tag):
    return {
        "role": "reverie_val_unseen",
        "batch_id": batch_id,
        "run_tag": run_tag,
    }


def _release(batch_id, job, attempt, attempt_dir):
    if _pid_alive(attempt_dir):
        raise UserError(
            "refusing to release reservation while attempt descendants are alive"
        )
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
    estimate = spec["execution"]["estimated_gpu_memory_mib_by_method"][job["method"]]
    expected_metadata = _reservation_metadata(batch_id, metadata["run_tag"])
    with shared_gpu_launch_guard(job["gpu"]) as ledger:
        record = ledger.document["reservations"].get(token)
        if record is None:
            _, used, _ = _gpu_memory_mib(job["gpu"])
            owner = process_identity()
            if owner is None:
                raise UserError("cannot bind REVERIE scheduler process identity")
            ledger.reserve(
                token, gpu_memory_mib=estimate, cgroup_memory_gib=0.0,
                observed_gpu_memory_mib=used, observed_cgroup_memory_gib=0.0,
                owner=owner, metadata=expected_metadata,
            )
            record = ledger.document["reservations"][token]
        elif (
            record.get("gpu_memory_mib") != estimate
            or not math.isclose(
                float(record.get("cgroup_memory_gib", math.nan)), 0.0,
                rel_tol=0.0, abs_tol=0.0,
            )
            or record.get("metadata") != expected_metadata
        ):
            raise UserError("live REVERIE worker reservation binding mismatch")
        owner_keys = ("pid", "start_token", "cmdline_sha256")
        for identity in identities:
            if not any(
                all(owner.get(key) == identity.get(key) for key in owner_keys)
                for owner in record.get("owners", [])
            ):
                ledger.add_owner(token, identity)
    _authorize_launch(attempt_dir)
    return token


def run_model_phase(spec, spec_path, batch_id, batch_root, jobs,
                    retry_failed=False, materializer=None, validator=None):
    materializer = materializer or materialize_attempt
    validator = validator or validate_attempt
    model = jobs[0]["model"]
    if any(item["model"] != model for item in jobs):
        raise UserError("one phase may contain only one model")
    _assert_clean_execution_tree(spec_path)
    cap = spec["execution"]["max_workers_by_model"][model]
    estimates = spec["execution"]["estimated_gpu_memory_mib_by_method"]
    aggregate_cap = spec["execution"]["max_aggregate_gpu_memory_mib"]
    minimum_free = spec["execution"]["minimum_free_gpu_memory_mib"]
    emergency = spec["execution"]["emergency_abort_used_gpu_memory_mib"]
    stagger = spec["execution"]["launch_stagger_seconds"]
    poll = spec["execution"]["poll_seconds"]
    active = {}
    validated_complete = set()
    last_launch = 0.0
    try:
        while True:
            batch = _read_json(Path(batch_root) / "BATCH.json")
            if batch.get("git_commit") != _git_commit() or batch.get(
                "spec_sha256"
            ) != _sha256(spec_path):
                raise UserError("HEAD/spec drifted from immutable batch binding")
            completed, running, launchable = 0, [], []
            for job in jobs:
                latest = _latest_attempt(batch_root, job)
                if latest is None:
                    launchable.append((job, 0))
                    continue
                attempt, path = latest
                _recover_worker_identity(path)
                state = _state_for_attempt(path)
                if state in ("completed", "finished", "failed", "invalid", "orphaned"):
                    active_handle = active.pop(job["base_run_tag"], None)
                    if active_handle is not None:
                        active_handle[0].poll()
                        if active_handle[1] is not None:
                            active_handle[1].close()
                if state in ("completed", "finished"):
                    if job["base_run_tag"] not in validated_complete:
                        try:
                            result = validator(
                                path, expected_job=job,
                                expected_batch_id=batch_id,
                                expected_spec_path=spec_path,
                            )
                        except Exception as error:
                            _release(batch_id, job, attempt, path)
                            _atomic_json(path / "validation_error.json", {"error": str(error)})
                            if not retry_failed:
                                raise UserError("{} validation failed: {}".format(
                                    job["base_run_tag"], error
                                ))
                            launchable.append((job, attempt + 1))
                            continue
                        validated_complete.add(job["base_run_tag"])
                        if result.get("metrics") is None:
                            print("completed {} submission={}".format(
                                result["run_tag"], result["submission_path"]
                            ))
                        else:
                            print("completed {} RGSPL={:.2f}".format(
                                result["run_tag"], result["metrics"]["RGSPL"]
                            ))
                    _release(batch_id, job, attempt, path)
                    completed += 1
                elif state == "running":
                    metadata = _read_json(path / "job.json")
                    _validate_attempt_binding(
                        path, metadata, expected_job=job,
                        expected_batch_id=batch_id,
                        expected_spec_path=spec_path,
                        split=(
                            "test" if metadata.get("submission_generation_only")
                            is True else "val_unseen"
                        ),
                    )
                    _claim_running_reservation(
                        spec, batch_id, job, path, metadata
                    )
                    running.append(job)
                elif state in ("failed", "invalid", "orphaned"):
                    _release(batch_id, job, attempt, path)
                    if not retry_failed:
                        raise UserError("{} is {}; use --resume --retry-failed".format(
                            job["base_run_tag"], state
                        ))
                    launchable.append((job, attempt + 1))
                else:
                    launchable.append((job, attempt))
            if completed == len(jobs):
                return
            busy_methods = Counter(item["method"] for item in running)
            launched_any = False
            for job, attempt in launchable:
                if len(running) >= cap or busy_methods[job["method"]] >= 1:
                    continue
                if time.monotonic() - last_launch < stagger:
                    break
                estimate = estimates[job["method"]]
                with shared_gpu_launch_guard(job["gpu"]) as ledger:
                    total, used, free = _gpu_memory_mib(job["gpu"])
                    snapshot = ledger.snapshot(used, 0.0)
                    effective_used = snapshot["effective_gpu_memory_mib"]
                    if used >= emergency:
                        raise UserError("GPU memory reached emergency threshold")
                    if (
                        free < minimum_free
                        or total - (effective_used + estimate) < minimum_free
                        or effective_used + estimate > aggregate_cap
                    ):
                        continue
                    _assert_clean_execution_tree(spec_path)
                    attempt_dir, metadata = materializer(
                        spec, spec_path, batch_id, batch_root, job, attempt
                    )
                    token = _reservation_token(batch_id, metadata["run_tag"])
                    owner = process_identity()
                    if owner is None:
                        raise UserError("cannot bind REVERIE scheduler process identity")
                    ledger.reserve(
                        token, gpu_memory_mib=estimate, cgroup_memory_gib=0.0,
                        observed_gpu_memory_mib=used, observed_cgroup_memory_gib=0.0,
                        owner=owner,
                        metadata=_reservation_metadata(batch_id, metadata["run_tag"]),
                    )
                    process = None
                    handle = None
                    try:
                        process, handle = launch_attempt(attempt_dir, metadata)
                        worker_identity = _read_json(
                            attempt_dir / "process_identity.json"
                        )
                        ledger.add_owner(token, worker_identity)
                        _authorize_launch(attempt_dir)
                    except LaunchCleanupError as error:
                        _retain_reservation_owners(
                            ledger, token, attempt_dir, error.process
                        )
                        raise
                    except BaseException:
                        if handle is not None:
                            handle.close()
                        _rollback_failed_launch(
                            ledger, token, process, attempt_dir
                        )
                        raise
                    active[job["base_run_tag"]] = (
                        process, handle, attempt_dir, metadata, job
                    )
                    running.append(job)
                    busy_methods[job["method"]] += 1
                    last_launch = time.monotonic()
                    launched_any = True
                    print("launched {} pid={}".format(
                        metadata["run_tag"], process.pid
                    ))
                    # Keep the shared launch lock until CUDA allocation can
                    # become visible to the peer campaign.
                    time.sleep(stagger)
                    descendants = _live_worker_identities(attempt_dir)
                    for descendant in descendants:
                        if descendant.get("pid") != worker_identity.get("pid"):
                            ledger.add_owner(token, descendant)
            for tag, (process, handle, path, metadata, job) in list(active.items()):
                if process.poll() is not None and handle is not None:
                    handle.close()
                    active[tag] = (process, None, path, metadata, job)
            if not launched_any:
                time.sleep(min(float(poll), 2.0))
    except BaseException:
        for process, handle, path, metadata, job in active.values():
            try:
                cleaned = _terminate_attempt_process_group(path, process)
            except Exception:
                cleaned = False
            try:
                process.wait(timeout=10)
            except (subprocess.TimeoutExpired, AttributeError):
                pass
            if handle is not None:
                handle.close()
            if cleaned:
                try:
                    _release(batch_id, job, metadata["attempt"], path)
                except Exception:
                    pass
            else:
                try:
                    _claim_running_reservation(
                        spec, batch_id, job, path, metadata
                    )
                except Exception:
                    pass
        raise


def collect_states(batch_root, jobs, revalidate=False, validator=None,
                   cache_name="metrics.json"):
    validator = validator or validate_attempt
    states = Counter()
    rows = []
    for job in jobs:
        latest = _latest_attempt(batch_root, job)
        if latest is None:
            states["pending"] += 1
            continue
        _, path = latest
        state = _state_for_attempt(path)
        if revalidate and state in ("completed", "finished"):
            try:
                rows.append(validator(path, expected_job=job))
                state = "completed"
            except Exception:
                state = "invalid"
        elif state == "completed":
            rows.append(_read_json(path / cache_name))
        states[state] += 1
    return {key: states.get(key, 0) for key in (
        "completed", "finished", "failed", "invalid", "running",
        "orphaned", "pending",
    )}, rows


def write_summary(batch_root, spec, jobs):
    states, rows = collect_states(batch_root, jobs, revalidate=True)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "source_execution_jobs": 0,
        "total_jobs": 15,
        "complete": states["completed"] == 15,
        "states": states,
        "results": [{
            key: row[key] for key in (
                "setting", "method", "run_tag", "parameters", "metrics",
                "feedback_endpoint", "formal_manifest",
                "formal_manifest_sha256",
            )
        } for row in sorted(rows, key=lambda item: item["ordinal"])],
    }
    _atomic_json(Path(batch_root) / "SUMMARY.json", payload)
    csv_path = Path(batch_root) / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "setting", "method", "SR", "SPL", "RGS", "RGSPL", "run_tag"
        ))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["ordinal"]):
            writer.writerow({
                "setting": row["setting"], "method": row["method"],
                **row["metrics"], "run_tag": row["run_tag"],
            })
    return payload


def write_test_summary(batch_root, spec, jobs):
    states, rows = collect_states(
        batch_root, jobs, revalidate=True, validator=validate_test_attempt,
        cache_name="submission.json",
    )
    payload = {
        "schema": "navtta.vln_reverie_test_submission_summary.v1",
        "experiment_id": spec["experiment_id"],
        "split": "test",
        "hidden_ground_truth": True,
        "submission_generation_only": True,
        "local_metrics": None,
        "source_execution_jobs": 0,
        "reused_source_submissions": 3,
        "tta_submission_jobs": 9,
        "unavailable_methods": {
            "feedtta": "N/A: no legal online binary feedback",
            "atena": "N/A: no legal online binary feedback",
        },
        "complete": states["completed"] == 9,
        "states": states,
        "submissions": [{
            key: row[key] for key in (
                "setting", "method", "run_tag", "parameters",
                "submission_path", "submission_sha256", "submission_size",
                "formal_manifest", "formal_manifest_sha256",
            )
        } for row in sorted(rows, key=lambda item: item["ordinal"])],
    }
    _atomic_json(Path(batch_root) / "SUBMISSIONS.json", payload)
    return payload


def print_plan(spec, jobs, batch_id, gpu, print_commands=False):
    print(
        "batch_id={} gpu={} tta_jobs=15 source_jobs=0 split=val_unseen "
        "episodes=3521 canonical_order_seed=0".format(batch_id, gpu)
    )
    for phase in model_phases(jobs):
        print("model {}: methods={} max_workers={} strict_barrier_after=true".format(
            phase[0]["model"], ",".join(item["method"] for item in phase),
            spec["execution"]["max_workers_by_model"][phase[0]["model"]],
        ))
        if print_commands:
            for job in phase:
                print(shlex.join([
                    str(RUNNER), job["setting"], "val_unseen", str(gpu),
                    "--run-tag", job["base_run_tag"], "--tta-config",
                    "<ATTEMPT_DIR>/parameters.json",
                ]))
    policy = spec["test_submission_policy"]
    print(
        "test=hidden_submission_only eligible_tta={} feedtta=N/A atena=N/A "
        "without_legal_online_feedback".format(
            ",".join(policy["eligible_tta_methods"])
        )
    )


def print_test_plan(spec, jobs, batch_id, gpu, print_commands=False):
    print(
        "batch_id={} gpu={} tta_submission_jobs=9 source_jobs=0 "
        "reused_source_submissions=3 split=test episodes=6292 "
        "canonical_order_seed=0 hidden_ground_truth=true local_metrics=forbidden"
        .format(batch_id, gpu)
    )
    for phase in test_model_phases(jobs):
        print("model {}: methods={} max_workers={} strict_barrier_after=true".format(
            phase[0]["model"], ",".join(item["method"] for item in phase),
            spec["execution"]["max_workers_by_model"][phase[0]["model"]],
        ))
        if print_commands:
            for job in phase:
                print(shlex.join([
                    str(RUNNER), job["setting"], "test", str(gpu),
                    "--run-tag", job["base_run_tag"], "--tta-config",
                    "<ATTEMPT_DIR>/parameters.json",
                ]))
    print("feedtta=N/A atena=N/A reason=no_legal_online_binary_feedback")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    default_batch_id = "vln-reverie-val-unseen-frozen-eval-v1-seed0"
    parser.add_argument("--batch-id", default=default_batch_id)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--test-submissions", action="store_true",
        help="generate hidden-test submissions for Tent/FSTTA/EAM only",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if args.test_submissions and args.batch_id == default_batch_id:
        args.batch_id = "vln-reverie-test-submissions-v1-seed0"
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.batch_id):
        parser.error("invalid --batch-id")
    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    test_mode = args.test_submissions
    jobs = (
        expand_test_jobs(spec, args.batch_id, args.gpu)
        if test_mode else expand_jobs(spec, args.batch_id, args.gpu)
    )
    batch_root = (TEST_LOG_ROOT if test_mode else LOG_ROOT) / args.batch_id
    if args.plan_only:
        if test_mode:
            print_test_plan(spec, jobs, args.batch_id, args.gpu, args.print_commands)
        else:
            print_plan(spec, jobs, args.batch_id, args.gpu, args.print_commands)
        return 0
    if args.status:
        states, rows = collect_states(
            batch_root, jobs,
            validator=validate_test_attempt if test_mode else validate_attempt,
            cache_name="submission.json" if test_mode else "metrics.json",
        )
        print(json.dumps({
            "batch_id": args.batch_id,
            "started": (batch_root / "BATCH.json").is_file(),
            "states": states,
            "completed_results": len(rows),
        }, indent=2, sort_keys=True))
        return 0
    if not args.confirm_reviewed:
        raise UserError("formal execution requires --confirm-reviewed")
    _assert_clean_execution_tree(spec_path)
    try:
        if test_mode:
            plan_builder.validate_test_source(
                _resolve(spec["test_submission_policy"]["source"]["ledger"]),
                require_submission_artifacts=True,
            )
        else:
            plan_builder.validate_val_unseen_source(
                _resolve(spec["source_control"]["manifest"]),
                require_metric_artifacts=True,
            )
    except plan_builder.PlanError as error:
        raise UserError(str(error))
    batch_root.mkdir(parents=True, exist_ok=True)
    _prepare_batch(
        spec_path, spec, args.batch_id, batch_root, args.gpu, args.resume,
        mode="test_submission" if test_mode else "val_unseen",
    )
    phases = test_model_phases(jobs) if test_mode else model_phases(jobs)
    for phase in phases:
        run_model_phase(
            spec, spec_path, args.batch_id, batch_root, phase,
            retry_failed=args.retry_failed,
            materializer=(materialize_test_attempt if test_mode else materialize_attempt),
            validator=(validate_test_attempt if test_mode else validate_attempt),
        )
        (write_test_summary if test_mode else write_summary)(batch_root, spec, jobs)
    summary = (write_test_summary if test_mode else write_summary)(
        batch_root, spec, jobs
    )
    if not summary["complete"]:
        raise UserError("campaign ended before all jobs completed")
    if test_mode:
        print("all 9 legal REVERIE test TTA submissions completed; FeedTTA/ATENA N/A")
    else:
        print("all 15 REVERIE val_unseen frozen TTA jobs completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (UserError, ReservationLedgerError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
