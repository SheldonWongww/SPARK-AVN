#!/usr/bin/env python3
"""Run the 3-model x 5-method frozen R2R-to-REVERIE transfer matrix.

This is deliberately not a hyperparameter-search runner.  Every job is a
single, immutable R2R-selected configuration, and Source is always reused from
its tracked provenance registry.  Methods form strict barriers; models within
one method run concurrently up to the reviewed cap.
"""

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "vln/experiments/reverie_r2r_frozen_transfer_v1.json"
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
LOG_ROOT = REPO_ROOT / "vln/results/logs/reverie/frozen_transfer"
RESULT_ROOT = REPO_ROOT / "vln/results/tuning/reverie/frozen_transfer"
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_r2r_final_registry as r2r_registry_builder  # noqa: E402
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402
from shared_gpu_launch_guard import (  # noqa: E402
    ReservationLedgerError,
    release_shared_gpu_reservation,
    shared_gpu_launch_guard,
)
from joint_campaign_contract import (  # noqa: E402
    campaign_lifetime_lock,
    JointLaunchError,
    process_identity,
    process_identity_alive,
    validate_joint_launch,
    wait_for_joint_release,
    write_ready_ack,
)

SCHEMA = "navtta.vln_reverie_r2r_frozen_transfer.v1"
REGISTRY_SCHEMA = "navtta.vln_r2r_final_registry.v1"
SOURCE_SCHEMA = "navtta.vln_reverie_reused_source_controls.v1"
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
SETTINGS = ("duet-reverie", "hamt-reverie", "goat-reverie")
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
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _canonical_repo_file(value, label, required_root=None):
    path = _resolve_repo_path(value).resolve()
    root = (required_root or REPO_ROOT).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise UserError("{} is outside {}: {}".format(label, root, path))
    if not path.is_file():
        raise UserError("missing {}: {}".format(label, path))
    return path


def _validate_source_registry(spec, require_metrics_artifacts=False):
    path = _canonical_repo_file(
        spec["source_control"]["manifest"],
        "REVERIE Source provenance registry",
    )
    expected_registry_sha256 = spec["source_control"].get("sha256")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(expected_registry_sha256 or ""))
        or _sha256(path) != expected_registry_sha256
    ):
        raise UserError("REVERIE Source provenance registry SHA256 mismatch")
    source = _read_json(path)
    if source.get("schema") != SOURCE_SCHEMA:
        raise UserError("unsupported REVERIE Source registry schema")
    if source.get("benchmark") != "reverie" or source.get("split") != "val_seen":
        raise UserError("REVERIE Source registry benchmark/split mismatch")
    if source.get("episode_count") != 1423:
        raise UserError("REVERIE Source registry must describe 1,423 episodes")
    if set(source.get("records", {})) != set(SETTINGS):
        raise UserError("REVERIE Source registry must contain exactly three models")
    for setting, record in source["records"].items():
        metrics = record.get("metrics", {})
        if set(metrics) != {"SR", "SPL", "RGS", "RGSPL"}:
            raise UserError("{} Source metrics are incomplete".format(setting))
        for key in ("formal_manifest_sha256", "immutable_identity_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(key, ""))):
                raise UserError("{} has invalid {}".format(setting, key))
        manifest_path = _canonical_repo_file(
            record.get("formal_manifest_path", ""),
            "{} canonical Source manifest".format(setting),
            FORMAL_ROOT,
        )
        if _sha256(manifest_path) != record["formal_manifest_sha256"]:
            raise UserError("{} Source manifest SHA256 mismatch".format(setting))
        manifest = _read_json(manifest_path)
        expected_manifest = {
            "task": "vln",
            "benchmark": (
                "reverie_discrete_goat"
                if setting == "goat-reverie"
                else "reverie_discrete_duet_hamt"
            ),
            "status": "completed",
            "exit_code": 0,
            "model": record["model"],
            "method": "source",
            "run_tag": source["source_batch"]["batch_id"],
            "seed": source["order_seed"],
            "git_commit": source["source_batch"]["git_commit"],
            "source_setting": "{}:val_seen:native".format(setting),
            "immutable_identity_sha256": record["immutable_identity_sha256"],
        }
        for key, expected in expected_manifest.items():
            if manifest.get(key) != expected:
                raise UserError(
                    "{} Source manifest {} mismatch: expected {!r}, got {!r}"
                    .format(setting, key, expected, manifest.get(key))
                )
        identity = manifest.get("immutable_identity_sha256")
        if (
            identity != record["immutable_identity_sha256"]
            or immutable_identity_sha256(manifest) != identity
        ):
            raise UserError("{} Source immutable identity mismatch".format(setting))
        dataset = manifest.get("dataset", {})
        if dataset.get("stream_order_sha256") != source["episode_order_sha256"]:
            raise UserError("{} Source stream-order SHA256 mismatch".format(setting))
        if dataset.get("stream_content_sha256") != record["dataset_sha256"]:
            raise UserError("{} Source dataset SHA256 mismatch".format(setting))
        if manifest.get("checkpoint", {}).get("sha256") != record[
            "checkpoint_sha256"
        ]:
            raise UserError("{} Source checkpoint SHA256 mismatch".format(setting))

        metric_path = _resolve_repo_path(record.get("metrics_artifact_path", "")).resolve()
        if metric_path.is_file():
            if _sha256(metric_path) != record.get("metrics_artifact_sha256"):
                raise UserError("{} Source metrics artifact SHA256 mismatch".format(setting))
            authenticated = [
                item for item in manifest.get("result_artifacts", [])
                if isinstance(item, dict)
                and item.get("sha256") == record.get("metrics_artifact_sha256")
                and str(item.get("name", "")).endswith("valid.txt")
            ]
            if len(authenticated) != 1:
                raise UserError(
                    "{} Source metrics artifact is not authenticated by its "
                    "formal manifest".format(setting)
                )
            if authenticated[0].get("size") != metric_path.stat().st_size:
                raise UserError("{} Source metrics artifact size mismatch".format(setting))
        elif require_metrics_artifacts:
            raise UserError(
                "{} Source metrics artifact is unavailable on the execution "
                "server: {}".format(setting, metric_path)
            )
    return path, source


def load_spec(path=DEFAULT_SPEC):
    path = Path(path).resolve()
    spec = _read_json(path)
    if spec.get("schema") != SCHEMA:
        raise UserError("unsupported frozen-transfer specification schema")
    protocol = spec.get("protocol", {})
    if (
        protocol.get("benchmark") != "reverie"
        or protocol.get("split") != "val_seen"
        or protocol.get("episode_count") != 1423
        or protocol.get("order_seed") != 0
    ):
        raise UserError("frozen transfer requires canonical REVERIE val_seen")
    if tuple(protocol.get("settings", ())) != SETTINGS:
        raise UserError("frozen transfer setting order changed")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise UserError("frozen transfer method set changed")
    if protocol.get("reported_metrics") != ["SR", "SPL", "RGS", "RGSPL"]:
        raise UserError("REVERIE must report SR/SPL/RGS/RGSPL")
    feedback = protocol.get("binary_feedback", {})
    if (
        feedback.get("label") != "navigation_success_only"
        or not feedback.get("grounding_feedback_forbidden")
        or not feedback.get("simulator_distance_fallback_forbidden")
        or feedback.get("feedtta_timing") != "eager_once_per_episode"
        or feedback.get("atena_timing")
        != "lazy_only_when_entropy_gate_queries"
    ):
        raise UserError("REVERIE binary-feedback contract is incomplete")
    execution = spec.get("execution", {})
    if tuple(execution.get("method_order", ())) != METHODS:
        raise UserError("execution method order changed")
    if tuple(execution.get("setting_order", ())) != SETTINGS:
        raise UserError("execution setting order changed")
    if not execution.get("strict_method_barrier_required"):
        raise UserError("method barriers are mandatory")
    if execution.get("default_concurrency_profile") != "shared_gpu_with_r2r_ce":
        raise UserError("shared-GPU execution must remain the default")
    if not all(
        execution.get(key)
        for key in (
            "joint_launch_required",
            "joint_readiness_ack_required",
            "campaign_lifetime_lock_required",
            "shared_active_reservation_required",
        )
    ):
        raise UserError(
            "REVERIE must use the paired readiness/lock/reservation protocol"
        )
    profiles = execution.get("concurrency_profiles", {})
    shared = profiles.get("shared_gpu_with_r2r_ce", {})
    exclusive = profiles.get("exclusive_gpu", {})
    if shared.get("max_workers_by_method") != {method: 1 for method in METHODS}:
        raise UserError("shared-GPU REVERIE queue must remain single-worker")
    if shared.get("estimated_job_gpu_memory_mib", {}).keys() != set(METHODS):
        raise UserError("shared-GPU REVERIE estimates must cover every method")
    if any(
        not isinstance(value, int) or value <= 0
        for value in shared["estimated_job_gpu_memory_mib"].values()
    ):
        raise UserError("shared-GPU REVERIE memory estimates are invalid")
    for key in (
        "max_aggregate_gpu_memory_mib",
        "max_cgroup_memory_gib_before_launch",
        "estimated_job_memory_gib",
        "max_aggregate_cgroup_memory_gib",
        "shared_launch_settle_seconds",
    ):
        if not isinstance(shared.get(key), (int, float)) or shared[key] <= 0:
            raise UserError("shared-GPU profile has invalid {}".format(key))
    if exclusive.get("max_workers_by_method") != {
        "tent": 3,
        "fstta": 3,
        "eam": 3,
        "feedtta": 2,
        "atena": 2,
    }:
        raise UserError("reviewed exclusive-GPU concurrency caps changed")
    for profile_name, profile in profiles.items():
        free_memory = profile.get("minimum_free_mib_before_launch", {})
        if set(free_memory) != set(METHODS) or any(
            not isinstance(value, int) or value <= 0
            for value in free_memory.values()
        ):
            raise UserError(
                "{} has an invalid GPU memory launch gate".format(profile_name)
            )
    if (
        spec.get("source_control", {}).get("execution") != "reuse_only"
        or not spec.get("source_control", {}).get("rerun_forbidden")
    ):
        raise UserError("REVERIE Source must be reused, not rerun")

    jobs = spec.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 15:
        raise UserError("frozen transfer must contain exactly 15 TTA jobs")
    seen = set()
    for job in jobs:
        setting = job.get("setting")
        method = job.get("method")
        pair = (setting, method)
        if setting not in SETTINGS or method not in METHODS or pair in seen:
            raise UserError("invalid or duplicate frozen-transfer cell {}".format(pair))
        seen.add(pair)
        if job.get("r2r_setting") != setting.replace("reverie", "r2r"):
            raise UserError("{} has the wrong R2R anchor setting".format(pair))
        if not isinstance(job.get("parameters"), dict) or not job["parameters"]:
            raise UserError("{} has no frozen parameters".format(pair))
        if method == "tent" and job["parameters"].get("update_interval") != 1:
            raise UserError("Tent update_interval must stay fixed at one")
        if method in ("feedtta", "atena") and job["parameters"].get(
            "action_selection"
        ) != "argmax":
            raise UserError("{} must preserve target-native argmax".format(pair))
    if seen != {(setting, method) for method in METHODS for setting in SETTINGS}:
        raise UserError("frozen transfer matrix is incomplete")
    _validate_source_registry(spec)
    return spec


def _registry_entry(registry, r2r_setting, method):
    records = registry.get("records", {})
    setting_record = records.get(r2r_setting)
    if isinstance(setting_record, dict) and method in setting_record:
        return setting_record[method]
    return records.get("{}:{}".format(r2r_setting, method))


def validate_r2r_registry(spec):
    dependency = spec.get("r2r_registry_dependency", {})
    path = _resolve_repo_path(dependency.get("path", ""))
    if not path.is_file():
        raise UserError(
            "R2R final registry is not ready: {}. Plan-only is allowed, but "
            "formal REVERIE execution is fail-closed.".format(path)
        )
    registry = _read_json(path)
    if registry.get("schema") != dependency.get("schema", REGISTRY_SCHEMA):
        raise UserError("unsupported R2R final registry schema")
    if registry.get("registry_status") != dependency.get(
        "required_registry_status", "complete"
    ):
        raise UserError("R2R final registry is not complete")
    selected = registry.get("selected_winners", {})
    source = registry.get("source_ledger", {})
    for binding, label in (
        (selected, "R2R selected-winners evidence"),
        (source, "R2R Source-ledger evidence"),
    ):
        evidence_path = _canonical_repo_file(binding.get("path", ""), label)
        if not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", ""))):
            raise UserError("{} has an invalid SHA256".format(label))
        if _sha256(evidence_path) != binding["sha256"]:
            raise UserError("{} SHA256 mismatch".format(label))
    try:
        # Rebuild the registry from its pinned selection, Source ledger, and
        # all 18 canonical formal manifests.  Merely checking that digest
        # fields look like SHA256 strings would allow a hand-written registry
        # to bypass the launch gate.
        registry = r2r_registry_builder.validate_registry(
            path,
            _resolve_repo_path(selected["path"]),
            REPO_ROOT,
        )
    except r2r_registry_builder.RegistryError as error:
        raise UserError("R2R final registry provenance failed: {}".format(error))
    for job in spec["jobs"]:
        entry = _registry_entry(registry, job["r2r_setting"], job["method"])
        if not isinstance(entry, dict):
            raise UserError(
                "R2R registry lacks {} {}".format(
                    job["r2r_setting"], job["method"]
                )
            )
        if _canonical(entry.get("parameters")) != _canonical(job["parameters"]):
            raise UserError(
                "R2R registry parameters differ for {} {}".format(
                    job["r2r_setting"], job["method"]
                )
            )
        missing = set(dependency.get("required_record_fields", ())).difference(entry)
        if missing:
            raise UserError(
                "R2R registry {} {} lacks provenance fields: {}".format(
                    job["r2r_setting"], job["method"], ", ".join(sorted(missing))
                )
            )
        if not isinstance(entry.get("run_tag"), str) or entry["run_tag"].startswith(
            "pending-"
        ):
            raise UserError("R2R registry contains a pending run tag")
        if not {"SR", "SPL"}.issubset(entry.get("metrics", {})):
            raise UserError("R2R registry record lacks SR/SPL")
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(entry.get("formal_manifest_sha256", ""))
        ):
            raise UserError("R2R registry record has an invalid manifest digest")
    return path, registry


def bind_jobs_to_registry(jobs, registry):
    """Replace planning anchors with the completed registry provenance."""
    bound = []
    for job in jobs:
        entry = _registry_entry(registry, job["r2r_setting"], job["method"])
        item = dict(job)
        item["planning_r2r_anchor"] = item["r2r_anchor"]
        item["r2r_anchor"] = entry
        bound.append(item)
    return bound


def _slug_digest(job):
    identity = {
        "r2r_setting": job["r2r_setting"],
        "method": job["method"],
        "parameters": job["parameters"],
    }
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:10]


def expand_jobs(spec, batch_id, gpu=0):
    method_order = spec["execution"]["method_order"]
    setting_order = spec["execution"]["setting_order"]
    indexed = {(job["setting"], job["method"]): job for job in spec["jobs"]}
    expanded = []
    ordinal = 0
    for phase_index, method in enumerate(method_order):
        for setting_index, setting in enumerate(setting_order):
            source = indexed[(setting, method)]
            base_tag = (
                "{}-transfer-{:02d}-{}-{:02d}-{}-{}".format(
                    batch_id,
                    phase_index,
                    method,
                    setting_index,
                    setting,
                    _slug_digest(source),
                )
            )
            expanded.append({
                **source,
                "model": MODEL_FOR_SETTING[setting],
                "phase_index": phase_index,
                "setting_index": setting_index,
                "ordinal": ordinal,
                "base_run_tag": base_tag,
                "gpu": int(gpu),
            })
            ordinal += 1
    return expanded


def _attempt_tag(base_tag, attempt):
    return base_tag if attempt == 0 else "{}-retry{}".format(base_tag, attempt)


def _reservation_token(batch_id, run_tag):
    return "reverie:{}:{}".format(batch_id, run_tag)


def _attempt_dir(batch_root, job, attempt):
    return (
        batch_root
        / "phases"
        / "{:02d}-{}".format(job["phase_index"], job["method"])
        / "jobs"
        / job["base_run_tag"]
        / "attempt-{:02d}".format(attempt)
    )


def materialize_attempt(spec, spec_path, batch_id, batch_root, job, attempt):
    run_tag = _attempt_tag(job["base_run_tag"], attempt)
    attempt_dir = _attempt_dir(batch_root, job, attempt)
    result_root = (
        RESULT_ROOT
        / batch_id
        / job["model"]
        / job["method"]
        / "jobs"
        / run_tag
        / "val_seen"
    )
    formal_manifest = (
        FORMAL_ROOT
        / "{}-{}-val_seen-native".format(run_tag, job["setting"])
        / "manifest.json"
    )
    config = {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": batch_id,
        "stage": "frozen_transfer",
        "setting": job["setting"],
        "method": job["method"],
        "search_method": job["method"],
        "episodes": -1,
        "parameters": job["parameters"],
        "transfer_provenance": {
            "source_benchmark": "r2r",
            "source_setting": job["r2r_setting"],
            "r2r_registry": spec["r2r_registry_dependency"]["path"],
            "r2r_anchor": job["r2r_anchor"],
            "selection_on_reverie": False,
        },
    }
    config_path = attempt_dir / "parameters.json"
    source_registry_path, source_registry = _validate_source_registry(spec)
    source_anchor = source_registry["records"][job["setting"]]
    command = [
        str(RUNNER),
        job["setting"],
        "val_seen",
        str(job["gpu"]),
        "--run-tag",
        run_tag,
        "--tta-config",
        str(config_path),
        "--result-root",
        str(result_root),
    ]
    metadata = {
        "schema": "navtta.vln_reverie_frozen_transfer_job.v1",
        "batch_id": batch_id,
        "spec_path": str(spec_path),
        "spec_sha256": _sha256(spec_path),
        "attempt": attempt,
        "run_tag": run_tag,
        "base_run_tag": job["base_run_tag"],
        "phase_index": job["phase_index"],
        "ordinal": job["ordinal"],
        "setting": job["setting"],
        "model": job["model"],
        "method": job["method"],
        "r2r_setting": job["r2r_setting"],
        "r2r_anchor": job["r2r_anchor"],
        "parameters": job["parameters"],
        "episode_count": spec["protocol"]["episode_count"],
        "git_commit": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "expected_benchmark": (
            "reverie_discrete_goat"
            if job["setting"] == "goat-reverie"
            else "reverie_discrete_duet_hamt"
        ),
        "expected_checkpoint_sha256": source_anchor["checkpoint_sha256"],
        "expected_dataset_sha256": source_anchor["dataset_sha256"],
        "expected_episode_order_sha256": source_registry[
            "episode_order_sha256"
        ],
        "source_registry_path": str(source_registry_path),
        "source_registry_sha256": _sha256(source_registry_path),
        "result_root": str(result_root),
        "formal_manifest": str(formal_manifest),
        "command": command,
    }
    _atomic_json(config_path, config)
    _atomic_json(attempt_dir / "job.json", metadata)
    return attempt_dir, metadata


def _existing_attempts(batch_root, job):
    root = _attempt_dir(batch_root, job, 0).parent
    if not root.is_dir():
        return []
    attempts = []
    for path in root.glob("attempt-[0-9][0-9]"):
        try:
            attempts.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            continue
    return sorted(attempts)


def _pid_alive(path):
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        identity_path = path.with_name("process_identity.json")
        if identity_path.is_file():
            return process_identity_alive(_read_json(identity_path))
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _state_for_attempt(attempt_dir):
    if (attempt_dir / "metrics.json").is_file():
        return "completed"
    exit_path = attempt_dir / "exitcode"
    if exit_path.is_file():
        try:
            return "finished" if int(exit_path.read_text().strip()) == 0 else "failed"
        except ValueError:
            return "invalid"
    if _pid_alive(attempt_dir / "pid"):
        return "running"
    if (attempt_dir / "job.json").is_file():
        return "orphaned"
    return "pending"


def _latest_attempt(batch_root, job):
    attempts = _existing_attempts(batch_root, job)
    return attempts[-1] if attempts else None


METRIC_RE = re.compile(r"\b(sr|spl|rgs|rgspl):\s*(-?[0-9]+(?:\.[0-9]+)?)")


def _parse_metrics(result_root):
    matches = []
    for path in sorted(Path(result_root).rglob("valid.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            values = {key.upper(): float(value) for key, value in METRIC_RE.findall(line)}
            if set(values) == {"SR", "SPL", "RGS", "RGSPL"}:
                matches.append((path, values, line))
    if len(matches) != 1:
        raise UserError(
            "expected one REVERIE val_seen metric line under {}, found {}"
            .format(result_root, len(matches))
        )
    path, values, line = matches[0]
    for key, value in values.items():
        if not 0.0 <= value <= 100.0:
            raise UserError("invalid {} value {}".format(key, value))
    return path, values, line


def _validate_formal_manifest(path, metadata, required_artifacts=()):
    path = Path(path).resolve()
    try:
        path.relative_to(FORMAL_ROOT.resolve())
    except ValueError:
        raise UserError("formal run manifest is outside canonical results/runs")
    if not path.is_file():
        raise UserError("missing formal run manifest: {}".format(path))
    expected_run_id = "{}-{}-val_seen-native".format(
        metadata["run_tag"], metadata["setting"]
    )
    if path.name != "manifest.json" or path.parent.name != expected_run_id:
        raise UserError("formal run manifest has a noncanonical path")
    manifest = _read_json(path)
    expected = {
        "run_id": expected_run_id,
        "task": "vln",
        "benchmark": metadata["expected_benchmark"],
        "model": metadata["model"],
        "method": metadata["method"],
        "run_tag": metadata["run_tag"],
        "source_setting": "{}:val_seen:native:{}".format(
            metadata["setting"], metadata["method"]
        ),
        "seed": 0,
        "git_commit": metadata["git_commit"],
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise UserError(
                "formal manifest {} mismatch: expected {!r}, got {!r}".format(
                    key, value, manifest.get(key)
                )
            )
    if manifest.get("checkpoint", {}).get("sha256") != metadata[
        "expected_checkpoint_sha256"
    ]:
        raise UserError("formal manifest checkpoint SHA256 mismatch")
    dataset = manifest.get("dataset", {})
    if dataset.get("stream_content_sha256") != metadata[
        "expected_dataset_sha256"
    ]:
        raise UserError("formal manifest dataset SHA256 mismatch")
    if dataset.get("stream_order_sha256") != metadata[
        "expected_episode_order_sha256"
    ]:
        raise UserError("formal manifest episode-order SHA256 mismatch")
    identity = manifest.get("immutable_identity_sha256")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(identity or ""))
        or immutable_identity_sha256(manifest) != identity
    ):
        raise UserError("formal manifest immutable identity mismatch")
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise UserError("formal manifest has no result artifacts")
    authenticated = {}
    result_root = Path(metadata["result_root"]).resolve()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise UserError("formal manifest has malformed result artifacts")
        artifact_path = Path(str(artifact.get("path", ""))).resolve()
        try:
            artifact_path.relative_to(result_root)
        except ValueError:
            raise UserError("formal result artifact escapes the result root")
        if not artifact_path.is_file():
            raise UserError("formal result artifact is missing: {}".format(artifact_path))
        if artifact.get("size") != artifact_path.stat().st_size:
            raise UserError("formal result artifact size mismatch")
        if _sha256(artifact_path) != artifact.get("sha256"):
            raise UserError("formal result artifact SHA256 mismatch")
        authenticated[artifact_path] = artifact
    for required in required_artifacts:
        if Path(required).resolve() not in authenticated:
            raise UserError(
                "required result artifact is absent from formal manifest: {}"
                .format(required)
            )
    return manifest


def validate_attempt(attempt_dir):
    attempt_dir = Path(attempt_dir)
    metadata = _read_json(attempt_dir / "job.json")
    exit_code = int((attempt_dir / "exitcode").read_text().strip())
    if exit_code != 0:
        raise UserError("worker exited with status {}".format(exit_code))
    diagnostics_path = Path(metadata["result_root"]) / "tta_diagnostics.json"
    if not diagnostics_path.is_file():
        raise UserError("missing TTA diagnostics: {}".format(diagnostics_path))
    diagnostics = _read_json(diagnostics_path)
    expected_episodes = metadata["episode_count"]
    if diagnostics.get("method") != metadata["method"]:
        raise UserError("TTA diagnostics method mismatch")
    if diagnostics.get("episode_count") != expected_episodes:
        raise UserError("TTA diagnostics episode count mismatch")
    adapter = diagnostics.get("adapter", {})
    if adapter.get("episodes") != expected_episodes:
        raise UserError("adapter episode count mismatch")
    method = metadata["method"]
    if method in EXPECTED_FEEDBACK_ENDPOINT:
        endpoint = diagnostics.get("binary_feedback_endpoint")
        if endpoint != EXPECTED_FEEDBACK_ENDPOINT[method] or "distance" in endpoint:
            raise UserError("invalid REVERIE binary-feedback endpoint")
        if diagnostics.get("supervision") != "binary_navigation_success_feedback":
            raise UserError("binary-feedback supervision label mismatch")
        if method == "feedtta":
            if adapter.get("feedback_episodes") != expected_episodes:
                raise UserError("FeedTTA must consume one eager label per episode")
            if adapter.get("successful_feedback_episodes", 0) + adapter.get(
                "failed_feedback_episodes", 0
            ) != expected_episodes:
                raise UserError("FeedTTA feedback accounting mismatch")
        else:
            queries = adapter.get("queries")
            self_labels = adapter.get("self_label_episodes")
            if not isinstance(queries, int) or not isinstance(self_labels, int):
                raise UserError("ATENA query accounting is missing")
            if queries + self_labels != expected_episodes:
                raise UserError("ATENA query/self-label accounting mismatch")
    else:
        if diagnostics.get("supervision") != "unsupervised":
            raise UserError("unsupervised method has the wrong supervision label")
        if diagnostics.get("binary_feedback_endpoint") is not None:
            raise UserError("unsupervised method unexpectedly reports feedback")

    metric_path, metrics, metric_line = _parse_metrics(metadata["result_root"])
    manifest = _validate_formal_manifest(
        metadata["formal_manifest"],
        metadata,
        required_artifacts=(metric_path, diagnostics_path),
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
        "formal_immutable_identity_sha256": manifest.get(
            "immutable_identity_sha256"
        ),
    }
    _atomic_json(attempt_dir / "metrics.json", result)
    return result


def _write_worker(attempt_dir, command):
    exit_path = attempt_dir / "exitcode"
    script = "\n".join((
        "#!/usr/bin/env bash",
        "set +e",
        shlex.join(command),
        "status=$?",
        "printf '%s\\n' \"$status\" > {}".format(
            shlex.quote(str(exit_path) + ".tmp")
        ),
        "mv {} {}".format(
            shlex.quote(str(exit_path) + ".tmp"), shlex.quote(str(exit_path))
        ),
        "exit \"$status\"",
        "",
    ))
    path = attempt_dir / "worker.sh"
    _atomic_text(path, script)
    path.chmod(0o755)
    return path


def launch_attempt(attempt_dir, metadata):
    worker = _write_worker(attempt_dir, metadata["command"])
    launcher_log = (attempt_dir / "launcher.log").open("ab", buffering=0)
    process = subprocess.Popen(
        [str(worker)],
        cwd=str(REPO_ROOT),
        stdout=launcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _atomic_text(attempt_dir / "pid", "{}\n".format(process.pid))
    identity = process_identity(process.pid)
    if identity is None:
        try:
            os.killpg(process.pid, 15)
        except ProcessLookupError:
            pass
        raise UserError("cannot bind REVERIE worker process identity")
    _atomic_json(attempt_dir / "process_identity.json", identity)
    return process, launcher_log


def _collect_results(batch_root, jobs):
    rows = []
    states = {key: 0 for key in (
        "completed", "finished", "failed", "invalid", "running", "orphaned", "pending"
    )}
    for job in jobs:
        latest = _latest_attempt(batch_root, job)
        if latest is None:
            states["pending"] += 1
            continue
        _, attempt_dir = latest
        state = _state_for_attempt(attempt_dir)
        states[state] += 1
        if state == "completed":
            rows.append(_read_json(attempt_dir / "metrics.json"))
    return states, rows


def write_summary(batch_root, spec, jobs):
    states, rows = _collect_results(batch_root, jobs)
    payload = {
        "schema": "navtta.vln_reverie_frozen_transfer_summary.v1",
        "experiment_id": spec["experiment_id"],
        "states": states,
        "total_jobs": len(jobs),
        "complete": states["completed"] == len(jobs),
        "results": [{
            "setting": row["setting"],
            "method": row["method"],
            "run_tag": row["run_tag"],
            "parameters": row["parameters"],
            "metrics": row["metrics"],
            "feedback_endpoint": row["feedback_endpoint"],
            "formal_manifest": row["formal_manifest"],
            "formal_manifest_sha256": row["formal_manifest_sha256"],
        } for row in sorted(rows, key=lambda item: item["ordinal"])],
    }
    _atomic_json(batch_root / "SUMMARY.json", payload)
    with (batch_root / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["setting", "method", "SR", "SPL", "RGS", "RGSPL", "run_tag"],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["ordinal"]):
            writer.writerow({
                "setting": row["setting"],
                "method": row["method"],
                **row["metrics"],
                "run_tag": row["run_tag"],
            })
    return payload


def _prepare_batch(
    spec_path, spec, batch_id, batch_root, concurrency_profile,
    joint_launch_manifest,
):
    registry_path = _canonical_repo_file(
        spec["r2r_registry_dependency"]["path"], "R2R final registry"
    )
    source_path = _canonical_repo_file(
        spec["source_control"]["manifest"], "REVERIE Source registry"
    )
    snapshot = {
        "schema": "navtta.vln_reverie_frozen_transfer_batch.v1",
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "spec_path": str(spec_path),
        "spec_sha256": _sha256(spec_path),
        "r2r_registry": spec["r2r_registry_dependency"]["path"],
        "r2r_registry_sha256": _sha256(registry_path),
        "source_registry": spec["source_control"]["manifest"],
        "source_registry_sha256": _sha256(source_path),
        "concurrency_profile": concurrency_profile,
        "joint_launch_manifest": str(Path(joint_launch_manifest).resolve()),
    }
    path = batch_root / "BATCH.json"
    if path.is_file() and _canonical(_read_json(path)) != _canonical(snapshot):
        raise UserError("batch ID is already bound to a different immutable plan")
    _atomic_json(path, snapshot)


def _gpu_memory_mib(gpu):
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        first_line = output.splitlines()[0]
        total, used = (int(value.strip()) for value in first_line.split(","))
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as error:
        raise UserError("cannot query GPU memory for shared launch gate: {}".format(error))
    return total, used, total - used


def _cgroup_memory_gib():
    for candidate in (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        if candidate.is_file():
            try:
                return int(candidate.read_text(encoding="utf-8").strip()) / 2**30
            except (OSError, ValueError) as error:
                raise UserError("cannot query cgroup memory: {}".format(error))
    raise UserError("cannot query cgroup memory for shared launch gate")


def run_phase(
    spec,
    spec_path,
    batch_id,
    batch_root,
    jobs,
    concurrency_profile,
    retry_failed=False,
):
    method = jobs[0]["method"]
    profile = spec["execution"]["concurrency_profiles"][concurrency_profile]
    cap = profile["max_workers_by_method"][method]
    minimum_free_mib = profile["minimum_free_mib_before_launch"][method]
    poll_seconds = profile["gpu_poll_seconds"]
    launch_stagger_seconds = profile.get("launch_stagger_seconds", 0)
    shared_profile = concurrency_profile == "shared_gpu_with_r2r_ce"
    estimated_gpu_mib = (
        profile.get("estimated_job_gpu_memory_mib", {}).get(method, 0)
    )
    aggregate_gpu_cap = profile.get("max_aggregate_gpu_memory_mib")
    max_memory_gib = profile.get("max_cgroup_memory_gib_before_launch")
    estimated_memory_gib = profile.get("estimated_job_memory_gib", 0.0)
    aggregate_memory_cap = profile.get("max_aggregate_cgroup_memory_gib")
    launch_settle_seconds = profile.get("shared_launch_settle_seconds", 0.0)
    active = {}
    pending = list(jobs)
    last_gate_message = 0.0
    last_launch = 0.0
    while pending or active:
        external_running = sum(
            1
            for job in pending
            if (
                (latest := _latest_attempt(batch_root, job)) is not None
                and _state_for_attempt(latest[1]) == "running"
            )
        )
        next_pending = []
        for job in pending:
            latest = _latest_attempt(batch_root, job)
            if latest is not None:
                attempt, attempt_dir = latest
                state = _state_for_attempt(attempt_dir)
                if state == "completed":
                    if shared_profile:
                        release_shared_gpu_reservation(
                            job["gpu"],
                            _reservation_token(batch_id, _attempt_tag(
                                job["base_run_tag"], attempt
                            )),
                        )
                    continue
                if state == "finished":
                    try:
                        validate_attempt(attempt_dir)
                    except Exception as error:
                        _atomic_json(
                            attempt_dir / "validation_error.json",
                            {"error": str(error)},
                        )
                        raise
                    if shared_profile:
                        release_shared_gpu_reservation(
                            job["gpu"],
                            _reservation_token(batch_id, _attempt_tag(
                                job["base_run_tag"], attempt
                            )),
                        )
                    continue
                if state == "running":
                    next_pending.append(job)
                    continue
                if state in ("failed", "invalid", "orphaned"):
                    if not retry_failed:
                        raise UserError(
                            "{} is {}; rerun with --retry-failed".format(
                                job["base_run_tag"], state
                            )
                        )
                    attempt += 1
            else:
                attempt = 0

            if len(active) + external_running >= cap:
                next_pending.append(job)
                continue
            if time.monotonic() - last_launch < launch_stagger_seconds:
                next_pending.append(job)
                continue
            guard = (
                shared_gpu_launch_guard(job["gpu"])
                if shared_profile else nullcontext()
            )
            with guard as reservation_ledger:
                # Recheck resources while holding the same per-GPU lock used
                # by the R2R-CE scheduler.  This closes the race where both
                # campaigns observe pre-launch memory and start together.
                total_mib, used_mib, free_mib = _gpu_memory_mib(job["gpu"])
                memory_gib = _cgroup_memory_gib() if shared_profile else 0.0
                reservation_snapshot = (
                    reservation_ledger.snapshot(used_mib, memory_gib)
                    if shared_profile else {
                        "effective_gpu_memory_mib": used_mib,
                        "effective_cgroup_memory_gib": memory_gib,
                    }
                )
                effective_used_mib = reservation_snapshot[
                    "effective_gpu_memory_mib"
                ]
                effective_memory_gib = reservation_snapshot[
                    "effective_cgroup_memory_gib"
                ]
                effective_free_mib = total_mib - effective_used_mib
                resources_ok = effective_free_mib >= minimum_free_mib
                if shared_profile:
                    resources_ok = resources_ok and (
                        effective_used_mib + estimated_gpu_mib
                        <= aggregate_gpu_cap
                        and effective_memory_gib <= max_memory_gib
                        and effective_memory_gib + estimated_memory_gib
                        <= aggregate_memory_cap
                    )
                if not resources_ok:
                    next_pending.append(job)
                    now = time.monotonic()
                    if now - last_gate_message >= 60.0:
                        print(
                            "waiting for shared resources: method={} "
                            "gpu_used={}/{} MiB gpu_projected={} MiB "
                            "gpu_cap={} MiB memory={:.3f} GiB "
                            "memory_projected={:.3f} GiB memory_cap={} GiB"
                            .format(
                                method, effective_used_mib, total_mib,
                                effective_used_mib + estimated_gpu_mib,
                                aggregate_gpu_cap, effective_memory_gib,
                                effective_memory_gib + estimated_memory_gib,
                                aggregate_memory_cap,
                            )
                        )
                        last_gate_message = now
                    continue
                attempt_dir, metadata = materialize_attempt(
                    spec, spec_path, batch_id, batch_root, job, attempt
                )
                token = _reservation_token(batch_id, metadata["run_tag"])
                if shared_profile:
                    reservation_ledger.reserve(
                        token,
                        gpu_memory_mib=estimated_gpu_mib,
                        cgroup_memory_gib=estimated_memory_gib,
                        observed_gpu_memory_mib=used_mib,
                        observed_cgroup_memory_gib=memory_gib,
                        owner=process_identity(),
                        metadata={
                            "role": "reverie",
                            "batch_id": batch_id,
                            "run_tag": metadata["run_tag"],
                        },
                    )
                try:
                    process, launcher_log = launch_attempt(attempt_dir, metadata)
                    if shared_profile:
                        reservation_ledger.add_owner(
                            token,
                            _read_json(attempt_dir / "process_identity.json"),
                        )
                except Exception:
                    if shared_profile:
                        reservation_ledger.release(token)
                    raise
                active[job["base_run_tag"]] = (
                    process, launcher_log, attempt_dir, job, token
                )
                last_launch = time.monotonic()
                print("launched {} pid={}".format(metadata["run_tag"], process.pid))
                if shared_profile and launch_settle_seconds > 0:
                    time.sleep(launch_settle_seconds)
        pending = next_pending

        finished_tags = []
        for tag, (
            process, launcher_log, attempt_dir, job, reservation_token
        ) in active.items():
            return_code = process.poll()
            if return_code is None:
                continue
            launcher_log.close()
            finished_tags.append(tag)
            if shared_profile:
                release_shared_gpu_reservation(
                    job["gpu"], reservation_token
                )
            # The worker writes exitcode atomically just before it exits.
            for _ in range(20):
                if (attempt_dir / "exitcode").is_file():
                    break
                time.sleep(0.1)
            if return_code != 0:
                raise UserError(
                    "{} failed with exit code {}; inspect {}".format(
                        tag, return_code, attempt_dir / "launcher.log"
                    )
                )
            result = validate_attempt(attempt_dir)
            print(
                "completed {} SR/SPL/RGS/RGSPL={:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
                    result["run_tag"],
                    result["metrics"]["SR"],
                    result["metrics"]["SPL"],
                    result["metrics"]["RGS"],
                    result["metrics"]["RGSPL"],
                )
            )
        for tag in finished_tags:
            active.pop(tag)
        write_summary(batch_root, spec, expand_jobs(spec, batch_id, jobs[0]["gpu"]))
        if pending or active:
            time.sleep(min(2, poll_seconds))


def print_plan(
    spec, jobs, batch_id, gpu, concurrency_profile, print_commands=False
):
    profile = spec["execution"]["concurrency_profiles"][concurrency_profile]
    print(
        "batch_id={} gpu={} jobs=15 source_jobs=0 concurrency_profile={}".format(
            batch_id, gpu, concurrency_profile
        )
    )
    for method in METHODS:
        phase = [job for job in jobs if job["method"] == method]
        cap = profile["max_workers_by_method"][method]
        free = profile["minimum_free_mib_before_launch"][method]
        print(
            "{}: jobs={} max_workers={} minimum_free_mib={}".format(
                method, len(phase), cap, free
            )
        )
        for job in phase:
            print(
                "  {} <- {} {}".format(
                    job["setting"], job["r2r_setting"], _canonical(job["parameters"])
                )
            )
            if print_commands:
                config = "<BATCH_LOG_ROOT>/{}/parameters.json".format(
                    job["base_run_tag"]
                )
                result = "<RESULT_ROOT>/{}/val_seen".format(job["base_run_tag"])
                command = [
                    str(RUNNER), job["setting"], "val_seen", str(gpu),
                    "--run-tag", job["base_run_tag"], "--tta-config", config,
                    "--result-root", result,
                ]
                print("    " + shlex.join(command))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--batch-id", default="vln-reverie-r2r-frozen-transfer-v1-seed0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--concurrency-profile",
        choices=("shared_gpu_with_r2r_ce", "exclusive_gpu"),
        default=None,
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    parser.add_argument("--joint-launch-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.batch_id):
        raise UserError("invalid batch ID")
    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    concurrency_profile = (
        args.concurrency_profile
        or spec["execution"]["default_concurrency_profile"]
    )
    jobs = expand_jobs(spec, args.batch_id, args.gpu)
    batch_root = LOG_ROOT / args.batch_id

    if args.plan_only:
        print_plan(
            spec,
            jobs,
            args.batch_id,
            args.gpu,
            concurrency_profile,
            args.print_commands,
        )
        dependency = _resolve_repo_path(spec["r2r_registry_dependency"]["path"])
        print(
            "execution_gate={}".format(
                "ready" if dependency.is_file() else "waiting_for_r2r_registry"
            )
        )
        return 0
    if args.status:
        if not batch_root.is_dir():
            print("batch has not started: {}".format(args.batch_id))
            return 0
        summary = write_summary(batch_root, spec, jobs)
        print(json.dumps(summary["states"], sort_keys=True))
        return 0
    if not args.confirm_reviewed:
        raise UserError("formal launch requires --confirm-reviewed")
    if concurrency_profile != "shared_gpu_with_r2r_ce":
        raise UserError(
            "the joint campaign forbids the exclusive-GPU REVERIE profile"
        )
    try:
        authorization = validate_joint_launch(
            args.joint_launch_manifest,
            repo_root=REPO_ROOT,
            role="reverie",
            batch_id=args.batch_id,
            gpu=args.gpu,
            spec_path=spec_path,
            spec_sha256=_sha256(spec_path),
        )
    except JointLaunchError as error:
        raise UserError(str(error))
    lock_path = LOG_ROOT / ".locks" / "{}.lock".format(args.batch_id)
    try:
        with campaign_lifetime_lock(
            lock_path, role="reverie", batch_id=args.batch_id
        ):
            _validate_source_registry(spec, require_metrics_artifacts=True)
            _, registry = validate_r2r_registry(spec)
            jobs = bind_jobs_to_registry(jobs, registry)
            if batch_root.exists() and any(batch_root.iterdir()) and not args.resume:
                raise UserError("batch already exists; use --resume")
            _prepare_batch(
                spec_path, spec, args.batch_id, batch_root,
                concurrency_profile, args.joint_launch_manifest,
            )
            with shared_gpu_launch_guard(args.gpu) as reservation_ledger:
                _, used_mib, _ = _gpu_memory_mib(args.gpu)
                reservation_ledger.snapshot(used_mib, _cgroup_memory_gib())
            _, launch_document = authorization
            write_ready_ack(
                args.joint_launch_manifest,
                launch_document,
                role="reverie",
                batch_id=args.batch_id,
                spec_sha256=_sha256(spec_path),
            )
            wait_for_joint_release(
                args.joint_launch_manifest,
                launch_document,
                role="reverie",
            )

            for method in METHODS:
                phase_jobs = [job for job in jobs if job["method"] == method]
                print("starting strict method phase {}".format(method))
                run_phase(
                    spec,
                    spec_path,
                    args.batch_id,
                    batch_root,
                    phase_jobs,
                    concurrency_profile,
                    retry_failed=args.retry_failed,
                )
            summary = write_summary(batch_root, spec, jobs)
            if not summary["complete"]:
                raise UserError("frozen transfer ended without 15 validated jobs")
    except JointLaunchError as error:
        raise UserError(str(error))
    print("all 15 REVERIE frozen-transfer jobs completed and validated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (UserError, ReservationLedgerError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
