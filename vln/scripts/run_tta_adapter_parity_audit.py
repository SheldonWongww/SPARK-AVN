#!/usr/bin/env python3
"""Plan, run, and strictly validate the 256-episode VLN adapter parity audit.

This command is intentionally separate from ``run_tta_hparam_search.py``.
It consumes already-frozen winners, creates exactly 56 evidence jobs, and can
never promote or search a configuration.  Adapter jobs execute their complete
native control flow with parameter writes suppressed in ``navtta_core``.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from navtta_core.experiment.episode_order import (  # noqa: E402
    load_episode_order_manifest,
    prefix_episode_order_manifest,
)
from run_manifest_identity import (  # noqa: E402
    IMMUTABLE_IDENTITY_SHA256_FIELD,
    immutable_identity_sha256,
)

SPEC_PATH = REPO_ROOT / "vln/experiments/tta_adapter_parity_audit_v1.json"
SEARCH_ROOT = REPO_ROOT / "vln/results/logs/hparam_search"
AUDIT_ROOT = REPO_ROOT / "vln/results/audits/adapter_parity/campaigns"
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
PREFLIGHT = REPO_ROOT / "vln/scripts/verify_preflight.py"
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
SETTINGS = (
    "duet-r2r", "duet-reverie", "hamt-r2r", "hamt-reverie",
    "goat-r2r", "goat-reverie", "etpnav-r2r-ce", "bevbert-r2r-ce",
)
MODEL = {
    "duet-r2r": "duet", "duet-reverie": "duet",
    "hamt-r2r": "hamt", "hamt-reverie": "hamt",
    "goat-r2r": "goat", "goat-reverie": "goat",
    "etpnav-r2r-ce": "etpnav", "bevbert-r2r-ce": "bevbert",
}
CONTINUOUS = {"etpnav-r2r-ce", "bevbert-r2r-ce"}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class AuditError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_sha256_matches(path, digest):
    """Return whether a JSON-recorded regular file still has its exact hash."""
    if (
        not isinstance(path, str)
        or not Path(path).is_file()
        or HEX_SHA256.fullmatch(str(digest)) is None
    ):
        return False
    try:
        return sha256(path) == digest
    except OSError:
        return False


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT)] + list(args), text=True
    ).strip()


def read_json(path, label="JSON file"):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError("cannot read {} {}: {}".format(label, path, error))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if canonical(read_json(path)) != canonical(value):
            raise AuditError("immutable audit artifact changed: {}".format(path))
        return
    atomic_json(path, value)


def load_spec(path=SPEC_PATH):
    spec = read_json(path, "audit specification")
    if spec.get("schema") != "navtta.vln_tta_adapter_parity_audit.v1":
        raise AuditError("unsupported adapter-parity audit schema")
    if spec.get("namespace") != "adapter_parity_audit":
        raise AuditError("adapter-parity audit namespace is invalid")
    if spec.get("split") != "val_seen":
        raise AuditError("adapter-parity audit is val_seen-only")
    if int(spec.get("canonical_prefix_episodes", -1)) != 256:
        raise AuditError("adapter-parity audit requires the canonical 256 prefix")
    if tuple(spec.get("methods", ())) != METHODS:
        raise AuditError("adapter-parity method set/order changed")
    if tuple(spec.get("settings", ())) != SETTINGS:
        raise AuditError("adapter-parity setting set/order changed")
    if spec.get("expected_jobs") != {
        "adapter_zero_update": 40,
        "source_argmax": 8,
        "source_sampled": 8,
        "total": 56,
    }:
        raise AuditError("adapter-parity audit must contain exactly 56 jobs")
    protocol = spec.get("protocol", {})
    required_guards = {
        "formal_evidence", "canonical_prefix_only",
        "frozen_hyperparameters_only", "full_adapter_control_flow",
        "suppress_all_parameter_writes",
        "require_positive_suppressed_write_attempts",
        "require_zero_updates", "require_zero_relative_parameter_drift",
        "require_equal_parameter_state_hash",
        "require_equal_full_model_state_hash",
        "require_method_specific_positive_evidence",
        "require_formal_run_manifest_per_job",
        "require_canonical_prefix_manifest",
        "require_exact_episode_ids_and_order",
        "require_exact_action_trajectory_hash",
        "require_exact_matched_source_metrics",
        "require_all_asset_preflight",
        "ordinary_hparam_search_entry_forbidden",
    }
    if not required_guards.issubset(protocol) or not all(
        protocol[name] is True for name in required_guards
    ):
        raise AuditError("adapter-parity protocol guards cannot be weakened")
    return spec


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def order_identity(setting, spec):
    path = resolve_repo_path(spec["episode_order_manifests"][setting])
    try:
        document = load_episode_order_manifest(
            str(path), expected_split="val_seen"
        )
    except (OSError, ValueError) as error:
        raise AuditError(
            "invalid canonical order manifest for {}: {}".format(
                setting, error
            )
        )
    count = int(spec["canonical_prefix_episodes"])
    episodes = document.get("episodes")
    if (
        document.get("schema") != "navtta.episode_order.v1"
        or document.get("split") != "val_seen"
        or not isinstance(episodes, list)
        or len(episodes) < count
    ):
        raise AuditError("invalid canonical order manifest for {}".format(setting))
    identifiers = [str(item["episode_id"]) for item in episodes[:count]]
    if len(set(identifiers)) != count:
        raise AuditError("canonical prefix IDs are not unique for {}".format(setting))
    prefix = prefix_episode_order_manifest(document, count)
    identifier_sha = hashlib.sha256(
        canonical(identifiers).encode("utf-8")
    ).hexdigest()
    dataset = document.get("dataset", {})
    return {
        "path": str(path),
        "manifest_sha256": sha256(path),
        "benchmark": document.get("benchmark"),
        "dataset_path": dataset.get("path"),
        "dataset_sha256": dataset.get("sha256"),
        "full_order_sha256": document.get("order_sha256"),
        "prefix_episode_count": count,
        "prefix_episode_ids": identifiers,
        "prefix_episode_ids_sha256": identifier_sha,
        "prefix_order_sha256": prefix["order_sha256"],
    }


def validate_git_commit_object(commit, label):
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise AuditError("{} is not a full Git commit".format(label))
    try:
        git("cat-file", "-e", commit + "^{commit}")
    except (OSError, subprocess.CalledProcessError):
        raise AuditError("{} is not present in repository history".format(label))


def validate_search_stage(
    method_root, search_batch_id, method, stage, commit, spec_sha,
    search_spec,
):
    stage_root = Path(method_root) / "stages" / stage
    manifest_path = stage_root / "stage_manifest.json"
    manifest = read_json(manifest_path, "search stage manifest")
    expected = {
        "schema": "navtta.vln_tta_search_stage.v1",
        "batch_id": search_batch_id,
        "git_commit": commit,
        "spec_sha256": spec_sha,
        "method": method,
        "stage": stage,
        "episodes": -1,
        "settings": list(SETTINGS),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise AuditError(
            "{} {} search stage identity is invalid".format(method, stage)
        )
    job_paths = sorted((stage_root / "jobs").glob("*/job.json"))
    if (
        not job_paths
        or isinstance(manifest.get("job_count"), bool)
        or manifest.get("job_count") != len(job_paths)
    ):
        raise AuditError("{} {} search stage jobs are incomplete".format(
            method, stage
        ))
    summary_path = stage_root / "SUMMARY.json"
    summary = read_json(summary_path, "search stage summary")
    if (
        summary.get("schema") != "navtta.vln_tta_stage_summary.v1"
        or summary.get("planned") != len(job_paths)
        or summary.get("validated") != len(job_paths)
        or summary.get("errors") != []
        or summary.get("terminal") is not True
        or summary.get("complete") is not True
    ):
        raise AuditError("{} {} search stage is not complete".format(
            method, stage
        ))
    jobs = {}
    evidence_sha256 = {}
    ordinals = []
    expected_config_method = method if stage == "final" else "source"
    for job_path in job_paths:
        job = read_json(job_path, "search stage job")
        job_dir = Path(job.get("job_dir", ""))
        config_path = Path(job.get("config_path", ""))
        metrics_path = job_dir / "metrics.json"
        exit_path = job_dir / "exitcode"
        state_path = job_dir / "worker_state.json"
        if (
            job.get("search_method") != method
            or job.get("config_method") != expected_config_method
            or job.get("stage") != stage
            or job.get("setting") not in SETTINGS
            or job.get("episodes") != -1
            or not isinstance(job.get("run_tag"), str)
            or not job["run_tag"]
            or job.get("job_dir") != str(job_path.parent)
            or config_path != job_path.parent / "parameters.json"
            or isinstance(job.get("ordinal"), bool)
            or not isinstance(job.get("ordinal"), int)
        ):
            raise AuditError("invalid search job provenance: {}".format(job_path))
        if job["run_tag"] in jobs:
            raise AuditError("duplicate search run tag: {}".format(job["run_tag"]))
        ordinals.append(job["ordinal"])

        config = read_json(config_path, "search job config")
        expected_config = {
            "schema": "navtta.vln_tta_job.v1",
            "method": expected_config_method,
            "search_method": method,
            "stage": stage,
            "episodes": -1,
            "order_seed": job.get("order_seed"),
            "parameters": job.get("parameters"),
        }
        if any(config.get(key) != value for key, value in expected_config.items()):
            raise AuditError("search job config mismatch: {}".format(config_path))

        try:
            exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as error:
            raise AuditError("invalid search exit evidence {}: {}".format(
                exit_path, error
            ))
        state = read_json(state_path, "search worker state")
        if (
            exit_code != 0
            or state.get("status") != "finished"
            or state.get("exit_code") != 0
        ):
            raise AuditError("search job did not complete: {}".format(
                job["run_tag"]
            ))

        metrics = read_json(metrics_path, "search metrics evidence")
        identity_fields = (
            "run_tag", "setting", "search_method", "config_method", "stage",
            "episodes", "order_seed", "parameters", "config_path", "job_dir",
            "result_root",
        )
        if any(
            canonical(metrics.get(key)) != canonical(job.get(key))
            for key in identity_fields
        ):
            raise AuditError("search metrics/job identity mismatch: {}".format(
                job["run_tag"]
            ))
        expected_episodes = int(
            search_spec["setting_episode_counts"][job["setting"]]
        )
        metric_values = metrics.get("metrics")
        if (
            metrics.get("expected_episodes") != expected_episodes
            or not isinstance(metric_values, dict)
            or not metric_values
            or not _finite(metric_values)
            or not required_metric_keys(job["setting"]).issubset(metric_values)
        ):
            raise AuditError("search metric evidence is incomplete: {}".format(
                job["run_tag"]
            ))

        diagnostics_path = metrics.get("diagnostics_path")
        diagnostics_sha = metrics.get("diagnostics_sha256")
        if expected_config_method == method:
            if (
                not recorded_sha256_matches(
                    diagnostics_path, diagnostics_sha
                )
                or not isinstance(metrics.get("adapter_diagnostics"), dict)
            ):
                raise AuditError("winner diagnostics evidence is invalid: {}".format(
                    job["run_tag"]
                ))
        else:
            if (diagnostics_path is None) != (diagnostics_sha is None):
                raise AuditError(
                    "Source diagnostics path/hash mismatch: {}".format(
                        job["run_tag"]
                    )
                )
            if diagnostics_path is None:
                if metrics.get("adapter_diagnostics") is not None:
                    raise AuditError(
                        "Source diagnostics payload has no evidence: {}".format(
                            job["run_tag"]
                        )
                    )
            elif not recorded_sha256_matches(
                diagnostics_path, diagnostics_sha
            ):
                raise AuditError(
                    "Source diagnostics evidence is invalid: {}".format(
                        job["run_tag"]
                    )
                )

        evidence_paths = [
            job_path, config_path, metrics_path, exit_path, state_path,
        ]
        if diagnostics_path is not None:
            evidence_paths.append(Path(diagnostics_path))
        evidence_sha256[job["run_tag"]] = {
            str(path.resolve()): sha256(path) for path in evidence_paths
        }
        jobs[job["run_tag"]] = {
            "job": job,
            "config": config,
            "metrics": metrics,
        }
    if sorted(ordinals) != list(range(len(job_paths))):
        raise AuditError("{} {} search ordinals are invalid".format(method, stage))
    setting_counts = {
        setting: sum(
            record["job"]["setting"] == setting for record in jobs.values()
        )
        for setting in SETTINGS
    }
    expected_per_setting = (
        int(search_spec["protocol"][
            "full_val_seen_finalists_per_setting"
        ])
        if stage == "final" else 1
    )
    if (
        expected_per_setting <= 0
        or any(
            count != expected_per_setting
            for count in setting_counts.values()
        )
    ):
        raise AuditError("{} {} setting coverage is invalid".format(method, stage))
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
        "job_count": len(job_paths),
        "job_sha256": {
            str(path.resolve()): sha256(path) for path in job_paths
        },
        "evidence_sha256": evidence_sha256,
        "jobs": jobs,
    }


def load_frozen(search_root, search_batch_id, spec):
    search_root = Path(search_root)
    search_spec_path = resolve_repo_path(spec["identity_files"]["search_spec"])
    expected_search_sha = sha256(search_spec_path)
    search_spec = read_json(search_spec_path, "hyperparameter search spec")
    if (
        search_spec.get("schema") != "navtta.vln_tta_hparam_search.v1"
        or tuple(search_spec.get("settings", ())) != SETTINGS
        or int(search_spec.get("primary_order_seed", -1)) != 0
    ):
        raise AuditError("hyperparameter search specification is invalid")
    frozen = {}
    search_commits = set()
    for method in METHODS:
        method_root = search_root / method / search_batch_id
        path = method_root / "FROZEN_HPARAMETERS.json"
        selection_path = method_root / "FINAL_SELECTION.json"
        document = read_json(path, "frozen hyperparameters")
        selection = read_json(selection_path, "final selection")
        if (
            document.get("schema") != "navtta.vln_tta_frozen_hparams.v1"
            or document.get("method") != method
            or document.get("split") != "val_seen"
            or document.get("generated_from") != "FINAL_SELECTION.json"
            or document.get("spec_sha256") != expected_search_sha
            or set(document.get("settings", {})) != set(SETTINGS)
        ):
            raise AuditError("invalid frozen search artifact for {}".format(method))
        commit = document.get("git_commit")
        validate_git_commit_object(commit, "{} frozen commit".format(method))
        if (
            selection.get("schema") != "navtta.vln_tta_final_selection.v1"
            or selection.get("method") != method
            or selection.get("split") != "val_seen"
            or selection.get("finalist_stage") != "final"
            or selection.get("matched_source_stage") != "final_controls"
            or selection.get("git_commit") != commit
            or selection.get("spec_sha256") != expected_search_sha
            or set(selection.get("settings", {})) != set(SETTINGS)
        ):
            raise AuditError("invalid final-selection artifact for {}".format(method))
        final_stage = validate_search_stage(
            method_root, search_batch_id, method, "final", commit,
            expected_search_sha, search_spec,
        )
        control_stage = validate_search_stage(
            method_root, search_batch_id, method, "final_controls", commit,
            expected_search_sha, search_spec,
        )
        search_commits.add(commit)
        for setting in SETTINGS:
            selected = selection["settings"].get(setting)
            winner = final_stage["jobs"].get(
                selected.get("winner_run_tag") if isinstance(selected, dict)
                else None
            )
            source = control_stage["jobs"].get(
                selected.get("source_run_tag") if isinstance(selected, dict)
                else None
            )
            expected_source_parameters = {
                "action_selection": (
                    "sample" if method == "feedtta" else "argmax"
                ),
                "action_seed": int(search_spec["primary_order_seed"]),
            }
            if (
                not isinstance(document["settings"][setting], dict)
                or not isinstance(selected, dict)
                or canonical(selected.get("frozen_parameters"))
                != canonical(document["settings"][setting])
                or winner is None
                or source is None
                or winner["job"].get("setting") != setting
                or winner["job"].get("search_method") != method
                or winner["job"].get("config_method") != method
                or canonical(winner["job"].get("parameters"))
                != canonical(document["settings"][setting])
                or canonical(winner["config"].get("parameters"))
                != canonical(document["settings"][setting])
                or canonical(selected.get("winner_metrics"))
                != canonical(winner["metrics"].get("metrics"))
                or source["job"].get("setting") != setting
                or source["job"].get("search_method") != method
                or source["job"].get("config_method") != "source"
                or canonical(source["job"].get("parameters"))
                != canonical(expected_source_parameters)
                or canonical(source["config"].get("parameters"))
                != canonical(expected_source_parameters)
                or canonical(selected.get("source_metrics"))
                != canonical(source["metrics"].get("metrics"))
            ):
                raise AuditError("frozen parameters are invalid for {} {}".format(
                    method, setting
                ))
        frozen[method] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": sha256(selection_path),
            "final_stage": {
                key: value for key, value in final_stage.items()
                if key != "jobs"
            },
            "final_controls_stage": {
                key: value for key, value in control_stage.items()
                if key != "jobs"
            },
            "document": document,
        }
    if len(search_commits) != 1:
        raise AuditError("frozen methods do not share one search commit")
    return frozen, next(iter(search_commits))


def point_digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:10]


def campaign_root(batch_id):
    return AUDIT_ROOT / batch_id


def result_root(run_tag, setting):
    return (
        REPO_ROOT / "vln/results/audits/adapter_parity/runs"
        / run_tag / setting / "val_seen"
    )


def run_data_version(setting):
    return "v1.3-unified" if setting in CONTINUOUS else "native"


def formal_manifest_path(run_tag, setting):
    run_id = "{}-{}-val_seen-{}".format(
        run_tag, setting, run_data_version(setting)
    )
    return REPO_ROOT / "vln/results/runs" / run_id / "manifest.json"


def _job(batch_id, ordinal, kind, setting, gpu, method, parameters,
         campaign, frozen_source=None):
    identity = {
        "kind": kind, "setting": setting, "method": method,
        "parameters": parameters,
    }
    digest = point_digest(identity)
    run_tag = "{}-{:02d}-{}-{}-{}".format(
        batch_id, ordinal, kind.replace("source_", "src_"), setting, digest
    )
    job_dir = Path(campaign) / "jobs" / "{:02d}-{}-{}".format(
        ordinal, setting, digest
    )
    config_path = job_dir / "audit_config.json"
    command = [
        str(RUNNER), setting, "val_seen", str(gpu),
        "--run-tag", run_tag, "--tta-config", str(config_path),
        "--adapter-parity-audit", "--episode-limit", "256",
    ]
    return {
        "schema": "navtta.vln_tta_adapter_parity_planned_job.v1",
        "namespace": "adapter_parity_audit",
        "batch_id": batch_id,
        "ordinal": ordinal,
        "kind": kind,
        "setting": setting,
        "model": MODEL[setting],
        "family": "continuous" if setting in CONTINUOUS else "discrete",
        "method": method,
        "comparison_control": (
            "source_sampled" if method == "feedtta" else "source_argmax"
        ) if kind == "adapter_zero_update" else None,
        "episodes": 256,
        "order_seed": 0,
        "parameters": parameters,
        "frozen_source": frozen_source,
        "run_tag": run_tag,
        "job_dir": str(job_dir),
        "config_path": str(config_path),
        "result_root": str(result_root(run_tag, setting)),
        "prefix_manifest_path": str(
            result_root(run_tag, setting) / "episode_order_prefix.json"
        ),
        "formal_manifest_path": str(formal_manifest_path(run_tag, setting)),
        "run_data_version": run_data_version(setting),
        "command": command,
    }


def build_jobs(batch_id, search_batch_id, gpu, campaign, frozen):
    jobs = []
    ordinal = 0
    for kind, selection in (
        ("source_argmax", "argmax"),
        ("source_sampled", "sample"),
    ):
        for setting in SETTINGS:
            jobs.append(_job(
                batch_id, ordinal, kind, setting, gpu, "source",
                {"action_selection": selection, "action_seed": 0}, campaign,
            ))
            ordinal += 1
    for method in METHODS:
        for setting in SETTINGS:
            parameters = dict(frozen[method]["document"]["settings"][setting])
            jobs.append(_job(
                batch_id, ordinal, "adapter_zero_update", setting, gpu,
                method, parameters, campaign,
                frozen_source={
                    "search_batch_id": search_batch_id,
                    "path": frozen[method]["path"],
                    "sha256": frozen[method]["sha256"],
                },
            ))
            ordinal += 1
    if len(jobs) != 56:
        raise AuditError("internal audit plan did not produce 56 jobs")
    return jobs


def job_config(job, spec_sha):
    adapter = job["kind"] == "adapter_zero_update"
    return {
        "schema": "navtta.vln_tta_adapter_parity_job.v1",
        "namespace": "adapter_parity_audit",
        "audit_spec_sha256": spec_sha,
        "audit_zero_update": adapter,
        "audit_control": not adapter,
        "method": job["method"],
        "setting": job["setting"],
        "split": "val_seen",
        "episodes": 256,
        "order_seed": 0,
        "parameters": job["parameters"],
        "frozen_source": job["frozen_source"],
    }


def create_plan(batch_id, search_batch_id, gpu=0, search_root=SEARCH_ROOT):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", batch_id):
        raise AuditError("invalid audit batch ID")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", search_batch_id):
        raise AuditError("invalid search batch ID")
    if int(gpu) < 0:
        raise AuditError("GPU index must be nonnegative")
    root = campaign_root(batch_id)
    if root.exists() and any(root.iterdir()):
        raise AuditError("audit campaign already exists: {}".format(root))
    spec = load_spec()
    frozen, search_commit = load_frozen(search_root, search_batch_id, spec)
    orders = {setting: order_identity(setting, spec) for setting in SETTINGS}
    jobs = build_jobs(batch_id, search_batch_id, gpu, root, frozen)
    spec_sha = sha256(SPEC_PATH)
    root.mkdir(parents=True, exist_ok=False)
    job_records = []
    for job in jobs:
        directory = Path(job["job_dir"])
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(job["config_path"], job_config(job, spec_sha))
        atomic_json(directory / "job.json", job)
        job_records.append({
            "ordinal": job["ordinal"],
            "job_path": str(directory / "job.json"),
            "job_sha256": sha256(directory / "job.json"),
            "config_path": job["config_path"],
            "config_sha256": sha256(job["config_path"]),
        })
    identities = {}
    for name, relative in spec["identity_files"].items():
        path = resolve_repo_path(relative)
        identities[name] = {"path": str(path), "sha256": sha256(path)}
    plan = {
        "schema": "navtta.vln_tta_adapter_parity_plan.v1",
        "namespace": "adapter_parity_audit",
        "batch_id": batch_id,
        "search_batch_id": search_batch_id,
        "git_commit": git("rev-parse", "HEAD"),
        "search_git_commit": search_commit,
        "spec_path": str(SPEC_PATH),
        "spec_sha256": spec_sha,
        "identity_files": identities,
        "frozen_hyperparameters": {
            method: {
                key: value for key, value in item.items()
                if key != "document"
            }
            for method, item in frozen.items()
        },
        "episode_orders": orders,
        "split": "val_seen",
        "canonical_prefix_episodes": 256,
        "order_seed": 0,
        "job_count": 56,
        "job_records": job_records,
    }
    write_immutable_json(root / "PLAN.json", plan)
    return root, plan


def load_plan(batch_id):
    root = campaign_root(batch_id)
    return root, read_json(root / "PLAN.json", "audit plan")


def validate_plan(root, plan, require_current_commit=True):
    spec = load_spec()
    if (
        plan.get("schema") != "navtta.vln_tta_adapter_parity_plan.v1"
        or plan.get("namespace") != "adapter_parity_audit"
        or plan.get("job_count") != 56
        or plan.get("canonical_prefix_episodes") != 256
        or plan.get("order_seed") != 0
    ):
        raise AuditError("audit PLAN.json identity is invalid")
    if require_current_commit and plan.get("git_commit") != git("rev-parse", "HEAD"):
        raise AuditError("audit plan commit differs from current HEAD")
    validate_git_commit_object(plan.get("git_commit"), "audit plan commit")
    validate_git_commit_object(
        plan.get("search_git_commit"), "audit search commit"
    )
    if plan.get("spec_sha256") != sha256(SPEC_PATH):
        raise AuditError("audit specification changed after planning")
    if set(plan.get("identity_files", {})) != set(spec["identity_files"]):
        raise AuditError("audit identity-file set is incomplete")
    for name, expected in plan.get("identity_files", {}).items():
        path = Path(expected["path"])
        if sha256(path) != expected["sha256"]:
            raise AuditError("pinned identity file changed: {}".format(name))
    if set(plan.get("frozen_hyperparameters", {})) != set(METHODS):
        raise AuditError("audit frozen-method set is incomplete")
    for method, expected in plan.get("frozen_hyperparameters", {}).items():
        if (
            method not in METHODS
            or sha256(expected["path"]) != expected["sha256"]
            or sha256(expected["selection_path"])
            != expected["selection_sha256"]
        ):
            raise AuditError("frozen parameters changed: {}".format(method))
        frozen_document = read_json(expected["path"], "frozen parameters")
        selection_document = read_json(
            expected["selection_path"], "final selection"
        )
        if (
            frozen_document.get("git_commit") != plan["search_git_commit"]
            or selection_document.get("git_commit")
            != plan["search_git_commit"]
            or frozen_document.get("generated_from")
            != "FINAL_SELECTION.json"
        ):
            raise AuditError("frozen provenance changed: {}".format(method))
        for stage_name in ("final_stage", "final_controls_stage"):
            stage = expected.get(stage_name, {})
            if not recorded_sha256_matches(
                stage.get("manifest_path"), stage.get("manifest_sha256")
            ):
                raise AuditError("frozen stage manifest changed: {}".format(method))
            if not recorded_sha256_matches(
                stage.get("summary_path"), stage.get("summary_sha256")
            ):
                raise AuditError("frozen stage summary changed: {}".format(method))
            job_hashes = stage.get("job_sha256")
            if (
                not isinstance(job_hashes, dict)
                or len(job_hashes) != stage.get("job_count")
                or any(not recorded_sha256_matches(path, digest)
                       for path, digest in job_hashes.items())
            ):
                raise AuditError("frozen stage jobs changed: {}".format(method))
            jobs_by_tag = {}
            for job_path in job_hashes:
                job = read_json(job_path, "frozen stage job")
                run_tag = job.get("run_tag")
                if not isinstance(run_tag, str) or run_tag in jobs_by_tag:
                    raise AuditError(
                        "frozen stage job identities changed: {}".format(method)
                    )
                jobs_by_tag[run_tag] = (Path(job_path).resolve(), job)

            evidence_hashes = stage.get("evidence_sha256")
            if (
                not isinstance(evidence_hashes, dict)
                or set(evidence_hashes) != set(jobs_by_tag)
            ):
                raise AuditError(
                    "frozen stage evidence set changed: {}".format(method)
                )
            for run_tag, (job_path, job) in jobs_by_tag.items():
                evidence = evidence_hashes.get(run_tag)
                if not isinstance(evidence, dict):
                    raise AuditError(
                        "frozen stage evidence changed: {} {}".format(
                            method, run_tag
                        )
                    )
                job_dir = Path(job.get("job_dir", ""))
                config_path = Path(job.get("config_path", ""))
                metrics_path = job_dir / "metrics.json"
                metrics = read_json(metrics_path, "frozen search metrics")
                expected_paths = {
                    str(job_path),
                    str(config_path.resolve()),
                    str(metrics_path.resolve()),
                    str((job_dir / "exitcode").resolve()),
                    str((job_dir / "worker_state.json").resolve()),
                }
                diagnostics_path = metrics.get("diagnostics_path")
                if diagnostics_path is not None:
                    expected_paths.add(str(Path(diagnostics_path).resolve()))
                if (
                    set(evidence) != expected_paths
                    or any(
                        not recorded_sha256_matches(path, digest)
                        for path, digest in evidence.items()
                    )
                ):
                    raise AuditError(
                        "frozen stage evidence changed: {} {}".format(
                            method, run_tag
                        )
                    )
    for setting in SETTINGS:
        current = order_identity(setting, spec)
        if canonical(current) != canonical(plan["episode_orders"].get(setting)):
            raise AuditError("canonical order identity changed: {}".format(setting))
    records = plan.get("job_records")
    if not isinstance(records, list) or len(records) != 56:
        raise AuditError("audit job record count is not 56")
    jobs = []
    for ordinal, record in enumerate(records):
        if record.get("ordinal") != ordinal:
            raise AuditError("audit job ordinals are not contiguous")
        if sha256(record["job_path"]) != record["job_sha256"]:
            raise AuditError("planned job changed: {}".format(ordinal))
        if sha256(record["config_path"]) != record["config_sha256"]:
            raise AuditError("audit config changed: {}".format(ordinal))
        job = read_json(record["job_path"], "planned audit job")
        config = read_json(record["config_path"], "audit job config")
        command = job.get("command")
        if not isinstance(command, list) or len(command) != 11:
            raise AuditError("planned job command shape is invalid")
        expected_command = [
            str(RUNNER), job.get("setting"), "val_seen",
            str(command[3]),
            "--run-tag", job.get("run_tag"),
            "--tta-config", job.get("config_path"),
            "--adapter-parity-audit", "--episode-limit", "256",
        ]
        if (
            job.get("schema")
            != "navtta.vln_tta_adapter_parity_planned_job.v1"
            or job.get("namespace") != "adapter_parity_audit"
            or job.get("ordinal") != ordinal
            or job.get("setting") not in SETTINGS
            or job.get("model") != MODEL.get(job.get("setting"))
            or job.get("method") not in METHODS + ("source",)
            or job.get("episodes") != 256
            or job.get("order_seed") != 0
            or job.get("result_root") != str(result_root(
                job.get("run_tag"), job.get("setting")
            ))
            or job.get("prefix_manifest_path") != str(
                result_root(job.get("run_tag"), job.get("setting"))
                / "episode_order_prefix.json"
            )
            or job.get("formal_manifest_path") != str(formal_manifest_path(
                job.get("run_tag"), job.get("setting")
            ))
            or job.get("run_data_version") != run_data_version(
                job.get("setting")
            )
            or job.get("command") != expected_command
            or "run_tta_hparam_search.py" in " ".join(job.get("command", []))
            or job.get("command", []).count("--adapter-parity-audit") != 1
            or job.get("command", []).count("--episode-limit") != 1
            or job["command"][job["command"].index("--episode-limit") + 1]
            != "256"
        ):
            raise AuditError("planned job can enter an invalid protocol")
        adapter = job.get("kind") == "adapter_zero_update"
        if (
            config.get("schema") != "navtta.vln_tta_adapter_parity_job.v1"
            or config.get("namespace") != "adapter_parity_audit"
            or config.get("audit_spec_sha256") != plan["spec_sha256"]
            or config.get("audit_zero_update") is not adapter
            or config.get("audit_control") is not (not adapter)
            or config.get("method") != job.get("method")
            or config.get("setting") != job.get("setting")
            or config.get("split") != "val_seen"
            or config.get("episodes") != 256
            or config.get("order_seed") != 0
            or canonical(config.get("parameters"))
            != canonical(job.get("parameters"))
        ):
            raise AuditError("audit config/job mismatch: {}".format(ordinal))
        if adapter:
            frozen_identity = plan["frozen_hyperparameters"].get(job["method"])
            frozen_document = read_json(
                frozen_identity["path"], "frozen audit parameter source"
            )
            expected_source = {
                "search_batch_id": plan["search_batch_id"],
                "path": frozen_identity["path"],
                "sha256": frozen_identity["sha256"],
            }
            if (
                canonical(job.get("frozen_source"))
                != canonical(expected_source)
                or canonical(config.get("frozen_source"))
                != canonical(expected_source)
                or canonical(job.get("parameters"))
                != canonical(frozen_document["settings"][job["setting"]])
            ):
                raise AuditError("adapter job is not the frozen winner")
        else:
            expected_selection = (
                "argmax" if job.get("kind") == "source_argmax" else "sample"
            )
            if (
                job.get("method") != "source"
                or job.get("frozen_source") is not None
                or config.get("frozen_source") is not None
                or job.get("parameters") != {
                    "action_selection": expected_selection,
                    "action_seed": 0,
                }
            ):
                raise AuditError("Source control configuration drifted")
        jobs.append(job)
    kinds = [job["kind"] for job in jobs]
    if (
        kinds.count("adapter_zero_update") != 40
        or kinds.count("source_argmax") != 8
        or kinds.count("source_sampled") != 8
    ):
        raise AuditError("audit plan is not the required 40+8+8 design")
    identities = [(job["kind"], job["method"], job["setting"]) for job in jobs]
    expected = []
    for kind in ("source_argmax", "source_sampled"):
        expected.extend((kind, "source", setting) for setting in SETTINGS)
    expected.extend(
        ("adapter_zero_update", method, setting)
        for method in METHODS for setting in SETTINGS
    )
    if identities != expected:
        raise AuditError("audit job identity/order differs from the formal plan")
    return jobs


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def worker_main(job_path):
    job = read_json(job_path, "worker job")
    directory = Path(job["job_dir"])
    state_path = directory / "worker_state.json"
    exit_path = directory / "exitcode"
    launcher_log = directory / "launcher.log"
    child = {"process": None}

    def forward(signum, frame):
        del frame
        process = child["process"]
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    atomic_json(state_path, {
        "status": "starting", "worker_pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    with launcher_log.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            job["command"], cwd=str(REPO_ROOT), stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        )
        child["process"] = process
        atomic_json(state_path, {
            "status": "running", "worker_pid": os.getpid(),
            "runner_pid": process.pid,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        code = process.wait()
    atomic_text(exit_path, str(code) + "\n")
    atomic_json(state_path, {
        "status": "finished", "worker_pid": os.getpid(),
        "runner_pid": process.pid, "exit_code": code,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return code


def cgroup_memory_gib():
    for path in (Path("/sys/fs/cgroup/memory.current"),
                 Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            return int(path.read_text().strip()) / 1024 ** 3
        except (OSError, ValueError):
            pass
    return 0.0


def gpu_memory_mib():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL)
        return int(output.splitlines()[0].strip())
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return 0


def run_preflight(root):
    log_path = Path(root) / "ASSET_PREFLIGHT.log"
    exit_path = Path(root) / "ASSET_PREFLIGHT.exitcode"
    with log_path.open("wb") as stream:
        completed = subprocess.run(
            [sys.executable, str(PREFLIGHT), "--hash", "all"],
            cwd=str(REPO_ROOT), stdout=stream, stderr=subprocess.STDOUT,
        )
    atomic_text(exit_path, str(completed.returncode) + "\n")
    if completed.returncode != 0:
        raise AuditError("all-asset preflight failed; see {}".format(log_path))


def run_campaign(args):
    root, plan = load_plan(args.batch_id)
    jobs = validate_plan(root, plan)
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise AuditError("tracked worktree must be clean before an audit run")
    preflight_exit = root / "ASSET_PREFLIGHT.exitcode"
    if not preflight_exit.is_file() or preflight_exit.read_text().strip() != "0":
        run_preflight(root)
    running = {}
    pending = []
    failed = []
    for job in jobs:
        exit_path = Path(job["job_dir"]) / "exitcode"
        if exit_path.is_file():
            try:
                code = int(exit_path.read_text().strip())
            except ValueError:
                code = 255
            if code == 0:
                continue
            failed.append(job)
            continue
        state_path = Path(job["job_dir"]) / "worker_state.json"
        state = read_json(state_path) if state_path.is_file() else {}
        worker_pid = state.get("worker_pid")
        if state.get("status") in ("starting", "running") and process_alive(worker_pid):
            running[int(worker_pid)] = {"process": None, "job": job}
        else:
            pending.append(job)
    if failed:
        raise AuditError(
            "audit has failed jobs; use a new batch after diagnosis: {}".format(
                ", ".join(job["run_tag"] for job in failed)
            )
        )
    stop = {"requested": False}

    def request_stop(signum, frame):
        del signum, frame
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    scheduler_log = root / "scheduler.log"
    with scheduler_log.open("a", encoding="utf-8", buffering=1) as log:
        while pending or running:
            for pid, active in list(running.items()):
                exit_path = Path(active["job"]["job_dir"]) / "exitcode"
                if exit_path.is_file():
                    code = int(exit_path.read_text().strip())
                    log.write("finish {} {}\n".format(active["job"]["run_tag"], code))
                    del running[pid]
                    if code != 0:
                        stop["requested"] = True
                elif not process_alive(pid):
                    log.write("orphan {} {}\n".format(active["job"]["run_tag"], pid))
                    del running[pid]
                    stop["requested"] = True
            if stop["requested"]:
                for pid in list(running):
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                if not running:
                    break
                time.sleep(1)
                continue
            model_counts = {}
            for active in running.values():
                model = active["job"]["model"]
                model_counts[model] = model_counts.get(model, 0) + 1
            launched = False
            if (
                len(running) < args.max_workers
                and gpu_memory_mib() <= args.max_gpu_memory_mib
                and cgroup_memory_gib() <= args.max_memory_gib
            ):
                selected = next((index for index, job in enumerate(pending)
                                 if model_counts.get(job["model"], 0)
                                 < args.max_per_model), None)
                if selected is not None:
                    job = pending.pop(selected)
                    process = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()),
                         "--worker-job", str(Path(job["job_dir"]) / "job.json")],
                        cwd=str(REPO_ROOT), start_new_session=True,
                    )
                    running[process.pid] = {"process": process, "job": job}
                    log.write("launch {} {}\n".format(job["run_tag"], process.pid))
                    launched = True
                    time.sleep(args.launch_stagger)
            atomic_json(root / "progress.json", {
                "schema": "navtta.vln_tta_adapter_parity_progress.v1",
                "planned": 56,
                "pending": len(pending),
                "running": len(running),
                "completed": sum(
                    (Path(job["job_dir"]) / "exitcode").is_file()
                    and (Path(job["job_dir"]) / "exitcode").read_text().strip() == "0"
                    for job in jobs
                ),
                "running_jobs": [item["job"]["run_tag"]
                                 for item in running.values()],
            })
            if not launched:
                time.sleep(2)
    if stop["requested"]:
        raise AuditError("audit campaign stopped or a job failed")
    validate_campaign(args.batch_id)


def _finite(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def parse_metrics(console_path):
    text = Path(console_path).read_text(encoding="utf-8", errors="replace")
    values = {}
    lines = [line for line in text.splitlines() if "Env name: val_seen" in line]
    if lines:
        for key, value in re.findall(
            r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)",
            lines[-1],
        ):
            values[key.upper()] = float(value)
    else:
        for key, value in re.findall(
            r"Average episode ([A-Za-z0-9_]+):\s*(-?[0-9]+(?:\.[0-9]+)?)",
            text,
        ):
            metric = key.upper()
            number = float(value)
            values[metric] = 100.0 * number if metric in {
                "SUCCESS", "ORACLE_SUCCESS", "SPL", "NDTW", "SDTW"
            } else number
        if "SUCCESS" in values:
            values["SR"] = values["SUCCESS"]
        if "ORACLE_SUCCESS" in values:
            values["OSR"] = values["ORACLE_SUCCESS"]
    if not values or not _finite(values):
        raise AuditError("missing/non-finite metrics in {}".format(console_path))
    return values


def one_file(paths, label):
    paths = sorted(paths)
    if len(paths) != 1:
        raise AuditError("expected one {}, found {}".format(label, len(paths)))
    return paths[0]


def validate_file_metadata(metadata, hash_field, label, manifest_dir):
    if not isinstance(metadata, dict):
        raise AuditError("{} metadata is invalid".format(label))
    path = metadata.get("path")
    if not isinstance(path, str) or not path:
        raise AuditError("{} path is missing".format(label))
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = (Path(manifest_dir) / resolved).resolve()
    if (
        not resolved.is_file()
        or isinstance(metadata.get("size"), bool)
        or metadata.get("size") != resolved.stat().st_size
        or HEX_SHA256.fullmatch(str(metadata.get(hash_field, ""))) is None
        or sha256(resolved) != metadata[hash_field]
    ):
        raise AuditError("{} file identity changed".format(label))
    return resolved


def validate_prefix_manifest(job, plan):
    path = Path(job["prefix_manifest_path"])
    try:
        document = load_episode_order_manifest(
            str(path), expected_split="val_seen"
        )
    except (OSError, ValueError) as error:
        raise AuditError("invalid formal prefix manifest: {}".format(error))
    order = plan["episode_orders"][job["setting"]]
    ids = [str(item["episode_id"]) for item in document["episodes"]]
    parent = document.get("canonical_parent")
    protocol = document.get("audit_prefix")
    if (
        document.get("episode_count") != 256
        or ids != order["prefix_episode_ids"]
        or document.get("order_sha256") != order["prefix_order_sha256"]
        or document.get("benchmark") != order["benchmark"]
        or document.get("dataset", {}).get("path") != order["dataset_path"]
        or document.get("dataset", {}).get("sha256")
        != order["dataset_sha256"]
        or not isinstance(parent, dict)
        or Path(parent.get("path", "")).resolve()
        != Path(order["path"]).resolve()
        or parent.get("sha256") != order["manifest_sha256"]
        or parent.get("order_sha256") != order["full_order_sha256"]
        or not isinstance(parent.get("episode_count"), int)
        or parent["episode_count"] < 256
        or protocol != {
            "protocol": "zero_update_adapter_parity",
            "canonical_prefix": True,
            "episode_count": 256,
        }
    ):
        raise AuditError("formal prefix manifest identity mismatch")
    return document


def validate_formal_manifest(job, plan):
    manifest_path = Path(job["formal_manifest_path"])
    manifest = read_json(manifest_path, "formal run manifest")
    manifest_dir = manifest_path.parent
    run_id = "{}-{}-val_seen-{}".format(
        job["run_tag"], job["setting"], job["run_data_version"]
    )
    order = plan["episode_orders"][job["setting"]]
    expected = {
        "run_id": run_id,
        "task": "vln",
        "benchmark": order["benchmark"],
        "model": job["model"],
        "method": job["method"],
        "run_tag": job["run_tag"],
        "source_setting": "{}:val_seen:{}:{}".format(
            job["setting"], job["run_data_version"], job["method"]
        ),
        "seed": 0,
        "git_commit": plan["git_commit"],
        "config": job["config_path"],
        "status": "completed",
        "exit_code": 0,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise AuditError("formal run manifest identity mismatch")
    identity = manifest.get(IMMUTABLE_IDENTITY_SHA256_FIELD)
    if (
        HEX_SHA256.fullmatch(str(identity)) is None
        or immutable_identity_sha256(manifest) != identity
    ):
        raise AuditError("formal run immutable identity mismatch")
    if not isinstance(manifest.get("config_overrides"), list) or not manifest[
        "config_overrides"
    ]:
        raise AuditError("formal evaluated command is absent")
    hardware = manifest.get("hardware")
    required_hardware = {
        "hostname", "platform", "python", "cuda_visible_devices", "torch",
        "torch_cuda", "cuda_available", "cudnn", "gpu_name",
        "gpu_capability",
    }
    if (
        not isinstance(hardware, dict)
        or not required_hardware.issubset(hardware)
        or hardware.get("cuda_available") is not True
        or not isinstance(hardware.get("gpu_name"), str)
        or not hardware["gpu_name"]
    ):
        raise AuditError("formal hardware identity is incomplete")

    validate_file_metadata(
        manifest.get("checkpoint"), "sha256", "formal checkpoint",
        manifest_dir,
    )
    auxiliary = manifest.get("auxiliary_checkpoints")
    if not isinstance(auxiliary, list) or not auxiliary:
        raise AuditError("formal auxiliary assets are absent")
    auxiliary_by_name = {}
    for item in auxiliary:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name or name in auxiliary_by_name:
            raise AuditError("formal auxiliary asset names are invalid")
        validate_file_metadata(
            item, "sha256", "formal auxiliary {}".format(name), manifest_dir
        )
        auxiliary_by_name[name] = item
    if (
        Path(auxiliary_by_name.get("audit_job_config", {}).get("path", ""))
        .resolve() != Path(job["config_path"]).resolve()
        or auxiliary_by_name.get("audit_job_config", {}).get("sha256")
        != sha256(job["config_path"])
        or Path(auxiliary_by_name.get(
            "canonical_episode_order_parent", {}
        ).get("path", "")).resolve() != Path(order["path"]).resolve()
        or auxiliary_by_name.get("canonical_episode_order_parent", {}).get(
            "sha256"
        ) != order["manifest_sha256"]
    ):
        raise AuditError("formal audit config/order-parent binding mismatch")

    dataset = manifest.get("dataset")
    dataset_path = validate_file_metadata(
        dataset, "index_sha256", "formal dataset", manifest_dir
    )
    expected_dataset_path = resolve_repo_path(order["dataset_path"]).resolve()
    if (
        dataset_path.resolve() != expected_dataset_path
        or dataset.get("index_sha256") != order["dataset_sha256"]
        or dataset.get("stream_order_sha256") != order["prefix_order_sha256"]
        or dataset.get("stream_content_sha256") != order["dataset_sha256"]
        or dataset.get("version") != order["benchmark"]
    ):
        raise AuditError("formal dataset/prefix stream identity mismatch")

    prefix = validate_prefix_manifest(job, plan)
    pinned = manifest.get("pinned_manifests")
    if not isinstance(pinned, dict) or set(pinned) != {
        "assets", "environment", "episode_order"
    }:
        raise AuditError("formal pinned manifests are incomplete")
    for name, metadata in pinned.items():
        validate_file_metadata(
            metadata, "sha256", "formal pinned {}".format(name), manifest_dir
        )
    identities = plan.get("identity_files", {})
    if (
        pinned["assets"].get("sha256")
        != identities.get("asset_manifest", {}).get("sha256")
        or Path(pinned["assets"].get("path", "")).resolve()
        != Path(identities.get("asset_manifest", {}).get("path", "")).resolve()
        or pinned["environment"].get("sha256")
        != identities.get("environment_manifest", {}).get("sha256")
        or Path(pinned["environment"].get("path", "")).resolve()
        != Path(identities.get("environment_manifest", {}).get(
            "path", ""
        )).resolve()
    ):
        raise AuditError("formal asset/environment manifest binding mismatch")
    if (
        Path(pinned["episode_order"].get("path", "")).resolve()
        != Path(job["prefix_manifest_path"]).resolve()
        or pinned["episode_order"].get("sha256")
        != sha256(job["prefix_manifest_path"])
        or prefix["order_sha256"] != dataset["stream_order_sha256"]
    ):
        raise AuditError("formal pinned prefix mismatch")

    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AuditError("formal compact result artifacts are absent")
    artifact_names = set()
    artifact_records = []
    result = Path(job["result_root"]).resolve()
    for item in artifacts:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name or name in artifact_names:
            raise AuditError("formal result artifact names are invalid")
        artifact_names.add(name)
        path = validate_file_metadata(
            item, "sha256", "formal result {}".format(name), manifest_dir
        )
        try:
            path.resolve().relative_to(result)
        except ValueError:
            raise AuditError("formal result artifact escapes the result root")
        artifact_records.append({
            "name": name,
            "path": str(path),
            "size": item["size"],
            "sha256": item["sha256"],
        })
    required_paths = {
        str(Path(job["prefix_manifest_path"]).resolve()),
        str((result / "tta_diagnostics.json").resolve()),
    }
    if not required_paths.issubset({item["path"] for item in artifact_records}):
        raise AuditError("formal manifest omits required audit artifacts")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "immutable_identity_sha256": identity,
        "prefix_manifest_path": job["prefix_manifest_path"],
        "prefix_manifest_sha256": sha256(job["prefix_manifest_path"]),
        "result_artifacts": artifact_records,
    }


def required_metric_keys(setting):
    if setting.endswith("reverie"):
        return {"SR", "SPL", "RGS", "RGSPL"}
    if setting in CONTINUOUS:
        return {"SR", "SPL", "OSR"}
    return {"SR", "SPL", "ORACLE_SR"}


def output_evidence(job, expected_ids):
    root = Path(job["result_root"])
    if job["family"] == "continuous":
        path = one_file(root.rglob("stats_ep_*_val_seen_r0_w1.json"),
                        "continuous per-episode stats")
        document = read_json(path, "continuous per-episode stats")
        if list(map(str, document)) != expected_ids or not _finite(document):
            raise AuditError("continuous episode metrics/order mismatch")
        return {
            "artifact_path": str(path),
            "artifact_sha256": sha256(path),
            "episode_payload_sha256": hashlib.sha256(
                canonical(document).encode("utf-8")
            ).hexdigest(),
        }
    path = one_file(root.rglob("submit_val_seen*.json"),
                    "discrete trajectory artifact")
    document = read_json(path, "discrete trajectory artifact")
    if not isinstance(document, list):
        raise AuditError("discrete trajectory artifact is not a list")
    identifiers = []
    trajectories = []
    for item in document:
        identifier = item.get(
            "instr_id", item.get("episode_id", item.get("instruction_id"))
        )
        trajectory = item.get("trajectory", item.get("path"))
        if not isinstance(trajectory, list) or not trajectory:
            raise AuditError("discrete prediction has no trajectory")
        identifiers.append(str(identifier))
        trajectories.append({
            "episode_id": str(identifier),
            "trajectory": trajectory,
            "predObjId": item.get("predObjId") if job["setting"].endswith(
                "reverie"
            ) else None,
        })
    if identifiers != expected_ids or not _finite(trajectories):
        raise AuditError("discrete trajectory IDs/order mismatch")
    return {
        "artifact_path": str(path),
        "artifact_sha256": sha256(path),
        "episode_payload_sha256": hashlib.sha256(
            canonical(trajectories).encode("utf-8")
        ).hexdigest(),
    }


def validate_job(job, plan):
    exit_path = Path(job["job_dir"]) / "exitcode"
    if not exit_path.is_file() or exit_path.read_text().strip() != "0":
        raise AuditError("job did not exit successfully: {}".format(job["run_tag"]))
    result = Path(job["result_root"])
    diagnostics_path = result / "tta_diagnostics.json"
    console_path = result / "console.log"
    diagnostics = read_json(diagnostics_path, "TTA diagnostics")
    if not _finite(diagnostics):
        raise AuditError("non-finite diagnostics: {}".format(job["run_tag"]))
    if (
        diagnostics.get("episode_count") != 256
        or diagnostics.get("action_steps") != diagnostics.get(
            "trajectory_steps"
        )
        or diagnostics.get("trajectory_steps", 0) <= 0
        or HEX_SHA256.fullmatch(str(diagnostics.get("trajectory_sha256", "")))
        is None
    ):
        raise AuditError("incomplete trajectory evidence: {}".format(job["run_tag"]))
    expected_ids = plan["episode_orders"][job["setting"]][
        "prefix_episode_ids"
    ]
    evidence = output_evidence(job, expected_ids)
    metrics = parse_metrics(console_path)
    missing_metrics = required_metric_keys(job["setting"]).difference(metrics)
    if missing_metrics:
        raise AuditError("required benchmark metrics are absent: {}".format(
            ", ".join(sorted(missing_metrics))
        ))
    formal_manifest = validate_formal_manifest(job, plan)
    adapter = diagnostics.get("adapter")
    if job["kind"] == "adapter_zero_update":
        required = {
            "updates": 0,
            "slow_updates": 0,
            "relative_param_drift": 0.0,
            "audit_mode": "zero_update_adapter_parity",
            "audit_expected_episodes": 256,
            "parameter_writes_suppressed": True,
            "parameter_state_hash_match": True,
            "model_state_hash_match": True,
            "deployed_model_states_hash_match": True,
            "episodes": 256,
        }
        if not isinstance(adapter, dict) or any(
            adapter.get(key) != value for key, value in required.items()
        ):
            raise AuditError("zero-update adapter invariants failed: {}".format(
                job["run_tag"]
            ))
        attempts = adapter.get("parameter_write_attempts")
        suppressed = adapter.get("suppressed_parameter_write_attempts")
        optimizer_attempts = adapter.get("optimizer_step_attempts")
        suppressed_optimizer = adapter.get(
            "suppressed_optimizer_step_attempts"
        )
        before = str(adapter.get("parameter_state_before_sha256", ""))
        after = str(adapter.get("parameter_state_after_sha256", ""))
        model_before = str(adapter.get("model_state_before_sha256", ""))
        model_after = str(adapter.get("model_state_after_sha256", ""))
        if (
            not isinstance(attempts, int) or isinstance(attempts, bool)
            or attempts <= 0 or suppressed != attempts
            or adapter.get("action_steps", 0) <= 0
            or not isinstance(optimizer_attempts, int)
            or isinstance(optimizer_attempts, bool)
            or optimizer_attempts <= 0
            or suppressed_optimizer != optimizer_attempts
            or HEX_SHA256.fullmatch(before) is None
            or before != after
            or HEX_SHA256.fullmatch(model_before) is None
            or model_before != model_after
            or adapter.get("action_steps") != diagnostics["trajectory_steps"]
            or diagnostics.get("audit_zero_update") is not True
            or diagnostics.get("audit_control") is not False
        ):
            raise AuditError("suppression/state-hash evidence failed: {}".format(
                job["run_tag"]
            ))
        if job["method"] == "fstta":
            if (
                adapter.get("use_slow") is not True
                or adapter.get("fast_optimizer_attempts", 0) <= 0
                or adapter.get("fast_optimizer_attempts_suppressed")
                != adapter.get("fast_optimizer_attempts")
                or adapter.get("completed_slow_windows", 0) <= 0
                or adapter.get("slow_optimizer_attempts", 0) <= 0
                or adapter.get("slow_optimizer_attempts_suppressed")
                != adapter.get("slow_optimizer_attempts")
                or adapter.get("relative_slow_anchor_drift") != 0.0
                or adapter.get("relative_fast_to_slow_anchor") != 0.0
                or adapter.get("slow_anchor_state_hash_match") is not True
                or adapter.get("slow_anchor_state_before_sha256")
                != adapter.get("slow_anchor_state_after_sha256")
            ):
                raise AuditError("FSTTA fast/slow audit evidence is incomplete")
        elif job["method"] == "eam":
            if (
                adapter.get("update_attempts", 0) <= 0
                or adapter.get("replayed_steps", 0) <= 0
                or adapter.get("accepted_samples", 0) <= 0
                or adapter.get("optimizer_step_attempts", 0) <= 0
                or adapter.get("auxiliary_model_state_hash_match") is not True
                or adapter.get("auxiliary_model_state_before_sha256")
                != adapter.get("auxiliary_model_state_after_sha256")
            ):
                raise AuditError("EAM replay/update audit evidence is incomplete")
        elif job["method"] == "feedtta":
            if (
                adapter.get("feedback_episodes") != 256
                or adapter.get("policy_gradient_steps", 0) <= 0
                or adapter.get("policy_gradient_steps")
                != adapter.get("action_steps")
                or adapter.get("episode_end_optimizer_attempts", 0) <= 0
                or adapter.get("successful_feedback_episodes", 0)
                + adapter.get("failed_feedback_episodes", 0) != 256
            ):
                raise AuditError("FeedTTA feedback audit evidence is incomplete")
        elif job["method"] == "atena":
            if (
                adapter.get("query_gate_evaluations") != 256
                or adapter.get("self_prediction_evaluations") != 256
                or adapter.get("replayed_steps", 0) <= 0
                or adapter.get("auxiliary_head_constructed") is not True
                or adapter.get("queries", 0)
                + adapter.get("self_label_episodes", 0) != 256
                or adapter.get("optimizer_step_attempts", 0) <= 0
                or adapter.get("auxiliary_head_state_hash_match") is not True
                or adapter.get("auxiliary_head_state_before_sha256")
                != adapter.get("auxiliary_head_state_after_sha256")
            ):
                raise AuditError("ATENA gate/replay audit evidence is incomplete")
    else:
        if (
            diagnostics.get("method") != "source"
            or diagnostics.get("audit_control") is not True
            or diagnostics.get("audit_zero_update") is not False
        ):
            raise AuditError("Source control marker failed: {}".format(job["run_tag"]))
        if (
            not isinstance(adapter, dict)
            or adapter.get("updates") != 0
            or adapter.get("episodes") != 256
            or adapter.get("action_steps") != diagnostics["trajectory_steps"]
            or adapter.get("relative_param_drift") != 0.0
            or adapter.get("audit_expected_episodes") != 256
            or adapter.get("model_state_hash_match") is not True
            or HEX_SHA256.fullmatch(str(
                adapter.get("model_state_before_sha256", "")
            )) is None
            or adapter.get("model_state_before_sha256")
            != adapter.get("model_state_after_sha256")
        ):
            raise AuditError("Source control changed parameters")
    return {
        "run_tag": job["run_tag"],
        "ordinal": job["ordinal"],
        "kind": job["kind"],
        "setting": job["setting"],
        "method": job["method"],
        "metrics": metrics,
        "trajectory_sha256": diagnostics["trajectory_sha256"],
        "trajectory_steps": diagnostics["trajectory_steps"],
        "console_path": str(console_path),
        "console_sha256": sha256(console_path),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": sha256(diagnostics_path),
        "output_evidence": evidence,
        "formal_manifest": formal_manifest,
        "adapter": adapter,
    }


def validate_campaign(batch_id):
    root, plan = load_plan(batch_id)
    jobs = validate_plan(root, plan)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise AuditError("tracked worktree must be clean for formal validation")
    preflight_exit = root / "ASSET_PREFLIGHT.exitcode"
    preflight_log = root / "ASSET_PREFLIGHT.log"
    if (
        not preflight_exit.is_file() or preflight_exit.read_text().strip() != "0"
        or not preflight_log.is_file()
        or "VLN preflight passed" not in preflight_log.read_text(
            encoding="utf-8", errors="replace"
        )
    ):
        raise AuditError("formal all-asset preflight evidence is absent")
    results = [validate_job(job, plan) for job in jobs]
    controls = {
        (item["kind"], item["setting"]): item
        for item in results if item["kind"].startswith("source_")
    }
    comparisons = []
    for result in results:
        if result["kind"] != "adapter_zero_update":
            continue
        control_kind = (
            "source_sampled" if result["method"] == "feedtta"
            else "source_argmax"
        )
        control = controls[(control_kind, result["setting"])]
        exact_fields = (
            "metrics", "trajectory_sha256", "trajectory_steps"
        )
        if any(canonical(result[field]) != canonical(control[field])
               for field in exact_fields):
            raise AuditError("adapter/Source parity failed: {} {}".format(
                result["method"], result["setting"]
            ))
        if (
            result["output_evidence"]["episode_payload_sha256"]
            != control["output_evidence"]["episode_payload_sha256"]
        ):
            raise AuditError("per-episode output parity failed: {} {}".format(
                result["method"], result["setting"]
            ))
        comparisons.append({
            "method": result["method"],
            "setting": result["setting"],
            "adapter_run_tag": result["run_tag"],
            "control_kind": control_kind,
            "control_run_tag": control["run_tag"],
            "metrics": result["metrics"],
            "trajectory_sha256": result["trajectory_sha256"],
            "episode_payload_sha256": result["output_evidence"][
                "episode_payload_sha256"
            ],
            "suppressed_parameter_write_attempts": result["adapter"][
                "suppressed_parameter_write_attempts"
            ],
            "suppressed_optimizer_step_attempts": result["adapter"][
                "suppressed_optimizer_step_attempts"
            ],
            "parameter_state_sha256": result["adapter"][
                "parameter_state_after_sha256"
            ],
            "model_state_sha256": result["adapter"][
                "model_state_after_sha256"
            ],
            "diagnostics_sha256": result["diagnostics_sha256"],
            "console_sha256": result["console_sha256"],
            "output_artifact_sha256": result["output_evidence"][
                "artifact_sha256"
            ],
            "formal_manifest_sha256": result["formal_manifest"]["sha256"],
        })
    if len(comparisons) != 40:
        raise AuditError("formal parity summary does not contain 40 comparisons")
    summary = {
        "schema": "navtta.vln_tta_adapter_parity_results.v1",
        "namespace": "adapter_parity_audit",
        "batch_id": batch_id,
        "git_commit": plan["git_commit"],
        "search_batch_id": plan["search_batch_id"],
        "search_git_commit": plan["search_git_commit"],
        "spec_sha256": plan["spec_sha256"],
        "asset_manifest_sha256": plan["identity_files"]["asset_manifest"][
            "sha256"
        ],
        "environment_manifest_sha256": plan["identity_files"][
            "environment_manifest"
        ]["sha256"],
        "all_asset_preflight_log_sha256": sha256(preflight_log),
        "split": "val_seen",
        "canonical_prefix_episodes": 256,
        "order_seed": 0,
        "job_count": 56,
        "comparison_count": 40,
        "status": "passed",
        "job_evidence": results,
        "comparisons": comparisons,
    }
    write_immutable_json(root / "RESULTS.json", summary)
    return summary


def status(batch_id):
    root, plan = load_plan(batch_id)
    jobs = validate_plan(root, plan)
    counts = {"pending": 0, "running": 0, "succeeded": 0, "failed": 0}
    for job in jobs:
        directory = Path(job["job_dir"])
        exit_path = directory / "exitcode"
        if exit_path.is_file():
            try:
                code = int(exit_path.read_text().strip())
            except ValueError:
                code = 255
            counts["succeeded" if code == 0 else "failed"] += 1
            continue
        state_path = directory / "worker_state.json"
        state = read_json(state_path) if state_path.is_file() else {}
        if state.get("status") in ("starting", "running") and process_alive(
            state.get("worker_pid")
        ):
            counts["running"] += 1
        else:
            counts["pending"] += 1
    print(json.dumps({
        "schema": "navtta.vln_tta_adapter_parity_status.v1",
        "batch_id": batch_id, "planned": 56, **counts,
        "validated": (root / "RESULTS.json").is_file(),
    }, indent=2, sort_keys=True))


def parse_args(argv=None):
    spec = load_spec()
    defaults = spec["scheduler_defaults"]
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--batch-id", required=True)
    plan.add_argument("--search-batch-id", required=True)
    plan.add_argument("--search-root", default=str(SEARCH_ROOT))
    plan.add_argument("--gpu", type=int, default=0)
    run = subparsers.add_parser("run")
    run.add_argument("--batch-id", required=True)
    run.add_argument("--max-workers", type=int, default=defaults["max_workers"])
    run.add_argument("--max-per-model", type=int,
                     default=defaults["max_per_model"])
    run.add_argument("--max-gpu-memory-mib", type=int,
                     default=defaults["max_gpu_memory_mib_before_launch"])
    run.add_argument("--max-memory-gib", type=float,
                     default=defaults["max_cgroup_memory_gib_before_launch"])
    run.add_argument("--launch-stagger", type=float,
                     default=defaults["launch_stagger_seconds"])
    validate = subparsers.add_parser("validate")
    validate.add_argument("--batch-id", required=True)
    show = subparsers.add_parser("status")
    show.add_argument("--batch-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "run" and min(
        args.max_workers, args.max_per_model, args.max_gpu_memory_mib
    ) < 1:
        parser.error("run limits must be positive")
    if args.command == "run" and (
        args.max_memory_gib <= 0 or args.launch_stagger < 0
    ):
        parser.error("invalid memory/stagger limit")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "plan":
        root, plan = create_plan(
            args.batch_id, args.search_batch_id, gpu=args.gpu,
            search_root=args.search_root,
        )
        print("planned {} jobs at {}".format(plan["job_count"], root))
    elif args.command == "run":
        run_campaign(args)
    elif args.command == "validate":
        summary = validate_campaign(args.batch_id)
        print("adapter parity audit passed: {} comparisons".format(
            summary["comparison_count"]
        ))
    elif args.command == "status":
        status(args.batch_id)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-job":
        raise SystemExit(worker_main(sys.argv[2]))
    try:
        main()
    except AuditError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
