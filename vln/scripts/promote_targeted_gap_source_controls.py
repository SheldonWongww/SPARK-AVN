#!/usr/bin/env python3
"""Review Source evidence and prepare the launch-ready targeted-gap successor.

This utility deliberately separates three irreversible-looking decisions:

``review-ce``
    Independently re-authenticate all four native-v1.2 ETPNav/BEVBert Source
    jobs and write reviewed candidate copies plus the two promoted ledgers.
``audit-discrete``
    Re-authenticate the four discrete Source ledgers used by the campaign and
    write new audit-upgraded copies without modifying the historical ledgers.
``create-successor``
    Ask the campaign runner to validate all six Source bindings, then write a
    v2 successor and (only with explicit confirmation) mark v1 superseded.

The generated manifest/spec files must still be reviewed, ``git add``-ed, and
committed before the campaign runner will accept them for formal execution.
"""

from __future__ import print_function

import argparse
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
# The campaign runner's formal-manifest validator imports the tracked
# ``tools`` namespace.  A script launched by absolute path otherwise has only
# ``vln/scripts`` on sys.path, so make the repository import root explicit.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CE_WORKFLOW_PATH = REPO_ROOT / "vln/scripts/run_r2r_ce_v12_source_controls.py"
CAMPAIGN_RUNNER_PATH = REPO_ROOT / "vln/scripts/run_targeted_gap_campaign.py"
BASE_SPEC_PATH = REPO_ROOT / "vln/experiments/vln_targeted_gap_campaign_v1.json"
SUCCESSOR_SPEC_PATH = REPO_ROOT / "vln/experiments/vln_targeted_gap_campaign_v2.json"

PROMOTED_SCHEMA = "navtta.vln_r2r_ce_v1_2_source_controls.v1"
CANDIDATE_SCHEMA = "navtta.vln_r2r_ce_v1_2_source_controls_candidate.v1"
AUDIT_STATUS = "independently_reauthenticated"
PROMOTION_STATUS = "independently_reviewed_and_promoted"
CE_BINDINGS = {
    "val_unseen": "r2r_ce_v1_2_val_unseen",
    "val_seen": "r2r_ce_v1_2_val_seen",
}
DISCRETE_BINDINGS = (
    "reverie_val_unseen",
    "reverie_val_seen",
    "r2r_val_unseen",
    "r2r_val_seen",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class PromotionError(RuntimeError):
    pass


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise PromotionError("cannot import {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def campaign_runner():
    return _load_module(CAMPAIGN_RUNNER_PATH, "navtta_targeted_gap_runner")


def ce_workflow():
    return _load_module(CE_WORKFLOW_PATH, "navtta_ce_source_workflow")


def canonical_bytes(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def pretty_bytes(value):
    return (
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def spec_bytes(value):
    """Serialize a spec deterministically while preserving reviewed key order."""
    return (
        json.dumps(
            value, indent=2, sort_keys=False, ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def superseded_v1_bytes(original, successor_id):
    """Change only v1's two lifecycle values, preserving its hand formatting."""
    text = original.decode("utf-8")
    old = (
        '  "status": "active",\n'
        '  "supersedes": [],\n'
        '  "superseded_by": null,'
    )
    new = (
        '  "status": "superseded",\n'
        '  "supersedes": [],\n'
        '  "superseded_by": "{}",'.format(successor_id)
    )
    if text.count(old) != 1:
        raise PromotionError("v1 lifecycle block is missing or ambiguous")
    return text.replace(old, new, 1).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, label="JSON"):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise PromotionError("{} must contain an object: {}".format(label, path))
    return value


def repo_relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        raise PromotionError("path escapes repository: {}".format(path))


def repo_path(value, label, require=True):
    if not isinstance(value, str) or not value:
        raise PromotionError("{} path is missing".format(label))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise PromotionError("{} escapes repository: {}".format(label, path))
    if require and not path.is_file():
        raise PromotionError("missing {}: {}".format(label, path))
    return path


def resolve_relocated_artifact(value):
    """Resolve absolute server paths after an evidence checkout relocation."""
    if not isinstance(value, str) or not value:
        raise PromotionError("artifact path is missing")
    recorded = Path(value).expanduser()
    if recorded.is_file():
        return recorded.resolve()
    candidates = []
    parts = recorded.parts
    for anchor in ("vln", "core", "docs", "tools"):
        if anchor in parts:
            index = len(parts) - 1 - tuple(reversed(parts)).index(anchor)
            candidates.append(REPO_ROOT.joinpath(*parts[index:]).resolve())
    matches = []
    for path in candidates:
        if path.is_file() and path not in matches:
            matches.append(path)
    if len(matches) != 1:
        raise PromotionError(
            "artifact is missing or ambiguously relocated: {}".format(value)
        )
    return matches[0]


def require_tracked(path, label):
    import subprocess
    completed = subprocess.run(
        [
            "git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch",
            "--", repo_relative(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PromotionError("{} must be tracked by Git: {}".format(label, path))


def atomic_bytes(path, data, refuse_change=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return
        if refuse_change:
            raise PromotionError("refusing to overwrite different file: {}".format(path))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def valid_sha256(value):
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_review_identity(reviewer, reviewed_at):
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise PromotionError("--reviewed-by must identify the human reviewer")
    try:
        parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise PromotionError("--reviewed-at must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise PromotionError("--reviewed-at must include an explicit timezone")


def validate_plan_identity(plan):
    if (
        plan.get("schema")
        != "navtta.vln_r2r_ce_v1_2_source_control_workflow.v1"
        or plan.get("schema_version") != 1
    ):
        raise PromotionError("unsupported CE Source PLAN schema")
    recorded = plan.get("plan_identity_sha256")
    payload = dict(plan)
    payload.pop("plan_identity_sha256", None)
    if recorded != sha256_bytes(canonical_bytes(payload)):
        raise PromotionError("CE Source PLAN identity mismatch")
    jobs = plan.get("jobs")
    observed = {
        (item.get("setting"), item.get("split")) for item in jobs or ()
        if isinstance(item, dict)
    }
    expected = {
        (setting, split)
        for setting in ("etpnav-r2r-ce", "bevbert-r2r-ce")
        for split in ("val_unseen", "val_seen")
    }
    if len(jobs or ()) != 4 or observed != expected:
        raise PromotionError("CE Source PLAN is not the exact four-job matrix")
    if plan.get("protocol", {}).get("data_version") != "v1.2-native":
        raise PromotionError("CE Source PLAN is not native v1.2")


def candidate_ledgers(batch_root, plan):
    directory = Path(batch_root) / "candidate-ledgers"
    found = {}
    for path in sorted(directory.glob("*.json")):
        document = read_json(path, "candidate ledger")
        if (
            document.get("schema") == CANDIDATE_SCHEMA
            and document.get("schema_version") == 1
            and document.get("batch_id") == plan.get("batch_id")
            and document.get("plan_identity_sha256")
            == plan.get("plan_identity_sha256")
        ):
            split = document.get("split")
            if split in found:
                raise PromotionError("multiple candidate ledgers for {}".format(split))
            found[split] = (path, document)
    if set(found) != set(CE_BINDINGS):
        raise PromotionError(
            "expected one candidate ledger for each of val_unseen and val_seen"
        )
    return found


def _checked_artifact(record, label):
    if not isinstance(record, dict):
        raise PromotionError("{} record is missing".format(label))
    path = resolve_relocated_artifact(record.get("path"))
    if (
        isinstance(record.get("size"), bool)
        or not isinstance(record.get("size"), int)
        or record["size"] <= 0
        or path.stat().st_size != record["size"]
        or not valid_sha256(record.get("sha256"))
        or sha256_file(path) != record["sha256"]
    ):
        raise PromotionError("{} size/SHA256 mismatch".format(label))
    return path


def _validate_ce_aggregate_and_episodes(validation, order_document):
    aggregate_path = _checked_artifact(
        validation.get("aggregate_artifact"), "native aggregate"
    )
    episode_path = _checked_artifact(
        validation.get("per_episode_artifact"), "per-episode metrics"
    )
    aggregate = read_json(aggregate_path, "native aggregate")
    per_episode = read_json(episode_path, "per-episode metrics")
    expected_ids = [str(item.get("episode_id")) for item in order_document["episodes"]]
    if list(per_episode) != expected_ids:
        raise PromotionError("per-episode IDs/order differ from the canonical stream")
    if len(per_episode) != order_document["episode_count"]:
        raise PromotionError("per-episode result count is incomplete")
    maximum_delta = 0.0
    for metric, expected_mean in aggregate.items():
        if not finite_number(expected_mean):
            raise PromotionError("native aggregate contains a non-finite metric")
        values = []
        for episode_id in expected_ids:
            item = per_episode.get(episode_id)
            if not isinstance(item, dict) or not finite_number(item.get(metric)):
                raise PromotionError(
                    "per-episode metric {} is missing for {}".format(
                        metric, episode_id
                    )
                )
            values.append(float(item[metric]))
        observed_mean = math.fsum(values) / len(values)
        maximum_delta = max(maximum_delta, abs(observed_mean - float(expected_mean)))
    if maximum_delta > 1e-9:
        raise PromotionError(
            "native aggregate/per-episode mismatch (max delta {})".format(
                maximum_delta
            )
        )
    metrics = validation.get("metrics")
    if (
        not isinstance(metrics, dict)
        or not math.isclose(
            float(metrics.get("SR", float("nan"))),
            100.0 * float(aggregate.get("success", float("nan"))),
            rel_tol=0.0, abs_tol=1e-9,
        )
        or not math.isclose(
            float(metrics.get("SPL", float("nan"))),
            100.0 * float(aggregate.get("spl", float("nan"))),
            rel_tol=0.0, abs_tol=1e-9,
        )
    ):
        raise PromotionError("review report/native aggregate metric mismatch")
    return aggregate_path, episode_path


def review_ce_record(record, job, plan, campaign):
    split = job["split"]
    order_binding = plan["orders"][split]
    order_path = repo_path(order_binding["path"], "CE episode-order manifest")
    if sha256_file(order_path) != order_binding["sha256"]:
        raise PromotionError("CE episode-order manifest SHA256 mismatch")
    order = read_json(order_path, "CE episode-order manifest")
    if (
        order.get("schema") != "navtta.episode_order.v1"
        or order.get("split") != split
        or order.get("episode_count") != job["expected_episodes"]
        or order.get("order_sha256") != job["episode_order_sha256"]
        or order.get("dataset", {}).get("sha256") != job["dataset_sha256"]
        or not isinstance(order.get("episodes"), list)
    ):
        raise PromotionError("CE episode-order identity mismatch")

    checks = {
        "model": job["model"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "dataset_sha256": job["dataset_sha256"],
        "episode_order_sha256": job["episode_order_sha256"],
        "episode_count": job["expected_episodes"],
        "evidence_status": "ready",
        "git_commit": plan["git_commit"],
    }
    for key, expected in checks.items():
        if record.get(key) != expected:
            raise PromotionError(
                "{} {} record mismatch".format(job["key"], key)
            )
    if record.get("parameters") != {
        "action_selection": "argmax", "action_seed": 0,
    }:
        raise PromotionError("{} action protocol mismatch".format(job["key"]))

    validation_path = repo_path(
        record.get("metrics_artifact_path"), "CE validation report"
    )
    if sha256_file(validation_path) != record.get("metrics_artifact_sha256"):
        raise PromotionError("CE validation report SHA256 mismatch")
    validation = read_json(validation_path, "CE validation report")
    validation_checks = {
        "schema": "navtta.vln_r2r_ce_source_result_validation.v1",
        "setting": job["setting"],
        "model": job["model"],
        "split": split,
        "data_version": "v1.2-native",
        "seed": 0,
        "order_seed": 0,
        "action_selection": "target_native_argmax",
        "episode_count": job["expected_episodes"],
        "episode_order_sha256": job["episode_order_sha256"],
        "dataset_sha256": job["dataset_sha256"],
        "ordered_episode_ids_match": True,
        "aggregate_recomputed_from_per_episode": True,
    }
    for key, expected in validation_checks.items():
        if validation.get(key) != expected:
            raise PromotionError("CE validation {} mismatch".format(key))
    if validation.get("metrics") != record.get("metrics"):
        raise PromotionError("CE candidate/validation metrics mismatch")
    aggregate_path, episode_path = _validate_ce_aggregate_and_episodes(
        validation, order
    )
    if (
        sha256_file(aggregate_path) != record.get("aggregate_artifact_sha256")
        or sha256_file(episode_path) != record.get("per_episode_artifact_sha256")
    ):
        raise PromotionError("CE candidate native artifact digest mismatch")

    expected = {
        "setting": job["setting"],
        "split": split,
        "data_version": "v1.2-native",
        "model": job["model"],
        "benchmark": "r2r_ce_v1_2_etpnav_bevbert",
        "dataset_version": "r2r_ce_v1_2_etpnav_bevbert",
        "checkpoint_sha256": job["checkpoint_sha256"],
        "dataset_sha256": job["dataset_sha256"],
        "order_sha256": job["episode_order_sha256"],
        "order_manifest_sha256": order_binding["sha256"],
        "metric_artifact_sha256": record["metrics_artifact_sha256"],
        "metric_artifact_size": validation_path.stat().st_size,
        "metric_artifact_path": validation_path.resolve(),
        "selection_benchmark": "r2r-ce",
        "required_metrics": ("SPL", "SR"),
        "source_metrics": record["metrics"],
    }
    formal_path, identity, formal_digest = campaign._validate_formal_source_manifest(
        record.get("formal_manifest_path"),
        record.get("formal_manifest_sha256"), record, expected,
    )
    if identity != record.get("immutable_identity_sha256"):
        raise PromotionError("CE formal immutable identity mismatch")
    artifacts = read_json(formal_path, "CE formal manifest").get(
        "result_artifacts", []
    )
    artifact_digests = {item.get("sha256") for item in artifacts}
    required_digests = {
        record["metrics_artifact_sha256"],
        record["aggregate_artifact_sha256"],
        record["per_episode_artifact_sha256"],
    }
    if not required_digests.issubset(artifact_digests):
        raise PromotionError("CE formal manifest does not bind all reviewed artifacts")
    return {
        "formal_manifest_path": repo_relative(formal_path),
        "formal_manifest_sha256": formal_digest,
        "immutable_identity_sha256": identity,
        "aggregate_artifact_sha256": record["aggregate_artifact_sha256"],
        "per_episode_artifact_sha256": record["per_episode_artifact_sha256"],
    }


def build_promoted_ledger(candidate, candidate_path, candidate_digest,
                          reviewer, reviewed_at, evidence):
    document = copy.deepcopy(candidate)
    document.pop("candidate_status", None)
    document["schema"] = PROMOTED_SCHEMA
    document["promotion_status"] = PROMOTION_STATUS
    document["promotion_review"] = {
        "decision": "approved",
        "reviewed_by": reviewer,
        "reviewed_at": reviewed_at,
        "candidate_ledger_path": repo_relative(candidate_path),
        "candidate_ledger_sha256": candidate_digest,
        "review_tool_path": repo_relative(SCRIPT_PATH),
        "review_tool_sha256": sha256_file(SCRIPT_PATH),
        "checks": [
            "candidate_content_address_and_plan_identity",
            "four_formal_manifest_immutable_identities",
            "native_v1_2_dataset_checkpoint_split_seed_and_action_identity",
            "native_aggregate_recomputed_from_canonical_per_episode_stream",
            "formal_manifest_binds_validation_aggregate_and_per_episode_artifacts",
        ],
        "record_evidence": evidence,
    }
    return document


def review_ce(batch_id, reviewer, reviewed_at, batch_root_override=None):
    validate_review_identity(reviewer, reviewed_at)
    workflow = ce_workflow()
    campaign = campaign_runner()
    root = (
        Path(batch_root_override).resolve()
        if batch_root_override is not None
        else workflow.batch_root(batch_id).resolve()
    )
    plan_path = root / "PLAN.json"
    plan = read_json(plan_path, "CE Source PLAN")
    validate_plan_identity(plan)
    if plan.get("batch_id") != batch_id:
        raise PromotionError("CE Source PLAN batch mismatch")
    candidates = candidate_ledgers(root, plan)
    pending = []
    for split in ("val_unseen", "val_seen"):
        source_path, candidate = candidates[split]
        source_digest = sha256_file(source_path)
        if not source_path.name.endswith(".{}.json".format(source_digest)):
            raise PromotionError("candidate filename is not content-addressed")
        if (
            candidate.get("candidate_status")
            != "review_required_not_yet_promoted"
            or candidate.get("data_version") != "v1.2-native"
            or candidate.get("canonical_order_seed") != 0
            or candidate.get("source_protocol") != "target_native_argmax"
            or candidate.get("episode_count")
            != plan["orders"][split]["episodes"]
        ):
            raise PromotionError("{} candidate header mismatch".format(split))
        records = candidate.get("records")
        if not isinstance(records, dict) or set(records) != {
            "etpnav-r2r-ce", "bevbert-r2r-ce",
        }:
            raise PromotionError("{} candidate record coverage mismatch".format(split))
        evidence = {}
        for setting in ("etpnav-r2r-ce", "bevbert-r2r-ce"):
            job = next(
                item for item in plan["jobs"]
                if item["setting"] == setting and item["split"] == split
            )
            evidence[setting] = review_ce_record(
                records[setting], job, plan, campaign
            )
        tracked_candidate = (
            REPO_ROOT / "vln/manifests/source_candidates"
            / "r2r_ce_v1_2_{}.{}.json".format(split, source_digest)
        )
        promoted_path = (
            REPO_ROOT / "vln/manifests"
            / "r2r_ce_v1_2_{}_source_controls.json".format(split)
        )
        promoted = build_promoted_ledger(
            candidate, tracked_candidate, source_digest,
            reviewer, reviewed_at, evidence,
        )
        pending.append((source_path, tracked_candidate, promoted_path, promoted))

    # Validate everything before writing either split.
    outputs = []
    for source_path, tracked_candidate, promoted_path, promoted in pending:
        atomic_bytes(tracked_candidate, source_path.read_bytes())
        atomic_bytes(promoted_path, pretty_bytes(promoted))
        outputs.append({
            "split": promoted["split"],
            "candidate_path": repo_relative(tracked_candidate),
            "candidate_sha256": sha256_file(tracked_candidate),
            "promoted_path": repo_relative(promoted_path),
            "promoted_sha256": sha256_file(promoted_path),
        })
    return outputs


def _binding_benchmark_split(binding_key):
    if binding_key.startswith("reverie_"):
        benchmark = "reverie"
    elif binding_key.startswith("r2r_"):
        benchmark = "r2r"
    else:
        raise PromotionError("not a discrete Source binding: {}".format(binding_key))
    split = "val_unseen" if binding_key.endswith("val_unseen") else "val_seen"
    return benchmark, split


def _required_settings(spec, benchmark):
    return sorted({
        cell["setting"] for cell in spec["cells"]
        if cell["benchmark"] == benchmark
    })


def audit_discrete_record(record, setting, split, benchmark, spec, campaign):
    upgraded = copy.deepcopy(record)
    formal_path = repo_path(
        upgraded.get("formal_manifest_path"), "Source formal manifest"
    )
    require_tracked(formal_path, "Source formal manifest")
    actual_formal_digest = sha256_file(formal_path)
    if actual_formal_digest != upgraded.get("formal_manifest_sha256"):
        raise PromotionError("Source formal-manifest SHA256 mismatch")
    manifest = read_json(formal_path, "Source formal manifest")
    identity = manifest.get("immutable_identity_sha256")
    if not valid_sha256(identity):
        raise PromotionError("formal manifest has no immutable identity")

    def fill_from_formal(field, value):
        existing = upgraded.get(field)
        if existing not in (None, "") and existing != value:
            raise PromotionError(
                "historical record/formal manifest {} mismatch".format(field)
            )
        upgraded[field] = value

    fill_from_formal("model", manifest.get("model"))
    fill_from_formal("run_id", manifest.get("run_id"))
    fill_from_formal("run_tag", manifest.get("run_tag"))
    fill_from_formal("git_commit", manifest.get("git_commit"))
    fill_from_formal("immutable_identity_sha256", identity)
    fill_from_formal(
        "checkpoint_sha256", manifest.get("checkpoint", {}).get("sha256")
    )
    fill_from_formal(
        "dataset_sha256",
        manifest.get("dataset", {}).get("stream_content_sha256"),
    )
    fill_from_formal(
        "episode_order_sha256",
        manifest.get("dataset", {}).get("stream_order_sha256"),
    )
    upgraded["formal_manifest_sha256"] = actual_formal_digest
    if not COMMIT_RE.fullmatch(upgraded.get("git_commit", "")):
        raise PromotionError("formal manifest has no full Git commit")
    order = campaign.order_binding(spec, setting, split, require_dataset=False)
    fill_from_formal("episode_count", order["episode_count"])
    fill_from_formal("evidence_status", "ready")
    expected = {
        "setting": setting,
        "split": split,
        "data_version": "discrete-native",
        "model": next(
            cell["model"] for cell in spec["cells"]
            if cell["setting"] == setting
        ),
        "benchmark": order["benchmark"],
        "dataset_version": order["benchmark"],
        "checkpoint_sha256": spec["data_bindings"]["checkpoints"][setting],
        "dataset_sha256": order["dataset_sha256"],
        "order_sha256": order["order_sha256"],
        "order_manifest_sha256": order["sha256"],
    }
    for key in ("checkpoint_sha256", "dataset_sha256"):
        if upgraded.get(key) != expected[key]:
            raise PromotionError("Source record {} mismatch".format(key))
    record_order = upgraded.get("episode_order_sha256")
    if record_order != expected["order_sha256"]:
        raise PromotionError("Source record episode order mismatch")
    if upgraded.get("parameters") != {
        "action_selection": "argmax", "action_seed": 0,
    }:
        raise PromotionError("Source record action protocol mismatch")
    metrics = upgraded.get("metrics")
    needed = (
        ("RGSPL", "RGS", "SPL", "SR")
        if benchmark == "reverie" else ("SPL", "SR")
    )
    if not isinstance(metrics, dict) or any(
        not finite_number(metrics.get(name)) for name in needed
    ):
        raise PromotionError("Source record metrics are incomplete")
    metric_value = (
        upgraded.get("metrics_artifact_path")
        or upgraded.get("aggregate_artifact_path")
        or upgraded.get("metrics_json_path")
    )
    metric_digest = (
        upgraded.get("metrics_artifact_sha256")
        or upgraded.get("aggregate_artifact_sha256")
        or upgraded.get("metrics_json_sha256")
    )
    metric_path = repo_path(metric_value, "Source metric artifact")
    if not valid_sha256(metric_digest) or sha256_file(metric_path) != metric_digest:
        raise PromotionError("Source metric artifact changed")
    parsed = campaign._source_metrics_from_artifact(metric_path, benchmark, split)
    if any(
        name not in parsed
        or not math.isclose(
            float(parsed[name]), float(metrics[name]), rel_tol=0.0, abs_tol=1e-6
        )
        for name in needed
    ):
        raise PromotionError("Source metric artifact disagrees with ledger")
    expected.update({
        "metric_artifact_sha256": metric_digest,
        "metric_artifact_size": metric_path.stat().st_size,
        "metric_artifact_path": metric_path.resolve(),
        "selection_benchmark": benchmark,
        "required_metrics": needed,
        "source_metrics": metrics,
    })
    checked_path, checked_identity, checked_digest = (
        campaign._validate_formal_source_manifest(
            upgraded["formal_manifest_path"], upgraded["formal_manifest_sha256"],
            upgraded, expected,
        )
    )
    if checked_path != formal_path or checked_identity != identity:
        raise PromotionError("Source formal identity changed during review")
    upgraded["formal_manifest_sha256"] = checked_digest
    return upgraded


def build_audited_discrete_ledger(source, source_path, reviewer, reviewed_at,
                                  records):
    document = copy.deepcopy(source)
    document["records"] = records
    document["audit_status"] = AUDIT_STATUS
    document["audit_review"] = {
        "decision": "approved",
        "reviewed_by": reviewer,
        "reviewed_at": reviewed_at,
        "predecessor_ledger_path": repo_relative(source_path),
        "predecessor_ledger_sha256": sha256_file(source_path),
        "review_tool_path": repo_relative(SCRIPT_PATH),
        "review_tool_sha256": sha256_file(SCRIPT_PATH),
        "scope": "campaign_required_records_only",
        "checks": [
            "formal_manifest_content_address_and_immutable_identity",
            "checkpoint_dataset_split_order_seed_action_and_horizon_identity",
            "native_aggregate_reparsed_and_matched_to_reviewed_metrics",
        ],
    }
    return document


def audit_discrete(spec_path, reviewer, reviewed_at):
    validate_review_identity(reviewer, reviewed_at)
    campaign = campaign_runner()
    _, spec = campaign.load_spec(spec_path)
    outputs = []
    failures = []
    pending = []
    for binding_key in DISCRETE_BINDINGS:
        benchmark, split = _binding_benchmark_split(binding_key)
        binding = spec["source_control"]["bindings"][binding_key]
        source_path = repo_path(binding.get("path"), "historical Source ledger")
        if sha256_file(source_path) != binding.get("sha256"):
            failures.append("{}: historical ledger digest mismatch".format(binding_key))
            continue
        source = read_json(source_path, "historical Source ledger")
        source_records = source.get("records")
        if not isinstance(source_records, dict):
            failures.append("{}: records are missing".format(binding_key))
            continue
        upgraded_records = {}
        for setting in _required_settings(spec, benchmark):
            pair = "{}:{}".format(setting, split)
            try:
                if not isinstance(source_records.get(setting), dict):
                    raise PromotionError("historical record is missing")
                upgraded_records[setting] = audit_discrete_record(
                    source_records[setting], setting, split, benchmark,
                    spec, campaign,
                )
            except Exception as error:
                failures.append("{}: {}".format(pair, error))
        if len(upgraded_records) == len(_required_settings(spec, benchmark)):
            document = build_audited_discrete_ledger(
                source, source_path, reviewer, reviewed_at, upgraded_records
            )
            output_path = (
                REPO_ROOT / "vln/manifests/audited_source_controls"
                / (binding_key + ".json")
            )
            pending.append((binding_key, output_path, document))
    if failures:
        raise PromotionError(
            "discrete Source audit failed closed; rerun/retrieve evidence for:\n- "
            + "\n- ".join(failures)
        )
    for binding_key, output_path, document in pending:
        atomic_bytes(output_path, pretty_bytes(document))
        outputs.append({
            "binding": binding_key,
            "path": repo_relative(output_path),
            "sha256": sha256_file(output_path),
        })
    return outputs


def source_bindings_for_successor(v1):
    bindings = {}
    for key in DISCRETE_BINDINGS:
        audited = (
            REPO_ROOT / "vln/manifests/audited_source_controls" / (key + ".json")
        )
        path = audited if audited.is_file() else repo_path(
            v1["source_control"]["bindings"][key]["path"],
            "Source ledger", require=True,
        )
        bindings[key] = {
            "mode": "reuse", "path": repo_relative(path),
            "sha256": sha256_file(path),
        }
    suggested = v1["source_control"]["native_v1_2_promotion_gate"][
        "suggested_tracked_bindings"
    ]
    for split, key in CE_BINDINGS.items():
        path = repo_path(suggested[key], "promoted CE Source ledger", require=True)
        bindings[key] = {
            "mode": "reuse", "path": repo_relative(path),
            "sha256": sha256_file(path),
        }
    return bindings


def build_successor_documents(v1, bindings, reviewer, reviewed_at,
                              runner_digest, serialized_predecessor=None):
    successor_id = "vln-targeted-gap-campaign-v2"
    predecessor = copy.deepcopy(v1)
    if predecessor.get("spec_id") != "vln-targeted-gap-campaign-v1":
        raise PromotionError("unexpected predecessor spec_id")
    if predecessor.get("status") != "active" or predecessor.get(
        "superseded_by"
    ) is not None:
        raise PromotionError("v1 is not the active unsuperseded predecessor")
    predecessor["status"] = "superseded"
    predecessor["superseded_by"] = successor_id
    if serialized_predecessor is None:
        serialized_predecessor = spec_bytes(predecessor)
    else:
        try:
            serialized_document = json.loads(serialized_predecessor.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise PromotionError("invalid serialized predecessor: {}".format(error))
        if canonical_bytes(serialized_document) != canonical_bytes(predecessor):
            raise PromotionError("serialized predecessor differs from lifecycle update")
    predecessor_digest = sha256_bytes(serialized_predecessor)

    successor = copy.deepcopy(v1)
    successor.update({
        "spec_id": successor_id,
        "experiment_id": successor_id,
        "status": "active",
        "supersedes": [{
            "spec_id": "vln-targeted-gap-campaign-v1",
            "path": repo_relative(BASE_SPEC_PATH),
            "sha256": predecessor_digest,
        }],
        "superseded_by": None,
    })
    successor["source_control"]["bindings"] = copy.deepcopy(bindings)
    gate = successor["source_control"]["native_v1_2_promotion_gate"]
    gate["status"] = "promoted_authenticated_source_ledgers"
    gate["review"] = {
        "decision": "approved",
        "reviewed_by": reviewer,
        "reviewed_at": reviewed_at,
    }
    gate["promoted_bindings"] = {
        key: {"path": value["path"], "sha256": value["sha256"]}
        for key, value in bindings.items() if key.startswith("r2r_ce_v1_2_")
    }
    successor["launch_readiness"] = {
        "status": "ready",
        "blockers": [],
        "review": {
            "decision": "approved_for_formal_execution",
            "reviewed_by": reviewer,
            "reviewed_at": reviewed_at,
            "predecessor_sha256": predecessor_digest,
            "runner_sha256": runner_digest,
            "source_ledgers": {
                key: {"path": value["path"], "sha256": value["sha256"]}
                for key, value in sorted(bindings.items())
            },
        },
    }
    successor["freeze"]["artifact"] = (
        "vln/results/logs/targeted_gap/{}-seed0/FROZEN.json".format(successor_id)
    )
    provenance = copy.deepcopy(successor.get("provenance", {}))
    provenance["successor_review"] = {
        "predecessor_path": repo_relative(BASE_SPEC_PATH),
        "predecessor_sha256": predecessor_digest,
        "runner_path": repo_relative(CAMPAIGN_RUNNER_PATH),
        "runner_sha256": runner_digest,
        "reviewed_by": reviewer,
        "reviewed_at": reviewed_at,
    }
    successor["provenance"] = provenance
    return predecessor, successor


def create_successor(spec_path, output_path, reviewer, reviewed_at,
                     confirm_supersession):
    validate_review_identity(reviewer, reviewed_at)
    if not confirm_supersession:
        raise PromotionError(
            "refusing to supersede v1 without --confirm-v1-supersession"
        )
    spec_path = Path(spec_path).resolve()
    output_path = Path(output_path).resolve()
    if spec_path != BASE_SPEC_PATH.resolve():
        raise PromotionError("successor predecessor must be the canonical v1 spec")
    if output_path != SUCCESSOR_SPEC_PATH.resolve():
        raise PromotionError("successor output must be the canonical v2 path")
    require_tracked(spec_path, "v1 campaign spec")
    require_tracked(CAMPAIGN_RUNNER_PATH, "campaign runner")
    original_v1_bytes = spec_path.read_bytes()
    v1 = read_json(spec_path, "v1 campaign spec")
    bindings = source_bindings_for_successor(v1)
    for key, binding in bindings.items():
        require_tracked(
            repo_path(binding["path"], "{} Source ledger".format(key)),
            "{} Source ledger".format(key),
        )
    campaign = campaign_runner()
    runner_digest = sha256_file(CAMPAIGN_RUNNER_PATH)
    predecessor_data = superseded_v1_bytes(
        original_v1_bytes, "vln-targeted-gap-campaign-v2"
    )
    predecessor, successor = build_successor_documents(
        v1, bindings, reviewer, reviewed_at, runner_digest,
        serialized_predecessor=predecessor_data,
    )
    # This is the authoritative all-six-ledger gate.  It runs before either
    # lifecycle file is changed and fails closed on missing raw evidence.
    campaign.validate_source_controls(successor)
    if canonical_bytes(campaign._successor_scientific_projection(predecessor)) != (
        canonical_bytes(campaign._successor_scientific_projection(successor))
    ):
        raise PromotionError("successor changed the v1 scientific contract")

    successor_data = spec_bytes(successor)
    if output_path.exists() and output_path.read_bytes() != successor_data:
        raise PromotionError("refusing to overwrite a different v2 spec")
    # Write v2 first.  A crash before the v1 update leaves an invalid, hence
    # non-launchable, successor rather than a v1 pointing at a missing file.
    atomic_bytes(output_path, successor_data)
    atomic_bytes(spec_path, predecessor_data, refuse_change=False)
    return {
        "predecessor_path": repo_relative(spec_path),
        "predecessor_sha256": sha256_file(spec_path),
        "successor_path": repo_relative(output_path),
        "successor_sha256": sha256_file(output_path),
        "runner_sha256": runner_digest,
        "source_ledgers": bindings,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action")

    review = subparsers.add_parser(
        "review-ce", help="independently review and promote native-v1.2 ledgers"
    )
    review.add_argument("--batch-id", required=True)
    review.add_argument("--batch-root", type=Path)
    review.add_argument("--reviewed-by", required=True)
    review.add_argument("--reviewed-at", required=True)
    review.add_argument("--confirm-independent-review", action="store_true")

    audit = subparsers.add_parser(
        "audit-discrete", help="write audit-upgraded discrete Source ledgers"
    )
    audit.add_argument("--spec", type=Path, default=BASE_SPEC_PATH)
    audit.add_argument("--reviewed-by", required=True)
    audit.add_argument("--reviewed-at", required=True)
    audit.add_argument("--confirm-independent-review", action="store_true")

    successor = subparsers.add_parser(
        "create-successor", help="validate all Source evidence and create v2"
    )
    successor.add_argument("--spec", type=Path, default=BASE_SPEC_PATH)
    successor.add_argument("--output", type=Path, default=SUCCESSOR_SPEC_PATH)
    successor.add_argument("--reviewed-by", required=True)
    successor.add_argument("--reviewed-at", required=True)
    successor.add_argument("--confirm-v1-supersession", action="store_true")

    args = parser.parse_args(argv)
    if args.action is None:
        parser.error("one action is required")
    if args.action in ("review-ce", "audit-discrete") and not (
        args.confirm_independent_review
    ):
        parser.error("--confirm-independent-review is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.action == "review-ce":
        result = review_ce(
            args.batch_id, args.reviewed_by, args.reviewed_at, args.batch_root
        )
    elif args.action == "audit-discrete":
        result = audit_discrete(args.spec, args.reviewed_by, args.reviewed_at)
    elif args.action == "create-successor":
        result = create_successor(
            args.spec, args.output, args.reviewed_by, args.reviewed_at,
            args.confirm_v1_supersession,
        )
    else:  # pragma: no cover - argparse owns this branch.
        raise PromotionError("unsupported action")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PromotionError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
