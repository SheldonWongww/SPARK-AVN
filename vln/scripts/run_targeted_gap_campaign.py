#!/usr/bin/env python3
"""Run the frozen, seed-0 VLN targeted-gap campaign.

This launcher intentionally does not reuse the older Cartesian/three-seed
search scheduler. It executes the sixteen cells declared by the explicitly
selected active campaign spec as sixteen serial queues: 55 development jobs,
16 frozen validation jobs, and five REVERIE test submissions. Queue ``q`` is
permanently assigned to GPU slot ``q % 4``; therefore at most four model
processes can be active on one GPU.

The detailed lifecycle remains available as separate commands::

    plan -> search -> freeze -> val-seen -> reverie-test

For the common path, ``run`` executes ``search -> freeze -> val-seen`` in one
foreground command while retaining the same barriers. ``reverie-test`` stays
separate because its FeedTTA-LLM submission needs a live provider. ``status``
is read-only. A resume skips revalidated successes, retains all failed
evidence, and requires ``--retry-failed`` before making a new attempt for a
failed, invalid, or orphaned job. Administrative logs live below
``vln/results/logs/targeted_gap/BATCH`` and model outputs below
``vln/results/tuning/targeted_gap/BATCH``.
"""

import argparse
from collections import Counter, defaultdict
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
BASE_SPEC = REPO_ROOT / "vln/experiments/vln_targeted_gap_campaign_v1.json"
SUCCESSOR_SPEC = REPO_ROOT / "vln/experiments/vln_targeted_gap_campaign_v2.json"
DEFAULT_SPEC = SUCCESSOR_SPEC
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
TRANSLATOR = REPO_ROOT / "vln/scripts/tta_config_cli.py"
ENVIRONMENT_MANIFEST = (
    REPO_ROOT / "vln/manifests/environments/eval_environments.json"
)
LOG_ROOT = REPO_ROOT / "vln/results/logs/targeted_gap"
TUNING_ROOT = REPO_ROOT / "vln/results/tuning/targeted_gap"
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"
DEFAULT_BATCH_ID = "vln-targeted-gap-campaign-v2-seed0"

SPEC_SCHEMA = "navtta.vln_targeted_gap_campaign.v1"
SUCCESSOR_SPEC_SCHEMA = "navtta.vln_targeted_gap_campaign.successor.v1"
BATCH_SCHEMA = "navtta.vln_targeted_gap_batch.v1"
PLAN_SCHEMA = "navtta.vln_targeted_gap_plan.v1"
JOB_SCHEMA = "navtta.vln_targeted_gap_job.v1"
RESULT_SCHEMA = "navtta.vln_targeted_gap_result.v1"
FREEZE_SCHEMA = "navtta.vln_targeted_gap_frozen.v1"
SUMMARY_SCHEMA = "navtta.vln_targeted_gap_stage_summary.v1"
PER_EPISODE_SCHEMA = "navtta.vln_per_episode_metrics.v1"
PROCESS_SCHEMA = "navtta.vln_targeted_gap_process.v1"
RETRY_ARCHIVE_SCHEMA = "navtta.vln_targeted_gap_retry_archive.v2"
TERMINATION_SCHEMA = "navtta.vln_targeted_gap_scheduler_termination.v1"
LLM_PREFLIGHT_SCHEMA = "navtta.reverie_llm_feedback_preflight.v1"
RENDER_PREFLIGHT_SCHEMA = "navtta.reverie_mattersim_render_preflight.v1"
LLM_PREFLIGHT_ENV = "NAVTTA_REVERIE_LLM_PREFLIGHT"
RENDER_BUILD_ENV = "NAVTTA_REVERIE_RENDER_MATTERSIM_BUILD"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SPEC_ID_RE = re.compile(r"^vln-targeted-gap-campaign-v([1-9][0-9]*)$")

EXPECTED_CELLS = (
    (0, "hamt-reverie-eam", "reverie", "hamt-reverie", "hamt", "eam", "reverie_eam_v2", 0, "discrete-native", "unsupervised"),
    (1, "duet-reverie-eam", "reverie", "duet-reverie", "duet", "eam", "reverie_eam_v2", 1, "discrete-native", "unsupervised"),
    (2, "goat-reverie-eam", "reverie", "goat-reverie", "goat", "eam", "reverie_eam_v2", 2, "discrete-native", "unsupervised"),
    (3, "goat-reverie-feedtta", "reverie", "goat-reverie", "goat", "feedtta", "reverie_feedtta_v2", 3, "discrete-native", "feedback_supervised"),
    (4, "goat-reverie-idea", "reverie", "goat-reverie", "goat", "idea", "reverie_idea_v2", 0, "discrete-native", "unsupervised"),
    (5, "hamt-r2r-tent", "r2r", "hamt-r2r", "hamt", "tent", "r2r_tent_v2", 1, "discrete-native", "unsupervised"),
    (6, "hamt-r2r-fstta", "r2r", "hamt-r2r", "hamt", "fstta", "r2r_fstta_v2", 2, "discrete-native", "unsupervised"),
    (7, "hamt-r2r-feedtta", "r2r", "hamt-r2r", "hamt", "feedtta", "r2r_feedtta_v2", 3, "discrete-native", "feedback_supervised"),
    (8, "hamt-r2r-atena", "r2r", "hamt-r2r", "hamt", "atena", "r2r_atena_capped_v1", 0, "discrete-native", "feedback_supervised"),
    (9, "hamt-r2r-idea", "r2r", "hamt-r2r", "hamt", "idea", "r2r_idea_v2", 1, "discrete-native", "unsupervised"),
    (10, "goat-r2r-tent", "r2r", "goat-r2r", "goat", "tent", "r2r_tent_v2", 2, "discrete-native", "unsupervised"),
    (11, "goat-r2r-eam", "r2r", "goat-r2r", "goat", "eam", "r2r_eam_v2", 3, "discrete-native", "unsupervised"),
    (12, "goat-r2r-feedtta", "r2r", "goat-r2r", "goat", "feedtta", "r2r_feedtta_v2", 0, "discrete-native", "feedback_supervised"),
    (13, "goat-r2r-idea", "r2r", "goat-r2r", "goat", "idea", "r2r_idea_v2", 1, "discrete-native", "unsupervised"),
    (14, "etpnav-r2r-ce-eam", "r2r-ce", "etpnav-r2r-ce", "etpnav", "eam", "r2r_ce_v1_2_eam_v1", 2, "v1.2-native", "unsupervised"),
    (15, "bevbert-r2r-ce-eam", "r2r-ce", "bevbert-r2r-ce", "bevbert", "eam", "r2r_ce_v1_2_eam_v1", 3, "v1.2-native", "unsupervised"),
)
CELL_FIELDS = (
    "queue_id", "cell_id", "benchmark", "setting", "model", "method",
    "grid", "gpu_slot", "data_version", "supervision",
)

EXPECTED_TEST_SUBMISSIONS = (
    ("hamt-reverie-eam-test", "hamt-reverie-eam", "EAM", "unsupervised", "none"),
    ("duet-reverie-eam-test", "duet-reverie-eam", "EAM", "unsupervised", "none"),
    ("goat-reverie-eam-test", "goat-reverie-eam", "EAM", "unsupervised", "none"),
    ("goat-reverie-idea-test", "goat-reverie-idea", "IDEA", "unsupervised", "none"),
    ("goat-reverie-feedtta-llm-qwen2-vl-2b-v1-test", "goat-reverie-feedtta", "FeedTTA-LLM", "pseudo_label", "qwen2_vl_2b_v1"),
)
TEST_FIELDS = (
    "submission_id", "source_cell_id", "reported_method_label",
    "supervision", "feedback_provider",
)

CHECKPOINT_ASSET = {
    "duet-reverie": "duet_reverie_checkpoint",
    "hamt-reverie": "hamt_reverie_checkpoint",
    "goat-reverie": "goat_reverie_checkpoint",
    "hamt-r2r": "hamt_r2r_e2e_checkpoint",
    "goat-r2r": "goat_r2r_checkpoint",
    "etpnav-r2r-ce": "etpnav_checkpoint",
    "bevbert-r2r-ce": "bevbert_ce_checkpoint",
}

ORDER_KEY = {
    ("duet-reverie", "val_unseen"): "reverie_duet_hamt_val_unseen",
    ("duet-reverie", "val_seen"): "reverie_duet_hamt_val_seen",
    ("duet-reverie", "test"): "reverie_duet_hamt_test",
    ("hamt-reverie", "val_unseen"): "reverie_duet_hamt_val_unseen",
    ("hamt-reverie", "val_seen"): "reverie_duet_hamt_val_seen",
    ("hamt-reverie", "test"): "reverie_duet_hamt_test",
    ("goat-reverie", "val_unseen"): "reverie_goat_val_unseen",
    ("goat-reverie", "val_seen"): "reverie_goat_val_seen",
    ("goat-reverie", "test"): "reverie_goat_test",
    ("hamt-r2r", "val_unseen"): "r2r_duet_hamt_val_unseen",
    ("hamt-r2r", "val_seen"): "r2r_duet_hamt_val_seen",
    ("goat-r2r", "val_unseen"): "r2r_goat_val_unseen",
    ("goat-r2r", "val_seen"): "r2r_goat_val_seen",
    ("etpnav-r2r-ce", "val_unseen"): "r2r_ce_v1_2_val_unseen",
    ("etpnav-r2r-ce", "val_seen"): "r2r_ce_v1_2_val_seen",
    ("bevbert-r2r-ce", "val_unseen"): "r2r_ce_v1_2_val_unseen",
    ("bevbert-r2r-ce", "val_seen"): "r2r_ce_v1_2_val_seen",
}

SOURCE_BINDING = {
    ("reverie", "val_unseen"): "reverie_val_unseen",
    ("reverie", "val_seen"): "reverie_val_seen",
    ("r2r", "val_unseen"): "r2r_val_unseen",
    ("r2r", "val_seen"): "r2r_val_seen",
    ("r2r-ce", "val_unseen"): "r2r_ce_v1_2_val_unseen",
    ("r2r-ce", "val_seen"): "r2r_ce_v1_2_val_seen",
}

SOURCE_LEDGER_SCHEMA = {
    "reverie_val_unseen": "navtta.vln_reverie_val_unseen_reused_source_controls.v1",
    "reverie_val_seen": "navtta.vln_reverie_reused_source_controls.v1",
    "r2r_val_unseen": "navtta.vln_r2r_val_unseen_reused_source_controls.v1",
    "r2r_val_seen": "navtta.vln_r2r_reused_source_controls.v1",
    "r2r_ce_v1_2_val_unseen": "navtta.vln_r2r_ce_v1_2_source_controls.v1",
    "r2r_ce_v1_2_val_seen": "navtta.vln_r2r_ce_v1_2_source_controls.v1",
}

REQUIRED_METRICS = {
    "reverie": ("RGSPL", "RGS", "SPL", "SR"),
    "r2r": ("SPL", "SR", "NDTW", "SDTW"),
    "r2r-ce": ("SPL", "SR", "NDTW", "SDTW", "COLLISIONS"),
}
PER_EPISODE_METRIC_KEY = {
    "RGSPL": "rgspl",
    "RGS": "rgs",
    "SPL": "spl",
    "SR": "success",
    "NDTW": "ndtw",
    "SDTW": "sdtw",
    "COLLISIONS": "collisions",
}

# A successor may clear operational blockers and bind independently promoted
# native-v1.2 Source ledgers.  Every scientific choice outside this narrow
# allowlist must remain byte-equivalent to v1.
SUCCESSOR_LIFECYCLE_FIELDS = {
    "spec_id", "experiment_id", "status", "supersedes", "superseded_by",
    "launch_readiness", "provenance",
}

_ACTIVE_PROCESSES = {}
_ACTIVE_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
_VERIFIED_LARGE_FILES = set()


class CampaignError(RuntimeError):
    """A fail-closed campaign protocol or evidence error."""


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError("cannot read JSON {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise CampaignError("JSON document must be an object: {}".format(path))
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value):
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def repo_file(value, label, require=False):
    if not isinstance(value, str) or not value:
        raise CampaignError("{} path is missing".format(label))
    path = Path(value)
    path = path if path.is_absolute() else REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise CampaignError("{} escapes repository: {}".format(label, path))
    if require and not path.is_file():
        raise CampaignError("missing {}: {}".format(label, path))
    return path


def runtime_file(value, label, require=False):
    """Resolve a manifest-bound runtime asset, including external storage.

    Specs, scripts, and tracked manifests use :func:`repo_file`. Large
    datasets and checkpoints are intentionally stored outside Git (often via
    repository symlinks), so containment is inappropriate for those files;
    their callers authenticate the resolved bytes with the manifest SHA256.
    """
    if not isinstance(value, str) or not value:
        raise CampaignError("{} path is missing".format(label))
    path = Path(value).expanduser()
    path = path if path.is_absolute() else REPO_ROOT / path
    path = path.resolve()
    if require and not path.is_file():
        raise CampaignError("missing {}: {}".format(label, path))
    return path


def git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def exact_mapping(record, fields):
    return tuple(record.get(field) for field in fields)


def _validate_reference(binding, label):
    if not isinstance(binding, dict):
        raise CampaignError("{} binding must be an object".format(label))
    path = repo_file(binding.get("path"), label, require=True)
    digest = binding.get("sha256")
    if not valid_sha256(digest) or sha256(path) != digest:
        raise CampaignError("{} SHA256 mismatch".format(label))
    return path


def _successor_scientific_projection(document):
    """Return the v1 scientific contract with only reviewed transitions masked.

    Lifecycle metadata, the content-addressed native-v1.2 Source promotion,
    and the derived freeze path are the only fields a launch-ready successor
    may change.  This makes ``supersedes`` a verifiable relationship instead
    of a free-form label.
    """
    value = json.loads(json.dumps(document))
    for key in SUCCESSOR_LIFECYCLE_FIELDS:
        value.pop(key, None)
    source = value.get("source_control", {})
    bindings = source.get("bindings", {})
    for key in SOURCE_LEDGER_SCHEMA:
        if key in bindings:
            bindings[key] = "<authenticated-matched-source-binding>"
    gate = source.get("native_v1_2_promotion_gate")
    if isinstance(gate, dict):
        gate.pop("status", None)
        gate.pop("review", None)
        gate.pop("promoted_bindings", None)
    freeze = value.get("freeze")
    if isinstance(freeze, dict):
        freeze["artifact"] = "<derived-from-successor-spec-id>"
    return value


def _validate_successor_contract(spec_path, spec):
    """Authenticate a launch-ready successor against the tracked v1 file."""
    require_tracked(spec_path, "campaign successor spec")
    predecessor_path = BASE_SPEC.resolve()
    require_tracked(predecessor_path, "campaign predecessor spec")
    predecessor = read_json(predecessor_path)
    predecessor_digest = sha256(predecessor_path)

    declarations = spec.get("supersedes")
    if not isinstance(declarations, list) or len(declarations) != 1:
        raise CampaignError("successor spec must bind exactly one predecessor")
    declaration = declarations[0]
    if not isinstance(declaration, dict):
        raise CampaignError("successor supersedes entry must be content-addressed")
    declared_path = repo_file(
        declaration.get("path"), "superseded campaign spec", require=True
    )
    expected_declaration = {
        "spec_id": "vln-targeted-gap-campaign-v1",
        "path": str(BASE_SPEC.relative_to(REPO_ROOT)),
        "sha256": predecessor_digest,
    }
    if declared_path != predecessor_path or declaration != expected_declaration:
        raise CampaignError("successor predecessor path/SHA256 binding mismatch")
    if (
        predecessor.get("spec_id") != "vln-targeted-gap-campaign-v1"
        or predecessor.get("status") != "superseded"
        or predecessor.get("superseded_by") != spec.get("spec_id")
    ):
        raise CampaignError("tracked v1 lifecycle does not point to this successor")
    if canonical(_successor_scientific_projection(predecessor)) != canonical(
        _successor_scientific_projection(spec)
    ):
        raise CampaignError("successor changed the reviewed v1 scientific contract")

    expected_freeze = (
        "vln/results/logs/targeted_gap/{}-seed0/FROZEN.json"
        .format(spec["spec_id"])
    )
    if spec.get("freeze", {}).get("artifact") != expected_freeze:
        raise CampaignError("successor freeze artifact is not derived from spec_id")
    readiness = spec.get("launch_readiness", {})
    review = readiness.get("review")
    if not isinstance(review, dict):
        raise CampaignError("ready successor lacks an implementation review record")
    expected_review = {
        "decision": "approved_for_formal_execution",
        "predecessor_sha256": predecessor_digest,
        "runner_sha256": sha256(SCRIPT_PATH),
    }
    for key, expected in expected_review.items():
        if review.get(key) != expected:
            raise CampaignError("successor review.{} mismatch".format(key))
    for key in ("reviewed_by", "reviewed_at"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            raise CampaignError("successor review.{} is missing".format(key))

    suggested = predecessor["source_control"][
        "native_v1_2_promotion_gate"
    ]["suggested_tracked_bindings"]
    bindings = spec.get("source_control", {}).get("bindings", {})
    if set(bindings) != set(SOURCE_LEDGER_SCHEMA):
        raise CampaignError("successor Source binding set changed")
    for key, binding in bindings.items():
        expected_path = suggested.get(key)
        if (
            not isinstance(binding, dict)
            or binding.get("mode") != "reuse"
            or not valid_sha256(binding.get("sha256"))
        ):
            raise CampaignError("{} is not a promoted Source binding".format(key))
        if expected_path is not None and binding.get("path") != expected_path:
            raise CampaignError("{} promoted Source path changed".format(key))
        path = _validate_reference(binding, "{} promoted Source ledger".format(key))
        require_tracked(path, "{} promoted Source ledger".format(key))
    gate = spec.get("source_control", {}).get("native_v1_2_promotion_gate", {})
    if gate.get("status") != "promoted_authenticated_source_ledgers":
        raise CampaignError("native-v1.2 Source promotion gate is not complete")


def _expand_simple_successor(spec_path, successor):
    """Resolve the small v2 execution overlay against the frozen v1 matrix.

    The user-facing campaign only needs the reviewed 16-cell matrix and its
    search/evaluation barriers. Existing Source runs are useful for reporting,
    but their ledger packaging must not prevent TTA jobs from starting.
    Keeping this as a narrow overlay avoids duplicating the 55-candidate matrix
    while making the one intentional protocol change explicit.
    """
    allowed_keys = {
        "schema", "schema_version", "spec_id", "experiment_id", "status",
        "supersedes", "superseded_by", "base_spec", "launch_readiness",
        "execution_policy", "freeze_artifact", "provenance",
    }
    if set(successor) != allowed_keys:
        raise CampaignError("simple successor contains unsupported fields")
    if (
        successor.get("schema") != SUCCESSOR_SPEC_SCHEMA
        or successor.get("schema_version") != 1
        or successor.get("spec_id") != "vln-targeted-gap-campaign-v2"
        or successor.get("experiment_id") != successor.get("spec_id")
        or successor.get("status") != "active"
        or successor.get("superseded_by") is not None
    ):
        raise CampaignError("invalid simple successor lifecycle")

    base_binding = successor.get("base_spec")
    base_path = _validate_reference(base_binding, "simple successor base spec")
    if base_path != BASE_SPEC.resolve():
        raise CampaignError("simple successor must extend the tracked v1 matrix")
    require_tracked(base_path, "simple successor base spec")
    expected_predecessor = {
        "spec_id": "vln-targeted-gap-campaign-v1",
        "path": str(BASE_SPEC.relative_to(REPO_ROOT)),
        "sha256": sha256(base_path),
    }
    if successor.get("supersedes") != [expected_predecessor]:
        raise CampaignError("simple successor predecessor binding mismatch")
    base = read_json(base_path)
    if (
        base.get("schema") != SPEC_SCHEMA
        or base.get("schema_version") != 1
        or base.get("spec_id") != "vln-targeted-gap-campaign-v1"
        or base.get("status") != "superseded"
        or base.get("superseded_by") != successor["spec_id"]
    ):
        raise CampaignError("v1 lifecycle does not point to the simple successor")

    readiness = successor.get("launch_readiness")
    review = readiness.get("review") if isinstance(readiness, dict) else None
    if (
        not isinstance(readiness, dict)
        or readiness.get("status") != "ready"
        or readiness.get("blockers") != []
        or not isinstance(review, dict)
        or review.get("decision") != "approved_for_direct_tta_search"
        or any(
            not isinstance(review.get(key), str) or not review[key].strip()
            for key in ("reviewed_by", "reviewed_at", "reason")
        )
    ):
        raise CampaignError("simple successor lacks its execution review")
    expected_policy = {
        "source_controls": "reporting_only_not_launch_gate",
        "development": "run_all_55_val_unseen_candidates",
        "freeze": (
            "select_one_winner_for_each_of_16_cells_without_source_ledger_dependency"
        ),
        "evaluation": "run_all_16_frozen_winners_on_val_seen",
        "hidden_test": "run_the_5_reverie_submission_jobs_separately",
    }
    if successor.get("execution_policy") != expected_policy:
        raise CampaignError("simple successor execution policy changed")
    expected_freeze = (
        "vln/results/logs/targeted_gap/{}-seed0/FROZEN.json"
        .format(successor["spec_id"])
    )
    if successor.get("freeze_artifact") != expected_freeze:
        raise CampaignError("simple successor freeze artifact changed")

    resolved = json.loads(json.dumps(base))
    for key in (
        "spec_id", "experiment_id", "status", "supersedes",
        "superseded_by", "launch_readiness", "provenance",
    ):
        resolved[key] = successor[key]
    resolved["execution_policy"] = successor["execution_policy"]
    resolved["source_control"]["execution_gate"] = False
    resolved["source_control"]["role"] = (
        "existing_source_results_are_posthoc_reporting_references_only"
    )
    resolved["source_control"]["native_v1_2_promotion_gate"] = {
        "status": "not_required_for_tta_execution",
        "reason": (
            "Source results do not select hyperparameters and are compared "
            "after the TTA campaign."
        ),
    }
    resolved["selection"]["constraints"] = [
        item for item in resolved["selection"]["constraints"]
        if item != "matched_source_available"
    ]
    resolved["selection"]["source_comparison"] = (
        "posthoc_only_using_the_users_existing_source_results"
    )
    resolved["freeze"]["artifact"] = successor["freeze_artifact"]
    resolved["freeze"]["required_bindings"] = [
        item for item in resolved["freeze"]["required_bindings"]
        if item != "all_matched_source_manifest_sha256_values"
    ]
    return resolved


def load_spec(path=DEFAULT_SPEC):
    """Load and validate the complete static campaign contract."""
    path = repo_file(str(path), "campaign spec", require=True)
    spec = read_json(path)
    if spec.get("schema") == SUCCESSOR_SPEC_SCHEMA:
        spec = _expand_simple_successor(path, spec)
    if spec.get("schema") != SPEC_SCHEMA or spec.get("schema_version") != 1:
        raise CampaignError("unsupported targeted-gap campaign schema")
    spec_id = spec.get("spec_id")
    match = SPEC_ID_RE.fullmatch(spec_id) if isinstance(spec_id, str) else None
    if match is None:
        raise CampaignError("unexpected targeted-gap spec_id")
    scope = spec.get("scope", {})
    expected_scope = {
        "task": "vln", "matrix_type": "explicit_cells_not_cartesian",
        "cell_count": 16, "development_candidate_jobs": 55,
        "frozen_validation_jobs": 16, "hidden_test_transfer_jobs": 5,
        "source_jobs_in_cell_queues": 0,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            raise CampaignError("scope.{} must be {!r}".format(key, expected))
    if "streamvln" not in set(scope.get("excluded", ())):
        raise CampaignError("StreamVLN must remain excluded")

    cells = spec.get("cells")
    if not isinstance(cells, list):
        raise CampaignError("cells must be a list")
    observed_cells = tuple(exact_mapping(item, CELL_FIELDS) for item in cells)
    if observed_cells != EXPECTED_CELLS:
        raise CampaignError("the exact ordered 16-cell matrix changed")
    schedule = spec.get("schedule", {})
    expected_queues = {
        str(slot): list(range(slot, 16, 4)) for slot in range(4)
    }
    if (
        schedule.get("gpu_ids") != [0, 1, 2, 3]
        or schedule.get("queue_assignment")
        != "gpu_slot_equals_queue_id_modulo_4"
        or schedule.get("queues_per_gpu") != 4
        or schedule.get("max_active_jobs_per_cell") != 1
        or schedule.get("max_active_processes_per_gpu") != 4
        or schedule.get("work_stealing") is not False
        or schedule.get("gpu_queues") != expected_queues
        or schedule.get("development_jobs_by_gpu")
        != {"0": 15, "1": 15, "2": 13, "3": 12}
    ):
        raise CampaignError("the immutable four-GPU queue schedule changed")

    phase_order = spec.get("protocol", {}).get("phase_order", [])
    phases = [item.get("phase") for item in phase_order]
    if phases != [
        "development_search", "campaign_freeze", "frozen_evaluation",
        "optional_hidden_test_transfer",
    ]:
        raise CampaignError("campaign phase order changed")
    for item in (phase_order[0], phase_order[2], phase_order[3]):
        if item.get("model_seed") != 0 or item.get("episode_order_seed") != 0:
            raise CampaignError("all executable phases require model/order seed 0")
    if [phase_order[index].get("split") for index in (0, 2, 3)] != [
        "val_unseen", "val_seen", "test",
    ]:
        raise CampaignError("campaign split roles changed")

    grids = spec.get("candidate_grids")
    if not isinstance(grids, dict):
        raise CampaignError("candidate_grids must be an object")
    count = 0
    for cell in cells:
        grid = grids.get(cell["grid"])
        if not isinstance(grid, dict) or grid.get("method") != cell["method"]:
            raise CampaignError("invalid grid for {}".format(cell["cell_id"]))
        candidates = grid.get("candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != grid.get("candidate_count")
            or not 1 <= len(candidates) <= 5
        ):
            raise CampaignError("invalid candidate count for {}".format(cell["grid"]))
        ids = [item.get("id") for item in candidates]
        if (
            len(set(ids)) != len(ids)
            or any(not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value)
                   for value in ids)
        ):
            raise CampaignError("invalid candidate IDs for {}".format(cell["grid"]))
        for candidate in candidates:
            if not isinstance(candidate.get("parameters"), dict):
                raise CampaignError("candidate parameters must be objects")
        count += len(candidates)
        source_spec = grid.get("source_spec")
        _validate_reference(source_spec, "{} source spec".format(cell["grid"]))
    if count != 55 or spec.get("selection", {}).get("candidate_count") != 55:
        raise CampaignError("campaign must expand to exactly 55 candidates")

    data_bindings = spec.get("data_bindings", {})
    _validate_reference(data_bindings.get("asset_manifest"), "asset manifest")
    streams = data_bindings.get("streams", {})
    for key in set(ORDER_KEY.values()):
        binding = streams.get(key)
        order_path = _validate_reference(binding, "{} order manifest".format(key))
        order = read_json(order_path)
        if (
            order.get("schema") != "navtta.episode_order.v1"
            or order.get("split") not in ("val_seen", "val_unseen", "test")
            or order.get("order_seed") not in (None, 0)
            or order.get("episode_count") != binding.get("episodes")
            or order.get("order_sha256") != binding.get("order_sha256")
            or order.get("dataset", {}).get("sha256") != binding.get("dataset_sha256")
        ):
            raise CampaignError("{} order identity mismatch".format(key))
        if key.startswith("r2r_ce_v1_2") and binding.get("data_version") != "v1.2-native":
            raise CampaignError("R2R-CE stream must be v1.2-native")

    test = spec.get("hidden_test_transfer", {})
    submissions = test.get("submission_cells")
    observed_test = tuple(exact_mapping(item, TEST_FIELDS) for item in submissions or ())
    if (
        test.get("submission_job_count") != 5
        or observed_test != EXPECTED_TEST_SUBMISSIONS
    ):
        raise CampaignError("the exact five REVERIE submission jobs changed")
    expected_runtime_preflight = {
        "required_for_submission_id": (
            "goat-reverie-feedtta-llm-qwen2-vl-2b-v1-test"
        ),
        "producer": "vln/scripts/verify_reverie_llm_feedback_preflight.sh",
        "schema": LLM_PREFLIGHT_SCHEMA,
        "render_evidence_schema": RENDER_PREFLIGHT_SCHEMA,
        "path_env": LLM_PREFLIGHT_ENV,
        "render_build_env": RENDER_BUILD_ENV,
        "same_git_commit_required": True,
        "required_artifacts": [
            "model_verify", "render_evidence", "rendered_panorama",
            "provider_smoke", "provider_transcript",
        ],
        "manifest_binding": "feedback_provider_preflight",
    }
    if test.get("runtime_preflight") != expected_runtime_preflight:
        raise CampaignError("REVERIE hidden-test runtime preflight contract changed")
    return path, spec


def require_launch_ready(spec_path, spec):
    """Permit execution only from the active reviewed successor."""
    spec_path = Path(spec_path).resolve()
    loaded_path, loaded = load_spec(spec_path)
    if loaded_path != spec_path or canonical(loaded) != canonical(spec):
        raise CampaignError("in-memory campaign spec differs from spec_path")
    spec_id = spec.get("spec_id")
    match = SPEC_ID_RE.fullmatch(spec_id) if isinstance(spec_id, str) else None
    revision = int(match.group(1)) if match else 0
    readiness = spec.get("launch_readiness")
    if not isinstance(readiness, dict) or readiness.get("status") != "ready":
        raise CampaignError("campaign execution requires launch_readiness.status=ready")
    if spec.get("status") != "active":
        raise CampaignError("campaign execution requires status=active")
    if spec.get("experiment_id") != spec_id:
        raise CampaignError("active successor experiment_id must equal spec_id")
    if spec.get("superseded_by") is not None:
        raise CampaignError("superseded campaign specs cannot execute")
    blockers = readiness.get("blockers")
    if blockers not in (None, []):
        raise CampaignError("ready successor must have no launch blockers")
    if revision < 2:
        raise CampaignError("campaign execution requires the active v2 spec")
    require_tracked(spec_path, "campaign successor spec")
    raw = read_json(spec_path)
    if raw.get("schema") == SUCCESSOR_SPEC_SCHEMA:
        base_path = _validate_reference(
            raw.get("base_spec"), "simple successor base spec"
        )
        require_tracked(base_path, "simple successor base spec")
    else:
        _validate_successor_contract(spec_path, spec)


def direct_execution_mode(spec):
    return (
        spec.get("execution_policy", {}).get("source_controls")
        == "reporting_only_not_launch_gate"
    )


def combine_parameters(spec, cell, candidate):
    grid = spec["candidate_grids"][cell["grid"]]
    base = grid.get("base", {})
    overrides = candidate.get("parameters", {})
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        raise CampaignError("candidate base/parameters must be objects")
    conflicting = [key for key in set(base).intersection(overrides)
                   if base[key] != overrides[key]]
    if conflicting:
        raise CampaignError("candidate changes fixed base keys: {}".format(
            ", ".join(sorted(conflicting))))
    parameters = dict(base)
    parameters.update(overrides)
    # Seed zero is a derived method RNG, not an episode-order override.  The
    # command deliberately omits --order-seed so run_source_eval uses the
    # native canonical manifest and keeps model seed 0.
    if cell["method"] == "feedtta":
        parameters["sgr_seed"] = 0
    if cell["method"] == "idea":
        binding = spec["data_bindings"]["idea_source_statistics"].get(
            cell["setting"]
        )
        if not isinstance(binding, dict):
            raise CampaignError("missing IDEA Source statistics for {}".format(
                cell["setting"]))
        parameters.update({
            "source_stats_path": str(runtime_file(
                binding["path"], "IDEA source statistics", require=False
            )),
            "source_stats_sha256": binding["sha256"],
            "source_trajectories": binding.get("trajectory_count"),
        })
    # These are explanatory spec fields rather than accepted runtime flags.
    for key in ("trainable_scope", "policy_parameters_frozen"):
        parameters.pop(key, None)
    return parameters


def _cells_by_id(spec):
    return {cell["cell_id"]: cell for cell in spec["cells"]}


def expand_search_jobs(spec, gpus):
    jobs = []
    ordinal = 0
    for cell in spec["cells"]:
        grid = spec["candidate_grids"][cell["grid"]]
        for candidate in grid["candidates"]:
            jobs.append({
                "ordinal": ordinal,
                "stage": "search",
                "split": "val_unseen",
                "queue_id": cell["queue_id"],
                "gpu_slot": cell["gpu_slot"],
                "gpu": gpus[cell["gpu_slot"]],
                "cell_id": cell["cell_id"],
                "benchmark": cell["benchmark"],
                "setting": cell["setting"],
                "model": cell["model"],
                "method": cell["method"],
                "reported_method_label": cell["method"],
                "supervision": cell["supervision"],
                "feedback_provider": "task_evaluator" if cell["supervision"] == "feedback_supervised" else "none",
                "data_version": cell["data_version"],
                "candidate_id": candidate["id"],
                "candidate_role": candidate.get("role", "search"),
                "parameters": combine_parameters(spec, cell, candidate),
            })
            ordinal += 1
    if len(jobs) != 55:
        raise CampaignError("search expansion did not produce 55 jobs")
    return jobs


def expand_frozen_jobs(spec, gpus, frozen, stage, batch_id=DEFAULT_BATCH_ID):
    winners = {
        item["cell_id"]: item for item in frozen.get("winners", [])
        if isinstance(item, dict)
    }
    if set(winners) != {cell["cell_id"] for cell in spec["cells"]}:
        raise CampaignError("FROZEN.json does not contain all sixteen winners")
    if stage == "val-seen":
        selected = [
            {
                "submission_id": None,
                "source_cell_id": cell["cell_id"],
                "reported_method_label": cell["method"],
                "supervision": cell["supervision"],
                "feedback_provider": "task_evaluator" if cell["supervision"] == "feedback_supervised" else "none",
            }
            for cell in spec["cells"]
        ]
        split = "val_seen"
    elif stage == "reverie-test":
        selected = spec["hidden_test_transfer"]["submission_cells"]
        split = "test"
    else:
        raise CampaignError("invalid frozen job stage")
    cells = _cells_by_id(spec)
    jobs = []
    for ordinal, transfer in enumerate(selected):
        cell = cells[transfer["source_cell_id"]]
        winner = winners[cell["cell_id"]]
        jobs.append({
            "ordinal": ordinal,
            "stage": stage,
            "split": split,
            "queue_id": cell["queue_id"],
            "gpu_slot": cell["gpu_slot"],
            "gpu": gpus[cell["gpu_slot"]],
            "cell_id": cell["cell_id"],
            "benchmark": cell["benchmark"],
            "setting": cell["setting"],
            "model": cell["model"],
            "method": cell["method"],
            "reported_method_label": transfer["reported_method_label"],
            "supervision": transfer["supervision"],
            "feedback_provider": transfer["feedback_provider"],
            "data_version": cell["data_version"],
            "candidate_id": winner["candidate_id"],
            "candidate_role": winner.get("candidate_role", "frozen_winner"),
            "parameters": winner["parameters"],
            "frozen_config_sha256": winner["frozen_config_sha256"],
            "submission_id": transfer.get("submission_id"),
        })
    expected = 16 if stage == "val-seen" else 5
    if len(jobs) != expected:
        raise CampaignError("{} expansion did not produce {} jobs".format(stage, expected))
    return jobs


def bind_provider_preflight(jobs, binding):
    """Bind one authenticated PRECHECK to the sole FeedTTA-LLM job."""
    if not isinstance(binding, dict):
        raise CampaignError("FeedTTA-LLM provider preflight binding is missing")
    output = []
    bound = 0
    for job in jobs:
        item = dict(job)
        if item.get("feedback_provider") == "qwen2_vl_2b_v1":
            item["feedback_provider_preflight"] = dict(binding)
            bound += 1
        elif "feedback_provider_preflight" in item:
            raise CampaignError("non-LLM job contains provider preflight evidence")
        output.append(item)
    if bound != 1:
        raise CampaignError("reverie-test must bind exactly one FeedTTA-LLM job")
    return output


def plan_payload(spec_path, spec, batch_id, gpus):
    search = expand_search_jobs(spec, gpus)
    return {
        "schema": PLAN_SCHEMA,
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "git_commit": git_commit(),
        "spec_path": str(spec_path),
        "spec_sha256": sha256(spec_path),
        "runner_path": str(SCRIPT_PATH),
        "runner_sha256": sha256(SCRIPT_PATH),
        "gpus": list(gpus),
        "queue_count": 16,
        "queues_per_gpu": 4,
        "development_job_count": 55,
        "frozen_validation_job_count": 16,
        "reverie_test_job_count": 5,
        "jobs": search,
    }


def batch_root(batch_id):
    return LOG_ROOT / batch_id


def tuning_batch_root(batch_id):
    return TUNING_ROOT / batch_id


def batch_binding(spec_path, spec, batch_id, gpus, plan):
    return {
        "schema": BATCH_SCHEMA,
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "git_commit": git_commit(),
        "spec_path": str(spec_path),
        "spec_sha256": sha256(spec_path),
        "runner_path": str(SCRIPT_PATH),
        "runner_sha256": sha256(SCRIPT_PATH),
        "plan_sha256": hashlib.sha256(
            (json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False,
                        allow_nan=False) + "\n").encode("utf-8")
        ).hexdigest(),
        "gpus": list(gpus),
        "model_seed": 0,
        "episode_order_seed": 0,
        "created_at": utc_now(),
    }


def prepare_batch(spec_path, spec, batch_id, gpus, resume):
    root = batch_root(batch_id)
    binding_path = root / "BATCH.json"
    plan_path = root / "PLAN.json"
    plan = plan_payload(spec_path, spec, batch_id, gpus)
    if binding_path.exists():
        if not resume:
            raise CampaignError("batch already exists; use --resume")
        recorded = read_json(binding_path)
        immutable = {
            key: value for key, value in batch_binding(
                spec_path, spec, batch_id, gpus, plan
            ).items() if key != "created_at"
        }
        actual = {key: recorded.get(key) for key in immutable}
        if canonical(actual) != canonical(immutable):
            raise CampaignError("BATCH.json differs from current commit/spec/runner/GPU mapping")
        if not plan_path.is_file() or canonical(read_json(plan_path)) != canonical(plan):
            raise CampaignError("PLAN.json differs from the immutable batch plan")
    else:
        if resume:
            raise CampaignError("cannot resume a batch that does not exist")
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(plan_path, plan)
        binding = batch_binding(spec_path, spec, batch_id, gpus, plan)
        # Bind the exact bytes written, rather than relying on encoder parity.
        binding["plan_sha256"] = sha256(plan_path)
        atomic_json(binding_path, binding)
    return root, read_json(binding_path), read_json(plan_path)


def validate_batch(spec_path, spec, batch_id, gpus):
    root = batch_root(batch_id)
    binding_path = root / "BATCH.json"
    plan_path = root / "PLAN.json"
    if not binding_path.is_file() or not plan_path.is_file():
        raise CampaignError("missing BATCH.json/PLAN.json; run search first")
    binding = read_json(binding_path)
    plan = read_json(plan_path)
    expected = {
        "schema": BATCH_SCHEMA,
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "git_commit": git_commit(),
        "spec_path": str(spec_path),
        "spec_sha256": sha256(spec_path),
        "runner_path": str(SCRIPT_PATH),
        "runner_sha256": sha256(SCRIPT_PATH),
        "plan_sha256": sha256(plan_path),
        "gpus": list(gpus),
        "model_seed": 0,
        "episode_order_seed": 0,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise CampaignError("BATCH.json {} mismatch".format(key))
    if canonical(plan) != canonical(plan_payload(spec_path, spec, batch_id, gpus)):
        raise CampaignError("PLAN.json no longer matches the campaign expansion")
    return root, binding, plan


def tracked_worktree_dirty():
    return bool(subprocess.check_output([
        "git", "-C", str(REPO_ROOT), "status", "--porcelain",
        "--untracked-files=no",
    ], text=True).strip())


def require_tracked(path, label):
    path = Path(path).resolve()
    try:
        relative = str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        raise CampaignError("{} must be inside the repository".format(label))
    completed = subprocess.run([
        "git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", "--",
        relative,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        raise CampaignError("formal launch requires tracked {}: {}".format(label, relative))


def assert_clean_formal_tree(spec_path):
    if tracked_worktree_dirty():
        raise CampaignError("formal launch requires a clean tracked worktree")
    require_tracked(SCRIPT_PATH, "campaign runner")
    require_tracked(spec_path, "campaign spec")


def asset_index(spec):
    path = _validate_reference(
        spec["data_bindings"]["asset_manifest"], "asset manifest"
    )
    document = read_json(path)
    assets = document.get("assets")
    if not isinstance(assets, list):
        raise CampaignError("asset manifest has no assets list")
    values = {}
    for item in assets:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise CampaignError("asset manifest contains a malformed record")
        if item["id"] in values:
            raise CampaignError("asset manifest contains duplicate id {}".format(item["id"]))
        values[item["id"]] = item
    return path, values


def order_binding(spec, setting, split, require_dataset=False):
    key = ORDER_KEY.get((setting, split))
    if key is None:
        raise CampaignError("no canonical order binding for {}/{}".format(setting, split))
    binding = spec["data_bindings"]["streams"][key]
    path = _validate_reference(binding, "{} order manifest".format(key))
    document = read_json(path)
    dataset_value = document.get("dataset", {}).get("path")
    dataset_path = runtime_file(
        dataset_value, "{} dataset".format(key), require=require_dataset
    )
    if require_dataset and not direct_execution_mode(spec):
        identity = (str(dataset_path), document["dataset"]["sha256"])
        if identity not in _VERIFIED_LARGE_FILES:
            if sha256(dataset_path) != document["dataset"]["sha256"]:
                raise CampaignError("{} dataset SHA256 mismatch".format(key))
            _VERIFIED_LARGE_FILES.add(identity)
    return {
        "key": key,
        "path": path,
        "sha256": sha256(path),
        "document": document,
        "dataset_path": dataset_path,
        "dataset_sha256": document["dataset"]["sha256"],
        "order_sha256": document["order_sha256"],
        "episode_count": document["episode_count"],
        "benchmark": document["benchmark"],
    }


def checkpoint_binding(spec, setting, require_file=False):
    manifest_path, assets = asset_index(spec)
    asset_id = CHECKPOINT_ASSET[setting]
    asset = assets.get(asset_id)
    if not isinstance(asset, dict):
        raise CampaignError("missing checkpoint asset {}".format(asset_id))
    expected = spec["data_bindings"]["checkpoints"].get(setting)
    if asset.get("sha256") != expected or not valid_sha256(expected):
        raise CampaignError("{} checkpoint binding mismatch".format(setting))
    path = runtime_file(
        asset.get("path"), "{} checkpoint".format(setting),
        require=require_file,
    )
    if require_file and not direct_execution_mode(spec):
        identity = (str(path), expected)
        if identity not in _VERIFIED_LARGE_FILES:
            if sha256(path) != expected:
                raise CampaignError("{} checkpoint file SHA256 mismatch".format(setting))
            _VERIFIED_LARGE_FILES.add(identity)
    return {
        "asset_manifest_path": manifest_path,
        "asset_id": asset_id,
        "path": path,
        "sha256": expected,
        "size": path.stat().st_size if path.is_file() else asset.get("size"),
    }


def _resolve_manifest_artifact(value):
    """Resolve an artifact across repository relocations, then require it."""
    if not isinstance(value, str) or not value:
        raise CampaignError("formal result artifact path is missing")
    recorded = Path(value).expanduser()
    if recorded.is_file():
        return recorded.resolve()
    parts = recorded.parts
    candidates = []
    for anchor in ("vln", "core", "docs", "tools"):
        if anchor in parts:
            # Use the last matching anchor so a parent directory named like a
            # repository component cannot redirect the evidence lookup.
            index = len(parts) - 1 - tuple(reversed(parts)).index(anchor)
            candidates.append(REPO_ROOT.joinpath(*parts[index:]).resolve())
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise CampaignError(
            "formal result artifact is missing or ambiguously relocated: {}"
            .format(value)
        )
    return matches[0]


def _validate_formal_source_manifest(path, digest, record, expected):
    path = repo_file(path, "Source formal manifest", require=True)
    require_tracked(path, "Source formal manifest")
    if not valid_sha256(digest) or sha256(path) != digest:
        raise CampaignError("Source formal-manifest SHA256 mismatch: {}".format(path))
    manifest = read_json(path)
    checks = {
        "task": "vln",
        "benchmark": expected.get("benchmark"),
        "model": expected["model"],
        "method": "source",
        "seed": 0,
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in checks.items():
        if value is None:
            continue
        if manifest.get(key) != value:
            raise CampaignError("Source formal manifest {} mismatch".format(key))
    if manifest.get("checkpoint", {}).get("sha256") != expected["checkpoint_sha256"]:
        raise CampaignError("Source formal checkpoint mismatch")
    checkpoint = manifest.get("checkpoint", {})
    if (
        not isinstance(checkpoint.get("path"), str)
        or not checkpoint.get("path")
        or isinstance(checkpoint.get("size"), bool)
        or not isinstance(checkpoint.get("size"), int)
        or checkpoint.get("size") <= 0
    ):
        raise CampaignError("Source formal checkpoint record is sparse")
    dataset = manifest.get("dataset", {})
    if (
        not isinstance(dataset.get("path"), str)
        or not dataset.get("path")
        or isinstance(dataset.get("size"), bool)
        or not isinstance(dataset.get("size"), int)
        or dataset.get("size") <= 0
        or dataset.get("index_sha256") != expected["dataset_sha256"]
        or dataset.get("stream_content_sha256") != expected["dataset_sha256"]
        or dataset.get("stream_order_sha256") != expected["order_sha256"]
        or dataset.get("version") != expected["dataset_version"]
    ):
        raise CampaignError("Source formal dataset/order mismatch")
    source_setting = manifest.get("source_setting")
    if not isinstance(source_setting, str):
        raise CampaignError("Source formal source_setting is missing")
    source_parts = source_setting.split(":")
    if (
        len(source_parts) < 3
        or source_parts[0] != expected["setting"]
        or source_parts[1] != expected["split"]
        or (
            expected["data_version"] == "v1.2-native"
            and source_parts[2] != "v1.2-native"
        )
        or (
            expected["data_version"] == "discrete-native"
            and source_parts[2] not in ("native", "discrete-native")
        )
    ):
        raise CampaignError("Source formal physical split/data version mismatch")
    if (
        not isinstance(manifest.get("run_id"), str)
        or not manifest.get("run_id")
        or not isinstance(manifest.get("run_tag"), str)
        or not manifest.get("run_tag")
        or not isinstance(manifest.get("git_commit"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("git_commit", ""))
        or not isinstance(manifest.get("config"), str)
        or not manifest.get("config")
        or not isinstance(manifest.get("config_overrides"), list)
        or not manifest.get("config_overrides")
        or not isinstance(manifest.get("started_at"), str)
        or not isinstance(manifest.get("completed_at"), str)
    ):
        raise CampaignError("Source formal manifest is sparse")
    pinned = manifest.get("pinned_manifests")
    if not isinstance(pinned, dict):
        raise CampaignError("Source formal pinned manifests are missing")
    for name in ("assets", "environment", "episode_order"):
        item = pinned.get(name)
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item.get("path")
            or isinstance(item.get("size"), bool)
            or not isinstance(item.get("size"), int)
            or item.get("size") <= 0
            or not valid_sha256(item.get("sha256"))
        ):
            raise CampaignError("Source formal pinned {} is sparse".format(name))
    if pinned["episode_order"]["sha256"] != expected["order_manifest_sha256"]:
        raise CampaignError("Source formal episode-order manifest mismatch")
    hardware = manifest.get("hardware")
    if (
        not isinstance(hardware, dict)
        or any(not isinstance(hardware.get(key), str) or not hardware.get(key)
               for key in ("hostname", "platform", "python", "gpu_name"))
        or hardware.get("cuda_available") is not True
    ):
        raise CampaignError("Source formal hardware record is sparse")
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CampaignError("Source formal result artifacts are missing")
    artifact_digests = set()
    artifact_names = set()
    artifact_paths = set()
    validated_artifacts = []
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item.get("name")
            or not isinstance(item.get("path"), str)
            or not item.get("path")
            or isinstance(item.get("size"), bool)
            or not isinstance(item.get("size"), int)
            or item.get("size") <= 0
            or not valid_sha256(item.get("sha256"))
        ):
            raise CampaignError("Source formal result artifact is sparse")
        if item["name"] in artifact_names:
            raise CampaignError("Source formal result artifact name is duplicated")
        artifact_names.add(item["name"])
        artifact_path = _resolve_manifest_artifact(item["path"])
        if artifact_path in artifact_paths:
            raise CampaignError("Source formal result artifact path is duplicated")
        artifact_paths.add(artifact_path)
        if (
            artifact_path.stat().st_size != item["size"]
            or sha256(artifact_path) != item["sha256"]
        ):
            raise CampaignError("Source formal result artifact changed")
        artifact_digests.add(item["sha256"])
        validated_artifacts.append((item, artifact_path))
    if not any(
        "valid.txt" in str(item.get("name", "")).lower()
        or "stats_" in str(item.get("name", "")).lower()
        or "aggregate" in str(item.get("name", "")).lower()
        for item in artifacts
    ):
        raise CampaignError("Source formal manifest has no native aggregate artifact")
    metric_artifacts = [
        (item, artifact_path) for item, artifact_path in validated_artifacts
        if item.get("sha256") == expected.get("metric_artifact_sha256")
    ]
    if len(metric_artifacts) == 1:
        metric_item, metric_path = metric_artifacts[0]
        if (
            metric_item.get("size") != expected.get("metric_artifact_size")
            or metric_path != expected.get("metric_artifact_path")
        ):
            raise CampaignError(
                "Source metric artifact size/formal-manifest mismatch"
            )
    elif metric_artifacts:
        raise CampaignError(
            "Source metric artifact is not uniquely bound by the formal manifest"
        )
    else:
        # Some reviewed legacy ledgers point at a compact derived metrics.json
        # while the immutable run manifest binds the native valid.txt/stats
        # artifact.  Re-authenticate the native artifact and independently
        # reparse it; matching by run tag or plausible values alone is not
        # sufficient.
        native = [
            (item, artifact_path)
            for item, artifact_path in validated_artifacts
            if (
                "valid.txt" in str(item.get("name", "")).lower()
                or (
                    Path(str(item.get("name", ""))).name.lower().startswith("stats_")
                    and not Path(str(item.get("name", ""))).name.lower().startswith(
                        "stats_ep_"
                    )
                )
            )
        ]
        if len(native) != 1:
            raise CampaignError(
                "Source formal manifest has no unique native aggregate artifact"
            )
        _native_item, native_path = native[0]
        native_metrics = _source_metrics_from_artifact(
            native_path, expected["selection_benchmark"], expected["split"]
        )
        needed = expected["required_metrics"]
        if any(
            key not in native_metrics
            or not math.isclose(
                float(native_metrics[key]), float(expected["source_metrics"][key]),
                rel_tol=0.0, abs_tol=1e-6,
            )
            for key in needed
        ):
            raise CampaignError(
                "Source native aggregate disagrees with the reviewed ledger"
            )
    identity = manifest.get("immutable_identity_sha256")
    if not valid_sha256(identity):
        raise CampaignError("Source formal immutable identity is missing")
    from tools.run_manifest_identity import immutable_identity_sha256
    if immutable_identity_sha256(manifest) != identity:
        raise CampaignError("Source formal immutable identity mismatch")
    if record.get("immutable_identity_sha256") != identity:
        raise CampaignError("Source ledger/formal immutable identity mismatch")
    for key in ("model", "run_tag", "git_commit"):
        if record.get(key) != manifest.get(key):
            raise CampaignError("Source ledger/formal {} mismatch".format(key))
    return path, identity, sha256(path)


def _source_metrics_from_artifact(path, benchmark, split):
    path = Path(path)
    if path.suffix == ".json":
        document = read_json(path)
        nested = document.get("metrics") if isinstance(document.get("metrics"), dict) else document
        values = {
            key.upper(): float(value) for key, value in nested.items()
            if finite_number(value)
        }
        if "SUCCESS" in values and "SR" not in values:
            values["SR"] = 100.0 * values["SUCCESS"]
        if benchmark == "r2r-ce":
            for key in ("SPL", "NDTW", "SDTW"):
                if key in values and abs(values[key]) <= 1.0:
                    values[key] *= 100.0
        return values
    lines = [
        line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "Env name: {}".format(split) in line
    ]
    if len(lines) != 1:
        raise CampaignError("Source metric artifact has no unique split line")
    return {key.upper(): float(value) for key, value in METRIC_RE.findall(lines[0])}


def _validate_source_ledger_header(binding_key, benchmark, split, ledger_path,
                                   ledger):
    require_tracked(
        ledger_path, "{} Source control ledger".format(binding_key)
    )
    if ledger.get("schema") != SOURCE_LEDGER_SCHEMA[binding_key]:
        raise CampaignError("{} Source ledger schema mismatch".format(binding_key))
    if "candidate_status" in ledger:
        raise CampaignError(
            "{} is a candidate ledger and has not been promoted"
            .format(binding_key)
        )
    if benchmark != "r2r-ce":
        return
    review = ledger.get("promotion_review")
    if (
        ledger.get("schema_version") != 1
        or ledger.get("promotion_status")
        != "independently_reviewed_and_promoted"
        or not isinstance(review, dict)
        or review.get("decision") != "approved"
        or not isinstance(review.get("reviewed_by"), str)
        or not review["reviewed_by"].strip()
        or not isinstance(review.get("reviewed_at"), str)
        or not review["reviewed_at"].strip()
        or not valid_sha256(review.get("candidate_ledger_sha256"))
    ):
        raise CampaignError(
            "{} lacks an authenticated promotion review".format(binding_key)
        )
    candidate_path = repo_file(
        review.get("candidate_ledger_path"),
        "{} reviewed candidate ledger".format(binding_key),
        require=True,
    )
    require_tracked(
        candidate_path, "{} reviewed candidate ledger".format(binding_key)
    )
    if sha256(candidate_path) != review["candidate_ledger_sha256"]:
        raise CampaignError(
            "{} reviewed candidate ledger digest mismatch".format(binding_key)
        )
    candidate = read_json(candidate_path)
    if (
        candidate.get("schema")
        != "navtta.vln_r2r_ce_v1_2_source_controls_candidate.v1"
        or candidate.get("schema_version") != 1
        or candidate.get("candidate_status")
        != "review_required_not_yet_promoted"
        or candidate.get("split") != split
        or candidate.get("records") != ledger.get("records")
    ):
        raise CampaignError(
            "{} promoted ledger does not match its reviewed candidate"
            .format(binding_key)
        )


def validate_source_controls(spec):
    """Authenticate all Source controls needed on both campaign splits."""
    bindings = spec.get("source_control", {}).get("bindings", {})
    # Report the deliberate v1 native-CE prerequisite before auditing legacy
    # discrete evidence.  This keeps plan-only v1 diagnostics actionable while
    # a reviewed successor still validates every ledger below.
    for key in ("r2r_ce_v1_2_val_unseen", "r2r_ce_v1_2_val_seen"):
        binding = bindings.get(key)
        if not isinstance(binding, dict) or binding.get("mode") != "reuse":
            raise CampaignError(
                "{} matched Source control is not ready (mode must be reuse)"
                .format(key)
            )
    cells_by_benchmark = defaultdict(set)
    for cell in spec["cells"]:
        cells_by_benchmark[cell["benchmark"]].add(cell["setting"])
    evidence = {}
    for benchmark in ("reverie", "r2r", "r2r-ce"):
        for split in ("val_unseen", "val_seen"):
            binding_key = SOURCE_BINDING[(benchmark, split)]
            binding = bindings.get(binding_key)
            if not isinstance(binding, dict) or binding.get("mode") != "reuse":
                raise CampaignError(
                    "{} matched Source control is not ready (mode must be reuse)"
                    .format(binding_key)
                )
            ledger_path = _validate_reference(
                binding, "{} Source control".format(binding_key)
            )
            if ledger_path in [item["path"] for item in evidence.values()]:
                raise CampaignError("Source bindings unexpectedly reuse one ledger path")
            ledger = read_json(ledger_path)
            _validate_source_ledger_header(
                binding_key, benchmark, split, ledger_path, ledger
            )
            if ledger.get("split") != split:
                raise CampaignError("{} Source ledger split mismatch".format(binding_key))
            seed_values = [
                ledger[key] for key in ("canonical_order_seed", "order_seed")
                if key in ledger
            ]
            if not seed_values or any(value != 0 for value in seed_values):
                raise CampaignError("{} Source ledger is not canonical seed 0".format(binding_key))
            records = ledger.get("records")
            if not isinstance(records, dict):
                # Permit an explicitly named settings mapping in successor
                # native-v1.2 ledgers, but never guess another structure.
                records = ledger.get("settings")
            if not isinstance(records, dict):
                raise CampaignError("{} Source ledger has no records".format(binding_key))
            for setting in sorted(cells_by_benchmark[benchmark]):
                record = records.get(setting)
                if not isinstance(record, dict):
                    raise CampaignError("{} lacks Source record {}".format(binding_key, setting))
                order = order_binding(spec, setting, split, require_dataset=False)
                if ledger.get("episode_count") != order["episode_count"]:
                    raise CampaignError("{} Source horizon mismatch".format(binding_key))
                expected = {
                    "setting": setting,
                    "split": split,
                    "data_version": next(
                        cell["data_version"] for cell in spec["cells"]
                        if cell["setting"] == setting
                    ),
                    "model": next(cell["model"] for cell in spec["cells"] if cell["setting"] == setting),
                    "benchmark": order["benchmark"],
                    "dataset_version": order["benchmark"],
                    "checkpoint_sha256": spec["data_bindings"]["checkpoints"][setting],
                    "dataset_sha256": order["dataset_sha256"],
                    "order_sha256": order["order_sha256"],
                    "order_manifest_sha256": order["sha256"],
                }
                for field in ("checkpoint_sha256", "dataset_sha256"):
                    if record.get(field) != expected[field]:
                        raise CampaignError("{} {} {} mismatch".format(binding_key, setting, field))
                record_order = record.get("episode_order_sha256", ledger.get("episode_order_sha256"))
                if record_order != expected["order_sha256"]:
                    raise CampaignError("{} {} episode order mismatch".format(binding_key, setting))
                if record.get("evidence_status") != "ready":
                    raise CampaignError("{} {} Source evidence is not ready".format(binding_key, setting))
                parameters = record.get("parameters", {})
                if (
                    not isinstance(parameters, dict)
                    or parameters.get("action_selection") != "argmax"
                    or parameters.get("action_seed") != 0
                ):
                    raise CampaignError("{} {} Source action protocol is not argmax".format(binding_key, setting))
                metrics = record.get("metrics")
                needed = ("RGSPL", "RGS", "SPL", "SR") if benchmark == "reverie" else ("SPL", "SR")
                if not isinstance(metrics, dict) or any(not finite_number(metrics.get(name)) for name in needed):
                    raise CampaignError("{} {} Source metrics are incomplete".format(binding_key, setting))
                metric_value = (
                    record.get("metrics_artifact_path")
                    or record.get("aggregate_artifact_path")
                    or record.get("metrics_json_path")
                )
                metric_digest = (
                    record.get("metrics_artifact_sha256")
                    or record.get("aggregate_artifact_sha256")
                    or record.get("metrics_json_sha256")
                )
                metric_path = repo_file(metric_value, "Source metric artifact", require=True)
                if sha256(metric_path) != metric_digest:
                    raise CampaignError("{} {} Source metric artifact changed".format(binding_key, setting))
                expected["metric_artifact_sha256"] = metric_digest
                expected["metric_artifact_size"] = metric_path.stat().st_size
                expected["metric_artifact_path"] = metric_path.resolve()
                expected["selection_benchmark"] = benchmark
                expected["required_metrics"] = needed
                expected["source_metrics"] = metrics
                parsed_metrics = _source_metrics_from_artifact(
                    metric_path, benchmark, split
                )
                if any(
                    key not in parsed_metrics
                    or not math.isclose(
                        float(parsed_metrics[key]), float(metrics[key]),
                        rel_tol=0.0, abs_tol=1e-6,
                    )
                    for key in needed
                ):
                    raise CampaignError("{} {} Source metric values changed".format(
                        binding_key, setting))
                formal_path, identity, formal_digest = _validate_formal_source_manifest(
                    record.get("formal_manifest_path"),
                    record.get("formal_manifest_sha256"), record, expected,
                )
                record["_validated_formal_path"] = str(formal_path)
                record["_validated_formal_identity"] = identity
                record["_validated_formal_sha256"] = formal_digest
            evidence[binding_key] = {
                "path": ledger_path,
                "sha256": sha256(ledger_path),
                "document": ledger,
                "formal_manifests": {
                    setting: {
                        "path": records[setting]["_validated_formal_path"],
                        "sha256": records[setting]["_validated_formal_sha256"],
                        "immutable_identity_sha256": records[setting][
                            "_validated_formal_identity"
                        ],
                    }
                    for setting in sorted(cells_by_benchmark[benchmark])
                },
            }
    forbidden = {
        str(repo_file(path, "forbidden Source ledger", require=False))
        for path in spec["source_control"].get("forbidden_reuse", ())
    }
    if any(str(item["path"]) in forbidden for item in evidence.values()):
        raise CampaignError("a forbidden unified-v1.3 Source ledger was reused")
    return evidence


def validate_idea_assets(spec):
    evidence = {}
    for setting, binding in sorted(
        spec["data_bindings"].get("idea_source_statistics", {}).items()
    ):
        path = runtime_file(
            binding.get("path"), "IDEA Source statistics", require=True
        )
        if sha256(path) != binding.get("sha256"):
            raise CampaignError("{} IDEA Source-statistics SHA256 mismatch".format(setting))
        document = read_json(path)
        provenance = document.get("provenance", {})
        if (
            document.get("schema") != "navtta.idea.source_statistics"
            or document.get("version") != 1
            or binding.get("trajectory_count") != 128
            or provenance.get("trajectory_count") != 128
            or provenance.get("setting") != setting
            or provenance.get("checkpoint_sha256")
            != spec["data_bindings"]["checkpoints"][setting]
        ):
            raise CampaignError("{} IDEA Source-statistics provenance mismatch".format(setting))
        evidence[setting] = {"path": path, "sha256": binding["sha256"]}
    expected = {
        cell["setting"] for cell in spec["cells"] if cell["method"] == "idea"
    }
    if set(evidence) != expected:
        raise CampaignError("IDEA Source-statistics setting coverage mismatch")
    return evidence


def validate_runtime_assets(spec, stage):
    splits = {
        "search": ("val_unseen",),
        "val-seen": ("val_seen",),
        "reverie-test": ("test",),
    }[stage]
    cells = spec["cells"] if stage != "reverie-test" else [
        _cells_by_id(spec)[item["source_cell_id"]]
        for item in spec["hidden_test_transfer"]["submission_cells"]
    ]
    seen = set()
    for cell in cells:
        setting = cell["setting"]
        if setting not in seen:
            checkpoint_binding(spec, setting, require_file=True)
            seen.add(setting)
        for split in splits:
            order_binding(spec, setting, split, require_dataset=True)
    if not RUNNER.is_file() or not os.access(str(RUNNER), os.X_OK):
        raise CampaignError("run_source_eval.sh is missing or not executable")
    if not TRANSLATOR.is_file():
        raise CampaignError("TTA config translator is missing")
    if not ENVIRONMENT_MANIFEST.is_file():
        raise CampaignError("environment manifest is missing")


def _evidence_file(record, label, size_key="size"):
    if not isinstance(record, dict):
        raise CampaignError("{} evidence must be an object".format(label))
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise CampaignError("{} evidence path is missing".format(label))
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise CampaignError("{} evidence path must be absolute".format(label))
    path = path.resolve()
    size = record.get(size_key)
    digest = record.get("sha256")
    if (
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        or not valid_sha256(digest) or not path.is_file()
        or path.stat().st_size != size or sha256(path) != digest
    ):
        raise CampaignError("{} evidence changed or is invalid".format(label))
    return path


def _same_evidence(left, right, left_size="size", right_size="size_bytes"):
    try:
        return (
            Path(left["path"]).resolve() == Path(right["path"]).resolve()
            and left[left_size] == right[right_size]
            and left["sha256"] == right["sha256"]
        )
    except (KeyError, TypeError, OSError):
        return False


def _validate_provider_preflight_document(
    spec, preflight_path, render_build, live_health=None,
):
    """Authenticate the complete render/provider smoke evidence graph."""
    transfer = spec["hidden_test_transfer"]
    provider = transfer["provider"]
    preflight_contract = transfer["runtime_preflight"]
    runtime = provider["runtime"]
    model_contract = read_json(_validate_reference(
        runtime["contract"], "FeedTTA-LLM provider contract"
    ))

    preflight_path = Path(preflight_path).expanduser()
    render_build = Path(render_build).expanduser()
    if not preflight_path.is_absolute() or not preflight_path.is_file():
        raise CampaignError("FeedTTA-LLM PRECHECK must be an existing absolute file")
    if not render_build.is_absolute() or not render_build.is_dir():
        raise CampaignError("FeedTTA-LLM render build must be an existing absolute directory")
    preflight_path = preflight_path.resolve()
    render_build = render_build.resolve()
    try:
        preflight_path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise CampaignError("FeedTTA-LLM PRECHECK must remain outside Git")

    document = read_json(preflight_path)
    expected_top = {
        "schema": preflight_contract["schema"],
        "status": "passed",
        "git_commit": git_commit(),
        "render_mattersim_build": str(render_build),
        "service_url": runtime["service_url"],
    }
    for key, expected in expected_top.items():
        if document.get(key) != expected:
            raise CampaignError("FeedTTA-LLM PRECHECK {} mismatch".format(key))
    for key in ("created_at", "scan_id", "viewpoint_id"):
        if not isinstance(document.get(key), str) or not document[key]:
            raise CampaignError("FeedTTA-LLM PRECHECK {} is missing".format(key))

    model_identity = document.get("model_identity")
    expected_model = {
        "model_id": provider["model_id"],
        "model_dtype": "float16",
        "revision": provider["revision"],
        "weights_sha256": provider["weights_sha256"],
        "bundle_sha256": provider["model_bundle_sha256"],
        "prompt_bundle_sha256": provider["prompt_bundle_sha256"],
    }
    if not isinstance(model_identity, dict):
        raise CampaignError("FeedTTA-LLM PRECHECK model identity is missing")
    for key, expected in expected_model.items():
        if model_identity.get(key) != expected:
            raise CampaignError(
                "FeedTTA-LLM PRECHECK model {} mismatch".format(key)
            )
    if model_contract.get("bundle_sha256") != provider["model_bundle_sha256"]:
        raise CampaignError("FeedTTA-LLM model bundle binding mismatch")

    artifacts = document.get("artifacts")
    required_artifacts = preflight_contract["required_artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(required_artifacts):
        raise CampaignError("FeedTTA-LLM PRECHECK artifact set mismatch")
    artifact_paths = {
        name: _evidence_file(record, "PRECHECK {}".format(name))
        for name, record in artifacts.items()
    }

    model_verify = read_json(artifact_paths["model_verify"])
    for key, expected in expected_model.items():
        if model_verify.get(key) != expected:
            raise CampaignError("model verification {} mismatch".format(key))
    if (
        model_verify.get("runtime_device_type") != "cuda"
        or model_verify.get("runtime_dtype") != "float16"
        or model_verify.get("cuda_available") is not True
    ):
        raise CampaignError("model verification is not CUDA/float16")

    render = read_json(artifact_paths["render_evidence"])
    repository = render.get("repository", {})
    if (
        render.get("schema") != preflight_contract["render_evidence_schema"]
        or render.get("status") != "passed"
        or repository.get("git_commit") != git_commit()
        or repository.get("tracked_tree_clean") is not True
        or Path(str(repository.get("root", ""))).resolve() != REPO_ROOT.resolve()
        or render.get("scan_id") != document["scan_id"]
        or render.get("viewpoint_id") != document["viewpoint_id"]
    ):
        raise CampaignError("render preflight identity mismatch")
    build = render.get("headless_build")
    if not isinstance(build, dict) or Path(str(build.get("build_dir", ""))).resolve() != render_build:
        raise CampaignError("render preflight build path mismatch")
    if build.get("backend") not in ("egl", "osmesa"):
        raise CampaignError("render preflight headless backend is invalid")
    cmake_path = _evidence_file(
        build.get("cmake_cache"), "render CMake cache", "size_bytes"
    )
    module_path = _evidence_file(
        build.get("module_candidate"), "render MatterSim module", "size_bytes"
    )
    if cmake_path != render_build / "CMakeCache.txt":
        raise CampaignError("render CMake cache escaped the selected build")
    try:
        module_path.relative_to(render_build)
    except ValueError as error:
        raise CampaignError("render MatterSim module escaped the selected build") from error
    cache_text = cmake_path.read_text(encoding="utf-8", errors="strict")
    flags = {}
    for name in ("EGL_RENDERING", "OSMESA_RENDERING"):
        match = re.search(
            r"(?m)^{}:BOOL=(ON|OFF)$".format(re.escape(name)), cache_text
        )
        if match is None:
            raise CampaignError("render CMake cache lacks {}".format(name))
        flags[name] = match.group(1) == "ON"
    expected_backend = "egl" if flags["EGL_RENDERING"] else "osmesa"
    if sum(flags.values()) != 1 or build["backend"] != expected_backend:
        raise CampaignError("render CMake backend no longer matches PRECHECK")

    imports = render.get("imports")
    if not isinstance(imports, dict):
        raise CampaignError("render import evidence is missing")
    imported_mattersim = imports.get("mattersim")
    _evidence_file(imported_mattersim, "loaded render MatterSim", "size_bytes")
    if not _same_evidence(
        build["module_candidate"], imported_mattersim,
        left_size="size_bytes", right_size="size_bytes",
    ):
        raise CampaignError("loaded MatterSim differs from the selected module")
    core_module = _evidence_file(
        imports.get("navtta_core"), "render navtta_core", "size_bytes"
    )
    try:
        core_module.relative_to((REPO_ROOT / "core").resolve())
    except ValueError as error:
        raise CampaignError("render navtta_core escaped the current checkout") from error
    verifier = _evidence_file(
        render.get("verifier"), "render verifier", "size_bytes"
    )
    if verifier != (REPO_ROOT / "vln/scripts/verify_reverie_render_preflight.py").resolve():
        raise CampaignError("render verifier path mismatch")

    required_inputs = render.get("required_input_files")
    if not isinstance(required_inputs, list) or not required_inputs:
        raise CampaignError("render source-input evidence is missing")
    role_counts = Counter()
    for index, record in enumerate(required_inputs):
        _evidence_file(
            record, "render source input {}".format(index), "size_bytes"
        )
        role_counts[record.get("role")] += 1
    if (
        role_counts["connectivity_scan_list"] != 1
        or role_counts["connectivity_graph"] < 1
        or role_counts["raw_rgb_cubemap"] != 1
    ):
        raise CampaignError("render source-input roles are incomplete")
    raw_rgb = render.get("raw_rgb", {}).get("file")
    raw_records = [
        item for item in required_inputs if item.get("role") == "raw_rgb_cubemap"
    ]
    if not raw_records or raw_rgb != raw_records[0]:
        raise CampaignError("render raw-RGB evidence mismatch")

    rendered_panorama = artifacts["rendered_panorama"]
    render_output = render.get("render", {})
    if (
        Path(str(render_output.get("output_path", ""))).resolve()
        != artifact_paths["rendered_panorama"]
        or render_output.get("output_size_bytes") != rendered_panorama.get("size")
        or render_output.get("output_sha256") != rendered_panorama.get("sha256")
        or render_output.get("metadata", {}).get("view_count") != 36
        or render_output.get("metadata", {}).get("panorama_png_sha256")
        != rendered_panorama.get("sha256")
    ):
        raise CampaignError("rendered panorama evidence mismatch")

    smoke = read_json(artifact_paths["provider_smoke"])
    smoke_health = smoke.get("service")
    if (
        not isinstance(smoke_health, dict)
        or smoke.get("result", {}).get("available") is not True
    ):
        raise CampaignError("provider smoke evidence is invalid")
    expected_health = {
        "ready": True,
        "provider_id": "qwen2_vl_2b_v1",
        "model_id": provider["model_id"],
        "revision": provider["revision"],
        "weights_sha256": provider["weights_sha256"],
        "bundle_sha256": provider["model_bundle_sha256"],
        "prompt_bundle_sha256": provider["prompt_bundle_sha256"],
        "stage1_prompt_sha256": provider["pipeline"][0]["prompt_sha256"],
        "stage2_prompt_sha256": provider["pipeline"][1]["prompt_sha256"],
        "runtime_device_type": "cuda",
        "runtime_dtype": "float16",
        "cuda_available": True,
        "model_loaded": True,
        "model_eval_mode": True,
        "model_parameter_device_types": ["cuda"],
        "model_parameter_dtypes": ["float16"],
    }
    for key, expected in expected_health.items():
        if smoke_health.get(key) != expected:
            raise CampaignError("provider smoke health {} mismatch".format(key))
    if not valid_sha256(smoke_health.get("auth_token_sha256")):
        raise CampaignError("provider smoke token identity is invalid")
    if live_health is not None:
        for key, expected in expected_health.items():
            if live_health.get(key) != expected:
                raise CampaignError("live provider health {} mismatch".format(key))
        if live_health.get("auth_token_sha256") != smoke_health.get("auth_token_sha256"):
            raise CampaignError("provider token identity differs from PRECHECK")

    transcript_path = artifact_paths["provider_transcript"]
    try:
        transcript_rows = [
            json.loads(line) for line in transcript_path.read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError("provider smoke transcript is invalid") from error
    if (
        len(transcript_rows) != 1
        or transcript_rows[0].get("status") != "label"
        or transcript_rows[0].get("scan_id") != document["scan_id"]
        or transcript_rows[0].get("endpoint_viewpoint_id")
        != document["viewpoint_id"]
        or transcript_rows[0].get("panorama_png_sha256")
        != rendered_panorama.get("sha256")
    ):
        raise CampaignError("provider smoke transcript binding mismatch")

    return {
        "schema": LLM_PREFLIGHT_SCHEMA,
        "path": str(preflight_path),
        "size": preflight_path.stat().st_size,
        "sha256": sha256(preflight_path),
        "git_commit": document["git_commit"],
        "render_mattersim_build": str(render_build),
        "render_backend": build["backend"],
        "render_cmake_cache_sha256": build["cmake_cache"]["sha256"],
        "render_module_sha256": build["module_candidate"]["sha256"],
        "render_evidence_sha256": artifacts["render_evidence"]["sha256"],
        "rendered_panorama_sha256": rendered_panorama["sha256"],
        "provider_smoke_sha256": artifacts["provider_smoke"]["sha256"],
        "provider_transcript_sha256": artifacts["provider_transcript"]["sha256"],
        "model_bundle_sha256": provider["model_bundle_sha256"],
    }


def validate_provider_preflight(spec, provider_state):
    contract = spec["hidden_test_transfer"]["runtime_preflight"]
    preflight_value = os.environ.get(contract["path_env"], "")
    render_value = os.environ.get(contract["render_build_env"], "")
    if not preflight_value:
        raise CampaignError("{} is required for reverie-test".format(
            contract["path_env"]
        ))
    if not render_value:
        raise CampaignError("{} is required for reverie-test".format(
            contract["render_build_env"]
        ))
    return _validate_provider_preflight_document(
        spec, preflight_value, render_value, provider_state["health"]
    )


def validate_provider_preflight_binding(spec, binding):
    if not isinstance(binding, dict):
        raise CampaignError("FeedTTA-LLM provider preflight binding is missing")
    rebuilt = _validate_provider_preflight_document(
        spec, binding.get("path", ""), binding.get("render_mattersim_build", "")
    )
    if canonical(rebuilt) != canonical(binding):
        raise CampaignError("FeedTTA-LLM provider preflight binding changed")
    return rebuilt


def validate_provider(spec):
    transfer = spec.get("hidden_test_transfer", {})
    provider = transfer.get("provider", {})
    if transfer.get("status") not in ("ready", "frozen_ready"):
        raise CampaignError("FeedTTA-LLM provider contract is not ready")
    for key in (
        "model_id", "revision", "weights_sha256", "model_bundle_sha256",
    ):
        value = provider.get(key)
        if not isinstance(value, str) or not value:
            raise CampaignError("FeedTTA-LLM provider {} is missing".format(key))
    if not all(valid_sha256(provider[key]) for key in (
        "weights_sha256", "model_bundle_sha256",
    )):
        raise CampaignError("FeedTTA-LLM model SHA256 identity is invalid")
    pipeline = provider.get("pipeline")
    if not isinstance(pipeline, list) or len(pipeline) != 2:
        raise CampaignError("FeedTTA-LLM requires exactly two prompt stages")
    if any(not valid_sha256(item.get("prompt_sha256")) for item in pipeline):
        raise CampaignError("FeedTTA-LLM prompt identities are incomplete")
    runtime = provider.get("runtime")
    if not isinstance(runtime, dict):
        raise CampaignError("FeedTTA-LLM provider runtime binding is missing")
    contract_path = _validate_reference(
        runtime.get("contract"), "FeedTTA-LLM provider contract"
    )
    contract = read_json(contract_path)
    expected_contract = {
        "model_id": provider["model_id"],
        "revision": provider["revision"],
        "weights_sha256": provider["weights_sha256"],
    }
    for key, value in expected_contract.items():
        if contract.get(key) != value:
            raise CampaignError("FeedTTA-LLM model contract {} mismatch".format(key))
    if (
        contract.get("schema") != "navtta.external_model_asset.v1"
        or contract.get("bundle_sha256") != provider["model_bundle_sha256"]
    ):
        raise CampaignError("FeedTTA-LLM model contract schema/bundle is invalid")
    prompt_hashes = [item["prompt_sha256"] for item in pipeline]
    if provider.get("prompt_bundle_sha256") is None or not valid_sha256(
        provider.get("prompt_bundle_sha256")
    ):
        raise CampaignError("FeedTTA-LLM prompt bundle SHA256 is invalid")
    model_path = Path(str(runtime.get("model_path", ""))).expanduser()
    if not model_path.is_absolute() or not model_path.is_dir():
        raise CampaignError("FeedTTA-LLM model_path must be an existing absolute directory")
    # Recompute every file and both aggregate identities.  Merely seeing the
    # directory is not enough to establish which weights the service loaded.
    files = contract.get("files")
    if not isinstance(files, list) or not files:
        raise CampaignError("FeedTTA-LLM model contract has no files")
    bundle_lines = []
    weight_lines = []
    weight_names = {
        item.get("path") for item in contract.get("weight_files", [])
        if isinstance(item, dict)
    }
    if len(weight_names) != 2:
        raise CampaignError("FeedTTA-LLM model contract must pin two weight shards")
    for item in sorted(files, key=lambda value: value.get("path", "")):
        relative = item.get("path") if isinstance(item, dict) else None
        digest = item.get("sha256") if isinstance(item, dict) else None
        if (
            not isinstance(relative, str) or not relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts
            or not valid_sha256(digest)
        ):
            raise CampaignError("FeedTTA-LLM model contract has an invalid file")
        file_path = model_path / relative
        if not file_path.is_file() or sha256(file_path) != digest:
            raise CampaignError("FeedTTA-LLM model file changed: {}".format(file_path))
        bundle_lines.append("{}:{}\n".format(relative, digest))
        if relative in weight_names:
            weight_lines.append("{}:{}\n".format(relative, digest))
    bundle_digest = hashlib.sha256("".join(bundle_lines).encode("utf-8")).hexdigest()
    weights_digest = hashlib.sha256("".join(weight_lines).encode("utf-8")).hexdigest()
    if bundle_digest != contract.get("bundle_sha256"):
        raise CampaignError("FeedTTA-LLM model bundle SHA256 mismatch")
    if weights_digest != provider["weights_sha256"]:
        raise CampaignError("FeedTTA-LLM weight aggregate SHA256 mismatch")

    service_url = runtime.get("service_url")
    if not isinstance(service_url, str) or not service_url.startswith(
        ("http://127.0.0.1:", "http://localhost:", "http://[::1]:")
    ):
        raise CampaignError("FeedTTA-LLM service_url must be loopback HTTP")
    token_path = Path(os.environ.get(
        "NAVTTA_LLM_FEEDBACK_TOKEN_FILE", str(runtime.get("token_file", ""))
    )).expanduser()
    if not token_path.is_absolute() or not token_path.is_file():
        raise CampaignError("FeedTTA-LLM token_file must be an existing absolute file")
    if stat.S_IMODE(token_path.stat().st_mode) != 0o600:
        raise CampaignError("FeedTTA-LLM token_file permissions must be 0600")
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise CampaignError("cannot read FeedTTA-LLM token_file: {}".format(error))
    if not token:
        raise CampaignError("FeedTTA-LLM token_file is empty")
    request = urllib.request.Request(
        service_url.rstrip("/") + "/v1/health",
        headers={"Authorization": "Bearer " + token},
    )
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            health = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise CampaignError("FeedTTA-LLM health check failed: {}".format(error))
    expected_health = {
        "ready": True,
        "provider_id": "qwen2_vl_2b_v1",
        "model_id": provider["model_id"],
        "revision": provider["revision"],
        "weights_sha256": provider["weights_sha256"],
        "bundle_sha256": contract["bundle_sha256"],
        "prompt_bundle_sha256": provider.get("prompt_bundle_sha256"),
        "stage1_prompt_sha256": prompt_hashes[0],
        "stage2_prompt_sha256": prompt_hashes[1],
    }
    for key, value in expected_health.items():
        if health.get(key) != value:
            raise CampaignError("FeedTTA-LLM health {} mismatch".format(key))
    if not valid_sha256(health.get("auth_token_sha256")):
        raise CampaignError("FeedTTA-LLM health token identity is invalid")
    return {
        "contract_path": contract_path, "contract_sha256": sha256(contract_path),
        "model_path": model_path, "token_path": token_path.resolve(),
        "runtime": runtime, "health": health,
    }


def source_controls_are_execution_gate(spec):
    return not direct_execution_mode(spec)


def formal_preflight(spec_path, spec, stage):
    if direct_execution_mode(spec) and stage != "reverie-test":
        return {"source_controls": {}, "idea_assets": {}, "provider": None}
    assert_clean_formal_tree(spec_path)
    validate_runtime_assets(spec, stage)
    sources = (
        validate_source_controls(spec)
        if source_controls_are_execution_gate(spec)
        else {}
    )
    ideas = validate_idea_assets(spec)
    provider = validate_provider(spec) if stage == "reverie-test" else None
    if provider is not None:
        provider["preflight"] = validate_provider_preflight(spec, provider)
    return {"source_controls": sources, "idea_assets": ideas, "provider": provider}


def _safe_component(value, label):
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        raise CampaignError("invalid {}: {!r}".format(label, value))
    return value


def _job_key(job):
    suffix = job.get("submission_id") or job["candidate_id"]
    return "q{:02d}-{}-{}".format(
        int(job["queue_id"]), job["cell_id"], suffix
    )


def _job_parent(root, job):
    return (
        Path(root) / "stages" / job["stage"]
        / "q{:02d}".format(int(job["queue_id"]))
        / job["cell_id"] / (job.get("submission_id") or job["candidate_id"])
    )


def _attempts(root, job):
    parent = _job_parent(root, job)
    if not parent.is_dir():
        return []
    values = []
    for path in parent.glob("attempt-[0-9][0-9][0-9]"):
        try:
            values.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return sorted(values)


def _process_identity(pid):
    """Return a stable local process identity, not merely a reusable PID."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        pgid = os.getpgid(pid)
    except OSError:
        return None
    proc_stat = Path("/proc") / str(pid) / "stat"
    proc_cmdline = Path("/proc") / str(pid) / "cmdline"
    try:
        # The process name is parenthesized and may contain spaces.  Fields
        # after the final ``) `` start at proc(5) field 3; index 19 is the
        # kernel start-time tick (field 22), which is stable across PID reuse.
        remainder = proc_stat.read_text(encoding="utf-8").rsplit(") ", 1)[1]
        start_token = "linux-start-ticks:" + remainder.split()[19]
        command_bytes = proc_cmdline.read_bytes()
        if not command_bytes:
            return None
    except (OSError, IndexError, UnicodeError):
        started = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            check=False,
        )
        command = subprocess.run(
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            check=False,
        )
        if (
            started.returncode != 0 or command.returncode != 0
            or not started.stdout.strip() or not command.stdout.strip()
        ):
            return None
        start_token = "ps-lstart:" + started.stdout.strip()
        command_bytes = command.stdout.strip().encode("utf-8")
    return {
        "schema": PROCESS_SCHEMA,
        "hostname": socket.gethostname(),
        "pid": pid,
        "process_group_id": pgid,
        "start_token": start_token,
        "command_sha256": hashlib.sha256(command_bytes).hexdigest(),
    }


def _pid_alive(attempt_dir):
    attempt_dir = Path(attempt_dir)
    path = attempt_dir / "pid"
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        recorded = read_json(attempt_dir / "process_identity.json")
    except (CampaignError, OSError, ValueError):
        return False
    if (
        recorded.get("schema") != PROCESS_SCHEMA
        or recorded.get("hostname") != socket.gethostname()
        or recorded.get("pid") != pid
        or not valid_sha256(recorded.get("command_sha256"))
        or not isinstance(recorded.get("start_token"), str)
    ):
        return False
    current = _process_identity(pid)
    if current is None:
        return False
    return all(current.get(key) == recorded.get(key) for key in (
        "schema", "hostname", "pid", "process_group_id", "start_token",
        "command_sha256",
    ))


def _recorded_process_group_alive(attempt_dir):
    try:
        recorded = read_json(Path(attempt_dir) / "process_identity.json")
    except CampaignError:
        return False
    if recorded.get("hostname") != socket.gethostname():
        return True
    pgid = recorded.get("process_group_id")
    if isinstance(pgid, bool) or not isinstance(pgid, int) or pgid <= 0:
        return True
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def attempt_state(attempt_dir):
    attempt_dir = Path(attempt_dir)
    if (attempt_dir / "validation_error.json").is_file():
        return "invalid"
    if (attempt_dir / "result.json").is_file():
        return "completed"
    exit_path = attempt_dir / "exitcode"
    if exit_path.is_file():
        try:
            return "finished" if int(exit_path.read_text().strip()) == 0 else "failed"
        except ValueError:
            return "invalid"
    if _pid_alive(attempt_dir):
        return "running"
    if _recorded_process_group_alive(attempt_dir):
        return "running"
    if (attempt_dir / "job.json").is_file():
        return "orphaned"
    return "pending"


def latest_attempt(root, job):
    values = _attempts(root, job)
    return values[-1] if values else None


def _provider_parameters(spec, batch_id, job, result_root=None):
    if job.get("feedback_provider") != "qwen2_vl_2b_v1":
        return {}
    provider = spec["hidden_test_transfer"]["provider"]
    runtime = provider["runtime"]
    cache_root = tuning_batch_root(batch_id) / "provider_cache" / job["submission_id"]
    contract = read_json(_validate_reference(
        runtime["contract"], "FeedTTA-LLM provider contract"
    ))
    token_value = os.environ.get(
        "NAVTTA_LLM_FEEDBACK_TOKEN_FILE", str(runtime.get("token_file", ""))
    )
    token_path = Path(token_value or "/MISSING_NAVTTA_LLM_FEEDBACK_TOKEN_FILE").expanduser()
    transcript = (
        Path(result_root) / "llm_feedback_transcript.ndjson"
        if result_root is not None else
        batch_root(batch_id) / "stages/reverie-test/PENDING_TRANSCRIPT_PATH"
    )
    return {
        "feedback_provider": "qwen2_vl_2b_v1",
        "llm_feedback_url": runtime["service_url"],
        "llm_feedback_model_id": provider["model_id"],
        "llm_feedback_revision": provider["revision"],
        "llm_feedback_weights_sha256": provider["weights_sha256"],
        "llm_feedback_bundle_sha256": contract["bundle_sha256"],
        "llm_feedback_prompt_sha256": provider["prompt_bundle_sha256"],
        "llm_feedback_token_file": str(token_path.resolve()),
        "llm_feedback_cache_dir": str(cache_root.resolve()),
        "llm_feedback_transcript_path": str(transcript.resolve()),
        "llm_feedback_timeout_seconds": float(runtime.get("timeout_seconds", 120.0)),
        "llm_feedback_abort_on_failure": True,
    }


def _job_config(spec, batch_id, job, result_root=None):
    parameters = dict(job["parameters"])
    if job["benchmark"] != "r2r-ce":
        order_key = ORDER_KEY[(job["setting"], job["split"])]
        parameters["diagnostics_expected_episodes"] = spec[
            "data_bindings"
        ]["streams"][order_key]["episodes"]
    if job["stage"] == "reverie-test":
        parameters.update(_provider_parameters(spec, batch_id, job, result_root))
    return {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": batch_id,
        "stage": "targeted_gap_{}".format(job["stage"].replace("-", "_")),
        "setting": job["setting"],
        "method": job["method"],
        "search_method": job["method"],
        "episodes": -1,
        "parameters": parameters,
        "selection_provenance": {
            "cell_id": job["cell_id"],
            "candidate_id": job["candidate_id"],
            "development_split": "val_unseen",
            "model_seed": 0,
            "episode_order_seed": 0,
            "fresh_source_restart": True,
            "reported_method_label": job["reported_method_label"],
            "feedback_provider": job["feedback_provider"],
        },
    }


def _validate_config(setting, path, expected_method, diagnostics_path):
    diagnostics_path = Path(diagnostics_path).resolve()
    completed = subprocess.run([
        sys.executable, str(TRANSLATOR), "--setting", setting,
        "--config", str(path), "--diagnostics",
        str(diagnostics_path), "--print-method",
    ], cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
       text=True, check=False)
    if completed.returncode != 0 or completed.stdout.strip() != expected_method:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CampaignError("TTA translator rejected {}/{}: {}".format(
            setting, expected_method, detail))


def _attempt_tag(batch_id, job, attempt):
    stage = job["stage"].replace("-", "")
    suffix = job.get("submission_id") or job["candidate_id"]
    value = "{}-{}-q{:02d}-{}-a{:03d}".format(
        batch_id, stage, int(job["queue_id"]), suffix, attempt
    )
    return _safe_component(value, "attempt run tag")


def _result_root(batch_id, job, run_tag):
    return (
        tuning_batch_root(batch_id) / "stages" / job["stage"]
        / "q{:02d}".format(int(job["queue_id"])) / job["cell_id"]
        / run_tag / job["split"]
    ).resolve()


def _command(job, config_path, result_root, run_tag):
    command = [
        "bash", str(RUNNER), job["setting"], job["split"], str(job["gpu"]),
        "--run-tag", run_tag, "--tta-config", str(config_path),
        "--result-root", str(result_root),
    ]
    if job["benchmark"] == "r2r-ce":
        if job["data_version"] != "v1.2-native":
            raise CampaignError("continuous targeted-gap job is not v1.2-native")
        command.extend(["--ce-data-version", "v1.2-native"])
    if "--order-seed" in command:
        raise CampaignError("canonical seed-0 commands must omit --order-seed")
    return command


def _job_identity(spec_path, batch_id, job):
    payload = {
        "batch_id": batch_id,
        "spec_sha256": sha256(spec_path),
        "git_commit": git_commit(),
        "job": job,
        "model_seed": 0,
        "episode_order_seed": 0,
    }
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def materialize_attempt(spec_path, spec, batch_id, root, job, attempt):
    if job.get("feedback_provider") == "qwen2_vl_2b_v1":
        validate_provider_preflight_binding(
            spec, job.get("feedback_provider_preflight")
        )
    elif job.get("feedback_provider_preflight") is not None:
        raise CampaignError("provider preflight is forbidden for non-LLM jobs")
    attempt_dir = _job_parent(root, job) / "attempt-{:03d}".format(attempt)
    if attempt_dir.exists():
        raise CampaignError("attempt directory already exists: {}".format(attempt_dir))
    retry_history, retry_history_sha256 = _validated_retry_history(
        root, job, attempt
    )
    attempt_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = attempt_dir.with_name(
        attempt_dir.name + ".preparing.{}.{}".format(os.getpid(), time.time_ns())
    )
    staging_dir.mkdir()
    run_tag = _attempt_tag(batch_id, job, attempt)
    result_root = _result_root(batch_id, job, run_tag)
    config = _job_config(spec, batch_id, job, result_root)
    if job["stage"] == "val-seen":
        frozen_path = batch_root(batch_id) / "frozen_configs" / (
            "q{:02d}-{}.json".format(job["queue_id"], job["cell_id"])
        )
        if not frozen_path.is_file() or sha256(frozen_path) != job["frozen_config_sha256"]:
            raise CampaignError("frozen winner config is missing or changed")
        if canonical(read_json(frozen_path)) != canonical(config):
            raise CampaignError("val_seen config differs from FROZEN.json")
        config_path = frozen_path
        validation_config_path = frozen_path
    else:
        config_path = attempt_dir / "resolved_config.json"
        validation_config_path = staging_dir / "resolved_config.json"
        atomic_json(validation_config_path, config)
    _validate_config(
        job["setting"], validation_config_path, job["method"],
        result_root / "tta_diagnostics.json",
    )
    command = _command(job, config_path.resolve(), result_root, run_tag)
    order = order_binding(spec, job["setting"], job["split"], require_dataset=True)
    checkpoint = checkpoint_binding(spec, job["setting"], require_file=True)
    metadata = {
        "schema": JOB_SCHEMA,
        "batch_id": batch_id,
        "spec_path": str(spec_path),
        "spec_sha256": sha256(spec_path),
        "runner_sha256": sha256(SCRIPT_PATH),
        "git_commit": git_commit(),
        "job_identity_sha256": _job_identity(spec_path, batch_id, job),
        "retry_history": retry_history,
        "retry_history_sha256": retry_history_sha256,
        "retry_count": len(retry_history),
        "attempt": attempt,
        "run_tag": run_tag,
        "ordinal": job["ordinal"],
        "stage": job["stage"],
        "split": job["split"],
        "queue_id": job["queue_id"],
        "gpu_slot": job["gpu_slot"],
        "gpu": job["gpu"],
        "cell_id": job["cell_id"],
        "benchmark": job["benchmark"],
        "expected_benchmark": order["benchmark"],
        "setting": job["setting"],
        "model": job["model"],
        "method": job["method"],
        "reported_method_label": job["reported_method_label"],
        "supervision": job["supervision"],
        "feedback_provider": job["feedback_provider"],
        "feedback_provider_preflight": job.get("feedback_provider_preflight"),
        "data_version": job["data_version"],
        "candidate_id": job["candidate_id"],
        "candidate_role": job["candidate_role"],
        "submission_id": job.get("submission_id"),
        "parameters": job["parameters"],
        "runtime_parameters": config["parameters"],
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(validation_config_path),
        "model_seed": 0,
        "episode_order_seed": 0,
        "episode_count": order["episode_count"],
        "order_manifest_path": str(order["path"]),
        "order_manifest_sha256": order["sha256"],
        "order_sha256": order["order_sha256"],
        "dataset_path": str(order["dataset_path"]),
        "dataset_sha256": order["dataset_sha256"],
        "checkpoint_path": str(checkpoint["path"]),
        "checkpoint_sha256": checkpoint["sha256"],
        "result_root": str(result_root),
        "formal_manifest": str(
            FORMAL_ROOT / (run_tag + "-" + job["setting"] + "-" + job["split"]
                           + "-" + job["data_version"] + "-" + job["reported_method_label"].lower())
            / "manifest.json"
        ),
        "command": command,
    }
    atomic_json(staging_dir / "job.json", metadata)
    # Publish only a complete materialization.  A host failure before this
    # rename leaves a forensic ``.preparing.*`` directory that is not mistaken
    # for an executable/retryable attempt.
    os.rename(str(staging_dir), str(attempt_dir))
    return attempt_dir, metadata


def _file_metadata(path, hash_field="sha256"):
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, hash_field: sha256(path)}


def _hardware(gpu):
    value = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cuda_visible_devices": str(gpu),
    }
    completed = subprocess.run([
        "nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader,nounits",
        "--id={}".format(gpu),
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or len(lines) != 1:
        detail = completed.stderr.strip() or completed.stdout.strip() or "nvidia-smi failed"
        raise CampaignError("GPU hardware probe failed closed: {}".format(detail))
    fields = [item.strip() for item in lines[0].split(",")]
    try:
        memory_mib = int(fields[2])
    except (IndexError, TypeError, ValueError):
        raise CampaignError("GPU hardware probe returned malformed output")
    if len(fields) != 3 or not fields[0] or not fields[1] or memory_mib <= 0:
        raise CampaignError("GPU hardware probe returned incomplete identity")
    value.update({"gpu_name": fields[0], "gpu_uuid": fields[1],
                  "gpu_memory_mib": memory_mib})
    return value


def _create_run_manifest(metadata):
    from tools.run_manifest_identity import immutable_identity_sha256
    path = Path(metadata["formal_manifest"])
    if path.exists():
        raise CampaignError("formal run manifest already exists: {}".format(path))
    _, spec = load_spec(metadata["spec_path"])
    asset_manifest = spec["data_bindings"]["asset_manifest"]
    pinned = {
        "experiment_spec": _file_metadata(metadata["spec_path"]),
        "assets": _file_metadata(repo_file(asset_manifest["path"], "asset manifest", True)),
        "episode_order": _file_metadata(metadata["order_manifest_path"]),
        "environment": _file_metadata(ENVIRONMENT_MANIFEST),
    }
    checkpoint_path = Path(metadata["checkpoint_path"])
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "size": checkpoint_path.stat().st_size,
        "sha256": metadata["checkpoint_sha256"],
    }
    dataset_path = Path(metadata["dataset_path"])
    dataset = {
        "path": str(dataset_path.resolve()),
        "size": dataset_path.stat().st_size,
        "index_sha256": metadata["dataset_sha256"],
    }
    dataset.update({
        "version": metadata["data_version"],
        "stream_order_sha256": metadata["order_sha256"],
        "stream_content_sha256": metadata["dataset_sha256"],
    })
    auxiliary = []
    config_record = _file_metadata(metadata["config_path"])
    config_record["name"] = "resolved_config"
    auxiliary.append(config_record)
    source_stats = metadata["runtime_parameters"].get("source_stats_path")
    if source_stats:
        record = _file_metadata(source_stats)
        record["name"] = "idea_source_statistics"
        auxiliary.append(record)
    if metadata["feedback_provider"] == "qwen2_vl_2b_v1":
        provider_runtime = spec["hidden_test_transfer"]["provider"]["runtime"]
        record = _file_metadata(repo_file(
            provider_runtime["contract"]["path"], "provider contract", True
        ))
        record["name"] = "feedback_provider_contract"
        auxiliary.append(record)
        preflight = validate_provider_preflight_binding(
            spec,
            metadata.get("feedback_provider_preflight"),
        )
        record = _file_metadata(preflight["path"])
        if (
            record["size"] != preflight["size"]
            or record["sha256"] != preflight["sha256"]
        ):
            raise CampaignError("FeedTTA-LLM PRECHECK changed before launch")
        record["name"] = "feedback_provider_preflight"
        auxiliary.append(record)
    manifest = {
        "run_id": path.parent.name,
        "task": "vln",
        "benchmark": metadata["expected_benchmark"],
        "model": metadata["model"],
        "method": metadata["reported_method_label"].lower(),
        "run_tag": metadata["run_tag"],
        "source_setting": "{}:{}:{}:{}".format(
            metadata["setting"], metadata["split"], metadata["data_version"],
            metadata["reported_method_label"].lower(),
        ),
        "seed": 0,
        "git_commit": metadata["git_commit"] if "git_commit" in metadata else git_commit(),
        "config": metadata["config_path"],
        "config_overrides": list(metadata["command"]),
        "checkpoint": checkpoint,
        "auxiliary_checkpoints": auxiliary,
        "dataset": dataset,
        "pinned_manifests": pinned,
        "hardware": _hardware(metadata["gpu"]),
        "started_at": utc_now(),
        "completed_at": None,
        "status": "running",
        "exit_code": None,
        "campaign": {
            "batch_id": metadata["batch_id"],
            "stage": metadata["stage"],
            "cell_id": metadata["cell_id"],
            "candidate_id": metadata["candidate_id"],
            "queue_id": metadata["queue_id"],
            "gpu_slot": metadata["gpu_slot"],
            "episode_order_seed": 0,
            "config_sha256": metadata["config_sha256"],
            "job_identity_sha256": metadata["job_identity_sha256"],
            "supervision": metadata["supervision"],
            "feedback_provider": metadata["feedback_provider"],
            "feedback_provider_preflight": metadata.get(
                "feedback_provider_preflight"
            ),
            "attempt": metadata["attempt"],
            "retry_count": metadata["retry_count"],
            "retry_history_sha256": metadata["retry_history_sha256"],
        },
    }
    manifest["immutable_identity_sha256"] = immutable_identity_sha256(manifest)
    atomic_json(path, manifest)
    return path


def _finish_run_manifest(path, exit_code, artifacts=()):
    manifest = read_json(path)
    records = None
    if exit_code == 0:
        seen = set()
        records = []
        for name, item in artifacts:
            item = Path(item).resolve()
            if name in seen or not item.is_file():
                raise CampaignError("invalid or duplicate result artifact {}".format(name))
            seen.add(name)
            record = _file_metadata(item)
            record["name"] = name
            records.append(record)
        if not records:
            raise CampaignError("successful run manifest requires result artifacts")
    if manifest.get("status") in ("completed", "failed"):
        expected_status = "completed" if exit_code == 0 else "failed"
        if (
            manifest.get("status") != expected_status
            or manifest.get("exit_code") != int(exit_code)
            or (exit_code == 0 and manifest.get("result_artifacts") != records)
        ):
            raise CampaignError("finished run manifest or artifacts changed")
        return
    if manifest.get("status") != "running" or manifest.get("completed_at") is not None:
        raise CampaignError("formal run manifest has an invalid lifecycle state")
    manifest["completed_at"] = utc_now()
    manifest["exit_code"] = int(exit_code)
    manifest["status"] = "completed" if exit_code == 0 else "failed"
    if records is not None:
        manifest["result_artifacts"] = records
    atomic_json(path, manifest)


METRIC_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)"
)


def _metric_artifact(metadata):
    result_root = Path(metadata["result_root"])
    split = metadata["split"]
    matches = []
    if metadata["benchmark"] == "r2r-ce":
        for path in result_root.rglob("stats_*_{}.json".format(split)):
            if path.name.startswith("stats_ep_"):
                continue
            document = read_json(path)
            if finite_number(document.get("success")) and finite_number(document.get("spl")):
                metrics = {
                    key.upper(): float(value) for key, value in document.items()
                    if finite_number(value)
                }
                for key in ("SUCCESS", "ORACLE_SUCCESS", "SPL", "NDTW", "SDTW"):
                    if key in metrics:
                        metrics[key] *= 100.0
                metrics["SR"] = metrics["SUCCESS"]
                if "ORACLE_SUCCESS" in metrics:
                    metrics["OSR"] = metrics["ORACLE_SUCCESS"]
                matches.append((path, metrics))
    else:
        for path in result_root.rglob("valid.txt"):
            lines = [
                line for line in path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines() if "Env name: {}".format(split) in line
            ]
            if len(lines) == 1:
                metrics = {
                    key.upper(): float(value)
                    for key, value in METRIC_RE.findall(lines[0])
                }
                matches.append((path, metrics))
    if len(matches) != 1:
        raise CampaignError("expected one aggregate metric artifact for {}; found {}".format(
            metadata["run_tag"], len(matches)))
    needed = REQUIRED_METRICS[metadata["benchmark"]]
    if any(not finite_number(matches[0][1].get(key)) for key in needed):
        raise CampaignError("aggregate metrics are incomplete for {}".format(metadata["run_tag"]))
    return matches[0]


def _validate_prediction(metadata, attempt_dir):
    result_root = Path(metadata["result_root"])
    pattern = "submit_{}*.json".format(metadata["split"])
    matches = sorted(result_root.rglob(pattern))
    if len(matches) != 1:
        raise CampaignError("expected exactly one full prediction artifact for {}; found {}".format(
            metadata["run_tag"], len(matches)))
    path = matches[0]
    try:
        predictions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError("cannot read prediction artifact: {}".format(error))
    if not isinstance(predictions, list):
        raise CampaignError("prediction artifact must be a list")
    order = read_json(metadata["order_manifest_path"])
    expected_ids = [str(item["episode_id"]) for item in order["episodes"]]
    actual_ids = [str(item.get("instr_id")) for item in predictions
                  if isinstance(item, dict)]
    if len(actual_ids) != len(predictions) or actual_ids != expected_ids:
        raise CampaignError("prediction IDs/order do not match the full canonical stream")
    if any(not isinstance(item.get("trajectory"), list) or not item["trajectory"]
           for item in predictions):
        raise CampaignError("prediction artifact contains an empty trajectory")
    evidence = {
        "schema": "navtta.vln_trajectory_evidence.v1",
        "run_tag": metadata["run_tag"],
        "setting": metadata["setting"],
        "split": metadata["split"],
        "episode_count": len(actual_ids),
        "episode_ids_sha256": hashlib.sha256(
            canonical(actual_ids).encode("utf-8")
        ).hexdigest(),
        "order_sha256": metadata["order_sha256"],
        "order_manifest_sha256": metadata["order_manifest_sha256"],
        "prediction_path": str(path.resolve()),
        "prediction_sha256": sha256(path),
        "contains_per_episode_metrics": False,
    }
    evidence_path = Path(attempt_dir) / "trajectory_evidence.json"
    atomic_json(evidence_path, evidence)
    return path, evidence_path


def _validate_per_episode_document(path, metadata, aggregate_metrics=None):
    document = read_json(path)
    if document.get("schema") != PER_EPISODE_SCHEMA:
        raise CampaignError("per-episode metric schema mismatch")
    if (
        document.get("setting") != metadata["setting"]
        or document.get("split") != metadata["split"]
        or document.get("run_tag") != metadata["run_tag"]
        or document.get("episode_count") != metadata["episode_count"]
        or document.get("order_sha256") != metadata["order_sha256"]
        or document.get("episode_order_manifest_sha256")
        != metadata["order_manifest_sha256"]
    ):
        raise CampaignError("per-episode metric identity mismatch")
    rows = document.get("episodes")
    if not isinstance(rows, list) or len(rows) != metadata["episode_count"]:
        raise CampaignError("per-episode metric row count mismatch")
    order_rows = read_json(metadata["order_manifest_path"])["episodes"]
    expected = [str(item["episode_id"]) for item in order_rows]
    actual = [str(item.get("episode_id")) for item in rows if isinstance(item, dict)]
    if actual != expected:
        raise CampaignError("per-episode metric IDs/order mismatch")
    metric_keys = None
    for ordinal, (row, order_row) in enumerate(zip(rows, order_rows)):
        if row.get("ordinal") != ordinal:
            raise CampaignError("per-episode metric ordinal mismatch")
        if "scene_id" in row and row.get("scene_id") != order_row.get("scene_id"):
            raise CampaignError("per-episode metric scene key mismatch")
        values = row.get("metrics")
        if not isinstance(values, dict) or not values or any(
            not finite_number(value) for value in values.values()
        ):
            raise CampaignError("per-episode metrics contain invalid values")
        current_keys = tuple(sorted(str(key).lower() for key in values))
        if len(set(current_keys)) != len(values):
            raise CampaignError("per-episode metric keys are ambiguous")
        if metric_keys is None:
            metric_keys = current_keys
        elif current_keys != metric_keys:
            raise CampaignError("per-episode metric keys are inconsistent")
    source = document.get("source")
    source_artifact = document.get("source_artifact")
    if metadata["benchmark"] == "r2r-ce":
        if (
            source != "native_continuous_evaluator_stats"
            or not isinstance(source_artifact, dict)
        ):
            raise CampaignError("continuous per-episode source artifact is missing")
        source_path = Path(str(source_artifact.get("path", ""))).resolve()
        try:
            source_path.relative_to(Path(metadata["result_root"]).resolve())
        except ValueError:
            raise CampaignError("continuous per-episode source escapes result root")
        if (
            not source_path.is_file()
            or not source_path.name.startswith("stats_ep_")
            or source_path.stat().st_size != source_artifact.get("size")
            or sha256(source_path) != source_artifact.get("sha256")
        ):
            raise CampaignError("continuous per-episode source artifact changed")
    elif source != "native_discrete_evaluator_return":
        raise CampaignError("discrete per-episode source identity mismatch")
    if aggregate_metrics is not None:
        required = REQUIRED_METRICS[metadata["benchmark"]]
        missing = [
            name for name in required
            if metric_keys is None
            or PER_EPISODE_METRIC_KEY[name] not in metric_keys
            or not finite_number(aggregate_metrics.get(name))
        ]
        if missing:
            raise CampaignError(
                "per-episode/aggregate metrics are missing: {}"
                .format(", ".join(missing))
            )
        tolerance = 0.005001 if metadata["benchmark"] != "r2r-ce" else 1e-6
        for aggregate_name in required:
            episode_name = PER_EPISODE_METRIC_KEY[aggregate_name]
            values = []
            for row in rows:
                by_name = {
                    str(key).lower(): value
                    for key, value in row["metrics"].items()
                }
                values.append(float(by_name[episode_name]))
            multiplier = 1.0 if aggregate_name == "COLLISIONS" else 100.0
            recomputed = multiplier * sum(values) / len(values)
            if not math.isclose(
                recomputed, float(aggregate_metrics[aggregate_name]),
                rel_tol=0.0, abs_tol=tolerance,
            ):
                raise CampaignError(
                    "per-episode {} disagrees with aggregate"
                    .format(aggregate_name)
                )
    return path


def _per_episode_evidence(metadata, attempt_dir, aggregate_metrics=None):
    """Return paired evidence, or an explicit non-fabricated missing gate."""
    result_root = Path(metadata["result_root"])
    existing = sorted(result_root.rglob("per_episode_metrics.json"))
    if len(existing) > 1:
        raise CampaignError("multiple per_episode_metrics.json artifacts found")
    if existing:
        return _validate_per_episode_document(
            existing[0], metadata, aggregate_metrics
        ), True
    if metadata["benchmark"] != "r2r-ce":
        return None, False

    raw_matches = sorted(result_root.rglob(
        "stats_ep_*_{}_r0_w1.json".format(metadata["split"])
    ))
    if len(raw_matches) != 1:
        raise CampaignError("continuous run lacks one per-episode stats artifact")
    raw = read_json(raw_matches[0])
    order = read_json(metadata["order_manifest_path"])
    expected_ids = [str(item["episode_id"]) for item in order["episodes"]]
    if list(map(str, raw.keys())) != expected_ids:
        raise CampaignError("continuous per-episode stats order mismatch")
    rows = []
    for ordinal, episode_id in enumerate(expected_ids):
        metrics = raw[episode_id] if episode_id in raw else raw.get(int(episode_id))
        if not isinstance(metrics, dict) or not metrics or any(
            not finite_number(value) for value in metrics.values()
        ):
            raise CampaignError("continuous per-episode stats are invalid")
        rows.append({"ordinal": ordinal, "episode_id": episode_id,
                     "metrics": metrics})
    document = {
        "schema": PER_EPISODE_SCHEMA,
        "setting": metadata["setting"],
        "split": metadata["split"],
        "run_tag": metadata["run_tag"],
        "episode_count": metadata["episode_count"],
        "order_sha256": metadata["order_sha256"],
        "episode_order_manifest_sha256": metadata["order_manifest_sha256"],
        "source": "native_continuous_evaluator_stats",
        "source_artifact": _file_metadata(raw_matches[0]),
        "episodes": rows,
    }
    path = Path(attempt_dir) / "per_episode_metrics.json"
    atomic_json(path, document)
    return _validate_per_episode_document(path, metadata, aggregate_metrics), True


def _nonnegative_counter(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CampaignError("TTA {} counter is invalid".format(label))
    return value


def _diagnostic_accounting(metadata, diagnostics, adapter):
    method = metadata["method"]
    names = adapter.get("adapted_parameter_names", diagnostics.get("trainable_prefixes"))
    if (
        not isinstance(names, list) or not names
        or any(not isinstance(item, str) or not item for item in names)
        or len(set(names)) != len(names)
    ):
        raise CampaignError("TTA trainable tensor/prefix evidence is empty or invalid")
    parameter_count = adapter.get("adapted_parameter_count")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
    ):
        raise CampaignError("TTA adapted-parameter count is missing or empty")

    explicit_attempted = diagnostics.get(
        "attempted_updates", adapter.get("attempted_updates")
    )
    explicit_backward = diagnostics.get(
        "backward_passes", adapter.get("backward_passes")
    )
    raw_updates = adapter.get("updates")
    if method == "tent":
        attempted = adapter.get("action_steps") if explicit_attempted is None else explicit_attempted
        backward = attempted if explicit_backward is None else explicit_backward
        accepted = raw_updates
        consistent = attempted == backward == accepted
    elif method == "fstta":
        slow_attempts = adapter.get("slow_attempts")
        fast_attempts = adapter.get("fast_optimizer_attempts")
        slow_updates = adapter.get("slow_updates")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (slow_attempts, fast_attempts, slow_updates)
        ):
            raise CampaignError("FSTTA attempt/update diagnostics are incomplete")
        derived_attempted = fast_attempts + slow_attempts
        attempted = derived_attempted if explicit_attempted is None else explicit_attempted
        backward = (adapter.get("action_steps", 0)
                    if explicit_backward is None else explicit_backward)
        accepted = raw_updates + slow_updates if isinstance(raw_updates, int) else raw_updates
        consistent = attempted == derived_attempted and accepted <= attempted and backward > 0
    elif method == "eam":
        attempted = adapter.get("update_attempts") if explicit_attempted is None else explicit_attempted
        skipped = adapter.get("updates_skipped_no_reliable")
        backward = raw_updates if explicit_backward is None else explicit_backward
        accepted = raw_updates
        consistent = (
            isinstance(skipped, int) and attempted == accepted + skipped
            and backward == accepted
        )
    elif method == "feedtta":
        attempted = (adapter.get("episode_end_optimizer_attempts")
                     if explicit_attempted is None else explicit_attempted)
        backward = attempted if explicit_backward is None else explicit_backward
        accepted = raw_updates
        consistent = attempted == backward == accepted
    elif method == "atena":
        attempted = (adapter.get("query_gate_evaluations", adapter.get("episodes"))
                     if explicit_attempted is None else explicit_attempted)
        backward = attempted if explicit_backward is None else explicit_backward
        accepted = raw_updates
        consistent = (
            attempted == backward == accepted
            and adapter.get("queries", 0) + adapter.get("self_label_episodes", 0)
            == adapter.get("episodes")
        )
    elif method == "idea":
        new_domains = adapter.get("new_domain_steps")
        opt_steps = adapter.get("opt_steps")
        if (
            isinstance(new_domains, bool) or not isinstance(new_domains, int)
            or isinstance(opt_steps, bool) or not isinstance(opt_steps, int)
            or new_domains <= 0 or opt_steps <= 0
        ):
            raise CampaignError("IDEA optimization counters are incomplete")
        derived = new_domains * opt_steps
        prompt_attempts = _nonnegative_counter(
            adapter.get("prompt_optimizer_attempts"),
            "IDEA prompt_optimizer_attempts",
        )
        prompt_updates = _nonnegative_counter(
            adapter.get("prompt_optimizer_updates"),
            "IDEA prompt_optimizer_updates",
        )
        prompt_optimizations = _nonnegative_counter(
            adapter.get("prompt_optimizations"),
            "IDEA prompt_optimizations",
        )
        prompt_drift = adapter.get("relative_param_drift")
        max_prompt_drift = adapter.get("max_prompt_relative_drift")
        attempted = derived if explicit_attempted is None else explicit_attempted
        backward = derived if explicit_backward is None else explicit_backward
        accepted = diagnostics.get("accepted_updates", adapter.get("accepted_updates", derived))
        consistent = (
            attempted == backward == accepted == raw_updates == derived
            and prompt_attempts == derived
            and prompt_updates == derived
            and prompt_optimizations == new_domains
            and finite_number(prompt_drift)
            and float(prompt_drift) > 0.0
            and finite_number(max_prompt_drift)
            and float(max_prompt_drift) > 0.0
            and float(prompt_drift) <= float(max_prompt_drift) + 1e-15
        )
    else:
        raise CampaignError("unsupported adaptive method diagnostics")
    attempted = _nonnegative_counter(attempted, "attempted_updates")
    backward = _nonnegative_counter(backward, "backward_passes")
    accepted = _nonnegative_counter(accepted, "accepted_updates")
    if not consistent or not (attempted > 0 and backward > 0 and accepted > 0):
        raise CampaignError("TTA update accounting is inconsistent or no-op")
    return names, attempted, backward, accepted


def _diagnostic_value(container, key, expected, label):
    observed = container.get(key)
    if finite_number(expected) and finite_number(observed):
        matches = math.isclose(
            float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-15
        )
    else:
        matches = observed == expected
    if not matches:
        raise CampaignError(
            "TTA diagnostic {} mismatch: expected {!r}, got {!r}"
            .format(label, expected, observed)
        )


def _validate_method_diagnostic_contract(metadata, diagnostics, adapter, names):
    parameters = metadata.get("runtime_parameters")
    if not isinstance(parameters, dict):
        raise CampaignError("TTA runtime parameter evidence is missing")
    method = metadata["method"]
    if parameters.get("action_selection") != "argmax":
        raise CampaignError("formal targeted-gap action selection must be argmax")
    if diagnostics.get("action_selection") != "target_native_argmax":
        raise CampaignError("TTA diagnostic action protocol mismatch")
    if (
        diagnostics.get("batch_size") != 1
        or diagnostics.get("audit_zero_update") is not False
        or diagnostics.get("audit_control") is not False
        or diagnostics.get("stream") != metadata["split"]
    ):
        raise CampaignError("TTA diagnostic execution contract mismatch")
    expected_diagnostics = parameters.get("diagnostics_expected_episodes")
    if expected_diagnostics is not None and (
        expected_diagnostics != metadata["episode_count"]
        or diagnostics.get("diagnostics_expected_episodes")
        != metadata["episode_count"]
    ):
        raise CampaignError("TTA diagnostics full-horizon binding mismatch")

    prefixes = adapter.get("trainable_prefixes", diagnostics.get("trainable_prefixes"))
    if isinstance(prefixes, list) and prefixes:
        if any(
            not any(name == prefix or name.startswith(prefix + ".")
                    for prefix in prefixes)
            for name in names
        ):
            raise CampaignError("TTA adapted parameters escape the declared scope")

    pairs = []
    if method == "tent":
        pairs = [(adapter, "current_lr", parameters["lr"]),
                 (diagnostics, "tent_canonical_update_interval",
                  parameters["update_interval"] == 1)]
    elif method == "fstta":
        pairs = [
            (adapter, "fast_lr", parameters["lr_fast"]),
            (adapter, "slow_lr", parameters["lr_slow"]),
            (adapter, "fast_window", parameters["m"]),
            (adapter, "slow_window", parameters["n"]),
            (adapter, "q", parameters["q"]),
            (adapter, "fast_grad_mode", parameters["fast_grad_mode"]),
            (adapter, "slow_optimizer", parameters["slow_optimizer"]),
            (adapter, "reset_var_hist_each_episode",
             parameters["reset_var_hist_each_episode"]),
        ]
    elif method == "eam":
        pairs = [
            (adapter, "current_lr", parameters["lr"]),
            (adapter, "confidence_scale", parameters["confidence_scale"]),
            (adapter, "memory_size_steps", parameters["memory_size"]),
            (adapter, "batch_size_steps", parameters["batch_size"]),
            (adapter, "update_interval", parameters["update_interval"]),
            (adapter, "optimizer", parameters["optimizer"]),
            (adapter, "param_scope", "module_prefixes"),
        ]
    elif method == "feedtta":
        pairs = [
            (adapter, "current_lr", parameters["lr"]),
            (adapter, "reversal_probability", parameters["p"]),
            (adapter, "reversal_scale", parameters["alpha"]),
            (adapter, "sgr_seed", parameters["sgr_seed"]),
            (adapter, "sgr_mode", parameters["sgr_mode"]),
            (adapter, "gamma", parameters["gamma"]),
            (adapter, "normalize_gradient", parameters["normalize_gradient"]),
            (adapter, "action_selection", "argmax"),
            (diagnostics, "feedtta_scope_profile", parameters["scope_profile"]),
        ]
    elif method == "atena":
        pairs = [
            (adapter, "lr_query", parameters["lr_query"]),
            (adapter, "lr_self", parameters["lr_self"]),
            (adapter, "query_threshold", parameters["query_threshold"]),
            (adapter, "mix_lambda", parameters["mix_lambda"]),
            (adapter, "self_loss_weight", parameters["self_loss_weight"]),
            (adapter, "optimizer", parameters["optimizer"]),
            (adapter, "action_selection", "argmax"),
            (diagnostics, "atena_update_scope", parameters["update_scope"]),
            (diagnostics, "atena_exact_replay_within_declared_scope", True),
            (diagnostics, "atena_optimizer_scope_matches_reachable", True),
        ]
    elif method == "idea":
        if names != ["external_soft_prompt"]:
            raise CampaignError(
                "IDEA adapted parameters must be exactly external_soft_prompt"
            )
        pairs = [
            (adapter, "current_lr", parameters["lr"]),
            (adapter, "prompt_length", parameters["prompt_length"]),
            (adapter, "library_capacity", parameters["k_max"]),
            (adapter, "lambda", parameters["lambda"]),
            (adapter, "tau", parameters["tau"]),
            (adapter, "opt_steps", parameters["opt_steps"]),
            (adapter, "use_fisher", parameters["use_fisher"]),
            (adapter, "trains_base_policy", False),
            (adapter, "base_parameter_grads_none", True),
            (adapter, "base_parameter_integrity_schema",
             "navtta.idea.base_parameter_integrity.v1"),
            (adapter, "base_parameter_integrity_complete", True),
            (adapter, "base_parameter_unchanged", True),
            (adapter, "base_parameter_name_set_unchanged", True),
            (adapter, "base_parameter_content_hash_match", True),
            (adapter, "base_parameter_added_names", []),
            (adapter, "base_parameter_removed_names", []),
            (adapter, "base_parameter_content_modified_names", []),
            (adapter, "base_parameter_versions_unchanged", True),
            (adapter, "base_parameter_version_changed_names", []),
            (adapter, "base_parameter_modified_names", []),
        ]
    for container, key, expected in pairs:
        _diagnostic_value(container, key, expected, "{}.{}".format(method, key))
    if method == "idea":
        before_count = adapter.get("base_parameter_name_count_before")
        after_count = adapter.get("base_parameter_name_count_after")
        if (
            isinstance(before_count, bool) or not isinstance(before_count, int)
            or before_count <= 0 or after_count != before_count
        ):
            raise CampaignError("IDEA base parameter name counts are invalid")
        for before_key, after_key, label in (
            (
                "base_parameter_name_set_before_sha256",
                "base_parameter_name_set_after_sha256", "name-set",
            ),
            (
                "base_parameter_content_before_sha256",
                "base_parameter_content_after_sha256", "content",
            ),
        ):
            before_digest = adapter.get(before_key)
            after_digest = adapter.get(after_key)
            if (
                not valid_sha256(before_digest)
                or not valid_sha256(after_digest)
                or before_digest != after_digest
            ):
                raise CampaignError(
                    "IDEA base parameter {} digest mismatch".format(label)
                )


def _validate_diagnostics(metadata, path):
    diagnostics = read_json(path)
    expected_schema = (
        "navtta.vln_ce_tta.v1"
        if metadata["benchmark"] == "r2r-ce"
        else "navtta.vln_discrete_tta.v1"
    )
    if diagnostics.get("schema") != expected_schema:
        raise CampaignError("TTA diagnostics schema mismatch")
    if diagnostics.get("method") != metadata["method"]:
        raise CampaignError("TTA diagnostics method mismatch")
    if diagnostics.get("episode_count") != metadata["episode_count"]:
        raise CampaignError("TTA diagnostics episode count mismatch")
    adapter = diagnostics.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("episodes") != metadata["episode_count"]:
        raise CampaignError("TTA adapter episode accounting mismatch")
    drift = adapter.get("relative_param_drift")
    if (
        not finite_number(drift) or float(drift) < 0
        or (metadata["method"] != "idea" and float(drift) == 0)
    ):
        raise CampaignError("TTA parameter drift is invalid")
    names, attempted, backward, updates = _diagnostic_accounting(
        metadata, diagnostics, adapter
    )
    _validate_method_diagnostic_contract(metadata, diagnostics, adapter, names)

    provider_diag = diagnostics.get("pseudo_feedback")
    if metadata["feedback_provider"] == "qwen2_vl_2b_v1":
        if (
            diagnostics.get("supervision") != "external_mllm_pseudo_feedback"
            or diagnostics.get("feedback_provider") != "qwen2_vl_2b_v1"
            or diagnostics.get("reported_method_label") != "FeedTTA-LLM"
            or not isinstance(provider_diag, dict)
        ):
            raise CampaignError("FeedTTA-LLM diagnostics identity mismatch")
        runtime_parameters = metadata["runtime_parameters"]
        exact_provider = {
            "provider_id": "qwen2_vl_2b_v1",
            "model_id": runtime_parameters["llm_feedback_model_id"],
            "revision": runtime_parameters["llm_feedback_revision"],
            "weights_sha256": runtime_parameters["llm_feedback_weights_sha256"],
            "bundle_sha256": runtime_parameters["llm_feedback_bundle_sha256"],
            "prompt_bundle_sha256": runtime_parameters["llm_feedback_prompt_sha256"],
            "reported_method_label": "FeedTTA-LLM",
            "supervision": "external_mllm_pseudo_feedback",
        }
        for key, expected in exact_provider.items():
            if provider_diag.get(key) != expected:
                raise CampaignError("FeedTTA-LLM diagnostic {} mismatch".format(key))
        count = metadata["episode_count"]
        if (
            provider_diag.get("query_episodes") != count
            or provider_diag.get("pseudo_labels") != count
            or provider_diag.get("failed_closed_episodes") != 0
            or diagnostics.get("failed_closed_feedback_episodes") != 0
        ):
            raise CampaignError("FeedTTA-LLM did not produce one valid label per episode")
        for key in ("transcript_sha256", "weights_sha256", "bundle_sha256",
                    "prompt_bundle_sha256"):
            if not valid_sha256(provider_diag.get(key)):
                raise CampaignError("FeedTTA-LLM {} is invalid".format(key))
        transcript = Path(str(provider_diag.get("transcript_path", ""))).resolve()
        expected_transcript = Path(metadata["result_root"]) / "llm_feedback_transcript.ndjson"
        if transcript != expected_transcript.resolve() or not transcript.is_file():
            raise CampaignError("FeedTTA-LLM transcript path is invalid")
        _validate_feedback_transcript(transcript, provider_diag, metadata)
    else:
        if metadata["supervision"] == "unsupervised":
            expected = "none" if metadata["benchmark"] == "r2r-ce" else "unsupervised"
            observed = diagnostics.get(
                "feedback_supervision" if metadata["benchmark"] == "r2r-ce" else "supervision"
            )
            if observed != expected:
                raise CampaignError("unsupervised diagnostics label mismatch")
        elif metadata["supervision"] == "feedback_supervised":
            expected = (
                "binary_episode_success" if metadata["benchmark"] == "r2r-ce"
                else "binary_navigation_success_feedback"
            )
            observed = diagnostics.get(
                "feedback_supervision" if metadata["benchmark"] == "r2r-ce" else "supervision"
            )
            if observed != expected:
                raise CampaignError("feedback-supervised diagnostics label mismatch")
    return diagnostics, {
        "relative_param_drift": float(drift),
        "trainable_names": names,
        "attempted_updates": attempted,
        "backward_passes": backward,
        "updates": updates,
    }


def _prediction_viewpoints(prediction):
    trajectory = prediction.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise CampaignError("FeedTTA-LLM prediction trajectory is missing")
    viewpoints = []
    for step in trajectory:
        if isinstance(step, str) and step:
            viewpoints.append(step)
        elif (
            isinstance(step, (list, tuple)) and step
            and isinstance(step[0], str) and step[0]
        ):
            viewpoints.append(step[0])
        else:
            raise CampaignError("FeedTTA-LLM prediction trajectory is malformed")
    return viewpoints


def _validate_feedback_cache_record(row, metadata):
    """Rebind a transcript event to its immutable on-disk cache record."""
    cache_root = Path(str(
        metadata.get("runtime_parameters", {}).get("llm_feedback_cache_dir", "")
    )).resolve()
    cache_key = row.get("cache_key")
    if not valid_sha256(cache_key):
        raise CampaignError("FeedTTA-LLM transcript cache key is invalid")
    cache_path = (cache_root / cache_key[:2] / (cache_key + ".json")).resolve()
    try:
        cache_path.relative_to(cache_root)
    except ValueError:
        raise CampaignError("FeedTTA-LLM cache record escapes cache root")
    if not cache_path.is_file():
        raise CampaignError("FeedTTA-LLM cache record is missing")
    record = read_json(cache_path)
    actual_record_sha256 = hashlib.sha256(
        canonical(record).encode("utf-8")
    ).hexdigest()
    if actual_record_sha256 != row.get("record_sha256"):
        raise CampaignError("FeedTTA-LLM cache record digest mismatch")
    stage1 = record.get("stage1")
    stage2 = record.get("stage2")
    panorama = record.get("panorama")
    if not all(isinstance(value, dict) for value in (stage1, stage2, panorama)):
        raise CampaignError("FeedTTA-LLM cache record is incomplete")
    expected = {
        "cache_key": cache_key,
        "input_sha256": cache_key,
        "episode_id": row.get("episode_id"),
        "scan_id": row.get("scan_id"),
        "endpoint_viewpoint_id": row.get("endpoint_viewpoint_id"),
        "trajectory_sha256": row.get("trajectory_sha256"),
        "parsed_label": row.get("parsed_label"),
        "token_count": row.get("token_count"),
        "cost": row.get("cost"),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise CampaignError(
                "FeedTTA-LLM cache/transcript {} mismatch".format(key)
            )
    if (
        panorama.get("panorama_png_sha256") != row.get("panorama_png_sha256")
        or [stage1.get("request_sha256"), stage2.get("request_sha256")]
        != row.get("request_sha256_values")
        or [stage1.get("response_sha256"), stage2.get("response_sha256")]
        != row.get("response_sha256_values")
        or [stage1.get("latency_seconds"), stage2.get("latency_seconds")]
        != row.get("latency_seconds_values")
        or stage2.get("output") != row.get("parsed_label")
    ):
        raise CampaignError("FeedTTA-LLM cache/transcript stage binding mismatch")
    return cache_path


def _validate_feedback_transcript(path, diagnostics, metadata):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise CampaignError("FeedTTA-LLM transcript contains a blank record")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise CampaignError("invalid FeedTTA-LLM transcript: {}".format(error))
    if len(rows) != metadata["episode_count"]:
        raise CampaignError("FeedTTA-LLM transcript event count mismatch")
    previous = "0" * 64
    order_ids = [
        str(item["episode_id"])
        for item in read_json(metadata["order_manifest_path"])["episodes"]
    ]
    order_rows = read_json(metadata["order_manifest_path"])["episodes"]
    prediction_matches = sorted(Path(metadata["result_root"]).rglob(
        "submit_{}*.json".format(metadata["split"])
    ))
    if len(prediction_matches) != 1:
        raise CampaignError("FeedTTA-LLM transcript lacks one bound prediction")
    try:
        predictions = json.loads(prediction_matches[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError("cannot read FeedTTA-LLM prediction: {}".format(error))
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise CampaignError("FeedTTA-LLM prediction/transcript count mismatch")
    _, spec = load_spec(metadata["spec_path"])
    provider = spec["hidden_test_transfer"]["provider"]
    prompt_hashes = [item["prompt_sha256"] for item in provider["pipeline"]]
    expected_identity = {
        "provider_id": "qwen2_vl_2b_v1",
        "model_id": provider["model_id"],
        "revision": provider["revision"],
        "weights_sha256": provider["weights_sha256"],
        "bundle_sha256": metadata["runtime_parameters"][
            "llm_feedback_bundle_sha256"
        ],
    }
    prompt_tokens = completion_tokens = cache_hits = cache_misses = 0
    total_cost = 0.0
    positive = negative = 0
    for index, (row, episode_id, order_row, prediction) in enumerate(zip(
        rows, order_ids, order_rows, predictions
    )):
        if (
            not isinstance(row, dict)
            or row.get("schema") != "navtta.reverie_llm_feedback_transcript.v1"
            or row.get("sequence") != index
            or row.get("previous_event_sha256") != previous
            or row.get("status") != "label"
            or str(row.get("episode_id")) != episode_id
        ):
            raise CampaignError("FeedTTA-LLM transcript order/identity mismatch")
        for key, expected in expected_identity.items():
            if row.get(key) != expected:
                raise CampaignError("FeedTTA-LLM transcript {} mismatch".format(key))
        if row.get("prompt_sha256_values") != prompt_hashes:
            raise CampaignError("FeedTTA-LLM transcript prompt identity mismatch")
        for key in ("request_sha256_values", "response_sha256_values"):
            values = row.get(key)
            if (
                not isinstance(values, list) or len(values) != 2
                or any(not valid_sha256(value) for value in values)
            ):
                raise CampaignError("FeedTTA-LLM transcript {} is invalid".format(key))
        latencies = row.get("latency_seconds_values")
        if (
            not isinstance(latencies, list) or len(latencies) != 2
            or any(not finite_number(value) or float(value) < 0 for value in latencies)
        ):
            raise CampaignError("FeedTTA-LLM transcript latency is invalid")
        label = row.get("parsed_label")
        if label not in ("Yes", "No"):
            raise CampaignError("FeedTTA-LLM transcript label is invalid")
        positive += int(label == "Yes")
        negative += int(label == "No")
        if not valid_sha256(row.get("cache_key")) or not isinstance(
            row.get("cache_hit"), bool
        ) or not valid_sha256(row.get("record_sha256")):
            raise CampaignError("FeedTTA-LLM transcript cache identity is invalid")
        _validate_feedback_cache_record(row, metadata)
        cache_hits += int(row["cache_hit"])
        cache_misses += int(not row["cache_hit"])
        tokens = row.get("token_count")
        if (
            not isinstance(tokens, dict) or set(tokens) != {"prompt", "completion"}
            or any(isinstance(tokens[key], bool) or not isinstance(tokens[key], int)
                   or tokens[key] < 0 for key in tokens)
        ):
            raise CampaignError("FeedTTA-LLM transcript token accounting is invalid")
        prompt_tokens += tokens["prompt"]
        completion_tokens += tokens["completion"]
        if not finite_number(row.get("cost")) or float(row["cost"]) < 0:
            raise CampaignError("FeedTTA-LLM transcript cost is invalid")
        total_cost += float(row["cost"])
        if not valid_sha256(row.get("panorama_png_sha256")):
            raise CampaignError("FeedTTA-LLM transcript panorama identity is invalid")
        if not isinstance(prediction, dict) or str(prediction.get("instr_id")) != episode_id:
            raise CampaignError("FeedTTA-LLM prediction identity mismatch")
        viewpoints = _prediction_viewpoints(prediction)
        trajectory_sha256 = hashlib.sha256(
            canonical(viewpoints).encode("utf-8")
        ).hexdigest()
        if (
            row.get("scan_id") != order_row.get("scene_id")
            or row.get("endpoint_viewpoint_id") != viewpoints[-1]
            or row.get("trajectory_sha256") != trajectory_sha256
        ):
            raise CampaignError("FeedTTA-LLM transcript trajectory binding mismatch")
        recorded = row.get("event_sha256")
        body = dict(row)
        body.pop("event_sha256", None)
        actual = hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()
        if recorded != actual:
            raise CampaignError("FeedTTA-LLM transcript hash-chain mismatch")
        previous = recorded
    if diagnostics.get("transcript_events") != len(rows) or diagnostics.get(
        "transcript_sha256"
    ) != previous:
        raise CampaignError("FeedTTA-LLM transcript terminal digest mismatch")
    budgets = provider.get("budgets", {})
    diagnostic_cost = diagnostics.get("cost")
    if (
        len(rows) > budgets.get("binary_labels_maximum", -1)
        or diagnostics.get("http_requests") != 2 * cache_misses
        or diagnostics.get("http_requests") > budgets.get("provider_requests_maximum", -1)
        or diagnostics.get("requests_per_uncached_episode") != 2
        or diagnostics.get("positive_labels") != positive
        or diagnostics.get("negative_labels") != negative
        or diagnostics.get("cache_hits") != cache_hits
        or diagnostics.get("cache_misses") != cache_misses
        or diagnostics.get("prompt_tokens") != prompt_tokens
        or diagnostics.get("completion_tokens") != completion_tokens
        or not finite_number(diagnostic_cost)
        or not math.isclose(float(diagnostic_cost), total_cost,
                            rel_tol=0.0, abs_tol=1e-12)
        or diagnostics.get("max_labels") != budgets.get("binary_labels_maximum")
        or diagnostics.get("max_http_requests") != budgets.get("provider_requests_maximum")
        or diagnostics.get("feedback_timing")
        != "post_episode_affects_future_episodes_only"
    ):
        raise CampaignError("FeedTTA-LLM transcript budget/accounting mismatch")


def _reject_test_metric_artifacts(metadata):
    metric_keys = {
        "sr", "spl", "rgs", "rgspl", "success", "oracle_success",
        "oracle_sr", "ndtw", "sdtw", "collisions", "nav_error",
        "oracle_error", "distance_to_goal",
    }
    metric_keys_upper = {name.upper() for name in metric_keys}

    def contains_metric_field(value):
        if isinstance(value, dict):
            if any(str(key).strip().lower() in metric_keys for key in value):
                return True
            return any(contains_metric_field(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_metric_field(item) for item in value)
        return False

    forbidden = []
    for path in Path(metadata["result_root"]).rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        allowed_json = (
            name == "tta_diagnostics.json"
            or name == "resolved_config.json"
            or name.startswith("submit_test")
            or name == "trajectory_evidence.json"
        )
        valid_metrics = False
        text_metrics = False
        json_metrics = False
        if name == "valid.txt":
            content = path.read_text(encoding="utf-8", errors="replace")
            valid_metrics = any(
                "Env name:" in line
                and bool(METRIC_RE.search(line))
                for line in content.splitlines()
            )
        if path.suffix.lower() in {".txt", ".log", ".out"}:
            try:
                with path.open(encoding="utf-8", errors="replace") as stream:
                    text_metrics = any(
                        key.upper() in metric_keys_upper
                        for line in stream
                        for key, _value in METRIC_RE.findall(line)
                    )
            except OSError:
                text_metrics = True
        if path.suffix.lower() == ".json" and name not in {
            "tta_diagnostics.json", "resolved_config.json",
            "trajectory_evidence.json",
        }:
            try:
                json_metrics = contains_metric_field(json.loads(
                    path.read_text(encoding="utf-8")
                ))
            except (OSError, UnicodeError, json.JSONDecodeError):
                # Malformed prediction/output JSON is rejected by its own
                # validator; it is not silently classified as metric-free.
                json_metrics = True
        if (
            valid_metrics
            or text_metrics
            or json_metrics
            or name == "per_episode_metrics.json"
            or name.startswith("stats_")
            or (
                not allowed_json
                and re.search(
                    r"(^|[_-])(metrics?|scores?|results?|evaluation|eval)"
                    r"([_\-.]|$)",
                    name,
                )
            )
        ):
            forbidden.append(path)
    if forbidden:
        raise CampaignError(
            "hidden-test run produced forbidden local metric artifacts: {}"
            .format(", ".join(str(path) for path in forbidden[:5]))
        )


def validate_run_manifest(metadata, require_success=True):
    from tools.run_manifest_identity import immutable_identity_sha256
    path = Path(metadata["formal_manifest"])
    manifest = read_json(path)
    expected = {
        "run_id": path.parent.name,
        "task": "vln",
        "benchmark": metadata["expected_benchmark"],
        "model": metadata["model"],
        "method": metadata["reported_method_label"].lower(),
        "run_tag": metadata["run_tag"],
        "source_setting": "{}:{}:{}:{}".format(
            metadata["setting"], metadata["split"], metadata["data_version"],
            metadata["reported_method_label"].lower(),
        ),
        "seed": 0,
        "git_commit": metadata["git_commit"],
    }
    if require_success:
        expected.update({"status": "completed", "exit_code": 0})
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CampaignError("formal run manifest {} mismatch".format(key))
    identity = manifest.get("immutable_identity_sha256")
    if not valid_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise CampaignError("formal run immutable identity mismatch")
    if manifest.get("checkpoint", {}).get("sha256") != metadata["checkpoint_sha256"]:
        raise CampaignError("formal run checkpoint mismatch")
    config_path = Path(str(manifest.get("config", ""))).resolve()
    if (
        config_path != Path(metadata["config_path"]).resolve()
        or not config_path.is_file()
        or sha256(config_path) != metadata["config_sha256"]
    ):
        raise CampaignError("formal run resolved config mismatch")
    auxiliary = manifest.get("auxiliary_checkpoints")
    if not isinstance(auxiliary, list):
        raise CampaignError("formal run auxiliary bindings are missing")
    auxiliary_by_name = {
        item.get("name"): item for item in auxiliary if isinstance(item, dict)
    }
    required_auxiliary = {"resolved_config"}
    if metadata["method"] == "idea":
        required_auxiliary.add("idea_source_statistics")
    if metadata["feedback_provider"] == "qwen2_vl_2b_v1":
        required_auxiliary.update({
            "feedback_provider_contract", "feedback_provider_preflight",
        })
    if set(auxiliary_by_name) != required_auxiliary:
        raise CampaignError("formal run auxiliary binding set mismatch")
    for name, item in auxiliary_by_name.items():
        item_path = Path(str(item.get("path", ""))).resolve()
        if (
            not item_path.is_file()
            or item_path.stat().st_size != item.get("size")
            or sha256(item_path) != item.get("sha256")
        ):
            raise CampaignError("formal run auxiliary {} changed".format(name))
    dataset = manifest.get("dataset", {})
    if (
        dataset.get("version") != metadata["data_version"]
        or dataset.get("index_sha256") != metadata["dataset_sha256"]
        or dataset.get("stream_content_sha256") != metadata["dataset_sha256"]
        or dataset.get("stream_order_sha256") != metadata["order_sha256"]
    ):
        raise CampaignError("formal run dataset/order mismatch")
    campaign = manifest.get("campaign", {})
    for key in ("job_identity_sha256", "config_sha256", "cell_id", "queue_id",
                "gpu_slot", "episode_order_seed", "supervision", "feedback_provider",
                "feedback_provider_preflight", "attempt", "retry_count",
                "retry_history_sha256"):
        expected_value = (
            metadata.get(key) if key == "feedback_provider_preflight"
            else metadata[key] if key in metadata else 0
        )
        if campaign.get(key) != expected_value:
            raise CampaignError("formal run campaign.{} mismatch".format(key))
    if metadata["feedback_provider"] == "qwen2_vl_2b_v1":
        _, spec = load_spec(metadata["spec_path"])
        validated_preflight = validate_provider_preflight_binding(
            spec,
            metadata.get("feedback_provider_preflight"),
        )
        auxiliary_preflight = auxiliary_by_name[
            "feedback_provider_preflight"
        ]
        if (
            Path(auxiliary_preflight["path"]).resolve()
            != Path(validated_preflight["path"]).resolve()
            or auxiliary_preflight["size"] != validated_preflight["size"]
            or auxiliary_preflight["sha256"] != validated_preflight["sha256"]
        ):
            raise CampaignError("formal run PRECHECK binding mismatch")
    pinned = manifest.get("pinned_manifests", {})
    expected_pinned = {
        "experiment_spec": (Path(metadata["spec_path"]), metadata["spec_sha256"]),
        "episode_order": (Path(metadata["order_manifest_path"]), metadata["order_manifest_sha256"]),
    }
    for name, (path_value, digest) in expected_pinned.items():
        record = pinned.get(name, {})
        if Path(str(record.get("path", ""))).resolve() != path_value.resolve() or record.get("sha256") != digest:
            raise CampaignError("formal run pinned {} mismatch".format(name))
    hardware = manifest.get("hardware")
    if (
        not isinstance(hardware, dict)
        or not hardware.get("hostname")
        or hardware.get("cuda_visible_devices") != str(metadata["gpu"])
        or not isinstance(hardware.get("gpu_name"), str)
        or not hardware.get("gpu_name")
        or not isinstance(hardware.get("gpu_uuid"), str)
        or not hardware.get("gpu_uuid")
        or isinstance(hardware.get("gpu_memory_mib"), bool)
        or not isinstance(hardware.get("gpu_memory_mib"), int)
        or hardware.get("gpu_memory_mib") <= 0
    ):
        raise CampaignError("formal run hardware evidence is missing")
    artifacts = manifest.get("result_artifacts")
    if require_success and (not isinstance(artifacts, list) or not artifacts):
        raise CampaignError("formal run manifest has no result artifacts")
    for artifact in artifacts or ():
        path_value = Path(str(artifact.get("path", ""))).resolve()
        if (
            not path_value.is_file()
            or path_value.stat().st_size != artifact.get("size")
            or sha256(path_value) != artifact.get("sha256")
        ):
            raise CampaignError("formal result artifact changed")
    if require_success:
        names = {item.get("name") for item in artifacts}
        required_names = {"tta_diagnostics.json", "resolved_config.json"}
        if metadata["split"] == "test":
            required_names.update({"full_predictions", "trajectory_evidence.json"})
            if metadata["feedback_provider"] == "qwen2_vl_2b_v1":
                required_names.add("llm_feedback_transcript.ndjson")
            if names != required_names:
                raise CampaignError(
                    "formal hidden-test manifest contains unexpected artifacts"
                )
        elif metadata["benchmark"] == "r2r-ce":
            required_names.update({"aggregate_metrics", "per_episode_metrics.json"})
        else:
            required_names.update({
                "aggregate_metrics", "full_predictions", "trajectory_evidence.json",
            })
        if not required_names.issubset(names):
            raise CampaignError("formal run manifest lacks required result artifacts")
    return path, manifest


def _expected_attempt_job(attempt_dir, metadata):
    """Rebind mutable job.json to the independently regenerated stage plan."""
    attempt_dir = Path(attempt_dir).resolve()
    try:
        stage_dir = attempt_dir.parents[3]
        root = attempt_dir.parents[5]
    except IndexError:
        raise CampaignError("attempt path is outside a campaign stage")
    if stage_dir.parent.name != "stages" or root != stage_dir.parent.parent:
        raise CampaignError("attempt path is outside a campaign stage")
    binding = read_json(root / "BATCH.json")
    spec_path = repo_file(binding.get("spec_path"), "bound campaign spec", True)
    loaded_path, spec = load_spec(spec_path)
    if (
        binding.get("schema") != BATCH_SCHEMA
        or binding.get("model_seed") != 0
        or binding.get("episode_order_seed") != 0
        or binding.get("batch_id") != metadata.get("batch_id")
        or binding.get("spec_sha256") != sha256(loaded_path)
        or binding.get("runner_sha256") != sha256(SCRIPT_PATH)
        or binding.get("git_commit") != git_commit()
    ):
        raise CampaignError("attempt no longer matches BATCH.json")
    stage = stage_dir.name
    gpus = tuple(binding.get("gpus", ()))
    if stage == "search":
        expected_jobs = expand_search_jobs(spec, gpus)
        frozen_path = None
    elif stage in ("val-seen", "reverie-test"):
        frozen_path = root / "FROZEN.json"
        frozen = read_json(frozen_path)
        expected_jobs = expand_frozen_jobs(
            spec, gpus, frozen, stage, binding["batch_id"]
        )
    else:
        raise CampaignError("attempt stage is invalid")
    actual_plan = read_json(_stage_plan_path(root, stage))
    provider_preflight = actual_plan.get("provider_preflight")
    if stage == "reverie-test":
        provider_preflight = validate_provider_preflight_binding(
            spec, provider_preflight
        )
        expected_jobs = bind_provider_preflight(
            expected_jobs, provider_preflight
        )
    elif provider_preflight is not None:
        raise CampaignError("provider preflight is forbidden outside reverie-test")
    expected_plan = _stage_plan_payload(
        binding, stage, expected_jobs, frozen_path=frozen_path,
        provider_preflight=provider_preflight,
    )
    if canonical(actual_plan) != canonical(expected_plan):
        raise CampaignError("stage plan differs from independent expansion")
    matches = [
        job for job in expected_jobs
        if _job_parent(root, job).resolve() == attempt_dir.parent
    ]
    if len(matches) != 1:
        raise CampaignError("attempt does not identify one expected stage job")
    expected_job = matches[0]
    for key in (
        "ordinal", "stage", "split", "queue_id", "gpu_slot", "gpu",
        "cell_id", "benchmark", "setting", "model", "method",
        "reported_method_label", "supervision", "feedback_provider",
        "feedback_provider_preflight",
        "data_version", "candidate_id", "candidate_role", "parameters",
        "submission_id",
    ):
        if metadata.get(key) != expected_job.get(key):
            raise CampaignError("job.json {} differs from the stage plan".format(key))
    expected_identity = _job_identity(
        loaded_path, binding["batch_id"], expected_job
    )
    if metadata.get("job_identity_sha256") != expected_identity:
        raise CampaignError("job.json identity digest mismatch")
    return loaded_path, spec, binding, expected_job


def validate_attempt(attempt_dir, require_paired=False):
    attempt_dir = Path(attempt_dir)
    metadata = read_json(attempt_dir / "job.json")
    spec_path, spec, binding, expected_job = _expected_attempt_job(
        attempt_dir, metadata
    )
    try:
        attempt_number = int(attempt_dir.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        raise CampaignError("attempt directory number is invalid")
    retry_history, retry_history_sha256 = _validated_retry_history(
        root=attempt_dir.parents[5], job=expected_job,
        current_attempt=attempt_number,
    )
    if (
        metadata.get("retry_history") != retry_history
        or metadata.get("retry_history_sha256") != retry_history_sha256
        or metadata.get("retry_count") != len(retry_history)
    ):
        raise CampaignError("job.json retry history binding mismatch")
    expected_run_tag = _attempt_tag(binding["batch_id"], expected_job, attempt_number)
    expected_result_root = _result_root(
        binding["batch_id"], expected_job, expected_run_tag
    )
    if (
        metadata.get("attempt") != attempt_number
        or metadata.get("run_tag") != expected_run_tag
        or Path(str(metadata.get("result_root", ""))).resolve()
        != expected_result_root
        or metadata.get("spec_path") != str(spec_path)
        or metadata.get("spec_sha256") != sha256(spec_path)
        or metadata.get("runner_sha256") != sha256(SCRIPT_PATH)
        or metadata.get("git_commit") != binding["git_commit"]
    ):
        raise CampaignError("job.json execution identity mismatch")
    expected_config = _job_config(
        spec, binding["batch_id"], expected_job, expected_result_root
    )
    config_path = Path(str(metadata.get("config_path", ""))).resolve()
    if (
        not config_path.is_file()
        or canonical(read_json(config_path)) != canonical(expected_config)
        or metadata.get("config_sha256") != sha256(config_path)
        or metadata.get("runtime_parameters") != expected_config["parameters"]
        or metadata.get("command") != _command(
            expected_job, config_path, expected_result_root, expected_run_tag
        )
    ):
        raise CampaignError("job.json resolved execution plan mismatch")
    order = order_binding(spec, expected_job["setting"], expected_job["split"])
    checkpoint = checkpoint_binding(spec, expected_job["setting"])
    expected_derived = {
        "model_seed": 0,
        "episode_order_seed": 0,
        "episode_count": order["episode_count"],
        "expected_benchmark": order["benchmark"],
        "order_manifest_path": str(order["path"]),
        "order_manifest_sha256": order["sha256"],
        "order_sha256": order["order_sha256"],
        "dataset_path": str(order["dataset_path"]),
        "dataset_sha256": order["dataset_sha256"],
        "checkpoint_path": str(checkpoint["path"]),
        "checkpoint_sha256": checkpoint["sha256"],
        "formal_manifest": str(
            FORMAL_ROOT / (
                expected_run_tag + "-" + expected_job["setting"] + "-"
                + expected_job["split"] + "-" + expected_job["data_version"]
                + "-" + expected_job["reported_method_label"].lower()
            ) / "manifest.json"
        ),
    }
    for key, value in expected_derived.items():
        if metadata.get(key) != value:
            raise CampaignError("job.json {} differs from derived evidence".format(key))
    exit_path = attempt_dir / "exitcode"
    try:
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        raise CampaignError("attempt has no valid exit code")
    if exit_code != 0:
        raise CampaignError("worker exited with status {}".format(exit_code))
    require_paired = require_paired or metadata["stage"] in ("search", "val-seen")
    if metadata["split"] == "test":
        _reject_test_metric_artifacts(metadata)
    diagnostics_path = Path(metadata["result_root"]) / "tta_diagnostics.json"
    if not diagnostics_path.is_file():
        raise CampaignError("TTA diagnostics are missing")
    diagnostics, compact_diagnostics = _validate_diagnostics(metadata, diagnostics_path)
    prediction_path = None
    trajectory_evidence = None
    metrics_path = None
    metrics = None
    paired_path = None
    paired_ready = False
    if metadata["split"] == "test":
        prediction_path, trajectory_evidence = _validate_prediction(metadata, attempt_dir)
    else:
        metrics_path, metrics = _metric_artifact(metadata)
        if metadata["benchmark"] != "r2r-ce":
            prediction_path, trajectory_evidence = _validate_prediction(metadata, attempt_dir)
        paired_path, paired_ready = _per_episode_evidence(
            metadata, attempt_dir, metrics
        )
        if require_paired and not paired_ready:
            raise CampaignError(
                "paired per-episode metrics are required before campaign freeze; "
                "the full prediction is preserved but no metric sidecar exists"
            )
    artifacts = [("tta_diagnostics.json", diagnostics_path)]
    artifacts.append(("resolved_config.json", Path(metadata["config_path"])))
    if metrics_path is not None:
        artifacts.append(("aggregate_metrics", metrics_path))
    if prediction_path is not None:
        artifacts.append(("full_predictions", prediction_path))
    if trajectory_evidence is not None:
        artifacts.append(("trajectory_evidence.json", trajectory_evidence))
    if paired_path is not None:
        artifacts.append(("per_episode_metrics.json", paired_path))
    if metadata["feedback_provider"] == "qwen2_vl_2b_v1":
        artifacts.append((
            "llm_feedback_transcript.ndjson",
            Path(metadata["result_root"]) / "llm_feedback_transcript.ndjson",
        ))
    manifest_path = Path(metadata["formal_manifest"])
    _finish_run_manifest(manifest_path, 0, artifacts)
    manifest_path, manifest = validate_run_manifest(metadata)
    result = {
        **metadata,
        "schema": RESULT_SCHEMA,
        "metrics": metrics,
        "metric_artifact": str(metrics_path.resolve()) if metrics_path else None,
        "metric_artifact_sha256": sha256(metrics_path) if metrics_path else None,
        "prediction_artifact": str(prediction_path.resolve()) if prediction_path else None,
        "prediction_artifact_sha256": sha256(prediction_path) if prediction_path else None,
        "trajectory_evidence": str(trajectory_evidence.resolve()) if trajectory_evidence else None,
        "trajectory_evidence_sha256": sha256(trajectory_evidence) if trajectory_evidence else None,
        "paired_episode_metrics": str(paired_path.resolve()) if paired_path else None,
        "paired_episode_metrics_sha256": sha256(paired_path) if paired_path else None,
        "paired_episode_metrics_ready": paired_ready,
        "diagnostics_path": str(diagnostics_path.resolve()),
        "diagnostics_sha256": sha256(diagnostics_path),
        "diagnostics": compact_diagnostics,
        "formal_manifest_sha256": sha256(manifest_path),
        "formal_immutable_identity_sha256": manifest["immutable_identity_sha256"],
        "retry_history": retry_history,
        "retry_history_sha256": retry_history_sha256,
        "retry_count": len(retry_history),
    }
    atomic_json(attempt_dir / "result.json", result)
    return result


def _scheduler_log(root, message):
    line = "[{}] {}".format(utc_now(), message)
    with _LOG_LOCK:
        print(line, flush=True)
        path = Path(root) / "scheduler.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()


def _retry_reason(value):
    if not isinstance(value, str):
        raise CampaignError("--retry-failed requires a nonempty --retry-reason")
    reason = value.strip()
    if not reason or len(reason) > 1000 or any(ord(char) < 32 for char in reason):
        raise CampaignError("--retry-reason must be nonempty printable text")
    return reason


def _retryable_infrastructure_classification(attempt_dir, state):
    """Return machine evidence for the small set of retryable failures."""
    attempt_dir = Path(attempt_dir)
    if state == "orphaned":
        identity_path = attempt_dir / "process_identity.json"
        if identity_path.is_file() and not _pid_alive(attempt_dir):
            identity = read_json(identity_path)
            return {
                "basis": "recorded_worker_disappeared",
                "pid": identity.get("pid"),
                "start_token": identity.get("start_token"),
                "process_group_id": identity.get("process_group_id"),
            }
        raise CampaignError(
            "orphaned retry lacks a verifiable terminated worker identity"
        )
    if state == "failed":
        try:
            exit_code = int((attempt_dir / "exitcode").read_text(
                encoding="utf-8"
            ).strip())
        except (OSError, ValueError):
            raise CampaignError("failed retry lacks a valid exit code")
        marker_path = attempt_dir / "SCHEDULER_TERMINATION.json"
        if marker_path.is_file():
            marker = read_json(marker_path)
            metadata = read_json(attempt_dir / "job.json")
            identity = read_json(attempt_dir / "process_identity.json")
            if (
                marker.get("schema") == TERMINATION_SCHEMA
                and marker.get("job_identity_sha256")
                == metadata.get("job_identity_sha256")
                and marker.get("pid") == identity.get("pid")
                and marker.get("process_group_id")
                == identity.get("process_group_id")
                and marker.get("start_token") == identity.get("start_token")
                and marker.get("termination_signal") == int(signal.SIGTERM)
                and exit_code in (-int(signal.SIGTERM), 128 + int(signal.SIGTERM),
                                  -int(signal.SIGKILL), 128 + int(signal.SIGKILL))
                and isinstance(marker.get("reason"), str)
                and marker["reason"]
            ):
                return {
                    "basis": "scheduler_recorded_termination",
                    "exit_code": exit_code,
                    "termination_marker_sha256": sha256(marker_path),
                    "reason": marker["reason"],
                }
        raise CampaignError(
            "worker failure is not independently marked as scheduler infrastructure"
        )
    if state == "invalid":
        validation = read_json(attempt_dir / "validation_error.json")
        if validation.get("failure_classification") == "infrastructure":
            evidence = validation.get("classification_evidence")
            if not isinstance(evidence, dict) or not evidence:
                raise CampaignError("infrastructure validation failure lacks evidence")
            return {
                "basis": "validator_classified_infrastructure",
                "validation_evidence": evidence,
            }
        raise CampaignError("invalid result is not classified as infrastructure")
    raise CampaignError("attempt state is not eligible for infrastructure retry")


def _compact_prior_evidence(attempt_dir):
    """Hash the small immutable evidence needed to authorize one retry."""
    attempt_dir = Path(attempt_dir)
    job_path = attempt_dir / "job.json"
    if not job_path.is_file():
        raise CampaignError("retry requires the prior job.json")
    metadata = read_json(job_path)
    evidence = {}
    for name in (
        "job.json", "exitcode", "validation_error.json",
        "process_identity.json", "SCHEDULER_TERMINATION.json", "pid",
        "launcher.log", "result.json",
    ):
        path = attempt_dir / name
        if path.is_file():
            evidence[name] = {"size": path.stat().st_size, "sha256": sha256(path)}
    for name, value in (
        ("formal_manifest", metadata.get("formal_manifest")),
        ("resolved_config", metadata.get("config_path")),
    ):
        path = Path(str(value or "")).resolve()
        if path.is_file():
            evidence[name] = {"size": path.stat().st_size, "sha256": sha256(path)}
    if "job.json" not in evidence:
        raise CampaignError("retry evidence is missing job.json")
    return evidence


def _archive_for_retry(attempt_dir, next_attempt, state, job, retry_reason):
    attempt_dir = Path(attempt_dir)
    path = attempt_dir / "RETRY_ARCHIVE.json"
    reason = _retry_reason(retry_reason)
    prior_attempt = int(attempt_dir.name.rsplit("-", 1)[1])
    evidence = _compact_prior_evidence(attempt_dir)
    evidence_sha256 = hashlib.sha256(
        canonical(evidence).encode("utf-8")
    ).hexdigest()
    metadata = read_json(attempt_dir / "job.json")
    if not valid_sha256(metadata.get("job_identity_sha256")):
        raise CampaignError("retry requires the prior job identity")
    classification_evidence = _retryable_infrastructure_classification(
        attempt_dir, state
    )
    payload = {
        "schema": RETRY_ARCHIVE_SCHEMA,
        "failure_classification": "infrastructure",
        "operator_authorized": True,
        "classification_evidence": classification_evidence,
        "retry_reason": reason,
        "job_key": _job_key(job),
        "job_identity_sha256": metadata["job_identity_sha256"],
        "prior_attempt": prior_attempt,
        "prior_state": state,
        "next_attempt": next_attempt,
        "evidence_preserved_in_place": True,
        "prior_evidence": evidence,
        "prior_evidence_sha256": evidence_sha256,
        "recorded_at": utc_now(),
    }
    if path.exists():
        old = read_json(path)
        for key in (
            "schema", "failure_classification", "operator_authorized",
            "classification_evidence", "retry_reason", "job_key",
            "job_identity_sha256",
            "prior_attempt", "prior_state", "next_attempt",
            "evidence_preserved_in_place", "prior_evidence",
            "prior_evidence_sha256",
        ):
            if old.get(key) != payload[key]:
                raise CampaignError("retry archive changed: {}".format(path))
        return path
    atomic_json(path, payload)
    return path


def _validated_retry_history(root, job, current_attempt):
    prior = [(number, path) for number, path in _attempts(root, job)
             if number < current_attempt]
    if [number for number, _ in prior] != list(range(1, current_attempt)):
        raise CampaignError("retry attempt history is incomplete")
    entries = []
    for number, attempt_dir in prior:
        archive_path = attempt_dir / "RETRY_ARCHIVE.json"
        if not archive_path.is_file():
            raise CampaignError("prior attempt lacks RETRY_ARCHIVE.json")
        archive = read_json(archive_path)
        expected_state = attempt_state(attempt_dir)
        if expected_state not in ("failed", "invalid", "orphaned"):
            raise CampaignError("only terminal infrastructure failures may be retried")
        metadata = read_json(attempt_dir / "job.json")
        _expected_attempt_job(attempt_dir, metadata)
        evidence = _compact_prior_evidence(attempt_dir)
        evidence_sha256 = hashlib.sha256(
            canonical(evidence).encode("utf-8")
        ).hexdigest()
        expected = {
            "schema": RETRY_ARCHIVE_SCHEMA,
            "failure_classification": "infrastructure",
            "operator_authorized": True,
            "classification_evidence": _retryable_infrastructure_classification(
                attempt_dir, expected_state
            ),
            "job_key": _job_key(job),
            "job_identity_sha256": metadata.get("job_identity_sha256"),
            "prior_attempt": number,
            "prior_state": expected_state,
            "next_attempt": number + 1,
            "evidence_preserved_in_place": True,
            "prior_evidence": evidence,
            "prior_evidence_sha256": evidence_sha256,
        }
        for key, value in expected.items():
            if archive.get(key) != value:
                raise CampaignError("retry archive {} mismatch".format(key))
        _retry_reason(archive.get("retry_reason"))
        if not isinstance(archive.get("recorded_at"), str) or not archive["recorded_at"]:
            raise CampaignError("retry archive recorded_at is missing")
        entries.append({
            "prior_attempt": number,
            "next_attempt": number + 1,
            "failure_classification": "infrastructure",
            "operator_authorized": True,
            "classification_evidence": archive["classification_evidence"],
            "retry_reason": archive["retry_reason"],
            "prior_evidence_sha256": evidence_sha256,
            "archive_path": str(archive_path.resolve()),
            "archive_sha256": sha256(archive_path),
        })
    digest = hashlib.sha256(canonical(entries).encode("utf-8")).hexdigest()
    return entries, digest


def _run_attempt(spec_path, spec, batch_id, root, job, attempt):
    _assert_runtime_binding(root, spec_path)
    attempt_dir, metadata = materialize_attempt(
        spec_path, spec, batch_id, root, job, attempt
    )
    manifest_path = _create_run_manifest(metadata)
    launcher_log = (attempt_dir / "launcher.log").open("ab", buffering=0)
    _scheduler_log(root, "launch {} gpu={} queue=q{:02d}".format(
        metadata["run_tag"], job["gpu"], job["queue_id"]))
    process = None
    exit_code = -1
    try:
        child_environment = os.environ.copy()
        child_environment.pop(LLM_PREFLIGHT_ENV, None)
        child_environment.pop(RENDER_BUILD_ENV, None)
        if metadata["feedback_provider"] == "qwen2_vl_2b_v1":
            provider_preflight = validate_provider_preflight_binding(
                spec, metadata.get("feedback_provider_preflight")
            )
            child_environment[RENDER_BUILD_ENV] = provider_preflight[
                "render_mattersim_build"
            ]
        process = subprocess.Popen(
            metadata["command"], cwd=str(REPO_ROOT), stdout=launcher_log,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=child_environment,
        )
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES[process] = attempt_dir
        identity = _process_identity(process.pid)
        if identity is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait()
            raise CampaignError("could not persist a stable worker process identity")
        atomic_text(attempt_dir / "pid", "{}\n".format(process.pid))
        atomic_json(attempt_dir / "process_identity.json", identity)
        exit_code = process.wait()
    except Exception:
        atomic_text(attempt_dir / "exitcode", "{}\n".format(exit_code))
        _finish_run_manifest(manifest_path, exit_code)
        raise
    finally:
        launcher_log.close()
        if process is not None:
            with _ACTIVE_LOCK:
                _ACTIVE_PROCESSES.pop(process, None)
    atomic_text(attempt_dir / "exitcode", "{}\n".format(exit_code))
    if exit_code != 0:
        _finish_run_manifest(manifest_path, exit_code)
        _scheduler_log(root, "failed {} exit={}".format(metadata["run_tag"], exit_code))
        return False
    try:
        validate_attempt(
            attempt_dir,
            require_paired=job["stage"] in ("search", "val-seen"),
        )
    except Exception as error:
        atomic_json(attempt_dir / "validation_error.json", {
            "error": str(error), "recorded_at": utc_now(),
        })
        _scheduler_log(root, "invalid {}: {}".format(metadata["run_tag"], error))
        return False
    _scheduler_log(root, "completed {}".format(metadata["run_tag"]))
    return True


def _choose_attempt(root, job, retry_failed, retry_reason=None):
    latest = latest_attempt(root, job)
    if latest is None:
        return "run", 1, None
    number, path = latest
    state = attempt_state(path)
    if state == "completed":
        try:
            validate_attempt(
                path, require_paired=job["stage"] in ("search", "val-seen")
            )
        except Exception as error:
            atomic_json(path / "validation_error.json", {
                "error": str(error), "recorded_at": utc_now(),
            })
            state = "invalid"
        else:
            return "skip", number, None
    if state == "finished":
        try:
            validate_attempt(
                path, require_paired=job["stage"] in ("search", "val-seen")
            )
        except Exception as error:
            atomic_json(path / "validation_error.json", {
                "error": str(error), "recorded_at": utc_now(),
            })
            state = "invalid"
        else:
            return "skip", number, None
    if state == "running":
        return "blocked", number, "worker is still running"
    if state in ("failed", "invalid", "orphaned"):
        if not retry_failed:
            return "blocked", number, "{}; use --resume --retry-failed".format(state)
        next_attempt = number + 1
        _archive_for_retry(
            path, next_attempt, state, job, _retry_reason(retry_reason)
        )
        return "run", next_attempt, None
    return "run", max(1, number), None


def _run_cell_queue(spec_path, spec, batch_id, root, jobs, retry_failed,
                    retry_reason=None):
    failures = []
    for job in jobs:
        action, attempt, reason = _choose_attempt(
            root, job, retry_failed, retry_reason
        )
        if action == "skip":
            _scheduler_log(root, "skip validated {}".format(_job_key(job)))
            continue
        if action == "blocked":
            failures.append("{}: {}".format(_job_key(job), reason))
            # A live predecessor owns this serial queue.  Starting a later
            # candidate would violate max_active_jobs_per_cell=1.
            if reason == "worker is still running":
                break
            # Terminal failed evidence does not share mutable process state;
            # later fresh candidates may still run and be preserved.
            continue
        try:
            if not _run_attempt(spec_path, spec, batch_id, root, job, attempt):
                failures.append("{}: attempt failed".format(_job_key(job)))
        except Exception as error:
            failures.append("{}: {}".format(_job_key(job), error))
            _scheduler_log(root, "launcher error {}: {}".format(_job_key(job), error))
    return failures


def _stage_plan_path(root, stage):
    return Path(root) / "stages" / stage / "STAGE_PLAN.json"


def _stage_plan_payload(
    binding, stage, jobs, frozen_path=None, provider_preflight=None,
):
    return {
        "schema": "navtta.vln_targeted_gap_stage_plan.v1",
        "batch_id": binding["batch_id"],
        "git_commit": binding["git_commit"],
        "spec_sha256": binding["spec_sha256"],
        "runner_sha256": binding["runner_sha256"],
        "stage": stage,
        "split": jobs[0]["split"] if jobs else None,
        "job_count": len(jobs),
        "queue_ids": sorted({job["queue_id"] for job in jobs}),
        "frozen_path": str(Path(frozen_path).resolve()) if frozen_path else None,
        "frozen_sha256": sha256(frozen_path) if frozen_path else None,
        "provider_preflight": provider_preflight,
        "jobs": jobs,
    }


def write_stage_plan(
    root, binding, stage, jobs, frozen_path=None, provider_preflight=None,
):
    payload = _stage_plan_payload(
        binding, stage, jobs, frozen_path, provider_preflight
    )
    path = _stage_plan_path(root, stage)
    if path.exists() and canonical(read_json(path)) != canonical(payload):
        raise CampaignError("existing {} stage plan differs".format(stage))
    if not path.exists():
        atomic_json(path, payload)
    return path, payload


def _state_counts(root, jobs):
    counts = Counter()
    for job in jobs:
        latest = latest_attempt(root, job)
        counts["pending" if latest is None else attempt_state(latest[1])] += 1
    return {key: counts.get(key, 0) for key in (
        "completed", "finished", "failed", "invalid", "running",
        "orphaned", "pending",
    )}


def write_stage_summary(root, stage, jobs):
    states = _state_counts(root, jobs)
    payload = {
        "schema": SUMMARY_SCHEMA,
        "stage": stage,
        "job_count": len(jobs),
        "states": states,
        "complete": states["completed"] == len(jobs),
        "updated_at": utc_now(),
    }
    atomic_json(Path(root) / "stages" / stage / "SUMMARY.json", payload)
    return payload


def run_stage(spec_path, spec, batch_id, root, binding, stage, jobs,
              resume=False, retry_failed=False, retry_reason=None,
              provider_preflight=None):
    plan_path = _stage_plan_path(root, stage)
    if plan_path.exists() and not resume:
        raise CampaignError("{} stage already exists; use --resume".format(stage))
    write_stage_plan(
        root, binding, stage, jobs,
        frozen_path=(root / "FROZEN.json") if stage != "search" else None,
        provider_preflight=provider_preflight,
    )
    grouped = defaultdict(list)
    for job in jobs:
        grouped[job["queue_id"]].append(job)
    for queue_id, queue in grouped.items():
        if any(item["queue_id"] != queue_id for item in queue):
            raise CampaignError("cell queue contains a migrated job")
    # Fail before any worker is submitted if even one requested GPU cannot be
    # identified.  Per-run manifests probe again to bind the observed device.
    for gpu in sorted({job["gpu"] for job in jobs}):
        _hardware(gpu)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(grouped), thread_name_prefix="targeted-gap-q"
    ) as executor:
        futures = {
            executor.submit(
                _run_cell_queue, spec_path, spec, batch_id, root, queue,
                retry_failed, retry_reason,
            ): queue_id
            for queue_id, queue in sorted(grouped.items())
        }
        for future in concurrent.futures.as_completed(futures):
            queue_id = futures[future]
            try:
                failures.extend(future.result())
            except Exception as error:
                failures.append("q{:02d}: {}".format(queue_id, error))
    summary = write_stage_summary(root, stage, jobs)
    if failures or not summary["complete"]:
        raise CampaignError("{} stage incomplete: {}".format(
            stage, "; ".join(failures[:20]) or canonical(summary["states"])))
    return summary


def _completed_results(root, jobs, require_paired=False):
    values = []
    errors = []
    for job in jobs:
        latest = latest_attempt(root, job)
        if latest is None:
            errors.append("{} is pending".format(_job_key(job)))
            continue
        _, path = latest
        try:
            values.append(validate_attempt(path, require_paired=require_paired))
        except Exception as error:
            errors.append("{}: {}".format(_job_key(job), error))
    if errors:
        raise CampaignError("stage evidence is incomplete: {}".format(
            "; ".join(errors[:20])))
    return values


def _rank_key(result):
    metrics = result["metrics"]
    diagnostics = result["diagnostics"]
    if result["benchmark"] == "reverie":
        metric_part = tuple(-float(metrics[key]) for key in (
            "RGSPL", "RGS", "SPL", "SR"
        ))
    else:
        metric_part = (-float(metrics["SPL"]), -float(metrics["SR"]))
    return metric_part + (
        float(diagnostics["relative_param_drift"]),
        int(diagnostics["updates"]),
        result["candidate_id"],
    )


def select_winners(results):
    grouped = defaultdict(list)
    for result in results:
        if result.get("stage") != "search" or result.get("split") != "val_unseen":
            raise CampaignError("selection received non-development evidence")
        grouped[result["cell_id"]].append(result)
    winners = {}
    rankings = {}
    for cell_id, rows in grouped.items():
        ranked = sorted(rows, key=_rank_key)
        winners[cell_id] = ranked[0]
        rankings[cell_id] = ranked
    return winners, rankings


def _frozen_val_job(cell, winner, gpu):
    return {
        "ordinal": cell["queue_id"],
        "stage": "val-seen",
        "split": "val_seen",
        "queue_id": cell["queue_id"],
        "gpu_slot": cell["gpu_slot"],
        "gpu": gpu,
        "cell_id": cell["cell_id"],
        "benchmark": cell["benchmark"],
        "setting": cell["setting"],
        "model": cell["model"],
        "method": cell["method"],
        "reported_method_label": cell["method"],
        "supervision": cell["supervision"],
        "feedback_provider": (
            "task_evaluator" if cell["supervision"] == "feedback_supervised"
            else "none"
        ),
        "data_version": cell["data_version"],
        "candidate_id": winner["candidate_id"],
        "candidate_role": winner["candidate_role"],
        "parameters": winner["parameters"],
    }


def freeze_campaign(spec_path, spec, batch_id, root, binding, search_jobs,
                    source_evidence):
    if Path(root, "FROZEN.json").exists():
        # Recompute below and require byte-equivalent scientific content.
        pass
    results = _completed_results(root, search_jobs, require_paired=True)
    if len(results) != 55:
        raise CampaignError("freeze requires exactly 55 validated development runs")
    expected_keys = {_job_key(job) for job in search_jobs}
    if {_job_key(row) for row in results} != expected_keys:
        raise CampaignError("development evidence does not match the immutable plan")
    winners, rankings = select_winners(results)
    if set(winners) != {cell["cell_id"] for cell in spec["cells"]}:
        raise CampaignError("freeze requires one winner for every cell")

    frozen_winners = []
    for cell in spec["cells"]:
        result = winners[cell["cell_id"]]
        val_job = _frozen_val_job(cell, result, binding["gpus"][cell["gpu_slot"]])
        config = _job_config(spec, batch_id, val_job)
        config_path = Path(root) / "frozen_configs" / (
            "q{:02d}-{}.json".format(cell["queue_id"], cell["cell_id"])
        )
        if config_path.exists() and canonical(read_json(config_path)) != canonical(config):
            raise CampaignError("existing frozen config changed: {}".format(config_path))
        atomic_json(config_path, config)
        frozen_winners.append({
            "queue_id": cell["queue_id"],
            "cell_id": cell["cell_id"],
            "benchmark": cell["benchmark"],
            "setting": cell["setting"],
            "model": cell["model"],
            "method": cell["method"],
            "supervision": cell["supervision"],
            "data_version": cell["data_version"],
            "candidate_id": result["candidate_id"],
            "candidate_role": result["candidate_role"],
            "parameters": result["parameters"],
            "development_metrics": result["metrics"],
            "development_diagnostics": result["diagnostics"],
            "development_run_manifest": result["formal_manifest"],
            "development_run_manifest_sha256": result["formal_manifest_sha256"],
            "development_run_identity_sha256": result[
                "formal_immutable_identity_sha256"
            ],
            "frozen_config": str(config_path.resolve()),
            "frozen_config_sha256": sha256(config_path),
        })

    development = []
    for result in sorted(results, key=lambda item: item["ordinal"]):
        development.append({
            "ordinal": result["ordinal"],
            "queue_id": result["queue_id"],
            "cell_id": result["cell_id"],
            "candidate_id": result["candidate_id"],
            "config_path": result["config_path"],
            "config_sha256": result["config_sha256"],
            "run_manifest": result["formal_manifest"],
            "run_manifest_sha256": result["formal_manifest_sha256"],
            "immutable_identity_sha256": result[
                "formal_immutable_identity_sha256"
            ],
            "metric_artifact_sha256": result["metric_artifact_sha256"],
            "prediction_artifact_sha256": result["prediction_artifact_sha256"],
            "paired_episode_metrics_sha256": result[
                "paired_episode_metrics_sha256"
            ],
            "retry_count": result["retry_count"],
            "retry_history": result["retry_history"],
            "retry_history_sha256": result["retry_history_sha256"],
        })
    retry_histories = [{
        "job_key": "q{:02d}-{}-{}".format(
            item["queue_id"], item["cell_id"], item["candidate_id"]
        ),
        "retry_count": item["retry_count"],
        "retry_history_sha256": item["retry_history_sha256"],
    } for item in development]
    retry_history_sha256 = hashlib.sha256(
        canonical(retry_histories).encode("utf-8")
    ).hexdigest()
    source_bindings = {
        key: {
            "path": str(value["path"]),
            "sha256": value["sha256"],
            "formal_manifests": value["formal_manifests"],
        }
        for key, value in sorted(source_evidence.items())
    }
    ranking_payload = {}
    for cell_id, rows in rankings.items():
        ranking_payload[cell_id] = [{
            "rank": rank,
            "candidate_id": row["candidate_id"],
            "metrics": row["metrics"],
            "relative_param_drift": row["diagnostics"]["relative_param_drift"],
            "accepted_updates": row["diagnostics"]["updates"],
            "run_manifest_sha256": row["formal_manifest_sha256"],
        } for rank, row in enumerate(rows, 1)]
    payload = {
        "schema": FREEZE_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "git_commit": binding["git_commit"],
        "spec_path": str(spec_path),
        "spec_sha256": binding["spec_sha256"],
        "plan_sha256": binding["plan_sha256"],
        "selection_code_path": str(SCRIPT_PATH),
        "selection_code_sha256": binding["runner_sha256"],
        "development_split": "val_unseen",
        "model_seed": 0,
        "episode_order_seed": 0,
        "val_seen_consulted": False,
        "test_consulted": False,
        "selection_rule": spec["selection"]["ranking_by_benchmark"],
        "source_manifests": source_bindings,
        "retry_history": retry_histories,
        "retry_history_sha256": retry_history_sha256,
        "development_runs": development,
        "winners": frozen_winners,
        "rankings": ranking_payload,
    }
    path = Path(root) / "FROZEN.json"
    expected_path = repo_file(spec["freeze"]["artifact"], "freeze artifact", require=False)
    if path.resolve() != expected_path:
        raise CampaignError("batch_id does not map to the spec-pinned FROZEN.json path")
    if path.exists() and canonical(read_json(path)) != canonical(payload):
        raise CampaignError("existing FROZEN.json differs from authenticated selection")
    atomic_json(path, payload)
    print("frozen {} winners at {}".format(len(frozen_winners), path))
    return payload


def load_frozen(spec_path, spec, batch_id, root, binding):
    path = Path(root) / "FROZEN.json"
    frozen = read_json(path)
    expected = {
        "schema": FREEZE_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "batch_id": batch_id,
        "git_commit": binding["git_commit"],
        "spec_path": str(spec_path),
        "spec_sha256": binding["spec_sha256"],
        "plan_sha256": binding["plan_sha256"],
        "selection_code_path": str(SCRIPT_PATH),
        "selection_code_sha256": binding["runner_sha256"],
        "development_split": "val_unseen",
        "model_seed": 0,
        "episode_order_seed": 0,
        "val_seen_consulted": False,
        "test_consulted": False,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise CampaignError("FROZEN.json {} mismatch".format(key))
    winners = frozen.get("winners")
    runs = frozen.get("development_runs")
    if not isinstance(winners, list) or len(winners) != 16:
        raise CampaignError("FROZEN.json does not bind sixteen winners")
    if not isinstance(runs, list) or len(runs) != 55:
        raise CampaignError("FROZEN.json does not bind 55 development runs")
    retry_histories = frozen.get("retry_history")
    if (
        not isinstance(retry_histories, list)
        or len(retry_histories) != 55
        or not valid_sha256(frozen.get("retry_history_sha256"))
        or hashlib.sha256(canonical(retry_histories).encode("utf-8")).hexdigest()
        != frozen["retry_history_sha256"]
    ):
        raise CampaignError("FROZEN.json retry-history binding is invalid")
    expected_retry_histories = []
    for record in runs:
        manifest = repo_file(record.get("run_manifest"), "frozen development manifest", True)
        if sha256(manifest) != record.get("run_manifest_sha256"):
            raise CampaignError("frozen development run manifest changed")
        history = record.get("retry_history")
        if (
            not isinstance(history, list)
            or record.get("retry_count") != len(history)
            or hashlib.sha256(canonical(history).encode("utf-8")).hexdigest()
            != record.get("retry_history_sha256")
        ):
            raise CampaignError("frozen development retry history changed")
        for item in history:
            archive = Path(str(item.get("archive_path", ""))).resolve()
            if not archive.is_file() or sha256(archive) != item.get("archive_sha256"):
                raise CampaignError("frozen retry archive changed")
        expected_retry_histories.append({
            "job_key": "q{:02d}-{}-{}".format(
                record["queue_id"], record["cell_id"], record["candidate_id"]
            ),
            "retry_count": record["retry_count"],
            "retry_history_sha256": record["retry_history_sha256"],
        })
    if retry_histories != expected_retry_histories:
        raise CampaignError("FROZEN.json retry histories do not match development runs")
    for winner in winners:
        config = repo_file(winner.get("frozen_config"), "frozen winner config", True)
        if sha256(config) != winner.get("frozen_config_sha256"):
            raise CampaignError("frozen winner config changed")
    source_manifests = frozen.get("source_manifests")
    if not isinstance(source_manifests, dict):
        raise CampaignError("FROZEN.json source_manifests must be an object")
    if source_controls_are_execution_gate(spec) and not source_manifests:
        raise CampaignError("FROZEN.json does not bind Source manifests")
    if not source_controls_are_execution_gate(spec) and source_manifests:
        raise CampaignError(
            "reporting-only Source results must not be frozen into selection"
        )
    for binding_key, source in source_manifests.items():
        ledger = repo_file(source.get("path"), "frozen Source ledger", True)
        if sha256(ledger) != source.get("sha256"):
            raise CampaignError("frozen Source ledger changed: {}".format(binding_key))
        formal = source.get("formal_manifests")
        if not isinstance(formal, dict) or not formal:
            raise CampaignError("frozen Source formal-manifest bindings are missing")
        for setting, item in formal.items():
            manifest = repo_file(
                item.get("path"), "frozen Source formal manifest", True
            )
            document = read_json(manifest)
            from tools.run_manifest_identity import immutable_identity_sha256
            identity = item.get("immutable_identity_sha256")
            if (
                sha256(manifest) != item.get("sha256")
                or not valid_sha256(identity)
                or document.get("immutable_identity_sha256") != identity
                or immutable_identity_sha256(document) != identity
            ):
                raise CampaignError(
                    "frozen Source formal manifest changed: {} {}".format(
                        binding_key, setting
                    )
                )
    return path, frozen


def require_val_seen_barrier(root, jobs):
    results = _completed_results(root, jobs, require_paired=True)
    if len(results) != 16:
        raise CampaignError("all sixteen val_seen jobs must validate before test")
    return results


def _assert_runtime_binding(root, spec_path):
    binding = read_json(Path(root) / "BATCH.json")
    checks = {
        "git_commit": git_commit(),
        "spec_sha256": sha256(spec_path),
        "runner_sha256": sha256(SCRIPT_PATH),
    }
    for key, expected in checks.items():
        if binding.get(key) != expected:
            raise CampaignError("runtime {} drifted from BATCH.json".format(key))
    assert_clean_formal_tree(spec_path)


@contextmanager
def campaign_lock(batch_id):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / ("." + batch_id + ".lock")
    stream = path.open("a+")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CampaignError("another scheduler owns batch {}".format(batch_id))
        stream.seek(0)
        stream.truncate()
        stream.write("pid={} host={} acquired={}\n".format(
            os.getpid(), platform.node(), utc_now()))
        stream.flush()
        yield
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def terminate_active_processes(reason=None):
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES.items())
    for process, attempt_dir in processes:
        if process.poll() is not None:
            continue
        if reason:
            try:
                metadata = read_json(Path(attempt_dir) / "job.json")
                identity = read_json(Path(attempt_dir) / "process_identity.json")
                marker_path = Path(attempt_dir) / "SCHEDULER_TERMINATION.json"
                marker = {
                    "schema": TERMINATION_SCHEMA,
                    "reason": str(reason),
                    "job_identity_sha256": metadata["job_identity_sha256"],
                    "pid": identity["pid"],
                    "process_group_id": identity["process_group_id"],
                    "start_token": identity["start_token"],
                    "termination_signal": int(signal.SIGTERM),
                    "recorded_at": utc_now(),
                }
                if marker_path.exists():
                    existing = read_json(marker_path)
                    for key in marker:
                        if key != "recorded_at" and existing.get(key) != marker[key]:
                            raise CampaignError(
                                "scheduler termination marker changed"
                            )
                else:
                    atomic_json(marker_path, marker)
            except Exception as error:
                _scheduler_log(
                    Path(attempt_dir).parents[5],
                    "could not record scheduler termination for {}: {}"
                    .format(attempt_dir, error),
                )
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 10.0
    for process, _attempt_dir in processes:
        remaining = max(0.0, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _load_stage_jobs(root, stage):
    path = _stage_plan_path(root, stage)
    plan = read_json(path)
    if plan.get("stage") != stage or not isinstance(plan.get("jobs"), list):
        raise CampaignError("invalid {} stage plan".format(stage))
    return plan["jobs"]


def write_val_seen_report(root, jobs):
    rows = _completed_results(root, jobs, require_paired=True)
    records = []
    for row in sorted(rows, key=lambda value: value["queue_id"]):
        records.append({
            "cell_id": row["cell_id"],
            "setting": row["setting"],
            "method": row["method"],
            "supervision": row["supervision"],
            "candidate_id": row["candidate_id"],
            "metrics": row["metrics"],
            "run_manifest": row["formal_manifest"],
            "run_manifest_sha256": row["formal_manifest_sha256"],
            "paired_episode_metrics": row["paired_episode_metrics"],
            "paired_episode_metrics_sha256": row["paired_episode_metrics_sha256"],
        })
    payload = {
        "schema": "navtta.vln_targeted_gap_val_seen_results.v1",
        "split": "val_seen",
        "used_for_selection": False,
        "job_count": 16,
        "unsupervised": [item for item in records if item["supervision"] == "unsupervised"],
        "feedback_supervised": [item for item in records if item["supervision"] == "feedback_supervised"],
    }
    atomic_json(Path(root) / "VAL_SEEN_RESULTS.json", payload)
    return payload


def write_test_report(root, jobs):
    rows = _completed_results(root, jobs, require_paired=False)
    if len(rows) != 5 or any(row.get("metrics") is not None for row in rows):
        raise CampaignError("REVERIE hidden test is submission-only")
    records = []
    for row in sorted(rows, key=lambda value: value["ordinal"]):
        records.append({
            "submission_id": row["submission_id"],
            "source_cell_id": row["cell_id"],
            "setting": row["setting"],
            "reported_method_label": row["reported_method_label"],
            "supervision": row["supervision"],
            "feedback_provider": row["feedback_provider"],
            "candidate_id": row["candidate_id"],
            "submission_path": row["prediction_artifact"],
            "submission_sha256": row["prediction_artifact_sha256"],
            "run_manifest": row["formal_manifest"],
            "run_manifest_sha256": row["formal_manifest_sha256"],
        })
    payload = {
        "schema": "navtta.vln_targeted_gap_reverie_submissions.v1",
        "split": "test",
        "local_metrics": None,
        "selection_performed": False,
        "job_count": 5,
        "unsupervised": [item for item in records if item["supervision"] == "unsupervised"],
        "pseudo_feedback": [item for item in records if item["supervision"] == "pseudo_label"],
    }
    atomic_json(Path(root) / "REVERIE_TEST_SUBMISSIONS.json", payload)
    return payload


def _dry_commands(spec, batch_id, jobs):
    values = []
    for job in jobs:
        run_tag = _attempt_tag(batch_id, job, 1)
        result_root = _result_root(batch_id, job, run_tag)
        config_path = _job_parent(batch_root(batch_id), job) / "attempt-001/resolved_config.json"
        command = _command(job, config_path.resolve(), result_root, run_tag)
        values.append({
            "queue": "q{:02d}".format(job["queue_id"]),
            "gpu": job["gpu"],
            "cell_id": job["cell_id"],
            "candidate_id": job["candidate_id"],
            "config": _job_config(spec, batch_id, job, result_root),
            "command": command,
            "shell": shlex.join(command),
        })
    return values


def print_plan(spec_path, spec, batch_id, gpus, stage="search", jobs=None):
    jobs = jobs or expand_search_jobs(spec, gpus)
    grouped = defaultdict(list)
    for job in jobs:
        grouped[job["gpu"]].append(job)
    output = {
        "schema": "navtta.vln_targeted_gap_dry_plan.v1",
        "batch_id": batch_id,
        "git_commit": git_commit(),
        "spec_path": str(spec_path),
        "spec_sha256": sha256(spec_path),
        "runner_sha256": sha256(SCRIPT_PATH),
        "stage": stage,
        "job_count": len(jobs),
        "queue_count": len({job["queue_id"] for job in jobs}),
        "max_active_processes_per_gpu": 4,
        "jobs_by_gpu": {str(gpu): len(grouped[gpu]) for gpu in gpus},
        "jobs": _dry_commands(spec, batch_id, jobs),
    }
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))
    return output


def status_payload(spec, batch_id, gpus, spec_path=None):
    root = batch_root(batch_id)
    payload = {
        "schema": "navtta.vln_targeted_gap_status.v1",
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "spec_path": str(Path(spec_path).resolve()) if spec_path else None,
        "spec_sha256": sha256(spec_path) if spec_path else None,
        "runner_sha256": sha256(SCRIPT_PATH),
        "log_root": str(root.resolve()),
        "tuning_root": str(tuning_batch_root(batch_id).resolve()),
        "expected": {"search": 55, "val-seen": 16, "reverie-test": 5},
        "batch_exists": (root / "BATCH.json").is_file(),
        "stages": {},
        "frozen": (root / "FROZEN.json").is_file(),
    }
    search = expand_search_jobs(spec, gpus)
    payload["stages"]["search"] = _state_counts(root, search)
    if payload["frozen"]:
        frozen = read_json(root / "FROZEN.json")
        for stage in ("val-seen", "reverie-test"):
            jobs = expand_frozen_jobs(spec, gpus, frozen, stage, batch_id)
            payload["stages"][stage] = _state_counts(root, jobs)
    return payload


def _parse_gpus(value):
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("GPU list must contain integers")
    if len(values) != 4 or len(set(values)) != 4 or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("exactly four unique nonnegative GPUs are required")
    return values


def _actual_batch_id_from_spec(spec):
    return Path(spec["freeze"]["artifact"]).parent.name


def _barrier_jobs(spec, gpus, frozen, stage, batch_id):
    return expand_frozen_jobs(spec, gpus, frozen, stage, batch_id)


def main(argv=None):
    def _term(_signum, _frame):
        terminate_active_processes("scheduler_signal_{}".format(_signum))
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "plan", "run", "search", "freeze", "val-seen",
            "reverie-test", "status",
        ),
        help=(
            "run executes search, freeze, and val-seen in order; "
            "reverie-test remains a separate submission phase"
        ),
    )
    parser.add_argument(
        "--spec", default=str(DEFAULT_SPEC),
        help=(
            "campaign JSON; defaults to the active direct-search v2 spec"
        ),
    )
    parser.add_argument(
        "--batch-id",
        help=(
            "immutable batch identity; defaults to <spec-id>-seed0 as bound "
            "by freeze.artifact"
        ),
    )
    parser.add_argument(
        "--gpus", type=_parse_gpus, default=(0, 1, 2, 3),
        help="exactly 0,1,2,3; the reviewed queue-to-GPU mapping is immutable",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the same stage identity and skip only revalidated successes",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="with --resume, retry only independently classified infrastructure failures",
    )
    parser.add_argument(
        "--retry-reason",
        help="required operator justification when --retry-failed is used",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print a phase plan without launching workers",
    )
    args = parser.parse_args(argv)

    spec_path, spec = load_spec(args.spec)
    args.batch_id = args.batch_id or _actual_batch_id_from_spec(spec)
    _safe_component(args.batch_id, "batch id")
    gpus = tuple(args.gpus)
    if list(gpus) != spec["schedule"]["gpu_ids"]:
        raise CampaignError("GPU reassignment is forbidden by the campaign spec")
    if args.retry_failed and not args.resume:
        raise CampaignError("--retry-failed requires --resume")
    if args.retry_failed:
        args.retry_reason = _retry_reason(args.retry_reason)
    elif args.retry_reason is not None:
        raise CampaignError("--retry-reason requires --retry-failed")
    if args.stage in ("plan", "status") and (
        args.resume or args.retry_failed or args.retry_reason is not None
    ):
        raise CampaignError("plan/status do not accept resume flags")

    if args.stage == "plan":
        print_plan(spec_path, spec, args.batch_id, gpus)
        return 0
    if args.stage == "status":
        print(json.dumps(status_payload(
            spec, args.batch_id, gpus, spec_path=spec_path
        ), indent=2,
                         sort_keys=True, ensure_ascii=False))
        return 0

    if args.stage == "run":
        if args.dry_run:
            print_plan(
                spec_path, spec, args.batch_id, gpus, "search",
                expand_search_jobs(spec, gpus),
            )
            return 0
        forwarded_common = [
            "--spec", str(spec_path),
            "--batch-id", args.batch_id,
            "--gpus", ",".join(str(value) for value in gpus),
        ]
        for phase in ("search", "freeze", "val-seen"):
            forwarded = [phase] + forwarded_common
            if args.resume and phase in ("search", "val-seen"):
                forwarded.append("--resume")
            if args.retry_failed and phase in ("search", "val-seen"):
                forwarded.extend([
                    "--retry-failed", "--retry-reason", args.retry_reason,
                ])
            main(forwarded)
        print(
            "completed search, freeze, and val-seen for {}"
            .format(args.batch_id)
        )
        return 0

    require_launch_ready(spec_path, spec)

    root = batch_root(args.batch_id)
    if args.stage == "search":
        jobs = expand_search_jobs(spec, gpus)
        if args.dry_run:
            print_plan(spec_path, spec, args.batch_id, gpus, "search", jobs)
            return 0
        if args.batch_id != _actual_batch_id_from_spec(spec):
            raise CampaignError("formal batch id must match freeze.artifact")
        formal_preflight(spec_path, spec, "search")
        with campaign_lock(args.batch_id):
            root, binding, _ = prepare_batch(
                spec_path, spec, args.batch_id, gpus, args.resume
            )
            run_stage(
                spec_path, spec, args.batch_id, root, binding, "search", jobs,
                resume=args.resume,
                retry_failed=args.retry_failed,
                retry_reason=args.retry_reason,
            )
        return 0

    root, binding, _ = validate_batch(
        spec_path, spec, args.batch_id, gpus
    )
    if args.stage == "freeze":
        if args.dry_run:
            print(json.dumps({
                "stage": "freeze",
                "search_states": _state_counts(root, expand_search_jobs(spec, gpus)),
                "would_write": str(root / "FROZEN.json"),
            }, indent=2, sort_keys=True))
            return 0
        if direct_execution_mode(spec):
            sources = {}
        else:
            assert_clean_formal_tree(spec_path)
            sources = validate_source_controls(spec)
            validate_idea_assets(spec)
        with campaign_lock(args.batch_id):
            freeze_campaign(
                spec_path, spec, args.batch_id, root, binding,
                expand_search_jobs(spec, gpus), sources,
            )
        return 0

    frozen_path, frozen = load_frozen(
        spec_path, spec, args.batch_id, root, binding
    )
    jobs = _barrier_jobs(spec, gpus, frozen, args.stage, args.batch_id)
    if args.dry_run:
        print_plan(spec_path, spec, args.batch_id, gpus, args.stage, jobs)
        return 0
    if args.stage == "val-seen":
        preflight = formal_preflight(spec_path, spec, "val-seen")
        with campaign_lock(args.batch_id):
            freeze_campaign(
                spec_path, spec, args.batch_id, root, binding,
                expand_search_jobs(spec, gpus), preflight["source_controls"],
            )
            run_stage(
                spec_path, spec, args.batch_id, root, binding, "val-seen", jobs,
                resume=args.resume,
                retry_failed=args.retry_failed,
                retry_reason=args.retry_reason,
            )
            write_val_seen_report(root, jobs)
        return 0
    if args.stage == "reverie-test":
        preflight = formal_preflight(spec_path, spec, "reverie-test")
        provider_preflight = preflight["provider"]["preflight"]
        with campaign_lock(args.batch_id):
            freeze_campaign(
                spec_path, spec, args.batch_id, root, binding,
                expand_search_jobs(spec, gpus), preflight["source_controls"],
            )
            _, frozen = load_frozen(
                spec_path, spec, args.batch_id, root, binding
            )
            jobs = _barrier_jobs(
                spec, gpus, frozen, "reverie-test", args.batch_id
            )
            jobs = bind_provider_preflight(jobs, provider_preflight)
            val_jobs = _barrier_jobs(
                spec, gpus, frozen, "val-seen", args.batch_id
            )
            require_val_seen_barrier(root, val_jobs)
            run_stage(
                spec_path, spec, args.batch_id, root, binding,
                "reverie-test", jobs, resume=args.resume,
                retry_failed=args.retry_failed,
                retry_reason=args.retry_reason,
                provider_preflight=provider_preflight,
            )
            write_test_report(root, jobs)
        return 0
    raise CampaignError("unhandled stage")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        terminate_active_processes("operator_keyboard_interrupt")
        print("interrupted; active worker process groups were terminated", file=sys.stderr)
        raise SystemExit(130)
    except CampaignError as error:
        terminate_active_processes()
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
