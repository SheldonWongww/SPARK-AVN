#!/usr/bin/env python3
"""Materialize the immutable REVERIE val_unseen plan from final winners.

Run this only after the small val_seen search has been downloaded and
``build_reverie_final_registry.py`` has produced a complete registry.  The
generated specification contains zero Source jobs and pins all 15 TTA winner
identities.  It also records the separate hidden-test submission policy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for value in (str(REPO_ROOT), str(SCRIPT_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

import build_reverie_final_registry as registry_builder  # noqa: E402
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


SCHEMA = "navtta.vln_reverie_val_unseen_frozen_eval.v1"
SOURCE_SCHEMA = "navtta.vln_reverie_val_unseen_reused_source_controls.v1"
TEST_SOURCE_SCHEMA = "navtta.vln_reverie_test_reused_source_submissions.v1"
DEFAULT_REGISTRY = REPO_ROOT / "vln/results/final/reverie/registry.json"
DEFAULT_SOURCE = (
    REPO_ROOT / "vln/manifests/reverie_val_unseen_reused_source_controls.json"
)
DEFAULT_TEST_SOURCE = (
    REPO_ROOT / "vln/manifests/reverie_test_reused_source_submissions.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "vln/experiments/reverie_val_unseen_frozen_eval_v1.json"
)
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"

SETTINGS = registry_builder.SETTINGS
MODELS = registry_builder.MODELS
METHODS = registry_builder.METHODS
MODEL_FOR_SETTING = registry_builder.MODEL_FOR_SETTING
METRICS = registry_builder.METRICS
VAL_UNSEEN_EPISODES = 3521
VAL_UNSEEN_ORDER = (
    "67c909272166bb3d8e7ee612d597de190f9218a6342da6357b0c7aea8832baca"
)
TEST_EPISODES = 6292
TEST_ORDER = (
    "5f527c1d8442ae2b003cc5b429164eaf003b2e7acc16a8babd818874d3e1ab1d"
)
HEX64 = re.compile(r"[0-9a-f]{64}")


class PlanError(RuntimeError):
    pass


def _read_json(path, label="JSON"):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PlanError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise PlanError("{} must be an object".format(label))
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _repo_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        raise PlanError("plan evidence escapes repository: {}".format(path))


def _resolve(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _validate_order_manifest(path, digest, benchmark, split, episodes, order):
    path = Path(path).resolve()
    if not path.is_file() or _sha256(path) != digest:
        raise PlanError("{} {} order-manifest digest mismatch".format(benchmark, split))
    value = _read_json(path, "episode-order manifest")
    expected = {
        "schema": "navtta.episode_order.v1",
        "benchmark": benchmark,
        "split": split,
        "episode_count": episodes,
        "order_sha256": order,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise PlanError("episode-order {} mismatch".format(key))
    return path


def validate_val_unseen_source(path=DEFAULT_SOURCE,
                               require_metric_artifacts=False):
    path = Path(path).resolve()
    ledger = _read_json(path, "REVERIE val_unseen Source ledger")
    expected = {
        "schema": SOURCE_SCHEMA,
        "benchmark": "reverie",
        "split": "val_unseen",
        "source_protocol": "standard_argmax",
        "episode_count": VAL_UNSEEN_EPISODES,
        "canonical_order_seed": 0,
        "episode_order_sha256": VAL_UNSEEN_ORDER,
    }
    for key, expected_value in expected.items():
        if ledger.get(key) != expected_value:
            raise PlanError("val_unseen Source ledger {} mismatch".format(key))
    policy = ledger.get("source_execution_policy", {})
    if policy.get("execution") != "reuse_only" or policy.get(
        "rerun_forbidden"
    ) is not True:
        raise PlanError("val_unseen Source must be reuse-only")
    source_batch = ledger.get("source_batch", {})
    if (
        source_batch.get("batch_id") != "grouped-source-20260810T080743Z"
        or not re.fullmatch(r"[0-9a-f]{40}", str(source_batch.get("git_commit", "")))
    ):
        raise PlanError("val_unseen Source batch provenance is invalid")
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise PlanError("val_unseen Source ledger matrix is incomplete")
    for setting in SETTINGS:
        record = records[setting]
        label = "{} val_unseen Source".format(setting)
        if record.get("model") != MODEL_FOR_SETTING[setting]:
            raise PlanError("{} model mismatch".format(label))
        if record.get("evidence_status") != "ready":
            raise PlanError("{} evidence is not ready".format(label))
        if record.get("parameters") != {
            "action_selection": "argmax", "action_seed": 0
        }:
            raise PlanError("{} protocol mismatch".format(label))
        try:
            registry_builder._validate_metrics(record.get("metrics"), label)
        except registry_builder.RegistryError as error:
            raise PlanError(str(error))
        if record.get("episode_order_sha256") != VAL_UNSEEN_ORDER:
            raise PlanError("{} order mismatch".format(label))
        formal = _resolve(record.get("formal_manifest_path", "")).resolve()
        expected_run_id = "{}-{}-val_unseen-native".format(
            record["run_tag"], setting
        )
        try:
            formal.relative_to(FORMAL_ROOT.resolve())
        except ValueError:
            raise PlanError("{} manifest is noncanonical".format(label))
        if formal.name != "manifest.json" or formal.parent.name != expected_run_id:
            raise PlanError("{} manifest path is noncanonical".format(label))
        if not formal.is_file() or _sha256(formal) != record.get(
            "formal_manifest_sha256"
        ):
            raise PlanError("{} manifest digest mismatch".format(label))
        manifest = _read_json(formal, label + " manifest")
        expected_manifest = {
            "run_id": expected_run_id,
            "task": "vln",
            "benchmark": registry_builder.EXPECTED_BENCHMARK[setting],
            "model": MODEL_FOR_SETTING[setting],
            "method": "source",
            "run_tag": record["run_tag"],
            "source_setting": "{}:val_unseen:native".format(setting),
            "seed": 0,
            "git_commit": source_batch["git_commit"],
            "status": "completed",
            "exit_code": 0,
        }
        for key, expected_value in expected_manifest.items():
            if manifest.get(key) != expected_value:
                raise PlanError("{} manifest {} mismatch".format(label, key))
        identity = manifest.get("immutable_identity_sha256")
        if (
            identity != record.get("immutable_identity_sha256")
            or immutable_identity_sha256(manifest) != identity
        ):
            raise PlanError("{} immutable identity mismatch".format(label))
        dataset = manifest.get("dataset", {})
        if (
            dataset.get("stream_content_sha256") != record.get("dataset_sha256")
            or dataset.get("stream_order_sha256") != VAL_UNSEEN_ORDER
        ):
            raise PlanError("{} dataset/order mismatch".format(label))
        order_digest = (
            "1e5ca8cd785c54def1a7225ecc3cf420a1ef28c1e90e8d02780d58f2709c2db3"
            if setting == "goat-reverie"
            else "09b2d4984d0073a6b31dc0ea6057a70c1436b8d54ebb463b901065b23f0c9e47"
        )
        order_name = "reverie_goat" if setting == "goat-reverie" else "reverie_duet_hamt"
        order_path = REPO_ROOT / "vln/manifests/episode_order" / order_name / "val_unseen.json"
        _validate_order_manifest(
            order_path, order_digest, registry_builder.EXPECTED_BENCHMARK[setting],
            "val_unseen", VAL_UNSEEN_EPISODES, VAL_UNSEEN_ORDER,
        )
        order_document = _read_json(order_path, label + " episode order")
        if order_document.get("dataset", {}).get("sha256") != record.get(
            "dataset_sha256"
        ):
            raise PlanError("{} order-manifest dataset mismatch".format(label))
        if manifest.get("pinned_manifests", {}).get(
            "episode_order", {}
        ).get("sha256") != order_digest:
            raise PlanError("{} pinned order-manifest digest mismatch".format(label))
        if manifest.get("checkpoint", {}).get("sha256") != record.get(
            "checkpoint_sha256"
        ):
            raise PlanError("{} checkpoint mismatch".format(label))
        authenticated = [
            item for item in manifest.get("result_artifacts", [])
            if isinstance(item, dict)
            and item.get("sha256") == record.get("metrics_artifact_sha256")
            and str(item.get("name", "")).endswith("valid.txt")
        ]
        if len(authenticated) != 1:
            raise PlanError("{} metric artifact is not manifest-authenticated".format(label))
        artifact = _resolve(record.get("metrics_artifact_path", "")).resolve()
        if artifact.is_file():
            if (
                artifact.stat().st_size != authenticated[0].get("size")
                or _sha256(artifact) != record.get("metrics_artifact_sha256")
            ):
                raise PlanError("{} raw metric evidence changed".format(label))
            try:
                parsed = registry_builder._parse_reverie_metrics(
                    artifact, "val_unseen", label
                )
                registry_builder._assert_metrics_equal(
                    parsed, record["metrics"], label
                )
            except registry_builder.RegistryError as error:
                raise PlanError(str(error))
        elif require_metric_artifacts:
            raise PlanError("{} raw metric evidence is unavailable".format(label))
    return path, ledger


def validate_test_source(path=DEFAULT_TEST_SOURCE,
                         require_submission_artifacts=False):
    path = Path(path).resolve()
    ledger = _read_json(path, "REVERIE test Source-submission ledger")
    expected = {
        "schema": TEST_SOURCE_SCHEMA,
        "benchmark": "reverie",
        "split": "test",
        "episode_count": TEST_EPISODES,
        "canonical_order_seed": 0,
        "episode_order_sha256": TEST_ORDER,
        "hidden_ground_truth": True,
        "local_metrics_available": False,
    }
    for key, expected_value in expected.items():
        if ledger.get(key) != expected_value:
            raise PlanError("test Source ledger {} mismatch".format(key))
    policy = ledger.get("source_execution_policy", {})
    if policy != {
        "execution": "reuse_existing_submission_only",
        "rerun_forbidden": True,
    }:
        raise PlanError("test Source reuse policy mismatch")
    source_batch = ledger.get("source_batch", {})
    if (
        source_batch.get("batch_id") != "grouped-source-20260810T080743Z"
        or not re.fullmatch(r"[0-9a-f]{40}", str(source_batch.get("git_commit", "")))
    ):
        raise PlanError("test Source batch provenance is invalid")
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise PlanError("test Source submission matrix is incomplete")
    for setting in SETTINGS:
        record = records[setting]
        label = "{} test Source".format(setting)
        if record.get("model") != MODEL_FOR_SETTING[setting]:
            raise PlanError("{} model mismatch".format(label))
        formal = _resolve(record.get("formal_manifest_path", "")).resolve()
        expected_run_id = "{}-{}-test-native".format(record["run_tag"], setting)
        try:
            formal.relative_to(FORMAL_ROOT.resolve())
        except ValueError:
            raise PlanError("{} formal manifest is noncanonical".format(label))
        if formal.name != "manifest.json" or formal.parent.name != expected_run_id:
            raise PlanError("{} formal manifest path is noncanonical".format(label))
        if not formal.is_file() or _sha256(formal) != record.get(
            "formal_manifest_sha256"
        ):
            raise PlanError("{} formal manifest mismatch".format(label))
        manifest = _read_json(formal, label + " manifest")
        expected_manifest = {
            "run_id": expected_run_id,
            "task": "vln",
            "benchmark": registry_builder.EXPECTED_BENCHMARK[setting],
            "model": MODEL_FOR_SETTING[setting],
            "method": "source",
            "run_tag": record["run_tag"],
            "source_setting": "{}:test:native".format(setting),
            "seed": 0,
            "git_commit": ledger["source_batch"]["git_commit"],
            "status": "completed",
            "exit_code": 0,
        }
        for key, expected_value in expected_manifest.items():
            if manifest.get(key) != expected_value:
                raise PlanError("{} manifest {} mismatch".format(label, key))
        dataset = manifest.get("dataset", {})
        if (
            record.get("episode_order_sha256") != TEST_ORDER
            or dataset.get("stream_order_sha256") != TEST_ORDER
            or dataset.get("stream_content_sha256") != record.get("dataset_sha256")
        ):
            raise PlanError("{} dataset/episode-order mismatch".format(label))
        if manifest.get("checkpoint", {}).get("sha256") != record.get(
            "checkpoint_sha256"
        ):
            raise PlanError("{} checkpoint mismatch".format(label))
        order_key = "goat" if setting == "goat-reverie" else "duet_hamt"
        order_path = REPO_ROOT / "vln/manifests/episode_order/reverie_{}/test.json".format(
            order_key
        )
        order_digest = (
            "543b7de61221e1a4e8a29c074ea0f8abf7551da9ceeb198614a8c55288435146"
            if order_key == "goat"
            else "003f882a1e43537b2f046590e2fd3215f3a02266b57fb8ab51b1363d41f3382e"
        )
        _validate_order_manifest(
            order_path, order_digest, registry_builder.EXPECTED_BENCHMARK[setting],
            "test", TEST_EPISODES, TEST_ORDER,
        )
        order_document = _read_json(order_path, label + " episode order")
        if order_document.get("dataset", {}).get("sha256") != record.get(
            "dataset_sha256"
        ):
            raise PlanError("{} order-manifest dataset mismatch".format(label))
        if manifest.get("pinned_manifests", {}).get(
            "episode_order", {}
        ).get("sha256") != order_digest:
            raise PlanError("{} pinned order-manifest digest mismatch".format(label))
        identity = manifest.get("immutable_identity_sha256")
        if (
            identity != record.get("immutable_identity_sha256")
            or immutable_identity_sha256(manifest) != identity
        ):
            raise PlanError("{} immutable identity mismatch".format(label))
        artifacts = [
            item for item in manifest.get("result_artifacts", [])
            if isinstance(item, dict)
            and item.get("sha256") == record.get("submission_artifact_sha256")
            and item.get("size") == record.get("submission_artifact_size")
        ]
        if len(artifacts) != 1:
            raise PlanError("{} submission is not manifest-authenticated".format(label))
        submission = _resolve(record.get("submission_artifact_path", "")).resolve()
        if submission.is_file():
            if (
                submission.stat().st_size != record["submission_artifact_size"]
                or _sha256(submission) != record["submission_artifact_sha256"]
            ):
                raise PlanError("{} submission artifact changed".format(label))
            try:
                payload = json.loads(submission.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise PlanError("{} invalid submission JSON: {}".format(label, error))
            expected_ids = [
                str(item["episode_id"]) for item in order_document["episodes"]
            ]
            if not isinstance(payload, list) or len(payload) != TEST_EPISODES:
                raise PlanError("{} submission episode count mismatch".format(label))
            actual_ids = []
            for index, item in enumerate(payload):
                if (
                    not isinstance(item, dict)
                    or not {"instr_id", "trajectory", "predObjId"}.issubset(item)
                    or not isinstance(item["trajectory"], list)
                ):
                    raise PlanError("{} submission row {} is malformed".format(label, index))
                actual_ids.append(str(item["instr_id"]))
            if actual_ids != expected_ids or len(set(actual_ids)) != TEST_EPISODES:
                raise PlanError("{} submission IDs/order mismatch".format(label))
        elif require_submission_artifacts:
            raise PlanError("{} submission artifact is unavailable".format(label))
    return path, ledger


def load_registry(path=DEFAULT_REGISTRY):
    path = Path(path).resolve()
    raw = _read_json(path, "REVERIE final registry")
    selection = _resolve(raw.get("selected_winners", {}).get("path", ""))
    try:
        registry = registry_builder.validate_registry(path, selection)
    except registry_builder.RegistryError as error:
        raise PlanError("REVERIE winner registry failed: {}".format(error))
    return path, registry


def validate_cross_split_checkpoints(registry, val_unseen_source,
                                     test_source):
    for setting in SETTINGS:
        registry_hashes = {
            record.get("checkpoint_sha256")
            for record in registry["records"][setting].values()
        }
        val_unseen_hash = val_unseen_source["records"][setting].get(
            "checkpoint_sha256"
        )
        test_hash = test_source["records"][setting].get("checkpoint_sha256")
        if (
            len(registry_hashes) != 1
            or registry_hashes != {val_unseen_hash}
            or test_hash != val_unseen_hash
        ):
            raise PlanError(
                "{} checkpoint differs across val_seen registry, val_unseen "
                "Source, and test Source".format(setting)
            )
    return True


def _test_policy(test_source_path):
    return {
        "benchmark": "reverie",
        "native_split": "test",
        "episode_count": TEST_EPISODES,
        "canonical_order_seed": 0,
        "episode_order_sha256": TEST_ORDER,
        "hidden_ground_truth": True,
        "execution_purpose": "submission_generation_only",
        "local_metric_computation_forbidden": True,
        "selection_or_ranking_on_test_forbidden": True,
        "source": {
            "execution": "reuse_existing_submission_only",
            "rerun_forbidden": True,
            "ledger": _repo_path(test_source_path),
            "sha256": _sha256(test_source_path),
        },
        "order_manifests": {
            "duet_hamt": {
                "path": "vln/manifests/episode_order/reverie_duet_hamt/test.json",
                "sha256": "003f882a1e43537b2f046590e2fd3215f3a02266b57fb8ab51b1363d41f3382e",
                "settings": ["duet-reverie", "hamt-reverie"],
            },
            "goat": {
                "path": "vln/manifests/episode_order/reverie_goat/test.json",
                "sha256": "543b7de61221e1a4e8a29c074ea0f8abf7551da9ceeb198614a8c55288435146",
                "settings": ["goat-reverie"],
            },
        },
        "eligible_tta_methods": ["tent", "fstta", "eam"],
        "eligible_tta_submission_jobs": 9,
        "unavailable_without_legal_online_feedback": {
            "feedtta": {
                "status": "N/A",
                "launch_policy": "fail_closed",
                "reason": "hidden test labels provide no per-episode binary navigation-success feedback",
            },
            "atena": {
                "status": "N/A",
                "launch_policy": "fail_closed",
                "reason": "hidden test labels provide no legal queried binary navigation-success feedback",
            },
        },
        "legal_online_feedback_interface": None,
    }


def assert_test_method_allowed(policy, method):
    if method in policy.get("unavailable_without_legal_online_feedback", {}):
        raise PlanError(
            "{} is N/A on hidden REVERIE test without a legal online binary "
            "feedback interface".format(method)
        )
    if method not in policy.get("eligible_tta_methods", []):
        raise PlanError("{} is not an eligible test TTA method".format(method))
    return True


def build_spec(registry_path=DEFAULT_REGISTRY, source_path=DEFAULT_SOURCE,
               test_source_path=DEFAULT_TEST_SOURCE):
    registry_path, registry = load_registry(registry_path)
    source_path, source = validate_val_unseen_source(source_path)
    test_source_path, test_source = validate_test_source(test_source_path)
    validate_cross_split_checkpoints(registry, source, test_source)
    order_bindings = {
        "duet_hamt": {
            "path": "vln/manifests/episode_order/reverie_duet_hamt/val_unseen.json",
            "sha256": "09b2d4984d0073a6b31dc0ea6057a70c1436b8d54ebb463b901065b23f0c9e47",
            "settings": ["duet-reverie", "hamt-reverie"],
        },
        "goat": {
            "path": "vln/manifests/episode_order/reverie_goat/val_unseen.json",
            "sha256": "1e5ca8cd785c54def1a7225ecc3cf420a1ef28c1e90e8d02780d58f2709c2db3",
            "settings": ["goat-reverie"],
        },
    }
    _validate_order_manifest(
        _resolve(order_bindings["duet_hamt"]["path"]),
        order_bindings["duet_hamt"]["sha256"],
        "reverie_discrete_duet_hamt", "val_unseen",
        VAL_UNSEEN_EPISODES, VAL_UNSEEN_ORDER,
    )
    _validate_order_manifest(
        _resolve(order_bindings["goat"]["path"]),
        order_bindings["goat"]["sha256"],
        "reverie_discrete_goat", "val_unseen",
        VAL_UNSEEN_EPISODES, VAL_UNSEEN_ORDER,
    )
    jobs = []
    for setting in SETTINGS:
        for method in METHODS:
            record = registry["records"][setting][method]
            jobs.append({
                "setting": setting,
                "model": MODEL_FOR_SETTING[setting],
                "method": method,
                "selected_run_tag": record["run_tag"],
                "selected_formal_manifest_sha256": record[
                    "formal_manifest_sha256"
                ],
            })
    test_policy = _test_policy(test_source_path)
    _validate_order_manifest(
        _resolve(test_policy["order_manifests"]["duet_hamt"]["path"]),
        test_policy["order_manifests"]["duet_hamt"]["sha256"],
        "reverie_discrete_duet_hamt", "test", TEST_EPISODES, TEST_ORDER,
    )
    _validate_order_manifest(
        _resolve(test_policy["order_manifests"]["goat"]["path"]),
        test_policy["order_manifests"]["goat"]["sha256"],
        "reverie_discrete_goat", "test", TEST_EPISODES, TEST_ORDER,
    )
    spec = {
        "schema": SCHEMA,
        "experiment_id": "vln-reverie-val-unseen-frozen-eval-v1",
        "status": "materialized_from_complete_val_seen_registry",
        "purpose": (
            "Evaluate all 15 frozen REVERIE val_seen winners on canonical "
            "val_unseen without Source execution or val_unseen reselection."
        ),
        "registry_dependency": {
            "path": _repo_path(registry_path),
            "sha256": _sha256(registry_path),
            "schema": registry_builder.REGISTRY_SCHEMA,
            "required_status": "complete",
        },
        "source_control": {
            "manifest": _repo_path(source_path),
            "sha256": _sha256(source_path),
            "execution": "reuse_only",
            "rerun_forbidden": True,
        },
        "protocol": {
            "benchmark": "reverie",
            "split": "val_unseen",
            "episode_count": VAL_UNSEEN_EPISODES,
            "canonical_order_seed": 0,
            "episode_order_sha256": VAL_UNSEEN_ORDER,
            "order_seed_cli_forbidden": True,
            "full_split_only": True,
            "frozen_from": "reverie_val_seen_seed0_final_registry",
            "selection_on_val_unseen": False,
            "reported_metrics": list(METRICS),
            "order_manifests": order_bindings,
            "binary_feedback": {
                "label": "navigation_success_only",
                "grounding_feedback_forbidden": True,
                "simulator_distance_fallback_forbidden": True,
                "feedtta_timing": "eager_once_per_episode",
                "atena_timing": "lazy_only_when_entropy_gate_queries",
            },
        },
        "matrix": {
            "model_order": list(MODELS),
            "setting_order": list(SETTINGS),
            "method_order": list(METHODS),
            "strict_model_barrier": True,
            "parallel_methods_within_model": True,
            "jobs": jobs,
        },
        "execution": {
            "model_phase_order": "duet_then_hamt_then_goat",
            "max_workers_by_model": {"duet": 2, "hamt": 2, "goat": 4},
            "estimated_gpu_memory_mib_by_method": {
                "tent": 4096,
                "fstta": 5120,
                "eam": 6144,
                "feedtta": 8192,
                "atena": 10240,
            },
            "max_aggregate_gpu_memory_mib": 29000,
            "minimum_free_gpu_memory_mib": 4096,
            "emergency_abort_used_gpu_memory_mib": 30500,
            "launch_stagger_seconds": 5,
            "poll_seconds": 2,
            "formal_requires_clean_tracked_tree": True,
            "formal_requires_review_confirmation": True,
            "retry_assigns_new_run_tag": True,
        },
        "budget": {
            "source_execution_jobs": 0,
            "tta_jobs": 15,
            "total_executed_jobs": 15,
        },
        "test_submission_policy": test_policy,
    }
    validate_spec_document(spec)
    return spec


def validate_spec_document(spec):
    if spec.get("schema") != SCHEMA:
        raise PlanError("unsupported frozen-eval schema")
    protocol = spec.get("protocol", {})
    expected = {
        "benchmark": "reverie",
        "split": "val_unseen",
        "episode_count": VAL_UNSEEN_EPISODES,
        "canonical_order_seed": 0,
        "episode_order_sha256": VAL_UNSEEN_ORDER,
        "order_seed_cli_forbidden": True,
        "full_split_only": True,
        "selection_on_val_unseen": False,
        "reported_metrics": list(METRICS),
    }
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise PlanError("frozen-eval protocol {} mismatch".format(key))
    matrix = spec.get("matrix", {})
    if (
        tuple(matrix.get("model_order", ())) != MODELS
        or tuple(matrix.get("setting_order", ())) != SETTINGS
        or tuple(matrix.get("method_order", ())) != METHODS
        or matrix.get("strict_model_barrier") is not True
        or matrix.get("parallel_methods_within_model") is not True
    ):
        raise PlanError("frozen-eval model/method ordering contract mismatch")
    jobs = matrix.get("jobs")
    expected_pairs = {(setting, method) for setting in SETTINGS for method in METHODS}
    if (
        not isinstance(jobs, list)
        or len(jobs) != 15
        or {(item.get("setting"), item.get("method")) for item in jobs}
        != expected_pairs
        or any(item.get("model") != MODEL_FOR_SETTING.get(item.get("setting")) for item in jobs)
    ):
        raise PlanError("frozen-eval matrix must be exact 3 x 5 TTA")
    if spec.get("budget") != {
        "source_execution_jobs": 0,
        "tta_jobs": 15,
        "total_executed_jobs": 15,
    }:
        raise PlanError("frozen-eval budget must be 15 TTA + 0 Source")
    execution = spec.get("execution", {})
    if execution.get("max_workers_by_model") != {
        "duet": 2, "hamt": 2, "goat": 4
    }:
        raise PlanError("reviewed per-model concurrency changed")
    expected_execution = {
        "model_phase_order": "duet_then_hamt_then_goat",
        "estimated_gpu_memory_mib_by_method": {
            "tent": 4096,
            "fstta": 5120,
            "eam": 6144,
            "feedtta": 8192,
            "atena": 10240,
        },
        "max_aggregate_gpu_memory_mib": 29000,
        "minimum_free_gpu_memory_mib": 4096,
        "emergency_abort_used_gpu_memory_mib": 30500,
        "launch_stagger_seconds": 5,
        "poll_seconds": 2,
        "formal_requires_clean_tracked_tree": True,
        "formal_requires_review_confirmation": True,
        "retry_assigns_new_run_tag": True,
    }
    for key, expected_value in expected_execution.items():
        if execution.get(key) != expected_value:
            raise PlanError("reviewed execution {} changed".format(key))
    policy = spec.get("test_submission_policy", {})
    if (
        policy.get("benchmark") != "reverie"
        or policy.get("native_split") != "test"
        or policy.get("episode_count") != TEST_EPISODES
        or policy.get("canonical_order_seed") != 0
        or policy.get("episode_order_sha256") != TEST_ORDER
        or policy.get("hidden_ground_truth") is not True
        or policy.get("execution_purpose") != "submission_generation_only"
        or policy.get("local_metric_computation_forbidden") is not True
        or policy.get("selection_or_ranking_on_test_forbidden") is not True
        or policy.get("eligible_tta_methods") != ["tent", "fstta", "eam"]
        or policy.get("eligible_tta_submission_jobs") != 9
        or set(policy.get("unavailable_without_legal_online_feedback", {}))
        != {"feedtta", "atena"}
        or policy.get("legal_online_feedback_interface") is not None
    ):
        raise PlanError("hidden-test fail-closed policy mismatch")
    test_orders = policy.get("order_manifests", {})
    if (
        set(test_orders) != {"duet_hamt", "goat"}
        or tuple(test_orders["duet_hamt"].get("settings", ()))
        != ("duet-reverie", "hamt-reverie")
        or tuple(test_orders["goat"].get("settings", ()))
        != ("goat-reverie",)
    ):
        raise PlanError("hidden-test order-manifest bindings mismatch")
    for method in ("feedtta", "atena"):
        item = policy["unavailable_without_legal_online_feedback"][method]
        if item.get("status") != "N/A" or item.get("launch_policy") != "fail_closed":
            raise PlanError("{} hidden-test policy is not fail-closed".format(method))
    source = policy.get("source", {})
    if (
        source.get("execution") != "reuse_existing_submission_only"
        or source.get("rerun_forbidden") is not True
        or not isinstance(source.get("ledger"), str)
        or not _is_sha256(source.get("sha256"))
    ):
        raise PlanError("hidden-test Source reuse policy mismatch")
    return spec


def _atomic_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--source-ledger", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--test-source-ledger", type=Path, default=DEFAULT_TEST_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = build_spec(args.registry, args.source_ledger, args.test_source_ledger)
        if args.check_only:
            print(json.dumps(spec, indent=2, sort_keys=True))
        else:
            _atomic_json(args.output, spec)
            print("wrote {} (15 TTA jobs, 0 Source jobs)".format(args.output))
    except PlanError as error:
        print("REVERIE frozen-plan error: {}".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
