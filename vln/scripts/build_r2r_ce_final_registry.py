#!/usr/bin/env python3
"""Freeze R2R-CE val_seen winners and materialize val_unseen evaluation.

Selection is deliberately performed only from canonical val_seen evidence.
The generated val_unseen spec contains ten TTA jobs and references the two
already-completed Source controls; it never schedules Source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if value not in sys.path:
        sys.path.insert(0, value)

from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402
import run_r2r_ce_targeted_supplement as targeted_runner  # noqa: E402
import run_tta_hparam_search as staged_runner  # noqa: E402
import tta_config_cli as config_cli  # noqa: E402


SPEC_SCHEMA = "navtta.vln_r2r_ce_postsearch_finalization.v1"
SUPPLEMENT_SCHEMA = "navtta.vln_r2r_ce_targeted_supplement_results.v1"
SELECTION_SCHEMA = "navtta.vln_r2r_ce_final_selection.v1"
REGISTRY_SCHEMA = "navtta.vln_r2r_ce_final_registry.v1"
FROZEN_SPEC_SCHEMA = "navtta.vln_r2r_ce_val_unseen_frozen_eval.v1"
VAL_SEEN_SOURCE_SCHEMA = "navtta.vln_reused_source_controls.v1"
VAL_UNSEEN_SOURCE_SCHEMA = (
    "navtta.vln_r2r_ce_val_unseen_reused_source_controls.v1"
)

DEFAULT_SPEC = (
    REPO_ROOT / "vln/experiments/r2r_ce_postsearch_finalization_v1.json"
)
SETTINGS = ("etpnav-r2r-ce", "bevbert-r2r-ce")
MODELS = ("etpnav", "bevbert")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
MODEL_FOR_SETTING = dict(zip(SETTINGS, MODELS))
SUPERVISION = {
    "source": "source_no_adaptation",
    "tent": "unsupervised_tta",
    "fstta": "unsupervised_tta",
    "eam": "unsupervised_tta",
    "feedtta": "binary_episode_feedback_tta",
    "atena": "binary_episode_feedback_tta",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
METRIC_MAP = {
    "steps_taken": "STEPS_TAKEN",
    "distance_to_goal": "DISTANCE_TO_GOAL",
    "success": "SR",
    "oracle_success": "OSR",
    "path_length": "PATH_LENGTH",
    "collisions": "COLLISIONS",
    "spl": "SPL",
    "ndtw": "NDTW",
    "sdtw": "SDTW",
    "ghost_cnt": "GHOST_CNT",
}


class RegistryError(RuntimeError):
    pass


def _read_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryError("invalid {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise RegistryError("{} must be a JSON object".format(label))
    return value


def _json_bytes(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value):
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def _resolve(repo_root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(repo_root) / path


def _repo_path(repo_root, path):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError as error:
        raise RegistryError("path escapes repository: {}".format(path)) from error


def _bound_file(repo_root, binding, label):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise RegistryError("{} binding is malformed".format(label))
    path = _resolve(repo_root, binding["path"]).resolve()
    _repo_path(repo_root, path)
    if not path.is_file():
        raise RegistryError("missing {}: {}".format(label, path))
    if not _valid_sha256(binding["sha256"]) or _sha256(path) != binding["sha256"]:
        raise RegistryError("{} SHA256 mismatch".format(label))
    return path


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError("{} must be numeric".format(label))
    if not math.isfinite(float(value)):
        raise RegistryError("{} must be finite".format(label))
    return float(value)


def _metrics(value, label):
    if not isinstance(value, dict):
        raise RegistryError("{} metrics are missing".format(label))
    output = {}
    for name, metric in value.items():
        output[name] = _number(metric, "{} {}".format(label, name))
    if not {"SR", "SPL"}.issubset(output):
        raise RegistryError("{} metrics lack SR/SPL".format(label))
    if any(not 0.0 <= output[name] <= 100.0 for name in ("SR", "SPL")):
        raise RegistryError("{} SR/SPL are outside [0, 100]".format(label))
    return output


def _canonical_manifest_path(repo_root, raw_path, run_id):
    candidate = Path(str(raw_path))
    if candidate.name != "manifest.json" or candidate.parent.name != run_id:
        raise RegistryError("formal manifest path/run_id mismatch")
    path = Path(repo_root) / "vln/results/runs" / run_id / "manifest.json"
    return path.resolve()


def _validate_manifest(repo_root, record, setting, method, split):
    label = "{} {} {}".format(setting, method, split)
    run_tag = record.get("run_tag")
    suffix = "v1.3-unified"
    run_id = "{}-{}-{}-{}".format(run_tag, setting, split, suffix)
    path = _canonical_manifest_path(
        repo_root, record.get("formal_manifest_path"), run_id
    )
    if not path.is_file():
        raise RegistryError("missing canonical {} manifest: {}".format(label, path))
    if _sha256(path) != record.get("formal_manifest_sha256"):
        raise RegistryError("{} formal manifest SHA256 mismatch".format(label))
    manifest = _read_json(path, "{} formal manifest".format(label))
    expected = {
        "run_id": run_id,
        "task": "vln",
        "benchmark": "r2r_ce_v1_3_unified_etpnav_bevbert",
        "model": MODEL_FOR_SETTING[setting],
        "method": method,
        "run_tag": run_tag,
        "source_setting": "{}:{}:{}:{}".format(
            setting, split, suffix, method
        ),
        "seed": 0,
        "status": "completed",
        "exit_code": 0,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise RegistryError("{} manifest {} mismatch".format(label, key))
    if manifest.get("checkpoint", {}).get("sha256") != record.get(
        "checkpoint_sha256"
    ):
        raise RegistryError("{} checkpoint mismatch".format(label))
    order = (
        "93f44aab1be2e3d96b867a323172ab3bbbaa4dd97e5b3fa450fe839a6c4bd94e"
        if split == "val_seen"
        else "e5a86bf609770d0525efc0be6527d8b88c02f932a0a35e1c4162b06172307a8a"
    )
    expected_content = (
        "4cb1d97271017982103458fc9156e73ae4771e9b811b15ce9a7f959ad38c43ce"
        if split == "val_seen"
        else "128dde1534a235148252c098f5e2abc6a8e8b262c9322e074b52997ec3fb224d"
    )
    dataset = manifest.get("dataset", {})
    if dataset.get("stream_order_sha256") != order:
        raise RegistryError("{} episode-order mismatch".format(label))
    if dataset.get("stream_content_sha256") != expected_content or dataset.get(
        "index_sha256"
    ) != expected_content:
        raise RegistryError("{} dataset-content digest mismatch".format(label))
    identity = manifest.get("immutable_identity_sha256")
    if not _valid_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise RegistryError("{} immutable identity mismatch".format(label))
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RegistryError("{} manifest has no result artifacts".format(label))
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not _valid_sha256(
            artifact.get("sha256")
        ):
            raise RegistryError("{} has malformed artifact metadata".format(label))
    return _repo_path(repo_root, path), manifest


def _validate_manifest_parameters(setting, method, parameters, manifest):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "parameters.json"
        config.write_text(json.dumps({
            "schema": "navtta.vln_tta_job.v1",
            "method": method,
            "parameters": parameters,
        }), encoding="utf-8")
        translated_method, tokens = config_cli.translate(
            setting, config, root / "tta_diagnostics.json"
        )
    overrides = manifest.get("config_overrides")
    if translated_method != method or not isinstance(overrides, list) or len(
        overrides
    ) < len(tokens):
        raise RegistryError("{} {} parameter provenance is missing".format(setting, method))
    actual = overrides[-len(tokens):]
    # The diagnostics path is run-specific; every other translated token must
    # match the frozen parameter object exactly.
    tokens[3] = actual[3]
    if actual != tokens:
        raise RegistryError("{} {} parameters disagree with manifest".format(setting, method))


def _local_artifact_path(repo_root, raw_path):
    normalized = str(raw_path).replace("\\", "/")
    marker = "/vln/results/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        return (Path(repo_root) / "vln/results" / suffix).resolve()
    path = Path(normalized)
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def _validate_aggregate_metrics(repo_root, manifest, expected_metrics, label):
    artifacts = [
        item for item in manifest.get("result_artifacts", [])
        if isinstance(item, dict)
        and str(item.get("name", "")).startswith("metrics/")
        and "/stats_ckpt_" in "/" + str(item.get("name", ""))
        and "/stats_ep_" not in "/" + str(item.get("name", ""))
    ]
    if len(artifacts) != 1:
        raise RegistryError("{} must authenticate one aggregate".format(label))
    artifact = artifacts[0]
    path = _local_artifact_path(repo_root, artifact.get("path", ""))
    if not path.is_file() or path.stat().st_size != artifact.get("size") or (
        _sha256(path) != artifact.get("sha256")
    ):
        raise RegistryError("{} aggregate is missing or digest-mismatched".format(label))
    raw = _read_json(path, "{} aggregate".format(label))
    actual = {}
    for raw_name, metric_name in METRIC_MAP.items():
        value = _number(raw.get(raw_name), "{} {}".format(label, raw_name))
        actual[metric_name] = 100.0 * value if metric_name in {
            "SR", "OSR", "SPL", "NDTW", "SDTW"
        } else value
    for name in ("SR", "SPL"):
        if not math.isclose(
            actual[name], float(expected_metrics[name]), rel_tol=0.0, abs_tol=1e-3
        ):
            raise RegistryError("{} {} disagrees with aggregate".format(label, name))
    return actual


def _validate_diagnostics(repo_root, manifest, setting, method, parameters,
                          metrics, expected_episodes, label):
    artifacts = [
        item for item in manifest.get("result_artifacts", [])
        if isinstance(item, dict) and item.get("name") == "tta_diagnostics.json"
    ]
    if len(artifacts) != 1:
        raise RegistryError("{} must authenticate one diagnostics artifact".format(label))
    artifact = artifacts[0]
    path = _local_artifact_path(repo_root, artifact.get("path", ""))
    if not path.is_file() or path.stat().st_size != artifact.get("size") or (
        _sha256(path) != artifact.get("sha256")
    ):
        raise RegistryError("{} diagnostics are missing or digest-mismatched".format(label))
    diagnostics = _read_json(path, "{} diagnostics".format(label))
    if (
        diagnostics.get("method") != method
        or diagnostics.get("episode_count") != expected_episodes
        or diagnostics.get("action_selection") != "target_native_argmax"
    ):
        raise RegistryError("{} diagnostics identity/count/action mismatch".format(label))
    adapter = diagnostics.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("episodes") != expected_episodes:
        raise RegistryError("{} adapter episode count mismatch".format(label))
    updates = adapter.get("updates")
    if type(updates) is not int or updates < 0:
        raise RegistryError("{} adapter.updates is invalid".format(label))
    drift = adapter.get("relative_param_drift")
    if isinstance(drift, bool) or not isinstance(drift, (int, float)) or (
        not math.isfinite(float(drift)) or float(drift) < 0.0
    ):
        raise RegistryError("{} adapter.relative_param_drift is invalid".format(label))

    def count(key):
        value = adapter.get(key)
        if type(value) is not int or not 0 <= value <= expected_episodes:
            raise RegistryError(
                "{} adapter.{} must be an integer in [0, {}]".format(
                    label, key, expected_episodes
                )
            )
        return value

    if method in ("tent", "fstta", "eam"):
        if diagnostics.get("feedback_supervision") != "none" or diagnostics.get(
            "binary_feedback_endpoint"
        ) is not None:
            raise RegistryError("{} unsupervised method reports feedback".format(label))
        forbidden = (
            "feedback_episodes", "successful_feedback_episodes",
            "failed_feedback_episodes", "queries", "self_label_episodes",
            "feedback_observed_episodes", "query_gate_evaluations",
            "self_prediction_evaluations", "queried_feedback_successes",
            "self_feedback_successes",
        )
        if any(adapter.get(key) not in (None, 0) for key in forbidden):
            raise RegistryError("{} unsupervised adapter consumed feedback".format(label))
        if method == "fstta" and adapter.get(
            "variance_history_lifetime"
        ) != "test_stream":
            raise RegistryError("{} FSTTA stream lifetime mismatch".format(label))
    elif method == "feedtta":
        if diagnostics.get("feedback_supervision") != "binary_episode_success" or (
            diagnostics.get("binary_feedback_endpoint") is not None
        ):
            raise RegistryError("{} FeedTTA supervision mismatch".format(label))
        feedback = count("feedback_episodes")
        successes = count("successful_feedback_episodes")
        failures = count("failed_feedback_episodes")
        if (
            feedback != expected_episodes
            or successes + failures != expected_episodes
            or successes != int(round(metrics["SR"] * expected_episodes / 100.0))
            or adapter.get("feedback_type") != "binary_episode_success"
            or adapter.get("action_selection_protocol") != "target_native_argmax"
            or diagnostics.get("feedtta_scope_profile") != parameters.get(
                "scope_profile"
            )
        ):
            raise RegistryError("{} FeedTTA feedback accounting mismatch".format(label))
    elif method == "atena":
        if diagnostics.get("feedback_supervision") != "binary_episode_success" or (
            diagnostics.get("binary_feedback_endpoint") is not None
        ):
            raise RegistryError("{} ATENA supervision mismatch".format(label))
        queries = count("queries")
        self_labels = count("self_label_episodes")
        observed = count("feedback_observed_episodes")
        gate = count("query_gate_evaluations")
        self_evaluations = count("self_prediction_evaluations")
        queried_successes = count("queried_feedback_successes")
        self_successes = count("self_feedback_successes")
        if (
            queries + self_labels != expected_episodes
            or observed != queries
            or gate != expected_episodes
            or self_evaluations != expected_episodes
            or queried_successes > queries
            or self_successes > self_labels
            or adapter.get("action_selection") != "argmax"
        ):
            raise RegistryError("{} ATENA feedback accounting mismatch".format(label))
        if float(parameters.get("query_threshold", math.nan)) == 0.0 and (
            queries != expected_episodes
        ):
            raise RegistryError("{} zero-threshold ATENA did not query all".format(label))
    return {
        "path": _repo_path(repo_root, path),
        "sha256": artifact["sha256"],
        "adapter": adapter,
        "feedback_supervision": diagnostics.get("feedback_supervision"),
    }


def _validate_source_manifest(repo_root, record, setting, split):
    label = "{} Source {}".format(setting, split)
    path = _resolve(repo_root, record["formal_manifest_path"]).resolve()
    canonical_root = (Path(repo_root) / "vln/results/runs").resolve()
    try:
        path.relative_to(canonical_root)
    except ValueError as error:
        raise RegistryError("{} manifest is noncanonical".format(label)) from error
    expected_run_id = "{}-{}-{}-v1.3-unified".format(
        record["run_tag"], setting, split
    )
    if path.name != "manifest.json" or path.parent.name != expected_run_id:
        raise RegistryError("{} manifest path/run_id mismatch".format(label))
    if not path.is_file() or _sha256(path) != record["formal_manifest_sha256"]:
        raise RegistryError("{} manifest missing or digest-mismatched".format(label))
    manifest = _read_json(path, "{} manifest".format(label))
    expected = {
        "run_id": expected_run_id,
        "task": "vln",
        "benchmark": "r2r_ce_v1_3_unified_etpnav_bevbert",
        "model": MODEL_FOR_SETTING[setting],
        "method": "source",
        "run_tag": record["run_tag"],
        "source_setting": "{}:{}:v1.3-unified".format(setting, split),
        "seed": 0,
        "status": "completed",
        "exit_code": 0,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise RegistryError("{} manifest {} mismatch".format(label, key))
    if manifest.get("checkpoint", {}).get("sha256") != record[
        "checkpoint_sha256"
    ]:
        raise RegistryError("{} checkpoint mismatch".format(label))
    dataset = manifest.get("dataset", {})
    if dataset.get("stream_order_sha256") != record["episode_order_sha256"]:
        raise RegistryError("{} order mismatch".format(label))
    if split == "val_unseen" and dataset.get(
        "stream_content_sha256"
    ) != record["dataset_sha256"] or (
        split == "val_unseen"
        and dataset.get("index_sha256") != record["dataset_sha256"]
    ):
        raise RegistryError("{} dataset mismatch".format(label))
    if split == "val_seen" and (
        dataset.get("stream_content_sha256")
        != "4cb1d97271017982103458fc9156e73ae4771e9b811b15ce9a7f959ad38c43ce"
        or dataset.get("index_sha256")
        != "4cb1d97271017982103458fc9156e73ae4771e9b811b15ce9a7f959ad38c43ce"
    ):
        raise RegistryError("{} val_seen content digest mismatch".format(label))
    identity = manifest.get("immutable_identity_sha256")
    expected_identity = record.get("immutable_identity_sha256", identity)
    if identity != expected_identity or immutable_identity_sha256(manifest) != identity:
        raise RegistryError("{} immutable identity mismatch".format(label))
    return manifest


def _validate_source_artifact(repo_root, binding, manifest, label,
                              required_root=None):
    if not isinstance(binding, dict) or set(binding) != {
        "name", "path", "size", "sha256"
    }:
        raise RegistryError("{} binding is malformed".format(label))
    if (
        not isinstance(binding["name"], str)
        or not binding["name"]
        or type(binding["size"]) is not int
        or binding["size"] < 0
        or not _valid_sha256(binding["sha256"])
    ):
        raise RegistryError("{} binding is malformed".format(label))
    path = _resolve(repo_root, binding["path"]).resolve()
    _repo_path(repo_root, path)
    if required_root is not None:
        try:
            path.relative_to(Path(required_root).resolve())
        except ValueError as error:
            raise RegistryError("{} is outside its canonical Source root".format(label)) from error
    if (
        not path.is_file()
        or path.stat().st_size != binding["size"]
        or _sha256(path) != binding["sha256"]
    ):
        raise RegistryError("{} is missing or digest-mismatched".format(label))
    matches = [
        artifact for artifact in manifest.get("result_artifacts", [])
        if isinstance(artifact, dict)
        and artifact.get("name") == binding["name"]
        and artifact.get("size") == binding["size"]
        and artifact.get("sha256") == binding["sha256"]
        and _local_artifact_path(repo_root, artifact.get("path", "")) == path
    ]
    if len(matches) != 1:
        raise RegistryError("{} is not exactly formal-manifest authenticated".format(label))
    return path


def _validate_flat_source_artifact(repo_root, record, kind, manifest, label,
                                   required_root, require_file):
    path_key = "{}_artifact_path".format(kind)
    digest_key = "{}_artifact_sha256".format(kind)
    path = _resolve(repo_root, record.get(path_key, "")).resolve()
    _repo_path(repo_root, path)
    try:
        path.relative_to(Path(required_root).resolve())
    except ValueError as error:
        raise RegistryError("{} is outside its canonical Source root".format(label)) from error
    digest = record.get(digest_key)
    if not _valid_sha256(digest):
        raise RegistryError("{} digest is malformed".format(label))
    matches = [
        artifact for artifact in manifest.get("result_artifacts", [])
        if isinstance(artifact, dict)
        and artifact.get("sha256") == digest
        and _local_artifact_path(repo_root, artifact.get("path", "")) == path
        and type(artifact.get("size")) is int
        and artifact["size"] >= 0
        and isinstance(artifact.get("name"), str)
    ]
    if len(matches) != 1:
        raise RegistryError("{} is not exactly formal-manifest authenticated".format(label))
    artifact = matches[0]
    name = artifact["name"]
    if (
        not name.startswith("metrics/source_val_unseen/")
        or (kind == "aggregate" and "stats_ep_" in name)
        or (kind == "per_episode" and "stats_ep_" not in name)
    ):
        raise RegistryError("{} has the wrong formal artifact role".format(label))
    if require_file and (
        not path.is_file()
        or path.stat().st_size != artifact["size"]
        or _sha256(path) != digest
    ):
        raise RegistryError("{} is missing or digest-mismatched".format(label))
    return path


def _validate_parameters(method, parameters, label):
    if not isinstance(parameters, dict) or not parameters:
        raise RegistryError("{} has no parameter object".format(label))
    if method == "tent" and parameters.get("update_interval") != 1:
        raise RegistryError("{} Tent update_interval must equal one".format(label))
    if method in ("feedtta", "atena") and parameters.get(
        "action_selection"
    ) != "argmax":
        raise RegistryError("{} must use target-native argmax".format(label))


def load_spec(path=DEFAULT_SPEC, repo_root=REPO_ROOT):
    path = Path(path).resolve()
    spec = _read_json(path, "R2R-CE post-search spec")
    if spec.get("schema") != SPEC_SCHEMA:
        raise RegistryError("unsupported R2R-CE post-search schema")
    if tuple(spec.get("settings", ())) != SETTINGS or tuple(
        spec.get("methods", ())
    ) != METHODS:
        raise RegistryError("R2R-CE setting/method matrix mismatch")
    protocol = spec.get("protocol", {})
    expected = {
        "benchmark": "r2r_ce_v1_3_unified_etpnav_bevbert",
        "selection_split": "val_seen",
        "selection_episode_count": 778,
        "selection_order_seed": 0,
        "selection_order_sha256": "93f44aab1be2e3d96b867a323172ab3bbbaa4dd97e5b3fa450fe839a6c4bd94e",
        "evaluation_split": "val_unseen",
        "evaluation_episode_count": 1839,
        "evaluation_order_seed": 0,
        "evaluation_order_sha256": "e5a86bf609770d0525efc0be6527d8b88c02f932a0a35e1c4162b06172307a8a",
        "selection_on_val_unseen": False,
        "source_execution_jobs": 0,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RegistryError("R2R-CE protocol {} mismatch".format(key))
    expected_targets = {
        ("etpnav-r2r-ce", "tent"),
        ("etpnav-r2r-ce", "fstta"),
        ("bevbert-r2r-ce", "fstta"),
        ("etpnav-r2r-ce", "feedtta"),
        ("bevbert-r2r-ce", "feedtta"),
    }
    targets = {tuple(item) for item in spec.get("targeted_cells", ())}
    if targets != expected_targets:
        raise RegistryError("targeted-cell set mismatch")
    if spec.get("selection", {}).get("candidate_scope") != (
        "initial_full_incumbent_plus_same_cell_targeted_full_confirmation_only"
    ):
        raise RegistryError("selection candidate scope mismatch")
    if spec.get("val_unseen_execution", {}).get("max_workers") != 3:
        raise RegistryError("R2R-CE val_unseen concurrency must equal three")
    initial = spec.get("initial_search", {}).get("candidate_inventory")
    initial_search = spec.get("initial_search", {})
    if not re.fullmatch(r"[0-9a-f]{40}", str(initial_search.get("git_commit", ""))):
        raise RegistryError("initial-search Git commit is invalid")
    search_spec_path = _bound_file(
        repo_root, initial_search.get("search_spec"), "initial search spec"
    )
    search_spec = _read_json(search_spec_path, "initial search spec")
    if any(search_spec.get(key) != value for key, value in {
        "schema": "navtta.vln_tta_hparam_search.v1",
        "experiment_id": initial_search.get("experiment_id"),
        "split": "val_seen",
        "primary_order_seed": 0,
        "settings": list(SETTINGS),
        "setting_episode_counts": {setting: 778 for setting in SETTINGS},
        "reused_source_controls": spec.get("val_seen_source_control"),
    }.items()):
        raise RegistryError("initial search spec protocol mismatch")
    _bound_file(repo_root, initial, "initial full-candidate inventory")
    _bound_file(
        repo_root, spec["val_seen_source_control"], "val_seen Source ledger"
    )
    unseen_binding = spec.get("val_unseen_source_control", {})
    if unseen_binding.get("execution") != "reuse_only" or unseen_binding.get(
        "rerun_forbidden"
    ) is not True:
        raise RegistryError("val_unseen Source must be reuse-only")
    _bound_file(
        repo_root,
        {"path": unseen_binding.get("path"), "sha256": unseen_binding.get("sha256")},
        "val_unseen Source ledger",
    )
    supplement = spec.get("targeted_supplement", {})
    _bound_file(repo_root, supplement.get("spec"), "targeted supplement spec")
    return path, spec


def _val_seen_source_records(spec, repo_root):
    path = _bound_file(
        repo_root, spec["val_seen_source_control"], "val_seen Source ledger"
    )
    ledger = _read_json(path, "val_seen Source ledger")
    expected = {
        "schema": VAL_SEEN_SOURCE_SCHEMA,
        "benchmark": spec["protocol"]["benchmark"],
        "split": "val_seen",
        "episode_count": 778,
        "seed": 0,
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise RegistryError("val_seen Source ledger {} mismatch".format(key))
    reuse_policy = ledger.get("reuse_policy", {})
    if (
        ledger.get("action_selection") != "target_native_argmax"
        or reuse_policy.get("source_execution_jobs") != 0
        or reuse_policy.get("missing_or_mismatched_evidence")
        != "fail_closed_without_launching_source"
    ):
        raise RegistryError("val_seen Source ledger reuse policy mismatch")
    order = ledger.get("episode_order", {})
    if not isinstance(order, dict) or set(order) != {
        "path", "sha256", "order_sha256"
    }:
        raise RegistryError("val_seen Source order binding is malformed")
    if order["order_sha256"] != spec["protocol"]["selection_order_sha256"]:
        raise RegistryError("val_seen Source order digest mismatch")
    order_path = _bound_file(
        repo_root,
        {"path": order["path"], "sha256": order["sha256"]},
        "val_seen Source episode order",
    )
    order_document = _read_json(order_path, "val_seen Source episode order")
    if any(order_document.get(key) != value for key, value in {
        "schema": "navtta.episode_order.v1",
        "benchmark": spec["protocol"]["benchmark"],
        "split": "val_seen",
        "episode_count": 778,
        "order_sha256": spec["protocol"]["selection_order_sha256"],
    }.items()) or order_document.get("dataset", {}).get("sha256") != (
        "4cb1d97271017982103458fc9156e73ae4771e9b811b15ce9a7f959ad38c43ce"
    ):
        raise RegistryError("val_seen Source episode order is inconsistent")
    ordered_ids = [
        str(item.get("episode_id")) for item in order_document.get("episodes", [])
        if isinstance(item, dict)
    ]
    if len(ordered_ids) != 778 or len(set(ordered_ids)) != 778:
        raise RegistryError("val_seen Source episode order coverage mismatch")
    records = ledger.get("settings")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("val_seen Source ledger settings mismatch")
    output = {}
    for setting in SETTINGS:
        source = records[setting]
        if source.get("model") != MODEL_FOR_SETTING[setting] or not re.fullmatch(
            r"[0-9a-f]{40}", str(source.get("git_commit", ""))
        ):
            raise RegistryError("{} Source identity is malformed".format(setting))
        if source.get("dataset_index_sha256") != (
            "4cb1d97271017982103458fc9156e73ae4771e9b811b15ce9a7f959ad38c43ce"
        ):
            raise RegistryError("{} Source dataset-index digest mismatch".format(setting))
        formal = source["formal_manifest"]
        record = {
            "run_id": source["run_id"],
            "run_tag": source["run_id"].rsplit("-{}-val_seen".format(setting), 1)[0],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "formal_manifest_path": formal["path"],
            "formal_manifest_sha256": formal["sha256"],
            "episode_order_sha256": spec["protocol"]["selection_order_sha256"],
            "per_episode_artifact_sha256": source["per_episode_artifact"]["sha256"],
        }
        manifest = _validate_source_manifest(
            repo_root, record, setting, "val_seen"
        )
        if manifest.get("git_commit") != source["git_commit"]:
            raise RegistryError("{} Source Git commit mismatch".format(setting))
        if manifest.get("pinned_manifests", {}).get("episode_order", {}).get(
            "sha256"
        ) != order["sha256"]:
            raise RegistryError("{} Source order manifest mismatch".format(setting))
        source_root = (
            Path(repo_root) / "vln/results/source" / record["run_tag"] /
            setting / "val_seen"
        )
        metrics_path = _validate_source_artifact(
            repo_root, source.get("aggregate_artifact"), manifest,
            "{} Source aggregate".format(setting),
            required_root=source_root,
        )
        per_episode_path = _validate_source_artifact(
            repo_root, source.get("per_episode_artifact"), manifest,
            "{} Source per-episode artifact".format(setting),
            required_root=source_root,
        )
        if metrics_path == per_episode_path:
            raise RegistryError("{} Source artifacts alias each other".format(setting))
        raw_metrics = _read_json(metrics_path, "{} Source aggregate".format(setting))
        per_episode = _read_json(
            per_episode_path, "{} Source per-episode artifact".format(setting)
        )
        if set(per_episode) != set(ordered_ids):
            raise RegistryError("{} Source per-episode coverage mismatch".format(setting))
        try:
            expected_raw, metrics = staged_runner._source_metric_values(
                per_episode, ordered_ids
            )
        except Exception as error:
            raise RegistryError(
                "{} Source per-episode metrics are invalid: {}".format(
                    setting, error
                )
            ) from error
        if set(raw_metrics) != set(expected_raw) or any(
            not math.isclose(
                _number(raw_metrics.get(name), "Source {}".format(name)),
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name, expected_value in expected_raw.items()
        ):
            raise RegistryError("{} Source aggregate disagrees with episodes".format(setting))
        record["metrics"] = metrics
        record["immutable_identity_sha256"] = manifest[
            "immutable_identity_sha256"
        ]
        output[setting] = record
    return path, output


def validate_val_unseen_source_ledger(spec, repo_root=REPO_ROOT,
                                      require_artifacts=True):
    binding = spec["val_unseen_source_control"]
    path = _bound_file(
        repo_root,
        {"path": binding["path"], "sha256": binding["sha256"]},
        "val_unseen Source ledger",
    )
    ledger = _read_json(path, "val_unseen Source ledger")
    expected = {
        "schema": VAL_UNSEEN_SOURCE_SCHEMA,
        "benchmark": spec["protocol"]["benchmark"],
        "split": "val_unseen",
        "episode_count": 1839,
        "canonical_order_seed": 0,
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise RegistryError("val_unseen Source ledger {} mismatch".format(key))
    if ledger.get("source_execution_policy", {}).get("execution") != "reuse_only" or (
        ledger.get("source_execution_policy", {}).get("rerun_forbidden") is not True
    ):
        raise RegistryError("val_unseen Source ledger permits execution")
    order = ledger.get("episode_order", {})
    if order.get("order_sha256") != spec["protocol"]["evaluation_order_sha256"]:
        raise RegistryError("val_unseen Source order mismatch")
    order_path = _resolve(repo_root, order.get("path", ""))
    if not order_path.is_file() or _sha256(order_path) != order.get("sha256"):
        raise RegistryError("val_unseen order manifest is missing or changed")
    order_document = _read_json(order_path, "val_unseen order manifest")
    if any(order_document.get(key) != value for key, value in {
        "schema": "navtta.episode_order.v1",
        "benchmark": spec["protocol"]["benchmark"],
        "split": "val_unseen",
        "episode_count": 1839,
        "order_sha256": spec["protocol"]["evaluation_order_sha256"],
    }.items()) or not _valid_sha256(
        order_document.get("dataset", {}).get("sha256")
    ):
        raise RegistryError("val_unseen order manifest is inconsistent")
    ordered_ids = [
        str(item.get("episode_id")) for item in order_document.get("episodes", [])
        if isinstance(item, dict)
    ]
    if len(ordered_ids) != 1839 or len(set(ordered_ids)) != 1839:
        raise RegistryError("val_unseen order manifest coverage mismatch")
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("val_unseen Source records mismatch")
    for setting in SETTINGS:
        record = records[setting]
        if record.get("evidence_status") != "ready":
            raise RegistryError("{} val_unseen Source is not ready".format(setting))
        if record.get("model") != MODEL_FOR_SETTING[setting]:
            raise RegistryError("{} val_unseen Source model mismatch".format(setting))
        if record.get("parameters") != {
            "action_selection": "argmax", "action_seed": 0
        }:
            raise RegistryError("{} val_unseen Source protocol mismatch".format(setting))
        if record.get("episode_order_sha256") != order["order_sha256"] or (
            record.get("dataset_sha256") != order_document["dataset"]["sha256"]
        ):
            raise RegistryError("{} val_unseen Source dataset/order mismatch".format(setting))
        _metrics(record.get("metrics"), "{} val_unseen Source".format(setting))
        manifest = _validate_source_manifest(
            repo_root, record, setting, "val_unseen"
        )
        if manifest.get("pinned_manifests", {}).get("episode_order", {}).get(
            "sha256"
        ) != order["sha256"]:
            raise RegistryError("{} val_unseen Source order manifest mismatch".format(setting))
        source_root = (
            Path(repo_root) / "vln/results/source" / record["run_tag"] /
            setting / "val_unseen"
        )
        aggregate_path = _validate_flat_source_artifact(
            repo_root, record, "aggregate", manifest,
            "{} val_unseen Source aggregate".format(setting), source_root,
            require_artifacts,
        )
        per_episode_path = _validate_flat_source_artifact(
            repo_root, record, "per_episode", manifest,
            "{} val_unseen Source per-episode artifact".format(setting),
            source_root, require_artifacts,
        )
        if aggregate_path == per_episode_path:
            raise RegistryError("{} val_unseen Source artifacts alias".format(setting))
        raw = _read_json(
            aggregate_path,
            "{} val_unseen Source aggregate".format(setting),
        ) if require_artifacts else None
        if raw is not None:
            per_episode = _read_json(
                per_episode_path,
                "{} val_unseen Source per-episode artifact".format(setting),
            )
            if set(per_episode) != set(ordered_ids):
                raise RegistryError(
                    "{} val_unseen Source per-episode coverage mismatch".format(setting)
                )
            try:
                expected_raw, expected_metrics = staged_runner._source_metric_values(
                    per_episode, ordered_ids
                )
            except Exception as error:
                raise RegistryError(
                    "{} val_unseen Source episodes are invalid: {}".format(
                        setting, error
                    )
                ) from error
            if set(raw) != set(expected_raw) or any(
                not math.isclose(
                    _number(raw.get(name), "aggregate {}".format(name)),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for name, value in expected_raw.items()
            ):
                raise RegistryError(
                    "{} val_unseen Source aggregate disagrees with episodes".format(setting)
                )
            for name in ("SR", "SPL"):
                value = expected_metrics[name]
                if not math.isclose(
                    value, float(record["metrics"][name]), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise RegistryError(
                        "{} Source {} does not match aggregate".format(setting, name)
                    )
    return path, ledger


def _initial_candidates(spec, repo_root, source_records):
    path = _bound_file(
        repo_root,
        spec["initial_search"]["candidate_inventory"],
        "initial full-candidate inventory",
    )
    document = _read_json(path, "initial full-candidate inventory")
    expected = {
        "schema": "navtta.vln_r2r_ce_initial_full_candidates.v1",
        "experiment_id": spec["initial_search"]["experiment_id"],
        "batch_id": spec["initial_search"]["batch_id"],
        "benchmark": spec["protocol"]["benchmark"],
        "split": "val_seen",
        "episode_count": 778,
        "order_seed": 0,
        "episode_order_sha256": spec["protocol"]["selection_order_sha256"],
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RegistryError("initial candidate inventory {} mismatch".format(key))
    records = document.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("initial candidate setting matrix mismatch")
    source_selections = document.get("source_selection_files")
    if not isinstance(source_selections, dict) or set(
        source_selections
    ) != set(METHODS):
        raise RegistryError("initial source-selection bindings are incomplete")
    authenticated_selections = {}
    initial_search = spec["initial_search"]
    for method in METHODS:
        selection_path = _bound_file(
            repo_root, source_selections[method],
            "{} source FINAL_SELECTION".format(method),
        )
        selection = _read_json(
            selection_path, "{} source FINAL_SELECTION".format(method)
        )
        if (
            selection.get("schema") != "navtta.vln_tta_final_selection.v1"
            or selection.get("method") != method
            or selection.get("split") != "val_seen"
            or selection.get("spec_sha256")
            != initial_search["search_spec"]["sha256"]
            or selection.get("git_commit") != initial_search["git_commit"]
            or selection.get("finalist_stage") != "final"
            or selection.get("matched_source_stage")
            != "reused_full_778_episode_source"
            or set(selection.get("settings", {})) != set(SETTINGS)
        ):
            raise RegistryError(
                "{} source FINAL_SELECTION identity mismatch".format(method)
            )
        authenticated_selections[method] = selection
    candidates = {setting: {} for setting in SETTINGS}
    for setting in SETTINGS:
        if not isinstance(records[setting], dict) or set(
            records[setting]
        ) != set(METHODS):
            raise RegistryError("{} initial method matrix mismatch".format(setting))
        for method in METHODS:
            item = records[setting][method]
            source_item = authenticated_selections[method]["settings"][setting]
            source_provenance = source_item.get("winner_formal_provenance", {})
            source_selection = source_item.get("selection", {})
            source_control = source_item.get("source_provenance", {})
            expected_source = source_records[setting]
            if (
                source_item.get("source_run_tag")
                != "{}-778ep".format(expected_source["run_id"])
                or source_control.get("reused_formal_manifest_sha256")
                != expected_source["formal_manifest_sha256"]
                or _local_artifact_path(
                    repo_root,
                    source_control.get("reused_formal_manifest", ""),
                ) != _resolve(
                    repo_root, expected_source["formal_manifest_path"]
                ).resolve()
                or source_control.get("per_episode_artifact_sha256")
                != expected_source["per_episode_artifact_sha256"]
                or source_item.get("source_metrics") != expected_source["metrics"]
                or source_selection.get("method") != method
                or source_selection.get("setting") != setting
                or not math.isclose(
                    _number(
                        source_selection.get("matched_source_sr"),
                        "{} {} matched Source SR".format(setting, method),
                    ),
                    float(expected_source["metrics"]["SR"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or source_selection.get("selected_run_tags")
                != [source_item.get("winner_run_tag")]
            ):
                raise RegistryError(
                    "{} {} source FINAL_SELECTION provenance mismatch".format(
                        setting, method
                    )
                )
            expected_inventory = {
                "parameters": source_item.get("frozen_parameters"),
                "run_tag": source_item.get("winner_run_tag"),
                "metrics": source_item.get("winner_metrics"),
                "formal_manifest_path": (
                    "vln/results/runs/{}/manifest.json".format(
                        Path(str(source_provenance.get("formal_manifest_path", "")))
                        .parent.name
                    )
                ),
                "formal_manifest_sha256": source_provenance.get(
                    "formal_manifest_sha256"
                ),
                "formal_immutable_identity_sha256": source_provenance.get(
                    "formal_immutable_identity_sha256"
                ),
            }
            if json.dumps(
                item, sort_keys=True, separators=(",", ":")
            ) != json.dumps(
                expected_inventory, sort_keys=True, separators=(",", ":")
            ):
                raise RegistryError(
                    "{} {} inventory differs from authenticated FINAL_SELECTION"
                    .format(setting, method)
                )
            run_tag = item.get("run_tag")
            record = {
                "origin": "initial_search_full",
                "run_tag": run_tag,
                "parameters": item.get("parameters"),
                "metrics": _metrics(
                    item.get("metrics"), "{} {} initial".format(setting, method)
                ),
                "checkpoint_sha256": source_records[setting]["checkpoint_sha256"],
                "formal_manifest_path": item.get("formal_manifest_path"),
                "formal_manifest_sha256": item.get("formal_manifest_sha256"),
                "formal_immutable_identity_sha256": item.get(
                    "formal_immutable_identity_sha256"
                ),
                "adapter_diagnostics": {},
            }
            _validate_parameters(method, record["parameters"], "{} {}".format(setting, method))
            manifest_path, manifest = _validate_manifest(
                repo_root, record, setting, method, "val_seen"
            )
            _validate_manifest_parameters(
                setting, method, record["parameters"], manifest
            )
            record["metrics"] = _validate_aggregate_metrics(
                repo_root, manifest, record["metrics"],
                "{} {} initial".format(setting, method),
            )
            diagnostics = _validate_diagnostics(
                repo_root, manifest, setting, method, record["parameters"],
                record["metrics"], 778,
                "{} {} initial".format(setting, method),
            )
            record["diagnostics_artifact_path"] = diagnostics["path"]
            record["diagnostics_artifact_sha256"] = diagnostics["sha256"]
            record["adapter_diagnostics"] = diagnostics["adapter"]
            if manifest["immutable_identity_sha256"] != record[
                "formal_immutable_identity_sha256"
            ]:
                raise RegistryError("{} {} initial identity mismatch".format(setting, method))
            if manifest.get("git_commit") != initial_search["git_commit"]:
                raise RegistryError("{} {} initial Git commit mismatch".format(setting, method))
            record["formal_manifest_path"] = manifest_path
            candidates[setting][method] = record
    return candidates


def _portable_repo_path(repo_root, raw_path, label):
    normalized = str(raw_path).replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute():
        candidate = (Path(repo_root) / path).resolve()
    else:
        candidate = path.resolve()
        try:
            candidate.relative_to(Path(repo_root).resolve())
        except ValueError:
            candidate = None
            for marker in ("/vln/", "/core/", "/tools/"):
                if marker in normalized:
                    prefix, suffix = normalized.split(marker, 1)
                    del prefix
                    candidate = (
                        Path(repo_root) / marker.strip("/") / suffix
                    ).resolve()
                    break
            if candidate is None:
                raise RegistryError("{} is not repository-relative".format(label))
    _repo_path(repo_root, candidate)
    return candidate


def _rebase_targeted_job(job, repo_root):
    output = json.loads(json.dumps(job))
    for key in (
        "config_path", "job_dir", "result_root", "retry_result_root_parent",
    ):
        output[key] = str(_portable_repo_path(
            repo_root, output.get(key, ""), "targeted job {}".format(key)
        ))
    command = output.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise RegistryError("targeted job command is malformed")
    rebased_command = []
    for index, token in enumerate(command):
        if index == 0 or token.startswith("vln/") or any(
            marker in token.replace("\\", "/")
            for marker in ("/vln/", "/core/", "/tools/")
        ):
            token = str(_portable_repo_path(
                repo_root, token, "targeted job command path"
            ))
        rebased_command.append(token)
    output["command"] = rebased_command
    return output


def _targeted_job_evidence(job, repo_root):
    job_dir = Path(job["job_dir"])
    paths = {
        "job": job_dir / "job.json",
        "parameters": job_dir / "parameters.json",
        "metrics": job_dir / "metrics.json",
        "exitcode": job_dir / "exitcode",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise RegistryError(
                "targeted job lacks {}: {}".format(label, path)
            )
    return {
        "run_tag": job["run_tag"],
        **{
            "{}_path".format(label): _repo_path(repo_root, path)
            for label, path in paths.items()
        },
        **{
            "{}_sha256".format(label): _sha256(path)
            for label, path in paths.items()
        },
    }


def _load_targeted_phase_jobs(root, expected_jobs, repo_root):
    root = Path(root).resolve()
    paths = sorted((root / "jobs").glob("*/job.json"))
    if len(paths) != len(expected_jobs):
        raise RegistryError("targeted phase persisted-job count mismatch")
    actual = []
    for path in paths:
        raw = _read_json(path, "targeted job")
        rebased = _rebase_targeted_job(raw, repo_root)
        if Path(rebased["job_dir"]).resolve() != path.parent.resolve():
            raise RegistryError("targeted job directory is noncanonical")
        actual.append((raw, rebased))
    by_base = {item[1].get("base_run_tag"): item for item in actual}
    if len(by_base) != len(actual) or set(by_base) != {
        job["base_run_tag"] for job in expected_jobs
    }:
        raise RegistryError("targeted phase base-run tags mismatch")
    output = []
    identity_keys = tuple(targeted_runner._job_identity(expected_jobs[0])) if expected_jobs else ()
    all_keys = set(identity_keys) | {"run_tag", "attempt", "result_root", "command"}
    for expected in expected_jobs:
        raw, job = by_base[expected["base_run_tag"]]
        if set(raw) != all_keys:
            raise RegistryError("targeted job fields differ from producer contract")
        expected = _rebase_targeted_job(expected, repo_root)
        for key in identity_keys:
            if _json_bytes(job.get(key)) != _json_bytes(expected.get(key)):
                raise RegistryError(
                    "targeted job identity mismatch for {}".format(key)
                )
        attempt = job.get("attempt")
        if type(attempt) is not int or attempt < 0:
            raise RegistryError("targeted retry attempt is invalid")
        expected_tag = job["base_run_tag"] + (
            "-retry{}".format(attempt) if attempt else ""
        )
        if job.get("run_tag") != expected_tag:
            raise RegistryError("targeted retry run tag is invalid")
        expected_result = (
            Path(job["retry_result_root_parent"]) / expected_tag / "val_seen"
        ).resolve()
        if Path(job["result_root"]).resolve() != expected_result:
            raise RegistryError("targeted retry result root is noncanonical")
        expected_command = list(expected["command"])
        expected_command[expected_command.index("--run-tag") + 1] = expected_tag
        expected_command[expected_command.index("--result-root") + 1] = str(
            expected_result
        )
        if job["command"] != expected_command:
            raise RegistryError("targeted job command changed")
        config_path = Path(job["config_path"])
        if not config_path.is_file() or _read_json(
            config_path, "targeted parameters"
        ) != staged_runner._job_config(job):
            raise RegistryError("targeted runtime parameters changed")
        try:
            exit_code = int((Path(job["job_dir"]) / "exitcode").read_text().strip())
        except (OSError, ValueError) as error:
            raise RegistryError("targeted job exit status is invalid") from error
        if exit_code != 0:
            raise RegistryError("targeted job did not complete successfully")
        attempts_root = Path(job["job_dir"]) / "attempts"
        archived = sorted(
            path for path in attempts_root.glob("attempt-*") if path.is_dir()
        )
        if [item.name for item in archived] != [
            "attempt-{:02d}".format(index) for index in range(attempt)
        ]:
            raise RegistryError("targeted retry archive is not contiguous")
        for index, archive in enumerate(archived):
            archived_job_path = archive / "job.json"
            archived_job = _read_json(
                archived_job_path, "targeted retry job"
            )
            archived_rebased = _rebase_targeted_job(archived_job, repo_root)
            expected_archived = dict(expected)
            expected_archived["attempt"] = index
            expected_archived["run_tag"] = expected["base_run_tag"] + (
                "-retry{}".format(index) if index else ""
            )
            expected_archived["result_root"] = str(
                Path(expected["retry_result_root_parent"])
                / expected_archived["run_tag"] / "val_seen"
            )
            expected_archived["command"] = list(expected["command"])
            expected_archived["command"][
                expected_archived["command"].index("--run-tag") + 1
            ] = expected_archived["run_tag"]
            expected_archived["command"][
                expected_archived["command"].index("--result-root") + 1
            ] = expected_archived["result_root"]
            archived_config_path = archive / "parameters.json"
            archive_manifest_path = archive / "archived_evidence.json"
            if (
                archived_job.get("base_run_tag") != raw.get("base_run_tag")
                or archived_job.get("attempt") != index
                or archived_job.get("run_tag") != raw.get("base_run_tag") + (
                    "-retry{}".format(index) if index else ""
                )
                or json.dumps(
                    archived_rebased, sort_keys=True, separators=(",", ":")
                ) != json.dumps(
                    expected_archived, sort_keys=True, separators=(",", ":")
                )
                or not archived_config_path.is_file()
                or _read_json(
                    archived_config_path, "targeted archived parameters"
                ) != staged_runner._job_config(archived_rebased)
                or not archive_manifest_path.is_file()
            ):
                raise RegistryError("targeted retry archive identity mismatch")
            archived_evidence = _read_json(
                archive_manifest_path, "targeted archived evidence"
            )
            if set(archived_evidence) - {"result_root", "formal_run_manifest"}:
                raise RegistryError("targeted retry archive evidence is malformed")
            for value in archived_evidence.values():
                archived_path = _portable_repo_path(
                    repo_root, value, "targeted archived evidence"
                )
                try:
                    archived_path.relative_to(archive.resolve())
                except ValueError as error:
                    raise RegistryError(
                        "targeted retry evidence escaped its archive"
                    ) from error
                if not archived_path.exists():
                    raise RegistryError("targeted retry evidence is missing")
        output.append(job)
    return output


def _screening_result(job, supplement_spec):
    try:
        result = staged_runner.parse_metrics(job, supplement_spec)
        result = targeted_runner.enrich_screening_result(
            job, result, supplement_spec
        )
    except Exception as error:
        raise RegistryError(
            "targeted screening evidence is invalid for {}: {}".format(
                job.get("run_tag"), error
            )
        ) from error
    persisted = _read_json(
        Path(job["job_dir"]) / "metrics.json", "targeted screening metrics"
    )
    for key in (
        "run_tag", "setting", "config_method", "point_index", "parameters",
        "metrics", "adapter_diagnostics", "expected_episodes",
        "diagnostics_sha256",
    ):
        if json.dumps(persisted.get(key), sort_keys=True, separators=(",", ":")) != json.dumps(
            result.get(key), sort_keys=True, separators=(",", ":")
        ):
            raise RegistryError(
                "targeted screening metrics mismatch for {} {}".format(
                    job["run_tag"], key
                )
            )
    # The scheduler may refresh metrics.json from the base parser after the
    # promotion pass.  In that case the three enrichment fields are absent,
    # while PROMOTION.json still contains their producer output.  Recompute
    # them from the episode artifacts below; if metrics.json retained them,
    # require the complete set and verify it byte-for-byte.
    enrichment_keys = (
        "navigation_record_change_count",
        "navigation_record_artifact_sha256",
        "matched_source_record_artifact_sha256",
    )
    present_enrichment = [key for key in enrichment_keys if key in persisted]
    if present_enrichment and len(present_enrichment) != len(enrichment_keys):
        raise RegistryError(
            "targeted screening metrics have partial navigation enrichment for {}"
            .format(job["run_tag"])
        )
    for key in present_enrichment:
        if json.dumps(
            persisted[key], sort_keys=True, separators=(",", ":")
        ) != json.dumps(result.get(key), sort_keys=True, separators=(",", ":")):
            raise RegistryError(
                "targeted screening metrics mismatch for {} {}".format(
                    job["run_tag"], key
                )
            )
    candidate_path = Path(result["navigation_record_artifact"])
    candidate = _read_json(candidate_path, "targeted screening episode records")
    identifiers = targeted_runner._canonical_episode_ids(supplement_spec, 100)
    try:
        _, episode_metrics = staged_runner._source_metric_values(
            candidate, identifiers
        )
    except Exception as error:
        raise RegistryError("targeted screening episode metrics are invalid") from error
    for name in ("SR", "SPL"):
        if not math.isclose(
            float(result["metrics"][name]), float(episode_metrics[name]),
            rel_tol=0.0, abs_tol=1e-3,
        ):
            raise RegistryError(
                "targeted screening {} disagrees with episode records".format(name)
            )
    return result


def _authenticate_supplement_jobs(campaign_root, phases, supplement_spec,
                                  promotions, confirmations, repo_root):
    evidence = {}
    for setting_index, setting in enumerate(SETTINGS):
        screening_phase = phases[setting_index * 2]
        full_phase = phases[setting_index * 2 + 1]
        screening_root = Path(campaign_root) / "phases" / screening_phase["phase_id"]
        full_root = Path(campaign_root) / "phases" / full_phase["phase_id"]
        persisted_paths = sorted((screening_root / "jobs").glob("*/job.json"))
        if not persisted_paths:
            raise RegistryError("{} screening has no persisted jobs".format(setting))
        first = _read_json(persisted_paths[0], "targeted screening job")
        command = first.get("command")
        try:
            gpu = int(command[3])
        except (IndexError, TypeError, ValueError) as error:
            raise RegistryError("targeted screening GPU binding is invalid") from error
        expected_screening = targeted_runner.build_screening_jobs(
            screening_phase, supplement_spec["_batch_id"], supplement_spec, gpu
        )
        screening_jobs = _load_targeted_phase_jobs(
            screening_root, expected_screening, repo_root
        )
        results = [_screening_result(job, supplement_spec) for job in screening_jobs]
        source = staged_runner.reused_source_result(
            setting, 100, supplement_spec
        )
        promotion = promotions[setting]
        expected_source = {
            "run_tag": source["run_tag"],
            "metrics": source["metrics"],
            "formal_manifest": source["reused_formal_manifest"],
            "formal_manifest_sha256": source["reused_formal_manifest_sha256"],
        }
        actual_source = dict(promotion.get("source", {}))
        if "formal_manifest" in actual_source:
            actual_source["formal_manifest"] = str(_portable_repo_path(
                repo_root, actual_source["formal_manifest"],
                "targeted screening Source manifest",
            ))
            expected_source["formal_manifest"] = str(_portable_repo_path(
                repo_root, expected_source["formal_manifest"],
                "targeted screening Source manifest",
            ))
        if json.dumps(actual_source, sort_keys=True, separators=(",", ":")) != json.dumps(
            expected_source, sort_keys=True, separators=(",", ":")
        ):
            raise RegistryError("{} PROMOTION Source evidence mismatch".format(setting))
        for method in targeted_runner.enabled_methods(setting, supplement_spec):
            method_results = [
                item for item in results if item["config_method"] == method
            ]
            selected, selection = targeted_runner.select_finalist(
                method, setting, method_results, source, supplement_spec
            )
            expected_cell = {
                "selected": ({
                    "run_tag": selected["run_tag"],
                    "point_index": selected["point_index"],
                    "parameters": selected["parameters"],
                    "metrics": selected["metrics"],
                    "adapter_diagnostics": selected["adapter_diagnostics"],
                    "navigation_record_change_count": selected[
                        "navigation_record_change_count"
                    ],
                } if selected is not None else None),
                "selection": selection,
            }
            if json.dumps(
                promotion["cells"][method], sort_keys=True, separators=(",", ":")
            ) != json.dumps(expected_cell, sort_keys=True, separators=(",", ":")):
                raise RegistryError(
                    "{} {} PROMOTION does not replay from screening jobs".format(
                        setting, method
                    )
                )
            evidence[(setting, method)] = {
                "screening_jobs": [
                    _targeted_job_evidence(job, repo_root)
                    for job in screening_jobs
                    if job["config_method"] == method
                ]
            }

        expected_full = targeted_runner.build_full_jobs(
            full_phase, supplement_spec["_batch_id"], promotion,
            supplement_spec, gpu,
        )
        if expected_full:
            full_jobs = _load_targeted_phase_jobs(
                full_root, expected_full, repo_root
            )
        else:
            if list((full_root / "jobs").glob("**/job.json")):
                raise RegistryError("empty targeted full phase contains jobs")
            full_jobs = []
        confirmation_cells = confirmations[setting].get("cells", {})
        if set(confirmation_cells) != {
            job["config_method"] for job in full_jobs
        }:
            raise RegistryError("{} full jobs/CONFIRMATION cells mismatch".format(setting))
        for job in full_jobs:
            method = job["config_method"]
            cell = confirmation_cells[method]
            persisted = _read_json(
                Path(job["job_dir"]) / "metrics.json", "targeted full metrics"
            )
            for key in (
                "run_tag", "parent_run_tags", "parameters",
                "adapter_diagnostics", "formal_manifest_sha256",
                "formal_immutable_identity_sha256",
            ):
                expected_value = (
                    [cell["parent_screening_run_tag"]]
                    if key == "parent_run_tags" else cell.get(key)
                )
                if json.dumps(
                    persisted.get(key), sort_keys=True, separators=(",", ":")
                ) != json.dumps(
                    expected_value, sort_keys=True, separators=(",", ":")
                ):
                    raise RegistryError(
                        "{} {} full metrics/CONFIRMATION {} mismatch".format(
                            setting, method, key
                        )
                    )
            for name in ("SR", "SPL"):
                if not math.isclose(
                    float(persisted.get("metrics", {}).get(name, math.nan)),
                    float(cell.get("metrics", {}).get(name, math.nan)),
                    rel_tol=0.0,
                    abs_tol=1e-3,
                ):
                    raise RegistryError(
                        "{} {} full metrics/CONFIRMATION {} mismatch".format(
                            setting, method, name
                        )
                    )
            path_pairs = (
                ("config_path", "job_config_path"),
                ("formal_manifest_path", "formal_manifest_path"),
            )
            for result_key, confirmation_key in path_pairs:
                if _portable_repo_path(
                    repo_root, persisted.get(result_key, ""),
                    "targeted full {}".format(result_key),
                ) != _portable_repo_path(
                    repo_root, cell.get(confirmation_key, ""),
                    "targeted confirmation {}".format(confirmation_key),
                ):
                    raise RegistryError(
                        "{} {} full path binding mismatch".format(setting, method)
                    )
            if _sha256(Path(job["config_path"])) != cell.get("job_config_sha256"):
                raise RegistryError(
                    "{} {} full digest binding mismatch".format(setting, method)
                )
            evidence[(setting, method)]["full_job"] = _targeted_job_evidence(
                job, repo_root
            )
    return evidence


def _validate_supplement_campaign_artifacts(path, document, spec,
                                            supplement_spec, repo_root,
                                            source_records):
    expected_results = _resolve(
        repo_root, spec["targeted_supplement"]["results_relative_path"]
    ).resolve()
    if Path(path).resolve() != expected_results:
        raise RegistryError(
            "targeted supplement RESULTS must use the canonical campaign path"
        )
    campaign_root = expected_results.parent
    plan_path = campaign_root / "PLAN.json"
    plan = _read_json(plan_path, "targeted supplement PLAN")
    expected_shared_gpu_coordination = {
        "peer_campaign": supplement_spec["execution"]["parallel_peer_campaign"],
        "launch_guard_required": True,
        "active_reservation_required": True,
        "reservation_role": targeted_runner.RESERVATION_ROLE,
    }
    expected_plan = {
        "schema": "navtta.vln_r2r_ce_targeted_supplement_plan.v1",
        "experiment_id": document["experiment_id"],
        "batch_id": document["batch_id"],
        "benchmark": document["benchmark"],
        "split": "val_seen",
        "git_commit": document["git_commit"],
        "spec_sha256": document["spec_sha256"],
        "source_execution_jobs": 0,
        "screening_jobs": 15,
        "full_jobs_max": 5,
        "total_jobs_max": 20,
        "strict_model_barrier": True,
        "shared_gpu_coordination": expected_shared_gpu_coordination,
        "canonical_order_seed": 0,
        "canonical_order_sha256": supplement_spec["canonical_order"][
            "manifest"
        ]["order_sha256"],
    }
    expected_plan_keys = set(expected_plan) | {"spec_path", "phases"}
    if set(plan) != expected_plan_keys:
        raise RegistryError("targeted supplement PLAN fields mismatch")
    for key, value in expected_plan.items():
        if plan.get(key) != value:
            raise RegistryError("targeted supplement PLAN {} mismatch".format(key))
    supplement_spec_path = _resolve(
        repo_root, spec["targeted_supplement"]["spec"]["path"]
    ).resolve()
    if _portable_repo_path(
        repo_root, plan.get("spec_path", ""), "targeted supplement PLAN spec"
    ) != supplement_spec_path:
        raise RegistryError("targeted supplement PLAN spec path mismatch")
    phases = plan.get("phases")
    if not isinstance(phases, list) or len(phases) != 4:
        raise RegistryError("targeted supplement PLAN must contain four phases")
    expected_phases = []
    for setting in SETTINGS:
        phase_spec = supplement_spec["execution"]["phases"][setting]
        model = MODEL_FOR_SETTING[setting]
        expected_phases.extend((
            {
                "kind": "screening",
                "setting": setting,
                "model": model,
                "stage": "r2r_ce_targeted_screening",
                "planned_jobs": phase_spec["screening"]["planned_jobs"],
                "max_jobs": None,
                "max_workers": phase_spec["screening"]["max_workers"],
            },
            {
                "kind": "full_confirmation",
                "setting": setting,
                "model": model,
                "stage": "r2r_ce_targeted_full",
                "planned_jobs": None,
                "max_jobs": phase_spec["full_confirmation"]["max_jobs"],
                "max_workers": phase_spec["full_confirmation"]["max_workers"],
            },
        ))
    dispositions = {}
    promotions = {}
    confirmations = {}
    for index, (phase, expected_phase) in enumerate(zip(phases, expected_phases)):
        expected_phase = dict(expected_phase)
        expected_phase.update({
            "index": index,
            "phase_id": "{:02d}-{}-{}".format(
                index, expected_phase["setting"], expected_phase["kind"]
            ),
        })
        if json.dumps(phase, sort_keys=True, separators=(",", ":")) != json.dumps(
            expected_phase, sort_keys=True, separators=(",", ":")
        ):
            raise RegistryError("targeted supplement phase {} mismatch".format(index))
    for setting_index, setting in enumerate(SETTINGS):
        screening_phase = phases[setting_index * 2]
        full_phase = phases[setting_index * 2 + 1]
        screening_root = campaign_root / "phases" / screening_phase["phase_id"]
        full_root = campaign_root / "phases" / full_phase["phase_id"]
        phase_documents = []
        for phase, phase_root in (
            (screening_phase, screening_root), (full_phase, full_root)
        ):
            phase_path = phase_root / "PHASE.json"
            phase_document = _read_json(phase_path, "{} PHASE".format(phase["phase_id"]))
            if (
                phase_document.get("schema")
                != "navtta.vln_r2r_ce_targeted_supplement_phase.v1"
                or phase_document.get("experiment_id") != document["experiment_id"]
                or phase_document.get("batch_id") != document["batch_id"]
                or phase_document.get("git_commit") != document["git_commit"]
                or phase_document.get("spec_sha256") != document["spec_sha256"]
                or phase_document.get("phase") != phase
                or phase_document.get("source_execution_jobs") != 0
                or phase_document.get("restart_from_source_checkpoint") is not True
            ):
                raise RegistryError("{} PHASE binding mismatch".format(phase["phase_id"]))
            expected_phase_keys = {
                "schema", "experiment_id", "batch_id", "phase", "job_count",
                "job_count_cap", "git_commit", "spec_path", "spec_sha256",
                "canonical_order_seed", "canonical_order_sha256",
                "source_execution_jobs", "restart_from_source_checkpoint",
                "shared_gpu_coordination", "resource_limits",
            }
            if set(phase_document) != expected_phase_keys:
                raise RegistryError("{} PHASE fields mismatch".format(phase["phase_id"]))
            expected_cap = (
                phase["planned_jobs"]
                if phase["kind"] == "screening" else phase["max_jobs"]
            )
            if (
                phase_document.get("job_count_cap") != expected_cap
                or phase_document.get("canonical_order_seed") != 0
                or phase_document.get("canonical_order_sha256")
                != supplement_spec["canonical_order"]["manifest"]["order_sha256"]
                or phase_document.get("resource_limits")
                != targeted_runner.runtime_limits(phase, supplement_spec)
                or phase_document.get("shared_gpu_coordination")
                != expected_shared_gpu_coordination
                or _portable_repo_path(
                    repo_root, phase_document.get("spec_path", ""),
                    "{} PHASE spec".format(phase["phase_id"]),
                ) != supplement_spec_path
            ):
                raise RegistryError("{} PHASE runtime contract mismatch".format(phase["phase_id"]))
            phase_documents.append((phase_path, phase_document))

        promotion_path = screening_root / "PROMOTION.json"
        promotion = _read_json(promotion_path, "{} PROMOTION".format(setting))
        promotions[setting] = promotion
        if any(promotion.get(key) != value for key, value in {
            "schema": "navtta.vln_r2r_ce_targeted_supplement_promotion.v1",
            "experiment_id": document["experiment_id"],
            "batch_id": document["batch_id"],
            "setting": setting,
            "git_commit": document["git_commit"],
            "spec_sha256": document["spec_sha256"],
            "canonical_prefix_episodes": 100,
            "canonical_order_seed": 0,
        }.items()):
            raise RegistryError("{} PROMOTION binding mismatch".format(setting))
        expected_source = source_records[setting]
        if promotion.get("source", {}).get(
            "formal_manifest_sha256"
        ) != expected_source["formal_manifest_sha256"]:
            raise RegistryError("{} PROMOTION Source mismatch".format(setting))
        allowed = tuple(
            method for method in METHODS
            if setting in supplement_spec["methods"].get(method, {}).get("settings", {})
        )
        cells = promotion.get("cells")
        if not isinstance(cells, dict) or set(cells) != set(allowed):
            raise RegistryError("{} PROMOTION cell set mismatch".format(setting))

        confirmation_path = full_root / "CONFIRMATION.json"
        confirmation = _read_json(
            confirmation_path, "{} CONFIRMATION".format(setting)
        )
        confirmations[setting] = confirmation
        embedded = document.get("settings", {}).get(setting)
        if json.dumps(
            confirmation, sort_keys=True, separators=(",", ":")
        ) != json.dumps(embedded, sort_keys=True, separators=(",", ":")):
            raise RegistryError("{} RESULTS/CONFIRMATION mismatch".format(setting))
        confirmed_cells = confirmation.get("cells")
        if not isinstance(confirmed_cells, dict):
            raise RegistryError("{} CONFIRMATION cells malformed".format(setting))
        promoted = set()
        for method in allowed:
            promotion_cell = cells[method]
            selection = promotion_cell.get("selection")
            selected = promotion_cell.get("selected")
            if not isinstance(selection, dict) or selection.get("method") != method or (
                selection.get("setting") != setting
            ):
                raise RegistryError("{} {} promotion disposition malformed".format(setting, method))
            ranked = selection.get("ranked")
            if not isinstance(ranked, list) or len(ranked) != 3 or len({
                item.get("run_tag") for item in ranked if isinstance(item, dict)
            }) != 3:
                raise RegistryError("{} {} promotion ranking is incomplete".format(setting, method))
            if selected is None:
                if selection.get("selected_run_tag") is not None or method in confirmed_cells:
                    raise RegistryError("{} {} rejected promotion has a full result".format(setting, method))
                status = "screening_rejected"
                full_run_tag = None
            else:
                if not isinstance(selected, dict) or selection.get(
                    "selected_run_tag"
                ) != selected.get("run_tag"):
                    raise RegistryError("{} {} selected promotion mismatch".format(setting, method))
                if method not in confirmed_cells:
                    raise RegistryError("{} {} promotion lacks confirmation".format(setting, method))
                confirmation_cell = confirmed_cells[method]
                if (
                    confirmation_cell.get("parent_screening_run_tag")
                    != selected.get("run_tag")
                    or json.dumps(
                        confirmation_cell.get("parameters"), sort_keys=True,
                        separators=(",", ":"),
                    ) != json.dumps(
                        selected.get("parameters"), sort_keys=True,
                        separators=(",", ":"),
                    )
                ):
                    raise RegistryError("{} {} promotion/confirmation mismatch".format(setting, method))
                promoted.add(method)
                status = "full_confirmed"
                full_run_tag = confirmation_cell.get("run_tag")
            dispositions[(setting, method)] = {
                "status": status,
                "selection_decision": selection.get("decision"),
                "screening_run_tag": (
                    selected.get("run_tag") if isinstance(selected, dict) else None
                ),
                "full_run_tag": full_run_tag,
                "promotion_path": _repo_path(repo_root, promotion_path),
                "promotion_sha256": _sha256(promotion_path),
                "confirmation_path": _repo_path(repo_root, confirmation_path),
                "confirmation_sha256": _sha256(confirmation_path),
                "screening_phase_path": _repo_path(
                    repo_root, phase_documents[0][0]
                ),
                "screening_phase_sha256": _sha256(phase_documents[0][0]),
                "full_phase_path": _repo_path(repo_root, phase_documents[1][0]),
                "full_phase_sha256": _sha256(phase_documents[1][0]),
            }
        if set(confirmed_cells) != promoted:
            raise RegistryError("{} CONFIRMATION/promoted cell set mismatch".format(setting))
        screening_phase_document = phase_documents[0][1]
        full_phase_document = phase_documents[1][1]
        if screening_phase_document.get("job_count") != screening_phase[
            "planned_jobs"
        ] or full_phase_document.get("job_count") != len(promoted):
            raise RegistryError("{} phase job count disagrees with disposition".format(setting))
        empty_path = full_root / "EMPTY_COMPLETE.json"
        if not promoted:
            empty = _read_json(empty_path, "{} EMPTY_COMPLETE".format(setting))
            if any(empty.get(key) != value for key, value in {
                "schema": "navtta.vln_r2r_ce_targeted_supplement_empty_phase.v1",
                "phase_id": full_phase["phase_id"],
                "git_commit": document["git_commit"],
                "spec_sha256": document["spec_sha256"],
                "reason": "no_candidate_passed_the_predeclared_promotion_gate",
            }.items()):
                raise RegistryError("{} EMPTY_COMPLETE binding mismatch".format(setting))
            for method in allowed:
                dispositions[(setting, method)]["empty_complete_path"] = _repo_path(
                    repo_root, empty_path
                )
                dispositions[(setting, method)]["empty_complete_sha256"] = _sha256(
                    empty_path
                )
        elif empty_path.exists():
            raise RegistryError("{} nonempty full phase has stale EMPTY_COMPLETE".format(setting))
    expected_targets = {tuple(item) for item in spec["targeted_cells"]}
    if set(dispositions) != expected_targets:
        raise RegistryError("targeted supplement lacks an exact five-cell disposition")
    expected_result_keys = {
        "schema", "experiment_id", "batch_id", "benchmark", "split",
        "git_commit", "spec_sha256", "canonical_order_seed",
        "source_execution_jobs", "supervision_groups", "settings",
    }
    if set(document) != expected_result_keys or document.get(
        "supervision_groups"
    ) != supplement_spec["protocol"]["supervision_groups"]:
        raise RegistryError("targeted supplement RESULTS contract mismatch")
    job_evidence = _authenticate_supplement_jobs(
        campaign_root, phases, supplement_spec, promotions, confirmations,
        repo_root,
    )
    if set(job_evidence) != expected_targets:
        raise RegistryError("targeted supplement job evidence is incomplete")
    for cell, evidence in job_evidence.items():
        dispositions[cell]["job_evidence"] = evidence
    return {
        "plan": {
            "path": _repo_path(repo_root, plan_path),
            "sha256": _sha256(plan_path),
        },
        "cells": {
            "{}::{}".format(setting, method): dispositions[(setting, method)]
            for setting, method in sorted(dispositions)
        },
    }


def _load_supplement(path, spec, repo_root, source_records):
    path = Path(path).resolve()
    document = _read_json(path, "targeted supplement results")
    expected = {
        "schema": SUPPLEMENT_SCHEMA,
        "experiment_id": spec["targeted_supplement"]["experiment_id"],
        "batch_id": spec["targeted_supplement"]["batch_id"],
        "benchmark": spec["protocol"]["benchmark"],
        "split": "val_seen",
        "canonical_order_seed": 0,
        "source_execution_jobs": 0,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RegistryError("targeted supplement {} mismatch".format(key))
    if document.get("spec_sha256") != spec["targeted_supplement"]["spec"]["sha256"]:
        raise RegistryError("targeted supplement spec digest mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(document.get("git_commit", ""))):
        raise RegistryError("targeted supplement Git commit is invalid")
    supplement_spec_path = _bound_file(
        repo_root, spec["targeted_supplement"]["spec"],
        "targeted supplement spec",
    )
    supplement_spec = _read_json(
        supplement_spec_path, "targeted supplement spec"
    )
    supplement_spec["_path"] = str(supplement_spec_path)
    supplement_spec["_sha256"] = spec["targeted_supplement"]["spec"]["sha256"]
    supplement_spec["_batch_id"] = document["batch_id"]
    disposition_evidence = _validate_supplement_campaign_artifacts(
        path, document, spec, supplement_spec, repo_root, source_records
    )
    settings = document.get("settings")
    if not isinstance(settings, dict) or set(settings) != set(SETTINGS):
        raise RegistryError("targeted supplement settings mismatch")
    output = {setting: {} for setting in SETTINGS}
    targets = {tuple(item) for item in spec["targeted_cells"]}
    for setting in SETTINGS:
        confirmation = settings[setting]
        if confirmation.get("schema") != (
            "navtta.vln_r2r_ce_targeted_supplement_confirmation.v1"
        ):
            raise RegistryError("{} confirmation schema mismatch".format(setting))
        if confirmation.get("canonical_full_episodes") != 778 or (
            confirmation.get("canonical_order_seed") != 0
        ):
            raise RegistryError("{} confirmation protocol mismatch".format(setting))
        for key in ("experiment_id", "batch_id", "git_commit", "spec_sha256"):
            if confirmation.get(key) != document.get(key):
                raise RegistryError(
                    "{} confirmation {} mismatch".format(setting, key)
                )
        if confirmation.get("restart_from_source_checkpoint") is not True:
            raise RegistryError("{} confirmation did not restart from Source".format(setting))
        source = confirmation.get("source", {})
        expected_source = source_records[setting]
        if source.get("formal_manifest_sha256") != expected_source[
            "formal_manifest_sha256"
        ]:
            raise RegistryError("{} confirmation Source mismatch".format(setting))
        cells = confirmation.get("cells")
        if not isinstance(cells, dict):
            raise RegistryError("{} confirmation cells are malformed".format(setting))
        allowed = {method for cell_setting, method in targets if cell_setting == setting}
        if not set(cells).issubset(allowed):
            raise RegistryError("{} confirmation contains an untargeted cell".format(setting))
        for method, item in cells.items():
            run_tag = item.get("run_tag")
            run_id = "{}-{}-val_seen-v1.3-unified".format(run_tag, setting)
            raw_path = item.get("formal_manifest_path")
            if Path(str(raw_path)).parent.name != run_id:
                raise RegistryError("{} {} confirmation path mismatch".format(setting, method))
            record = {
                "origin": "targeted_supplement_full",
                "run_tag": run_tag,
                "parameters": item.get("parameters"),
                "metrics": _metrics(item.get("metrics"), "{} {} supplement".format(setting, method)),
                "checkpoint_sha256": expected_source["checkpoint_sha256"],
                "formal_manifest_path": "vln/results/runs/{}/manifest.json".format(run_id),
                "formal_manifest_sha256": item.get("formal_manifest_sha256"),
                "formal_immutable_identity_sha256": item.get(
                    "formal_immutable_identity_sha256"
                ),
                "adapter_diagnostics": item.get("adapter_diagnostics") or {},
                "parent_screening_run_tag": item.get("parent_screening_run_tag"),
            }
            if not isinstance(record["parent_screening_run_tag"], str) or not record[
                "parent_screening_run_tag"
            ]:
                raise RegistryError(
                    "{} {} lacks a screening parent".format(setting, method)
                )
            try:
                method_spec = supplement_spec["methods"][method]["settings"][setting]
            except (KeyError, TypeError) as error:
                raise RegistryError(
                    "{} {} is outside the targeted search spec".format(setting, method)
                ) from error
            allowed_parameters = []
            for point in method_spec.get("candidates", []):
                merged = dict(method_spec.get("fixed", {}))
                merged.update(point)
                allowed_parameters.append(merged)
            if not any(
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                == json.dumps(record["parameters"], sort_keys=True, separators=(",", ":"))
                for value in allowed_parameters
            ):
                raise RegistryError(
                    "{} {} parameters are outside the targeted candidate set"
                    .format(setting, method)
                )
            _validate_parameters(method, record["parameters"], "{} {}".format(setting, method))
            manifest_path, manifest = _validate_manifest(
                repo_root, record, setting, method, "val_seen"
            )
            _validate_manifest_parameters(
                setting, method, record["parameters"], manifest
            )
            record["metrics"] = _validate_aggregate_metrics(
                repo_root, manifest, record["metrics"],
                "{} {} supplement".format(setting, method),
            )
            aggregate_artifacts = [
                artifact for artifact in manifest.get("result_artifacts", [])
                if isinstance(artifact, dict)
                and str(artifact.get("name", "")).startswith("metrics/")
                and "/stats_ckpt_" in "/" + str(artifact.get("name", ""))
                and "/stats_ep_" not in "/" + str(artifact.get("name", ""))
            ]
            aggregate_artifact = aggregate_artifacts[0]
            if (
                _portable_repo_path(
                    repo_root, item.get("aggregate_artifact_path", ""),
                    "{} {} declared aggregate".format(setting, method),
                )
                != _local_artifact_path(
                    repo_root, aggregate_artifact.get("path", "")
                )
                or item.get("aggregate_artifact_sha256")
                != aggregate_artifact.get("sha256")
            ):
                raise RegistryError(
                    "{} {} confirmation aggregate binding mismatch".format(
                        setting, method
                    )
                )
            diagnostics = _validate_diagnostics(
                repo_root, manifest, setting, method, record["parameters"],
                record["metrics"], 778,
                "{} {} supplement".format(setting, method),
            )
            if json.dumps(
                record["adapter_diagnostics"], sort_keys=True, separators=(",", ":")
            ) != json.dumps(
                diagnostics["adapter"], sort_keys=True, separators=(",", ":")
            ):
                raise RegistryError(
                    "{} {} RESULTS diagnostics disagree with authenticated artifact"
                    .format(setting, method)
                )
            record["diagnostics_artifact_path"] = diagnostics["path"]
            record["diagnostics_artifact_sha256"] = diagnostics["sha256"]
            record["adapter_diagnostics"] = diagnostics["adapter"]
            if manifest.get("git_commit") != document.get("git_commit"):
                raise RegistryError(
                    "{} {} supplement Git commit mismatch".format(setting, method)
                )
            if manifest["immutable_identity_sha256"] != record[
                "formal_immutable_identity_sha256"
            ]:
                raise RegistryError("{} {} supplement identity mismatch".format(setting, method))
            record["formal_manifest_path"] = manifest_path
            output[setting][method] = record
    return path, document, output, disposition_evidence


def _eligible(candidate, method, source_metrics, spec):
    reasons = []
    tolerance = float(
        spec["selection"]["source_sr_floor_tolerance_percentage_points"]
    )
    if candidate["metrics"]["SR"] < source_metrics["SR"] - tolerance:
        reasons.append("below_source_sr_floor")
    if method == "fstta":
        maximum = int(spec["selection"]["fstta"]["max_fast_window_m"])
        m_value = candidate["parameters"].get("m")
        if isinstance(m_value, bool) or not isinstance(m_value, int) or not (
            1 <= m_value <= maximum
        ):
            reasons.append("invalid_fast_window")
        updates = candidate["adapter_diagnostics"].get("updates")
        if isinstance(updates, bool) or not isinstance(updates, (int, float)) or (
            not math.isfinite(float(updates)) or float(updates) <= 0
        ):
            reasons.append("no_effective_updates")
    return not reasons, reasons


def select_winners(spec, initial, supplement, source_records):
    targets = {tuple(item) for item in spec["targeted_cells"]}
    winners = {setting: {} for setting in SETTINGS}
    decisions = {setting: {} for setting in SETTINGS}
    tie_tolerance = float(spec["selection"]["metric_tie_absolute_tolerance"])
    for setting in SETTINGS:
        for method in METHODS:
            pool = [initial[setting][method]]
            if (setting, method) in targets and method in supplement[setting]:
                pool.append(supplement[setting][method])
            reviewed = []
            for candidate in pool:
                eligible, reasons = _eligible(
                    candidate, method, source_records[setting]["metrics"], spec
                )
                reviewed.append({
                    "candidate": candidate,
                    "eligible": eligible,
                    "reasons": reasons,
                })
            eligible = [item["candidate"] for item in reviewed if item["eligible"]]
            if not eligible:
                raise RegistryError(
                    "{} {} has no valid val_seen finalist; further search is required"
                    .format(setting, method)
                )
            eligible.sort(
                key=lambda item: (
                    float(item["metrics"]["SPL"]),
                    float(item["metrics"]["SR"]),
                    item["origin"] == "targeted_supplement_full",
                ),
                reverse=True,
            )
            winner = eligible[0]
            if len(eligible) > 1:
                left, right = eligible[:2]
                if all(math.isclose(
                    float(left["metrics"][name]), float(right["metrics"][name]),
                    rel_tol=0.0, abs_tol=tie_tolerance,
                ) for name in ("SPL", "SR")):
                    winner = next(
                        (item for item in eligible if item["origin"] == "targeted_supplement_full"),
                        left,
                    )
            winners[setting][method] = winner
            decisions[setting][method] = {
                "selected_run_tag": winner["run_tag"],
                "selected_origin": winner["origin"],
                "ranked_candidates": [
                    {
                        "run_tag": item["candidate"]["run_tag"],
                        "origin": item["candidate"]["origin"],
                        "metrics": item["candidate"]["metrics"],
                        "eligible": item["eligible"],
                        "reasons": item["reasons"],
                    }
                    for item in reviewed
                ],
            }
    return winners, decisions


def _selection_document(spec_path, spec, supplement_path, supplement_document,
                        supplement_dispositions, winners, decisions, source_path,
                        repo_root=REPO_ROOT):
    records = {}
    for setting in SETTINGS:
        records[setting] = {}
        for method in METHODS:
            winner = winners[setting][method]
            records[setting][method] = {
                "selection_status": "ready",
                "supervision_category": SUPERVISION[method],
                "selected_origin": winner["origin"],
                "parameters": winner["parameters"],
                "run_tag": winner["run_tag"],
                "metrics": winner["metrics"],
                "formal_manifest_path": winner["formal_manifest_path"],
                "formal_manifest_sha256": winner["formal_manifest_sha256"],
                "formal_immutable_identity_sha256": winner[
                    "formal_immutable_identity_sha256"
                ],
                "decision": decisions[setting][method],
            }
    return {
        "schema": SELECTION_SCHEMA,
        "selection_status": "complete",
        "protocol": {
            "benchmark": spec["protocol"]["benchmark"],
            "split": "val_seen",
            "episode_count": 778,
            "order_seed": 0,
            "episode_order_sha256": spec["protocol"]["selection_order_sha256"],
            "selection_scope": "val_seen_seed0_initial_plus_targeted_supplement",
            "selection_on_val_unseen": False,
        },
        "source_ledger": {
            "path": _repo_path(repo_root, source_path),
            "sha256": _sha256(source_path),
        },
        "evidence": {
            "finalization_spec": {
                "path": _repo_path(repo_root, spec_path),
                "sha256": _sha256(spec_path),
            },
            "targeted_supplement_results": {
                "path": _repo_path(repo_root, supplement_path),
                "sha256": _sha256(supplement_path),
                "git_commit": supplement_document.get("git_commit"),
            },
            "targeted_supplement_dispositions": supplement_dispositions,
        },
        "selection_policy": spec["selection"],
        "records": records,
    }


def _registry_document(selection, selection_path, source_records, source_path,
                       repo_root=REPO_ROOT):
    records = {}
    for setting in SETTINGS:
        source = source_records[setting]
        source_entry = {
            "method": "source",
            "supervision_category": SUPERVISION["source"],
            "parameters": {"action_selection": "argmax", "action_seed": 0},
            "run_tag": source["run_tag"],
            "metrics": source["metrics"],
            "delta_vs_source_pp": {"SR": 0.0, "SPL": 0.0},
            "checkpoint_sha256": source["checkpoint_sha256"],
            "episode_order_sha256": source["episode_order_sha256"],
            "formal_manifest_path": source["formal_manifest_path"],
            "formal_manifest_sha256": source["formal_manifest_sha256"],
            "formal_immutable_identity_sha256": source[
                "immutable_identity_sha256"
            ],
        }
        records[setting] = {"source": source_entry}
        for method in METHODS:
            item = selection["records"][setting][method]
            records[setting][method] = {
                "method": method,
                "supervision_category": item["supervision_category"],
                "selected_origin": item["selected_origin"],
                "parameters": item["parameters"],
                "run_tag": item["run_tag"],
                "metrics": item["metrics"],
                "delta_vs_source_pp": {
                    name: round(
                        float(item["metrics"][name]) - float(source["metrics"][name]), 8
                    )
                    for name in ("SR", "SPL")
                },
                "checkpoint_sha256": source["checkpoint_sha256"],
                "formal_manifest_path": item["formal_manifest_path"],
                "formal_manifest_sha256": item["formal_manifest_sha256"],
                "formal_immutable_identity_sha256": item[
                    "formal_immutable_identity_sha256"
                ],
            }
    return {
        "schema": REGISTRY_SCHEMA,
        "registry_status": "complete",
        "protocol_status": "complete_val_seen_seed0_selection",
        "provenance_status": "complete_formal_manifest_identity_verified",
        "publication_status": "selection_registry_not_publication_final",
        "benchmark": "r2r-ce",
        "split": "val_seen",
        "protocol": selection["protocol"],
        "source_ledger": {
            "path": _repo_path(repo_root, source_path),
            "sha256": _sha256(source_path),
        },
        "selected_winners": {
            "path": _repo_path(repo_root, selection_path),
            "sha256": _bytes_sha256(_json_bytes(selection)),
        },
        "supervision_categories": {
            "source_no_adaptation": {"methods": ["source"], "uses_episode_feedback": False},
            "unsupervised_tta": {"methods": ["tent", "fstta", "eam"], "uses_episode_feedback": False},
            "binary_episode_feedback_tta": {
                "methods": ["feedtta", "atena"],
                "uses_episode_feedback": True,
                "feedback": "binary_navigation_success",
            },
        },
        "records": records,
    }


def validate_registry_document(registry):
    if registry.get("schema") != REGISTRY_SCHEMA or registry.get(
        "registry_status"
    ) != "complete":
        raise RegistryError("R2R-CE registry is incomplete")
    if registry.get("benchmark") != "r2r-ce" or registry.get("split") != "val_seen":
        raise RegistryError("R2R-CE registry protocol mismatch")
    protocol = registry.get("protocol", {})
    expected_protocol = {
        "benchmark": "r2r_ce_v1_3_unified_etpnav_bevbert",
        "split": "val_seen",
        "episode_count": 778,
        "order_seed": 0,
        "episode_order_sha256": "93f44aab1be2e3d96b867a323172ab3bbbaa4dd97e5b3fa450fe839a6c4bd94e",
        "selection_scope": "val_seen_seed0_initial_plus_targeted_supplement",
        "selection_on_val_unseen": False,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise RegistryError("R2R-CE registry protocol {} mismatch".format(key))
    records = registry.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("R2R-CE registry setting matrix mismatch")
    for setting in SETTINGS:
        if set(records[setting]) != {"source", *METHODS}:
            raise RegistryError("{} registry method matrix mismatch".format(setting))
        source = records[setting]["source"]
        if source.get("method") != "source" or source.get(
            "supervision_category"
        ) != SUPERVISION["source"]:
            raise RegistryError("{} Source registry record malformed".format(setting))
        if source.get("delta_vs_source_pp") != {"SR": 0.0, "SPL": 0.0}:
            raise RegistryError("{} Source delta is nonzero".format(setting))
        for method in METHODS:
            item = records[setting][method]
            if item.get("parameters") is None or item.get("method") != method:
                raise RegistryError("{} {} registry record malformed".format(setting, method))
            if item.get("supervision_category") != SUPERVISION[method]:
                raise RegistryError("{} {} supervision mismatch".format(setting, method))
    return registry


def _rebuild_selection(selection_path, selection, repo_root):
    evidence = selection.get("evidence", {})
    spec_binding = evidence.get("finalization_spec")
    results_binding = evidence.get("targeted_supplement_results")
    spec_path = _bound_file(repo_root, spec_binding, "finalization spec")
    results_path = _bound_file(
        repo_root,
        {"path": results_binding.get("path"), "sha256": results_binding.get("sha256")}
        if isinstance(results_binding, dict) else results_binding,
        "targeted supplement results",
    )
    _, spec = load_spec(spec_path, repo_root)
    expected_selection_path = _resolve(
        repo_root, spec["outputs"]["selected_winners"]
    ).resolve()
    if Path(selection_path).resolve() != expected_selection_path:
        raise RegistryError("selected-winner output path differs from finalization spec")
    source_path, source_records = _val_seen_source_records(spec, repo_root)
    initial = _initial_candidates(spec, repo_root, source_records)
    (
        authenticated_results_path,
        supplement_document,
        supplement,
        dispositions,
    ) = _load_supplement(results_path, spec, repo_root, source_records)
    winners, decisions = select_winners(spec, initial, supplement, source_records)
    expected = _selection_document(
        spec_path, spec, authenticated_results_path, supplement_document,
        dispositions, winners, decisions, source_path, repo_root,
    )
    if json.dumps(selection, sort_keys=True, separators=(",", ":")) != json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ):
        raise RegistryError(
            "selected winners do not deterministically replay from val_seen evidence"
        )
    return expected, source_path, source_records


def validate_registry(path, repo_root=REPO_ROOT):
    registry = validate_registry_document(_read_json(path, "R2R-CE registry"))
    selection_binding = registry.get("selected_winners", {})
    selection_path = _resolve(repo_root, selection_binding.get("path", ""))
    if not selection_path.is_file() or _sha256(selection_path) != selection_binding.get("sha256"):
        raise RegistryError("R2R-CE selected-winner binding mismatch")
    selection = _read_json(selection_path, "R2R-CE selected winners")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get(
        "selection_status"
    ) != "complete":
        raise RegistryError("R2R-CE selected winners are incomplete")
    selection, replayed_source_path, replayed_sources = _rebuild_selection(
        selection_path, selection, repo_root
    )
    source_binding = registry.get("source_ledger", {})
    source_path = _resolve(repo_root, source_binding.get("path", ""))
    if not source_path.is_file() or _sha256(source_path) != source_binding.get("sha256"):
        raise RegistryError("R2R-CE val_seen Source-ledger binding mismatch")
    source_ledger = _read_json(source_path, "R2R-CE val_seen Source ledger")
    if source_ledger.get("schema") != VAL_SEEN_SOURCE_SCHEMA or set(
        source_ledger.get("settings", {})
    ) != set(SETTINGS):
        raise RegistryError("R2R-CE val_seen Source ledger is malformed")
    if source_path.resolve() != replayed_source_path.resolve():
        raise RegistryError("registry Source ledger differs from replayed selection")
    for setting in SETTINGS:
        source = registry["records"][setting]["source"]
        if source.get("metrics") != replayed_sources[setting].get("metrics"):
            raise RegistryError("{} Source metrics differ from replay".format(setting))
        ledger_source = source_ledger["settings"][setting]
        if source.get("formal_manifest_sha256") != ledger_source.get(
            "formal_manifest", {}
        ).get("sha256") or source.get("checkpoint_sha256") != ledger_source.get(
            "checkpoint_sha256"
        ):
            raise RegistryError("{} registry/Source-ledger mismatch".format(setting))
        _validate_source_manifest(repo_root, source, setting, "val_seen")
        for method in METHODS:
            registry_record = registry["records"][setting][method]
            selection_record = selection.get("records", {}).get(setting, {}).get(method)
            if not isinstance(selection_record, dict):
                raise RegistryError("selected-winner matrix is incomplete")
            for key in (
                "parameters", "run_tag", "metrics", "formal_manifest_path",
                "formal_manifest_sha256", "formal_immutable_identity_sha256",
            ):
                if registry_record.get(key) != selection_record.get(key):
                    raise RegistryError(
                        "{} {} registry/selection {} mismatch".format(
                            setting, method, key
                        )
                    )
            _validate_manifest(repo_root, registry_record, setting, method, "val_seen")
    expected_registry = _registry_document(
        selection, selection_path, replayed_sources, replayed_source_path,
        repo_root,
    )
    if json.dumps(registry, sort_keys=True, separators=(",", ":")) != json.dumps(
        expected_registry, sort_keys=True, separators=(",", ":")
    ):
        raise RegistryError(
            "registry does not deterministically replay from selected winners"
        )
    return registry


def _frozen_spec(spec, registry, registry_path, registry_sha256,
                 unseen_source_path, repo_root=REPO_ROOT):
    jobs = []
    for setting in SETTINGS:
        for method in METHODS:
            item = registry["records"][setting][method]
            jobs.append({
                "setting": setting,
                "model": MODEL_FOR_SETTING[setting],
                "method": method,
                "selected_run_tag": item["run_tag"],
                "selected_formal_manifest_sha256": item[
                    "formal_manifest_sha256"
                ],
            })
    execution = spec["val_unseen_execution"]
    return {
        "schema": FROZEN_SPEC_SCHEMA,
        "experiment_id": "vln-r2r-ce-val-unseen-frozen-eval-v1",
        "canonical_batch_id": "vln-r2r-ce-val-unseen-frozen-eval-v1-seed0",
        "status": "reviewed_and_source_evidence_complete",
        "purpose": (
            "Evaluate ten frozen R2R-CE val_seen winners on canonical "
            "val_unseen without Source execution or val_unseen reselection."
        ),
        "registry_dependency": {
            "path": _repo_path(repo_root, registry_path),
            "sha256": registry_sha256,
            "schema": REGISTRY_SCHEMA,
            "required_status": "complete",
        },
        "source_control": {
            "manifest": _repo_path(repo_root, unseen_source_path),
            "sha256": _sha256(unseen_source_path),
            "execution": "reuse_only",
            "rerun_forbidden": True,
        },
        "protocol": {
            "benchmark": spec["protocol"]["benchmark"],
            "split": "val_unseen",
            "episode_count": 1839,
            "canonical_order_seed": 0,
            "episode_order_sha256": spec["protocol"]["evaluation_order_sha256"],
            "order_manifest": {
                "path": "vln/manifests/episode_order/r2r_ce_v1_3_unified/val_unseen.json",
                "sha256": "dc9600fd768f1c885601928c5599e7dc8833f07da57c0aa22d01a9d4fe79047f",
            },
            "order_seed_cli_forbidden": True,
            "full_split_only": True,
            "frozen_from": "r2r_ce_val_seen_seed0_final_registry",
            "selection_on_val_unseen": False,
            "report_all_cells": True,
        },
        "matrix": {
            "model_order": list(MODELS),
            "setting_order": list(SETTINGS),
            "method_order": list(METHODS),
            "strict_model_barrier": True,
            "max_workers": 3,
            "jobs": jobs,
        },
        "execution": execution,
        "budget": {
            "source_execution_jobs": 0,
            "tta_jobs": 10,
            "total_executed_jobs": 10,
        },
    }


def build_documents(spec_path=DEFAULT_SPEC, supplement_results=None,
                    repo_root=REPO_ROOT):
    repo_root = Path(repo_root).resolve()
    spec_path, spec = load_spec(spec_path, repo_root)
    source_path, source_records = _val_seen_source_records(spec, repo_root)
    unseen_source_path, _ = validate_val_unseen_source_ledger(
        spec, repo_root, require_artifacts=True
    )
    if supplement_results is None:
        supplement_results = _resolve(
            repo_root, spec["targeted_supplement"]["results_relative_path"]
        )
    (
        supplement_path,
        supplement_document,
        supplement,
        supplement_dispositions,
    ) = _load_supplement(
        supplement_results, spec, repo_root, source_records
    )
    initial = _initial_candidates(spec, repo_root, source_records)
    winners, decisions = select_winners(
        spec, initial, supplement, source_records
    )
    outputs = spec["outputs"]
    selection_path = _resolve(repo_root, outputs["selected_winners"]).resolve()
    registry_path = _resolve(repo_root, outputs["registry"]).resolve()
    frozen_path = _resolve(repo_root, outputs["val_unseen_spec"]).resolve()
    selection = _selection_document(
        spec_path, spec, supplement_path, supplement_document,
        supplement_dispositions, winners, decisions, source_path, repo_root,
    )
    registry = _registry_document(
        selection, selection_path, source_records, source_path, repo_root
    )
    validate_registry_document(registry)
    registry_sha256 = _bytes_sha256(_json_bytes(registry))
    frozen = _frozen_spec(
        spec, registry, registry_path, registry_sha256, unseen_source_path,
        repo_root,
    )
    return {
        "selection": (selection_path, selection),
        "registry": (registry_path, registry),
        "frozen_spec": (frozen_path, frozen),
    }


def _atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
    os.replace(str(temporary), str(path))


def write_documents(documents):
    for path, value in documents.values():
        _atomic_write(path, value)
    return documents


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--supplement-results", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    documents = build_documents(args.spec, args.supplement_results)
    if args.write:
        write_documents(documents)
        validate_registry(documents["registry"][0])
        frozen = _read_json(
            documents["frozen_spec"][0], "generated frozen val_unseen spec"
        )
        if _sha256(documents["registry"][0]) != frozen[
            "registry_dependency"
        ]["sha256"]:
            raise RegistryError("generated frozen spec has a stale registry digest")
    for label, (path, value) in documents.items():
        print("{}={} sha256={}".format(
            label, path, _bytes_sha256(_json_bytes(value))
        ))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegistryError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
