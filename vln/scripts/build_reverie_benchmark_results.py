#!/usr/bin/env python3
"""Build the final cross-split REVERIE comparison from downloaded evidence.

The frozen ``val_seen`` registry is the only hyperparameter-selection source.
This offline tool requires a complete 15-job ``val_unseen`` batch and a
complete nine-job hidden-``test`` submission batch.  It never launches an
experiment, never reselects a configuration, and never represents a hidden
test submission as a locally measured result.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
import tta_config_cli  # noqa: E402


RESULT_SCHEMA = "navtta.vln_reverie_benchmark_final_results.v1"
REGISTRY_SCHEMA = "navtta.vln_reverie_final_registry.v1"
SPEC_SCHEMA = "navtta.vln_reverie_val_unseen_frozen_eval.v1"
VAL_SOURCE_SCHEMA = "navtta.vln_reverie_val_unseen_reused_source_controls.v1"
TEST_SOURCE_SCHEMA = "navtta.vln_reverie_test_reused_source_submissions.v1"
BATCH_SCHEMA = "navtta.vln_reverie_val_unseen_frozen_batch.v1"
VAL_SUMMARY_SCHEMA = "navtta.vln_reverie_val_unseen_frozen_summary.v1"
TEST_SUMMARY_SCHEMA = "navtta.vln_reverie_test_submission_summary.v1"
VAL_JOB_SCHEMA = "navtta.vln_reverie_val_unseen_frozen_job.v1"
TEST_JOB_SCHEMA = "navtta.vln_reverie_test_submission_job.v1"
VAL_SEEN_SOURCE_SCHEMA = "navtta.vln_reverie_reused_source_controls.v1"
SEARCH_SPEC_SCHEMA = "navtta.vln_reverie_val_seen_small_hparam_search.v1"
SEARCH_BATCH_SCHEMA = "navtta.vln_reverie_small_search_batch.v1"
SEARCH_SELECTION_SCHEMA = (
    "navtta.vln_reverie_val_seen_small_search_selection.v1"
)

DEFAULT_REGISTRY = REPO_ROOT / "vln/results/final/reverie/registry.json"
DEFAULT_SPEC = REPO_ROOT / "vln/experiments/reverie_val_unseen_frozen_eval_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "vln/results/final/reverie/benchmark_results.json"
DEFAULT_REPORT = REPO_ROOT / "vln/results/final/reverie/BENCHMARK_RESULTS.md"

SETTINGS = ("duet-reverie", "hamt-reverie", "goat-reverie")
MODELS = ("duet", "hamt", "goat")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
TEST_METHODS = ("tent", "fstta", "eam")
UNAVAILABLE_TEST_METHODS = ("feedtta", "atena")
ALL_METHODS = ("source",) + METHODS
METRICS = ("SR", "SPL", "RGS", "RGSPL")
MODEL_FOR_SETTING = dict(zip(SETTINGS, MODELS))
BENCHMARK_FOR_SETTING = {
    "duet-reverie": "reverie_discrete_duet_hamt",
    "hamt-reverie": "reverie_discrete_duet_hamt",
    "goat-reverie": "reverie_discrete_goat",
}
SUPERVISION_FOR_METHOD = {
    "source": "source_no_adaptation",
    "tent": "unsupervised_tta",
    "fstta": "unsupervised_tta",
    "eam": "unsupervised_tta",
    "feedtta": "binary_episode_feedback_tta",
    "atena": "binary_episode_feedback_tta",
}
FEEDBACK_ENDPOINT = {
    "feedtta": (
        "reverie_submitted_trajectory_evaluator_navigation_"
        "success_every_episode"
    ),
    "atena": (
        "reverie_submitted_trajectory_evaluator_navigation_"
        "success_lazy_query"
    ),
}
VAL_SEEN_EPISODES = 1423
VAL_UNSEEN_EPISODES = 3521
TEST_EPISODES = 6292
VAL_SEEN_ORDER = "aef9a2b094080347dc928cc19ddc40946299f77a91d380ee00bee10449cf1fd1"
VAL_UNSEEN_ORDER = "67c909272166bb3d8e7ee612d597de190f9218a6342da6357b0c7aea8832baca"
TEST_ORDER = "5f527c1d8442ae2b003cc5b429164eaf003b2e7acc16a8babd818874d3e1ab1d"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
METRIC_RE = re.compile(
    r"\b(sr|spl|rgs|rgspl):\s*(-?[0-9]+(?:\.[0-9]+)?)"
)


class ArchiveError(RuntimeError):
    """Downloaded evidence is incomplete, inconsistent, or unauthenticated."""


def _read_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArchiveError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise ArchiveError("{} must be a JSON object: {}".format(label, path))
    return value


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def _resolve(repo_root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def _display_path(repo_root, path):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _require_file(path, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise ArchiveError("missing {}: {}".format(label, path))
    return path


def _same(left, right):
    return _canonical(left) == _canonical(right)


def _metrics(value, label):
    if not isinstance(value, dict) or set(value) != set(METRICS):
        raise ArchiveError("{} metrics must contain exactly {}".format(label, METRICS))
    result = {}
    for name in METRICS:
        number = value[name]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or not 0.0 <= float(number) <= 100.0
        ):
            raise ArchiveError("{} {} is invalid".format(label, name))
        result[name] = number
    return result


def _validate_parameters(method, parameters, label):
    if not isinstance(parameters, dict) or not parameters:
        raise ArchiveError("{} frozen parameters are invalid".format(label))
    allowed = tta_config_cli.COMMON | tta_config_cli.METHOD_KEYS.get(method, set())
    unknown = set(parameters).difference(allowed)
    if unknown:
        raise ArchiveError("{} has unknown parameters: {}".format(
            label, ", ".join(sorted(unknown))
        ))
    for key, value in parameters.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ArchiveError("{} has non-finite parameter {}".format(label, key))
    for key in ("action_seed", "sgr_seed"):
        if key in parameters and type(parameters[key]) is not int:
            raise ArchiveError("{} {} must be an exact integer".format(label, key))
    if method == "tent" and parameters.get("update_interval") != 1:
        raise ArchiveError("{} Tent update_interval must equal 1".format(label))
    if method in UNAVAILABLE_TEST_METHODS and parameters.get(
        "action_selection"
    ) != "argmax":
        raise ArchiveError("{} must use target-native argmax".format(label))


def _job_digest(setting, method, record):
    identity = {
        "setting": setting,
        "method": method,
        "selected_run_tag": record["run_tag"],
        "parameters": record["parameters"],
    }
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:10]


def _expected_process_group_token(metadata, attempt_dir):
    del attempt_dir
    identity = {
        "schema": "navtta.vln_reverie_process_group.v1",
        "batch_id": metadata["batch_id"],
        "run_tag": metadata["run_tag"],
        "attempt": metadata["attempt"],
        "setting": metadata["setting"],
        "method": metadata["method"],
        "spec_sha256": metadata["spec_sha256"],
        "git_commit": metadata["git_commit"],
    }
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def _delta(metrics, source):
    return {
        name: round(float(metrics[name]) - float(source[name]), 6)
        for name in METRICS
    }


def _formal_path(repo_root, declared, run_id, label):
    relative = Path("vln/results/runs") / run_id / "manifest.json"
    normalized = str(declared or "").replace("\\", "/").rstrip("/")
    expected = relative.as_posix()
    if normalized != expected and not normalized.endswith("/" + expected):
        raise ArchiveError("{} formal-manifest path is noncanonical".format(label))
    return _require_file(Path(repo_root) / relative, "{} formal manifest".format(label))


def _artifact_index(manifest, label):
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ArchiveError("{} formal manifest has no result artifacts".format(label))
    by_digest = {}
    for item in artifacts:
        if not isinstance(item, dict):
            raise ArchiveError("{} has malformed artifact metadata".format(label))
        name = item.get("name")
        path = str(item.get("path", "")).replace("\\", "/")
        digest = item.get("sha256")
        size = item.get("size")
        if not isinstance(name, str) or not name:
            raise ArchiveError("{} artifact name is invalid".format(label))
        if path != name and not path.endswith("/" + name):
            raise ArchiveError("{} artifact path/name mismatch".format(label))
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ArchiveError("{} artifact size is invalid".format(label))
        if not _is_sha256(digest):
            raise ArchiveError("{} artifact digest is invalid".format(label))
        by_digest.setdefault(digest, []).append(item)
    return by_digest


def _artifact_path(repo_root, item, label):
    declared = str(item.get("path", ""))
    direct = Path(declared)
    candidates = []
    if direct.is_absolute():
        candidates.append(direct)
    else:
        candidates.append(Path(repo_root) / direct)
    normalized = declared.replace("\\", "/")
    marker = "vln/results/"
    offset = normalized.find(marker)
    if offset >= 0:
        candidates.append(Path(repo_root) / normalized[offset:])
    result_root = (Path(repo_root) / "vln/results").resolve()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            try:
                candidate.relative_to(result_root)
            except ValueError:
                raise ArchiveError("{} artifact escapes vln/results".format(label))
            if (
                candidate.stat().st_size != item.get("size")
                or _sha256(candidate) != item.get("sha256")
            ):
                raise ArchiveError("{} artifact changed".format(label))
            return candidate
    raise ArchiveError("missing downloaded {} artifact".format(label))


def _result_root_path(repo_root, declared, label):
    raw = str(declared or "")
    candidates = [Path(raw)]
    normalized = raw.replace("\\", "/")
    marker = "vln/results/"
    offset = normalized.find(marker)
    if offset >= 0:
        candidates.append(Path(repo_root) / normalized[offset:])
    allowed_root = (Path(repo_root) / "vln/results").resolve()
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_dir():
            continue
        try:
            candidate.relative_to(allowed_root)
        except ValueError:
            raise ArchiveError("{} result root escapes vln/results".format(label))
        return candidate
    raise ArchiveError("missing downloaded {} result root".format(label))


def _artifact_for_digest(repo_root, artifacts, digest, label, suffix=None):
    if not _is_sha256(digest):
        raise ArchiveError("{} digest is invalid".format(label))
    matches = artifacts.get(digest, [])
    if suffix is not None:
        matches = [
            item for item in matches
            if str(item.get("name", "")).endswith(suffix)
        ]
    if len(matches) != 1:
        raise ArchiveError("{} is not uniquely manifest-authenticated".format(label))
    return _artifact_path(repo_root, matches[0], label), matches[0]


def _declared_path_matches(repo_root, declared, actual):
    actual = Path(actual).resolve()
    candidate = Path(str(declared))
    if not candidate.is_absolute():
        candidate = (Path(repo_root) / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate == actual:
        return True
    normalized = str(declared or "").replace("\\", "/")
    return normalized.endswith("/" + _display_path(repo_root, actual))


def _assert_no_hidden_test_metrics(repo_root, artifacts, result_root, label):
    rows = []
    for items in artifacts.values():
        for item in items:
            if str(item.get("name", "")).endswith("valid.txt"):
                rows.append(item)
    if len(rows) != 1:
        raise ArchiveError("{} must authenticate exactly one test log".format(label))
    authenticated_log = _artifact_path(repo_root, rows[0], label + " test log")
    result_root = Path(result_root).resolve()
    logs = sorted(result_root.rglob("valid.txt"))
    if authenticated_log not in [path.resolve() for path in logs]:
        raise ArchiveError("{} authenticated test log is outside result tree".format(label))
    for path in logs:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            values = {key.upper() for key, _ in METRIC_RE.findall(line)}
            if "Env name: test" in line and values:
                raise ArchiveError(
                    "{} contains forbidden local hidden-test metrics".format(label)
                )


def _validate_val_metric_tree(result_root, metric_path, metrics, label):
    result_root = Path(result_root).resolve()
    matches = []
    for path in sorted(result_root.rglob("valid.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            values = {
                key.upper(): float(value) for key, value in METRIC_RE.findall(line)
            }
            if set(values) == set(METRICS) and "Env name: val_unseen" in line:
                matches.append((path.resolve(), _metrics(values, label + " result tree"), line))
    if len(matches) != 1 or matches[0][0] != Path(metric_path).resolve():
        raise ArchiveError(
            "{} result tree must contain exactly the authenticated metric row".format(label)
        )
    _assert_metrics_equal(matches[0][1], metrics, label)
    return matches[0][2]


def _validate_test_submission_tree(result_root, submission_path, label):
    submissions = sorted(Path(result_root).resolve().rglob("submit_test*.json"))
    if (
        len(submissions) != 1
        or submissions[0].resolve() != Path(submission_path).resolve()
    ):
        raise ArchiveError(
            "{} result tree must contain exactly the authenticated submission".format(label)
        )


def _validate_manifest(
    repo_root,
    declared_path,
    declared_sha256,
    setting,
    method,
    split,
    run_tag,
    checkpoint_sha256,
    order_sha256,
    dataset_sha256=None,
    immutable_sha256=None,
    git_commit=None,
    parameters=None,
    result_root=None,
    order_manifest=None,
    order_manifest_sha256=None,
):
    label = "{} {} {}".format(setting, method, split)
    if not isinstance(run_tag, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", run_tag):
        raise ArchiveError("{} run tag is invalid".format(label))
    run_id = "{}-{}-{}-native".format(run_tag, setting, split)
    path = _formal_path(repo_root, declared_path, run_id, label)
    if not _is_sha256(declared_sha256) or _sha256(path) != declared_sha256:
        raise ArchiveError("{} formal-manifest SHA256 mismatch".format(label))
    manifest = _read_json(path, "{} formal manifest".format(label))
    source_setting = "{}:{}:native".format(setting, split)
    if method != "source":
        source_setting += ":{}".format(method)
    expected = {
        "run_id": run_id,
        "task": "vln",
        "benchmark": BENCHMARK_FOR_SETTING[setting],
        "model": MODEL_FOR_SETTING[setting],
        "method": method,
        "run_tag": run_tag,
        "source_setting": source_setting,
        "seed": 0,
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ArchiveError("{} manifest {} mismatch".format(label, key))
    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or HEX_COMMIT.fullmatch(commit) is None:
        raise ArchiveError("{} manifest git commit is invalid".format(label))
    if git_commit is not None and commit != git_commit:
        raise ArchiveError("{} manifest git commit changed".format(label))
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise ArchiveError("{} checkpoint digest mismatch".format(label))
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("stream_order_sha256") != order_sha256:
        raise ArchiveError("{} episode-order digest mismatch".format(label))
    if dataset_sha256 is not None and dataset.get("stream_content_sha256") != dataset_sha256:
        raise ArchiveError("{} dataset digest mismatch".format(label))
    identity = manifest.get("immutable_identity_sha256")
    if not _is_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise ArchiveError("{} immutable identity mismatch".format(label))
    if immutable_sha256 is not None and identity != immutable_sha256:
        raise ArchiveError("{} pinned immutable identity mismatch".format(label))
    artifacts = _artifact_index(manifest, label)
    for items in artifacts.values():
        for item in items:
            _artifact_path(
                repo_root, item,
                "{} manifest artifact {}".format(label, item.get("name")),
            )
    if method != "source" and parameters is not None:
        overrides = manifest.get("config_overrides")
        if not isinstance(overrides, list):
            raise ArchiveError("{} formal manifest lacks config_overrides".format(label))

        def flag_value(flag):
            positions = [
                index for index, value in enumerate(overrides) if value == flag
            ]
            if len(positions) != 1 or positions[0] + 1 >= len(overrides):
                raise ArchiveError("{} formal manifest has invalid {}".format(label, flag))
            return str(overrides[positions[0] + 1])

        if "--test" not in overrides:
            raise ArchiveError("{} formal command is not an evaluation command".format(label))
        if flag_value("--eval_splits") != split or flag_value("--seed") != "0":
            raise ArchiveError("{} formal command split/seed mismatch".format(label))
        if result_root is not None and Path(flag_value("--output_dir")).resolve() != Path(
            result_root
        ).resolve():
            raise ArchiveError("{} formal output directory mismatch".format(label))
        if order_manifest is not None and not _declared_path_matches(
            repo_root,
            flag_value("--episode_order_manifest"),
            Path(order_manifest).resolve().parent,
        ):
            raise ArchiveError("{} formal episode-order path mismatch".format(label))
        try:
            start = overrides.index("--tta_method")
            actual_tta = [str(value) for value in overrides[start:]]
            expected_tta = tta_config_cli._discrete(
                method, parameters, "__DIAGNOSTICS__"
            )
        except ValueError as error:
            raise ArchiveError("{} parameters cannot translate: {}".format(label, error))
        for tokens in (actual_tta, expected_tta):
            try:
                position = tokens.index("--tta_diagnostics")
                tokens[position + 1] = "__DIAGNOSTICS__"
            except (ValueError, IndexError):
                raise ArchiveError("{} TTA diagnostics CLI is malformed".format(label))
        if actual_tta != expected_tta:
            raise ArchiveError("{} effective TTA parameters mismatch".format(label))
        root = Path(result_root).resolve() if result_root is not None else None
        if root is not None:
            for items in artifacts.values():
                for item in items:
                    try:
                        Path(str(item["path"])).resolve().relative_to(root)
                    except ValueError:
                        raise ArchiveError("{} formal artifact escapes result root".format(label))
    if order_manifest_sha256 is not None and manifest.get(
        "pinned_manifests", {}
    ).get("episode_order", {}).get("sha256") != order_manifest_sha256:
        raise ArchiveError("{} pinned episode-order manifest mismatch".format(label))
    provenance = {
        "run_tag": run_tag,
        "git_commit": commit,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_sha256": dataset.get("stream_content_sha256"),
        "episode_order_sha256": dataset.get("stream_order_sha256"),
        "formal_manifest_path": _display_path(repo_root, path),
        "formal_manifest_sha256": declared_sha256,
        "formal_immutable_identity_sha256": identity,
    }
    return manifest, artifacts, provenance


def _parse_metrics(path, split, label):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        values = {
            key.upper(): float(value) for key, value in METRIC_RE.findall(line)
        }
        if set(values) == set(METRICS) and "Env name: {}".format(split) in line:
            rows.append(values)
    if len(rows) != 1:
        raise ArchiveError(
            "{} must contain exactly one authenticated {} metric row".format(label, split)
        )
    return _metrics(rows[0], label + " artifact")


def _assert_metrics_equal(actual, expected, label):
    if any(
        not math.isclose(float(actual[name]), float(expected[name]), abs_tol=1e-9)
        for name in METRICS
    ):
        raise ArchiveError("{} metrics differ from authenticated artifact".format(label))


def _validate_order_binding(
    repo_root, binding, settings, benchmark, split, episodes, order_sha256, label
):
    if not isinstance(binding, dict) or tuple(binding.get("settings", ())) != tuple(settings):
        raise ArchiveError("{} setting binding mismatch".format(label))
    path = _require_file(_resolve(repo_root, binding.get("path", "")), label)
    if not _is_sha256(binding.get("sha256")) or _sha256(path) != binding["sha256"]:
        raise ArchiveError("{} digest mismatch".format(label))
    document = _read_json(path, label)
    expected = {
        "schema": "navtta.episode_order.v1",
        "benchmark": benchmark,
        "split": split,
        "episode_count": episodes,
        "order_sha256": order_sha256,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise ArchiveError("{} {} mismatch".format(label, key))
    dataset_sha256 = document.get("dataset", {}).get("sha256")
    if not _is_sha256(dataset_sha256):
        raise ArchiveError("{} dataset digest is invalid".format(label))
    return path, document


def _load_val_seen_source_ledger(path, registry, protocol):
    ledger = _read_json(path, "REVERIE val_seen Source ledger")
    expected = {
        "schema": VAL_SEEN_SOURCE_SCHEMA,
        "benchmark": "reverie",
        "split": "val_seen",
        "source_protocol": "standard_argmax",
        "episode_count": VAL_SEEN_EPISODES,
        "order_seed": 0,
        "episode_order_sha256": VAL_SEEN_ORDER,
    }
    if set(ledger) != set(expected) | {
        "dataset_version", "hardware", "source_batch", "records"
    }:
        raise ArchiveError("REVERIE val_seen Source ledger contract is malformed")
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise ArchiveError("REVERIE val_seen Source ledger {} mismatch".format(key))
    if any(
        not isinstance(ledger.get(key), str) or not ledger[key]
        for key in ("dataset_version", "hardware")
    ):
        raise ArchiveError("REVERIE val_seen Source provenance is invalid")
    source_batch = ledger.get("source_batch")
    if (
        not isinstance(source_batch, dict)
        or set(source_batch) != {"batch_id", "git_commit"}
        or not isinstance(source_batch.get("batch_id"), str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", source_batch["batch_id"])
        or not isinstance(source_batch.get("git_commit"), str)
        or HEX_COMMIT.fullmatch(source_batch["git_commit"]) is None
    ):
        raise ArchiveError("REVERIE val_seen Source batch provenance is invalid")
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise ArchiveError("REVERIE val_seen Source matrix is incomplete")
    expected_record_keys = {
        "model", "run_tag", "parameters", "metrics", "checkpoint_sha256",
        "dataset_sha256", "formal_manifest_path", "formal_manifest_sha256",
        "immutable_identity_sha256", "metrics_artifact_path",
        "metrics_artifact_sha256",
    }
    for setting in SETTINGS:
        record = records[setting]
        registry_record = registry["records"][setting]["source"]
        label = "{} Source".format(setting)
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise ArchiveError("{} ledger record contract is malformed".format(label))
        if (
            record.get("model") != MODEL_FOR_SETTING[setting]
            or record.get("run_tag") != source_batch["batch_id"]
            or record.get("parameters") != {
                "action_selection": "argmax", "action_seed": 0,
            }
            or not _is_sha256(record.get("dataset_sha256"))
            or not isinstance(record.get("metrics_artifact_path"), str)
            or not record["metrics_artifact_path"]
        ):
            raise ArchiveError("{} ledger identity/protocol is invalid".format(label))
        _metrics(record.get("metrics"), label)
        comparisons = {
            "run_tag": "run_tag",
            "parameters": "parameters",
            "metrics": "metrics",
            "checkpoint_sha256": "checkpoint_sha256",
            "formal_manifest_path": "formal_manifest_path",
            "formal_manifest_sha256": "formal_manifest_sha256",
            "immutable_identity_sha256": "formal_immutable_identity_sha256",
            "metrics_artifact_sha256": "metric_artifact_sha256",
        }
        for ledger_key, registry_key in comparisons.items():
            if not _same(record.get(ledger_key), registry_record.get(registry_key)):
                raise ArchiveError(
                    "{} differs from registry: {}".format(label, ledger_key)
                )
    if protocol.get("episode_order_sha256") != ledger["episode_order_sha256"]:
        raise ArchiveError("REVERIE val_seen Source/order protocol mismatch")
    return ledger


def _validate_search_evidence(repo_root, selection, source_binding):
    evidence = selection.get("search_evidence")
    expected_keys = {
        "experiment_id", "git_commit", "spec_sha256",
        "final_selection_sha256", "batch_id", "batch_binding_sha256",
        "search_spec_path", "search_spec_sha256",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_keys:
        raise ArchiveError("REVERIE selected-winner search provenance is malformed")
    if (
        not isinstance(evidence.get("experiment_id"), str)
        or not evidence["experiment_id"]
        or not isinstance(evidence.get("batch_id"), str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", evidence["batch_id"])
        or not isinstance(evidence.get("git_commit"), str)
        or HEX_COMMIT.fullmatch(evidence["git_commit"]) is None
        or any(
            not _is_sha256(evidence.get(key))
            for key in (
                "spec_sha256", "final_selection_sha256",
                "batch_binding_sha256", "search_spec_sha256",
            )
        )
    ):
        raise ArchiveError("REVERIE selected-winner search provenance is invalid")
    search_spec_path = _require_file(
        _resolve(repo_root, evidence["search_spec_path"]),
        "REVERIE val_seen search specification",
    )
    search_spec_sha = _sha256(search_spec_path)
    if (
        search_spec_sha != evidence["search_spec_sha256"]
        or search_spec_sha != evidence["spec_sha256"]
    ):
        raise ArchiveError("REVERIE search-spec evidence digest mismatch")
    search_spec = _read_json(search_spec_path, "REVERIE val_seen search specification")
    dependencies = search_spec.get("dependencies", {})
    source_dependency = dependencies.get("source_registry")
    incumbent_dependency = dependencies.get("incumbent_spec")
    if (
        search_spec.get("schema") != SEARCH_SPEC_SCHEMA
        or search_spec.get("experiment_id") != evidence["experiment_id"]
        or source_dependency != source_binding
        or not isinstance(incumbent_dependency, dict)
        or not _is_sha256(incumbent_dependency.get("sha256"))
    ):
        raise ArchiveError("REVERIE search-spec provenance mismatch")
    incumbent_path = _require_file(
        _resolve(repo_root, incumbent_dependency.get("path", "")),
        "REVERIE incumbent transfer specification",
    )
    if _sha256(incumbent_path) != incumbent_dependency["sha256"]:
        raise ArchiveError("REVERIE incumbent-spec evidence digest mismatch")
    evidence_root = (
        Path(repo_root) / "vln/results/logs/reverie/hparam_search"
        / evidence["batch_id"]
    )
    batch_path = _require_file(evidence_root / "BATCH.json", "REVERIE search batch")
    raw_path = _require_file(
        evidence_root / "FINAL_SELECTION.json", "REVERIE raw search selection"
    )
    if (
        _sha256(batch_path) != evidence["batch_binding_sha256"]
        or _sha256(raw_path) != evidence["final_selection_sha256"]
    ):
        raise ArchiveError("REVERIE raw search evidence digest mismatch")
    batch = _read_json(batch_path, "REVERIE search batch")
    expected_batch = {
        "schema": SEARCH_BATCH_SCHEMA,
        "batch_id": evidence["batch_id"],
        "experiment_id": evidence["experiment_id"],
        "spec_sha256": search_spec_sha,
        "git_commit": evidence["git_commit"],
        "source_registry_sha256": source_binding["sha256"],
        "incumbent_spec_sha256": incumbent_dependency["sha256"],
    }
    for key, value in expected_batch.items():
        if batch.get(key) != value:
            raise ArchiveError("REVERIE search batch {} mismatch".format(key))
    raw = _read_json(raw_path, "REVERIE raw search selection")
    expected_raw = {
        "schema": SEARCH_SELECTION_SCHEMA,
        "experiment_id": evidence["experiment_id"],
        "split": "val_seen",
        "primary_metric": "RGSPL",
        "spec_sha256": search_spec_sha,
        "git_commit": evidence["git_commit"],
    }
    for key, value in expected_raw.items():
        if raw.get(key) != value:
            raise ArchiveError("REVERIE raw search selection {} mismatch".format(key))
    records = raw.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise ArchiveError("REVERIE raw search selection matrix is incomplete")
    for setting in SETTINGS:
        if not isinstance(records[setting], dict) or set(records[setting]) != set(METHODS):
            raise ArchiveError("{} raw search method matrix is incomplete".format(setting))
        for method in METHODS:
            raw_record = records[setting][method]
            selected = selection["records"][setting][method]
            if not isinstance(raw_record, dict):
                raise ArchiveError(
                    "{} {} raw search record is malformed".format(setting, method)
                )
            for key in (
                "origin", "parameters", "run_tag", "metrics", "source_metrics",
                "delta_vs_source_pp", "formal_manifest_sha256",
            ):
                if not _same(raw_record.get(key), selected.get(key)):
                    raise ArchiveError(
                        "{} {} differs from raw search evidence: {}".format(
                            setting, method, key
                        )
                    )


def _load_registry(repo_root, registry_path):
    registry_path = _require_file(registry_path, "REVERIE val_seen registry")
    registry = _read_json(registry_path, "REVERIE val_seen registry")
    expected = {
        "schema": REGISTRY_SCHEMA,
        "registry_status": "complete",
        "protocol_status": "complete_val_seen_seed0_selection",
        "provenance_status": "complete_formal_manifest_identity_verified",
        "publication_status": "selection_registry_not_test_result",
        "benchmark": "reverie",
        "split": "val_seen",
    }
    for key, value in expected.items():
        if registry.get(key) != value:
            raise ArchiveError("val_seen registry {} mismatch".format(key))
    if set(registry) != set(expected) | {
        "protocol", "source_ledger", "selected_winners",
        "supervision_categories", "records",
    }:
        raise ArchiveError("val_seen registry contract is malformed")
    if registry.get("supervision_categories") != {
        "source_no_adaptation": {
            "methods": ["source"], "uses_episode_feedback": False,
        },
        "unsupervised_tta": {
            "methods": list(TEST_METHODS), "uses_episode_feedback": False,
        },
        "binary_episode_feedback_tta": {
            "methods": list(UNAVAILABLE_TEST_METHODS),
            "uses_episode_feedback": True,
            "feedback": "binary_navigation_success",
        },
    }:
        raise ArchiveError("val_seen registry supervision taxonomy is invalid")
    protocol = registry.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("benchmark") != "reverie"
        or protocol.get("split") != "val_seen"
        or protocol.get("episode_count") != VAL_SEEN_EPISODES
        or protocol.get("order_seed") != 0
        or protocol.get("episode_order_sha256") != VAL_SEEN_ORDER
    ):
        raise ArchiveError("val_seen registry protocol is invalid")
    selected_binding = registry.get("selected_winners", {})
    selected_path = _require_file(
        _resolve(repo_root, selected_binding.get("path", "")),
        "REVERIE selected winners",
    )
    if (
        not _is_sha256(selected_binding.get("sha256"))
        or _sha256(selected_path) != selected_binding["sha256"]
    ):
        raise ArchiveError("REVERIE selected-winner binding mismatch")
    selection = _read_json(selected_path, "REVERIE selected winners")
    if (
        selection.get("schema") != "navtta.vln_reverie_final_selection.v1"
        or selection.get("selection_status") != "complete"
        or selection.get("protocol") != protocol
        or not isinstance(selection.get("records"), dict)
        or set(selection["records"]) != set(SETTINGS)
    ):
        raise ArchiveError("REVERIE selected-winner document is invalid")
    if selection.get("source_ledger") != registry.get("source_ledger"):
        raise ArchiveError("REVERIE selected-winner Source binding mismatch")
    source_binding = registry.get("source_ledger", {})
    source_path = _require_file(
        _resolve(repo_root, source_binding.get("path", "")),
        "REVERIE val_seen Source ledger",
    )
    if (
        not _is_sha256(source_binding.get("sha256"))
        or _sha256(source_path) != source_binding["sha256"]
    ):
        raise ArchiveError("REVERIE val_seen Source-ledger binding mismatch")
    records = registry.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise ArchiveError("val_seen registry setting matrix is incomplete")
    source_ledger = _load_val_seen_source_ledger(
        source_path, registry, protocol
    )
    _validate_search_evidence(repo_root, selection, source_binding)
    source_batch = source_ledger["source_batch"]
    archived = {}
    for setting in SETTINGS:
        methods = records[setting]
        if not isinstance(methods, dict) or set(methods) != set(ALL_METHODS):
            raise ArchiveError("{} val_seen method matrix is incomplete".format(setting))
        selected_methods = selection["records"].get(setting)
        if not isinstance(selected_methods, dict) or set(selected_methods) != set(METHODS):
            raise ArchiveError("{} selected-winner method matrix is incomplete".format(setting))
        source_metrics = _metrics(methods["source"].get("metrics"), setting + " source")
        source_checkpoint = methods["source"].get("checkpoint_sha256")
        if not _is_sha256(source_checkpoint):
            raise ArchiveError("{} Source checkpoint digest is invalid".format(setting))
        archived[setting] = {}
        for method in ALL_METHODS:
            label = "{} {}".format(setting, method)
            record = methods[method]
            if not isinstance(record, dict):
                raise ArchiveError("{} val_seen record is malformed".format(label))
            if (
                record.get("method") != method
                or record.get("supervision_category") != SUPERVISION_FOR_METHOD[method]
            ):
                raise ArchiveError("{} val_seen method/supervision mismatch".format(label))
            parameters = record.get("parameters")
            _validate_parameters(method, parameters, label)
            metrics = _metrics(record.get("metrics"), label)
            if record.get("delta_vs_source_pp") != _delta(metrics, source_metrics):
                raise ArchiveError("{} val_seen Source delta mismatch".format(label))
            if record.get("checkpoint_sha256") != source_checkpoint:
                raise ArchiveError("{} checkpoint differs within model".format(label))
            if method != "source":
                selected = selected_methods[method]
                for key in (
                    "supervision_category", "parameters", "run_tag", "metrics",
                    "delta_vs_source_pp", "checkpoint_sha256",
                    "formal_manifest_path", "formal_manifest_sha256",
                    "formal_immutable_identity_sha256", "metric_artifact_sha256",
                ):
                    if not _same(selected.get(key), record.get(key)):
                        raise ArchiveError(
                            "{} differs from selected-winner evidence: {}".format(label, key)
                        )
                if selected.get("selection_status") != "ready":
                    raise ArchiveError("{} selected winner is not ready".format(label))
            _, artifacts, provenance = _validate_manifest(
                repo_root,
                record.get("formal_manifest_path"),
                record.get("formal_manifest_sha256"),
                setting,
                method,
                "val_seen",
                record.get("run_tag"),
                source_checkpoint,
                protocol["episode_order_sha256"],
                dataset_sha256=source_ledger["records"][setting]["dataset_sha256"],
                immutable_sha256=record.get("formal_immutable_identity_sha256"),
                git_commit=(source_batch["git_commit"] if method == "source" else None),
                parameters=parameters,
            )
            metric_path, _ = _artifact_for_digest(
                repo_root, artifacts, record.get("metric_artifact_sha256"),
                label + " val_seen metric", "valid.txt",
            )
            parsed = _parse_metrics(metric_path, "val_seen", label)
            _assert_metrics_equal(parsed, metrics, label)
            if method == "source" and not _declared_path_matches(
                repo_root,
                source_ledger["records"][setting]["metrics_artifact_path"],
                metric_path,
            ):
                raise ArchiveError("{} Source metric path binding mismatch".format(label))
            archived[setting][method] = {
                "parameters": parameters,
                "metrics": metrics,
                "delta_vs_source_pp": record["delta_vs_source_pp"],
                "provenance": provenance,
            }
    return registry_path, registry, archived


def _load_spec(repo_root, spec_path, registry_path, registry):
    spec_path = _require_file(spec_path, "REVERIE frozen-evaluation specification")
    spec = _read_json(spec_path, "REVERIE frozen-evaluation specification")
    if spec.get("schema") != SPEC_SCHEMA:
        raise ArchiveError("unsupported REVERIE frozen-evaluation specification")
    if spec.get("status") != "materialized_from_complete_val_seen_registry":
        raise ArchiveError("REVERIE frozen-evaluation specification is not ready")
    dependency = spec.get("registry_dependency")
    if (
        not isinstance(dependency, dict)
        or dependency.get("schema") != REGISTRY_SCHEMA
        or dependency.get("required_status") != "complete"
        or _resolve(repo_root, dependency.get("path", "")) != registry_path
        or dependency.get("sha256") != _sha256(registry_path)
    ):
        raise ArchiveError("frozen specification registry binding mismatch")
    protocol = spec.get("protocol")
    expected_protocol = {
        "benchmark": "reverie",
        "split": "val_unseen",
        "episode_count": VAL_UNSEEN_EPISODES,
        "canonical_order_seed": 0,
        "order_seed_cli_forbidden": True,
        "full_split_only": True,
        "frozen_from": "reverie_val_seen_seed0_final_registry",
        "selection_on_val_unseen": False,
        "reported_metrics": list(METRICS),
    }
    if not isinstance(protocol, dict):
        raise ArchiveError("val_unseen protocol is missing")
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise ArchiveError("val_unseen protocol {} mismatch".format(key))
    order_sha = protocol.get("episode_order_sha256")
    if order_sha != VAL_UNSEEN_ORDER:
        raise ArchiveError("val_unseen order digest is noncanonical")
    if protocol.get("binary_feedback") != {
        "label": "navigation_success_only",
        "grounding_feedback_forbidden": True,
        "simulator_distance_fallback_forbidden": True,
        "feedtta_timing": "eager_once_per_episode",
        "atena_timing": "lazy_only_when_entropy_gate_queries",
    }:
        raise ArchiveError("val_unseen binary-feedback policy mismatch")
    source_binding = spec.get("source_control", {})
    if (
        source_binding.get("execution") != "reuse_only"
        or source_binding.get("rerun_forbidden") is not True
        or not isinstance(source_binding.get("manifest"), str)
        or not _is_sha256(source_binding.get("sha256"))
    ):
        raise ArchiveError("val_unseen Source binding/reuse policy mismatch")
    order_bindings = protocol.get("order_manifests", {})
    if set(order_bindings) != {"duet_hamt", "goat"}:
        raise ArchiveError("val_unseen order bindings are incomplete")
    val_orders = {
        "duet_hamt": _validate_order_binding(
            repo_root, order_bindings["duet_hamt"], SETTINGS[:2],
            BENCHMARK_FOR_SETTING[SETTINGS[0]], "val_unseen",
            VAL_UNSEEN_EPISODES, order_sha, "DUET/HAMT val_unseen order manifest",
        )[1],
        "goat": _validate_order_binding(
            repo_root, order_bindings["goat"], SETTINGS[2:],
            BENCHMARK_FOR_SETTING[SETTINGS[2]], "val_unseen",
            VAL_UNSEEN_EPISODES, order_sha, "GOAT val_unseen order manifest",
        )[1],
    }
    matrix = spec.get("matrix", {})
    if (
        tuple(matrix.get("model_order", ())) != MODELS
        or tuple(matrix.get("setting_order", ())) != SETTINGS
        or tuple(matrix.get("method_order", ())) != METHODS
        or matrix.get("strict_model_barrier") is not True
        or matrix.get("parallel_methods_within_model") is not True
    ):
        raise ArchiveError("val_unseen frozen matrix identity mismatch")
    jobs = matrix.get("jobs")
    expected_pairs = [(setting, method) for setting in SETTINGS for method in METHODS]
    if (
        not isinstance(jobs, list)
        or len(jobs) != 15
        or [(item.get("setting"), item.get("method")) for item in jobs]
        != expected_pairs
    ):
        raise ArchiveError("val_unseen specification is not the model-major 3x5 matrix")
    for job in jobs:
        selected = registry["records"][job["setting"]][job["method"]]
        if (
            job.get("model") != MODEL_FOR_SETTING[job["setting"]]
            or job.get("selected_run_tag") != selected["run_tag"]
            or job.get("selected_formal_manifest_sha256")
            != selected["formal_manifest_sha256"]
        ):
            raise ArchiveError("val_unseen winner binding mismatch")
    expected_execution = {
        "model_phase_order": "duet_then_hamt_then_goat",
        "max_workers_by_model": {"duet": 2, "hamt": 2, "goat": 4},
        "estimated_gpu_memory_mib_by_method": {
            "tent": 4096, "fstta": 5120, "eam": 6144,
            "feedtta": 8192, "atena": 10240,
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
    execution = spec.get("execution", {})
    for key, value in expected_execution.items():
        if execution.get(key) != value:
            raise ArchiveError("reviewed execution {} changed".format(key))
    if spec.get("budget") != {
        "source_execution_jobs": 0,
        "tta_jobs": 15,
        "total_executed_jobs": 15,
    }:
        raise ArchiveError("val_unseen budget must be 15 TTA + 0 Source")
    policy = spec.get("test_submission_policy")
    if not isinstance(policy, dict):
        raise ArchiveError("hidden-test policy is missing")
    expected_test = {
        "benchmark": "reverie",
        "native_split": "test",
        "episode_count": TEST_EPISODES,
        "canonical_order_seed": 0,
        "hidden_ground_truth": True,
        "execution_purpose": "submission_generation_only",
        "local_metric_computation_forbidden": True,
        "selection_or_ranking_on_test_forbidden": True,
        "eligible_tta_methods": list(TEST_METHODS),
        "eligible_tta_submission_jobs": 9,
        "legal_online_feedback_interface": None,
    }
    for key, value in expected_test.items():
        if policy.get(key) != value:
            raise ArchiveError("hidden-test policy {} mismatch".format(key))
    test_order_sha = policy.get("episode_order_sha256")
    if test_order_sha != TEST_ORDER:
        raise ArchiveError("hidden-test order digest is noncanonical")
    unavailable = policy.get("unavailable_without_legal_online_feedback")
    if not isinstance(unavailable, dict) or set(unavailable) != set(UNAVAILABLE_TEST_METHODS):
        raise ArchiveError("hidden-test unavailable-method policy is incomplete")
    for method in UNAVAILABLE_TEST_METHODS:
        item = unavailable[method]
        if (
            item.get("status") != "N/A"
            or item.get("launch_policy") != "fail_closed"
            or not isinstance(item.get("reason"), str)
            or not item["reason"]
        ):
            raise ArchiveError("{} hidden-test N/A policy is invalid".format(method))
    test_source_binding = policy.get("source", {})
    if (
        test_source_binding.get("execution")
        != "reuse_existing_submission_only"
        or test_source_binding.get("rerun_forbidden") is not True
        or not isinstance(test_source_binding.get("ledger"), str)
        or not _is_sha256(test_source_binding.get("sha256"))
    ):
        raise ArchiveError("hidden-test Source binding/reuse policy mismatch")
    test_bindings = policy.get("order_manifests", {})
    if set(test_bindings) != {"duet_hamt", "goat"}:
        raise ArchiveError("hidden-test order bindings are incomplete")
    test_orders = {
        "duet_hamt": _validate_order_binding(
            repo_root, test_bindings["duet_hamt"], SETTINGS[:2],
            BENCHMARK_FOR_SETTING[SETTINGS[0]], "test", TEST_EPISODES,
            test_order_sha, "DUET/HAMT test order manifest",
        )[1],
        "goat": _validate_order_binding(
            repo_root, test_bindings["goat"], SETTINGS[2:],
            BENCHMARK_FOR_SETTING[SETTINGS[2]], "test", TEST_EPISODES,
            test_order_sha, "GOAT test order manifest",
        )[1],
    }
    return spec_path, spec, val_orders, test_orders


def _order_for_setting(orders, setting):
    return orders["goat" if setting == "goat-reverie" else "duet_hamt"]


def _order_binding_for_setting(spec, setting, split):
    key = "goat" if setting == "goat-reverie" else "duet_hamt"
    if split == "val_unseen":
        binding = spec["protocol"]["order_manifests"][key]
    elif split == "test":
        binding = spec["test_submission_policy"]["order_manifests"][key]
    else:
        raise ArchiveError("unsupported downstream split {}".format(split))
    return binding


def _expected_job_identity(batch, registry_record, setting, method, mode):
    methods = METHODS if mode == "val_unseen" else TEST_METHODS
    model_index = SETTINGS.index(setting)
    method_index = methods.index(method)
    infix = "frozen" if mode == "val_unseen" else "submission"
    base_run_tag = "{}-{}-{:02d}-{}-{:02d}-{}-{}".format(
        batch["batch_id"], infix, model_index, MODEL_FOR_SETTING[setting],
        method_index, method, _job_digest(setting, method, registry_record),
    )
    return {
        "model_index": model_index,
        "method_index": method_index,
        "ordinal": model_index * len(methods) + method_index,
        "base_run_tag": base_run_tag,
        "gpu": batch["gpu"],
    }


def _validate_submission(path, order_document, label):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArchiveError("cannot read {} submission: {}".format(label, error))
    expected_ids = [str(item["episode_id"]) for item in order_document.get("episodes", [])]
    if len(expected_ids) != TEST_EPISODES:
        raise ArchiveError("{} order manifest episode list is incomplete".format(label))
    if not isinstance(payload, list) or len(payload) != TEST_EPISODES:
        raise ArchiveError("{} submission episode count mismatch".format(label))
    actual_ids = []
    for index, item in enumerate(payload):
        if (
            not isinstance(item, dict)
            or not {"instr_id", "trajectory", "predObjId"}.issubset(item)
            or not isinstance(item["trajectory"], list)
        ):
            raise ArchiveError("{} submission row {} is malformed".format(label, index))
        actual_ids.append(str(item["instr_id"]))
    if actual_ids != expected_ids or len(set(actual_ids)) != TEST_EPISODES:
        raise ArchiveError("{} submission IDs/order differ from canonical test stream".format(label))


def _load_val_unseen_sources(
    repo_root, source_path, spec, registry, seen_archive, orders
):
    source_path = _require_file(source_path, "REVERIE val_unseen Source ledger")
    ledger = _read_json(source_path, "REVERIE val_unseen Source ledger")
    expected = {
        "schema": VAL_SOURCE_SCHEMA,
        "benchmark": "reverie",
        "split": "val_unseen",
        "source_protocol": "standard_argmax",
        "episode_count": VAL_UNSEEN_EPISODES,
        "canonical_order_seed": 0,
        "episode_order_sha256": spec["protocol"]["episode_order_sha256"],
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise ArchiveError("val_unseen Source ledger {} mismatch".format(key))
    policy = ledger.get("source_execution_policy", {})
    if policy.get("execution") != "reuse_only" or policy.get("rerun_forbidden") is not True:
        raise ArchiveError("val_unseen Source ledger is not reuse-only")
    source_batch = ledger.get("source_batch", {})
    if not isinstance(source_batch.get("git_commit"), str) or HEX_COMMIT.fullmatch(
        source_batch["git_commit"]
    ) is None:
        raise ArchiveError("val_unseen Source batch provenance is invalid")
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise ArchiveError("val_unseen Source ledger matrix is incomplete")
    archived = {}
    for setting in SETTINGS:
        label = setting + " val_unseen Source"
        record = records[setting]
        if record.get("model") != MODEL_FOR_SETTING[setting] or record.get("evidence_status") != "ready":
            raise ArchiveError("{} identity/evidence status mismatch".format(label))
        if not _same(record.get("parameters"), seen_archive[setting]["source"]["parameters"]):
            raise ArchiveError("{} protocol changed across splits".format(label))
        checkpoint = record.get("checkpoint_sha256")
        if checkpoint != registry["records"][setting]["source"]["checkpoint_sha256"]:
            raise ArchiveError("{} checkpoint changed across splits".format(label))
        order = _order_for_setting(orders, setting)
        order_binding = _order_binding_for_setting(spec, setting, "val_unseen")
        if (
            record.get("episode_order_sha256")
            != spec["protocol"]["episode_order_sha256"]
            or order.get("dataset", {}).get("sha256")
            != record.get("dataset_sha256")
        ):
            raise ArchiveError("{} order-manifest dataset mismatch".format(label))
        metrics = _metrics(record.get("metrics"), label)
        _, artifacts, provenance = _validate_manifest(
            repo_root, record.get("formal_manifest_path"),
            record.get("formal_manifest_sha256"), setting, "source",
            "val_unseen", record.get("run_tag"), checkpoint,
            spec["protocol"]["episode_order_sha256"],
            dataset_sha256=record.get("dataset_sha256"),
            immutable_sha256=record.get("immutable_identity_sha256"),
            git_commit=source_batch["git_commit"],
            order_manifest_sha256=order_binding["sha256"],
        )
        metric_path, _ = _artifact_for_digest(
            repo_root, artifacts, record.get("metrics_artifact_sha256"),
            label + " metric", "valid.txt",
        )
        _assert_metrics_equal(
            _parse_metrics(metric_path, "val_unseen", label), metrics, label
        )
        archived[setting] = {
            "metrics": metrics,
            "delta_vs_source_pp": {name: 0.0 for name in METRICS},
            "provenance": provenance,
        }
    return source_path, ledger, archived


def _load_test_sources(repo_root, source_path, spec, registry, orders):
    source_path = _require_file(source_path, "REVERIE test Source-submission ledger")
    ledger = _read_json(source_path, "REVERIE test Source-submission ledger")
    policy = spec["test_submission_policy"]
    expected = {
        "schema": TEST_SOURCE_SCHEMA,
        "benchmark": "reverie",
        "split": "test",
        "episode_count": TEST_EPISODES,
        "canonical_order_seed": 0,
        "episode_order_sha256": policy["episode_order_sha256"],
        "hidden_ground_truth": True,
        "local_metrics_available": False,
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise ArchiveError("test Source ledger {} mismatch".format(key))
    if "metrics" in ledger or ledger.get("source_execution_policy") != {
        "execution": "reuse_existing_submission_only", "rerun_forbidden": True
    }:
        raise ArchiveError("test Source ledger is not reuse-only")
    source_batch = ledger.get("source_batch", {})
    if not isinstance(source_batch.get("git_commit"), str) or HEX_COMMIT.fullmatch(
        source_batch["git_commit"]
    ) is None:
        raise ArchiveError("test Source batch provenance is invalid")
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise ArchiveError("test Source submission matrix is incomplete")
    archived = {}
    for setting in SETTINGS:
        label = setting + " test Source"
        record = records[setting]
        if record.get("model") != MODEL_FOR_SETTING[setting] or "metrics" in record:
            raise ArchiveError("{} identity/hidden-metric contract mismatch".format(label))
        checkpoint = record.get("checkpoint_sha256")
        if checkpoint != registry["records"][setting]["source"]["checkpoint_sha256"]:
            raise ArchiveError("{} checkpoint changed across splits".format(label))
        order = _order_for_setting(orders, setting)
        order_binding = _order_binding_for_setting(spec, setting, "test")
        if (
            record.get("episode_order_sha256")
            != policy["episode_order_sha256"]
            or order.get("dataset", {}).get("sha256")
            != record.get("dataset_sha256")
        ):
            raise ArchiveError("{} order-manifest dataset mismatch".format(label))
        _, artifacts, provenance = _validate_manifest(
            repo_root, record.get("formal_manifest_path"),
            record.get("formal_manifest_sha256"), setting, "source", "test",
            record.get("run_tag"), checkpoint, policy["episode_order_sha256"],
            dataset_sha256=record.get("dataset_sha256"),
            immutable_sha256=record.get("immutable_identity_sha256"),
            git_commit=source_batch["git_commit"],
            order_manifest_sha256=order_binding["sha256"],
        )
        submission_path, item = _artifact_for_digest(
            repo_root, artifacts, record.get("submission_artifact_sha256"),
            label + " submission", ".json",
        )
        if item.get("size") != record.get("submission_artifact_size"):
            raise ArchiveError("{} submission size binding mismatch".format(label))
        declared_submission = _resolve(repo_root, record.get("submission_artifact_path", ""))
        if declared_submission != submission_path:
            normalized = str(record.get("submission_artifact_path", "")).replace("\\", "/")
            if not normalized.endswith("/" + _display_path(repo_root, submission_path)):
                raise ArchiveError("{} submission path binding mismatch".format(label))
        _validate_submission(submission_path, order, label)
        result_root = submission_path.parents[1]
        _validate_test_submission_tree(result_root, submission_path, label)
        _assert_no_hidden_test_metrics(repo_root, artifacts, result_root, label)
        archived[setting] = {
            "status": "reused_submission_ready",
            "metrics": None,
            "submission": {
                "path": _display_path(repo_root, submission_path),
                "sha256": item["sha256"],
                "size": item["size"],
            },
            "provenance": provenance,
        }
    return source_path, ledger, archived


def _states_complete(states, expected, label):
    required = {"completed", "finished", "failed", "invalid", "running", "orphaned", "pending"}
    if not isinstance(states, dict) or set(states) != required:
        raise ArchiveError("{} states are malformed".format(label))
    if states["completed"] != expected or any(
        states[key] != 0 for key in required if key != "completed"
    ):
        raise ArchiveError("{} is not a clean {}/{} completion".format(label, expected, expected))


def _load_batch(
    batch_root, spec_path, spec, registry_path, source_path, mode
):
    if mode not in ("val_unseen", "test_submission"):
        raise ArchiveError("invalid batch mode")
    batch_root = Path(batch_root).resolve()
    batch_path = _require_file(batch_root / "BATCH.json", "downloaded {} batch".format(mode))
    summary_name = "SUMMARY.json" if mode == "val_unseen" else "SUBMISSIONS.json"
    summary_path = _require_file(batch_root / summary_name, "downloaded {} summary".format(mode))
    batch = _read_json(batch_path, "downloaded {} batch".format(mode))
    summary = _read_json(summary_path, "downloaded {} summary".format(mode))
    count = 15 if mode == "val_unseen" else 9
    expected_batch = {
        "schema": BATCH_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "spec_sha256": _sha256(spec_path),
        "mode": mode,
        "source_execution_jobs": 0,
        "tta_jobs": count,
    }
    for key, value in expected_batch.items():
        if batch.get(key) != value:
            raise ArchiveError("downloaded {} batch {} mismatch".format(mode, key))
    expected_batch_keys = set(expected_batch) | {
        "batch_id", "spec_path", "git_commit", "registry", "gpu",
        "source_ledger" if mode == "val_unseen"
        else "test_source_submission_ledger",
    }
    if set(batch) != expected_batch_keys:
        raise ArchiveError("downloaded {} batch contract is malformed".format(mode))
    if not isinstance(batch.get("batch_id"), str) or not re.fullmatch(
        r"[A-Za-z0-9._-]+", batch["batch_id"]
    ):
        raise ArchiveError("downloaded {} batch ID is invalid".format(mode))
    if not _declared_path_matches(Path(spec_path).parents[2], batch["spec_path"], spec_path):
        raise ArchiveError("downloaded {} batch specification path mismatch".format(mode))
    if type(batch.get("gpu")) is not int or batch["gpu"] < 0:
        raise ArchiveError("downloaded {} batch GPU is invalid".format(mode))
    if batch.get("registry") != spec["registry_dependency"]:
        raise ArchiveError("downloaded {} batch registry binding mismatch".format(mode))
    if _sha256(registry_path) != batch["registry"]["sha256"]:
        raise ArchiveError("downloaded {} batch registry digest mismatch".format(mode))
    source_key = "source_ledger" if mode == "val_unseen" else "test_source_submission_ledger"
    expected_source = (
        spec["source_control"] if mode == "val_unseen"
        else spec["test_submission_policy"]["source"]
    )
    if batch.get(source_key) != expected_source or _sha256(source_path) != expected_source["sha256"]:
        raise ArchiveError("downloaded {} batch Source-ledger binding mismatch".format(mode))
    if not isinstance(batch.get("git_commit"), str) or HEX_COMMIT.fullmatch(batch["git_commit"]) is None:
        raise ArchiveError("downloaded {} batch git commit is invalid".format(mode))

    if mode == "val_unseen":
        expected_summary = {
            "schema": VAL_SUMMARY_SCHEMA,
            "experiment_id": spec["experiment_id"],
            "source_execution_jobs": 0,
            "total_jobs": 15,
            "complete": True,
        }
        rows = summary.get("results")
        methods = METHODS
        cache_name = "metrics.json"
        cache_schema = VAL_JOB_SCHEMA
        expected_summary_keys = set(expected_summary) | {"states", "results"}
    else:
        expected_summary = {
            "schema": TEST_SUMMARY_SCHEMA,
            "experiment_id": spec["experiment_id"],
            "split": "test",
            "hidden_ground_truth": True,
            "submission_generation_only": True,
            "local_metrics": None,
            "source_execution_jobs": 0,
            "reused_source_submissions": 3,
            "tta_submission_jobs": 9,
            "complete": True,
        }
        rows = summary.get("submissions")
        methods = TEST_METHODS
        cache_name = "submission.json"
        cache_schema = TEST_JOB_SCHEMA
        expected_summary_keys = set(expected_summary) | {
            "states", "unavailable_methods", "submissions"
        }
        if summary.get("unavailable_methods") != {
            "feedtta": "N/A: no legal online binary feedback",
            "atena": "N/A: no legal online binary feedback",
        }:
            raise ArchiveError("hidden-test unavailable-method summary mismatch")
        if any("metrics" in row for row in rows or []):
            raise ArchiveError("hidden-test summary must not contain local metrics")
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ArchiveError("downloaded {} summary {} mismatch".format(mode, key))
    if set(summary) != expected_summary_keys:
        raise ArchiveError("downloaded {} summary contract is malformed".format(mode))
    _states_complete(summary.get("states"), count, "downloaded {} summary".format(mode))
    expected_pairs = [(setting, method) for setting in SETTINGS for method in methods]
    if (
        not isinstance(rows, list)
        or len(rows) != count
        or [(row.get("setting"), row.get("method")) for row in rows] != expected_pairs
    ):
        raise ArchiveError("downloaded {} summary matrix is incomplete/noncanonical".format(mode))
    summary_row_keys = {
        "setting", "method", "run_tag", "parameters", "metrics",
        "feedback_endpoint", "formal_manifest", "formal_manifest_sha256",
    } if mode == "val_unseen" else {
        "setting", "method", "run_tag", "parameters", "submission_path",
        "submission_sha256", "submission_size", "formal_manifest",
        "formal_manifest_sha256",
    }
    if any(not isinstance(row, dict) or set(row) != summary_row_keys for row in rows):
        raise ArchiveError("downloaded {} summary row contract is malformed".format(mode))
    caches = {}
    for path in sorted(batch_root.rglob(cache_name)):
        if (path.parent / "validation_error.json").is_file():
            continue
        value = _read_json(path, "downloaded {} cache".format(mode))
        if value.get("schema") != cache_schema:
            continue
        tag = value.get("run_tag")
        if not isinstance(tag, str) or not tag or tag in caches:
            raise ArchiveError("duplicate/invalid downloaded {} run tag".format(mode))
        caches[tag] = (path, value)
    if set(caches) != {row.get("run_tag") for row in rows}:
        raise ArchiveError("downloaded {} caches do not exactly cover summary".format(mode))
    return batch_root, batch_path, batch, summary_path, summary, caches


def _validate_config(path, row, spec, registry_record, mode):
    config = _read_json(path, "downloaded frozen job configuration")
    split = "val_unseen" if mode == "val_unseen" else "test"
    expected = {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": row["batch_id"],
        "stage": "frozen_val_unseen" if mode == "val_unseen" else "frozen_test_submission",
        "setting": row["setting"],
        "method": row["method"],
        "search_method": row["method"],
        "episodes": -1,
        "parameters": registry_record["parameters"],
    }
    if set(config) != set(expected) | {"frozen_evaluation_provenance"}:
        raise ArchiveError(
            "{} configuration has unexpected/missing fields".format(row["run_tag"])
        )
    for key, value in expected.items():
        if not _same(config.get(key), value):
            raise ArchiveError("{} configuration {} mismatch".format(row["run_tag"], key))
    if "order_seed" in config:
        raise ArchiveError("{} configuration illegally overrides order_seed".format(row["run_tag"]))
    provenance = config.get("frozen_evaluation_provenance")
    expected_provenance_keys = {
        "selection_benchmark", "selection_split", "evaluation_split",
        "canonical_order_seed", "registry", "selected_anchor",
        "selection_on_val_unseen", "source_ledger",
    } if mode == "val_unseen" else {
        "selection_benchmark", "selection_split", "evaluation_split",
        "canonical_order_seed", "registry", "selected_anchor",
        "selection_on_test", "hidden_ground_truth",
        "submission_generation_only", "source_submission_ledger",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != expected_provenance_keys
        or provenance.get("selection_benchmark") != "reverie"
        or provenance.get("selection_split") != "val_seen"
        or provenance.get("evaluation_split") != split
        or provenance.get("canonical_order_seed") != 0
        or provenance.get("registry") != spec["registry_dependency"]
        or provenance.get("selected_anchor") != registry_record
    ):
        raise ArchiveError("{} frozen-selection provenance mismatch".format(row["run_tag"]))
    if mode == "val_unseen":
        expected_source = {
            "path": spec["source_control"]["manifest"],
            "sha256": spec["source_control"]["sha256"],
        }
        if (
            provenance.get("selection_on_val_unseen") is not False
            or provenance.get("source_ledger") != expected_source
        ):
            raise ArchiveError("{} val_unseen selection policy mismatch".format(row["run_tag"]))
    elif (
        provenance.get("selection_on_test") is not False
        or provenance.get("hidden_ground_truth") is not True
        or provenance.get("submission_generation_only") is not True
        or provenance.get("source_submission_ledger")
        != spec["test_submission_policy"]["source"]
    ):
        raise ArchiveError("{} hidden-test provenance mismatch".format(row["run_tag"]))


def _validate_attempt_files(
    cache_path, row, label, spec, spec_path, mode, repo_root
):
    attempt_dir = Path(cache_path).parent
    if (attempt_dir / "validation_error.json").exists():
        raise ArchiveError("{} attempt carries a validation error".format(label))
    try:
        exit_code = int((attempt_dir / "exitcode").read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise ArchiveError("{} exit status is missing/invalid: {}".format(label, error))
    if exit_code != 0:
        raise ArchiveError("{} worker did not exit successfully".format(label))
    metadata = _read_json(attempt_dir / "job.json", label + " job metadata")
    ledger_prefix = "source" if mode == "val_unseen" else "test_source"
    required = {
        "schema", "batch_id", "spec_path", "spec_sha256", "attempt",
        "run_tag", "base_run_tag", "model_index", "method_index",
        "ordinal", "gpu", "setting", "model", "method", "parameters",
        "selected_anchor", "episode_count", "canonical_order_seed",
        "git_commit", "expected_benchmark", "expected_checkpoint_sha256",
        "expected_dataset_sha256", "expected_episode_order_sha256",
        "episode_order_manifest", "result_root", "formal_manifest",
        "command", "parameters_path", "parameters_sha256",
        ledger_prefix + "_ledger_path", ledger_prefix + "_ledger_sha256",
    }
    if mode == "test_submission":
        required.update({"hidden_ground_truth", "submission_generation_only"})
    optional = {"process_group_token"}
    if set(metadata) not in (required, required | optional):
        raise ArchiveError("{} job metadata contract is incomplete".format(label))
    cache_fields = {
        "metrics", "metric_artifact", "metric_artifact_sha256", "metric_line",
        "diagnostics_path", "diagnostics_sha256", "feedback_endpoint",
        "adapter_diagnostics", "formal_manifest_sha256",
        "formal_immutable_identity_sha256",
    } if mode == "val_unseen" else {
        "submission_path", "submission_sha256", "submission_size",
        "diagnostics_path", "diagnostics_sha256", "formal_manifest_sha256",
        "formal_immutable_identity_sha256", "metrics",
    }
    if set(row) != set(metadata) | cache_fields:
        raise ArchiveError("{} result cache contract is malformed".format(label))
    for key, value in metadata.items():
        if key not in row or not _same(row[key], value):
            raise ArchiveError("{} cache differs from job metadata: {}".format(label, key))
    if "process_group_token" in metadata and metadata["process_group_token"] != (
        _expected_process_group_token(metadata, attempt_dir)
    ):
        raise ArchiveError("{} process-group token mismatch".format(label))
    attempt = metadata.get("attempt")
    if type(attempt) is not int or attempt < 0:
        raise ArchiveError("{} attempt index is invalid".format(label))
    expected_tag = (
        metadata["base_run_tag"] if attempt == 0
        else "{}-retry{}".format(metadata["base_run_tag"], attempt)
    )
    if (
        metadata["run_tag"] != expected_tag
        or attempt_dir.name != "attempt-{:02d}".format(attempt)
        or attempt_dir.parent.name != metadata["base_run_tag"]
    ):
        raise ArchiveError("{} attempt/run-tag binding mismatch".format(label))
    expected_attempt_suffix = Path(
        "models", "{:02d}-{}".format(metadata["model_index"], metadata["model"]),
        "jobs", metadata["base_run_tag"], "attempt-{:02d}".format(attempt),
    )
    if Path(*attempt_dir.parts[-5:]) != expected_attempt_suffix:
        raise ArchiveError("{} attempt directory is noncanonical".format(label))
    if type(metadata.get("gpu")) is not int or metadata["gpu"] < 0:
        raise ArchiveError("{} GPU index is invalid".format(label))
    if not _declared_path_matches(repo_root, metadata["spec_path"], spec_path):
        raise ArchiveError("{} specification path binding mismatch".format(label))
    config_path = attempt_dir / "parameters.json"
    if (
        not _declared_path_matches(repo_root, metadata["parameters_path"], config_path)
        or metadata.get("parameters_sha256") != _sha256(config_path)
    ):
        raise ArchiveError("{} parameters.json path/digest mismatch".format(label))
    split = "val_unseen" if mode == "val_unseen" else "test"
    result_suffix = "vln/results/tuning/{}/{}/{}".format(
        metadata["run_tag"], metadata["setting"], split
    )
    normalized_root = str(metadata.get("result_root", "")).replace("\\", "/").rstrip("/")
    if normalized_root != result_suffix and not normalized_root.endswith("/" + result_suffix):
        raise ArchiveError("{} result root is noncanonical".format(label))
    order_binding = _order_binding_for_setting(spec, metadata["setting"], split)
    expected_order = _resolve(repo_root, order_binding["path"])
    if not _declared_path_matches(
        repo_root, metadata["episode_order_manifest"], expected_order
    ):
        raise ArchiveError("{} episode-order path binding mismatch".format(label))
    if mode == "val_unseen":
        source = spec["source_control"]
        source_path = source["manifest"]
    else:
        source = spec["test_submission_policy"]["source"]
        source_path = source["ledger"]
    if (
        metadata[ledger_prefix + "_ledger_sha256"] != source["sha256"]
        or not _declared_path_matches(
            repo_root, metadata[ledger_prefix + "_ledger_path"],
            _resolve(repo_root, source_path),
        )
    ):
        raise ArchiveError("{} Source-ledger attempt binding mismatch".format(label))
    command = metadata["command"]
    if (
        not isinstance(command, list)
        or len(command) != 8
        or not str(command[0]).replace("\\", "/").endswith(
            "/vln/scripts/run_source_eval.sh"
        )
    ):
        raise ArchiveError("{} worker command mismatch".format(label))
    expected_command = [
        command[0], metadata["setting"], split,
        str(metadata["gpu"]), "--run-tag", metadata["run_tag"],
        "--tta-config", metadata["parameters_path"],
    ]
    if command != expected_command:
        raise ArchiveError("{} worker command mismatch".format(label))
    return attempt_dir


def _validate_diagnostics(path, row, metrics, hidden_test=False):
    diagnostics = _read_json(path, "{} diagnostics".format(row["run_tag"]))
    episodes = TEST_EPISODES if hidden_test else VAL_UNSEEN_EPISODES
    if (
        diagnostics.get("method") != row["method"]
        or diagnostics.get("episode_count") != episodes
        or diagnostics.get("action_selection") != "target_native_argmax"
    ):
        raise ArchiveError("{} diagnostics identity mismatch".format(row["run_tag"]))
    adapter = diagnostics.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("episodes") != episodes:
        raise ArchiveError("{} adapter episode accounting mismatch".format(row["run_tag"]))
    for key in ("updates", "relative_param_drift"):
        value = adapter.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ArchiveError("{} adapter.{} is invalid".format(row["run_tag"], key))
    method = row["method"]
    if hidden_test:
        if (
            method not in TEST_METHODS
            or diagnostics.get("supervision") != "unsupervised"
            or diagnostics.get("binary_feedback_endpoint") is not None
        ):
            raise ArchiveError("{} hidden-test diagnostics consumed feedback".format(row["run_tag"]))
        return adapter
    endpoint = FEEDBACK_ENDPOINT.get(method)
    if method not in UNAVAILABLE_TEST_METHODS:
        if (
            diagnostics.get("supervision") != "unsupervised"
            or diagnostics.get("binary_feedback_endpoint") is not None
        ):
            raise ArchiveError("{} unsupervised diagnostics consumed feedback".format(row["run_tag"]))
    else:
        if (
            diagnostics.get("supervision") != "binary_navigation_success_feedback"
            or diagnostics.get("binary_feedback_endpoint") != endpoint
            or "distance" in str(diagnostics.get("binary_feedback_endpoint"))
        ):
            raise ArchiveError("{} binary-feedback endpoint mismatch".format(row["run_tag"]))
        if method == "feedtta":
            feedback = adapter.get("feedback_episodes")
            successes = adapter.get("successful_feedback_episodes")
            failures = adapter.get("failed_feedback_episodes")
            if (
                type(feedback) is not int or type(successes) is not int
                or type(failures) is not int or feedback != episodes
                or successes + failures != episodes
                or successes != int(round(float(metrics["SR"]) * episodes / 100.0))
                or adapter.get("action_selection_protocol") != "target_native_argmax"
            ):
                raise ArchiveError("{} FeedTTA feedback accounting mismatch".format(row["run_tag"]))
        else:
            keys = (
                "queries", "self_label_episodes", "feedback_observed_episodes",
                "query_gate_evaluations", "self_prediction_evaluations",
            )
            if any(type(adapter.get(key)) is not int for key in keys):
                raise ArchiveError("{} ATENA feedback accounting is invalid".format(row["run_tag"]))
            if any(not 0 <= adapter[key] <= episodes for key in keys):
                raise ArchiveError("{} ATENA feedback count is out of range".format(row["run_tag"]))
            if (
                adapter["queries"] + adapter["self_label_episodes"] != episodes
                or adapter["feedback_observed_episodes"] != adapter["queries"]
                or adapter["query_gate_evaluations"] != episodes
                or adapter["self_prediction_evaluations"] != episodes
            ):
                raise ArchiveError("{} ATENA feedback accounting mismatch".format(row["run_tag"]))
            try:
                zero_threshold = float(
                    row["parameters"].get("query_threshold", math.nan)
                ) == 0.0
            except (TypeError, ValueError):
                raise ArchiveError(
                    "{} ATENA query threshold is invalid".format(row["run_tag"])
                )
            if zero_threshold and adapter["queries"] != episodes:
                raise ArchiveError(
                    "{} zero-threshold ATENA must query every episode".format(
                        row["run_tag"]
                    )
                )
    return adapter


def _archive_val_unseen_tta(
    repo_root, spec_path, spec, registry, sources, batch, summary, caches
):
    archived = {setting: {} for setting in SETTINGS}
    for summary_row in summary["results"]:
        setting, method = summary_row["setting"], summary_row["method"]
        label = "{} {} val_unseen".format(setting, method)
        registry_record = registry["records"][setting][method]
        if not _same(summary_row.get("parameters"), registry_record["parameters"]):
            raise ArchiveError("{} parameters drifted from val_seen".format(label))
        metrics = _metrics(summary_row.get("metrics"), label)
        cache_path, row = caches[summary_row["run_tag"]]
        expected = {
            "schema": VAL_JOB_SCHEMA,
            "batch_id": batch["batch_id"],
            "spec_sha256": batch["spec_sha256"],
            "setting": setting,
            "model": MODEL_FOR_SETTING[setting],
            "method": method,
            "parameters": registry_record["parameters"],
            "selected_anchor": registry_record,
            "episode_count": VAL_UNSEEN_EPISODES,
            "canonical_order_seed": 0,
            "git_commit": batch["git_commit"],
            "expected_benchmark": BENCHMARK_FOR_SETTING[setting],
            "expected_checkpoint_sha256": registry["records"][setting]["source"]["checkpoint_sha256"],
            "expected_dataset_sha256": sources[setting]["provenance"]["dataset_sha256"],
            "expected_episode_order_sha256": spec["protocol"]["episode_order_sha256"],
            "run_tag": summary_row["run_tag"],
            "metrics": metrics,
            "formal_manifest_sha256": summary_row.get("formal_manifest_sha256"),
            **_expected_job_identity(
                batch, registry_record, setting, method, "val_unseen"
            ),
        }
        for key, value in expected.items():
            if not _same(row.get(key), value):
                raise ArchiveError("{} downloaded cache {} mismatch".format(label, key))
        if row.get("formal_manifest") != summary_row.get("formal_manifest"):
            raise ArchiveError("{} formal-manifest path differs between outputs".format(label))
        attempt_dir = _validate_attempt_files(
            cache_path, row, label, spec, spec_path, "val_unseen", repo_root
        )
        _validate_config(attempt_dir / "parameters.json", row, spec, registry_record, "val_unseen")
        order_binding = _order_binding_for_setting(spec, setting, "val_unseen")
        _, artifacts, provenance = _validate_manifest(
            repo_root, row.get("formal_manifest"), row.get("formal_manifest_sha256"),
            setting, method, "val_unseen", row["run_tag"],
            row["expected_checkpoint_sha256"], row["expected_episode_order_sha256"],
            dataset_sha256=row["expected_dataset_sha256"],
            immutable_sha256=row.get("formal_immutable_identity_sha256"),
            git_commit=row.get("git_commit"),
            parameters=registry_record["parameters"],
            result_root=row.get("result_root"),
            order_manifest=_resolve(repo_root, order_binding["path"]),
            order_manifest_sha256=order_binding["sha256"],
        )
        metric_path, _ = _artifact_for_digest(
            repo_root, artifacts, row.get("metric_artifact_sha256"),
            label + " metric", "valid.txt",
        )
        if not _declared_path_matches(repo_root, row.get("metric_artifact"), metric_path):
            raise ArchiveError("{} metric artifact path binding mismatch".format(label))
        parsed_metrics = _parse_metrics(metric_path, "val_unseen", label)
        _assert_metrics_equal(parsed_metrics, metrics, label)
        local_result_root = _result_root_path(repo_root, row["result_root"], label)
        metric_line = _validate_val_metric_tree(
            local_result_root, metric_path, metrics, label
        )
        if row.get("metric_line") != metric_line:
            raise ArchiveError("{} cached metric line mismatch".format(label))
        diagnostics_path, _ = _artifact_for_digest(
            repo_root, artifacts, row.get("diagnostics_sha256"),
            label + " diagnostics", "tta_diagnostics.json",
        )
        if not _declared_path_matches(repo_root, row.get("diagnostics_path"), diagnostics_path):
            raise ArchiveError("{} diagnostics path binding mismatch".format(label))
        if diagnostics_path.resolve() != (
            local_result_root / "tta_diagnostics.json"
        ):
            raise ArchiveError("{} diagnostics path is noncanonical".format(label))
        adapter = _validate_diagnostics(diagnostics_path, row, metrics)
        if not _same(row.get("adapter_diagnostics"), adapter):
            raise ArchiveError("{} cached adapter diagnostics mismatch".format(label))
        if row.get("feedback_endpoint") != FEEDBACK_ENDPOINT.get(method):
            raise ArchiveError("{} supervision endpoint mismatch".format(label))
        if summary_row.get("feedback_endpoint") != row.get("feedback_endpoint"):
            raise ArchiveError("{} summary feedback endpoint mismatch".format(label))
        archived[setting][method] = {
            "metrics": metrics,
            "delta_vs_source_pp": _delta(metrics, sources[setting]["metrics"]),
            "provenance": provenance,
        }
    return archived


def _archive_test_tta(
    repo_root, spec_path, spec, registry, test_sources, test_orders,
    batch, summary, caches
):
    archived = {setting: {} for setting in SETTINGS}
    for summary_row in summary["submissions"]:
        setting, method = summary_row["setting"], summary_row["method"]
        label = "{} {} test".format(setting, method)
        registry_record = registry["records"][setting][method]
        if method not in TEST_METHODS:
            raise ArchiveError("{} feedback method cannot appear in hidden-test batch".format(label))
        if not _same(summary_row.get("parameters"), registry_record["parameters"]):
            raise ArchiveError("{} parameters drifted from val_seen".format(label))
        cache_path, row = caches[summary_row["run_tag"]]
        expected = {
            "schema": TEST_JOB_SCHEMA,
            "batch_id": batch["batch_id"],
            "spec_sha256": batch["spec_sha256"],
            "setting": setting,
            "model": MODEL_FOR_SETTING[setting],
            "method": method,
            "parameters": registry_record["parameters"],
            "selected_anchor": registry_record,
            "episode_count": TEST_EPISODES,
            "canonical_order_seed": 0,
            "hidden_ground_truth": True,
            "submission_generation_only": True,
            "git_commit": batch["git_commit"],
            "expected_benchmark": BENCHMARK_FOR_SETTING[setting],
            "expected_checkpoint_sha256": registry["records"][setting]["source"]["checkpoint_sha256"],
            "expected_dataset_sha256": _order_for_setting(test_orders, setting)["dataset"]["sha256"],
            "expected_episode_order_sha256": spec["test_submission_policy"]["episode_order_sha256"],
            "run_tag": summary_row["run_tag"],
            "submission_sha256": summary_row.get("submission_sha256"),
            "submission_size": summary_row.get("submission_size"),
            "formal_manifest_sha256": summary_row.get("formal_manifest_sha256"),
            "metrics": None,
            **_expected_job_identity(
                batch, registry_record, setting, method, "test_submission"
            ),
        }
        for key, value in expected.items():
            if not _same(row.get(key), value):
                raise ArchiveError("{} downloaded cache {} mismatch".format(label, key))
        for key in ("formal_manifest", "submission_path"):
            if row.get(key) != summary_row.get(key):
                raise ArchiveError("{} {} differs between outputs".format(label, key))
        attempt_dir = _validate_attempt_files(
            cache_path, row, label, spec, spec_path,
            "test_submission", repo_root
        )
        _validate_config(attempt_dir / "parameters.json", row, spec, registry_record, "test_submission")
        order_binding = _order_binding_for_setting(spec, setting, "test")
        _, artifacts, provenance = _validate_manifest(
            repo_root, row.get("formal_manifest"), row.get("formal_manifest_sha256"),
            setting, method, "test", row["run_tag"],
            row["expected_checkpoint_sha256"], row["expected_episode_order_sha256"],
            dataset_sha256=row["expected_dataset_sha256"],
            immutable_sha256=row.get("formal_immutable_identity_sha256"),
            git_commit=row.get("git_commit"),
            parameters=registry_record["parameters"],
            result_root=row.get("result_root"),
            order_manifest=_resolve(repo_root, order_binding["path"]),
            order_manifest_sha256=order_binding["sha256"],
        )
        submission_path, item = _artifact_for_digest(
            repo_root, artifacts, row.get("submission_sha256"),
            label + " submission", ".json",
        )
        if not _declared_path_matches(repo_root, row.get("submission_path"), submission_path):
            raise ArchiveError("{} submission path binding mismatch".format(label))
        if item.get("size") != row.get("submission_size"):
            raise ArchiveError("{} submission size mismatch".format(label))
        local_result_root = _result_root_path(repo_root, row["result_root"], label)
        _validate_test_submission_tree(local_result_root, submission_path, label)
        _validate_submission(
            submission_path, _order_for_setting(test_orders, setting), label
        )
        diagnostics_path, _ = _artifact_for_digest(
            repo_root, artifacts, row.get("diagnostics_sha256"),
            label + " diagnostics", "tta_diagnostics.json",
        )
        if not _declared_path_matches(repo_root, row.get("diagnostics_path"), diagnostics_path):
            raise ArchiveError("{} diagnostics path binding mismatch".format(label))
        if diagnostics_path.resolve() != (
            local_result_root / "tta_diagnostics.json"
        ):
            raise ArchiveError("{} diagnostics path is noncanonical".format(label))
        _assert_no_hidden_test_metrics(
            repo_root, artifacts, local_result_root, label
        )
        _validate_diagnostics(diagnostics_path, row, None, hidden_test=True)
        archived[setting][method] = {
            "status": "frozen_submission_ready",
            "metrics": None,
            "submission": {
                "path": _display_path(repo_root, submission_path),
                "sha256": item["sha256"],
                "size": item["size"],
            },
            "provenance": provenance,
        }
    return archived


def build_results(
    val_unseen_batch_root,
    test_batch_root,
    repo_root=REPO_ROOT,
    registry_path=None,
    spec_path=None,
):
    """Return a deterministic final-results document without writing inputs."""
    repo_root = Path(repo_root).resolve()
    registry_path = _resolve(
        repo_root, registry_path or "vln/results/final/reverie/registry.json"
    )
    spec_path = _resolve(
        repo_root, spec_path or "vln/experiments/reverie_val_unseen_frozen_eval_v1.json"
    )
    registry_path, registry, seen = _load_registry(repo_root, registry_path)
    spec_path, spec, val_orders, test_orders = _load_spec(
        repo_root, spec_path, registry_path, registry
    )
    val_binding = spec.get("source_control", {})
    val_source_path = _resolve(repo_root, val_binding.get("manifest", ""))
    if val_binding.get("sha256") != _sha256(_require_file(val_source_path, "val_unseen Source ledger")):
        raise ArchiveError("val_unseen specification Source-ledger digest mismatch")
    val_source_path, val_source_ledger, val_sources = _load_val_unseen_sources(
        repo_root, val_source_path, spec, registry, seen, val_orders
    )
    test_binding = spec["test_submission_policy"].get("source", {})
    test_source_path = _resolve(repo_root, test_binding.get("ledger", ""))
    if test_binding.get("sha256") != _sha256(_require_file(test_source_path, "test Source ledger")):
        raise ArchiveError("hidden-test specification Source-ledger digest mismatch")
    test_source_path, test_source_ledger, test_sources = _load_test_sources(
        repo_root, test_source_path, spec, registry, test_orders
    )
    for setting in SETTINGS:
        if val_sources[setting]["provenance"]["checkpoint_sha256"] != test_sources[setting]["provenance"]["checkpoint_sha256"]:
            raise ArchiveError("{} checkpoint differs across splits".format(setting))

    (
        _, val_batch_path, val_batch, val_summary_path, val_summary, val_caches
    ) = _load_batch(
        val_unseen_batch_root, spec_path, spec, registry_path,
        val_source_path, "val_unseen",
    )
    val_tta = _archive_val_unseen_tta(
        repo_root, spec_path, spec, registry, val_sources,
        val_batch, val_summary, val_caches
    )
    (
        _, test_batch_path, test_batch, test_summary_path, test_summary,
        test_caches,
    ) = _load_batch(
        test_batch_root, spec_path, spec, registry_path,
        test_source_path, "test_submission",
    )
    test_tta = _archive_test_tta(
        repo_root, spec_path, spec, registry, test_sources, test_orders,
        test_batch, test_summary, test_caches,
    )

    unavailable = spec["test_submission_policy"][
        "unavailable_without_legal_online_feedback"
    ]
    records = {}
    for setting in SETTINGS:
        cells = {}
        for method in ALL_METHODS:
            unseen = val_sources[setting] if method == "source" else val_tta[setting][method]
            if method == "source":
                test = test_sources[setting]
            elif method in TEST_METHODS:
                test = test_tta[setting][method]
            else:
                test = {
                    "status": "N/A",
                    "metrics": None,
                    "submission": None,
                    "reason": unavailable[method]["reason"],
                    "legal_online_feedback_interface": None,
                }
            cells[method] = {
                "method": method,
                "supervision_category": SUPERVISION_FOR_METHOD[method],
                "frozen_parameters": seen[setting][method]["parameters"],
                "val_seen": {
                    key: seen[setting][method][key]
                    for key in ("metrics", "delta_vs_source_pp", "provenance")
                },
                "val_unseen": unseen,
                "test": test,
            }
        records[setting] = {
            "model": MODEL_FOR_SETTING[setting], "methods": cells
        }

    document = {
        "schema": RESULT_SCHEMA,
        "result_status": (
            "complete_val_seen_selection_val_unseen_frozen_evaluation_"
            "and_hidden_test_submissions"
        ),
        "benchmark": "reverie",
        "metric_unit": "percentage_points",
        "selection_policy": {
            "selection_split": "val_seen",
            "evaluation_split": "val_unseen",
            "hidden_submission_split": "test",
            "selection_on_val_unseen": False,
            "selection_or_ranking_on_test": False,
            "order_seed": 0,
            "qualification": (
                "val_seen selected each model-method configuration; val_unseen "
                "evaluated each frozen configuration once; hidden test stores "
                "submission evidence only and contains no local metrics."
            ),
        },
        "protocols": {
            "val_seen": registry["protocol"],
            "val_unseen": {
                "episode_count": VAL_UNSEEN_EPISODES,
                "order_seed": 0,
                "episode_order_sha256": spec["protocol"]["episode_order_sha256"],
                "source_protocol": val_source_ledger["source_protocol"],
                "full_split_only": True,
                "reported_metrics": list(METRICS),
            },
            "test": {
                "native_split": "test",
                "episode_count": TEST_EPISODES,
                "order_seed": 0,
                "episode_order_sha256": spec["test_submission_policy"]["episode_order_sha256"],
                "hidden_ground_truth": True,
                "local_metrics_available": False,
                "execution_purpose": "submission_generation_only",
                "eligible_tta_methods": list(TEST_METHODS),
                "unavailable_tta_methods": list(UNAVAILABLE_TEST_METHODS),
            },
        },
        "inputs": {
            "val_seen_registry": {
                "path": _display_path(repo_root, registry_path),
                "sha256": _sha256(registry_path),
            },
            "frozen_evaluation_specification": {
                "path": _display_path(repo_root, spec_path),
                "sha256": _sha256(spec_path),
            },
            "val_unseen_source_ledger": {
                "path": _display_path(repo_root, val_source_path),
                "sha256": _sha256(val_source_path),
            },
            "test_source_submission_ledger": {
                "path": _display_path(repo_root, test_source_path),
                "sha256": _sha256(test_source_path),
            },
            "val_unseen_batch": {
                "batch_id": val_batch["batch_id"],
                "batch_path": _display_path(repo_root, val_batch_path),
                "batch_sha256": _sha256(val_batch_path),
                "summary_path": _display_path(repo_root, val_summary_path),
                "summary_sha256": _sha256(val_summary_path),
            },
            "test_submission_batch": {
                "batch_id": test_batch["batch_id"],
                "batch_path": _display_path(repo_root, test_batch_path),
                "batch_sha256": _sha256(test_batch_path),
                "summary_path": _display_path(repo_root, test_summary_path),
                "summary_sha256": _sha256(test_summary_path),
            },
        },
        "supervision_categories": registry["supervision_categories"],
        "records": records,
    }
    validate_results_document(document)
    return document


def validate_results_document(document):
    if not isinstance(document, dict) or document.get("schema") != RESULT_SCHEMA:
        raise ArchiveError("unsupported REVERIE benchmark-results schema")
    if (
        document.get("benchmark") != "reverie"
        or document.get("metric_unit") != "percentage_points"
        or document.get("result_status")
        != "complete_val_seen_selection_val_unseen_frozen_evaluation_and_hidden_test_submissions"
    ):
        raise ArchiveError("REVERIE benchmark-results status is incomplete")
    policy = document.get("selection_policy", {})
    if (
        policy.get("selection_split") != "val_seen"
        or policy.get("evaluation_split") != "val_unseen"
        or policy.get("hidden_submission_split") != "test"
        or policy.get("selection_on_val_unseen") is not False
        or policy.get("selection_or_ranking_on_test") is not False
        or policy.get("order_seed") != 0
    ):
        raise ArchiveError("REVERIE benchmark-results selection policy is invalid")
    protocols = document.get("protocols")
    if not isinstance(protocols, dict) or set(protocols) != {
        "val_seen", "val_unseen", "test"
    }:
        raise ArchiveError("REVERIE benchmark protocols are incomplete")
    seen_protocol = protocols["val_seen"]
    if (
        seen_protocol.get("benchmark") != "reverie"
        or seen_protocol.get("split") != "val_seen"
        or seen_protocol.get("episode_count") != VAL_SEEN_EPISODES
        or seen_protocol.get("order_seed") != 0
        or not _is_sha256(seen_protocol.get("episode_order_sha256"))
    ):
        raise ArchiveError("REVERIE val_seen protocol is invalid")
    unseen_protocol = protocols["val_unseen"]
    if (
        unseen_protocol.get("episode_count") != VAL_UNSEEN_EPISODES
        or unseen_protocol.get("order_seed") != 0
        or not _is_sha256(unseen_protocol.get("episode_order_sha256"))
        or unseen_protocol.get("full_split_only") is not True
        or unseen_protocol.get("reported_metrics") != list(METRICS)
    ):
        raise ArchiveError("REVERIE val_unseen protocol is invalid")
    test_protocol = protocols["test"]
    if (
        test_protocol.get("native_split") != "test"
        or test_protocol.get("episode_count") != TEST_EPISODES
        or test_protocol.get("order_seed") != 0
        or not _is_sha256(test_protocol.get("episode_order_sha256"))
        or test_protocol.get("hidden_ground_truth") is not True
        or test_protocol.get("local_metrics_available") is not False
        or test_protocol.get("execution_purpose") != "submission_generation_only"
        or tuple(test_protocol.get("eligible_tta_methods", ())) != TEST_METHODS
        or tuple(test_protocol.get("unavailable_tta_methods", ()))
        != UNAVAILABLE_TEST_METHODS
    ):
        raise ArchiveError("REVERIE hidden-test protocol is invalid")
    expected_supervision = {
        "source_no_adaptation": {
            "methods": ["source"], "uses_episode_feedback": False,
        },
        "unsupervised_tta": {
            "methods": list(TEST_METHODS), "uses_episode_feedback": False,
        },
        "binary_episode_feedback_tta": {
            "methods": list(UNAVAILABLE_TEST_METHODS),
            "uses_episode_feedback": True,
            "feedback": "binary_navigation_success",
        },
    }
    if document.get("supervision_categories") != expected_supervision:
        raise ArchiveError("REVERIE supervision categories are invalid")
    inputs = document.get("inputs")
    expected_inputs = {
        "val_seen_registry", "frozen_evaluation_specification",
        "val_unseen_source_ledger", "test_source_submission_ledger",
        "val_unseen_batch", "test_submission_batch",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_inputs:
        raise ArchiveError("REVERIE benchmark input provenance is incomplete")
    for name, binding in inputs.items():
        if not isinstance(binding, dict):
            raise ArchiveError("{} input provenance is malformed".format(name))
        digest_keys = [key for key in binding if key.endswith("sha256")]
        if not digest_keys or any(not _is_sha256(binding[key]) for key in digest_keys):
            raise ArchiveError("{} input digest is invalid".format(name))
        path_keys = [key for key in binding if key.endswith("path")]
        if not path_keys or any(
            not isinstance(binding[key], str) or not binding[key]
            for key in path_keys
        ):
            raise ArchiveError("{} input path is invalid".format(name))
    records = document.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise ArchiveError("REVERIE benchmark-results setting matrix is incomplete")
    provenance_keys = {
        "run_tag", "git_commit", "checkpoint_sha256", "dataset_sha256",
        "episode_order_sha256", "formal_manifest_path",
        "formal_manifest_sha256", "formal_immutable_identity_sha256",
    }
    for setting in SETTINGS:
        if records[setting].get("model") != MODEL_FOR_SETTING[setting]:
            raise ArchiveError("{} benchmark-results model mismatch".format(setting))
        methods = records[setting].get("methods")
        if not isinstance(methods, dict) or set(methods) != set(ALL_METHODS):
            raise ArchiveError("{} benchmark-results method matrix is incomplete".format(setting))
        for split in ("val_seen", "val_unseen"):
            source = _metrics(methods["source"][split].get("metrics"), setting + " source")
            for method in ALL_METHODS:
                cell = methods[method]
                if (
                    cell.get("method") != method
                    or cell.get("supervision_category") != SUPERVISION_FOR_METHOD[method]
                    or not isinstance(cell.get("frozen_parameters"), dict)
                    or not cell["frozen_parameters"]
                ):
                    raise ArchiveError("{} {} method metadata mismatch".format(setting, method))
                result = cell.get(split)
                metrics = _metrics(result.get("metrics"), "{} {} {}".format(setting, method, split))
                if result.get("delta_vs_source_pp") != _delta(metrics, source):
                    raise ArchiveError("{} {} {} Source delta mismatch".format(setting, method, split))
                provenance = result.get("provenance")
                if not isinstance(provenance, dict) or set(provenance) != provenance_keys:
                    raise ArchiveError("{} {} {} provenance is incomplete".format(setting, method, split))
                for key in (
                    "checkpoint_sha256", "dataset_sha256",
                    "episode_order_sha256", "formal_manifest_sha256",
                    "formal_immutable_identity_sha256",
                ):
                    if not _is_sha256(provenance.get(key)):
                        raise ArchiveError("{} {} {} provenance digest is invalid".format(setting, method, split))
                if (
                    not isinstance(provenance.get("run_tag"), str)
                    or not isinstance(provenance.get("formal_manifest_path"), str)
                    or HEX_COMMIT.fullmatch(str(provenance.get("git_commit", ""))) is None
                ):
                    raise ArchiveError("{} {} {} provenance identity is invalid".format(setting, method, split))
        for method in ALL_METHODS:
            test = methods[method].get("test")
            if method in UNAVAILABLE_TEST_METHODS:
                if (
                    test.get("status") != "N/A"
                    or test.get("metrics") is not None
                    or test.get("submission") is not None
                    or test.get("legal_online_feedback_interface") is not None
                    or not isinstance(test.get("reason"), str)
                    or not test["reason"]
                    or "provenance" in test
                ):
                    raise ArchiveError("{} {} hidden-test N/A record is invalid".format(setting, method))
            else:
                expected_status = (
                    "reused_submission_ready" if method == "source"
                    else "frozen_submission_ready"
                )
                if test.get("status") != expected_status or test.get("metrics") is not None:
                    raise ArchiveError("{} {} hidden-test submission status is invalid".format(setting, method))
                submission = test.get("submission")
                if (
                    not isinstance(submission, dict)
                    or not isinstance(submission.get("path"), str)
                    or not _is_sha256(submission.get("sha256"))
                    or isinstance(submission.get("size"), bool)
                    or not isinstance(submission.get("size"), int)
                    or submission["size"] < 0
                    or not isinstance(test.get("provenance"), dict)
                    or set(test["provenance"]) != provenance_keys
                ):
                    raise ArchiveError("{} {} hidden-test submission evidence is incomplete".format(setting, method))
                provenance = test["provenance"]
                if any(
                    not _is_sha256(provenance.get(key))
                    for key in (
                        "checkpoint_sha256", "dataset_sha256",
                        "episode_order_sha256", "formal_manifest_sha256",
                        "formal_immutable_identity_sha256",
                    )
                ) or HEX_COMMIT.fullmatch(str(provenance.get("git_commit", ""))) is None:
                    raise ArchiveError("{} {} hidden-test provenance is invalid".format(setting, method))
    return document


def _fmt_metrics(value):
    return "{:.2f} / {:.2f} / {:.2f} / {:.2f}".format(
        *(float(value[name]) for name in METRICS)
    )


def _fmt_delta(value):
    return "{:+.2f} / {:+.2f} / {:+.2f} / {:+.2f}".format(
        *(float(value[name]) for name in METRICS)
    )


def render_report(document):
    validate_results_document(document)
    lines = [
        "# REVERIE final benchmark results",
        "",
        "Hyperparameters were selected only on `val_seen`. `val_unseen` "
        "evaluates each frozen configuration once and does not reselect it.",
        "",
        "The official hidden split is named `test`. It has no local ground "
        "truth, so this archive records submission readiness—not test metrics. "
        "FeedTTA and ATENA are N/A on `test` because no legal online binary "
        "navigation-success feedback interface is available.",
        "",
        "FeedTTA and ATENA consume binary episode-success feedback on splits "
        "where that feedback is legally available. Tent, FSTTA, and EAM are "
        "unsupervised TTA.",
        "",
        "## Main comparison",
        "",
        "Metric order and deltas are `SR / SPL / RGS / RGSPL` in percentage points.",
        "",
        "| Model | Method | Supervision | val_seen | val_unseen | val_unseen delta vs Source | test |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for setting in SETTINGS:
        model = document["records"][setting]["model"].upper()
        for method in ALL_METHODS:
            cell = document["records"][setting]["methods"][method]
            test_status = (
                "N/A (no legal online feedback)"
                if method in UNAVAILABLE_TEST_METHODS
                else "submission ready; no local metrics"
            )
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                model, method, cell["supervision_category"],
                _fmt_metrics(cell["val_seen"]["metrics"]),
                _fmt_metrics(cell["val_unseen"]["metrics"]),
                _fmt_delta(cell["val_unseen"]["delta_vs_source_pp"]),
                test_status,
            ))
    lines.extend([
        "",
        "## Frozen parameters",
        "",
        "Source uses deterministic argmax. Every TTA parameter object is "
        "copied verbatim from the frozen `val_seen` registry.",
        "",
        "| Model | Method | Parameters |",
        "|---|---|---|",
    ])
    for setting in SETTINGS:
        model = document["records"][setting]["model"].upper()
        for method in ALL_METHODS:
            params = document["records"][setting]["methods"][method]["frozen_parameters"]
            lines.append("| {} | {} | `{}` |".format(model, method, _canonical(params)))
    lines.extend([
        "",
        "## Hidden-test submission evidence",
        "",
        "No value in this section is a local test metric.",
        "",
        "| Model | Method | Status | Submission SHA256 |",
        "|---|---|---|---|",
    ])
    for setting in SETTINGS:
        model = document["records"][setting]["model"].upper()
        for method in ALL_METHODS:
            test = document["records"][setting]["methods"][method]["test"]
            digest = "—" if test.get("submission") is None else "`{}`".format(test["submission"]["sha256"])
            lines.append("| {} | {} | {} | {} |".format(model, method, test["status"], digest))
    lines.extend([
        "",
        "## Metric provenance",
        "",
        "| Model | Method | Split | Run tag | Git commit | Formal manifest SHA256 |",
        "|---|---|---|---|---|---|",
    ])
    for setting in SETTINGS:
        model = document["records"][setting]["model"].upper()
        for method in ALL_METHODS:
            cell = document["records"][setting]["methods"][method]
            for split in ("val_seen", "val_unseen"):
                provenance = cell[split]["provenance"]
                lines.append("| {} | {} | {} | `{}` | `{}` | `{}` |".format(
                    model, method, split, provenance["run_tag"],
                    provenance["git_commit"], provenance["formal_manifest_sha256"],
                ))
    lines.append("")
    return "\n".join(lines)


def _atomic_text(path, content):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def write_results(document, output_path=DEFAULT_OUTPUT, report_path=DEFAULT_REPORT):
    validate_results_document(document)
    payload = json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    _atomic_text(output_path, payload)
    _atomic_text(report_path, render_report(document))
    return Path(output_path).resolve(), Path(report_path).resolve()


def validate_outputs(document, output_path=DEFAULT_OUTPUT, report_path=DEFAULT_REPORT):
    output_path = _require_file(output_path, "REVERIE final benchmark JSON")
    report_path = _require_file(report_path, "REVERIE final benchmark report")
    actual = _read_json(output_path, "REVERIE final benchmark JSON")
    validate_results_document(actual)
    if not _same(actual, document):
        raise ArchiveError("final benchmark JSON differs from downloaded evidence")
    if report_path.read_text(encoding="utf-8") != render_report(document):
        raise ArchiveError("final benchmark report differs from downloaded evidence")
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-unseen-batch-root", type=Path, required=True)
    parser.add_argument("--test-batch-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    document = build_results(
        args.val_unseen_batch_root,
        args.test_batch_root,
        repo_root=args.repo_root,
        registry_path=args.registry,
        spec_path=args.spec,
    )
    if args.check_only:
        print("validated complete REVERIE val_unseen metrics and hidden-test submissions")
        return 0
    if args.validate:
        validate_outputs(document, args.output, args.report)
        print("validated REVERIE final benchmark outputs")
        return 0
    output, report = write_results(document, args.output, args.report)
    print("wrote {}".format(output))
    print("wrote {}".format(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArchiveError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
