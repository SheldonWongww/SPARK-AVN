#!/usr/bin/env python3
"""Build the frozen REVERIE val_seen winner registry offline.

The input is the authenticated ``FINAL_SELECTION.json`` emitted by
``run_reverie_small_hparam_search.py``.  No experiment is launched here.  The
builder resolves every winner to its canonical manifest under
``vln/results/runs``, validates the complete 3 x 5 matrix, and writes a
portable selected-winners document plus its derived registry atomically.
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
from vln.scripts import tta_config_cli  # noqa: E402


SEARCH_SELECTION_SCHEMA = (
    "navtta.vln_reverie_val_seen_small_search_selection.v1"
)
SELECTION_SCHEMA = "navtta.vln_reverie_final_selection.v1"
REGISTRY_SCHEMA = "navtta.vln_reverie_final_registry.v1"
SOURCE_SCHEMA = "navtta.vln_reverie_reused_source_controls.v1"

DEFAULT_SOURCE = REPO_ROOT / "vln/manifests/reverie_reused_source_controls.json"
DEFAULT_SEARCH_SPEC = (
    REPO_ROOT / "vln/experiments/reverie_val_seen_small_hparam_search_v1.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "vln/results/final/reverie"
DEFAULT_SELECTION = DEFAULT_OUTPUT_DIR / "selected_winners.json"
DEFAULT_REGISTRY = DEFAULT_OUTPUT_DIR / "registry.json"
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"

SETTINGS = ("duet-reverie", "hamt-reverie", "goat-reverie")
MODELS = ("duet", "hamt", "goat")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
ALL_METHODS = ("source",) + METHODS
MODEL_FOR_SETTING = dict(zip(SETTINGS, MODELS))
EXPECTED_BENCHMARK = {
    "duet-reverie": "reverie_discrete_duet_hamt",
    "hamt-reverie": "reverie_discrete_duet_hamt",
    "goat-reverie": "reverie_discrete_goat",
}
SUPERVISION = {
    "source": "source_no_adaptation",
    "tent": "unsupervised_tta",
    "fstta": "unsupervised_tta",
    "eam": "unsupervised_tta",
    "feedtta": "binary_episode_feedback_tta",
    "atena": "binary_episode_feedback_tta",
}
METRICS = ("SR", "SPL", "RGS", "RGSPL")
EXPECTED_EPISODES = 1423
EXPECTED_ORDER_SEED = 0
EXPECTED_ORDER_SHA256 = (
    "aef9a2b094080347dc928cc19ddc40946299f77a91d380ee00bee10449cf1fd1"
)
HEX64 = re.compile(r"[0-9a-f]{64}")
METRIC_RE = re.compile(
    r"\b(sr|spl|rgs|rgspl):\s*(-?[0-9]+(?:\.[0-9]+)?)"
)


class RegistryError(RuntimeError):
    pass


def _read_json(path, label="JSON"):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise RegistryError("{} must be a JSON object".format(label))
    return value


def _json_bytes(value):
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")


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


def _bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _resolve(value, root=REPO_ROOT):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _repo_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        raise RegistryError("evidence is outside the repository: {}".format(path))


def _validate_metrics(metrics, label):
    if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
        raise RegistryError("{} must contain exactly {}".format(label, METRICS))
    for key in METRICS:
        value = metrics[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 100.0
        ):
            raise RegistryError("{} has invalid {}".format(label, key))


def _validate_parameters(method, parameters, label):
    if not isinstance(parameters, dict) or not parameters:
        raise RegistryError("{} parameters are empty".format(label))
    allowed = tta_config_cli.COMMON | tta_config_cli.METHOD_KEYS.get(method, set())
    unknown = set(parameters).difference(allowed)
    if unknown:
        raise RegistryError("{} has unknown parameters: {}".format(
            label, ", ".join(sorted(unknown))
        ))
    for key, value in parameters.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RegistryError("{} has non-finite parameter {}".format(label, key))
    for key in ("action_seed", "sgr_seed"):
        if key in parameters and type(parameters[key]) is not int:
            raise RegistryError("{} {} must be an exact integer".format(label, key))
    if method == "tent" and parameters.get("update_interval") != 1:
        raise RegistryError("{} Tent update_interval must equal 1".format(label))
    if method in ("feedtta", "atena") and parameters.get(
        "action_selection"
    ) != "argmax":
        raise RegistryError("{} must use target-native argmax".format(label))


def _validate_artifact_metadata(manifest, label):
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RegistryError("{} manifest has no result artifacts".format(label))
    names = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise RegistryError("{} has malformed artifact metadata".format(label))
        name = item.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise RegistryError("{} has duplicate/missing artifact name".format(label))
        names.add(name)
        if not isinstance(item.get("size"), int) or item["size"] < 0:
            raise RegistryError("{} artifact size is invalid".format(label))
        if not _is_sha256(item.get("sha256")):
            raise RegistryError("{} artifact digest is invalid".format(label))
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise RegistryError("{} artifact path is invalid".format(label))
        normalized = path.replace("\\", "/")
        if normalized != name and not normalized.endswith("/" + name):
            raise RegistryError("{} artifact path/name mismatch".format(label))


def _relocate_result_artifact(value):
    """Resolve an artifact path after an AutoDL result tree is downloaded."""
    path = Path(str(value))
    if path.is_file():
        return path.resolve()
    normalized = str(value).replace("\\", "/")
    marker = "vln/results/"
    position = normalized.find(marker)
    if position >= 0:
        candidate = REPO_ROOT / normalized[position:]
        if candidate.is_file():
            return candidate.resolve()
    raise RegistryError("downloaded result artifact is missing: {}".format(value))


def _manifest_artifact(manifest, suffix, label):
    matches = [
        item for item in manifest.get("result_artifacts", [])
        if isinstance(item, dict) and str(item.get("name", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise RegistryError(
            "{} must authenticate exactly one {} artifact".format(label, suffix)
        )
    item = matches[0]
    path = _relocate_result_artifact(item.get("path", ""))
    if path.stat().st_size != item.get("size") or _sha256(path) != item.get("sha256"):
        raise RegistryError("{} {} artifact changed".format(label, suffix))
    return path, item


def _parse_reverie_metrics(path, split, label):
    matches = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        values = {
            key.upper(): float(value) for key, value in METRIC_RE.findall(line)
        }
        if set(values) == set(METRICS) and "Env name: {}".format(split) in line:
            matches.append(values)
    if len(matches) != 1:
        raise RegistryError(
            "{} must contain exactly one authenticated {} metric row".format(
                label, split
            )
        )
    return matches[0]


def _assert_metrics_equal(actual, expected, label):
    _validate_metrics(actual, label + " artifact")
    _validate_metrics(expected, label + " selection")
    if any(
        not math.isclose(float(actual[key]), float(expected[key]), abs_tol=1e-9)
        for key in METRICS
    ):
        raise RegistryError("{} metrics differ from authenticated valid.txt".format(label))


def _assert_effective_parameters(manifest, setting, method, parameters, label):
    overrides = manifest.get("config_overrides")
    if not isinstance(overrides, list):
        raise RegistryError("{} manifest lacks config_overrides".format(label))
    try:
        start = overrides.index("--tta_method")
    except ValueError:
        raise RegistryError("{} manifest has no TTA CLI segment".format(label))
    actual = [str(value) for value in overrides[start:]]
    try:
        expected = tta_config_cli._discrete(
            method, parameters, "__DIAGNOSTICS__"
        )
    except ValueError as error:
        raise RegistryError("{} parameters cannot translate: {}".format(label, error))
    for tokens in (actual, expected):
        try:
            index = tokens.index("--tta_diagnostics")
            tokens[index + 1] = "__DIAGNOSTICS__"
        except (ValueError, IndexError):
            raise RegistryError("{} TTA diagnostics CLI is malformed".format(label))
    if actual != expected:
        raise RegistryError(
            "{} parameters differ from immutable manifest config_overrides".format(label)
        )


def _selection_score(record):
    return tuple(float(record["metrics"][key]) for key in (
        "RGSPL", "RGS", "SPL", "SR"
    ))


def _formal_path(run_tag, setting, split):
    return FORMAL_ROOT / "{}-{}-{}-native".format(run_tag, setting, split) / "manifest.json"


def _validate_formal(record, setting, method, split, checkpoint_sha256,
                     order_sha256, dataset_sha256):
    label = "{} {}".format(setting, method)
    run_tag = record.get("run_tag")
    if not isinstance(run_tag, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", run_tag):
        raise RegistryError("{} run tag is invalid".format(label))
    path = _formal_path(run_tag, setting, split).resolve()
    if not path.is_file():
        raise RegistryError("{} canonical manifest is missing: {}".format(label, path))
    expected_digest = record.get("formal_manifest_sha256")
    if not _is_sha256(expected_digest) or _sha256(path) != expected_digest:
        raise RegistryError("{} formal manifest SHA256 mismatch".format(label))
    manifest = _read_json(path, "{} formal manifest".format(label))
    expected_run_id = "{}-{}-{}-native".format(run_tag, setting, split)
    expected = {
        "run_id": expected_run_id,
        "task": "vln",
        "benchmark": EXPECTED_BENCHMARK[setting],
        "model": MODEL_FOR_SETTING[setting],
        "method": method,
        "run_tag": run_tag,
        "seed": 0,
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RegistryError("{} manifest {} mismatch".format(label, key))
    expected_source = "{}:{}:native".format(setting, split)
    if method != "source":
        expected_source += ":{}".format(method)
    if manifest.get("source_setting") != expected_source:
        raise RegistryError("{} source_setting mismatch".format(label))
    if manifest.get("checkpoint", {}).get("sha256") != checkpoint_sha256:
        raise RegistryError("{} checkpoint digest mismatch".format(label))
    dataset = manifest.get("dataset", {})
    if dataset.get("stream_order_sha256") != order_sha256:
        raise RegistryError("{} episode order digest mismatch".format(label))
    if dataset.get("stream_content_sha256") != dataset_sha256:
        raise RegistryError("{} dataset digest mismatch".format(label))
    identity = manifest.get("immutable_identity_sha256")
    if not _is_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise RegistryError("{} immutable identity mismatch".format(label))
    _validate_artifact_metadata(manifest, label)
    return path, manifest


def load_source_ledger(path=DEFAULT_SOURCE):
    path = Path(path).resolve()
    ledger = _read_json(path, "REVERIE val_seen Source ledger")
    expected = {
        "schema": SOURCE_SCHEMA,
        "benchmark": "reverie",
        "split": "val_seen",
        "source_protocol": "standard_argmax",
        "episode_count": EXPECTED_EPISODES,
        "order_seed": EXPECTED_ORDER_SEED,
        "episode_order_sha256": EXPECTED_ORDER_SHA256,
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise RegistryError("Source ledger {} mismatch".format(key))
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("Source ledger must contain exactly three settings")
    source_batch = ledger.get("source_batch", {})
    if (
        not isinstance(source_batch.get("batch_id"), str)
        or not re.fullmatch(r"[0-9a-f]{40}", str(source_batch.get("git_commit", "")))
    ):
        raise RegistryError("Source batch provenance is invalid")
    for setting in SETTINGS:
        record = records[setting]
        label = "{} Source".format(setting)
        if record.get("model") != MODEL_FOR_SETTING[setting]:
            raise RegistryError("{} model mismatch".format(label))
        if record.get("parameters") != {
            "action_selection": "argmax", "action_seed": 0
        }:
            raise RegistryError("{} protocol mismatch".format(label))
        _validate_metrics(record.get("metrics"), label)
        if not _is_sha256(record.get("checkpoint_sha256")):
            raise RegistryError("{} checkpoint digest is invalid".format(label))
        formal_record = dict(record)
        formal_record["formal_manifest_sha256"] = record.get(
            "formal_manifest_sha256"
        )
        path_expected = _formal_path(record["run_tag"], setting, "val_seen")
        if _repo_path(path_expected) != record.get("formal_manifest_path"):
            raise RegistryError("{} manifest path is noncanonical".format(label))
        path_actual, manifest = _validate_formal(
            formal_record, setting, "source", "val_seen",
            record["checkpoint_sha256"], EXPECTED_ORDER_SHA256,
            record["dataset_sha256"],
        )
        if manifest.get("immutable_identity_sha256") != record.get(
            "immutable_identity_sha256"
        ):
            raise RegistryError("{} ledger identity mismatch".format(label))
        if (
            manifest.get("run_tag") != source_batch["batch_id"]
            or manifest.get("git_commit") != source_batch["git_commit"]
        ):
            raise RegistryError("{} Source batch/commit mismatch".format(label))
        if _repo_path(path_actual) != record["formal_manifest_path"]:
            raise RegistryError("{} formal path mismatch".format(label))
        metric_path, metric_meta = _manifest_artifact(
            manifest, "logs/valid.txt", label
        )
        if (
            _sha256(metric_path) != record.get("metrics_artifact_sha256")
            or metric_meta.get("sha256") != record.get("metrics_artifact_sha256")
        ):
            raise RegistryError("{} metric ledger digest mismatch".format(label))
        _assert_metrics_equal(
            _parse_reverie_metrics(metric_path, "val_seen", label),
            record["metrics"], label,
        )
    return path, ledger


def _load_full_candidates(selection_path, search_spec, source, batch_id,
                          selection_git_commit, spec_sha256):
    target_cells = {
        (setting, method)
        for method in METHODS
        for setting in search_spec.get("methods", {}).get(
            method, {}
        ).get("candidates_by_setting", {})
    }
    promotions = {}
    for setting in SETTINGS:
        setting_targets = {
            method for candidate_setting, method in target_cells
            if candidate_setting == setting
        }
        if not setting_targets:
            continue
        path = (
            Path(selection_path).parent / "stages" / "screening" / setting
            / "PROMOTIONS.json"
        )
        document = _read_json(path, "{} promotions".format(setting))
        if (
            document.get("schema")
            != "navtta.vln_reverie_small_search_promotions.v1"
            or document.get("experiment_id") != search_spec["experiment_id"]
            or document.get("spec_sha256") != spec_sha256
            or document.get("selection_scope")
            != "canonical_first_256_episodes_only"
            or document.get("primary_metric") != "RGSPL"
        ):
            raise RegistryError("{} promotions provenance mismatch".format(setting))
        items = document.get("promotions")
        if not isinstance(items, list):
            raise RegistryError("{} promotions are malformed".format(setting))
        for item in items:
            cell = (item.get("setting"), item.get("method"))
            if cell not in target_cells or cell in promotions:
                raise RegistryError("invalid/duplicate promotion {}".format(cell))
            promotions[cell] = item
    if set(promotions) != target_cells:
        raise RegistryError("screening promotions do not cover target cells")
    by_cell = {}
    for path in Path(selection_path).parent.rglob("metrics.json"):
        if not path.parent.name.startswith("attempt-"):
            continue
        try:
            record = _read_json(path, "full-run metrics")
        except RegistryError:
            continue
        if record.get("stage") != "full":
            continue
        cell = (record.get("setting"), record.get("method"))
        if cell not in target_cells:
            raise RegistryError("unexpected full candidate cell {}".format(cell))
        if cell in by_cell:
            raise RegistryError("multiple completed full candidates for {}".format(cell))
        setting, method = cell
        label = "{} {} full candidate".format(setting, method)
        expected = {
            "batch_id": batch_id,
            "setting": setting,
            "method": method,
            "git_commit": selection_git_commit,
            "spec_sha256": spec_sha256,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise RegistryError("{} {} mismatch".format(label, key))
        _validate_parameters(method, record.get("parameters"), label)
        method_spec = search_spec["methods"][method]
        allowed_parameters = []
        for candidate in method_spec["candidates_by_setting"][setting]:
            parameters = {
                key: value for key, value in candidate.items() if key != "role"
            }
            overlap = set(parameters).intersection(method_spec.get("fixed", {}))
            if any(
                parameters[key] != method_spec["fixed"][key] for key in overlap
            ):
                raise RegistryError("{} search grid overrides fixed keys".format(label))
            parameters.update(method_spec.get("fixed", {}))
            allowed_parameters.append(parameters)
        if not any(
            _canonical(record["parameters"]) == _canonical(parameters)
            for parameters in allowed_parameters
        ):
            raise RegistryError("{} parameters are outside the reviewed grid".format(label))
        promotion = promotions[cell]
        if (
            _canonical(promotion.get("parameters"))
            != _canonical(record["parameters"])
            or promotion.get("screening_run_tag")
            != record.get("promoted_from_run_tag")
            or promotion.get("role") != record.get("role")
        ):
            raise RegistryError("{} differs from screened promotion".format(label))
        _validate_metrics(record.get("metrics"), label)
        source_record = source["records"][setting]
        formal_record = {
            "run_tag": record.get("run_tag"),
            "formal_manifest_sha256": record.get("formal_manifest_sha256"),
        }
        formal_path, manifest = _validate_formal(
            formal_record, setting, method, "val_seen",
            source_record["checkpoint_sha256"], EXPECTED_ORDER_SHA256,
            source_record["dataset_sha256"],
        )
        if (
            manifest.get("git_commit") != selection_git_commit
            or not record["run_tag"].startswith(batch_id + "-full-")
            or batch_id not in str(manifest.get("config", ""))
        ):
            raise RegistryError("{} campaign identity mismatch".format(label))
        _assert_effective_parameters(
            manifest, setting, method, record["parameters"], label
        )
        metric_path, metric_meta = _manifest_artifact(
            manifest, "logs/valid.txt", label
        )
        _assert_metrics_equal(
            _parse_reverie_metrics(metric_path, "val_seen", label),
            record["metrics"], label,
        )
        if metric_meta["sha256"] != record.get("metric_artifact_sha256"):
            raise RegistryError("{} metric artifact binding mismatch".format(label))
        by_cell[cell] = {
            "run_tag": record["run_tag"],
            "parameters": record["parameters"],
            "metrics": record["metrics"],
            "formal_manifest_path": _repo_path(formal_path),
            "formal_manifest_sha256": record["formal_manifest_sha256"],
        }
    if set(by_cell) != target_cells:
        missing = sorted(target_cells.difference(by_cell))
        raise RegistryError(
            "full candidate evidence is incomplete; missing {}".format(missing)
        )
    return by_cell


def _normalize_search_selection(selection_path, source_path=DEFAULT_SOURCE,
                                search_spec_path=DEFAULT_SEARCH_SPEC):
    selection_path = Path(selection_path).resolve()
    source_path, source = load_source_ledger(source_path)
    search_spec_path = Path(search_spec_path).resolve()
    search_spec = _read_json(search_spec_path, "REVERIE small-search spec")
    if search_spec.get("schema") != (
        "navtta.vln_reverie_val_seen_small_hparam_search.v1"
    ):
        raise RegistryError("unsupported REVERIE small-search spec")
    dependency = search_spec.get("dependencies", {}).get("source_registry", {})
    if (
        _resolve(dependency.get("path", "")).resolve() != source_path
        or dependency.get("sha256") != _sha256(source_path)
    ):
        raise RegistryError("small-search Source-ledger dependency mismatch")
    incumbent_spec_binding = search_spec.get("dependencies", {}).get(
        "incumbent_spec", {}
    )
    incumbent_spec_path = _resolve(incumbent_spec_binding.get("path", "")).resolve()
    if (
        not incumbent_spec_path.is_file()
        or _sha256(incumbent_spec_path) != incumbent_spec_binding.get("sha256")
    ):
        raise RegistryError("small-search incumbent-spec dependency mismatch")
    incumbent_spec = _read_json(incumbent_spec_path, "incumbent transfer spec")
    incumbent_parameters = {
        (item["setting"], item["method"]): item["parameters"]
        for item in incumbent_spec.get("jobs", [])
    }
    expected_cells = {
        (setting, method) for setting in SETTINGS for method in METHODS
    }
    if set(incumbent_parameters) != expected_cells:
        raise RegistryError("incumbent transfer spec matrix is incomplete")
    batch_path = selection_path.parent / "BATCH.json"
    batch = _read_json(batch_path, "small-search batch binding")
    document = _read_json(selection_path, "REVERIE small-search selection")
    if document.get("schema") != SEARCH_SELECTION_SCHEMA:
        raise RegistryError("unsupported small-search selection schema")
    if (
        document.get("experiment_id") != search_spec.get("experiment_id")
        or document.get("split") != "val_seen" or document.get(
        "primary_metric"
        ) != "RGSPL"
    ):
        raise RegistryError("selection is not REVERIE val_seen RGSPL selection")
    expected_spec_sha = _sha256(search_spec_path)
    if document.get("spec_sha256") != expected_spec_sha:
        raise RegistryError("selection does not bind the tracked small-search spec")
    if not re.fullmatch(r"[0-9a-f]{40}", str(document.get("git_commit", ""))):
        raise RegistryError("selection git commit is invalid")
    tolerance = search_spec.get("protocol", {}).get(
        "source_sr_floor_tolerance_percentage_points"
    )
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or document.get("source_sr_floor_tolerance_percentage_points")
        != tolerance
    ):
        raise RegistryError("selection Source-SR floor differs from search spec")
    expected_batch = {
        "schema": "navtta.vln_reverie_small_search_batch.v1",
        "experiment_id": search_spec["experiment_id"],
        "spec_sha256": expected_spec_sha,
        "git_commit": document["git_commit"],
        "source_registry_sha256": dependency["sha256"],
        "incumbent_spec_sha256": incumbent_spec_binding["sha256"],
    }
    for key, value in expected_batch.items():
        if batch.get(key) != value:
            raise RegistryError("small-search BATCH.json {} mismatch".format(key))
    batch_id = batch.get("batch_id")
    if not isinstance(batch_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", batch_id):
        raise RegistryError("small-search BATCH.json has invalid batch_id")
    records = document.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("selection must contain exactly three settings")
    full_candidates = _load_full_candidates(
        selection_path, search_spec, source, batch_id, document["git_commit"],
        expected_spec_sha,
    )

    normalized = {}
    for setting in SETTINGS:
        methods = records[setting]
        if not isinstance(methods, dict) or set(methods) != set(METHODS):
            raise RegistryError("{} selection is not a five-method matrix".format(setting))
        source_record = source["records"][setting]
        normalized[setting] = {}
        for method in METHODS:
            label = "{} {}".format(setting, method)
            record = methods[method]
            if not isinstance(record, dict):
                raise RegistryError("{} winner is malformed".format(label))
            if record.get("origin") not in {
                "r2r_frozen_transfer_incumbent", "new_full_candidate"
            }:
                raise RegistryError("{} winner origin is invalid".format(label))
            _validate_parameters(method, record.get("parameters"), label)
            _validate_metrics(record.get("metrics"), label)
            _validate_metrics(record.get("source_metrics"), label + " source")
            if _canonical(record["source_metrics"]) != _canonical(
                source_record["metrics"]
            ):
                raise RegistryError("{} Source metrics mismatch".format(label))
            expected_delta = {
                key: round(
                    float(record["metrics"][key])
                    - float(source_record["metrics"][key]),
                    6,
                )
                for key in METRICS
            }
            if record.get("delta_vs_source_pp") != expected_delta:
                raise RegistryError("{} Source delta mismatch".format(label))
            if record.get("better_than_source_on_rgspl") is not (
                record["metrics"]["RGSPL"] > source_record["metrics"]["RGSPL"]
            ):
                raise RegistryError("{} RGSPL comparison flag mismatch".format(label))
            formal_record = {
                "run_tag": record.get("run_tag"),
                "formal_manifest_sha256": record.get(
                    "formal_manifest_sha256"
                ),
            }
            formal_path, manifest = _validate_formal(
                formal_record, setting, method, "val_seen",
                source_record["checkpoint_sha256"], EXPECTED_ORDER_SHA256,
                source_record["dataset_sha256"],
            )
            origin = record.get("origin")
            incumbent = search_spec.get("incumbents", {}).get(
                setting, {}
            ).get(method, {})
            expected_incumbent = {
                "run_tag": incumbent.get("run_tag"),
                "formal_manifest_sha256": incumbent.get(
                    "formal_manifest_sha256"
                ),
                "metrics": incumbent.get("metrics"),
                "parameters": incumbent_parameters[(setting, method)],
            }
            candidate = full_candidates.get((setting, method))
            candidate_eligible = candidate is not None and float(
                candidate["metrics"]["SR"]
            ) >= float(source_record["metrics"]["SR"]) - float(tolerance)
            expected_origin = (
                "new_full_candidate"
                if candidate_eligible
                and _selection_score(candidate) > _selection_score(expected_incumbent)
                else "r2r_frozen_transfer_incumbent"
            )
            if origin != expected_origin:
                raise RegistryError(
                    "{} winner disagrees with authenticated full-vs-incumbent "
                    "selection policy".format(label)
                )
            if origin == "r2r_frozen_transfer_incumbent":
                actual_incumbent = {
                    "run_tag": record.get("run_tag"),
                    "formal_manifest_sha256": record.get(
                        "formal_manifest_sha256"
                    ),
                    "metrics": record.get("metrics"),
                    "parameters": record.get("parameters"),
                }
                if _canonical(actual_incumbent) != _canonical(expected_incumbent):
                    raise RegistryError("{} is not the pinned incumbent".format(label))
                expected_commit = search_spec["dependencies"][
                    "incumbent_git_commit"
                ]
                if manifest.get("git_commit") != expected_commit:
                    raise RegistryError("{} incumbent commit mismatch".format(label))
                if search_spec["dependencies"]["incumbent_batch_id"] not in str(
                    manifest.get("config", "")
                ):
                    raise RegistryError("{} incumbent campaign mismatch".format(label))
            else:
                if record.get("new_candidate_eligible") is not True:
                    raise RegistryError("{} selected new candidate was ineligible".format(label))
                if float(record["metrics"]["SR"]) < (
                    float(source_record["metrics"]["SR"]) - float(tolerance)
                ):
                    raise RegistryError("{} violates the Source-SR floor".format(label))
                if _selection_score(record) <= _selection_score(expected_incumbent):
                    raise RegistryError("{} does not beat the pinned incumbent".format(label))
                if (
                    manifest.get("git_commit") != document["git_commit"]
                    or not record["run_tag"].startswith(batch_id + "-full-")
                    or batch_id not in str(manifest.get("config", ""))
                ):
                    raise RegistryError("{} new candidate campaign mismatch".format(label))
                expected_candidate = {
                    key: candidate[key] for key in (
                        "run_tag", "parameters", "metrics",
                        "formal_manifest_sha256",
                    )
                }
                actual_candidate = {
                    key: record.get(key) for key in expected_candidate
                }
                if _canonical(actual_candidate) != _canonical(expected_candidate):
                    raise RegistryError("{} differs from authenticated full winner".format(label))
            _assert_effective_parameters(
                manifest, setting, method, record["parameters"], label
            )
            metric_path, metric_meta = _manifest_artifact(
                manifest, "logs/valid.txt", label
            )
            parsed_metrics = _parse_reverie_metrics(
                metric_path, "val_seen", label
            )
            _assert_metrics_equal(parsed_metrics, record["metrics"], label)
            declared_metric_sha = record.get("metric_artifact_sha256")
            if declared_metric_sha is not None and (
                not _is_sha256(declared_metric_sha)
                or declared_metric_sha != metric_meta["sha256"]
            ):
                raise RegistryError(
                    "{} selected metric-artifact digest mismatch".format(label)
                )
            normalized[setting][method] = {
                "selection_status": "ready",
                "origin": origin,
                "supervision_category": SUPERVISION[method],
                "run_tag": record["run_tag"],
                "parameters": record["parameters"],
                "metrics": record["metrics"],
                "source_metrics": source_record["metrics"],
                "delta_vs_source_pp": expected_delta,
                "formal_manifest_path": _repo_path(formal_path),
                "formal_manifest_sha256": record["formal_manifest_sha256"],
                "formal_immutable_identity_sha256": manifest[
                    "immutable_identity_sha256"
                ],
                "checkpoint_sha256": source_record["checkpoint_sha256"],
                "metric_artifact_sha256": metric_meta["sha256"],
            }
    return {
        "schema": SELECTION_SCHEMA,
        "selection_status": "complete",
        "purpose": (
            "Freeze the authenticated REVERIE val_seen order-seed-0 winners "
            "for evaluation on untouched splits."
        ),
        "protocol": {
            "benchmark": "reverie",
            "split": "val_seen",
            "episode_count": EXPECTED_EPISODES,
            "order_seed": EXPECTED_ORDER_SEED,
            "episode_order_sha256": EXPECTED_ORDER_SHA256,
            "primary_metric": "RGSPL",
            "selection_scope": "val_seen_seed0_hyperparameter_selection",
        },
        "source_ledger": {
            "path": _repo_path(source_path),
            "sha256": _sha256(source_path),
        },
        "search_evidence": {
            "experiment_id": document.get("experiment_id"),
            "git_commit": document.get("git_commit"),
            "spec_sha256": document["spec_sha256"],
            "final_selection_sha256": _sha256(selection_path),
            "batch_id": batch_id,
            "batch_binding_sha256": _sha256(batch_path),
            "search_spec_path": _repo_path(search_spec_path),
            "search_spec_sha256": expected_spec_sha,
        },
        "records": normalized,
    }


def validate_selection_document(selection, repo_root=REPO_ROOT):
    del repo_root
    if selection.get("schema") != SELECTION_SCHEMA:
        raise RegistryError("unsupported REVERIE final selection schema")
    if selection.get("selection_status") != "complete":
        raise RegistryError("REVERIE final selection is incomplete")
    protocol = selection.get("protocol", {})
    expected = {
        "benchmark": "reverie",
        "split": "val_seen",
        "episode_count": EXPECTED_EPISODES,
        "order_seed": EXPECTED_ORDER_SEED,
        "episode_order_sha256": EXPECTED_ORDER_SHA256,
        "primary_metric": "RGSPL",
        "selection_scope": "val_seen_seed0_hyperparameter_selection",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RegistryError("final selection protocol {} mismatch".format(key))
    records = selection.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("final selection setting matrix is incomplete")
    for setting in SETTINGS:
        if set(records[setting]) != set(METHODS):
            raise RegistryError("{} method matrix is incomplete".format(setting))
        for method in METHODS:
            record = records[setting][method]
            label = "{} {}".format(setting, method)
            if record.get("selection_status") != "ready":
                raise RegistryError("{} is not ready".format(label))
            if record.get("supervision_category") != SUPERVISION[method]:
                raise RegistryError("{} supervision mismatch".format(label))
            _validate_parameters(method, record.get("parameters"), label)
            _validate_metrics(record.get("metrics"), label)
            _validate_metrics(record.get("source_metrics"), label + " source")
            if not _is_sha256(record.get("formal_manifest_sha256")):
                raise RegistryError("{} formal digest invalid".format(label))
            if not _is_sha256(record.get("metric_artifact_sha256")):
                raise RegistryError("{} metric artifact digest invalid".format(label))
    return selection


def _load_final_selection(path):
    path = Path(path).resolve()
    selection = validate_selection_document(
        _read_json(path, "REVERIE final selection")
    )
    source_binding = selection.get("source_ledger", {})
    source_path = _resolve(source_binding.get("path", "")).resolve()
    if not source_path.is_file() or _sha256(source_path) != source_binding.get(
        "sha256"
    ):
        raise RegistryError("final selection Source-ledger binding mismatch")
    _, source = load_source_ledger(source_path)
    for setting in SETTINGS:
        for method in METHODS:
            record = selection["records"][setting][method]
            source_record = source["records"][setting]
            if record.get("checkpoint_sha256") != source_record[
                "checkpoint_sha256"
            ]:
                raise RegistryError("{} {} checkpoint binding mismatch".format(
                    setting, method
                ))
            if _canonical(record.get("source_metrics")) != _canonical(
                source_record["metrics"]
            ):
                raise RegistryError("{} {} Source metrics mismatch".format(
                    setting, method
                ))
            formal_path, manifest = _validate_formal(
                record, setting, method, "val_seen",
                source_record["checkpoint_sha256"],
                EXPECTED_ORDER_SHA256,
                source_record["dataset_sha256"],
            )
            if _repo_path(formal_path) != record["formal_manifest_path"]:
                raise RegistryError("{} {} formal path mismatch".format(setting, method))
            if manifest["immutable_identity_sha256"] != record.get(
                "formal_immutable_identity_sha256"
            ):
                raise RegistryError("{} {} identity mismatch".format(setting, method))
            label = "{} {}".format(setting, method)
            _assert_effective_parameters(
                manifest, setting, method, record["parameters"], label
            )
            metric_path, metric_meta = _manifest_artifact(
                manifest, "logs/valid.txt", label
            )
            if metric_meta["sha256"] != record.get("metric_artifact_sha256"):
                raise RegistryError("{} metric artifact binding mismatch".format(label))
            _assert_metrics_equal(
                _parse_reverie_metrics(metric_path, "val_seen", label),
                record["metrics"], label,
            )
    return path, selection, source_path, source


def build_registry(selection_path=DEFAULT_SELECTION):
    selection_path, selection, source_path, source = _load_final_selection(
        selection_path
    )
    records = {}
    for setting in SETTINGS:
        source_record = source["records"][setting]
        source_entry = {
            "method": "source",
            "supervision_category": SUPERVISION["source"],
            "parameters": source_record["parameters"],
            "run_tag": source_record["run_tag"],
            "metrics": source_record["metrics"],
            "delta_vs_source_pp": {key: 0.0 for key in METRICS},
            "checkpoint_sha256": source_record["checkpoint_sha256"],
            "formal_manifest_path": source_record["formal_manifest_path"],
            "formal_manifest_sha256": source_record["formal_manifest_sha256"],
            "formal_immutable_identity_sha256": source_record[
                "immutable_identity_sha256"
            ],
            "metric_artifact_sha256": source_record[
                "metrics_artifact_sha256"
            ],
        }
        records[setting] = {"source": source_entry}
        for method in METHODS:
            selected = selection["records"][setting][method]
            records[setting][method] = {
                key: selected[key] for key in (
                    "supervision_category", "parameters", "run_tag", "metrics",
                    "delta_vs_source_pp", "checkpoint_sha256",
                    "formal_manifest_path", "formal_manifest_sha256",
                    "formal_immutable_identity_sha256", "metric_artifact_sha256",
                )
            }
            records[setting][method]["method"] = method
    registry = {
        "schema": REGISTRY_SCHEMA,
        "registry_status": "complete",
        "protocol_status": "complete_val_seen_seed0_selection",
        "provenance_status": "complete_formal_manifest_identity_verified",
        "publication_status": "selection_registry_not_test_result",
        "benchmark": "reverie",
        "split": "val_seen",
        "protocol": selection["protocol"],
        "source_ledger": {
            "path": _repo_path(source_path), "sha256": _sha256(source_path)
        },
        "selected_winners": {
            "path": _repo_path(selection_path),
            "sha256": _sha256(selection_path),
        },
        "supervision_categories": {
            "source_no_adaptation": {
                "methods": ["source"], "uses_episode_feedback": False
            },
            "unsupervised_tta": {
                "methods": ["tent", "fstta", "eam"],
                "uses_episode_feedback": False,
            },
            "binary_episode_feedback_tta": {
                "methods": ["feedtta", "atena"],
                "uses_episode_feedback": True,
                "feedback": "binary_navigation_success",
            },
        },
        "records": records,
    }
    validate_registry_document(registry)
    return registry


def validate_registry_document(registry):
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise RegistryError("unsupported REVERIE registry schema")
    expected = {
        "registry_status": "complete",
        "protocol_status": "complete_val_seen_seed0_selection",
        "provenance_status": "complete_formal_manifest_identity_verified",
        "publication_status": "selection_registry_not_test_result",
        "benchmark": "reverie",
        "split": "val_seen",
    }
    for key, value in expected.items():
        if registry.get(key) != value:
            raise RegistryError("registry {} mismatch".format(key))
    records = registry.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("registry setting matrix is incomplete")
    for setting in SETTINGS:
        if set(records[setting]) != set(ALL_METHODS):
            raise RegistryError("{} registry matrix is incomplete".format(setting))
        source_metrics = records[setting]["source"].get("metrics")
        _validate_metrics(source_metrics, setting + " source")
        for method in ALL_METHODS:
            record = records[setting][method]
            label = "{} {}".format(setting, method)
            if record.get("method") != method or record.get(
                "supervision_category"
            ) != SUPERVISION[method]:
                raise RegistryError("{} identity/supervision mismatch".format(label))
            _validate_parameters(method, record.get("parameters"), label)
            _validate_metrics(record.get("metrics"), label)
            expected_delta = {
                key: round(
                    float(record["metrics"][key]) - float(source_metrics[key]), 6
                ) for key in METRICS
            }
            if record.get("delta_vs_source_pp") != expected_delta:
                raise RegistryError("{} Source delta mismatch".format(label))
            for key in (
                "checkpoint_sha256", "formal_manifest_sha256",
                "formal_immutable_identity_sha256",
            ):
                if not _is_sha256(record.get(key)):
                    raise RegistryError("{} {} invalid".format(label, key))
            if not _is_sha256(record.get("metric_artifact_sha256")):
                raise RegistryError("{} metric artifact digest invalid".format(label))
    return registry


def _atomic_write(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    os.replace(str(temporary), str(path))


def freeze_from_search(search_selection, output_dir=DEFAULT_OUTPUT_DIR,
                       source_path=DEFAULT_SOURCE,
                       search_spec_path=DEFAULT_SEARCH_SPEC):
    output_dir = Path(output_dir).resolve()
    selected_path = output_dir / "selected_winners.json"
    registry_path = output_dir / "registry.json"
    selection = _normalize_search_selection(
        search_selection, source_path, search_spec_path
    )
    # Compute and validate the complete registry in a private temporary
    # location before replacing either public output.
    selected_bytes = _json_bytes(selection)
    temporary_selection = output_dir / ".selected_winners.validating.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_selection.write_bytes(selected_bytes)
    try:
        registry = build_registry(temporary_selection)
        registry["selected_winners"] = {
            "path": _repo_path(selected_path),
            "sha256": _bytes_sha256(selected_bytes),
        }
        validate_registry_document(registry)
    finally:
        if temporary_selection.exists():
            temporary_selection.unlink()
    _atomic_write(selected_path, selection)
    _atomic_write(registry_path, registry)
    return selected_path, registry_path, registry


def validate_registry(registry_path=DEFAULT_REGISTRY,
                      selection_path=DEFAULT_SELECTION):
    actual = _read_json(registry_path, "REVERIE final registry")
    validate_registry_document(actual)
    expected = build_registry(selection_path)
    if _canonical(actual) != _canonical(expected):
        raise RegistryError("registry differs from pinned winner evidence")
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-search", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-ledger", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--search-spec", type=Path, default=DEFAULT_SEARCH_SPEC)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate:
            registry = validate_registry(args.registry, args.selection)
            print("validated {} records".format(sum(map(len, registry["records"].values()))))
        else:
            if args.from_search is None:
                raise RegistryError("--from-search FINAL_SELECTION.json is required")
            selected, registry_path, registry = freeze_from_search(
                args.from_search, args.output_dir, args.source_ledger,
                args.search_spec,
            )
            print("wrote {} and {} ({} records)".format(
                selected, registry_path, sum(map(len, registry["records"].values()))
            ))
    except RegistryError as error:
        print("REVERIE registry error: {}".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
