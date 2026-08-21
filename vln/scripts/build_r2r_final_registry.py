#!/usr/bin/env python3
"""Build and validate the frozen R2R val_seen seed-0 selection registry.

The builder is intentionally fail-closed.  It writes ``registry.json`` only
after every Source and selected TTA record has a completed, digest-pinned run
manifest whose immutable identity agrees with the frozen selection.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


REGISTRY_SCHEMA = "navtta.vln_r2r_final_registry.v1"
SELECTION_SCHEMA = "navtta.vln_r2r_final_selection.v1"
SOURCE_SCHEMA = "navtta.vln_r2r_reused_source_controls.v1"

DEFAULT_SELECTION = REPO_ROOT / "vln/results/final/r2r/selected_winners.json"
DEFAULT_OUTPUT = REPO_ROOT / "vln/results/final/r2r/registry.json"

SETTINGS = ("duet-r2r", "hamt-r2r", "goat-r2r")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
ALL_METHODS = ("source",) + METHODS
MODEL_FOR_SETTING = {
    "duet-r2r": "duet",
    "hamt-r2r": "hamt",
    "goat-r2r": "goat",
}
SUPERVISION_FOR_METHOD = {
    "source": "source_no_adaptation",
    "tent": "unsupervised_tta",
    "fstta": "unsupervised_tta",
    "eam": "unsupervised_tta",
    "feedtta": "binary_episode_feedback_tta",
    "atena": "binary_episode_feedback_tta",
}
EXPECTED_EPISODES = 1021
EXPECTED_ORDER_SEED = 0
EXPECTED_ORDER_SHA256 = (
    "5692a749af8fa361d4c43e171f6dffb8f0a39d759ab6efc70dcf6f3907a1fb80"
)
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class RegistryError(RuntimeError):
    """A frozen selection or its formal provenance is incomplete/invalid."""


def _read_json(path, label):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError("invalid {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise RegistryError("{} must be a JSON object: {}".format(label, path))
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and HEX_SHA256.fullmatch(value) is not None


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _resolve(repo_root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(repo_root) / path


def _repo_display_path(repo_root, path):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError("{} must be numeric".format(label))
    if not float("-inf") < float(value) < float("inf"):
        raise RegistryError("{} must be finite".format(label))
    return value


def _validate_metrics(metrics, label):
    if not isinstance(metrics, dict) or not {"SR", "SPL"}.issubset(metrics):
        raise RegistryError("{} must contain SR and SPL".format(label))
    for name, value in metrics.items():
        _number(value, "{} metric {}".format(label, name))
    for name in ("SR", "SPL"):
        if not 0.0 <= float(metrics[name]) <= 100.0:
            raise RegistryError("{} {} is outside [0, 100]".format(label, name))


def _validate_method_parameters(method, parameters, label):
    if not isinstance(parameters, dict) or not parameters:
        raise RegistryError("{} parameters must be a non-empty object".format(label))
    if method == "tent" and parameters.get("update_interval") != 1:
        raise RegistryError("{} Tent update_interval must equal one".format(label))
    if method in ("feedtta", "atena") and parameters.get(
        "action_selection"
    ) != "argmax":
        raise RegistryError("{} must use target-native argmax".format(label))


def load_selection(path=DEFAULT_SELECTION):
    """Load and validate the frozen 3x5 TTA selection document."""
    path = Path(path).resolve()
    selection = _read_json(path, "R2R final selection")
    if selection.get("schema") != SELECTION_SCHEMA:
        raise RegistryError("unsupported R2R final selection schema")

    protocol = selection.get("protocol", {})
    expected_protocol = {
        "benchmark": "r2r",
        "split": "val_seen",
        "episode_count": EXPECTED_EPISODES,
        "order_seed": EXPECTED_ORDER_SEED,
        "episode_order_sha256": EXPECTED_ORDER_SHA256,
        "selection_scope": "val_seen_seed0_hyperparameter_selection",
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise RegistryError(
                "R2R final selection protocol {} mismatch".format(key)
            )

    source = selection.get("source_ledger")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise RegistryError("R2R final selection lacks a Source ledger path")
    if not _is_sha256(source.get("sha256")):
        raise RegistryError("R2R final selection has an invalid Source ledger SHA256")

    records = selection.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("R2R final selection must contain exactly three settings")

    pending = []
    for setting in SETTINGS:
        methods = records[setting]
        if not isinstance(methods, dict) or set(methods) != set(METHODS):
            raise RegistryError(
                "{} must contain exactly the five TTA methods".format(setting)
            )
        for method in METHODS:
            label = "{} {}".format(setting, method)
            record = methods[method]
            if not isinstance(record, dict):
                raise RegistryError("{} selection is not an object".format(label))
            expected_supervision = SUPERVISION_FOR_METHOD[method]
            if record.get("supervision_category") != expected_supervision:
                raise RegistryError("{} supervision category mismatch".format(label))
            _validate_method_parameters(method, record.get("parameters"), label)
            status = record.get("selection_status")
            if status == "awaiting_formal_confirmation":
                pending.append(label)
                for field in (
                    "run_tag",
                    "metrics",
                    "formal_manifest_path",
                    "formal_manifest_sha256",
                ):
                    if record.get(field) is not None:
                        raise RegistryError(
                            "{} pending field {} must be null".format(label, field)
                        )
                continue
            if status != "ready":
                raise RegistryError("{} has invalid selection_status".format(label))
            run_tag = record.get("run_tag")
            if not isinstance(run_tag, str) or not run_tag or run_tag.startswith(
                "pending-"
            ):
                raise RegistryError("{} has an invalid run_tag".format(label))
            _validate_metrics(record.get("metrics"), label)
            if not isinstance(record.get("formal_manifest_path"), str):
                raise RegistryError("{} lacks formal_manifest_path".format(label))
            if not _is_sha256(record.get("formal_manifest_sha256")):
                raise RegistryError("{} has invalid formal manifest SHA256".format(label))
    expected_status = "complete" if not pending else "awaiting_formal_confirmation"
    if selection.get("selection_status") != expected_status:
        raise RegistryError(
            "selection_status must be {!r} for {} pending cells".format(
                expected_status, len(pending)
            )
        )
    return path, selection, pending


def _select_manifest_path(repo_root, record, label):
    expected_sha256 = record.get("formal_manifest_sha256")
    relative = record.get("formal_manifest_path")
    path = _resolve(repo_root, relative).resolve()
    canonical_root = (Path(repo_root) / "vln/results/runs").resolve()
    try:
        path.relative_to(canonical_root)
    except ValueError:
        raise RegistryError(
            "{} formal manifest is outside canonical vln/results/runs: {}".format(
                label, relative
            )
        )
    if path.name != "manifest.json":
        raise RegistryError("{} canonical manifest filename is invalid".format(label))
    if not path.is_file():
        raise RegistryError(
            "{} canonical formal manifest is missing: {}".format(label, relative)
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise RegistryError(
            "{} canonical formal manifest SHA256 mismatch: {} != {}".format(
                label, actual_sha256, expected_sha256
            )
        )
    return path


def _validate_artifact_metadata(manifest, label):
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RegistryError("{} formal manifest has no result artifacts".format(label))
    names = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RegistryError("{} has malformed result artifact metadata".format(label))
        name = artifact.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise RegistryError("{} has duplicate/missing artifact names".format(label))
        names.add(name)
        if not isinstance(artifact.get("path"), str) or not artifact["path"]:
            raise RegistryError("{} artifact {} lacks a path".format(label, name))
        normalized_path = artifact["path"].replace("\\", "/")
        if normalized_path != name and not normalized_path.endswith("/" + name):
            raise RegistryError(
                "{} artifact {} path/name mismatch".format(label, name)
            )
        size = artifact.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RegistryError("{} artifact {} has invalid size".format(label, name))
        if not _is_sha256(artifact.get("sha256")):
            raise RegistryError("{} artifact {} has invalid SHA256".format(label, name))


def _validate_formal_manifest(
    repo_root,
    record,
    setting,
    method,
    run_tag,
    checkpoint_sha256,
):
    label = "{} {}".format(setting, method)
    path = _select_manifest_path(repo_root, record, label)
    manifest = _read_json(path, "{} formal manifest".format(label))
    expected = {
        "task": "vln",
        "model": MODEL_FOR_SETTING[setting],
        "method": method,
        "run_tag": run_tag,
        "seed": EXPECTED_ORDER_SEED,
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RegistryError(
                "{} formal manifest {} mismatch: {!r} != {!r}".format(
                    label, key, manifest.get(key), value
                )
            )
    source_parts = str(manifest.get("source_setting", "")).split(":")
    if (
        len(source_parts) != 4
        or source_parts[0] != setting
        or source_parts[1] != "val_seen"
        or source_parts[3] != method
    ):
        raise RegistryError("{} formal source_setting mismatch".format(label))
    expected_run_id = "{}-{}-val_seen-{}".format(
        run_tag, setting, source_parts[2]
    )
    if manifest.get("run_id") != expected_run_id:
        raise RegistryError("{} formal run_id mismatch".format(label))
    if path.parent.name != expected_run_id:
        raise RegistryError("{} canonical manifest directory mismatch".format(label))
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get(
        "sha256"
    ) != checkpoint_sha256:
        raise RegistryError("{} checkpoint SHA256 mismatch".format(label))
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get(
        "stream_order_sha256"
    ) != EXPECTED_ORDER_SHA256:
        raise RegistryError("{} episode-order SHA256 mismatch".format(label))
    identity = manifest.get("immutable_identity_sha256")
    if not _is_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise RegistryError("{} immutable manifest identity mismatch".format(label))
    _validate_artifact_metadata(manifest, label)
    return path, manifest


def _load_source_ledger(repo_root, selection):
    binding = selection["source_ledger"]
    path = _resolve(repo_root, binding["path"])
    if not path.is_file():
        raise RegistryError("missing R2R Source ledger: {}".format(path))
    actual_sha256 = _sha256(path)
    if actual_sha256 != binding["sha256"]:
        raise RegistryError(
            "R2R Source ledger SHA256 mismatch: {} != {}".format(
                actual_sha256, binding["sha256"]
            )
        )
    ledger = _read_json(path, "R2R Source ledger")
    if ledger.get("schema") != SOURCE_SCHEMA:
        raise RegistryError("unsupported R2R Source ledger schema")
    expected = {
        "benchmark": "r2r",
        "split": "val_seen",
        "source_protocol": "standard_argmax",
        "episode_count": EXPECTED_EPISODES,
        "order_seed": EXPECTED_ORDER_SEED,
        "episode_order_sha256": EXPECTED_ORDER_SHA256,
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            raise RegistryError("R2R Source ledger {} mismatch".format(key))
    records = ledger.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("R2R Source ledger must contain exactly three settings")
    return path, ledger


def _record(
    repo_root,
    setting,
    method,
    selection_record,
    source_record,
    source_metrics,
):
    run_tag = selection_record["run_tag"]
    _, manifest = _validate_formal_manifest(
        repo_root,
        selection_record,
        setting,
        method,
        run_tag,
        source_record["checkpoint_sha256"],
    )
    metrics = selection_record["metrics"]
    delta = {
        name: round(float(metrics[name]) - float(source_metrics[name]), 8)
        for name in ("SR", "SPL")
    }
    result = {
        "method": method,
        "supervision_category": SUPERVISION_FOR_METHOD[method],
        "parameters": selection_record["parameters"],
        "run_tag": run_tag,
        "metrics": metrics,
        "delta_vs_source_pp": delta,
        "checkpoint_sha256": source_record["checkpoint_sha256"],
        "formal_manifest_path": selection_record["formal_manifest_path"],
        "formal_manifest_sha256": selection_record["formal_manifest_sha256"],
        "formal_immutable_identity_sha256": manifest[
            "immutable_identity_sha256"
        ],
    }
    return result


def _source_record(repo_root, setting, source):
    label = "{} source".format(setting)
    parameters = source.get("parameters")
    if parameters != {"action_selection": "argmax", "action_seed": 0}:
        raise RegistryError("{} parameters are not standard argmax".format(label))
    _validate_metrics(source.get("metrics"), label)
    _, manifest = _validate_formal_manifest(
        repo_root,
        source,
        setting,
        "source",
        source.get("run_tag"),
        source.get("checkpoint_sha256"),
    )
    result = {
        "method": "source",
        "supervision_category": SUPERVISION_FOR_METHOD["source"],
        "parameters": parameters,
        "run_tag": source["run_tag"],
        "metrics": source["metrics"],
        "delta_vs_source_pp": {"SR": 0.0, "SPL": 0.0},
        "checkpoint_sha256": source["checkpoint_sha256"],
        "formal_manifest_path": source["formal_manifest_path"],
        "formal_manifest_sha256": source["formal_manifest_sha256"],
        "formal_immutable_identity_sha256": manifest[
            "immutable_identity_sha256"
        ],
    }
    return result


def build_registry(selection_path=DEFAULT_SELECTION, repo_root=REPO_ROOT):
    """Return a complete deterministic registry or raise before any write."""
    repo_root = Path(repo_root).resolve()
    selection_path, selection, pending = load_selection(selection_path)
    if pending:
        raise RegistryError(
            "R2R final registry is not ready; awaiting {} formal confirmations: {}".format(
                len(pending), ", ".join(pending)
            )
        )
    source_path, source_ledger = _load_source_ledger(repo_root, selection)

    records = {}
    for setting in SETTINGS:
        source = source_ledger["records"][setting]
        source_entry = _source_record(repo_root, setting, source)
        setting_records = {"source": source_entry}
        for method in METHODS:
            setting_records[method] = _record(
                repo_root,
                setting,
                method,
                selection["records"][setting][method],
                source,
                source_entry["metrics"],
            )
        records[setting] = setting_records

    registry = {
        "schema": REGISTRY_SCHEMA,
        "registry_status": "complete",
        "protocol_status": "complete_val_seen_seed0_selection",
        "provenance_status": "complete_formal_manifest_identity_verified",
        "publication_status": "selection_registry_not_publication_final",
        "benchmark": "r2r",
        "split": "val_seen",
        "protocol": {
            "episode_count": EXPECTED_EPISODES,
            "order_seed": EXPECTED_ORDER_SEED,
            "episode_order_sha256": EXPECTED_ORDER_SHA256,
            "source_protocol": "standard_argmax",
            "metric_unit": "percentage_points",
            "selection_scope": "val_seen_seed0_hyperparameter_selection",
            "qualification": (
                "Frozen model-method selection evidence; independent split/order "
                "robustness is not established."
            ),
        },
        "source_ledger": {
            "path": _repo_display_path(repo_root, source_path),
            "sha256": _sha256(source_path),
        },
        "selected_winners": {
            "path": _repo_display_path(repo_root, selection_path),
            "sha256": _sha256(selection_path),
        },
        "supervision_categories": {
            "source_no_adaptation": {
                "methods": ["source"],
                "uses_episode_feedback": False,
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
    """Validate the public registry schema without consulting source files."""
    if not isinstance(registry, dict) or registry.get("schema") != REGISTRY_SCHEMA:
        raise RegistryError("unsupported R2R final registry schema")
    expected_status = {
        "registry_status": "complete",
        "protocol_status": "complete_val_seen_seed0_selection",
        "provenance_status": "complete_formal_manifest_identity_verified",
        "publication_status": "selection_registry_not_publication_final",
        "benchmark": "r2r",
        "split": "val_seen",
    }
    for key, value in expected_status.items():
        if registry.get(key) != value:
            raise RegistryError("R2R final registry {} mismatch".format(key))
    protocol = registry.get("protocol", {})
    if (
        protocol.get("episode_count") != EXPECTED_EPISODES
        or protocol.get("order_seed") != EXPECTED_ORDER_SEED
        or protocol.get("episode_order_sha256") != EXPECTED_ORDER_SHA256
    ):
        raise RegistryError("R2R final registry protocol identity mismatch")
    records = registry.get("records")
    if not isinstance(records, dict) or set(records) != set(SETTINGS):
        raise RegistryError("R2R final registry setting matrix is incomplete")
    for setting in SETTINGS:
        methods = records[setting]
        if not isinstance(methods, dict) or set(methods) != set(ALL_METHODS):
            raise RegistryError("{} registry method matrix is incomplete".format(setting))
        source_metrics = methods["source"].get("metrics")
        _validate_metrics(source_metrics, "{} source".format(setting))
        for method in ALL_METHODS:
            label = "{} {}".format(setting, method)
            record = methods[method]
            if not isinstance(record, dict):
                raise RegistryError("{} registry record is invalid".format(label))
            for field in (
                "parameters",
                "run_tag",
                "metrics",
                "formal_manifest_path",
                "formal_manifest_sha256",
            ):
                if field not in record:
                    raise RegistryError("{} lacks {}".format(label, field))
            _validate_method_parameters(method, record["parameters"], label)
            _validate_metrics(record["metrics"], label)
            if record.get("supervision_category") != SUPERVISION_FOR_METHOD[method]:
                raise RegistryError("{} supervision category mismatch".format(label))
            if not isinstance(record["run_tag"], str) or not record["run_tag"]:
                raise RegistryError("{} run_tag is invalid".format(label))
            if not isinstance(record["formal_manifest_path"], str):
                raise RegistryError("{} formal manifest path is invalid".format(label))
            if not _is_sha256(record["formal_manifest_sha256"]):
                raise RegistryError("{} formal manifest SHA256 is invalid".format(label))
            expected_delta = {
                name: round(
                    float(record["metrics"][name]) - float(source_metrics[name]), 8
                )
                for name in ("SR", "SPL")
            }
            if record.get("delta_vs_source_pp") != expected_delta:
                raise RegistryError("{} source delta mismatch".format(label))
    return registry


def validate_registry(
    registry_path=DEFAULT_OUTPUT,
    selection_path=DEFAULT_SELECTION,
    repo_root=REPO_ROOT,
):
    """Rebuild from pinned evidence and require byte-semantic equality."""
    registry_path = Path(registry_path).resolve()
    actual = _read_json(registry_path, "R2R final registry")
    validate_registry_document(actual)
    expected = build_registry(selection_path=selection_path, repo_root=repo_root)
    if _canonical(actual) != _canonical(expected):
        raise RegistryError(
            "R2R final registry differs from its pinned Source/selection evidence"
        )
    return actual


def write_registry(
    output_path=DEFAULT_OUTPUT,
    selection_path=DEFAULT_SELECTION,
    repo_root=REPO_ROOT,
):
    """Atomically write only a fully built and validated registry."""
    registry = build_registry(selection_path=selection_path, repo_root=repo_root)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                registry,
                stream,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(str(temporary), str(output_path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path, registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate an existing registry against all pinned evidence",
    )
    args = parser.parse_args(argv)
    try:
        if args.validate:
            registry = validate_registry(args.output, args.selection, REPO_ROOT)
            path = Path(args.output).resolve()
        else:
            path, registry = write_registry(args.output, args.selection, REPO_ROOT)
    except RegistryError as error:
        print("R2R final registry error: {}".format(error), file=sys.stderr)
        return 2
    print("{} ({} records)".format(path, sum(map(len, registry["records"].values()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
