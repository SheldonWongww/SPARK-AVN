#!/usr/bin/env python3
"""Select stable R2R TTA parameters, freeze them, then evaluate val_unseen.

Selection is restricted to shuffled ``val_seen`` streams.  Source is never
executed: its metrics and checkpoint identity are read from authenticated
ledgers.  A frozen choice is the only input allowed to the ``val_unseen``
stage, which prevents accidental test-split feedback into hyperparameter
selection.
"""

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import run_r2r_val_unseen_frozen_eval as frozen_eval  # noqa: E402
import run_tta_hparam_search as hparam_runner  # noqa: E402
from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


ORIGINAL_HPARAM_PARSE_METRICS = hparam_runner.parse_metrics


SCHEMA = "navtta.vln_r2r_cross_split_robust_eval.v1"
PLAN_SCHEMA = "navtta.vln_r2r_cross_split_robust_plan.v1"
PHASE_PLAN_SCHEMA = "navtta.vln_r2r_cross_split_phase_plan.v1"
SUMMARY_SCHEMA = "navtta.vln_r2r_cross_split_summary.v1"
FROZEN_SCHEMA = "navtta.vln_r2r_cross_split_frozen.v1"

DEFAULT_SPEC = (
    REPO_ROOT / "vln/experiments/r2r_cross_split_robust_eval_v1.json"
)
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
LOG_ROOT = REPO_ROOT / "vln/results/logs/r2r/cross_split_robust_eval"
TUNING_ROOT = REPO_ROOT / "vln/results/tuning/r2r/cross_split_robust_eval"
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"

SETTINGS = ("duet-r2r", "hamt-r2r", "goat-r2r")
MODELS = ("duet", "hamt", "goat")
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
MODEL_FOR_SETTING = dict(zip(SETTINGS, MODELS))
ORDER_FAMILY = {
    "duet-r2r": "r2r_duet_hamt",
    "hamt-r2r": "r2r_duet_hamt",
    "goat-r2r": "r2r_goat",
}
EXPECTED_BENCHMARK = {
    "duet-r2r": "r2r_discrete_duet_hamt",
    "hamt-r2r": "r2r_discrete_duet_hamt",
    "goat-r2r": "r2r_discrete_goat",
}
EXECUTION_SURFACE = (
    "core", "tools", "vln/baselines", "vln/navtta_vln",
    "vln/scripts", "vln/experiments", "vln/manifests",
)
METRIC_RE = re.compile(r"\b(sr|spl):\s*(-?[0-9]+(?:\.[0-9]+)?)", re.I)


class UserError(RuntimeError):
    pass


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("cannot read JSON {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise UserError("JSON document must be an object: {}".format(path))
    return value


def _repo_file(value, label):
    path = Path(value)
    path = path if path.is_absolute() else REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        raise UserError("{} escapes the repository: {}".format(label, path))
    if not path.is_file():
        raise UserError("missing {}: {}".format(label, path))
    return path


def _valid_sha256(value):
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None


def _finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _order_manifest_path(setting, split, order_seed):
    family = ORDER_FAMILY[setting]
    return (
        REPO_ROOT / "vln/manifests/episode_order"
        / "order_seed_{}".format(order_seed) / family
        / "{}.json".format(split)
    )


def _parse_source_metrics(record, split, setting):
    if record.get("metrics_artifact_path") is not None:
        path = _repo_file(
            record["metrics_artifact_path"],
            "{} {} Source metric artifact".format(split, setting),
        )
        expected_sha = record.get("metrics_artifact_sha256")
        if not _valid_sha256(expected_sha) or _sha256(path) != expected_sha:
            raise UserError("{} {} Source metric artifact SHA256 mismatch".format(split, setting))
        matches = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Env name: {}".format(split) not in line:
                continue
            values = {key.upper(): float(value) for key, value in METRIC_RE.findall(line)}
            if set(values) == {"SR", "SPL"}:
                matches.append(values)
        if len(matches) != 1:
            raise UserError("{} {} Source metric artifact is ambiguous".format(split, setting))
        return path, matches[0]

    path = _repo_file(
        record.get("metrics_json_path", ""),
        "{} {} Source metrics JSON".format(split, setting),
    )
    expected_sha = record.get("metrics_json_sha256")
    if not _valid_sha256(expected_sha) or _sha256(path) != expected_sha:
        raise UserError("{} {} Source metrics JSON SHA256 mismatch".format(split, setting))
    result = _read_json(path)
    if (
        result.get("setting") != setting
        or result.get("config_method") != "source"
        or result.get("run_tag") != record.get("run_tag")
    ):
        raise UserError("{} {} Source metrics JSON identity mismatch".format(split, setting))
    metrics = result.get("metrics", {})
    if any(not _finite_number(metrics.get(key)) for key in ("SR", "SPL")):
        raise UserError("{} {} Source metrics JSON is incomplete".format(split, setting))
    return path, {key: float(metrics[key]) for key in ("SR", "SPL")}


def _artifact_matches_local(manifest, local_path):
    local_path = Path(local_path).resolve()
    try:
        relative = local_path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return False
    for artifact in manifest.get("result_artifacts", []):
        if not isinstance(artifact, dict):
            continue
        artifact_path = str(artifact.get("path", "")).replace("\\", "/")
        if (
            artifact_path.endswith("/" + relative)
            and artifact.get("size") == local_path.stat().st_size
            and artifact.get("sha256") == _sha256(local_path)
        ):
            return True
    return False


def _validate_source_record(document, split, setting, record):
    if not isinstance(record, dict):
        raise UserError("invalid {} Source record for {}".format(split, setting))
    metrics = record.get("metrics", {})
    if (
        record.get("model") != MODEL_FOR_SETTING[setting]
        or record.get("parameters") != {
            "action_selection": "argmax", "action_seed": 0
        }
        or not _valid_sha256(record.get("checkpoint_sha256"))
        or any(not _finite_number(metrics.get(key)) for key in ("SR", "SPL"))
    ):
        raise UserError("invalid {} Source record for {}".format(split, setting))

    formal_path = _repo_file(
        record.get("formal_manifest_path", ""),
        "{} {} Source formal manifest".format(split, setting),
    )
    try:
        formal_path.relative_to(FORMAL_ROOT.resolve())
    except ValueError:
        raise UserError("Source formal manifest is outside vln/results/runs")
    expected_formal_sha = record.get("formal_manifest_sha256")
    if (
        not _valid_sha256(expected_formal_sha)
        or _sha256(formal_path) != expected_formal_sha
    ):
        raise UserError("{} {} Source formal manifest SHA256 mismatch".format(split, setting))
    evidence_copy = record.get("formal_manifest_evidence_copy_path")
    if evidence_copy is not None:
        # This ignored raw-log copy is a convenience when a historical batch
        # is still present.  The canonical tracked formal manifest above is
        # the authority, so a clean Git checkout must not require the copy.
        copy_path = (REPO_ROOT / evidence_copy).resolve()
        if copy_path.is_file() and _sha256(copy_path) != expected_formal_sha:
            raise UserError("Source formal-manifest evidence copy mismatch")

    manifest = _read_json(formal_path)
    run_tag = record.get("run_tag")
    expected_run_id = "{}-{}-{}-native".format(run_tag, setting, split)
    source_settings = {
        "{}:{}:native".format(setting, split),
        "{}:{}:native:source".format(setting, split),
    }
    expected = {
        "run_id": expected_run_id,
        "task": "vln",
        "benchmark": EXPECTED_BENCHMARK[setting],
        "model": MODEL_FOR_SETTING[setting],
        "method": "source",
        "run_tag": run_tag,
        "seed": 0,
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise UserError("{} {} Source formal manifest {} mismatch".format(split, setting, key))
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("git_commit", ""))):
        raise UserError("{} {} Source formal manifest has invalid Git commit".format(split, setting))
    if manifest.get("source_setting") not in source_settings:
        raise UserError("{} {} Source formal manifest split mismatch".format(split, setting))
    if manifest.get("checkpoint", {}).get("sha256") != record["checkpoint_sha256"]:
        raise UserError("{} {} Source checkpoint mismatch".format(split, setting))
    identity = manifest.get("immutable_identity_sha256")
    if not _valid_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise UserError("{} {} Source immutable identity mismatch".format(split, setting))
    if record.get("immutable_identity_sha256", identity) != identity:
        raise UserError("{} {} Source ledger immutable identity mismatch".format(split, setting))
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts or any(
        not isinstance(artifact, dict)
        or not _valid_sha256(artifact.get("sha256"))
        or type(artifact.get("size")) is not int
        or artifact["size"] < 0
        for artifact in artifacts
    ):
        raise UserError("{} {} Source formal artifacts are malformed".format(split, setting))
    dataset = manifest.get("dataset", {})
    if dataset.get("stream_order_sha256") != document.get("episode_order_sha256"):
        raise UserError("{} {} Source stream order mismatch".format(split, setting))
    if record.get("dataset_sha256", dataset.get("stream_content_sha256")) != dataset.get(
        "stream_content_sha256"
    ):
        raise UserError("{} {} Source dataset mismatch".format(split, setting))

    metric_path, parsed_metrics = _parse_source_metrics(record, split, setting)
    for key in ("SR", "SPL"):
        if float(metrics[key]) != float(parsed_metrics[key]):
            raise UserError("{} {} Source {} metric mismatch".format(split, setting, key))
    # Newer Source ledgers preserve the original evaluator artifact.  Require
    # it to be one of the artifacts authenticated by the formal manifest.
    if record.get("metrics_artifact_path") is not None and not _artifact_matches_local(
        manifest, metric_path
    ):
        raise UserError("Source metric artifact is absent from the formal manifest")
    parameters_path_value = record.get("parameters_json_path")
    if parameters_path_value is not None:
        parameters_path = _repo_file(
            parameters_path_value, "{} {} Source parameters".format(split, setting)
        )
        if _sha256(parameters_path) != record.get("parameters_json_sha256"):
            raise UserError("Source parameter evidence SHA256 mismatch")
        parameters_document = _read_json(parameters_path)
        if (
            parameters_document.get("method") != "source"
            or parameters_document.get("parameters") != record["parameters"]
            or not str(manifest.get("config", "")).replace("\\", "/").endswith(
                "/" + parameters_path_value.replace("\\", "/")
            )
        ):
            raise UserError("Source parameter evidence is not bound to the formal run")
    elif not str(manifest.get("config", "")).startswith(
        "vln/scripts/run_source_eval.sh#"
    ):
        raise UserError("legacy Source formal run lacks an immutable runner anchor")
    return {
        "metrics": {key: float(metrics[key]) for key in ("SR", "SPL")},
        "checkpoint_sha256": record["checkpoint_sha256"],
        "ledger_formal_manifest_path": str(formal_path),
        "ledger_formal_manifest_sha256": expected_formal_sha,
        "ledger_formal_immutable_identity_sha256": identity,
        "metric_artifact_path": str(metric_path),
        "metric_artifact_sha256": _sha256(metric_path),
    }


def _load_source_ledgers(spec):
    records = {}
    for split in ("val_seen", "val_unseen"):
        binding = spec["source_controls"][split]
        if not isinstance(binding, dict) or not {"path", "sha256"}.issubset(binding):
            raise UserError("source_controls.{} must pin path and sha256".format(split))
        if binding.get("execution", "reuse_only_order_invariant_argmax_aggregate") != (
            "reuse_only_order_invariant_argmax_aggregate"
        ):
            raise UserError("{} Source execution policy is not reuse-only".format(split))
        path = _repo_file(binding["path"], "{} Source ledger".format(split))
        if not _valid_sha256(binding["sha256"]) or _sha256(path) != binding["sha256"]:
            raise UserError("{} Source ledger SHA256 mismatch".format(split))
        document = _read_json(path)
        if document.get("benchmark") != "r2r" or document.get("split") != split:
            raise UserError("{} Source ledger benchmark/split mismatch".format(split))
        if document.get("source_protocol") != "standard_argmax":
            raise UserError("{} Source ledger is not standard argmax".format(split))
        ledger_seed = document.get(
            "order_seed", document.get("canonical_order_seed")
        )
        if type(ledger_seed) is not int or ledger_seed != 0:
            raise UserError("{} Source ledger must be seed 0".format(split))
        expected_count = spec["protocol"]["episodes_by_split"][split]
        if document.get("episode_count") != expected_count:
            raise UserError("{} Source ledger episode count mismatch".format(split))
        ledger_records = document.get("records")
        if not isinstance(ledger_records, dict) or set(ledger_records) != set(SETTINGS):
            raise UserError("{} Source ledger must contain all R2R settings".format(split))
        records[split] = {}
        for setting in SETTINGS:
            record = ledger_records[setting]
            records[split][setting] = _validate_source_record(
                document, split, setting, record
            )
            records[split][setting].update({
                "ledger_path": str(path), "ledger_sha256": binding["sha256"]
            })
    return records


def _load_order_bindings(spec):
    bindings = {}
    protocol = spec["protocol"]
    for split, seed_key in (
        ("val_seen", "selection_order_seeds"),
        ("val_unseen", "evaluation_order_seeds"),
    ):
        bindings[split] = {}
        for seed in protocol[seed_key]:
            bindings[split][seed] = {}
            for setting in SETTINGS:
                path = _order_manifest_path(setting, split, seed)
                document = _read_json(path)
                _validate_order_document(
                    document, setting, split, seed,
                    protocol["episodes_by_split"][split],
                )
                dataset = document["dataset"]
                bindings[split][seed][setting] = {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "benchmark": document["benchmark"],
                    "episode_count": document["episode_count"],
                    "order_sha256": document["order_sha256"],
                    "dataset_sha256": dataset["sha256"],
                }
            _validate_paired_order_hashes(bindings[split][seed], split, seed)
    return bindings


def _validate_order_document(document, setting, split, seed, episode_count):
    dataset = document.get("dataset", {})
    expected = {
        "schema": "navtta.episode_order.v1",
        "benchmark": EXPECTED_BENCHMARK[setting],
        "split": split,
        "episode_count": episode_count,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise UserError(
                "order manifest {} mismatch for {} seed {}".format(
                    key, setting, seed
                )
            )
    if type(document.get("order_seed")) is not int or document["order_seed"] != seed:
        raise UserError(
            "order manifest order_seed mismatch for {} seed {}".format(
                setting, seed
            )
        )
    if not _valid_sha256(document.get("order_sha256")):
        raise UserError("order manifest has invalid stream digest")
    if not _valid_sha256(dataset.get("sha256")):
        raise UserError("order manifest has invalid dataset digest")


def _validate_paired_order_hashes(bindings, split, seed):
    paired_hashes = {
        bindings[setting]["order_sha256"] for setting in SETTINGS
    }
    if len(paired_hashes) != 1:
        raise UserError(
            "DUET/HAMT and GOAT order hashes differ for {} seed {}".format(
                split, seed
            )
        )


def _validate_candidate(method, candidate, seen_ids):
    if not isinstance(candidate, dict) or set(candidate) != {
        "candidate_id", "parameters"
    }:
        raise UserError(
            "{} candidates require exactly candidate_id and parameters".format(method)
        )
    candidate_id = candidate["candidate_id"]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(candidate_id or "")):
        raise UserError("invalid candidate_id for {}".format(method))
    if candidate_id in seen_ids:
        raise UserError("duplicate {} candidate_id {}".format(method, candidate_id))
    seen_ids.add(candidate_id)
    parameters = candidate["parameters"]
    if not isinstance(parameters, dict):
        raise UserError("{} candidate parameters must be an object".format(method))
    # ``run_tta_hparam_search`` imports translator functions, not its module;
    # use the public key tables through the already importable sibling module.
    import tta_config_cli
    unknown = set(parameters).difference(
        tta_config_cli.COMMON | tta_config_cli.METHOD_KEYS[method]
    )
    if unknown:
        raise UserError(
            "unknown {} candidate parameters: {}".format(
                method, ", ".join(sorted(unknown))
            )
        )
    if any(not math.isfinite(value) for value in _float_leaves(parameters)):
        raise UserError("{} candidate contains a non-finite number".format(method))
    if parameters.get("episodic") is not False:
        raise UserError("{} candidates must use continual adaptation".format(method))
    if method == "tent" and parameters.get("update_interval") != 1:
        raise UserError("Tent update_interval is fixed at 1")
    if method == "fstta" and (
        parameters.get("m") != 3 or parameters.get("n") != 4
    ):
        raise UserError("FSTTA M/N are fixed at 3/4 in this campaign")
    if method == "eam" and (
        type(parameters.get("batch_size")) is not int
        or parameters["batch_size"] < 1
    ):
        raise UserError("EAM requires a positive integer replay batch_size")
    if method in ("feedtta", "atena") and parameters.get(
        "action_selection"
    ) != "argmax":
        raise UserError("{} must preserve target-native argmax".format(method))


def _float_leaves(value):
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_float_leaves(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_float_leaves(item))
        return result
    return []


def _worker_cap(spec, model, method):
    value = spec["execution"]["max_workers_by_model"][model]
    if isinstance(value, dict):
        value = value.get(method)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UserError("missing positive worker cap for {} {}".format(model, method))
    return value


def load_spec(path=DEFAULT_SPEC):
    path = Path(path).resolve()
    spec = _read_json(path)
    if spec.get("schema") != SCHEMA:
        raise UserError("unsupported R2R cross-split robust-eval schema")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", str(spec.get("experiment_id", ""))):
        raise UserError("invalid experiment_id")
    protocol = spec.get("protocol")
    if not isinstance(protocol, dict):
        raise UserError("missing protocol")
    required = {
        "selection_split": "val_seen",
        "evaluation_split": "val_unseen",
        "selection_order_seeds": [1, 2, 3],
        "evaluation_order_seeds": [1, 2, 3],
        "source_execution_jobs": 0,
        "no_val_unseen_selection": True,
        "freeze_before_evaluation": True,
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise UserError("protocol {} must equal {!r}".format(key, value))
    if protocol.get("episodes_by_split") != {
        "val_seen": 1021, "val_unseen": 2349
    }:
        raise UserError("protocol episodes_by_split must pin full R2R splits")
    for key in ("selection_order_seeds", "evaluation_order_seeds"):
        if any(type(seed) is not int for seed in protocol[key]):
            raise UserError("{} must contain exact integers".format(key))

    matrix = spec.get("matrix")
    if not isinstance(matrix, dict):
        raise UserError("missing matrix")
    if tuple(matrix.get("setting_order", ())) != SETTINGS:
        raise UserError("matrix.setting_order mismatch")
    if tuple(matrix.get("method_order", ())) != METHODS:
        raise UserError("matrix.method_order mismatch")
    candidates = matrix.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != set(SETTINGS):
        raise UserError("candidate matrix must contain exactly three R2R settings")
    for setting in SETTINGS:
        methods = candidates[setting]
        if not isinstance(methods, dict) or set(methods) != set(METHODS):
            raise UserError("{} candidate matrix must contain five TTA methods".format(setting))
        for method in METHODS:
            values = methods[method]
            if not isinstance(values, list) or not values:
                raise UserError("{} {} requires at least one candidate".format(setting, method))
            seen_ids = set()
            for candidate in values:
                _validate_candidate(method, candidate, seen_ids)

    execution = spec.get("execution")
    if not isinstance(execution, dict):
        raise UserError("missing execution policy")
    for key in (
        "formal_requires_clean_tracked_tree", "strict_model_barrier",
        "strict_method_barrier",
    ):
        if execution.get(key) is not True:
            raise UserError("execution.{} must be true".format(key))
    caps = execution.get("max_workers_by_model")
    if not isinstance(caps, dict) or set(caps) != set(MODELS):
        raise UserError("max_workers_by_model must cover duet, hamt, and goat")
    for model in MODELS:
        for method in METHODS:
            _worker_cap(spec, model, method)

    gates = spec.get("implementation_gates")
    required_gates = {
        "duet_atena_self_prediction": (
            "concat_global_and_local_cls_matching_official_2D_representation"
        ),
        "eam_warmup": "no_optimizer_update_until_reservoir_reaches_batch_size",
        "feedtta_action_protocol": "target_native_argmax",
        "atena_feedback": "lazy_submitted_trajectory_evaluator_query",
        "fstta_variance_history": "test_stream",
        "nonfinite_metrics_forbidden": True,
        "complete_diagnostics_required": True,
    }
    if not isinstance(gates, dict) or any(
        gates.get(key) != value for key, value in required_gates.items()
    ):
        raise UserError("implementation_gates do not match the reviewed protocol")

    candidate_count = sum(
        len(candidates[setting][method])
        for setting in SETTINGS for method in METHODS
    )
    expected_budget = {
        "candidate_configurations": candidate_count,
        "selection_tta_jobs": candidate_count * 3,
        "evaluation_tta_jobs": len(SETTINGS) * len(METHODS) * 3,
        "source_execution_jobs": 0,
        "total_tta_jobs": candidate_count * 3 + len(SETTINGS) * len(METHODS) * 3,
    }
    if "budget" in spec and spec["budget"] != expected_budget:
        raise UserError("budget does not match the registered candidate matrix")

    source_records = _load_source_ledgers(spec)
    order_bindings = _load_order_bindings(spec)
    spec["_spec_path"] = str(path)
    spec["_source_records"] = source_records
    spec["_order_bindings"] = order_bindings
    return spec


def _parameters_for_seed(method, parameters, order_seed):
    result = copy.deepcopy(parameters)
    if method == "feedtta":
        result["action_seed"] = order_seed
        result["sgr_seed"] = order_seed
    return result


def _slug(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-")


def _job_tag(batch_id, campaign_stage, setting, method, candidate_id, seed):
    identity = {
        "stage": campaign_stage,
        "setting": setting,
        "method": method,
        "candidate_id": candidate_id,
        "order_seed": seed,
    }
    return "{}-{}-{}-{}-{}-s{}-{}".format(
        batch_id, campaign_stage, MODEL_FOR_SETTING[setting], method,
        _slug(candidate_id), seed, _json_sha256(identity)[:10],
    )


def _make_job(
    spec, batch_id, batch_root, gpu, campaign_stage, phase_index, ordinal,
    setting, method, candidate_id, parameters, order_seed,
):
    split = (
        spec["protocol"]["selection_split"]
        if campaign_stage == "selection"
        else spec["protocol"]["evaluation_split"]
    )
    base_run_tag = _job_tag(
        batch_id, campaign_stage, setting, method, candidate_id, order_seed
    )
    phase_name = "{:02d}-{}-{}".format(
        phase_index, MODEL_FOR_SETTING[setting], method
    )
    phase_dir = Path(batch_root) / campaign_stage / "phases" / phase_name
    job_dir = phase_dir / "jobs" / setting / base_run_tag
    config_path = job_dir / "parameters.json"
    result_parent = (
        TUNING_ROOT / batch_id / campaign_stage / MODEL_FOR_SETTING[setting]
        / method / "jobs"
    )
    result_root = result_parent / base_run_tag / split
    order = spec["_order_bindings"][split][order_seed][setting]
    source = spec["_source_records"][split][setting]
    seeded_parameters = _parameters_for_seed(method, parameters, order_seed)
    command = [
        str(RUNNER), setting, split, str(gpu),
        "--run-tag", base_run_tag,
        "--tta-config", str(config_path),
        "--result-root", str(result_root),
        "--order-seed", str(order_seed),
    ]
    formal_manifest = (
        FORMAL_ROOT
        / "{}-{}-{}-native".format(base_run_tag, setting, split)
        / "manifest.json"
    )
    return {
        "batch_id": batch_id,
        "gpu": int(gpu),
        "ordinal": ordinal,
        "phase_index": phase_index,
        "phase_name": phase_name,
        "base_run_tag": base_run_tag,
        "run_tag": base_run_tag,
        "attempt": 0,
        "campaign_stage": campaign_stage,
        "split": split,
        "setting": setting,
        "model": MODEL_FOR_SETTING[setting],
        "family": "discrete",
        "benchmark": "r2r",
        "method": method,
        "search_method": method,
        "config_method": method,
        "candidate_id": candidate_id,
        "candidate_parameters": copy.deepcopy(parameters),
        "parameters": seeded_parameters,
        "stage": "orders",
        "config_stage": "orders",
        "episodes": -1,
        "order_seed": order_seed,
        "result_layout": None,
        "result_namespace": method,
        "parent_run_tags": [],
        "config_path": str(config_path),
        "job_dir": str(job_dir),
        "result_root": str(result_root),
        "retry_result_root_parent": str(result_parent),
        "formal_manifest": str(formal_manifest),
        "expected_episode_count": order["episode_count"],
        "expected_benchmark": order["benchmark"],
        "expected_order_manifest": order["path"],
        "expected_order_manifest_sha256": order["sha256"],
        "expected_episode_order_sha256": order["order_sha256"],
        "expected_dataset_sha256": order["dataset_sha256"],
        "expected_checkpoint_sha256": source["checkpoint_sha256"],
        "source_metrics": source["metrics"],
        "git_commit": _git_commit(),
        "command": command,
    }


def build_selection_jobs(spec, batch_id, gpu=0, batch_root=None):
    batch_root = Path(batch_root or (LOG_ROOT / batch_id))
    jobs = []
    ordinal = 0
    phase_index = 0
    for setting in spec["matrix"]["setting_order"]:
        for method in spec["matrix"]["method_order"]:
            for candidate in spec["matrix"]["candidates"][setting][method]:
                for seed in spec["protocol"]["selection_order_seeds"]:
                    jobs.append(_make_job(
                        spec, batch_id, batch_root, gpu, "selection",
                        phase_index, ordinal, setting, method,
                        candidate["candidate_id"], candidate["parameters"], seed,
                    ))
                    ordinal += 1
            phase_index += 1
    return jobs


def build_evaluation_jobs(spec, frozen, batch_id, gpu=0, batch_root=None):
    if not isinstance(frozen, dict) or frozen.get("schema") != FROZEN_SCHEMA:
        raise UserError("evaluation requires a validated frozen selection")
    batch_root = Path(batch_root or (LOG_ROOT / batch_id))
    jobs = []
    ordinal = 0
    phase_index = 0
    cells = frozen.get("cells", {})
    for setting in spec["matrix"]["setting_order"]:
        for method in spec["matrix"]["method_order"]:
            try:
                winner = cells[setting][method]
            except (KeyError, TypeError):
                raise UserError("frozen selection is missing {} {}".format(setting, method))
            for seed in spec["protocol"]["evaluation_order_seeds"]:
                job = _make_job(
                    spec, batch_id, batch_root, gpu, "evaluation",
                    phase_index, ordinal, setting, method,
                    winner["candidate_id"], winner["parameters"], seed,
                )
                job["frozen_selection_sha256"] = _json_sha256(frozen)
                jobs.append(job)
                ordinal += 1
            phase_index += 1
    return jobs


def phase_groups(jobs):
    groups = []
    for phase_index in sorted({job["phase_index"] for job in jobs}):
        group = [job for job in jobs if job["phase_index"] == phase_index]
        pairs = {(job["setting"], job["method"]) for job in group}
        if len(pairs) != 1:
            raise UserError("phase {} mixes model-method cells".format(phase_index))
        groups.append(sorted(group, key=lambda item: item["ordinal"]))
    return groups


def _job_identity(job):
    identity = {
        key: job[key] for key in (
            "batch_id", "gpu", "ordinal", "phase_index", "phase_name",
            "base_run_tag",
            "campaign_stage", "split", "setting", "model", "method",
            "candidate_id", "candidate_parameters", "parameters",
            "order_seed", "expected_episode_count", "expected_benchmark",
            "job_dir", "config_path", "retry_result_root_parent",
            "expected_order_manifest",
            "expected_order_manifest_sha256",
            "expected_episode_order_sha256", "expected_dataset_sha256",
            "expected_checkpoint_sha256", "source_metrics", "git_commit",
        )
    }
    if job["campaign_stage"] == "evaluation":
        identity["frozen_selection_sha256"] = job["frozen_selection_sha256"]
    return identity


def _validate_job_contract(job):
    if (
        job.get("config_method") != job.get("method")
        or job.get("search_method") != job.get("method")
        or job.get("method") == "source"
        or job.get("stage") != "orders"
        or job.get("config_stage") != "orders"
        or job.get("episodes") != -1
        or type(job.get("order_seed")) is not int
        or type(job.get("gpu")) is not int
        or job["gpu"] < 0
    ):
        raise UserError("job violates the non-Source orders-stage contract")
    attempt = job.get("attempt")
    if type(attempt) is not int or attempt < 0:
        raise UserError("job attempt must be a nonnegative integer")
    expected_base_tag = _job_tag(
        job["batch_id"], job["campaign_stage"], job["setting"],
        job["method"], job["candidate_id"], job["order_seed"],
    )
    if job.get("base_run_tag") != expected_base_tag:
        raise UserError("job base_run_tag does not match its immutable identity")
    expected_run_tag = (
        expected_base_tag if attempt == 0
        else "{}-retry{}".format(expected_base_tag, attempt)
    )
    if job.get("run_tag") != expected_run_tag:
        raise UserError("job run_tag does not match its retry attempt")
    expected_config_path = str(Path(job["job_dir"]) / "parameters.json")
    if job.get("config_path") != expected_config_path:
        raise UserError("job config_path is not inside its durable job directory")
    expected_result_root = str(
        Path(job["retry_result_root_parent"]) / expected_run_tag / job["split"]
    )
    if job.get("result_root") != expected_result_root:
        raise UserError("job result_root does not match its retry attempt")
    expected_formal_manifest = str(
        FORMAL_ROOT
        / "{}-{}-{}-native".format(
            expected_run_tag, job["setting"], job["split"]
        )
        / "manifest.json"
    )
    if job.get("formal_manifest") != expected_formal_manifest:
        raise UserError("job formal_manifest does not match its retry attempt")
    expected_parameters = _parameters_for_seed(
        job["method"], job["candidate_parameters"], job["order_seed"]
    )
    if _canonical(job.get("parameters")) != _canonical(expected_parameters):
        raise UserError("job parameters do not match candidate/order seed")
    expected_command = [
        str(RUNNER), job["setting"], job["split"], str(job["gpu"]),
        "--run-tag", expected_run_tag,
        "--tta-config", expected_config_path,
        "--result-root", expected_result_root,
        "--order-seed", str(job["order_seed"]),
    ]
    if job.get("command") != expected_command:
        raise UserError("job command differs from the complete registered command")


def _plan_payload(spec, spec_path, batch_id, gpu, selection_jobs):
    candidate_count = sum(
        len(spec["matrix"]["candidates"][setting][method])
        for setting in SETTINGS for method in METHODS
    )
    return {
        "schema": PLAN_SCHEMA,
        "batch_id": batch_id,
        "experiment_id": spec["experiment_id"],
        "git_commit": _git_commit(),
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "gpu": int(gpu),
        "source_execution_jobs": 0,
        "selection": {
            "split": "val_seen",
            "order_seeds": [1, 2, 3],
            "candidate_configs": candidate_count,
            "job_count": len(selection_jobs),
            "phase_count": len(SETTINGS) * len(METHODS),
        },
        "evaluation": {
            "split": "val_unseen",
            "order_seeds": [1, 2, 3],
            "job_count_after_freeze": len(SETTINGS) * len(METHODS) * 3,
            "phase_count_after_freeze": len(SETTINGS) * len(METHODS),
        },
        "selection_policy": {
            "feasibility": {
                "median_sr_strictly_above_source": True,
                "median_spl_strictly_above_source": True,
                "worst_sr_floor_pp": -0.2,
                "worst_spl_floor_pp": -0.2,
            },
            "ranking": [
                "feasible_first", "max_worst_sr_gain",
                "max_median_sr_gain", "max_worst_spl_gain",
                "max_median_spl_gain", "min_mean_relative_param_drift",
                "candidate_id_ascending_tie_break",
            ],
            "val_unseen_feedback_forbidden": True,
            "positive_gain_not_guaranteed": True,
        },
        "source_controls": copy.deepcopy(spec["source_controls"]),
        "order_manifests": copy.deepcopy(spec["_order_bindings"]),
        "selection_jobs": [_job_identity(job) for job in selection_jobs],
    }


def _prepare_batch(spec, spec_path, batch_id, batch_root, gpu, resume):
    selection_jobs = build_selection_jobs(
        spec, batch_id, gpu=gpu, batch_root=batch_root
    )
    payload = _plan_payload(spec, spec_path, batch_id, gpu, selection_jobs)
    path = Path(batch_root) / "PLAN.json"
    if path.is_file():
        if not resume:
            raise UserError("batch already exists; use --resume")
        if _canonical(_read_json(path)) != _canonical(payload):
            raise UserError("batch ID is bound to a different immutable PLAN")
    elif Path(batch_root).exists() and any(Path(batch_root).iterdir()):
        raise UserError("nonempty batch root lacks PLAN.json")
    else:
        _atomic_json(path, payload)
    return selection_jobs


def _ensure_phase_plan(batch_root, phase_jobs, resume):
    phase_dir = Path(phase_jobs[0]["job_dir"]).parents[2]
    plan_path = phase_dir / "PLAN.json"
    expected = {
        "schema": PHASE_PLAN_SCHEMA,
        "campaign_stage": phase_jobs[0]["campaign_stage"],
        "phase_index": phase_jobs[0]["phase_index"],
        "phase_name": phase_jobs[0]["phase_name"],
        "setting": phase_jobs[0]["setting"],
        "model": phase_jobs[0]["model"],
        "method": phase_jobs[0]["method"],
        "strict_barrier_after": True,
        "source_execution_jobs": 0,
        "job_count": len(phase_jobs),
        "jobs": [_job_identity(job) for job in phase_jobs],
    }
    if plan_path.is_file():
        if not resume:
            raise UserError("phase plan already exists; use --resume")
        if _canonical(_read_json(plan_path)) != _canonical(expected):
            raise UserError("persisted phase PLAN differs from requested phase")
        persisted = hparam_runner.load_jobs(phase_dir)
        if [_job_identity(job) for job in persisted] != expected["jobs"]:
            raise UserError("persisted phase jobs differ from immutable PLAN")
        for job in persisted:
            _validate_job_contract(job)
            expected_config = hparam_runner._job_config(job)
            if _canonical(_read_json(job["config_path"])) != _canonical(expected_config):
                raise UserError("persisted TTA config differs from immutable job")
        return persisted
    hparam_runner.write_plan(phase_dir, phase_jobs)
    for job in phase_jobs:
        _validate_job_contract(job)
    _atomic_json(plan_path, expected)
    return phase_jobs


def _load_persisted_stage_jobs(batch_root, campaign_stage, expected_jobs):
    """Load current attempts while authenticating every immutable phase plan."""
    persisted_jobs = []
    for expected_group in phase_groups(expected_jobs):
        phase_dir = Path(expected_group[0]["job_dir"]).parents[2]
        phase_plan_path = phase_dir / "PLAN.json"
        if not phase_plan_path.is_file():
            raise UserError(
                "missing persisted {} phase PLAN: {}".format(
                    campaign_stage, phase_plan_path
                )
            )
        phase_plan = _read_json(phase_plan_path)
        expected_identities = [_job_identity(job) for job in expected_group]
        checks = {
            "schema": PHASE_PLAN_SCHEMA,
            "campaign_stage": campaign_stage,
            "phase_index": expected_group[0]["phase_index"],
            "phase_name": expected_group[0]["phase_name"],
            "setting": expected_group[0]["setting"],
            "model": expected_group[0]["model"],
            "method": expected_group[0]["method"],
            "strict_barrier_after": True,
            "source_execution_jobs": 0,
            "job_count": len(expected_group),
            "jobs": expected_identities,
        }
        if _canonical(phase_plan) != _canonical(checks):
            raise UserError("persisted {} phase PLAN was modified".format(campaign_stage))
        current_group = hparam_runner.load_jobs(phase_dir)
        if [_job_identity(job) for job in current_group] != expected_identities:
            raise UserError("persisted {} jobs differ from their PLAN".format(campaign_stage))
        for job in current_group:
            _validate_job_contract(job)
            expected_config = hparam_runner._job_config(job)
            if _canonical(_read_json(job["config_path"])) != _canonical(expected_config):
                raise UserError("persisted TTA config differs from its current job")
        persisted_jobs.extend(current_group)
    if len(persisted_jobs) != len(expected_jobs):
        raise UserError("persisted {} job matrix is incomplete".format(campaign_stage))
    return persisted_jobs


def _metric_artifact(result_root, split):
    matches = []
    for path in sorted(Path(result_root).rglob("valid.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Env name: {}".format(split) not in line:
                continue
            values = {
                key.upper(): float(value) for key, value in METRIC_RE.findall(line)
            }
            if set(values) == {"SR", "SPL"}:
                matches.append((path, values, line))
    if len(matches) != 1:
        raise UserError(
            "expected exactly one {} SR/SPL metric line under {}; found {}".format(
                split, result_root, len(matches)
            )
        )
    path, values, line = matches[0]
    if any(not 0.0 <= value <= 100.0 for value in values.values()):
        raise UserError("R2R metric is outside [0, 100]")
    return path, values, line


def _validate_formal_manifest(job, required_artifacts):
    path = Path(job["formal_manifest"]).resolve()
    try:
        path.relative_to(FORMAL_ROOT.resolve())
    except ValueError:
        raise UserError("formal manifest is outside vln/results/runs")
    expected_run_id = "{}-{}-{}-native".format(
        job["run_tag"], job["setting"], job["split"]
    )
    if path.parent.name != expected_run_id or path.name != "manifest.json":
        raise UserError("formal manifest path is noncanonical")
    manifest = _read_json(path)
    expected = {
        "run_id": expected_run_id,
        "task": "vln",
        "benchmark": job["expected_benchmark"],
        "model": job["model"],
        "method": job["method"],
        "run_tag": job["run_tag"],
        "source_setting": "{}:{}:native:{}".format(
            job["setting"], job["split"], job["method"]
        ),
        "seed": job["order_seed"],
        "git_commit": job["git_commit"],
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise UserError("formal manifest {} mismatch".format(key))
    if manifest.get("checkpoint", {}).get("sha256") != job[
        "expected_checkpoint_sha256"
    ]:
        raise UserError("formal manifest checkpoint SHA256 mismatch")
    dataset = manifest.get("dataset", {})
    if dataset.get("stream_content_sha256") != job["expected_dataset_sha256"]:
        raise UserError("formal manifest dataset SHA256 mismatch")
    if dataset.get("stream_order_sha256") != job[
        "expected_episode_order_sha256"
    ]:
        raise UserError("formal manifest episode-order SHA256 mismatch")
    order_pin = manifest.get("pinned_manifests", {}).get("episode_order", {})
    if order_pin.get("sha256") != job["expected_order_manifest_sha256"]:
        raise UserError("formal manifest order-manifest file SHA256 mismatch")
    identity = manifest.get("immutable_identity_sha256")
    if not _valid_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise UserError("formal manifest immutable identity mismatch")
    artifacts = manifest.get("result_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise UserError("formal manifest has no result artifacts")
    authenticated = set()
    result_root = Path(job["result_root"]).resolve()
    for artifact in artifacts:
        artifact_path = Path(str(artifact.get("path", ""))).resolve()
        try:
            artifact_path.relative_to(result_root)
        except ValueError:
            raise UserError("formal artifact escapes result root")
        if (
            not artifact_path.is_file()
            or artifact.get("size") != artifact_path.stat().st_size
            or artifact.get("sha256") != _sha256(artifact_path)
        ):
            raise UserError("formal result artifact digest mismatch")
        authenticated.add(artifact_path)
    for required in required_artifacts:
        if Path(required).resolve() not in authenticated:
            raise UserError("required artifact is absent from formal manifest")
    return path, manifest


def _validate_implementation_gates(job, diagnostics, adapter):
    method = job["method"]
    parameters = job["parameters"]
    if method == "eam":
        batch_size = parameters.get("batch_size")
        if (
            type(batch_size) is not int
            or batch_size < 1
            or adapter.get("short_buffer_behavior") != "warmup_no_update"
            or adapter.get("warmup_no_update_steps") != batch_size - 1
        ):
            raise UserError("EAM replay warm-up diagnostics violate the reviewed contract")
        if adapter.get("batch_size_steps", batch_size) != batch_size:
            raise UserError("EAM diagnostics batch size mismatch")
    elif method == "fstta":
        if parameters.get("m") != 3 or parameters.get("n") != 4:
            raise UserError("FSTTA robust campaign requires M=3 and N=4")
        if (
            adapter.get("fast_window") != 3
            or adapter.get("slow_window") != 4
            or adapter.get("variance_history_lifetime") != "test_stream"
        ):
            raise UserError("FSTTA window/history diagnostics mismatch")
    elif method == "feedtta":
        # The shared validator checks endpoint, complete success/failure
        # accounting, evaluator agreement, and target-native argmax.
        if adapter.get("feedback_episodes") != job["expected_episode_count"]:
            raise UserError("FeedTTA did not consume one feedback item per episode")
    elif method == "atena":
        # The shared validator checks the lazy endpoint and requires
        # queries+self-labels == episodes with feedback observed iff queried.
        feature_dim = adapter.get("self_prediction_feature_dim")
        if type(feature_dim) is not int or feature_dim <= 0:
            raise UserError("ATENA self-prediction feature dimension is missing")
        if job["setting"] == "duet-r2r" and feature_dim != 1536:
            raise UserError("DUET ATENA must use 1536-D global/local CLS features")


def parse_job_result(job, _unused_spec=None):
    metric_path, metrics, metric_line = _metric_artifact(
        job["result_root"], job["split"]
    )
    diagnostics_path = Path(job["result_root"]) / "tta_diagnostics.json"
    diagnostics = _read_json(diagnostics_path)
    metadata = {
        "episode_count": job["expected_episode_count"],
        "method": job["method"],
        "parameters": job["parameters"],
    }
    adapter = frozen_eval._validate_diagnostics(metadata, diagnostics, metrics)
    _validate_implementation_gates(job, diagnostics, adapter)
    drift = adapter.get("relative_param_drift")
    if not _finite_number(drift) or float(drift) < 0.0:
        raise UserError("adapter relative_param_drift is missing or invalid")

    # Reuse the established val_seen parser and cross-check its SR/SPL values.
    if job["split"] == "val_seen":
        parsed = ORIGINAL_HPARAM_PARSE_METRICS(job, {
            "setting_episode_counts": {job["setting"]: job["expected_episode_count"]},
            "protocol": {"require_formal_final_manifest": False},
        })
        for key in ("SR", "SPL"):
            if float(parsed["metrics"][key]) != metrics[key]:
                raise UserError("val_seen parser/artifact {} mismatch".format(key))

    formal_path, manifest = _validate_formal_manifest(
        job, (metric_path, diagnostics_path)
    )
    result = dict(job)
    result.update({
        "metrics": metrics,
        "metric_artifact": str(metric_path),
        "metric_artifact_sha256": _sha256(metric_path),
        "metric_line": metric_line,
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "adapter_diagnostics": adapter,
        "relative_param_drift": float(drift),
        "formal_manifest_path": str(formal_path),
        "formal_manifest_sha256": _sha256(formal_path),
        "formal_immutable_identity_sha256": manifest["immutable_identity_sha256"],
    })
    return result


def _write_phase_summary(stage_dir, jobs, _unused_spec=None):
    results = []
    errors = []
    for job in jobs:
        exit_path = Path(job["job_dir"]) / "exitcode"
        if not exit_path.is_file():
            continue
        try:
            code = int(exit_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            errors.append({"run_tag": job["run_tag"], "error": "invalid exitcode"})
            continue
        if code != 0:
            errors.append({"run_tag": job["run_tag"], "exit_code": code})
            continue
        try:
            result = parse_job_result(job)
            _atomic_json(Path(job["job_dir"]) / "metrics.json", result)
            results.append(result)
        except Exception as error:
            errors.append({"run_tag": job["run_tag"], "error": str(error)})
    summary = {
        "schema": SUMMARY_SCHEMA,
        "scope": "phase",
        "campaign_stage": jobs[0]["campaign_stage"] if jobs else None,
        "phase_name": jobs[0]["phase_name"] if jobs else None,
        "source_execution_jobs": 0,
        "planned": len(jobs),
        "validated": len(results),
        "errors": errors,
        "complete": len(results) == len(jobs) and not errors,
        "results": results,
    }
    _atomic_json(Path(stage_dir) / "SUMMARY.json", summary)
    return results, errors


@contextmanager
def _patched_hparam_scheduler():
    original_parse = hparam_runner.parse_metrics
    original_summary = hparam_runner.write_summary
    original_shared = hparam_runner.uses_shared_gpu_peer
    hparam_runner.parse_metrics = parse_job_result
    hparam_runner.write_summary = _write_phase_summary
    # run_batch's shared path both serializes launch snapshots and initializes
    # the worker process identity used for durable resume.
    hparam_runner.uses_shared_gpu_peer = lambda _spec: True
    try:
        yield
    finally:
        hparam_runner.parse_metrics = original_parse
        hparam_runner.write_summary = original_summary
        hparam_runner.uses_shared_gpu_peer = original_shared


def _retry_failed_jobs(jobs):
    for job in jobs:
        job_dir = Path(job["job_dir"])
        exit_path = job_dir / "exitcode"
        should_retry = False
        if exit_path.is_file():
            try:
                should_retry = int(exit_path.read_text().strip()) != 0
            except ValueError:
                should_retry = True
            if not should_retry:
                try:
                    parse_job_result(job)
                except Exception:
                    should_retry = True
        else:
            pid = hparam_runner._worker_pid(job)
            identity = hparam_runner._worker_identity(job)
            should_retry = pid is not None and not hparam_runner.process_alive(pid, identity)
        if not should_retry:
            continue
        attempt_root = job_dir / "attempts" / "attempt-{:02d}".format(
            int(job.get("attempt", 0))
        )
        attempt_root.mkdir(parents=True, exist_ok=False)
        _atomic_json(attempt_root / "job.json", job)
        for name in (
            "console.log", "exitcode", "metrics.json", "worker_state.json",
            "validation_error.json",
        ):
            path = job_dir / name
            if path.exists():
                path.rename(attempt_root / name)
        old_result = Path(job["result_root"])
        if old_result.exists():
            old_result.rename(attempt_root / "result_root")
        old_formal = Path(job["formal_manifest"]).parent
        if old_formal.exists():
            old_formal.rename(attempt_root / "formal_run_manifest")
        job["attempt"] = int(job.get("attempt", 0)) + 1
        job["run_tag"] = "{}-retry{}".format(job["base_run_tag"], job["attempt"])
        job["result_root"] = str(
            Path(job["retry_result_root_parent"]) / job["run_tag"] / job["split"]
        )
        job["formal_manifest"] = str(
            FORMAL_ROOT
            / "{}-{}-{}-native".format(
                job["run_tag"], job["setting"], job["split"]
            )
            / "manifest.json"
        )
        command = list(job["command"])
        command[command.index("--run-tag") + 1] = job["run_tag"]
        command[command.index("--result-root") + 1] = job["result_root"]
        job["command"] = command
        _validate_job_contract(job)
        hparam_runner._write_job(job)


def _scheduler_args(args, spec, phase_jobs):
    model = phase_jobs[0]["model"]
    method = phase_jobs[0]["method"]
    cap = _worker_cap(spec, model, method)
    execution = spec["execution"]
    return argparse.Namespace(
        method=method,
        batch_id=args.batch_id,
        settings=[phase_jobs[0]["setting"]],
        gpu=args.gpu,
        max_workers=cap,
        max_per_model=cap,
        max_discrete_workers=cap,
        max_continuous_workers=1,
        max_gpu_memory_mib=int(execution.get("max_gpu_memory_mib_before_launch", 30000)),
        estimated_job_gpu_memory_mib=int(execution.get("estimated_job_gpu_memory_mib", 0)),
        max_aggregate_gpu_memory_mib=int(execution.get("max_aggregate_gpu_memory_mib", 32000)),
        max_memory_gib=float(execution.get("max_cgroup_memory_gib_before_launch", 220.0)),
        estimated_job_memory_gib=float(execution.get("estimated_job_memory_gib", 0.0)),
        max_aggregate_memory_gib=float(execution.get("max_aggregate_cgroup_memory_gib", 240.0)),
        launch_stagger=float(execution.get("launch_stagger_seconds", 0.0)),
        resource_wait_timeout=float(execution.get("resource_wait_timeout_seconds", 900.0)),
        resume=args.resume,
        retry_failed=False,
        fail_fast=True,
    )


def _run_phase(args, spec, phase_jobs):
    if args.retry_failed:
        _retry_failed_jobs(phase_jobs)
    stage_dir = Path(phase_jobs[0]["job_dir"]).parents[2]
    scheduler_args = _scheduler_args(args, spec, phase_jobs)
    with _patched_hparam_scheduler():
        hparam_runner.run_batch(scheduler_args, stage_dir, phase_jobs, spec)


def _result_from_job(job, require_cache=False, write_cache=True):
    _validate_job_contract(job)
    config = _read_json(job["config_path"])
    if _canonical(config) != _canonical(hparam_runner._job_config(job)):
        raise UserError("persisted TTA config differs from its current job")
    exit_path = Path(job["job_dir"]) / "exitcode"
    try:
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise UserError("job has no valid terminal exit code: {}".format(error))
    if exit_code != 0:
        raise UserError("job exited with status {}".format(exit_code))
    result = parse_job_result(job)
    path = Path(job["job_dir"]) / "metrics.json"
    if not path.is_file():
        if require_cache:
            raise UserError("completed job is missing its durable metrics cache")
        if write_cache:
            _atomic_json(path, result)
        return result
    persisted = _read_json(path)
    if persisted.get("split") != job["split"]:
        raise UserError("persisted result split mismatch")
    if _canonical(persisted) != _canonical(result):
        raise UserError("persisted result no longer matches authenticated artifacts")
    return result


def write_stage_summary(batch_root, campaign_stage, jobs, spec):
    results = []
    incomplete = []
    for job in jobs:
        if not (Path(job["job_dir"]) / "exitcode").is_file():
            incomplete.append(job["base_run_tag"])
            continue
        results.append(_result_from_job(job))
    payload = {
        "schema": SUMMARY_SCHEMA,
        "scope": campaign_stage,
        "experiment_id": spec["experiment_id"],
        "split": "val_seen" if campaign_stage == "selection" else "val_unseen",
        "source_execution_jobs": 0,
        "planned": len(jobs),
        "validated": len(results),
        "complete": not incomplete and len(results) == len(jobs),
        "incomplete_run_tags": incomplete,
        "results": sorted(results, key=lambda value: value["ordinal"]),
    }
    if campaign_stage == "evaluation" and payload["complete"]:
        payload["aggregates"] = _evaluation_aggregates(spec, results)
    _atomic_json(Path(batch_root) / campaign_stage / "SUMMARY.json", payload)
    return payload


def _authenticate_complete_stage(batch_root, campaign_stage, spec, expected_jobs):
    jobs = _load_persisted_stage_jobs(
        batch_root, campaign_stage, expected_jobs
    )
    results = [_result_from_job(job) for job in jobs]
    payload = {
        "schema": SUMMARY_SCHEMA,
        "scope": campaign_stage,
        "experiment_id": spec["experiment_id"],
        "split": "val_seen" if campaign_stage == "selection" else "val_unseen",
        "source_execution_jobs": 0,
        "planned": len(jobs),
        "validated": len(results),
        "complete": True,
        "incomplete_run_tags": [],
        "results": sorted(results, key=lambda value: value["ordinal"]),
    }
    if campaign_stage == "evaluation":
        payload["aggregates"] = _evaluation_aggregates(spec, results)
    return jobs, payload


def _candidate_statistics(rows, source_metrics):
    sr = [float(row["metrics"]["SR"]) for row in rows]
    spl = [float(row["metrics"]["SPL"]) for row in rows]
    drift = [float(row["relative_param_drift"]) for row in rows]
    source_sr = float(source_metrics["SR"])
    source_spl = float(source_metrics["SPL"])
    statistics_value = {
        "SR": {
            "per_seed": {str(row["order_seed"]): float(row["metrics"]["SR"]) for row in rows},
            "mean": statistics.fmean(sr),
            "median": statistics.median(sr),
            "std": statistics.stdev(sr),
            "worst": min(sr),
        },
        "SPL": {
            "per_seed": {str(row["order_seed"]): float(row["metrics"]["SPL"]) for row in rows},
            "mean": statistics.fmean(spl),
            "median": statistics.median(spl),
            "std": statistics.stdev(spl),
            "worst": min(spl),
        },
        "relative_param_drift": {
            "per_seed": {str(row["order_seed"]): float(row["relative_param_drift"]) for row in rows},
            "mean": statistics.fmean(drift),
            "worst": max(drift),
        },
    }
    gains = {
        "worst_sr": min(sr) - source_sr,
        "median_sr": statistics.median(sr) - source_sr,
        "worst_spl": min(spl) - source_spl,
        "median_spl": statistics.median(spl) - source_spl,
    }
    feasible = (
        gains["median_sr"] > 0.0
        and gains["median_spl"] > 0.0
        and gains["worst_sr"] >= -0.2 - 1e-12
        and gains["worst_spl"] >= -0.2 - 1e-12
    )
    return statistics_value, gains, feasible


def rank_cell(candidate_rows, source_metrics, expected_order_seeds=(1, 2, 3)):
    """Return the deterministic robust winner for one model-method cell."""
    expected_order_seeds = tuple(expected_order_seeds)
    by_candidate = {}
    for row in candidate_rows:
        if row.get("split") != "val_seen" or row.get("campaign_stage") != "selection":
            raise UserError("selection ranking accepts val_seen selection rows only")
        by_candidate.setdefault(row["candidate_id"], []).append(row)
    ranked = []
    for candidate_id, rows in by_candidate.items():
        rows = sorted(rows, key=lambda value: value["order_seed"])
        if tuple(row["order_seed"] for row in rows) != expected_order_seeds:
            raise UserError("candidate {} lacks the registered order seeds".format(candidate_id))
        base_parameters = rows[0]["candidate_parameters"]
        if any(_canonical(row["candidate_parameters"]) != _canonical(base_parameters) for row in rows):
            raise UserError("candidate {} changes parameters across seeds".format(candidate_id))
        summary, gains, feasible = _candidate_statistics(rows, source_metrics)
        ranked.append({
            "candidate_id": candidate_id,
            "parameters": base_parameters,
            "feasible": feasible,
            "gate_status": "pass" if feasible else "fail",
            "statistics": summary,
            "gains_pp": gains,
            "selection_runs": {
                str(row["order_seed"]): {
                    "run_tag": row["run_tag"],
                    "formal_manifest_path": row["formal_manifest_path"],
                    "formal_manifest_sha256": row["formal_manifest_sha256"],
                } for row in rows
            },
        })
    if not ranked:
        raise UserError("cannot freeze an empty candidate cell")
    ranked.sort(key=lambda value: (
        -int(value["feasible"]),
        -value["gains_pp"]["worst_sr"],
        -value["gains_pp"]["median_sr"],
        -value["gains_pp"]["worst_spl"],
        -value["gains_pp"]["median_spl"],
        value["statistics"]["relative_param_drift"]["mean"],
        value["candidate_id"],
    ))
    winner = copy.deepcopy(ranked[0])
    winner["feasible_candidate_count"] = sum(item["feasible"] for item in ranked)
    winner["candidate_count"] = len(ranked)
    winner["fallback_selected"] = not winner["feasible"]
    winner["ranking_position"] = 1
    return winner, ranked


def select_frozen(spec, selection_results, source_by_setting):
    if any(row.get("split") != "val_seen" for row in selection_results):
        raise UserError("val_unseen results are forbidden during freeze")
    expected = {}
    for setting in spec["matrix"]["setting_order"]:
        for method in spec["matrix"]["method_order"]:
            for candidate in spec["matrix"]["candidates"][setting][method]:
                for seed in spec["protocol"]["selection_order_seeds"]:
                    key = (setting, method, candidate["candidate_id"], seed)
                    expected[key] = candidate["parameters"]
    observed = {}
    for row in selection_results:
        key = (
            row.get("setting"), row.get("method"), row.get("candidate_id"),
            row.get("order_seed"),
        )
        if key not in expected:
            raise UserError("selection contains an unregistered candidate tuple: {}".format(key))
        if key in observed:
            raise UserError("selection contains a duplicate candidate tuple: {}".format(key))
        if row.get("campaign_stage") != "selection":
            raise UserError("selection row has the wrong campaign stage")
        if _canonical(row.get("candidate_parameters")) != _canonical(expected[key]):
            raise UserError("selection candidate parameters differ from the spec")
        seeded = _parameters_for_seed(key[1], expected[key], key[3])
        if _canonical(row.get("parameters")) != _canonical(seeded):
            raise UserError("selection executed parameters differ from the spec/order seed")
        observed[key] = row
    missing = set(expected).difference(observed)
    if missing:
        raise UserError(
            "selection is missing {} registered candidate tuples".format(len(missing))
        )
    if len(selection_results) != len(expected):
        raise UserError("selection result count differs from the registered matrix")
    cells = {}
    rankings = {}
    for setting in SETTINGS:
        cells[setting] = {}
        rankings[setting] = {}
        for method in METHODS:
            rows = [
                row for row in selection_results
                if row["setting"] == setting and row["method"] == method
            ]
            winner, ranked = rank_cell(
                rows, source_by_setting[setting],
                spec["protocol"]["selection_order_seeds"],
            )
            cells[setting][method] = winner
            rankings[setting][method] = ranked
    return cells, rankings


def _authenticate_selection_evidence(batch_root, spec):
    plan_path = Path(batch_root) / "PLAN.json"
    plan = _read_json(plan_path)
    expected_jobs = build_selection_jobs(
        spec, plan.get("batch_id", ""), plan.get("gpu", -1), batch_root
    )
    expected_plan = _plan_payload(
        spec, spec["_spec_path"], plan.get("batch_id", ""),
        plan.get("gpu", -1), expected_jobs,
    )
    if _canonical(plan) != _canonical(expected_plan):
        raise UserError("selection PLAN differs from the current immutable protocol")
    _, summary = _authenticate_complete_stage(
        batch_root, "selection", spec, expected_jobs
    )
    summary_path = Path(batch_root) / "selection" / "SUMMARY.json"
    _atomic_json(summary_path, summary)
    return plan, summary_path, summary


def freeze_selection(batch_root, spec):
    plan, summary_path, summary = _authenticate_selection_evidence(
        batch_root, spec
    )
    plan_path = Path(batch_root) / "PLAN.json"
    source = {
        setting: spec["_source_records"]["val_seen"][setting]["metrics"]
        for setting in SETTINGS
    }
    cells, rankings = select_frozen(spec, summary["results"], source)
    core = {
        "schema": FROZEN_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "selection_split": "val_seen",
        "selection_order_seeds": [1, 2, 3],
        "selection_summary_path": str(summary_path.resolve()),
        "selection_summary_sha256": _sha256(summary_path),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "git_commit": plan["git_commit"],
        "spec_sha256": plan["spec_sha256"],
        "source_metrics": source,
        "source_execution_jobs": 0,
        "val_unseen_consulted": False,
        "positive_gain_guaranteed": False,
        "cells": cells,
        "rankings": rankings,
    }
    path = Path(batch_root) / "FROZEN.json"
    if path.is_file():
        existing = _read_json(path)
        if _canonical(existing) != _canonical(core):
            raise UserError("existing FROZEN.json differs from recomputed selection")
        return existing
    _atomic_json(path, core)
    return core


def require_frozen(batch_root, spec):
    frozen_path = Path(batch_root) / "FROZEN.json"
    if not frozen_path.is_file():
        raise UserError("evaluation is blocked until selection and freeze complete")
    _, summary_path, summary = _authenticate_selection_evidence(
        batch_root, spec
    )
    frozen = _read_json(frozen_path)
    if frozen.get("selection_summary_sha256") != _sha256(summary_path):
        raise UserError("FROZEN.json is not bound to the current selection summary")
    plan_path = Path(batch_root) / "PLAN.json"
    if (
        frozen.get("plan_sha256") != _sha256(plan_path)
        or frozen.get("git_commit") != _git_commit()
    ):
        raise UserError("FROZEN.json is not bound to the current immutable PLAN")
    source = {
        setting: spec["_source_records"]["val_seen"][setting]["metrics"]
        for setting in SETTINGS
    }
    cells, rankings = select_frozen(spec, summary["results"], source)
    if _canonical(frozen.get("cells")) != _canonical(cells) or _canonical(
        frozen.get("rankings")
    ) != _canonical(rankings):
        raise UserError("FROZEN.json does not match deterministic val_seen selection")
    if frozen.get("val_unseen_consulted") is not False:
        raise UserError("FROZEN.json violates the no-val_unseen-selection contract")
    return frozen


def _evaluation_aggregates(spec, results):
    aggregates = {}
    expected_seeds = tuple(spec["protocol"]["evaluation_order_seeds"])
    for setting in SETTINGS:
        aggregates[setting] = {}
        source = spec["_source_records"]["val_unseen"][setting]["metrics"]
        for method in METHODS:
            rows = sorted(
                [row for row in results if row["setting"] == setting and row["method"] == method],
                key=lambda row: row["order_seed"],
            )
            if tuple(row["order_seed"] for row in rows) != expected_seeds:
                raise UserError("evaluation cell lacks registered order seeds")
            cell = {"source_metrics": source, "per_seed": {}}
            for row in rows:
                cell["per_seed"][str(row["order_seed"])] = {
                    "metrics": row["metrics"],
                    "gains_pp": {
                        key: row["metrics"][key] - source[key] for key in ("SR", "SPL")
                    },
                    "relative_param_drift": row["relative_param_drift"],
                    "run_tag": row["run_tag"],
                    "formal_manifest_sha256": row["formal_manifest_sha256"],
                }
            cell["aggregate"] = {}
            for key in ("SR", "SPL"):
                values = [row["metrics"][key] for row in rows]
                gains = [value - source[key] for value in values]
                cell["aggregate"][key] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.stdev(values),
                    "worst": min(values),
                    "mean_gain_pp": statistics.fmean(gains),
                    "worst_gain_pp": min(gains),
                }
            aggregates[setting][method] = cell
    return aggregates


def _progress_payload(batch_root, selection_jobs, evaluation_jobs=()):
    def counts(jobs):
        value = {
            "planned": len(jobs), "succeeded": 0, "failed": 0,
            "invalid": 0, "running": 0, "pending": 0,
        }
        for job in jobs:
            path = Path(job["job_dir"]) / "exitcode"
            if not path.is_file():
                pid = hparam_runner._worker_pid(job)
                identity = hparam_runner._worker_identity(job)
                if pid is None:
                    value["pending"] += 1
                elif hparam_runner.process_alive(pid, identity):
                    value["running"] += 1
                else:
                    value["invalid"] += 1
                continue
            try:
                code = int(path.read_text().strip())
            except ValueError:
                value["invalid"] += 1
                continue
            if code != 0:
                value["failed"] += 1
                continue
            try:
                _result_from_job(job, write_cache=False)
            except Exception:
                value["invalid"] += 1
            else:
                value["succeeded"] += 1
        return value
    return {
        "schema": "navtta.vln_r2r_cross_split_progress.v1",
        "selection": counts(selection_jobs),
        "evaluation": counts(evaluation_jobs),
        "frozen": (Path(batch_root) / "FROZEN.json").is_file(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _write_progress(batch_root, selection_jobs, evaluation_jobs=()):
    payload = _progress_payload(batch_root, selection_jobs, evaluation_jobs)
    _atomic_json(Path(batch_root) / "progress.json", payload)
    return payload


def _require_clean_execution_tree(spec_path, spec):
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise UserError("tracked worktree must be clean before formal execution")
    paths = [Path(__file__).resolve(), Path(spec_path).resolve()]
    for split in ("val_seen", "val_unseen"):
        paths.append(Path(spec["_source_records"][split][SETTINGS[0]]["ledger_path"]))
        for seed in (1, 2, 3):
            for setting in SETTINGS:
                paths.append(Path(spec["_order_bindings"][split][seed][setting]["path"]))
    relative = sorted({str(path.relative_to(REPO_ROOT.resolve())) for path in paths})
    checked = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", "--", *relative],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if checked.returncode != 0:
        raise UserError("runner, spec, ledgers, and order manifests must be tracked")
    untracked = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--others", "--exclude-standard", "--", *EXECUTION_SURFACE],
        text=True,
    ).strip()
    if untracked:
        raise UserError("formal execution refuses untracked execution files: {}".format(untracked))


def _materialize_plans(batch_root, jobs, resume):
    planned = []
    for group in phase_groups(jobs):
        planned.extend(_ensure_phase_plan(batch_root, group, resume))
    return planned


def _print_plan(spec, selection_jobs, frozen=None, print_commands=False):
    evaluation_count = len(SETTINGS) * len(METHODS) * 3
    print(json.dumps({
        "experiment_id": spec["experiment_id"],
        "selection_jobs": len(selection_jobs),
        "selection_phases": len(phase_groups(selection_jobs)),
        "evaluation_jobs_after_freeze": evaluation_count,
        "evaluation_plannable_now": frozen is not None,
        "source_execution_jobs": 0,
    }, indent=2, sort_keys=True))
    if print_commands:
        for job in selection_jobs:
            print(subprocess.list2cmdline(job["command"]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--batch-id")
    parser.add_argument(
        "--stage", choices=("selection", "freeze", "evaluation", "all"),
        default="all",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--confirm-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if args.stage == "all" and not args.plan_only and not args.status:
        parser.error(
            "formal --stage all is disabled; run selection, freeze, and "
            "evaluation as separate reviewed processes"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    spec_path = args.spec.resolve()
    spec = load_spec(spec_path)
    args.batch_id = args.batch_id or spec["experiment_id"]
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.batch_id):
        raise UserError("invalid --batch-id")
    batch_root = LOG_ROOT / args.batch_id

    if args.status:
        errors = []
        plan_path = batch_root / "PLAN.json"
        plan = _read_json(plan_path) if plan_path.is_file() else None
        plan_gpu = plan.get("gpu", args.gpu) if plan is not None else args.gpu
        selection_expected = build_selection_jobs(
            spec, args.batch_id, plan_gpu, batch_root
        )
        if plan is not None:
            expected_plan = _plan_payload(
                spec, spec_path, args.batch_id, plan_gpu, selection_expected
            )
            if _canonical(plan) != _canonical(expected_plan):
                errors.append("persisted PLAN differs from the current protocol")
        try:
            selection_jobs = (
                _load_persisted_stage_jobs(
                    batch_root, "selection", selection_expected
                ) if plan is not None else selection_expected
            )
        except UserError as error:
            selection_jobs = selection_expected
            errors.append(str(error))
        frozen = None
        frozen_path = batch_root / "FROZEN.json"
        if frozen_path.is_file():
            try:
                frozen = _read_json(frozen_path)
                evaluation_expected = build_evaluation_jobs(
                    spec, frozen, args.batch_id, plan_gpu, batch_root
                )
                evaluation_jobs = (
                    _load_persisted_stage_jobs(
                        batch_root, "evaluation", evaluation_expected
                    ) if (batch_root / "evaluation" / "phases").is_dir()
                    else evaluation_expected
                )
            except UserError as error:
                evaluation_jobs = []
                errors.append(str(error))
        else:
            evaluation_jobs = []
        payload = _progress_payload(batch_root, selection_jobs, evaluation_jobs)
        payload["integrity_errors"] = errors
        print(json.dumps(
            payload,
            indent=2, sort_keys=True,
        ))
        return 0

    selection_jobs = _prepare_batch(
        spec, spec_path, args.batch_id, batch_root, args.gpu, args.resume
    )
    selection_jobs = _materialize_plans(batch_root, selection_jobs, args.resume)
    frozen = None
    if (batch_root / "FROZEN.json").is_file():
        frozen = require_frozen(batch_root, spec)

    if args.plan_only:
        if frozen is not None and args.stage in ("evaluation", "all"):
            evaluation_jobs = build_evaluation_jobs(
                spec, frozen, args.batch_id, args.gpu, batch_root
            )
            _materialize_plans(batch_root, evaluation_jobs, args.resume)
        _print_plan(spec, selection_jobs, frozen, args.print_commands)
        _write_progress(batch_root, selection_jobs)
        return 0

    if not args.confirm_reviewed:
        raise UserError("formal execution requires --confirm-reviewed")
    _require_clean_execution_tree(spec_path, spec)

    if args.stage in ("selection", "all"):
        for group in phase_groups(selection_jobs):
            _run_phase(args, spec, group)
            write_stage_summary(batch_root, "selection", selection_jobs, spec)
            _write_progress(batch_root, selection_jobs)
        selection_summary = write_stage_summary(
            batch_root, "selection", selection_jobs, spec
        )
        if not selection_summary["complete"]:
            raise UserError("selection ended before every registered job completed")
        # ``--stage selection`` deliberately stops at authenticated val_seen
        # evidence.  A separate ``--stage freeze`` invocation creates the
        # immutable registry before any val_unseen job can be materialized.
        if args.stage == "all":
            frozen = freeze_selection(batch_root, spec)
    elif args.stage == "freeze":
        frozen = freeze_selection(batch_root, spec)

    if args.stage in ("evaluation", "all"):
        frozen = require_frozen(batch_root, spec)
        evaluation_jobs = build_evaluation_jobs(
            spec, frozen, args.batch_id, args.gpu, batch_root
        )
        evaluation_jobs = _materialize_plans(
            batch_root, evaluation_jobs, args.resume
        )
        for group in phase_groups(evaluation_jobs):
            _run_phase(args, spec, group)
            write_stage_summary(batch_root, "evaluation", evaluation_jobs, spec)
            _write_progress(batch_root, selection_jobs, evaluation_jobs)
        evaluation_summary = write_stage_summary(
            batch_root, "evaluation", evaluation_jobs, spec
        )
        if not evaluation_summary["complete"]:
            raise UserError("evaluation ended before every registered job completed")
    _write_progress(batch_root, selection_jobs, locals().get("evaluation_jobs", []))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (UserError, hparam_runner.UserError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
