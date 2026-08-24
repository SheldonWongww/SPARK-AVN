#!/usr/bin/env python3
"""Fail-closed, staged VLN TTA hyperparameter search.

Protocol:

* search every registered candidate on the globally shuffled ``val_unseen``
  streams with order seeds 1, 2, and 3;
* freeze one candidate per (setting, method) by a Source-relative
  ``mean(delta) - sample_std(delta)`` score;
* only after freezing, rerun that winner on ``val_seen`` seeds 1, 2, and 3 as
  a retention report.  ``val_seen`` can never affect the winner;
* authenticate every consumed metric and diagnostics file through its formal
  run manifest.  A failed, missing, incomplete, or inconsistent job aborts the
  whole stage.

R2R/R2R-CE rank SPL first and SR second.  REVERIE ranks RGSPL first and
RGS second.  There is deliberately no positive-gain gate or fallback.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_manifest_identity import immutable_identity_sha256  # noqa: E402


RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
CONFIG_TRANSLATOR = REPO_ROOT / "vln/scripts/tta_config_cli.py"
FORMAL_ROOT = REPO_ROOT / "vln/results/runs"
SEARCH_SCHEMA = "navtta.vln_tta_consistency_search.v2"
JOB_SCHEMA = "navtta.vln_tta_job.v1"
PLAN_SCHEMA = "navtta.vln_tta_consistency_plan.v2"
FROZEN_SCHEMA = "navtta.vln_tta_consistency_frozen.v2"
REPORT_SCHEMA = "navtta.vln_tta_consistency_report.v2"
ORDER_SEEDS = (1, 2, 3)
METHOD_ORDER = ("tent", "fstta", "eam", "feedtta", "atena", "idea")
CONTINUOUS_SETTINGS = {"etpnav-r2r-ce", "bevbert-r2r-ce"}
EXPECTED_SETTINGS = {
    "r2r": ["duet-r2r", "hamt-r2r", "goat-r2r"],
    "reverie": ["duet-reverie", "hamt-reverie", "goat-reverie"],
    "r2r-ce": ["etpnav-r2r-ce", "bevbert-r2r-ce"],
}
EXPECTED_CONCURRENCY = {
    "r2r": {
        "duet-r2r": {"tent": 10, "fstta": 12, "eam": 5, "feedtta": 6, "atena": 5, "idea": 1},
        "hamt-r2r": {"tent": 10, "fstta": 11, "eam": 5, "feedtta": 5, "atena": 5, "idea": 1},
        "goat-r2r": {"tent": 10, "fstta": 12, "eam": 5, "feedtta": 6, "atena": 5, "idea": 1},
    },
    "reverie": {
        setting: {method: 1 for method in METHOD_ORDER}
        for setting in ("duet-reverie", "hamt-reverie", "goat-reverie")
    },
    "r2r-ce": {
        setting: {
            method: (1 if method == "idea" else 3) for method in METHOD_ORDER
        }
        for setting in ("etpnav-r2r-ce", "bevbert-r2r-ce")
    },
}
MODEL_FOR_SETTING = {
    "duet-r2r": "duet",
    "hamt-r2r": "hamt",
    "goat-r2r": "goat",
    "duet-reverie": "duet",
    "hamt-reverie": "hamt",
    "goat-reverie": "goat",
    "etpnav-r2r-ce": "etpnav",
    "bevbert-r2r-ce": "bevbert",
}
ORDER_FAMILY = {
    "duet-r2r": "r2r_duet_hamt",
    "hamt-r2r": "r2r_duet_hamt",
    "goat-r2r": "r2r_goat",
    "duet-reverie": "reverie_duet_hamt",
    "hamt-reverie": "reverie_duet_hamt",
    "goat-reverie": "reverie_goat",
    "etpnav-r2r-ce": "r2r_ce_v1_3_unified",
    "bevbert-r2r-ce": "r2r_ce_v1_3_unified",
}
SOURCE_LEDGER = {
    ("r2r", "val_seen"): "vln/manifests/r2r_reused_source_controls.json",
    ("r2r", "val_unseen"): "vln/manifests/r2r_val_unseen_reused_source_controls.json",
    ("reverie", "val_seen"): "vln/manifests/reverie_reused_source_controls.json",
    ("reverie", "val_unseen"): "vln/manifests/reverie_val_unseen_reused_source_controls.json",
    ("r2r-ce", "val_seen"): "vln/manifests/r2r_ce_reused_source_controls.json",
    ("r2r-ce", "val_unseen"): "vln/manifests/r2r_ce_val_unseen_reused_source_controls.json",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDEA_SOURCE_STATS_SCHEMA = "navtta.idea.source_statistics"
IDEA_SOURCE_STATS_VERSION = 1
IDEA_SOURCE_STATS_ESTIMATOR = "global_sum_sumsq_count_sample_std"
IDEA_SOURCE_STATS_SCOPE = "all_steps_all_valid_tokens"
IDEA_SOURCE_TRAJECTORIES = 128
IDEA_SOURCE_COLLECTION_POLICY = "frozen_source_argmax_rollout"
SEARCH_PARAMETER_KEYS = {
    "tent": frozenset({"lr"}),
    "fstta": frozenset({"lr_fast", "lr_slow"}),
    "eam": frozenset({"lr"}),
    "feedtta": frozenset({"lr"}),
    "atena": frozenset({
        "lr_query", "lr_self", "query_threshold", "mix_lambda",
    }),
    "idea": frozenset({"lr", "tau"}),
}
COMMON_BASE_INVARIANTS = {
    "tent": {
        "norm_scope": "ln",
        "optimizer": "Adam",
        "action_selection": "argmax",
        "update_interval": 1,
        "max_grad_norm": 0.0,
    },
    "fstta": {
        "m": 3,
        "n": 4,
        "q": 0.1,
        "rho": 0.95,
        "tau": 0.7,
        "a": 0.9,
        "b": 1.1,
        "reset_var_hist_each_episode": False,
        "last_k_ln": 4,
        "norm_scope": "last_k_ln",
        "slow_optimizer": "AdamW",
        "beta1": 0.9,
        "beta2": 0.99,
        "fast_grad_mode": "concordant",
        "action_selection": "argmax",
        "max_grad_norm": 0.0,
    },
    "eam": {
        "confidence_scale": 0.4,
        "memory_size": 32,
        "batch_size": 8,
        "update_interval": 1,
        "optimizer": "Adam",
        "action_selection": "argmax",
    },
    "idea": {
        "prompt_length": 4,
        "k_max": 32,
        "lambda": 0.4,
        "fisher_beta": 0.1,
        "use_fisher": True,
        "opt_steps": 50,
        "action_selection": "argmax",
    },
}
FEEDTTA_BASE_INVARIANTS = {
    benchmark: {
        "gamma": 0.99,
        "p": 0.05,
        "alpha": -0.2 if benchmark == "reverie" else 0.1,
        "sgr_mode": "paper_main",
        "normalize_gradient": False,
        "optimizer_eps": 1e-5,
        "scope_profile": "paper_full",
        "action_selection": "argmax",
    }
    for benchmark in EXPECTED_SETTINGS
}
ATENA_BASE_INVARIANTS = {
    benchmark: {
        "action_selection": "argmax",
        "update_scope": "replay_reachable_high_level_navigation",
        "optimizer": "AdamW",
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "episodic": False,
        "max_grad_norm": 0.0,
        "self_loss_weight": 0.25 if benchmark == "reverie" else 0.1,
    }
    for benchmark in EXPECTED_SETTINGS
}
ATENA_REQUIRED_DIAGNOSTICS = {
    benchmark: {
        "exact_episode_replay_enabled": True,
        "gradient_reconstruction": "exact_step_replay_in_eval_mode",
        "retains_episode_graph": False,
        "requires_task_policy_scope_verification": True,
        "task_scope_evidence": (
            "high_level_waypoint_navigation_decision"
            if benchmark == "r2r-ce"
            else "replay_reachable_high_level_navigation"
        ),
    }
    for benchmark in EXPECTED_SETTINGS
}


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


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserError("cannot read JSON {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise UserError("JSON document must be an object: {}".format(path))
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value):
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _finite(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _selected(values, requested, label):
    requested = set(requested or ())
    unknown = requested.difference(values)
    if unknown:
        if label == "settings" and "streamvln-r2r-ce" in unknown:
            raise UserError(
                "StreamVLN TTA search is blocked: the launcher rejects "
                "shuffled order seeds and the config translator has no "
                "StreamVLN adapter"
            )
        raise UserError("unknown {}: {}".format(label, ", ".join(sorted(unknown))))
    return [value for value in values if not requested or value in requested]


def _require_exact_mapping(actual, expected, label):
    """Reject missing, additional, type-changed, or value-changed fields."""
    if not isinstance(actual, dict):
        raise UserError("{} must be an object".format(label))
    if _canonical(actual) == _canonical(expected):
        return
    actual_keys = set(actual)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    changed = sorted(
        key for key in actual_keys & expected_keys
        if _canonical(actual[key]) != _canonical(expected[key])
    )
    details = []
    if missing:
        details.append("missing={}".format(",".join(missing)))
    if extra:
        details.append("extra={}".format(",".join(extra)))
    if changed:
        details.append("changed={}".format(",".join(changed)))
    raise UserError(
        "{} differs from the registered exact invariant ({})".format(
            label, "; ".join(details) or "non-canonical value"
        )
    )


def _expected_base(method, benchmark):
    if method == "feedtta":
        return FEEDTTA_BASE_INVARIANTS[benchmark]
    if method == "atena":
        return ATENA_BASE_INVARIANTS[benchmark]
    return COMMON_BASE_INVARIANTS[method]


def _validate_candidates(method, method_spec, benchmark):
    base = method_spec.get("base")
    records = method_spec.get("candidates")
    if not isinstance(base, dict) or not isinstance(records, list) or not records:
        raise UserError("{} requires base plus a non-empty explicit candidate list".format(method))
    _require_exact_mapping(
        base, _expected_base(method, benchmark), "{}.base".format(method)
    )
    if method == "atena":
        _require_exact_mapping(
            method_spec.get("required_diagnostics"),
            ATENA_REQUIRED_DIAGNOSTICS[benchmark],
            "atena.required_diagnostics",
        )
    seen_ids = set()
    seen_parameters = set()
    anchors = 0
    for record in records:
        if not isinstance(record, dict):
            raise UserError("{} candidate is not an object".format(method))
        candidate_id = record.get("id")
        if not isinstance(candidate_id, str) or not re.fullmatch(r"[a-z0-9_]+", candidate_id):
            raise UserError("{} candidate id is invalid".format(method))
        if candidate_id in seen_ids:
            raise UserError("{} candidate id is duplicated: {}".format(method, candidate_id))
        seen_ids.add(candidate_id)
        overrides = record.get("parameters", {})
        if not isinstance(overrides, dict):
            raise UserError("{} candidate parameters must be an object".format(candidate_id))
        if set(overrides) != SEARCH_PARAMETER_KEYS[method]:
            raise UserError(
                "{} candidate {} must declare exactly the search keys {}"
                .format(
                    method,
                    candidate_id,
                    ", ".join(sorted(SEARCH_PARAMETER_KEYS[method])),
                )
            )
        parameters = dict(base)
        parameters.update(overrides)
        signature = _canonical(parameters)
        if signature in seen_parameters:
            raise UserError("{} has duplicate effective candidates".format(method))
        seen_parameters.add(signature)
        anchors += record.get("role") == "paper_anchor"

        if method == "idea" and parameters.get("opt_steps") != 50:
            raise UserError("IDEA must use paper O=50 in every stage")
        if method == "atena" and (
            float(parameters.get("query_threshold", 0.0)) <= 0.0
            or float(parameters.get("mix_lambda", 0.0)) <= 0.0
        ):
            raise UserError("ATENA candidates forbid delta=0 and lambda=0")
    if anchors != 1:
        raise UserError("{} requires exactly one paper_anchor candidate".format(method))

    expected_counts = {
        "tent": 4,
        "fstta": 4 if benchmark == "r2r-ce" else 3,
        "eam": 3,
        "feedtta": 3,
        "atena": 6,
        "idea": 4,
    }
    if len(records) != expected_counts[method]:
        raise UserError(
            "{} requires exactly {} constrained candidates".format(
                method, expected_counts[method]
            )
        )

    effective = [dict(base, **item.get("parameters", {})) for item in records]
    anchor = effective[next(
        index for index, item in enumerate(records)
        if item.get("role") == "paper_anchor"
    )]
    if method == "tent":
        if sorted(row["lr"] for row in effective) != sorted({
            "r2r": [1e-6, 3e-6, 1e-5, 3e-5],
            "reverie": [3e-6, 1e-5, 1.5625e-5, 3e-5],
            "r2r-ce": [3e-7, 1e-6, 3e-6, 1.5625e-5],
        }[benchmark]):
            raise UserError("Tent must use the registered four-LR grid")
    elif method == "fstta":
        pairs = sorted((row["lr_fast"], row["lr_slow"]) for row in effective)
        expected_pairs = [(6e-5, 1e-4), (1.8e-4, 3e-4), (6e-4, 1e-3)]
        if benchmark == "r2r-ce":
            expected_pairs.append((1.8e-5, 3e-5))
        expected_pairs.sort()
        if pairs != expected_pairs:
            raise UserError("FSTTA learning rates must be paired common-scale points")
    elif method == "eam":
        if sorted(row["lr"] for row in effective) != [1e-6, 3e-6, 1e-5]:
            raise UserError("EAM must vary only the registered three learning rates")
    elif method == "feedtta":
        if sorted(row["lr"] for row in effective) != [2e-6, 5e-6, 1e-5]:
            raise UserError("FeedTTA must use its registered benchmark anchor and LR-only grid")
    elif method == "atena":
        official = {
            "r2r": (8e-7, 1e-7, 0.1, 0.75),
            "reverie": (5e-6, 1e-7, 0.1, 0.5),
            "r2r-ce": (1e-6, 5e-7, 0.05, 0.5),
        }[benchmark]
        if (
            anchor["lr_query"], anchor["lr_self"],
            anchor["query_threshold"], anchor["mix_lambda"],
        ) != official:
            raise UserError("ATENA paper anchor differs from the official benchmark anchor")
    elif method == "idea":
        pairs = {(row["lr"], row["tau"]) for row in effective}
        if pairs != {(1e-3, 0.5), (1e-3, 0.7), (3e-3, 0.5), (3e-3, 0.7)}:
            raise UserError("IDEA requires the registered four lr/tau points")


def load_spec(path):
    spec = _read_json(path)
    if spec.get("schema") != SEARCH_SCHEMA:
        raise UserError("unsupported search spec schema: {}".format(spec.get("schema")))
    for key in ("experiment_id", "benchmark", "protocol", "settings", "methods", "concurrency"):
        if key not in spec:
            raise UserError("search spec missing required key: {}".format(key))
    protocol = spec["protocol"]
    expected = {
        "development_split": "val_unseen",
        "retention_split": "val_seen",
        "order_seeds": [1, 2, 3],
        "selection_score": "mean_delta_minus_sample_std_delta",
        "positive_result_required": False,
        "source_policy": "reuse_authenticated_order_invariant_argmax",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise UserError("protocol.{} must be {!r}".format(key, value))
    metric_pair = (
        ("RGSPL", "RGS") if spec["benchmark"] == "reverie" else ("SPL", "SR")
    )
    if (
        protocol.get("primary_metric"), protocol.get("secondary_metric")
    ) != metric_pair:
        raise UserError("benchmark metric policy must be {}/{}".format(*metric_pair))
    if spec["benchmark"] not in EXPECTED_SETTINGS:
        raise UserError("unsupported benchmark: {}".format(spec["benchmark"]))
    settings = spec["settings"]
    if not isinstance(settings, list) or not settings or len(settings) != len(set(settings)):
        raise UserError("settings must be a non-empty unique list")
    unknown_settings = set(settings).difference(MODEL_FOR_SETTING)
    if unknown_settings:
        if "streamvln-r2r-ce" in unknown_settings:
            raise UserError(
                "StreamVLN TTA search is blocked: tta_config_cli.py and "
                "run_source_eval.sh --order-seed do not support StreamVLN"
            )
        raise UserError("unsupported settings: {}".format(", ".join(sorted(unknown_settings))))
    if settings != EXPECTED_SETTINGS[spec["benchmark"]]:
        raise UserError("settings/order differ from the registered benchmark protocol")
    if set(spec["methods"]) != set(METHOD_ORDER):
        raise UserError("methods must define exactly {}".format(", ".join(METHOD_ORDER)))
    for method in METHOD_ORDER:
        _validate_candidates(method, spec["methods"][method], spec["benchmark"])
    if spec["concurrency"] != EXPECTED_CONCURRENCY[spec["benchmark"]]:
        raise UserError("concurrency differs from the registered per-cell caps")
    if spec["benchmark"] == "r2r-ce":
        if spec.get("model_barrier") is not True:
            raise UserError("R2R-CE requires a strict model barrier")
        blocked = spec.get("blocked_settings", {}).get("streamvln-r2r-ce", {})
        if blocked.get("status") != "blocked":
            raise UserError("R2R-CE must explicitly record StreamVLN as blocked")
    return spec


def expand_candidate_records(method, method_spec):
    records = []
    for ordinal, item in enumerate(method_spec["candidates"]):
        parameters = dict(method_spec["base"])
        parameters.update(item.get("parameters", {}))
        records.append({
            "candidate_id": item["id"],
            "role": item.get("role", "bounded_variant"),
            "ordinal": ordinal,
            "parameters": parameters,
        })
    return records


def _idea_source_parameters(spec, setting, expected_checkpoint_sha256=None):
    """Validate and materialize one setting's immutable IDEA source anchor.

    The per-setting binding deliberately lives outside the shared IDEA base:
    source statistics depend on the concrete policy/checkpoint, whereas the
    four searched ``lr x tau`` candidates do not.  Returning absolute paths
    also prevents a worker's baseline-specific cwd from changing which file is
    consumed.
    """
    method_spec = spec["methods"]["idea"]
    bindings = method_spec.get("source_statistics", {})
    binding = bindings.get(setting) if isinstance(bindings, dict) else None
    if not isinstance(binding, dict):
        raise UserError(
            "{}/IDEA is blocked until precomputed Source statistics are pinned"
            .format(setting)
        )
    path_value = binding.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise UserError("{}/IDEA Source-statistics path is missing".format(setting))
    path = Path(path_value)
    path = path if path.is_absolute() else REPO_ROOT / path
    expected_sha256 = binding.get("sha256")
    if not path.is_file() or not _valid_sha256(expected_sha256):
        raise UserError(
            "{}/IDEA Source-statistics binding is incomplete".format(setting)
        )
    if _sha256(path) != expected_sha256:
        raise UserError(
            "{}/IDEA Source-statistics SHA256 mismatch".format(setting)
        )
    # The formal campaign uses inspectable JSON artifacts.  Runtime performs
    # the tensor/dimension checks; here we reject wrong-domain assets before a
    # GPU process is launched.
    if path.suffix.lower() != ".json":
        raise UserError("IDEA consistency search requires a JSON source-statistics artifact")
    artifact = _read_json(path)
    provenance = artifact.get("provenance")
    if (
        artifact.get("schema") != IDEA_SOURCE_STATS_SCHEMA
        or artifact.get("version") != IDEA_SOURCE_STATS_VERSION
        or artifact.get("moment_estimator") != IDEA_SOURCE_STATS_ESTIMATOR
        or not isinstance(provenance, dict)
        or provenance.get("collection_scope") != IDEA_SOURCE_STATS_SCOPE
        or provenance.get("collection_policy")
        != IDEA_SOURCE_COLLECTION_POLICY
        or "train" not in str(provenance.get("split", "")).lower()
        or provenance.get("trajectory_count") != IDEA_SOURCE_TRAJECTORIES
    ):
        raise UserError(
            "{}/IDEA Source-statistics artifact violates the offline train-128 contract"
            .format(setting)
        )
    if (
        provenance.get("setting") != setting
        or provenance.get("model") != MODEL_FOR_SETTING[setting]
        or not provenance.get("dataset")
        or not provenance.get("dataset_version")
    ):
        raise UserError(
            "{}/IDEA Source-statistics provenance does not identify the model/dataset"
            .format(setting)
        )
    trajectory_ids = provenance.get("trajectory_ids")
    if (
        not isinstance(trajectory_ids, list)
        or len(trajectory_ids) != IDEA_SOURCE_TRAJECTORIES
        or len(set(map(str, trajectory_ids))) != IDEA_SOURCE_TRAJECTORIES
        or not _valid_sha256(provenance.get("trajectory_ids_sha256"))
    ):
        raise UserError("IDEA Source-statistics trajectory provenance is invalid")
    ids_bytes = json.dumps(
        sorted(map(str, trajectory_ids)), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(ids_bytes).hexdigest() != provenance["trajectory_ids_sha256"]:
        raise UserError("IDEA Source-statistics trajectory-id digest mismatch")
    artifact_checkpoint = provenance.get("checkpoint_sha256")
    if not _valid_sha256(artifact_checkpoint):
        raise UserError("IDEA Source-statistics checkpoint digest is invalid")
    if (
        expected_checkpoint_sha256 is not None
        and artifact_checkpoint != expected_checkpoint_sha256
    ):
        raise UserError(
            "{}/IDEA Source-statistics checkpoint differs from Source control"
            .format(setting)
        )
    binding_count = binding.get("trajectory_count", IDEA_SOURCE_TRAJECTORIES)
    if type(binding_count) is not int or binding_count != IDEA_SOURCE_TRAJECTORIES:
        raise UserError("IDEA source-statistics binding must declare 128 trajectories")
    if binding.get("collection_policy") != IDEA_SOURCE_COLLECTION_POLICY:
        raise UserError(
            "IDEA source-statistics binding must declare the frozen Source "
            "argmax rollout policy"
        )
    binding_checkpoint = binding.get("checkpoint_sha256")
    if binding_checkpoint != artifact_checkpoint:
        raise UserError(
            "{}/IDEA binding and artifact checkpoint digests differ".format(setting)
        )

    def bound_file(field, digest_field):
        value = binding.get(field)
        digest_value = binding.get(digest_field)
        if not isinstance(value, str) or not value or not _valid_sha256(digest_value):
            raise UserError(
                "{}/IDEA collection binding is missing {} provenance".format(
                    setting, field
                )
            )
        resolved = Path(value)
        resolved = resolved if resolved.is_absolute() else REPO_ROOT / resolved
        if not resolved.is_file() or _sha256(resolved) != digest_value:
            raise UserError(
                "{}/IDEA collection {} is missing or changed".format(
                    setting, field
                )
            )
        return resolved

    order_path = bound_file("order_manifest", "order_manifest_sha256")
    config_path = bound_file("collection_config", "collection_config_sha256")
    diagnostics_path = bound_file("diagnostics", "diagnostics_sha256")
    formal_path = bound_file("formal_manifest", "formal_manifest_sha256")
    order = _read_json(order_path)
    expected_ids = sorted(str(item.get("episode_id")) for item in order.get("episodes", []))
    if (
        order.get("split") != "train"
        or order.get("episode_count") != IDEA_SOURCE_TRAJECTORIES
        or len(expected_ids) != IDEA_SOURCE_TRAJECTORIES
        or provenance.get("trajectory_ids") != expected_ids
        or provenance.get("dataset") != order.get("dataset", {}).get("path")
        or provenance.get("dataset_version")
        != "sha256:" + str(order.get("dataset", {}).get("sha256", ""))
    ):
        raise UserError("{}/IDEA train-order provenance mismatch".format(setting))
    collection_config = _read_json(config_path)
    collection_parameters = collection_config.get("parameters", {})
    if (
        collection_config.get("schema") != JOB_SCHEMA
        or collection_config.get("namespace") != "idea_source_statistics"
        or collection_config.get("stage") != "source_statistics"
        or collection_config.get("method") != "idea"
        or collection_parameters.get("collect_source_stats") is not True
        or collection_parameters.get("action_selection") != "argmax"
        or collection_parameters.get("collection_policy")
        != IDEA_SOURCE_COLLECTION_POLICY
        or collection_parameters.get("source_checkpoint_sha256")
        != artifact_checkpoint
        or Path(str(collection_parameters.get("source_stats_output", ""))).resolve()
        != path.resolve()
    ):
        raise UserError("{}/IDEA collection config identity mismatch".format(setting))
    diagnostics = _read_json(diagnostics_path)
    collection = (
        diagnostics.get("adapter")
        if diagnostics.get("schema") == "navtta.vln_discrete_tta.v1"
        else diagnostics.get("idea_source_collection")
    )
    if (
        diagnostics.get("method") != "idea"
        or diagnostics.get("episode_count") != IDEA_SOURCE_TRAJECTORIES
        or not isinstance(collection, dict)
        or collection.get("complete") is not True
        or collection.get("trajectory_count") != IDEA_SOURCE_TRAJECTORIES
        or collection.get("artifact_sha256") != expected_sha256
        or collection.get("collection_scope") != IDEA_SOURCE_STATS_SCOPE
        or collection.get("provenance", {}).get("collection_policy")
        != IDEA_SOURCE_COLLECTION_POLICY
    ):
        raise UserError("{}/IDEA collection diagnostics mismatch".format(setting))
    formal = _read_json(formal_path)
    immutable = binding.get("formal_immutable_identity_sha256")
    expected_source_setting = "{}:train:{}:idea".format(
        setting, "v1.3-unified" if setting in CONTINUOUS_SETTINGS else "native"
    )
    if (
        formal.get("task") != "vln"
        or formal.get("model") != MODEL_FOR_SETTING[setting]
        or formal.get("method") != "idea"
        or formal.get("source_setting") != expected_source_setting
        or formal.get("status") != "completed"
        or formal.get("exit_code") != 0
        or formal.get("checkpoint", {}).get("sha256") != artifact_checkpoint
        or formal.get("dataset", {}).get("stream_order_sha256")
        != order.get("order_sha256")
        or formal.get("dataset", {}).get("stream_content_sha256")
        != order.get("dataset", {}).get("sha256")
        or not _valid_sha256(immutable)
        or formal.get("immutable_identity_sha256") != immutable
        or immutable_identity_sha256(formal) != immutable
        or formal.get("pinned_manifests", {}).get("episode_order", {}).get(
            "sha256"
        ) != binding["order_manifest_sha256"]
    ):
        raise UserError("{}/IDEA formal collection manifest mismatch".format(setting))
    auxiliary = {
        item.get("name"): item
        for item in formal.get("auxiliary_checkpoints", [])
        if isinstance(item, dict)
    }
    if auxiliary.get("tta_job_config", {}).get("sha256") != binding[
        "collection_config_sha256"
    ]:
        raise UserError("{}/IDEA formal collection config is unauthenticated".format(setting))
    artifacts = {
        item.get("name"): item
        for item in formal.get("result_artifacts", [])
        if isinstance(item, dict)
    }
    for required_path, required_sha in (
        (path, expected_sha256),
        (diagnostics_path, binding["diagnostics_sha256"]),
    ):
        try:
            name = required_path.resolve().relative_to(
                path.resolve().parent
            ).as_posix()
        except ValueError:
            raise UserError("IDEA collection artifacts do not share a result root")
        if artifacts.get(name, {}).get("sha256") != required_sha:
            raise UserError(
                "{}/IDEA formal collection result is unauthenticated".format(setting)
            )
    return {
        "source_stats_path": str(path.resolve()),
        "source_stats_sha256": expected_sha256,
        "source_trajectories": IDEA_SOURCE_TRAJECTORIES,
    }


def _candidate_records_for_setting(spec, setting, method):
    records = expand_candidate_records(method, spec["methods"][method])
    if method != "idea":
        return records
    source_parameters = _idea_source_parameters(spec, setting)
    bound = []
    for record in records:
        item = dict(record)
        item["parameters"] = dict(record["parameters"], **source_parameters)
        bound.append(item)
    return bound


def expand_candidates(method, method_spec, *, for_search=True):
    """Compatibility helper returning effective parameter dictionaries.

    ``for_search`` is intentionally ignored: IDEA uses O=50 everywhere.
    """
    del for_search
    return [item["parameters"] for item in expand_candidate_records(method, method_spec)]


def candidate_tag(method, params):
    digest = hashlib.sha256(_canonical(params).encode("utf-8")).hexdigest()[:10]
    return "{}-{}".format(method, digest)


def _parameters_for_seed(method, parameters, order_seed):
    seeded = dict(parameters)
    if method == "feedtta":
        seeded["sgr_seed"] = order_seed
        if seeded.get("action_selection") == "sample":
            seeded["action_seed"] = order_seed
        else:
            # Native-argmax VLN has no action-sampling RNG.  Do not put an
            # inert action seed into the formal config/manifest.
            seeded.pop("action_seed", None)
    return seeded


def _cell_dir(out_dir, spec, run_tag, setting, method):
    return Path(out_dir) / spec["benchmark"] / run_tag / setting / method


def _job_run_tag(run_tag, stage, setting, method, candidate_id, seed):
    return "{}-{}-{}-{}-{}-s{}".format(
        run_tag, stage, setting, method, candidate_id, seed
    )


def _data_version(setting):
    return "v1.3-unified" if setting in CONTINUOUS_SETTINGS else "native"


def _formal_manifest_path(run_tag, setting, split):
    return FORMAL_ROOT / "{}-{}-{}-{}".format(
        run_tag, setting, split, _data_version(setting)
    ) / "manifest.json"


def _order_metadata(setting, split, seed):
    path = (
        REPO_ROOT / "vln/manifests/episode_order"
        / "order_seed_{}".format(seed) / ORDER_FAMILY[setting]
        / "{}.json".format(split)
    )
    document = _read_json(path)
    if (
        document.get("split") != split
        or document.get("order_seed") != seed
        or type(document.get("episode_count")) is not int
        or document["episode_count"] <= 0
        or not _valid_sha256(document.get("order_sha256"))
        or not _valid_sha256(document.get("dataset", {}).get("sha256"))
    ):
        raise UserError("invalid shuffled order manifest: {}".format(path))
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256(path),
        "benchmark": document["benchmark"],
        "episode_count": document["episode_count"],
        "order_sha256": document["order_sha256"],
        "dataset_sha256": document["dataset"]["sha256"],
    }


def write_job_config(path, method, params, order_seed, protocol_stage, candidate_id):
    config = {
        "schema": JOB_SCHEMA,
        "stage": "orders",
        "episodes": -1,
        "order_seed": order_seed,
        "search_method": method,
        "method": method,
        "parameters": _parameters_for_seed(method, params, order_seed),
        "protocol_stage": protocol_stage,
        "candidate_id": candidate_id,
    }
    _atomic_json(path, config)
    return config


def _validate_job_config(setting, config_path, method):
    completed = subprocess.run(
        [
            sys.executable, str(CONFIG_TRANSLATOR), "--setting", setting,
            "--config", str(Path(config_path).resolve()),
            "--diagnostics", "/tmp/navtta-consistency-preflight.json",
            "--print-method",
        ],
        cwd=str(REPO_ROOT), text=True, capture_output=True,
    )
    if completed.returncode != 0 or completed.stdout.strip() != method:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise UserError("TTA config translator rejected {}/{}: {}".format(
            setting, method, detail
        ))


def build_command(setting, split, config_path, result_root, run_tag, order_seed, gpu=0):
    command = [
        "bash", str(RUNNER), setting, split, str(gpu),
        "--run-tag", run_tag,
        "--tta-config", str(Path(config_path).resolve()),
        "--result-root", str(Path(result_root).resolve()),
        "--order-seed", str(order_seed),
    ]
    if setting in CONTINUOUS_SETTINGS:
        command.extend(["--ce-data-version", "v1.3-unified"])
    return command


def _build_jobs(spec, out_dir, run_tag, setting, method, stage, candidates, gpu):
    split = (
        spec["protocol"]["development_split"]
        if stage == "search" else spec["protocol"]["retention_split"]
    )
    cell = _cell_dir(out_dir, spec, run_tag, setting, method)
    jobs = []
    for candidate in candidates:
        for seed in ORDER_SEEDS:
            job_run_tag = _job_run_tag(
                run_tag, stage, setting, method, candidate["candidate_id"], seed
            )
            job_dir = cell / stage / candidate["candidate_id"] / "seed_{}".format(seed)
            config_path = job_dir / "tta_config.json"
            config = write_job_config(
                config_path, method, candidate["parameters"], seed, stage,
                candidate["candidate_id"],
            )
            _validate_job_config(setting, config_path, method)
            result_root = job_dir / "result" / job_run_tag / split
            order = _order_metadata(setting, split, seed)
            jobs.append({
                "stage": stage,
                "setting": setting,
                "model": MODEL_FOR_SETTING[setting],
                "method": method,
                "candidate_id": candidate["candidate_id"],
                "candidate_role": candidate["role"],
                "candidate_parameters": candidate["parameters"],
                "parameters": config["parameters"],
                "order_seed": seed,
                "split": split,
                "run_tag": job_run_tag,
                "job_dir": str(job_dir.resolve()),
                "config_path": str(config_path.resolve()),
                "config_sha256": _sha256(config_path),
                "result_root": str(result_root.resolve()),
                "formal_manifest": str(
                    _formal_manifest_path(job_run_tag, setting, split).resolve()
                ),
                "expected_benchmark": order["benchmark"],
                "expected_episode_count": order["episode_count"],
                "expected_order_sha256": order["order_sha256"],
                "expected_dataset_sha256": order["dataset_sha256"],
                "expected_order_manifest_sha256": order["file_sha256"],
                "command": build_command(
                    setting, split, config_path, result_root, job_run_tag, seed, gpu
                ),
            })
    return jobs


def _plan_path(out_dir, spec, run_tag, setting, method, stage):
    return _cell_dir(out_dir, spec, run_tag, setting, method) / "{}_PLAN.json".format(
        stage.upper()
    )


def _write_plan(path, spec_path, spec, run_tag, setting, method, stage, jobs):
    payload = {
        "schema": PLAN_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "benchmark": spec["benchmark"],
        "protocol_stage": stage,
        "run_tag": run_tag,
        "setting": setting,
        "method": method,
        "git_commit": _git_commit(),
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "split": jobs[0]["split"],
        "order_seeds": list(ORDER_SEEDS),
        "job_count": len(jobs),
        "jobs": jobs,
    }
    if Path(path).is_file():
        if _canonical(_read_json(path)) != _canonical(payload):
            raise UserError("existing plan differs from requested immutable plan: {}".format(path))
    else:
        _atomic_json(path, payload)
    return payload


def _load_plan(path, spec_path, spec, run_tag, setting, method, stage):
    plan = _read_json(path)
    expected = {
        "schema": PLAN_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "benchmark": spec["benchmark"],
        "protocol_stage": stage,
        "run_tag": run_tag,
        "setting": setting,
        "method": method,
        "git_commit": _git_commit(),
        "spec_path": str(Path(spec_path).resolve()),
        "spec_sha256": _sha256(spec_path),
        "split": (
            spec["protocol"]["development_split"]
            if stage == "search" else spec["protocol"]["retention_split"]
        ),
        "order_seeds": list(ORDER_SEEDS),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise UserError("{} {} mismatch".format(path, key))
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != plan.get("job_count"):
        raise UserError("{} has an incomplete job matrix".format(path))
    registered = {
        item["candidate_id"]: item
        for item in _candidate_records_for_setting(spec, setting, method)
    }
    expected_count = len(registered) * len(ORDER_SEEDS) if stage == "search" else len(ORDER_SEEDS)
    if len(jobs) != expected_count:
        raise UserError("{} has the wrong number of {} jobs".format(path, stage))
    observed = []
    cell = Path(path).resolve().parent
    for job in jobs:
        if not isinstance(job, dict):
            raise UserError("{} contains a malformed job".format(path))
        candidate_id = job.get("candidate_id")
        seed = job.get("order_seed")
        candidate = registered.get(candidate_id)
        if candidate is None or type(seed) is not int or seed not in ORDER_SEEDS:
            raise UserError("{} contains an unregistered candidate/seed".format(path))
        observed.append((candidate_id, seed))
        expected_job_dir = cell / stage / candidate_id / "seed_{}".format(seed)
        expected_run_tag = _job_run_tag(
            run_tag, stage, setting, method, candidate_id, seed
        )
        expected_config_path = expected_job_dir / "tta_config.json"
        expected_result_root = (
            expected_job_dir / "result" / expected_run_tag / expected["split"]
        )
        fixed = {
            "stage": stage,
            "setting": setting,
            "model": MODEL_FOR_SETTING[setting],
            "method": method,
            "candidate_role": candidate["role"],
            "split": expected["split"],
            "run_tag": expected_run_tag,
            "job_dir": str(expected_job_dir),
            "config_path": str(expected_config_path),
            "result_root": str(expected_result_root),
            "formal_manifest": str(
                _formal_manifest_path(expected_run_tag, setting, expected["split"])
                .resolve()
            ),
        }
        for key, value in fixed.items():
            if job.get(key) != value:
                raise UserError("{} job {} mismatch".format(path, key))
        expected_parameters = candidate["parameters"]
        seeded_parameters = _parameters_for_seed(method, expected_parameters, seed)
        if (
            _canonical(job.get("candidate_parameters"))
            != _canonical(expected_parameters)
            or _canonical(job.get("parameters")) != _canonical(seeded_parameters)
        ):
            raise UserError("{} job parameters differ from the registered candidate".format(path))
        config_path = Path(job["config_path"])
        expected_config = {
            "schema": JOB_SCHEMA,
            "stage": "orders",
            "episodes": -1,
            "order_seed": seed,
            "search_method": method,
            "method": method,
            "parameters": seeded_parameters,
            "protocol_stage": stage,
            "candidate_id": candidate_id,
        }
        if (
            not config_path.is_file()
            or _canonical(_read_json(config_path)) != _canonical(expected_config)
            or job.get("config_sha256") != _sha256(config_path)
        ):
            raise UserError("{} job config binding is invalid".format(path))
        order = _order_metadata(setting, expected["split"], seed)
        order_expected = {
            "expected_benchmark": order["benchmark"],
            "expected_episode_count": order["episode_count"],
            "expected_order_sha256": order["order_sha256"],
            "expected_dataset_sha256": order["dataset_sha256"],
            "expected_order_manifest_sha256": order["file_sha256"],
        }
        for key, value in order_expected.items():
            if job.get(key) != value:
                raise UserError("{} job {} mismatch".format(path, key))
        command = job.get("command")
        if (
            not isinstance(command, list)
            or len(command) < 5
            or not str(command[4]).isdigit()
            or command != build_command(
                setting, expected["split"], expected_config_path,
                expected_result_root, expected_run_tag, seed, int(command[4]),
            )
        ):
            raise UserError("{} job launch command is invalid".format(path))
    if len(observed) != len(set(observed)):
        raise UserError("{} contains duplicate jobs".format(path))
    if stage == "search" and set(observed) != {
        (candidate_id, seed)
        for candidate_id in registered
        for seed in ORDER_SEEDS
    }:
        raise UserError("{} search job matrix is incomplete".format(path))
    if stage == "retention" and {
        seed for _, seed in observed
    } != set(ORDER_SEEDS):
        raise UserError("{} retention seed matrix is incomplete".format(path))
    return plan


def _preflight_method(spec, setting, method, expected_checkpoint_sha256=None):
    if setting == "streamvln-r2r-ce":
        raise UserError(
            "StreamVLN TTA search is blocked: the launcher rejects shuffled "
            "order seeds and the config translator has no StreamVLN adapter"
        )
    method_spec = spec["methods"][method]
    availability = method_spec.get("availability", {"status": "ready"})
    if availability.get("status") != "ready":
        raise UserError("{}/{} is blocked: {}".format(
            setting, method, availability.get("reason", "capability unavailable")
        ))
    if method == "idea":
        _idea_source_parameters(spec, setting, expected_checkpoint_sha256)
    if method == "atena":
        core_path = REPO_ROOT / "core/navtta_core/tta/tta_core.py"
        marker = '"exact_episode_replay_enabled": True'
        if marker not in core_path.read_text(encoding="utf-8"):
            raise UserError(
                "ATENA is blocked until exact full-episode replay capability is present"
            )


def _run_one(job, dry_run):
    printable = " ".join(job["command"])
    if dry_run:
        return printable
    result_root = Path(job["result_root"])
    if result_root.exists() and any(result_root.iterdir()):
        raise UserError("refusing to reuse non-empty result root: {}".format(result_root))
    completed = subprocess.run(job["command"], cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        raise UserError(
            "job failed with exit {}: {}".format(completed.returncode, printable)
        )
    return printable


def _run_jobs(jobs, concurrency, dry_run, launcher_preflight=False):
    commands = [" ".join(job["command"]) for job in jobs]
    if dry_run:
        if launcher_preflight and jobs:
            # Config translation already checks every candidate.  One shell
            # dry-run per cell additionally verifies the pinned environment,
            # checkpoint, dataset and baseline-specific command wiring without
            # starting a simulator or reserving GPU memory.
            probe = list(jobs[0]["command"]) + ["--dry-run"]
            completed = subprocess.run(probe, cwd=str(REPO_ROOT))
            if completed.returncode != 0:
                raise UserError(
                    "launcher preflight failed with exit {}: {}".format(
                        completed.returncode, " ".join(probe)
                    )
                )
        return commands
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run_one, job, False) for job in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return commands


def parse_console_metrics(text, split):
    values = {}
    matching = [line for line in text.splitlines() if "Env name: {}".format(split) in line]
    if matching:
        for key, value in re.findall(
            r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)",
            matching[-1],
        ):
            values[key.upper()] = float(value)
        return values
    for key, value in re.findall(
        r"Average episode ([A-Za-z0-9_]+):\s*(-?[0-9]+(?:\.[0-9]+)?)", text
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
    return values


def _metric_artifact(result_root, setting, split):
    result_root = Path(result_root)
    if setting in CONTINUOUS_SETTINGS:
        matches = []
        for path in result_root.rglob("stats_*_{}.json".format(split)):
            if re.search(r"_r\d+_w\d+$", path.stem):
                continue
            document = _read_json(path)
            if _finite(document.get("success")) and _finite(document.get("spl")):
                values = {key.upper(): float(value) for key, value in document.items() if _finite(value)}
                for key in ("SUCCESS", "ORACLE_SUCCESS", "SPL", "NDTW", "SDTW"):
                    if key in values:
                        values[key] *= 100.0
                values["SR"] = values["SUCCESS"]
                if "ORACLE_SUCCESS" in values:
                    values["OSR"] = values["ORACLE_SUCCESS"]
                matches.append((path, values))
    else:
        matches = []
        for path in result_root.rglob("valid.txt"):
            values = parse_console_metrics(
                path.read_text(encoding="utf-8", errors="replace"), split
            )
            if values:
                matches.append((path, values))
    if len(matches) != 1:
        raise UserError(
            "expected exactly one metric artifact under {}; found {}".format(
                result_root, len(matches)
            )
        )
    return matches[0]


def read_metrics(result_root, split, setting=None):
    """Read metrics, retaining console compatibility for small unit fixtures."""
    if setting is not None:
        return _metric_artifact(result_root, setting, split)[1]
    console = Path(result_root) / "console.log"
    if not console.is_file():
        raise UserError("missing console log: {}".format(console))
    values = parse_console_metrics(
        console.read_text(encoding="utf-8", errors="replace"), split
    )
    if not values:
        raise UserError("no validation metrics found in {}".format(console))
    return values


def _artifact_entry(manifest, result_root, required_path):
    relative = Path(required_path).resolve().relative_to(Path(result_root).resolve()).as_posix()
    matches = [
        item for item in manifest.get("result_artifacts", [])
        if isinstance(item, dict) and item.get("name") == relative
    ]
    if len(matches) != 1:
        raise UserError("formal manifest does not authenticate {}".format(relative))
    item = matches[0]
    if (
        item.get("size") != Path(required_path).stat().st_size
        or item.get("sha256") != _sha256(required_path)
    ):
        raise UserError("formal artifact digest mismatch: {}".format(relative))


def _validate_diagnostics(job, path):
    diagnostics = _read_json(path)
    if diagnostics.get("method") != job["method"]:
        raise UserError("TTA diagnostics method mismatch")
    episodes = diagnostics.get("episode_count")
    if episodes != job["expected_episode_count"]:
        raise UserError("TTA diagnostics episode count mismatch")
    adapter = diagnostics.get("adapter")
    if not isinstance(adapter, dict) or adapter.get("episodes") != episodes:
        raise UserError("adapter episode accounting mismatch")
    drift = adapter.get("relative_param_drift")
    updates = adapter.get("updates")
    if not _finite(drift) or float(drift) < 0.0:
        raise UserError("adapter relative_param_drift is missing or invalid")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise UserError("adapter updates is missing or invalid")
    supervised = job["method"] in {"feedtta", "atena"}
    if job.get("setting") in CONTINUOUS_SETTINGS:
        expected_supervision = "binary_episode_success" if supervised else "none"
        if diagnostics.get("feedback_supervision") != expected_supervision:
            raise UserError("continuous TTA supervision label mismatch")
    else:
        expected_supervision = (
            "binary_navigation_success_feedback" if supervised else "unsupervised"
        )
        if diagnostics.get("supervision") != expected_supervision:
            raise UserError("discrete TTA supervision label mismatch")
        endpoint = diagnostics.get("binary_feedback_endpoint")
        if supervised == (not isinstance(endpoint, str) or not endpoint):
            raise UserError("TTA binary-feedback endpoint mismatch")
    if job["method"] == "feedtta":
        if adapter.get("sgr_mode") != "paper_main":
            raise UserError("FeedTTA must attest sgr_mode=paper_main")
        if diagnostics.get("action_selection") != "target_native_argmax":
            raise UserError("FeedTTA formal VLN protocol requires target-native argmax")
        if adapter.get("action_selection_protocol") != "target_native_argmax":
            raise UserError("FeedTTA adapter action protocol is not target-native argmax")
        if (
            diagnostics.get("feedtta_sgr_mode") != "paper_main"
            or diagnostics.get("feedtta_scope_profile") != "paper_full"
            or diagnostics.get("feedtta_protocol")
            != "task_adapted_target_native_argmax"
            or diagnostics.get("feedtta_native_action_protocol") is not True
            or diagnostics.get("feedtta_paper_sampling_protocol") is not False
            or diagnostics.get("feedtta_canonical_protocol") is not False
        ):
            raise UserError("FeedTTA task-adapted native-action protocol mismatch")
    if job["method"] == "tent" and diagnostics.get(
        "tent_canonical_update_interval"
    ) is not True:
        raise UserError("Tent formal search requires update_interval=1")
    if job["method"] == "fstta" and (
        diagnostics.get("fstta_reset_var_hist_each_episode") is not False
        or diagnostics.get("fstta_variance_history_profile")
        != "paper_eq6_test_stream_history"
    ):
        raise UserError("FSTTA formal search requires paper Eq.6 stream history")
    if job["method"] == "atena":
        expected = {
            "exact_episode_replay_enabled": True,
            "gradient_reconstruction": "exact_step_replay_in_eval_mode",
            "retains_episode_graph": False,
            "requires_task_policy_scope_verification": True,
        }
        for key, value in expected.items():
            if adapter.get(key) != value:
                raise UserError("ATENA full-replay diagnostic {} mismatch".format(key))
        if (
            diagnostics.get("atena_update_scope")
            != "replay_reachable_high_level_navigation"
            or diagnostics.get("atena_exact_replay_within_declared_scope") is not True
            or diagnostics.get("atena_full_end_to_end_policy_claimed") is not False
            or diagnostics.get("atena_upstream_feature_extractors_adapted") is not False
        ):
            raise UserError("ATENA reachable policy scope evidence is incomplete")
    if job["method"] == "idea":
        provenance = adapter.get("source_statistics_provenance")
        if (
            adapter.get("source_statistics_mode") != "offline_artifact"
            or adapter.get("source_statistics_schema")
            != IDEA_SOURCE_STATS_SCHEMA
            or adapter.get("source_statistics_moment_estimator")
            != IDEA_SOURCE_STATS_ESTIMATOR
            or adapter.get("source_statistics_sha256")
            != job.get("parameters", {}).get("source_stats_sha256")
            or adapter.get("source_statistics_trajectory_count")
            != job.get("parameters", {}).get("source_trajectories")
            or not isinstance(provenance, dict)
            or provenance.get("collection_scope") != IDEA_SOURCE_STATS_SCOPE
            or provenance.get("collection_policy")
            != IDEA_SOURCE_COLLECTION_POLICY
            or provenance.get("setting") != job.get("setting")
            or provenance.get("model") != job.get("model")
        ):
            raise UserError("IDEA Source-statistics identity mismatch")
    result = {
        "relative_param_drift": float(drift),
        "updates": updates,
        "episode_count": episodes,
    }
    if job["method"] == "idea":
        result["source_checkpoint_sha256"] = provenance.get(
            "checkpoint_sha256"
        )
    return result


def validate_job_result(job, git_commit):
    config_path = Path(job["config_path"])
    if not config_path.is_file() or _sha256(config_path) != job["config_sha256"]:
        raise UserError("job config is missing or changed: {}".format(config_path))
    metric_path, metrics = _metric_artifact(
        job["result_root"], job["setting"], job["split"]
    )
    for metric in ("SPL", "SR") if "reverie" not in job["setting"] else (
        "RGSPL", "RGS", "SPL", "SR"
    ):
        if not _finite(metrics.get(metric)):
            raise UserError("required metric {} is missing or invalid".format(metric))
    diagnostics_path = Path(job["result_root"]) / "tta_diagnostics.json"
    diagnostics = _validate_diagnostics(job, diagnostics_path)
    manifest_path = Path(job["formal_manifest"])
    manifest = _read_json(manifest_path)
    expected_run_id = "{}-{}-{}-{}".format(
        job["run_tag"], job["setting"], job["split"], _data_version(job["setting"])
    )
    expected = {
        "run_id": expected_run_id,
        "task": "vln",
        "benchmark": job["expected_benchmark"],
        "model": job["model"],
        "method": job["method"],
        "run_tag": job["run_tag"],
        "seed": job["order_seed"],
        "git_commit": git_commit,
        "status": "completed",
        "exit_code": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise UserError("formal manifest {} mismatch for {}".format(key, job["run_tag"]))
    if manifest.get("source_setting") != "{}:{}:{}:{}".format(
        job["setting"], job["split"], _data_version(job["setting"]), job["method"]
    ):
        raise UserError("formal manifest source_setting mismatch")
    if Path(str(manifest.get("config", ""))).resolve() != config_path.resolve():
        raise UserError("formal manifest config path mismatch")
    if manifest.get("dataset", {}).get("stream_order_sha256") != job["expected_order_sha256"]:
        raise UserError("formal manifest stream-order digest mismatch")
    if manifest.get("dataset", {}).get("stream_content_sha256") != job["expected_dataset_sha256"]:
        raise UserError("formal manifest dataset digest mismatch")
    if manifest.get("pinned_manifests", {}).get("episode_order", {}).get("sha256") != job[
        "expected_order_manifest_sha256"
    ]:
        raise UserError("formal manifest order-file digest mismatch")
    if not _valid_sha256(manifest.get("checkpoint", {}).get("sha256")):
        raise UserError("formal manifest checkpoint digest is missing")
    auxiliary = manifest.get("auxiliary_checkpoints")
    if not isinstance(auxiliary, list):
        raise UserError("formal manifest auxiliary checkpoints are missing")
    auxiliary_by_name = {
        item.get("name"): item for item in auxiliary if isinstance(item, dict)
    }
    config_evidence = auxiliary_by_name.get("tta_job_config", {})
    if (
        config_evidence.get("sha256") != job["config_sha256"]
        or config_evidence.get("size") != config_path.stat().st_size
    ):
        raise UserError("formal manifest does not bind the TTA job config")
    if job["method"] == "idea":
        if diagnostics.get("source_checkpoint_sha256") != manifest["checkpoint"]["sha256"]:
            raise UserError("IDEA Source-statistics checkpoint digest mismatch")
        source_evidence = auxiliary_by_name.get("idea_source_statistics", {})
        source_path = Path(job["parameters"]["source_stats_path"])
        if (
            source_evidence.get("sha256")
            != job["parameters"]["source_stats_sha256"]
            or source_evidence.get("size") != source_path.stat().st_size
        ):
            raise UserError(
                "formal manifest does not bind IDEA Source statistics"
            )
    identity = manifest.get("immutable_identity_sha256")
    if not _valid_sha256(identity) or immutable_identity_sha256(manifest) != identity:
        raise UserError("formal manifest immutable identity mismatch")
    if not isinstance(manifest.get("result_artifacts"), list) or not manifest["result_artifacts"]:
        raise UserError("formal manifest has no result artifacts")
    result_root = Path(job["result_root"]).resolve()
    seen_artifacts = set()
    for artifact in manifest["result_artifacts"]:
        if not isinstance(artifact, dict):
            raise UserError("formal manifest contains a malformed result artifact")
        name = artifact.get("name")
        if not isinstance(name, str) or not name or name in seen_artifacts:
            raise UserError("formal manifest result artifact name is invalid")
        seen_artifacts.add(name)
        artifact_path = (result_root / name).resolve()
        try:
            artifact_path.relative_to(result_root)
        except ValueError:
            raise UserError("formal result artifact escapes the result root")
        if (
            not artifact_path.is_file()
            or artifact.get("size") != artifact_path.stat().st_size
            or artifact.get("sha256") != _sha256(artifact_path)
        ):
            raise UserError("formal result artifact digest mismatch: {}".format(name))
    _artifact_entry(manifest, job["result_root"], metric_path)
    _artifact_entry(manifest, job["result_root"], diagnostics_path)
    return {
        **job,
        "metrics": metrics,
        "metric_artifact": str(metric_path.resolve()),
        "metric_sha256": _sha256(metric_path),
        "diagnostics": diagnostics,
        "diagnostics_path": str(diagnostics_path.resolve()),
        "diagnostics_sha256": _sha256(diagnostics_path),
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "formal_manifest_sha256": _sha256(manifest_path),
        "formal_immutable_identity_sha256": identity,
    }


def _source_manifest_from_console(source_root):
    console = Path(source_root) / "console.log"
    if not console.is_file():
        raise UserError("missing Source console log: {}".format(console))
    candidates = sorted(set(
        line.strip() for line in console.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines() if line.strip().endswith("/manifest.json")
    ))
    if len(candidates) != 1:
        raise UserError("Source console must name exactly one formal manifest")
    original = Path(candidates[0])
    local = FORMAL_ROOT / original.parent.name / "manifest.json"
    return local if local.is_file() else original


def _benchmark_for_setting(setting):
    if setting.endswith("-reverie"):
        return "reverie"
    if setting.endswith("-r2r-ce"):
        return "r2r-ce"
    return "r2r"


def _source_evidence(source_root, setting, split, required_metrics):
    """Load Source only from the tracked, per-setting reuse ledger.

    ``source_root`` is retained as a CLI compatibility argument, but it is not
    an authority: a single symlink tree cannot represent the different R2R
    source batches.  The tracked ledger binds each setting independently.
    """
    del source_root
    ledger_relative = SOURCE_LEDGER[(_benchmark_for_setting(setting), split)]
    ledger_path = REPO_ROOT / ledger_relative
    ledger = _read_json(ledger_path)
    if ledger.get("split") != split:
        raise UserError("Source ledger split mismatch: {}".format(ledger_path))
    benchmark = _benchmark_for_setting(setting)
    expected_ledger_protocol = (
        "target_native_argmax" if benchmark == "r2r-ce"
        else "standard_argmax"
    )
    declared_ledger_protocol = ledger.get(
        "source_protocol", ledger.get("action_selection")
    )
    if declared_ledger_protocol != expected_ledger_protocol:
        raise UserError(
            "Source ledger action protocol mismatch: {}".format(ledger_path)
        )
    record = (ledger.get("records") or ledger.get("settings") or {}).get(setting)
    if not isinstance(record, dict):
        raise UserError("Source ledger lacks setting {}".format(setting))
    record_parameters = record.get("parameters")
    if isinstance(record_parameters, dict) and (
        record_parameters.get("action_selection") != "argmax"
        or record_parameters.get("matched_feedtta_source", False) is not False
    ):
        raise UserError(
            "Source ledger record is not an unmatched argmax control: {}"
            .format(setting)
        )

    metrics = record.get("metrics")
    metric_path = None
    metric_sha = None
    formal_metric_path = None
    formal_metric_sha = None
    formal_metric_size = None
    if isinstance(metrics, dict):
        metrics = {key.upper(): float(value) for key, value in metrics.items() if _finite(value)}
    else:
        aggregate = record.get("aggregate_artifact")
        if not isinstance(aggregate, dict):
            raise UserError("Source ledger lacks aggregate metrics for {}".format(setting))
        metric_path = REPO_ROOT / aggregate["path"]
        if (
            not metric_path.is_file()
            or _sha256(metric_path) != aggregate.get("sha256")
            or metric_path.stat().st_size != aggregate.get("size")
        ):
            raise UserError("Source aggregate artifact digest mismatch")
        raw = _read_json(metric_path)
        metrics = {key.upper(): float(value) for key, value in raw.items() if _finite(value)}
        for key in ("SUCCESS", "ORACLE_SUCCESS", "SPL", "NDTW", "SDTW"):
            if key in metrics:
                metrics[key] *= 100.0
        metrics["SR"] = metrics["SUCCESS"]
        metric_sha = aggregate["sha256"]
        formal_metric_path = metric_path
        formal_metric_sha = metric_sha
        formal_metric_size = metric_path.stat().st_size
    for metric in required_metrics:
        if not _finite(metrics.get(metric)):
            raise UserError("Source metric {} is missing for {}".format(metric, setting))

    artifact_value = record.get("metrics_artifact_path") or record.get("aggregate_artifact_path")
    artifact_sha = record.get("metrics_artifact_sha256") or record.get("aggregate_artifact_sha256")
    if artifact_value is not None:
        metric_path = REPO_ROOT / artifact_value
        if not metric_path.is_file() or _sha256(metric_path) != artifact_sha:
            raise UserError("Source metric artifact SHA256 mismatch")
        parsed = _metric_artifact(metric_path.parent.parent if metric_path.name == "valid.txt" else metric_path.parents[2], setting, split)[1]
        for metric in required_metrics:
            if not math.isclose(
                float(parsed[metric]), float(metrics[metric]),
                rel_tol=0.0, abs_tol=1e-9,
            ):
                raise UserError("Source ledger metric disagrees with its artifact")
        metric_sha = artifact_sha
        formal_metric_path = metric_path
        formal_metric_sha = artifact_sha
        formal_metric_size = metric_path.stat().st_size
    metrics_json_value = record.get("metrics_json_path")
    if metrics_json_value is not None:
        metrics_json = REPO_ROOT / metrics_json_value
        if not metrics_json.is_file() or _sha256(metrics_json) != record.get("metrics_json_sha256"):
            raise UserError("Source metrics JSON SHA256 mismatch")
        parsed = _read_json(metrics_json).get("metrics", {})
        for metric in required_metrics:
            if not math.isclose(
                float(parsed[metric]), float(metrics[metric]),
                rel_tol=0.0, abs_tol=1e-9,
            ):
                raise UserError("Source ledger metric disagrees with metrics JSON")
        metric_path = metrics_json
        metric_sha = record["metrics_json_sha256"]

    formal = record.get("formal_manifest")
    if isinstance(formal, dict):
        manifest_path = REPO_ROOT / formal["path"]
        expected_manifest_sha = formal["sha256"]
    else:
        manifest_path = REPO_ROOT / record["formal_manifest_path"]
        expected_manifest_sha = record["formal_manifest_sha256"]
    if not manifest_path.is_file() or _sha256(manifest_path) != expected_manifest_sha:
        raise UserError("Source formal-manifest SHA256 mismatch")
    manifest = _read_json(manifest_path)
    manifest_dataset = manifest.get("dataset")
    if (
        not isinstance(manifest_dataset, dict)
        or not _valid_sha256(manifest_dataset.get("stream_content_sha256"))
    ):
        raise UserError("Source formal manifest dataset identity is invalid")
    if (
        manifest.get("task") != "vln"
        or manifest.get("model") != MODEL_FOR_SETTING[setting]
        or manifest.get("method") != "source"
        or manifest.get("seed") != 0
        or manifest.get("status") != "completed"
        or manifest.get("exit_code") != 0
        or not _valid_sha256(manifest.get("checkpoint", {}).get("sha256"))
        or not _valid_sha256(manifest.get("immutable_identity_sha256"))
        or immutable_identity_sha256(manifest) != manifest["immutable_identity_sha256"]
    ):
        raise UserError("Source formal manifest identity is invalid for {}".format(setting))
    source_setting_base = "{}:{}:{}".format(
        setting, split, _data_version(setting)
    )
    if manifest.get("source_setting") not in {
        source_setting_base, source_setting_base + ":source"
    }:
        raise UserError("Source formal manifest split/setting mismatch")
    if manifest.get("checkpoint", {}).get("sha256") != record.get("checkpoint_sha256"):
        raise UserError("Source ledger checkpoint digest mismatch")
    record_dataset_sha = (
        record.get("dataset_sha256") or record.get("dataset_index_sha256")
    )
    if (
        record_dataset_sha is not None
        and record_dataset_sha != manifest_dataset["stream_content_sha256"]
    ):
        raise UserError("Source ledger dataset digest mismatch")
    if formal_metric_path is None:
        candidates = []
        for item in manifest.get("result_artifacts", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            if setting in CONTINUOUS_SETTINGS:
                selected = (
                    Path(name).name.startswith("stats_")
                    and Path(name).name.endswith("_{}.json".format(split))
                    and re.search(r"_r\d+_w\d+$", Path(name).stem) is None
                )
            else:
                selected = Path(name).name == "valid.txt"
            if not selected:
                continue
            recorded_path = Path(str(item.get("path", "")))
            local_path = recorded_path
            if not local_path.is_file() and "vln" in recorded_path.parts:
                # Historical formal manifests were created on AutoDL and keep
                # absolute server paths.  Resolve their repository-relative
                # suffix when authenticating a checked-out evidence bundle.
                offset = recorded_path.parts.index("vln")
                local_path = REPO_ROOT.joinpath(*recorded_path.parts[offset:])
            if local_path.is_file() and (
                item.get("size") != local_path.stat().st_size
                or item.get("sha256") != _sha256(local_path)
            ):
                raise UserError("Source formal metric artifact digest mismatch")
            if _valid_sha256(item.get("sha256")) and type(item.get("size")) is int:
                candidates.append((
                    local_path if local_path.is_file() else None,
                    item["sha256"],
                    item["size"],
                ))
        if len(candidates) != 1:
            raise UserError(
                "Source formal manifest must expose exactly one aggregate metric artifact"
            )
        formal_metric_path, formal_metric_sha, formal_metric_size = candidates[0]
        if formal_metric_path is not None and setting in CONTINUOUS_SETTINGS:
            parsed_source_metrics = {
                key.upper(): float(value)
                for key, value in _read_json(formal_metric_path).items()
                if _finite(value)
            }
            for key in ("SUCCESS", "ORACLE_SUCCESS", "SPL", "NDTW", "SDTW"):
                if key in parsed_source_metrics:
                    parsed_source_metrics[key] *= 100.0
            parsed_source_metrics["SR"] = parsed_source_metrics["SUCCESS"]
        elif formal_metric_path is not None:
            parsed_source_metrics = parse_console_metrics(
                formal_metric_path.read_text(encoding="utf-8", errors="replace"),
                split,
            )
        if formal_metric_path is not None:
            for metric in required_metrics:
                if not math.isclose(
                    float(parsed_source_metrics[metric]), float(metrics[metric]),
                    rel_tol=0.0, abs_tol=1e-9,
                ):
                    raise UserError("Source ledger metric disagrees with formal artifact")
    if formal_metric_sha is None or formal_metric_size is None:
        raise UserError("Source ledger has no authenticated metric artifact")
    matching_artifacts = [
        item for item in manifest.get("result_artifacts", [])
        if isinstance(item, dict)
        and item.get("sha256") == formal_metric_sha
        and item.get("size") == formal_metric_size
    ]
    if len(matching_artifacts) != 1:
        raise UserError("Source formal manifest does not authenticate its metric artifact")
    return {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "source_ledger": str(ledger_path.resolve()),
        "source_ledger_sha256": _sha256(ledger_path),
        "source_action_protocol": "target_native_argmax",
        "matched_feedtta_source": False,
        "metric_artifact": str(metric_path.resolve()) if metric_path else None,
        "metric_sha256": metric_sha,
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "dataset_sha256": manifest_dataset["stream_content_sha256"],
        "benchmark": manifest.get("benchmark"),
        "formal_manifest": str(Path(manifest_path).resolve()),
        "formal_manifest_sha256": expected_manifest_sha,
        "formal_immutable_identity_sha256": manifest["immutable_identity_sha256"],
    }


def _metric_summary(rows, metric, source_value):
    values = [float(row["metrics"][metric]) for row in rows]
    deltas = [value - float(source_value) for value in values]
    mean_delta = statistics.fmean(deltas)
    sample_std = statistics.stdev(deltas)
    return {
        "per_seed": {str(row["order_seed"]): float(row["metrics"][metric]) for row in rows},
        "delta_per_seed": {str(row["order_seed"]): delta for row, delta in zip(rows, deltas)},
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "mean_delta": mean_delta,
        "sample_std_delta": sample_std,
        "lcb": mean_delta - sample_std,
    }


def select_config(candidate_results, source_metrics, selection):
    """Rank complete three-seed candidates without a positive-result gate."""
    primary = selection["primary_metric"]
    secondary = selection["secondary_metric"]
    expected_seeds = tuple(selection.get("order_seeds", ORDER_SEEDS))
    grouped = {}
    for row in candidate_results:
        if row.get("split") != "val_unseen":
            raise UserError("selection accepts val_unseen development rows only")
        grouped.setdefault(row["candidate_id"], []).append(row)
    ranked = []
    for candidate_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["order_seed"])
        if tuple(row["order_seed"] for row in rows) != expected_seeds:
            raise UserError("candidate {} lacks complete seeds 1/2/3".format(candidate_id))
        parameters = rows[0]["candidate_parameters"]
        if any(_canonical(row["candidate_parameters"]) != _canonical(parameters) for row in rows):
            raise UserError("candidate {} changes parameters across seeds".format(candidate_id))
        primary_stats = _metric_summary(rows, primary, source_metrics[primary])
        secondary_stats = _metric_summary(rows, secondary, source_metrics[secondary])
        ranked.append({
            "candidate_id": candidate_id,
            "role": rows[0]["candidate_role"],
            "parameters": parameters,
            "primary": {"metric": primary, **primary_stats},
            "secondary": {"metric": secondary, **secondary_stats},
            "mean_relative_param_drift": statistics.fmean(
                row["diagnostics"]["relative_param_drift"] for row in rows
            ),
            "mean_updates": statistics.fmean(row["diagnostics"]["updates"] for row in rows),
            "selection_runs": {
                str(row["order_seed"]): {
                    "run_tag": row["run_tag"],
                    "formal_manifest": row["formal_manifest"],
                    "formal_manifest_sha256": row["formal_manifest_sha256"],
                    "diagnostics_sha256": row["diagnostics_sha256"],
                    "metric_sha256": row["metric_sha256"],
                } for row in rows
            },
        })
    if not ranked:
        raise UserError("cannot select from an empty candidate set")
    ranked.sort(key=lambda item: (
        -item["primary"]["lcb"],
        -item["secondary"]["lcb"],
        item["mean_relative_param_drift"],
        item["mean_updates"],
        item["candidate_id"],
    ))
    return ranked[0], ranked


def _validate_complete_matrix(plan, spec, source_root):
    results = [validate_job_result(job, plan["git_commit"]) for job in plan["jobs"]]
    candidates = _candidate_records_for_setting(
        spec, plan["setting"], plan["method"]
    )
    expected_parameters = {
        candidate["candidate_id"]: candidate["parameters"] for candidate in candidates
    }
    expected = {(candidate_id, seed) for candidate_id in expected_parameters for seed in ORDER_SEEDS}
    observed = [(row["candidate_id"], row["order_seed"]) for row in results]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise UserError("validated search results do not match the complete plan")
    for row in results:
        expected_base = expected_parameters[row["candidate_id"]]
        if _canonical(row["candidate_parameters"]) != _canonical(expected_base):
            raise UserError("candidate parameters differ from the registered specification")
        if _canonical(row["parameters"]) != _canonical(
            _parameters_for_seed(row["method"], expected_base, row["order_seed"])
        ):
            raise UserError("seeded job parameters differ from the registered specification")
    required_metrics = (
        spec["protocol"]["primary_metric"], spec["protocol"]["secondary_metric"]
    )
    source = _source_evidence(
        source_root, plan["setting"], plan["split"], required_metrics
    )
    if plan["method"] == "idea":
        _idea_source_parameters(
            spec, plan["setting"], source["checkpoint_sha256"]
        )
    checkpoint_digests = {row["checkpoint_sha256"] for row in results}
    if checkpoint_digests != {source["checkpoint_sha256"]}:
        raise UserError("candidate and Source checkpoint digests differ")
    dataset_digests = {row["expected_dataset_sha256"] for row in results}
    if dataset_digests != {source["dataset_sha256"]}:
        raise UserError("candidate and Source dataset digests differ")
    benchmark_names = {row["expected_benchmark"] for row in results}
    if benchmark_names != {source["benchmark"]}:
        raise UserError("candidate and Source benchmark identities differ")
    return results, source


def run_selection(spec_path, out_dir, source_root, run_tag, methods_filter,
                  settings_filter, final_stage=False):
    if final_stage:
        raise UserError("selection is development-only; final_stage is invalid")
    spec = load_spec(spec_path)
    settings = _selected(spec["settings"], settings_filter, "settings")
    methods = _selected(list(spec["methods"]), methods_filter, "methods")
    selections = {}
    for setting in settings:
        for method in methods:
            plan_path = _plan_path(out_dir, spec, run_tag, setting, method, "search")
            plan = _load_plan(plan_path, spec_path, spec, run_tag, setting, method, "search")
            results, source = _validate_complete_matrix(plan, spec, source_root)
            selection = {
                "primary_metric": spec["protocol"]["primary_metric"],
                "secondary_metric": spec["protocol"]["secondary_metric"],
                "order_seeds": list(ORDER_SEEDS),
            }
            winner, ranked = select_config(results, source["metrics"], selection)
            frozen = {
                "schema": FROZEN_SCHEMA,
                "experiment_id": spec["experiment_id"],
                "benchmark": spec["benchmark"],
                "setting": setting,
                "method": method,
                "git_commit": plan["git_commit"],
                "spec_sha256": plan["spec_sha256"],
                "search_plan": str(Path(plan_path).resolve()),
                "search_plan_sha256": _sha256(plan_path),
                "development_split": "val_unseen",
                "order_seeds": list(ORDER_SEEDS),
                "selection_rule": "mean_delta_minus_sample_std_delta",
                "positive_result_required": False,
                "val_seen_consulted": False,
                "source": source,
                "winner": winner,
                "ranked": ranked,
            }
            frozen_path = _cell_dir(out_dir, spec, run_tag, setting, method) / "FROZEN.json"
            if frozen_path.is_file() and _canonical(_read_json(frozen_path)) != _canonical(frozen):
                raise UserError("existing FROZEN.json differs from authenticated selection")
            _atomic_json(frozen_path, frozen)
            selections[(setting, method)] = frozen
            print("[freeze] {}/{} -> {} ({} LCB={:.6f})".format(
                setting, method, winner["candidate_id"],
                winner["primary"]["metric"], winner["primary"]["lcb"],
            ))
    return selections


def _load_frozen(out_dir, spec_path, spec, run_tag, setting, method):
    path = _cell_dir(out_dir, spec, run_tag, setting, method) / "FROZEN.json"
    frozen = _read_json(path)
    expected = {
        "schema": FROZEN_SCHEMA,
        "experiment_id": spec["experiment_id"],
        "benchmark": spec["benchmark"],
        "setting": setting,
        "method": method,
        "git_commit": _git_commit(),
        "spec_sha256": _sha256(spec_path),
        "development_split": "val_unseen",
        "order_seeds": list(ORDER_SEEDS),
        "positive_result_required": False,
        "val_seen_consulted": False,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise UserError("FROZEN.json {} mismatch".format(key))
    plan_path = Path(frozen.get("search_plan", ""))
    if not plan_path.is_file() or _sha256(plan_path) != frozen.get("search_plan_sha256"):
        raise UserError("FROZEN.json search-plan binding is invalid")
    return path, frozen


def run_search(spec_path, run_tag, methods_filter, settings_filter, out_dir,
               concurrency_override, dry_run, final_stage=False, gpu=0,
               source_root=None, launcher_preflight=False):
    if launcher_preflight and not dry_run:
        raise UserError("launcher preflight requires --dry-run")
    spec = load_spec(spec_path)
    stage = "retention" if final_stage else "search"
    settings = _selected(spec["settings"], settings_filter, "settings")
    methods = _selected(list(spec["methods"]), methods_filter, "methods")
    commands = []
    # Prove every requested cell launchable before starting the first GPU job;
    # otherwise a late blocked method would leave a misleading partial batch.
    for setting in settings:
        source = None
        if stage == "search":
            source = _source_evidence(
                None, setting, spec["protocol"]["development_split"],
                (
                    spec["protocol"]["primary_metric"],
                    spec["protocol"]["secondary_metric"],
                ),
            )
        for method in methods:
            if method == "idea" and source is None:
                source = _source_evidence(
                    None, setting, spec["protocol"]["development_split"],
                    (
                        spec["protocol"]["primary_metric"],
                        spec["protocol"]["secondary_metric"],
                    ),
                )
            _preflight_method(
                spec,
                setting,
                method,
                None if source is None else source["checkpoint_sha256"],
            )
    for setting in settings:
        for method in methods:
            if stage == "search":
                candidates = _candidate_records_for_setting(spec, setting, method)
            else:
                # Recompute the freeze from development evidence before allowing
                # any val_seen process to start.
                frozen_now = run_selection(
                    spec_path, out_dir, source_root, run_tag, {method}, {setting}
                )[(setting, method)]
                _, frozen = _load_frozen(out_dir, spec_path, spec, run_tag, setting, method)
                if _canonical(frozen_now) != _canonical(frozen):
                    raise UserError("frozen selection changed before retention")
                candidates = [{
                    "candidate_id": frozen["winner"]["candidate_id"],
                    "role": frozen["winner"]["role"],
                    "ordinal": 0,
                    "parameters": frozen["winner"]["parameters"],
                }]
            jobs = _build_jobs(
                spec, out_dir, run_tag, setting, method, stage, candidates, gpu
            )
            plan_path = _plan_path(out_dir, spec, run_tag, setting, method, stage)
            _write_plan(plan_path, spec_path, spec, run_tag, setting, method, stage, jobs)
            cap = spec["concurrency"][setting][method]
            if concurrency_override is not None:
                if concurrency_override < 1 or concurrency_override > cap:
                    raise UserError(
                        "--concurrency must be in [1, {}] for {}/{}".format(
                            cap, setting, method
                        )
                    )
                cap = concurrency_override
            commands.extend(
                _run_jobs(jobs, cap, dry_run, launcher_preflight)
            )
    if dry_run:
        print("# DRY RUN: {} {} commands".format(len(commands), stage))
        for command in commands:
            print(command)
    return commands


def run_report(spec_path, out_dir, source_root, run_tag, methods_filter, settings_filter):
    spec = load_spec(spec_path)
    settings = _selected(spec["settings"], settings_filter, "settings")
    methods = _selected(list(spec["methods"]), methods_filter, "methods")
    reports = {}
    for setting in settings:
        for method in methods:
            # Re-authenticate all development evidence at report time.  This
            # prevents a post-retention edit of FROZEN.json from changing the
            # reported winner without rerunning selection.
            frozen_now = run_selection(
                spec_path, out_dir, source_root, run_tag, {method}, {setting}
            )[(setting, method)]
            frozen_path, frozen = _load_frozen(
                out_dir, spec_path, spec, run_tag, setting, method
            )
            if _canonical(frozen_now) != _canonical(frozen):
                raise UserError("frozen selection changed before report")
            plan_path = _plan_path(out_dir, spec, run_tag, setting, method, "retention")
            plan = _load_plan(
                plan_path, spec_path, spec, run_tag, setting, method, "retention"
            )
            rows = [validate_job_result(job, plan["git_commit"]) for job in plan["jobs"]]
            if len(rows) != 3 or tuple(sorted(row["order_seed"] for row in rows)) != ORDER_SEEDS:
                raise UserError("retention requires exactly winner seeds 1/2/3")
            if any(row["candidate_id"] != frozen["winner"]["candidate_id"] for row in rows):
                raise UserError("retention plan contains a non-winner candidate")
            for row in rows:
                if _canonical(row["candidate_parameters"]) != _canonical(
                    frozen["winner"]["parameters"]
                ) or _canonical(row["parameters"]) != _canonical(
                    _parameters_for_seed(
                        method, frozen["winner"]["parameters"], row["order_seed"]
                    )
                ):
                    raise UserError("retention parameters differ from the frozen winner")
            required = (
                spec["protocol"]["primary_metric"], spec["protocol"]["secondary_metric"]
            )
            source = _source_evidence(source_root, setting, "val_seen", required)
            if {row["checkpoint_sha256"] for row in rows} != {source["checkpoint_sha256"]}:
                raise UserError("retention and Source checkpoint digests differ")
            if {row["expected_dataset_sha256"] for row in rows} != {
                source["dataset_sha256"]
            }:
                raise UserError("retention and Source dataset digests differ")
            if {row["expected_benchmark"] for row in rows} != {
                source["benchmark"]
            }:
                raise UserError("retention and Source benchmark identities differ")
            retention = {
                metric: _metric_summary(rows, metric, source["metrics"][metric])
                for metric in required
            }
            report = {
                "schema": REPORT_SCHEMA,
                "experiment_id": spec["experiment_id"],
                "benchmark": spec["benchmark"],
                "setting": setting,
                "method": method,
                "git_commit": plan["git_commit"],
                "spec_sha256": plan["spec_sha256"],
                "frozen_path": str(frozen_path.resolve()),
                "frozen_sha256": _sha256(frozen_path),
                "retention_plan": str(Path(plan_path).resolve()),
                "retention_plan_sha256": _sha256(plan_path),
                "winner": frozen["winner"],
                "development_selection": {
                    "split": "val_unseen",
                    "source": frozen["source"],
                    "primary": frozen["winner"]["primary"],
                    "secondary": frozen["winner"]["secondary"],
                },
                "post_freeze_retention": {
                    "split": "val_seen",
                    "used_for_selection": False,
                    "source": source,
                    "metrics": retention,
                    "runs": {
                        str(row["order_seed"]): {
                            "run_tag": row["run_tag"],
                            "formal_manifest": row["formal_manifest"],
                            "formal_manifest_sha256": row["formal_manifest_sha256"],
                            "diagnostics_sha256": row["diagnostics_sha256"],
                            "metric_sha256": row["metric_sha256"],
                        } for row in rows
                    },
                },
            }
            path = _cell_dir(out_dir, spec, run_tag, setting, method) / "selected_config.json"
            if path.is_file() and _canonical(_read_json(path)) != _canonical(report):
                raise UserError("existing selected_config.json differs from authenticated report")
            _atomic_json(path, report)
            reports[(setting, method)] = report
            print("[report] {}/{} -> {}".format(
                setting, method, frozen["winner"]["candidate_id"]
            ))
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--run-tag", default="consistency-v2")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stage", choices=("search", "select", "retention", "report"),
        default="search",
    )
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=None,
                        help="may lower, but never raise, the registered cell cap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--launcher-preflight", action="store_true",
        help="with --dry-run, execute one baseline shell dry-run per cell",
    )
    parser.add_argument(
        "--source-root", default=None, help=argparse.SUPPRESS,
    )
    # Backward-compatible spellings; their corrected semantics are strict.
    parser.add_argument("--select-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--final-stage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu < 0:
        raise UserError("--gpu must be nonnegative")
    stage = args.stage
    if args.select_only:
        if stage != "search" or args.final_stage:
            raise UserError("legacy stage flags cannot be combined")
        stage = "select"
    elif args.final_stage:
        if stage != "search":
            raise UserError("legacy stage flags cannot be combined")
        stage = "retention"
    if args.dry_run and stage not in ("search", "retention"):
        raise UserError("--dry-run applies only to launch stages")
    if args.launcher_preflight and not args.dry_run:
        raise UserError("--launcher-preflight requires --dry-run")
    methods = set(args.methods or ())
    settings = set(args.settings or ())
    if stage == "search":
        run_search(
            args.spec, args.run_tag, methods, settings, args.out_dir,
            args.concurrency, args.dry_run, False, args.gpu,
            launcher_preflight=args.launcher_preflight,
        )
    elif stage == "select":
        run_selection(
            args.spec, args.out_dir, args.source_root, args.run_tag,
            methods, settings,
        )
    elif stage == "retention":
        run_search(
            args.spec, args.run_tag, methods, settings, args.out_dir,
            args.concurrency, args.dry_run, True, args.gpu, args.source_root,
            args.launcher_preflight,
        )
    else:
        run_report(
            args.spec, args.out_dir, args.source_root, args.run_tag,
            methods, settings,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
