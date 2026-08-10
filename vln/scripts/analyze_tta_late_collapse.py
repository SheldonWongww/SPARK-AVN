#!/usr/bin/env python3
"""Posthoc fixed-order late-collapse analysis for a completed VLN TTA search.

The search protocol deliberately defers late-stream stability review until the
full ``val_seen`` finalists have completed.  This utility performs that review
without importing a model or mutating scheduler evidence.

Two evidence levels are intentionally distinguished:

* DUET/HAMT/GOAT only retain aggregate metrics.  Their first-256 screening run
  and full-stream run permit an algebraic remainder estimate, but no
  per-episode uncertainty or final-tail localization.  Those results are
  labelled ``coarse_fixed_order``.
* ETPNav/BEVBert retain one metric record per canonical episode.  Their paired
  TTA-minus-Source sequence supports chronological quartiles, rolling windows,
  and a deterministic circular moving-block bootstrap.  It is still a
  fixed-order analysis; robustness across orders requires separate runs.

Only JSON and CSV files are written.  Checkpoints, predictions, and raw model
outputs are neither read nor copied.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "vln/experiments/tta_hparam_search_v1.json"
DEFAULT_ORDER_ROOT = REPO_ROOT / "vln/manifests/episode_order"
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
LAST_SCREENING_STAGE = {
    "tent": "stage1",
    "fstta": "stage3",
    "eam": "stage3",
    "feedtta": "stage2",
    "atena": "stage3",
}
CONTINUOUS_SETTINGS = {"etpnav-r2r-ce", "bevbert-r2r-ce"}
SETTING_ORDER_DIRECTORY = {
    "duet-r2r": "r2r_duet_hamt",
    "duet-reverie": "reverie_duet_hamt",
    "hamt-r2r": "r2r_duet_hamt",
    "hamt-reverie": "reverie_duet_hamt",
    "goat-r2r": "r2r_goat",
    "goat-reverie": "reverie_goat",
    "etpnav-r2r-ce": "r2r_ce_v1_3_unified",
    "bevbert-r2r-ce": "r2r_ce_v1_3_unified",
}
PER_EPISODE_SCHEMA = "navtta.vln_per_episode_metrics.v1"
OUTPUT_SCHEMA = "navtta.vln_tta_late_collapse_analysis.v1"


class CollapseError(RuntimeError):
    """Evidence is incomplete, inconsistent, or unsuitable for the audit."""


def read_json(path):
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise CollapseError("cannot read JSON {}: {}".format(path, error))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollapseError("{} is not numeric".format(label))
    value = float(value)
    if not math.isfinite(value):
        raise CollapseError("{} is non-finite".format(label))
    return value


def validate_metric_mapping(metrics, label):
    if not isinstance(metrics, dict) or not metrics:
        raise CollapseError("{} is not a nonempty metric mapping".format(label))
    output = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise CollapseError("{} has an invalid metric name".format(label))
        output[key.lower()] = finite_number(
            value, "{}.{}".format(label, key)
        )
    return output


def percentile(values, probability):
    if not values:
        raise CollapseError("cannot take a percentile of an empty sequence")
    if probability < 0.0 or probability > 1.0:
        raise CollapseError("percentile probability is outside [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _circular_block_mean(values, block_length, rng):
    if not values:
        raise CollapseError("block bootstrap received an empty segment")
    if block_length < 1:
        raise CollapseError("block length must be positive")
    total = 0.0
    collected = 0
    size = len(values)
    while collected < size:
        start = rng.randrange(size)
        take = min(block_length, size - collected)
        for offset in range(take):
            total += values[(start + offset) % size]
        collected += take
    return total / size


def _normal_lower_tail(z_value):
    return 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))


def _bootstrap_summary(observed, replicates, seed, block_length, samples):
    standard_error = (
        statistics.stdev(replicates) if len(replicates) > 1 else 0.0
    )
    if standard_error == 0.0:
        if observed < 0.0:
            p_value = 0.0
        elif observed > 0.0:
            p_value = 1.0
        else:
            p_value = 0.5
    else:
        p_value = _normal_lower_tail(observed / standard_error)
    return {
        "estimate_percentage_points": observed,
        "ci95_lower_percentage_points": percentile(replicates, 0.025),
        "ci95_upper_percentage_points": percentile(replicates, 0.975),
        "one_sided_upper95_percentage_points": percentile(replicates, 0.95),
        "bootstrap_standard_error": standard_error,
        "one_sided_negative_p_value": min(1.0, max(0.0, p_value)),
        "bootstrap_seed": int(seed),
        "bootstrap_samples": int(samples),
        "block_length": int(block_length),
        "bootstrap": "independent_segment_circular_moving_blocks",
        "p_value_method": "normal_approximation_from_block_bootstrap_se",
    }


def block_bootstrap_contrast(first, last, samples=2000, block_length=32,
                             seed=0):
    """Bootstrap ``mean(last) - mean(first)`` using local circular blocks."""
    first = [finite_number(value, "first segment") for value in first]
    last = [finite_number(value, "last segment") for value in last]
    if not first or not last:
        raise CollapseError("contrast requires nonempty first and last segments")
    if samples < 2:
        raise CollapseError("bootstrap_samples must be at least 2")
    observed = statistics.mean(last) - statistics.mean(first)
    rng = random.Random(int(seed))
    replicates = [
        _circular_block_mean(last, block_length, rng)
        - _circular_block_mean(first, block_length, rng)
        for _ in range(int(samples))
    ]
    return _bootstrap_summary(
        observed, replicates, seed, block_length, samples
    )


def block_bootstrap_mean(values, samples=2000, block_length=32, seed=0):
    values = [finite_number(value, "bootstrap segment") for value in values]
    if not values:
        raise CollapseError("mean bootstrap requires a nonempty segment")
    if samples < 2:
        raise CollapseError("bootstrap_samples must be at least 2")
    observed = statistics.mean(values)
    rng = random.Random(int(seed))
    replicates = [
        _circular_block_mean(values, block_length, rng)
        for _ in range(int(samples))
    ]
    return _bootstrap_summary(
        observed, replicates, seed, block_length, samples
    )


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values for a key -> p mapping."""
    if not isinstance(p_values, dict) or not p_values:
        raise CollapseError("Holm adjustment requires at least one p-value")
    ordered = []
    for key, value in p_values.items():
        value = finite_number(value, "Holm p-value {}".format(key))
        if value < 0.0 or value > 1.0:
            raise CollapseError("Holm p-value is outside [0, 1]")
        ordered.append((value, key))
    ordered.sort(key=lambda item: (item[0], str(item[1])))
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, (value, key) in enumerate(ordered):
        candidate = min(1.0, (count - index) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def stable_seed(base_seed, *parts):
    text = ":".join([str(int(base_seed))] + [str(part) for part in parts])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def rolling_means(values, window):
    if window < 1:
        raise CollapseError("rolling window must be positive")
    if len(values) < window:
        raise CollapseError(
            "rolling window {} exceeds {} episodes".format(window, len(values))
        )
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + finite_number(value, "rolling value"))
    return [
        (prefix[index + window] - prefix[index]) / window
        for index in range(len(values) - window + 1)
    ]


def derive_discrete_metric(tta_prefix, tta_full, source_prefix, source_full,
                           episode_count, prefix_count,
                           rounding_half_unit=0.005):
    """Derive the post-prefix remainder and matched chronological erosion."""
    values = [tta_prefix, tta_full, source_prefix, source_full]
    values = [finite_number(value, "discrete aggregate") for value in values]
    tta_prefix, tta_full, source_prefix, source_full = values
    episode_count = int(episode_count)
    prefix_count = int(prefix_count)
    if not 0 < prefix_count < episode_count:
        raise CollapseError("prefix_count must be in (0, episode_count)")
    remainder_count = episode_count - prefix_count
    tta_suffix = (
        episode_count * tta_full - prefix_count * tta_prefix
    ) / remainder_count
    source_suffix = (
        episode_count * source_full - prefix_count * source_prefix
    ) / remainder_count
    prefix_delta = tta_prefix - source_prefix
    suffix_delta = tta_suffix - source_suffix
    erosion = suffix_delta - prefix_delta

    half = finite_number(rounding_half_unit, "rounding_half_unit")
    if half < 0.0:
        raise CollapseError("rounding_half_unit must be nonnegative")
    suffix_mean_bound = (
        (episode_count + prefix_count) * half / remainder_count
    )
    prefix_delta_bound = 2.0 * half
    suffix_delta_bound = 2.0 * suffix_mean_bound
    erosion_bound = prefix_delta_bound + suffix_delta_bound
    return {
        "episode_count": episode_count,
        "prefix_count": prefix_count,
        "suffix_count": remainder_count,
        "tta_prefix_mean": tta_prefix,
        "tta_full_mean": tta_full,
        "tta_suffix_mean": tta_suffix,
        "source_prefix_mean": source_prefix,
        "source_full_mean": source_full,
        "source_suffix_mean": source_suffix,
        "matched_prefix_delta": prefix_delta,
        "matched_suffix_delta": suffix_delta,
        "matched_delta_erosion": erosion,
        "rounding_bounds": {
            "input_aggregate": half,
            "derived_suffix_mean": suffix_mean_bound,
            "matched_prefix_delta": prefix_delta_bound,
            "matched_suffix_delta": suffix_delta_bound,
            "matched_delta_erosion": erosion_bound,
        },
    }


def analyze_discrete_aggregates(tta_prefix_metrics, tta_full_metrics,
                                source_prefix_metrics, source_full_metrics,
                                setting, episode_count, prefix_count,
                                threshold_pp=2.0):
    metric_sets = []
    normalized = []
    for label, metrics in (
        ("TTA prefix", tta_prefix_metrics),
        ("TTA full", tta_full_metrics),
        ("Source prefix", source_prefix_metrics),
        ("Source full", source_full_metrics),
    ):
        mapped = validate_metric_mapping(metrics, label)
        normalized.append(mapped)
        metric_sets.append(set(mapped))
    common = set.intersection(*metric_sets)
    primary = "rgspl" if setting.endswith("reverie") else "spl"
    for required in (primary, "sr"):
        if required not in common:
            raise CollapseError(
                "{} aggregate evidence lacks {}".format(setting, required.upper())
            )
    metric_results = {}
    for metric in sorted(common):
        metric_results[metric.upper()] = derive_discrete_metric(
            normalized[0][metric], normalized[1][metric],
            normalized[2][metric], normalized[3][metric],
            episode_count, prefix_count,
        )

    primary_result = metric_results[primary.upper()]
    sr_result = metric_results["SR"]
    threshold = finite_number(threshold_pp, "threshold_pp")
    if threshold <= 0.0:
        raise CollapseError("threshold_pp must be positive")
    nominal_erosion = primary_result["matched_delta_erosion"] <= -threshold
    robust_erosion = (
        primary_result["matched_delta_erosion"]
        + primary_result["rounding_bounds"]["matched_delta_erosion"]
        <= -threshold
    )
    nominal_tail = (
        primary_result["matched_suffix_delta"] <= -threshold
        or sr_result["matched_suffix_delta"] <= -threshold
    )
    robust_tail = (
        primary_result["matched_suffix_delta"]
        + primary_result["rounding_bounds"]["matched_suffix_delta"]
        <= -threshold
        or sr_result["matched_suffix_delta"]
        + sr_result["rounding_bounds"]["matched_suffix_delta"]
        <= -threshold
    )
    robust_flag = bool(robust_erosion and robust_tail)
    nominal_flag = bool(nominal_erosion and nominal_tail)
    return {
        "analysis_level": "coarse_fixed_order_prefix_remainder",
        "primary_metric": primary.upper(),
        "metric_unit": "native_console_units; bounded metrics are percentage_points",
        "materiality_threshold_percentage_points": threshold,
        "metrics": metric_results,
        "coarse_erosion_2pp_nominal": bool(nominal_erosion),
        "coarse_tail_deficit_2pp_nominal": bool(nominal_tail),
        "coarse_2pp_flag_nominal": nominal_flag,
        "coarse_erosion_2pp_rounding_robust": bool(robust_erosion),
        "coarse_tail_deficit_2pp_rounding_robust": bool(robust_tail),
        "coarse_2pp_flag": robust_flag,
        "classification": "coarse_flag" if robust_flag else (
            "coarse_warning" if nominal_erosion or nominal_tail else "no_coarse_flag"
        ),
        "limitations": [
            "Only the first canonical prefix and its complementary remainder are identifiable.",
            "No per-episode confidence interval or final-tail change point is available.",
            "A collapse confined to the last 10-20 percent can be diluted in the remainder mean.",
            "The result is a fixed-order diagnostic, not an order-robustness claim.",
        ],
    }


def load_canonical_order(order_root, setting, expected_count=None):
    directory = SETTING_ORDER_DIRECTORY.get(setting)
    if directory is None:
        raise CollapseError("no canonical order mapping for {}".format(setting))
    path = Path(order_root) / directory / "val_seen.json"
    document = read_json(path)
    if document.get("schema") != "navtta.episode_order.v1":
        raise CollapseError("unsupported episode-order schema in {}".format(path))
    if document.get("split") != "val_seen":
        raise CollapseError("canonical episode-order manifest is not val_seen")
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise CollapseError("canonical order has no episodes: {}".format(path))
    count = int(document.get("episode_count", -1))
    if count != len(episodes):
        raise CollapseError("canonical episode_count/list length mismatch")
    if expected_count is not None and count != int(expected_count):
        raise CollapseError(
            "canonical order count {} != expected {}".format(count, expected_count)
        )
    output = []
    seen = set()
    for ordinal, item in enumerate(episodes):
        if not isinstance(item, dict) or "episode_id" not in item:
            raise CollapseError("canonical episode {} is invalid".format(ordinal))
        episode_id = str(item["episode_id"])
        if episode_id in seen:
            raise CollapseError("canonical episode id is duplicated: {}".format(episode_id))
        seen.add(episode_id)
        output.append({
            "ordinal": ordinal,
            "episode_id": episode_id,
            "scene_id": None if item.get("scene_id") is None else str(item["scene_id"]),
        })
    return {
        "path": str(path),
        "sha256": sha256(path),
        "order_sha256": document.get("order_sha256"),
        "episodes": output,
    }


def validate_per_episode_document(document, canonical_order, setting=None,
                                  run_tag=None, source_path=None):
    """Normalize the portable per-episode schema and reject order drift."""
    if not isinstance(document, dict):
        raise CollapseError("per-episode metrics document is not an object")
    if document.get("schema") != PER_EPISODE_SCHEMA:
        raise CollapseError("unsupported per-episode metrics schema")
    if setting is not None and document.get("setting") != setting:
        raise CollapseError("per-episode metrics setting mismatch")
    if run_tag is not None and document.get("run_tag") != run_tag:
        raise CollapseError("per-episode metrics run_tag mismatch")
    if document.get("split") != "val_seen":
        raise CollapseError("per-episode metrics split is not val_seen")
    expected = canonical_order["episodes"]
    if int(document.get("episode_count", -1)) != len(expected):
        raise CollapseError("per-episode metrics episode_count mismatch")
    recorded_order_hash = document.get("order_sha256")
    canonical_order_hash = canonical_order.get("order_sha256")
    if (recorded_order_hash is not None and canonical_order_hash is not None
            and recorded_order_hash != canonical_order_hash):
        raise CollapseError("per-episode metrics order_sha256 mismatch")
    recorded_manifest_hash = document.get("episode_order_manifest_sha256")
    canonical_manifest_hash = canonical_order.get("sha256")
    if (recorded_manifest_hash is not None and canonical_manifest_hash is not None
            and recorded_manifest_hash != canonical_manifest_hash):
        raise CollapseError(
            "per-episode metrics episode-order manifest SHA256 mismatch"
        )
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(expected):
        raise CollapseError("per-episode metrics episodes are missing or incomplete")
    output = []
    for ordinal, (actual, wanted) in enumerate(zip(episodes, expected)):
        if not isinstance(actual, dict):
            raise CollapseError("per-episode record {} is invalid".format(ordinal))
        if actual.get("ordinal") != ordinal:
            raise CollapseError("per-episode ordinal mismatch at {}".format(ordinal))
        if str(actual.get("episode_id")) != wanted["episode_id"]:
            raise CollapseError("per-episode order mismatch at {}".format(ordinal))
        scene = actual.get("scene_id")
        if (scene is not None and wanted.get("scene_id") is not None
                and str(scene) != wanted["scene_id"]):
            raise CollapseError("per-episode scene mismatch at {}".format(ordinal))
        metrics = validate_metric_mapping(
            actual.get("metrics"), "episode {} metrics".format(ordinal)
        )
        for required in ("success", "spl"):
            if required not in metrics:
                raise CollapseError(
                    "episode {} lacks {}".format(ordinal, required)
                )
        if metrics["success"] not in (0.0, 1.0):
            raise CollapseError("episode {} success is not binary".format(ordinal))
        if metrics["spl"] < 0.0 or metrics["spl"] > 1.0:
            raise CollapseError("episode {} SPL is outside [0, 1]".format(ordinal))
        output.append({
            "ordinal": ordinal,
            "episode_id": wanted["episode_id"],
            "scene_id": wanted.get("scene_id"),
            "metrics": metrics,
        })
    return {
        "schema": PER_EPISODE_SCHEMA,
        "setting": setting or document.get("setting"),
        "run_tag": run_tag or document.get("run_tag"),
        "split": "val_seen",
        "episode_count": len(output),
        "episodes": output,
        "evidence_path": None if source_path is None else str(source_path),
        "evidence_sha256": (
            None if source_path is None else sha256(source_path)
        ),
    }


def validate_raw_stats_episode(document, canonical_order, setting=None,
                               run_tag=None, source_path=None):
    """Normalize the native ETPNav/BEVBert ``stats_ep`` JSON fallback."""
    if not isinstance(document, dict) or not document:
        raise CollapseError("native stats_ep document is empty or invalid")
    expected = canonical_order["episodes"]
    actual_ids = [str(episode_id) for episode_id in document.keys()]
    expected_ids = [item["episode_id"] for item in expected]
    if actual_ids != expected_ids:
        raise CollapseError("native stats_ep output is incomplete or out of order")
    output = []
    for ordinal, wanted in enumerate(expected):
        raw_metrics = document[actual_ids[ordinal]]
        metrics = validate_metric_mapping(
            raw_metrics, "episode {} metrics".format(ordinal)
        )
        for required in ("success", "spl"):
            if required not in metrics:
                raise CollapseError("episode {} lacks {}".format(ordinal, required))
        if metrics["success"] not in (0.0, 1.0):
            raise CollapseError("episode {} success is not binary".format(ordinal))
        if metrics["spl"] < 0.0 or metrics["spl"] > 1.0:
            raise CollapseError("episode {} SPL is outside [0, 1]".format(ordinal))
        output.append({
            "ordinal": ordinal,
            "episode_id": wanted["episode_id"],
            "scene_id": wanted.get("scene_id"),
            "metrics": metrics,
        })
    return {
        "schema": "native_stats_ep_fallback",
        "setting": setting,
        "run_tag": run_tag,
        "split": "val_seen",
        "episode_count": len(output),
        "episodes": output,
        "evidence_path": None if source_path is None else str(source_path),
        "evidence_sha256": (
            None if source_path is None else sha256(source_path)
        ),
    }


def _metric_delta_series(tta, source, metric):
    if len(tta["episodes"]) != len(source["episodes"]):
        raise CollapseError("TTA and Source per-episode counts differ")
    output = []
    for ordinal, (tta_episode, source_episode) in enumerate(zip(
            tta["episodes"], source["episodes"])):
        if (tta_episode["ordinal"] != ordinal
                or source_episode["ordinal"] != ordinal
                or tta_episode["episode_id"] != source_episode["episode_id"]):
            raise CollapseError("TTA and Source episode order differs at {}".format(
                ordinal
            ))
        if metric not in tta_episode["metrics"]:
            raise CollapseError("TTA episode {} lacks {}".format(ordinal, metric))
        if metric not in source_episode["metrics"]:
            raise CollapseError("Source episode {} lacks {}".format(ordinal, metric))
        # Native CE bounded metrics are in [0, 1].  Convert their paired
        # difference to percentage points for one shared materiality policy.
        output.append(100.0 * (
            tta_episode["metrics"][metric]
            - source_episode["metrics"][metric]
        ))
    return output


def analyze_continuous_per_episode(tta, source, method, setting, run_tag,
                                   bootstrap_samples=2000, block_length=32,
                                   rolling_window=64, bootstrap_seed=0,
                                   threshold_pp=2.0):
    threshold = finite_number(threshold_pp, "threshold_pp")
    if threshold <= 0:
        raise CollapseError("threshold_pp must be positive")
    if tta["episode_count"] != source["episode_count"]:
        raise CollapseError("TTA and Source per-episode counts differ")
    episode_count = int(tta["episode_count"])
    quartile_size = episode_count // 4
    if quartile_size < 1:
        raise CollapseError("at least four episodes are required")
    primary = _metric_delta_series(tta, source, "spl")
    success = _metric_delta_series(tta, source, "success")
    first_primary = primary[:quartile_size]
    last_primary = primary[-quartile_size:]
    first_success = success[:quartile_size]
    last_success = success[-quartile_size:]

    seeds = {
        "primary_erosion": stable_seed(
            bootstrap_seed, method, setting, run_tag, "primary_erosion"
        ),
        "primary_tail": stable_seed(
            bootstrap_seed, method, setting, run_tag, "primary_tail"
        ),
        "sr_erosion": stable_seed(
            bootstrap_seed, method, setting, run_tag, "sr_erosion"
        ),
        "sr_tail": stable_seed(
            bootstrap_seed, method, setting, run_tag, "sr_tail"
        ),
    }
    tests = {
        "primary_erosion": block_bootstrap_contrast(
            first_primary, last_primary, bootstrap_samples, block_length,
            seeds["primary_erosion"],
        ),
        "primary_tail": block_bootstrap_mean(
            last_primary, bootstrap_samples, block_length,
            seeds["primary_tail"],
        ),
        "sr_erosion": block_bootstrap_contrast(
            first_success, last_success, bootstrap_samples, block_length,
            seeds["sr_erosion"],
        ),
        "sr_tail": block_bootstrap_mean(
            last_success, bootstrap_samples, block_length,
            seeds["sr_tail"],
        ),
    }

    primary_rolling = rolling_means(primary, rolling_window)
    success_rolling = rolling_means(success, rolling_window)
    rolling = [{
        "start_ordinal": index,
        "end_ordinal": index + rolling_window - 1,
        "primary_delta_percentage_points": primary_value,
        "sr_delta_percentage_points": success_rolling[index],
    } for index, primary_value in enumerate(primary_rolling)]

    return {
        "analysis_level": "paired_per_episode_fixed_order",
        "primary_metric": "SPL",
        "metric_unit": "percentage_points",
        "episode_count": episode_count,
        "quartile_size": quartile_size,
        "first_quartile_ordinals": [0, quartile_size - 1],
        "last_quartile_ordinals": [episode_count - quartile_size, episode_count - 1],
        "first_quartile_primary_delta_percentage_points": statistics.mean(
            first_primary
        ),
        "last_quartile_primary_delta_percentage_points": statistics.mean(
            last_primary
        ),
        "first_quartile_sr_delta_percentage_points": statistics.mean(
            first_success
        ),
        "last_quartile_sr_delta_percentage_points": statistics.mean(
            last_success
        ),
        "tests": tests,
        "rolling_window": rolling_window,
        "rolling_deltas": rolling,
        "materiality_threshold_percentage_points": finite_number(
            threshold, "threshold_pp"
        ),
        "classification": "pending_holm_adjustment",
        "limitations": [
            "The bootstrap quantifies one realized canonical order and does not recreate counterfactual adaptation histories.",
            "Matched Source controls episode difficulty, but order-specific history interactions remain possible.",
            "Three separately executed orders are required for order-robustness claims.",
        ],
        "tta_per_episode_evidence": {
            "path": tta.get("evidence_path"),
            "sha256": tta.get("evidence_sha256"),
            "schema": tta.get("schema"),
        },
        "source_per_episode_evidence": {
            "path": source.get("evidence_path"),
            "sha256": source.get("evidence_sha256"),
            "schema": source.get("schema"),
        },
    }


def apply_continuous_holm(records, expected_count=5, alpha=0.05):
    if expected_count is not None and len(records) != int(expected_count):
        raise CollapseError(
            "Holm family has {} finalists; expected {}".format(
                len(records), expected_count
            )
        )
    if not records:
        raise CollapseError("cannot adjust an empty finalist family")
    tests = ("primary_erosion", "primary_tail", "sr_erosion", "sr_tail")
    for test_name in tests:
        adjusted = holm_adjust({
            record["run_tag"]: record["analysis"]["tests"][test_name][
                "one_sided_negative_p_value"
            ]
            for record in records
        })
        for record in records:
            record["analysis"]["tests"][test_name][
                "holm_adjusted_p_value"
            ] = adjusted[record["run_tag"]]
            record["analysis"]["tests"][test_name][
                "holm_family_size"
            ] = len(records)

    for record in records:
        analysis = record["analysis"]
        threshold = analysis["materiality_threshold_percentage_points"]
        erosion = analysis["tests"]["primary_erosion"]
        primary_tail = analysis["tests"]["primary_tail"]
        sr_tail = analysis["tests"]["sr_tail"]

        def evidence(test):
            return (
                test["estimate_percentage_points"] <= -threshold
                and test["one_sided_upper95_percentage_points"] < 0.0
                and test["holm_adjusted_p_value"] < alpha
            )

        erosion_evidence = evidence(erosion)
        primary_tail_evidence = evidence(primary_tail)
        sr_tail_evidence = evidence(sr_tail)
        severe = erosion_evidence and (
            primary_tail_evidence or sr_tail_evidence
        )
        nominal_warning = (
            erosion["estimate_percentage_points"] <= -threshold
            or primary_tail["estimate_percentage_points"] <= -threshold
            or sr_tail["estimate_percentage_points"] <= -threshold
        )
        analysis["holm_alpha"] = alpha
        analysis["severe_rule_components"] = {
            "material_and_significant_primary_erosion": erosion_evidence,
            "material_and_significant_primary_tail_deficit": primary_tail_evidence,
            "material_and_significant_sr_tail_deficit": sr_tail_evidence,
        }
        analysis["severe_fixed_order_collapse"] = bool(severe)
        analysis["warning_fixed_order_degradation"] = bool(
            not severe and nominal_warning
        )
        analysis["classification"] = (
            "severe_fixed_order_collapse" if severe
            else "warning_fixed_order_degradation" if nominal_warning
            else "no_fixed_order_collapse_signal"
        )
    return records


class EvidenceStore:
    """Read either raw scheduler roots or one compact export directory."""

    def __init__(self, root, batch_id=None):
        self.root = Path(root).absolute()
        self.batch_id = batch_id
        if (self.root / "report.json").is_file() and (self.root / "methods").is_dir():
            self.mode = "compact_export"
            self.report = read_json(self.root / "report.json")
            if self.report.get("schema") != "navtta.vln_tta_hparam_compact_export.v1":
                raise CollapseError("unsupported compact export schema")
            if batch_id is not None and self.report.get("batch_id") != batch_id:
                raise CollapseError("compact export batch_id mismatch")
            self.batch_id = self.report.get("batch_id")
        else:
            self.mode = "raw_scheduler"
            self.report = None
            if not batch_id:
                raise CollapseError("--batch-id is required for raw scheduler evidence")

    def method_root(self, method):
        if self.mode == "compact_export":
            return self.root / "methods" / method
        return self.root / method / self.batch_id

    def load_method_jobs(self, method):
        root = self.method_root(method)
        stages_root = root / "stages"
        if not stages_root.is_dir():
            raise CollapseError("missing method stages: {}".format(stages_root))
        for stage in ("controls", "final_controls", "final"):
            summary_path = stages_root / stage / "SUMMARY.json"
            summary = read_json(summary_path)
            if (summary.get("schema") != "navtta.vln_tta_stage_summary.v1"
                    or summary.get("terminal") is not True
                    or summary.get("complete") is not True
                    or summary.get("errors") != []):
                raise CollapseError(
                    "strict stage is not complete and error-free: {}".format(
                        summary_path
                    )
                )
        by_tag = {}
        by_stage = {}
        for job_path in sorted(stages_root.glob("*/jobs/*/job.json")):
            job_dir = job_path.parent
            job = read_json(job_path)
            tag = job.get("run_tag")
            stage = job.get("stage")
            if not isinstance(tag, str) or not tag or not isinstance(stage, str):
                raise CollapseError("invalid job identity: {}".format(job_path))
            if tag in by_tag:
                raise CollapseError("duplicate run_tag in evidence: {}".format(tag))
            metrics_path = job_dir / "metrics.json"
            result = read_json(metrics_path) if metrics_path.is_file() else None
            if result is not None:
                if result.get("run_tag") != tag:
                    raise CollapseError("job/metrics run_tag mismatch for {}".format(tag))
                if canonical(result.get("parameters")) != canonical(job.get("parameters")):
                    raise CollapseError("job/metrics parameters mismatch for {}".format(tag))
            item = {
                "job": job,
                "result": result,
                "job_dir": job_dir,
                "job_path": job_path,
                "metrics_path": metrics_path if result is not None else None,
            }
            by_tag[tag] = item
            by_stage.setdefault(stage, []).append(item)
        if not by_tag:
            raise CollapseError("method has no job evidence: {}".format(root))
        return by_tag, by_stage

    def winner_tags(self, method):
        root = self.method_root(method)
        candidates = []
        direct = root / "FINAL_SELECTION.json"
        if direct.is_file():
            candidates.append(direct)
        candidates.extend(sorted((root / "official").glob("**/FINAL_SELECTION.json")))
        if candidates:
            documents = [read_json(path) for path in candidates]
            canonical_documents = {canonical(document) for document in documents}
            if len(canonical_documents) != 1:
                raise CollapseError("conflicting FINAL_SELECTION documents")
            document = documents[0]
            return {
                setting: value["winner_run_tag"]
                for setting, value in document.get("settings", {}).items()
            }
        if self.report is not None:
            try:
                settings = self.report["methods"][method]["settings"]
                return {
                    setting: value["winner"]["run_tag"]
                    for setting, value in settings.items()
                }
            except (KeyError, TypeError):
                pass
        raise CollapseError("missing winner selection for {}".format(method))

    def load_per_episode(self, item, canonical_order):
        job = item["job"]
        setting = job["setting"]
        run_tag = job["run_tag"]
        job_dir = item["job_dir"]
        recorded_result_root = job.get("result_root")
        result_root = (
            Path(recorded_result_root)
            if isinstance(recorded_result_root, str) and recorded_result_root
            else None
        )
        preferred = [job_dir / "per_episode_metrics.json"]
        if result_root is not None:
            preferred.extend([
                result_root / "per_episode_metrics.json",
                result_root / "metrics" / "per_episode_metrics.json",
            ])
        result = item.get("result") or {}
        recorded = result.get("per_episode_metrics_path")
        if isinstance(recorded, str) and recorded:
            preferred.append(Path(recorded))
        seen = set()
        for path in preferred:
            absolute = path.absolute()
            if absolute in seen:
                continue
            seen.add(absolute)
            if path.is_file():
                if self.mode == "compact_export":
                    try:
                        relative = path.absolute().relative_to(self.root).as_posix()
                    except ValueError:
                        raise CollapseError(
                            "compact per-episode evidence escapes the export root"
                        )
                    metadata = [
                        value for value in self.report.get(
                            "per_episode_metrics", []
                        )
                        if value.get("exported_path") == relative
                    ]
                    if len(metadata) != 1:
                        raise CollapseError(
                            "compact per-episode evidence has no unique inventory record"
                        )
                    metadata = metadata[0]
                    if (int(metadata.get("exported_size", -1)) != path.stat().st_size
                            or metadata.get("exported_sha256") != sha256(path)):
                        raise CollapseError(
                            "compact per-episode evidence hash/size mismatch"
                        )
                return validate_per_episode_document(
                    read_json(path), canonical_order, setting=setting,
                    run_tag=run_tag, source_path=path,
                )
        if (self.mode == "raw_scheduler" and result_root is not None
                and result_root.is_dir()):
            fallbacks = sorted(result_root.glob(
                "metrics/**/stats_ep_ckpt_*_val_seen_r0_w1.json"
            ))
            if len(fallbacks) != 1:
                raise CollapseError(
                    "{} has {} native stats_ep candidates".format(
                        run_tag, len(fallbacks)
                    )
                )
            path = fallbacks[0]
            return validate_raw_stats_episode(
                read_json(path), canonical_order, setting=setting,
                run_tag=run_tag, source_path=path,
            )
        raise CollapseError("missing validated per_episode_metrics for {}".format(
            run_tag
        ))


def _require_result(item, label):
    if item is None or item.get("result") is None:
        raise CollapseError("{} lacks validated metrics".format(label))
    return item["result"]


def _one(items, label):
    if len(items) != 1:
        raise CollapseError("{} has {} records; expected 1".format(label, len(items)))
    return items[0]


def _validate_parent(final_item, parent_item, prefix_count):
    final_job = final_item["job"]
    parent_job = parent_item["job"]
    for key in ("setting", "search_method", "config_method"):
        if final_job.get(key) != parent_job.get(key):
            raise CollapseError("finalist/parent {} mismatch".format(key))
    if canonical(final_job.get("parameters")) != canonical(parent_job.get("parameters")):
        raise CollapseError("finalist and parent parameters differ")
    expected_stage = LAST_SCREENING_STAGE.get(final_job.get("search_method"))
    if expected_stage is None or parent_job.get("stage") != expected_stage:
        raise CollapseError(
            "finalist parent is not from the last screening stage"
        )
    parent_result = _require_result(parent_item, "screening parent")
    if int(parent_result.get("expected_episodes", -1)) != int(prefix_count):
        raise CollapseError("screening parent is not the canonical prefix")


def analyze_campaign(store, spec, order_root, methods=METHODS,
                     bootstrap_samples=2000, block_length=32,
                     rolling_window=64, bootstrap_seed=0,
                     threshold_pp=2.0):
    settings = list(spec.get("settings", []))
    episode_counts = spec.get("setting_episode_counts", {})
    prefix_count = int(spec.get("screening_episodes", -1))
    finalist_count = int(spec.get("protocol", {}).get(
        "full_val_seen_finalists_per_setting", -1
    ))
    if prefix_count < 1 or finalist_count < 1:
        raise CollapseError("search specification lacks protocol counts")
    if set(settings) != set(SETTING_ORDER_DIRECTORY):
        raise CollapseError("search settings differ from canonical collapse mapping")

    records = []
    rolling_rows = []
    order_cache = {}
    for method in methods:
        by_tag, by_stage = store.load_method_jobs(method)
        winners = store.winner_tags(method)
        finals = by_stage.get("final", [])
        controls = by_stage.get("controls", [])
        full_controls = by_stage.get("final_controls", [])
        for setting in settings:
            expected_count = int(episode_counts[setting])
            setting_finals = [
                item for item in finals if item["job"].get("setting") == setting
            ]
            if len(setting_finals) != finalist_count:
                raise CollapseError(
                    "{} {} has {} finalists; expected {}".format(
                        method, setting, len(setting_finals), finalist_count
                    )
                )
            source_prefix = _one([
                item for item in controls
                if item["job"].get("setting") == setting
                and item["job"].get("config_method") == "source"
            ], "{} {} prefix Source".format(method, setting))
            source_full = _one([
                item for item in full_controls
                if item["job"].get("setting") == setting
                and item["job"].get("config_method") == "source"
            ], "{} {} full Source".format(method, setting))
            source_prefix_result = _require_result(source_prefix, "prefix Source")
            source_full_result = _require_result(source_full, "full Source")
            if int(source_prefix_result.get("expected_episodes", -1)) != prefix_count:
                raise CollapseError("Source control is not the canonical prefix")
            if int(source_full_result.get("expected_episodes", -1)) != expected_count:
                raise CollapseError("Source final control is not the full stream")
            if canonical(source_prefix["job"].get("parameters")) != canonical(
                    source_full["job"].get("parameters")):
                raise CollapseError("prefix/full Source parameters differ")
            winner_tag = winners.get(setting)
            if winner_tag not in {item["job"]["run_tag"] for item in setting_finals}:
                raise CollapseError("winner is absent from finalists for {} {}".format(
                    method, setting
                ))

            continuous_group = []
            for final_item in setting_finals:
                final_job = final_item["job"]
                final_result = _require_result(final_item, "finalist")
                if int(final_result.get("expected_episodes", -1)) != expected_count:
                    raise CollapseError("finalist is not the full canonical stream")
                parent_tags = final_job.get("parent_run_tags")
                if not isinstance(parent_tags, list) or len(parent_tags) != 1:
                    raise CollapseError("finalist must identify one screening parent")
                parent = by_tag.get(parent_tags[0])
                if parent is None:
                    raise CollapseError("screening parent is absent: {}".format(
                        parent_tags[0]
                    ))
                _validate_parent(final_item, parent, prefix_count)
                base = {
                    "method": method,
                    "setting": setting,
                    "family": (
                        "continuous" if setting in CONTINUOUS_SETTINGS else "discrete"
                    ),
                    "run_tag": final_job["run_tag"],
                    "screening_parent_run_tag": parent_tags[0],
                    "matched_prefix_source_run_tag": source_prefix["job"]["run_tag"],
                    "matched_full_source_run_tag": source_full["job"]["run_tag"],
                    "is_scheduler_winner": final_job["run_tag"] == winner_tag,
                    "parameters": final_job.get("parameters"),
                }
                if setting in CONTINUOUS_SETTINGS:
                    if setting not in order_cache:
                        order_cache[setting] = load_canonical_order(
                            order_root, setting, expected_count
                        )
                    order = order_cache[setting]
                    tta_episodes = store.load_per_episode(final_item, order)
                    source_episodes = store.load_per_episode(source_full, order)
                    analysis = analyze_continuous_per_episode(
                        tta_episodes, source_episodes, method, setting,
                        final_job["run_tag"], bootstrap_samples, block_length,
                        rolling_window, bootstrap_seed, threshold_pp,
                    )
                    base["analysis"] = analysis
                    continuous_group.append(base)
                else:
                    parent_result = _require_result(parent, "screening parent")
                    base["analysis"] = analyze_discrete_aggregates(
                        parent_result["metrics"], final_result["metrics"],
                        source_prefix_result["metrics"],
                        source_full_result["metrics"], setting,
                        expected_count, prefix_count, threshold_pp,
                    )
                    records.append(base)
            if continuous_group:
                apply_continuous_holm(
                    continuous_group, expected_count=finalist_count
                )
                records.extend(continuous_group)

    records.sort(key=lambda row: (
        METHODS.index(row["method"]), settings.index(row["setting"]),
        row["run_tag"],
    ))
    for record in records:
        if record["family"] != "continuous":
            continue
        for window in record["analysis"]["rolling_deltas"]:
            rolling_rows.append({
                "method": record["method"],
                "setting": record["setting"],
                "run_tag": record["run_tag"],
                "is_scheduler_winner": record["is_scheduler_winner"],
                **window,
            })
    return {
        "schema": OUTPUT_SCHEMA,
        "batch_id": store.batch_id,
        "evidence_mode": store.mode,
        "policy": {
            "split": "val_seen",
            "canonical_order_only": True,
            "screening_prefix_episodes": prefix_count,
            "full_stream_finalists_per_setting": finalist_count,
            "materiality_threshold_percentage_points": threshold_pp,
            "continuous_first_last_fraction": "quartiles",
            "continuous_rolling_window": rolling_window,
            "bootstrap": "independent_segment_circular_moving_blocks",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_block_length": block_length,
            "bootstrap_seed": bootstrap_seed,
            "holm_family": "five finalists within each method x setting",
            "holm_alpha": 0.05,
        },
        "global_limitations": [
            "All classifications describe one fixed canonical order.",
            "Discrete evidence identifies only a prefix and complementary remainder.",
            "Continuous block-bootstrap intervals do not replay counterfactual stateful adaptation histories.",
            "Order robustness requires the separately executed three-order experiment.",
        ],
        "records": records,
        "rolling_rows": rolling_rows,
    }


def summary_row(record):
    analysis = record["analysis"]
    row = {
        "method": record["method"],
        "setting": record["setting"],
        "family": record["family"],
        "run_tag": record["run_tag"],
        "is_scheduler_winner": record["is_scheduler_winner"],
        "screening_parent_run_tag": record.get("screening_parent_run_tag"),
        "matched_prefix_source_run_tag": record.get(
            "matched_prefix_source_run_tag"
        ),
        "matched_full_source_run_tag": record.get(
            "matched_full_source_run_tag"
        ),
        "classification": analysis["classification"],
        "primary_metric": analysis["primary_metric"],
        "threshold_percentage_points": analysis[
            "materiality_threshold_percentage_points"
        ],
    }
    if record["family"] == "discrete":
        primary = analysis["metrics"][analysis["primary_metric"]]
        sr = analysis["metrics"]["SR"]
        row.update({
            "prefix_count": primary["prefix_count"],
            "suffix_count": primary["suffix_count"],
            "primary_prefix_delta_pp": primary["matched_prefix_delta"],
            "primary_suffix_delta_pp": primary["matched_suffix_delta"],
            "primary_erosion_pp": primary["matched_delta_erosion"],
            "sr_prefix_delta_pp": sr["matched_prefix_delta"],
            "sr_suffix_delta_pp": sr["matched_suffix_delta"],
            "sr_erosion_pp": sr["matched_delta_erosion"],
            "primary_erosion_rounding_bound_pp": primary[
                "rounding_bounds"
            ]["matched_delta_erosion"],
            "coarse_2pp_flag": analysis["coarse_2pp_flag"],
            "severe_fixed_order_collapse": "",
            "warning_fixed_order_degradation": "",
        })
    else:
        erosion = analysis["tests"]["primary_erosion"]
        primary_tail = analysis["tests"]["primary_tail"]
        sr_tail = analysis["tests"]["sr_tail"]
        row.update({
            "prefix_count": "",
            "suffix_count": "",
            "primary_prefix_delta_pp": analysis[
                "first_quartile_primary_delta_percentage_points"
            ],
            "primary_suffix_delta_pp": analysis[
                "last_quartile_primary_delta_percentage_points"
            ],
            "primary_erosion_pp": erosion["estimate_percentage_points"],
            "sr_prefix_delta_pp": analysis[
                "first_quartile_sr_delta_percentage_points"
            ],
            "sr_suffix_delta_pp": analysis[
                "last_quartile_sr_delta_percentage_points"
            ],
            "sr_erosion_pp": analysis["tests"]["sr_erosion"][
                "estimate_percentage_points"
            ],
            "primary_erosion_ci95_lower_pp": erosion[
                "ci95_lower_percentage_points"
            ],
            "primary_erosion_ci95_upper_pp": erosion[
                "ci95_upper_percentage_points"
            ],
            "primary_erosion_p_holm": erosion["holm_adjusted_p_value"],
            "primary_tail_p_holm": primary_tail["holm_adjusted_p_value"],
            "sr_tail_p_holm": sr_tail["holm_adjusted_p_value"],
            "coarse_2pp_flag": "",
            "severe_fixed_order_collapse": analysis[
                "severe_fixed_order_collapse"
            ],
            "warning_fixed_order_degradation": analysis[
                "warning_fixed_order_degradation"
            ],
        })
    return row


SUMMARY_FIELDS = (
    "method", "setting", "family", "run_tag", "is_scheduler_winner",
    "screening_parent_run_tag", "matched_prefix_source_run_tag",
    "matched_full_source_run_tag",
    "classification", "primary_metric", "threshold_percentage_points",
    "prefix_count", "suffix_count", "primary_prefix_delta_pp",
    "primary_suffix_delta_pp", "primary_erosion_pp", "sr_prefix_delta_pp",
    "sr_suffix_delta_pp", "sr_erosion_pp",
    "primary_erosion_rounding_bound_pp", "primary_erosion_ci95_lower_pp",
    "primary_erosion_ci95_upper_pp", "primary_erosion_p_holm",
    "primary_tail_p_holm", "sr_tail_p_holm", "coarse_2pp_flag",
    "severe_fixed_order_collapse", "warning_fixed_order_degradation",
)
ROLLING_FIELDS = (
    "method", "setting", "run_tag", "is_scheduler_winner",
    "start_ordinal", "end_ordinal", "primary_delta_percentage_points",
    "sr_delta_percentage_points",
)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def _atomic_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(str(temporary), str(path))


def write_outputs(output_dir, report, overwrite=False):
    output_dir = Path(output_dir).absolute()
    rolling_window = int(report.get("policy", {}).get(
        "continuous_rolling_window", 64
    ))
    targets = {
        "json": output_dir / "late_collapse_analysis.json",
        "summary": output_dir / "late_collapse_summary.csv",
        "rolling": output_dir / "late_collapse_rolling{}.csv".format(
            rolling_window
        ),
    }
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise CollapseError(
            "output files exist; pass --overwrite: {}".format(", ".join(existing))
        )
    # Per-finalist rolling values already live under each JSON record.  The
    # flattened copy is an in-memory convenience for CSV emission and would
    # otherwise double the portable bundle size.
    json_report = dict(report)
    json_report.pop("rolling_rows", None)
    _atomic_json(targets["json"], json_report)
    _atomic_csv(
        targets["summary"], [summary_row(record) for record in report["records"]],
        SUMMARY_FIELDS,
    )
    _atomic_csv(targets["rolling"], report["rolling_rows"], ROLLING_FIELDS)
    return targets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root", required=True,
        help="raw hparam_search root or one compact export directory",
    )
    parser.add_argument(
        "--batch-id", default=None,
        help="required for raw scheduler evidence; verified for compact exports",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--order-root", default=str(DEFAULT_ORDER_ROOT))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--rolling-window", type=int, default=64)
    parser.add_argument("--threshold-pp", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        spec = read_json(args.spec)
        if spec.get("schema") != "navtta.vln_tta_hparam_search.v1":
            raise CollapseError("unsupported search specification schema")
        store = EvidenceStore(args.evidence_root, args.batch_id)
        report = analyze_campaign(
            store, spec, args.order_root, methods=tuple(args.methods),
            bootstrap_samples=args.bootstrap_samples,
            block_length=args.block_length,
            rolling_window=args.rolling_window,
            bootstrap_seed=args.bootstrap_seed,
            threshold_pp=args.threshold_pp,
        )
        report["evidence_root"] = str(Path(args.evidence_root).absolute())
        report["search_spec"] = {
            "path": str(Path(args.spec).absolute()),
            "sha256": sha256(args.spec),
        }
        targets = write_outputs(args.output_dir, report, overwrite=args.overwrite)
    except CollapseError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    for path in targets.values():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
