#!/usr/bin/env python3
"""Resumable, staged VLN TTA hyperparameter-search scheduler.

The scheduler deliberately keeps model execution in ``run_source_eval.sh``.
It owns only immutable search plans, promotion, bounded concurrency, durable
worker state, and compact metric summaries.  A worker process writes its own
exit code, so killing/restarting the scheduler cannot silently duplicate a
still-running GPU job.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "vln/experiments/tta_hparam_search_v1.json"
LOG_ROOT = REPO_ROOT / "vln/results/logs/hparam_search"
TUNING_ROOT = REPO_ROOT / "vln/results/tuning"
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
RESULT_LAYOUT = "method_batch_stage_setting_run_v2"
METHODS = ("tent", "fstta", "eam", "feedtta", "atena")
DISCRETE_SETTINGS = {
    "duet-r2r", "duet-reverie", "hamt-r2r", "hamt-reverie",
    "goat-r2r", "goat-reverie",
}
CONTINUOUS_SETTINGS = {"etpnav-r2r-ce", "bevbert-r2r-ce"}
SETTING_MODEL = {
    "duet-r2r": "duet", "duet-reverie": "duet",
    "hamt-r2r": "hamt", "hamt-reverie": "hamt",
    "goat-r2r": "goat", "goat-reverie": "goat",
    "etpnav-r2r-ce": "etpnav", "bevbert-r2r-ce": "bevbert",
}
INTERMEDIATE_STAGES = {
    "tent": (),
    "fstta": ("stage2", "stage3"),
    "eam": ("stage2", "stage3"),
    "feedtta": ("stage2",),
    "atena": ("stage2", "stage3"),
}
STRICT_FULL_STAGES = {"final_controls", "final", "orders"}


class UserError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(path))


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT)] + list(args), text=True
    ).strip()


def slug(value):
    text = str(value).lower().replace("-", "m").replace(".", "p")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def load_spec(path=SPEC_PATH):
    with Path(path).open("r", encoding="utf-8") as stream:
        spec = json.load(stream)
    if spec.get("schema") != "navtta.vln_tta_hparam_search.v1":
        raise UserError("unsupported search specification schema")
    final_order_seeds = spec.get("final_order_seeds")
    if (final_order_seeds != [0, 1, 2]
            or any(type(value) is not int for value in final_order_seeds)):
        raise UserError("the search specification must use order seeds [0, 1, 2]")
    if type(spec.get("primary_order_seed")) is not int or spec.get(
            "primary_order_seed") != 0:
        raise UserError("the search specification primary order seed must be integer 0")
    protocol = spec.get("protocol", {})
    if not protocol.get("full_val_seen_matched_source_controls"):
        raise UserError("the search requires full-val matched Source controls")
    if not protocol.get("freeze_winner_before_order_robustness"):
        raise UserError("the canonical full-val winner must be frozen before orders")
    return spec


def _product(mapping):
    keys = list(mapping)
    for values in itertools.product(*(mapping[key] for key in keys)):
        yield dict(zip(keys, values))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _deduplicate_candidates(candidates):
    unique = []
    seen = set()
    for candidate in candidates:
        key = (_canonical(candidate["parameters"]), candidate.get("order_seed"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def anchor_for(method, setting, spec):
    method_spec = spec["methods"][method]
    if method == "tent":
        result = dict(method_spec["paper_anchor"])
        result.update(method_spec["stage1"]["fixed"])
        return result
    if method == "fstta":
        family = "continuous" if setting in CONTINUOUS_SETTINGS else "discrete"
        result = dict(method_spec["anchors"][family])
        result.update(method_spec["fixed"])
        return result
    if method == "eam":
        result = {"lr": 1e-6, "update_interval": 1}
        result.update(method_spec["fixed"])
        return result
    if method == "feedtta":
        result = dict(method_spec["paper_anchor"])
        result.update(method_spec["fixed"])
        return result
    if method == "atena":
        family = (
            "continuous_r2r_ce" if setting in CONTINUOUS_SETTINGS
            else "discrete_reverie" if setting.endswith("reverie")
            else "discrete_r2r"
        )
        result = dict(method_spec["anchors"][family])
        result.update(method_spec["fixed"])
        return result
    raise UserError("unknown method {}".format(method))


def stage1_points(method, setting, spec):
    method_spec = spec["methods"][method]
    if method == "tent":
        for point in _product(method_spec["stage1"]["grid"]):
            point.update(method_spec["stage1"]["fixed"])
            yield point
        return
    if method == "fstta":
        family = "continuous" if setting in CONTINUOUS_SETTINGS else "discrete"
        anchor = method_spec["anchors"][family]
        m_key = "m_continuous" if family == "continuous" else "m_discrete"
        for multiplier, m_value in itertools.product(
            method_spec["stage1_fast"]["grid"]["lr_fast_multiplier"],
            method_spec["stage1_fast"]["grid"][m_key],
        ):
            point = {
                "lr_fast": anchor["lr_fast"] * multiplier,
                "lr_slow": anchor["lr_slow"],
                "m": m_value,
                "n": anchor["n"],
            }
            point.update(method_spec["fixed"])
            yield point
        return
    if method == "eam":
        for point in _product(method_spec["stage1_intensity"]["grid"]):
            point.update(method_spec["fixed"])
            yield point
        return
    if method == "feedtta":
        for point in _product(method_spec["stage1_intensity"]["grid"]):
            point.update(method_spec["stage1_intensity"]["fixed"])
            point.update(method_spec["fixed"])
            yield point
        return
    if method == "atena":
        anchor = anchor_for(method, setting, spec)
        for multiplier in method_spec["stage1_lr_pair"]["grid"][
            "joint_lr_multiplier"
        ]:
            point = dict(anchor)
            point["lr_query"] = anchor["lr_query"] * multiplier
            point["lr_self"] = anchor["lr_self"] * multiplier
            yield point
        return
    raise UserError("unknown method {}".format(method))


def clean_parameters(point):
    ignored = {
        "matched_source_action_selection",
        "selection_requires_feedback_budget_reporting",
    }
    return {key: value for key, value in point.items() if key not in ignored}


def _validate_parameter_seeds(parameters, expected_order_seed=None):
    for key in ("action_seed", "sgr_seed"):
        if key in parameters and type(parameters[key]) is not int:
            raise UserError("{} must be an exact integer".format(key))
    if expected_order_seed is not None:
        for key in ("action_seed", "sgr_seed"):
            if parameters.get(key) != expected_order_seed:
                raise UserError(
                    "FeedTTA orders {} must equal order_seed".format(key)
                )


def _candidate(parameters, parents=(), order_seed=None, config_method=None):
    parameters = clean_parameters(parameters)
    _validate_parameter_seeds(parameters)
    value = {
        "parameters": parameters,
        "parent_run_tags": list(parents),
    }
    if order_seed is not None:
        if type(order_seed) is not int:
            raise UserError("order_seed must be an exact integer")
        value["order_seed"] = order_seed
    if config_method is not None:
        value["config_method"] = config_method
    return value


def _ensure_anchor(candidates, method, setting, spec):
    anchor = clean_parameters(anchor_for(method, setting, spec))
    if not any(_canonical(item["parameters"]) == _canonical(anchor)
               for item in candidates):
        candidates.append(_candidate(anchor))
    return _deduplicate_candidates(candidates)


def expand_stage(method, stage, setting, promoted, spec):
    """Expand promoted result dictionaries into the next-stage candidates."""
    method_spec = spec["methods"][method]
    candidates = []
    for result in promoted:
        parent = (result["run_tag"],)
        base = dict(result["parameters"])
        if stage == "stage2" and method == "fstta":
            family = "continuous" if setting in CONTINUOUS_SETTINGS else "discrete"
            base_lr = method_spec["anchors"][family]["lr_slow"]
            for multiplier in method_spec["stage2_slow_lr"]["grid"][
                "lr_slow_multiplier"
            ]:
                point = dict(base)
                point["lr_slow"] = base_lr * multiplier
                point["n"] = method_spec["stage2_slow_lr"]["fixed"]["n"]
                candidates.append(_candidate(point, parent))
        elif stage == "stage3" and method == "fstta":
            for value in method_spec["stage3_slow_window"]["grid"]["n"]:
                point = dict(base)
                point["n"] = value
                candidates.append(_candidate(point, parent))
        elif stage == "stage2" and method == "eam":
            for value in method_spec["stage2_confidence"]["grid"][
                "confidence_scale"
            ]:
                point = dict(base)
                point["confidence_scale"] = value
                candidates.append(_candidate(point, parent))
        elif stage == "stage3" and method == "eam":
            for memory_size, batch_size in method_spec["stage3_replay"]["grid"][
                "memory_batch"
            ]:
                point = dict(base)
                point["memory_size"] = memory_size
                point["batch_size"] = batch_size
                candidates.append(_candidate(point, parent))
        elif stage == "stage2" and method == "feedtta":
            for values in _product(method_spec["stage2_sgr"]["grid"]):
                point = dict(base)
                point.update(values)
                candidates.append(_candidate(point, parent))
            for control in method_spec["stage2_sgr"].get("controls", []):
                point = dict(base)
                point.update({key: value for key, value in control.items()
                              if key != "name"})
                candidates.append(_candidate(point, parent))
        elif stage == "stage2" and method == "atena":
            for value in method_spec["stage2_mixture"]["grid"]["mix_lambda"]:
                point = dict(base)
                point["mix_lambda"] = value
                candidates.append(_candidate(point, parent))
        elif stage == "stage3" and method == "atena":
            for value in method_spec["stage3_query"]["grid"]["query_threshold"]:
                point = dict(base)
                point["query_threshold"] = value
                candidates.append(_candidate(point, parent))
        else:
            raise UserError("{} has no expansion for {}".format(method, stage))
    return _ensure_anchor(candidates, method, setting, spec)


def workflow_stages(method, include_orders=False):
    stages = ("smoke", "controls", "stage1") + INTERMEDIATE_STAGES[method] + (
        "final_controls", "final",
    )
    return stages + (("orders",) if include_orders else ())


def previous_search_stage(method, stage):
    if stage == "stage2":
        return "stage1"
    if stage == "stage3":
        return "stage2"
    if stage == "final":
        return INTERMEDIATE_STAGES[method][-1] if INTERMEDIATE_STAGES[method] else "stage1"
    if stage == "orders":
        return "final"
    return None


def promotion_limit(method, next_stage, spec):
    method_spec = spec["methods"][method]
    if next_stage == "stage2":
        key = {
            "fstta": "stage2_slow_lr",
            "eam": "stage2_confidence",
            "feedtta": "stage2_sgr",
            "atena": "stage2_mixture",
        }[method]
        return int(method_spec[key].get(
            "promote", method_spec[key].get("promote_fast", 1)
        ))
    if next_stage == "stage3":
        key = {
            "fstta": "stage3_slow_window",
            "eam": "stage3_replay",
            "atena": "stage3_query",
        }[method]
        return int(method_spec[key].get("promote", method_spec[key].get(
            "promote_complete", 1
        )))
    if next_stage == "final":
        return int(spec["protocol"]["full_val_seen_finalists_per_setting"])
    if next_stage == "orders":
        return 1
    raise UserError("no promotion limit for {}".format(next_stage))


def stage_episode_count(stage, spec):
    if stage == "smoke":
        return int(spec["smoke_episodes"])
    if stage in ("controls", "stage1", "stage2", "stage3"):
        return int(spec["screening_episodes"])
    if stage in ("final_controls", "final", "orders"):
        return -1
    raise UserError("unknown stage {}".format(stage))


def resolve_stage_episodes(stage, override, spec):
    expected = stage_episode_count(stage, spec)
    if override is None:
        return expected
    value = int(override)
    if value != expected:
        raise UserError(
            "{} requires protocol episode count {}; got {}".format(
                stage, expected, value
            )
        )
    return value


def _adapter_diagnostics(document):
    if not isinstance(document, dict):
        return None
    adapter = document.get("adapter")
    return adapter if isinstance(adapter, dict) else None


def _all_finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def parse_metrics(job, spec=None):
    spec = spec or load_spec()
    console = Path(job["job_dir"]) / "console.log"
    if not console.is_file():
        raise UserError("missing console log: {}".format(console))
    diagnostics = Path(job["result_root"]) / "tta_diagnostics.json"
    values = {}
    text = console.read_text(encoding="utf-8", errors="replace")
    discrete_lines = [
        line for line in text.splitlines() if "Env name: val_seen" in line
    ]
    if discrete_lines:
        for key, value in re.findall(
            r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)",
            discrete_lines[-1],
        ):
            values[key.upper()] = float(value)
    else:
        for key, value in re.findall(
            r"Average episode ([A-Za-z0-9_]+):\s*(-?[0-9]+(?:\.[0-9]+)?)",
            text,
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
    if not values:
        raise UserError("no validation metrics found in {}".format(console))
    if not _all_finite(values):
        raise UserError("non-finite validation metric in {}".format(console))

    diag = None
    if diagnostics.is_file():
        with diagnostics.open("r", encoding="utf-8") as stream:
            diag = json.load(stream)
        if not _all_finite(diag):
            raise UserError("non-finite TTA diagnostics: {}".format(diagnostics))
    elif job.get("config_method") != "source":
        raise UserError("missing TTA diagnostics: {}".format(diagnostics))

    adapter = _adapter_diagnostics(diag)
    if job.get("config_method") != "source":
        if adapter is None:
            raise UserError("TTA diagnostics have no adapter payload")
        for required in ("relative_param_drift", "updates"):
            if required not in adapter:
                raise UserError(
                    "TTA diagnostics are missing adapter.{}".format(required)
                )

    expected_episodes = (
        int(job["episodes"]) if int(job["episodes"]) > 0
        else int(spec["setting_episode_counts"][job["setting"]])
    )
    if diag is not None:
        observed = diag.get("episode_count")
        if observed is None and adapter is not None:
            observed = adapter.get("episodes")
        if observed is not None and int(observed) != expected_episodes:
            raise UserError(
                "incomplete episode stream: expected {}, got {}".format(
                    expected_episodes, observed
                )
            )

    result = dict(job)
    result["metrics"] = values
    result["expected_episodes"] = expected_episodes
    result["diagnostics_path"] = str(diagnostics) if diag is not None else None
    result["diagnostics_sha256"] = sha256(diagnostics) if diag is not None else None
    result["adapter_diagnostics"] = adapter
    result["requires_posthoc_late_collapse_check"] = True
    return result


def _selection_spec(setting, spec):
    return spec["selection"][
        "reverie" if setting.endswith("reverie") else "r2r_and_r2r_ce"
    ]


def rank_and_promote(results, source_result, method, setting, limit, spec,
                     force_anchor=True):
    selection = _selection_spec(setting, spec)
    primary = selection["primary"].upper()
    floor_tolerance = float(
        selection["source_sr_floor_tolerance_percentage_points"]
    )
    source_sr = None
    if source_result is not None:
        source_sr = source_result.get("metrics", {}).get("SR")
    ranked = []
    for result in results:
        metrics = result.get("metrics", {})
        reasons = []
        if primary not in metrics:
            reasons.append("missing_primary_metric")
        if "SR" not in metrics:
            reasons.append("missing_sr")
        if source_sr is not None and metrics.get("SR", -math.inf) < (
            source_sr - floor_tolerance
        ):
            reasons.append("below_matched_source_sr_floor")
        adapter = result.get("adapter_diagnostics") or {}
        drift = float(adapter.get("relative_param_drift", math.inf))
        updates = float(adapter.get("updates", math.inf))
        tie_values = []
        for tie in selection["tie_breakers"]:
            if tie == "lower_parameter_drift":
                tie_values.append(-drift)
            elif tie == "fewer_updates":
                tie_values.append(-updates)
            else:
                if tie.upper() not in metrics:
                    reasons.append("missing_tie_metric_{}".format(tie.upper()))
                tie_values.append(float(metrics.get(tie.upper(), -1e300)))
        score = (float(metrics.get(primary, -1e300)),) + tuple(tie_values)
        ranked.append({
            "result": result,
            "eligible": not reasons,
            "reasons": reasons,
            "score": score,
        })
    eligible = [item for item in ranked if item["eligible"]]
    eligible.sort(key=lambda item: item["score"], reverse=True)
    if len(eligible) < limit:
        raise UserError(
            "{} has only {} eligible {} candidates; {} are required".format(
                setting, len(eligible), method, limit
            )
        )
    selected = [item["result"] for item in eligible[:limit]]

    anchor = clean_parameters(anchor_for(method, setting, spec))
    anchor_result = next((
        result for result in results
        if _canonical(result["parameters"]) == _canonical(anchor)
    ), None)
    if force_anchor:
        if anchor_result is None:
            raise UserError("paper/default anchor is missing for {}".format(setting))
        if not any(item["run_tag"] == anchor_result["run_tag"] for item in selected):
            if len(selected) >= limit:
                selected[-1] = anchor_result
            else:
                selected.append(anchor_result)

    record = {
        "setting": setting,
        "method": method,
        "primary_metric": primary,
        "matched_source_sr": source_sr,
        "source_sr_floor_tolerance_percentage_points": floor_tolerance,
        "selected_run_tags": [item["run_tag"] for item in selected],
        "ranked": [{
            "run_tag": item["result"]["run_tag"],
            "eligible": item["eligible"],
            "reasons": item["reasons"],
            "score": list(item["score"]),
        } for item in sorted(ranked, key=lambda item: item["score"], reverse=True)],
    }
    return selected, record


def point_tag(point):
    return hashlib.sha256(_canonical(point).encode("utf-8")).hexdigest()[:10]


def tuning_result_root(method, batch_id, stage, setting, run_tag,
                       result_namespace=None):
    """Return the canonical, human-browsable raw tuning-result directory."""
    namespace = result_namespace or method
    components = {
        "method": method,
        "result_namespace": namespace,
        "batch_id": batch_id,
        "stage": stage,
        "setting": setting,
        "run_tag": run_tag,
    }
    for label, value in components.items():
        if (not isinstance(value, str) or not value
                or Path(value).name != value
                or value in (".", "..")):
            raise UserError("unsafe {} path component: {!r}".format(label, value))
    if method not in METHODS:
        raise UserError("unknown tuning-result method {}".format(method))
    if namespace not in METHODS + ("_shared",):
        raise UserError(
            "unknown tuning-result namespace {}".format(namespace)
        )
    if setting not in SETTING_MODEL:
        raise UserError("unknown tuning-result setting {}".format(setting))
    return TUNING_ROOT / namespace / batch_id / stage / setting / run_tag / "val_seen"


def _source_candidates(method, settings, spec):
    candidates = {}
    for setting in settings:
        parameters = {
            "action_selection": "sample" if method == "feedtta" else "argmax",
            "action_seed": int(spec["primary_order_seed"]),
        }
        candidates[setting] = [
            _candidate(parameters, config_method="source")
        ]
    return candidates


def _validate_stage_manifest(stage_dir, batch_id, method, settings, spec,
                             expected_stage):
    manifest_path = Path(stage_dir) / "stage_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(
            "invalid prerequisite stage manifest {}: {}".format(
                manifest_path, error
            )
        )
    expected_settings = list(settings)
    checks = (
        ("schema", "navtta.vln_tta_search_stage.v1"),
        ("batch_id", batch_id),
        ("git_commit", git("rev-parse", "HEAD")),
        ("spec_sha256", sha256(SPEC_PATH)),
        ("method", method),
        ("stage", expected_stage),
        ("episodes", stage_episode_count(expected_stage, spec)),
        ("settings", expected_settings),
    )
    for key, expected in checks:
        if manifest.get(key) != expected:
            raise UserError(
                "stage manifest {} mismatch for {}: expected {!r}, got {!r}".format(
                    manifest_path, key, expected, manifest.get(key)
                )
                )
    recorded_layout = manifest.get("result_layout")
    if recorded_layout not in (None, RESULT_LAYOUT):
        raise UserError(
            "stage manifest {} has an unsupported result layout".format(
                manifest_path
            )
        )
    return manifest


def _validate_jobs(jobs, batch_id, method, settings, stage, spec,
                   expected_job_count=None):
    expected_episodes = stage_episode_count(stage, spec)
    expected_settings = set(settings)
    if expected_job_count is not None and len(jobs) != int(expected_job_count):
        raise UserError(
            "stage job count mismatch: manifest={}, plan={}".format(
                expected_job_count, len(jobs)
            )
        )
    ordinals = [int(job.get("ordinal", -1)) for job in jobs]
    if ordinals != list(range(len(jobs))):
        raise UserError("stage job ordinals are not contiguous")
    actual_settings = {job.get("setting") for job in jobs}
    if actual_settings != expected_settings:
        raise UserError(
            "stage job setting set mismatch: expected {}, got {}".format(
                sorted(expected_settings), sorted(actual_settings)
            )
        )
    order_seeds_by_setting = {setting: [] for setting in settings}
    for job in jobs:
        checks = (
            ("batch_id", batch_id),
            ("search_method", method),
            ("stage", stage),
            ("episodes", expected_episodes),
        )
        for key, expected in checks:
            if job.get(key) != expected:
                raise UserError(
                    "job {} mismatch for {}: expected {!r}, got {!r}".format(
                        job.get("run_tag", job.get("ordinal")), key,
                        expected, job.get(key),
                    )
                )
        if job.get("setting") not in expected_settings:
            raise UserError("job plan contains an unrequested setting")
        command = list(job.get("command", []))
        layout = job.get("result_layout")
        if layout not in (None, RESULT_LAYOUT):
            raise UserError("job plan has an unsupported result layout")
        result_values = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--result-root"
        ]
        if layout == RESULT_LAYOUT:
            namespace = job.get("result_namespace") or method
            if namespace == "_shared" and job.get("config_method") != "source":
                raise UserError("only Source jobs may use _shared results")
            expected_root = tuning_result_root(
                method, batch_id, stage, job["setting"], job["run_tag"],
                result_namespace=namespace,
            )
            if Path(job.get("result_root", "")) != expected_root:
                raise UserError(
                    "job {} result_root disagrees with the canonical layout".format(
                        job.get("run_tag", job.get("ordinal"))
                    )
                )
            if result_values != [str(expected_root)]:
                raise UserError(
                    "job {} command does not pin its result_root".format(
                        job.get("run_tag", job.get("ordinal"))
                    )
                )
        elif result_values and result_values != [job.get("result_root")]:
            raise UserError("legacy job result-root command disagrees with metadata")
        limit_values = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--episode-limit"
        ]
        expected_limits = [] if expected_episodes < 0 else [str(expected_episodes)]
        if limit_values != expected_limits:
            raise UserError(
                "job {} episode-limit command disagrees with protocol".format(
                    job.get("run_tag", job.get("ordinal"))
                )
            )
        order_values = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--order-seed"
        ]
        order_seed = job.get("order_seed")
        parameters = job.get("parameters", {})
        if not isinstance(parameters, dict):
            raise UserError("job parameters must be an object")
        _validate_parameter_seeds(parameters)
        if stage == "orders":
            if (type(order_seed) is not int
                    or order_seed not in spec["final_order_seeds"]):
                raise UserError("orders job has an invalid order_seed")
            if method == "feedtta":
                _validate_parameter_seeds(
                    parameters, expected_order_seed=order_seed
                )
            if order_values != [str(order_seed)]:
                raise UserError("orders job command does not pin its order_seed")
            order_seeds_by_setting[job["setting"]].append(order_seed)
        elif order_seed is not None or order_values:
            raise UserError(
                "only the orders stage may carry an order_seed"
            )
    if stage == "orders":
        for setting, seeds in order_seeds_by_setting.items():
            if seeds != spec["final_order_seeds"]:
                raise UserError(
                    "{} orders jobs must be exactly seeds {}".format(
                        setting, spec["final_order_seeds"]
                    )
                )


def _validate_persisted_stage(stage_dir, batch_id, method, settings, spec,
                              expected_stage):
    manifest = _validate_stage_manifest(
        stage_dir, batch_id, method, settings, spec, expected_stage
    )
    jobs = load_jobs(stage_dir)
    job_count = manifest.get("job_count")
    if not isinstance(job_count, int) or job_count < 1:
        raise UserError("stage manifest has invalid job_count")
    _validate_jobs(
        jobs, batch_id, method, settings, expected_stage, spec,
        expected_job_count=job_count,
    )
    return manifest, jobs


def _last_screening_stage(method):
    return INTERMEDIATE_STAGES[method][-1] if INTERMEDIATE_STAGES[method] else "stage1"


def _validate_stage_prerequisites(batch_root, batch_id, method, stage, settings,
                                  spec):
    batch_root = Path(batch_root)
    required = []
    if stage == "stage1":
        required = ["controls"]
    elif stage == "stage2":
        required = ["controls", "stage1"]
    elif stage == "stage3":
        required = ["controls", "stage2"]
    elif stage == "final_controls":
        required = ["controls", _last_screening_stage(method)]
    elif stage == "final":
        required = [
            "controls", _last_screening_stage(method), "final_controls",
        ]
    elif stage == "orders":
        required = ["final_controls", "final"]
    for prerequisite in required:
        _validate_persisted_stage(
            batch_root / "stages" / prerequisite,
            batch_id,
            method,
            settings,
            spec,
            prerequisite,
        )
    if stage == "final_controls":
        load_stage_results(
            batch_root / "stages" / _last_screening_stage(method),
            spec,
            allow_partial=True,
        )
        load_stage_results(batch_root / "stages" / "controls", spec)
    elif stage == "final":
        # This stage is a strict execution barrier.  Candidate promotion below
        # still reads the last screening stage, never final_controls.
        load_stage_results(batch_root / "stages" / "final_controls", spec)
    elif stage == "orders":
        _load_frozen_selection(batch_root, method, settings, spec)


def _validate_exact_setting_keys(document, settings, label):
    requested = list(settings)
    if len(set(requested)) != len(requested):
        raise UserError("duplicate requested settings are not allowed")
    values = document.get("settings")
    if not isinstance(values, dict):
        raise UserError("{} has no settings mapping".format(label))
    actual = set(values)
    expected = set(requested)
    if len(values) != len(requested) or actual != expected:
        raise UserError(
            "{} setting set mismatch: expected {}, got {}".format(
                label, sorted(expected), sorted(actual)
            )
        )


def _write_immutable_documents(documents):
    pending = []
    for path, expected in documents:
        path = Path(path)
        if not path.is_file():
            pending.append((path, expected))
            continue
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserError("invalid immutable output {}: {}".format(path, error))
        if _canonical(existing) != _canonical(expected):
            raise UserError(
                "immutable output differs from recomputed result: {}".format(path)
            )
    # Validate every existing document before creating any missing peer.  This
    # keeps FINAL_SELECTION and FROZEN_HPARAMETERS atomic as a logical pair.
    for path, expected in pending:
        atomic_json(path, expected)


def _load_frozen_selection(batch_root, method, settings, spec):
    batch_root = Path(batch_root)
    selection_path = batch_root / "FINAL_SELECTION.json"
    frozen_path = batch_root / "FROZEN_HPARAMETERS.json"
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(
            "orders require completed canonical final selection: {}".format(error)
        )
    if selection.get("schema") != "navtta.vln_tta_final_selection.v1":
        raise UserError("unsupported final-selection schema")
    if frozen.get("schema") != "navtta.vln_tta_frozen_hparams.v1":
        raise UserError("unsupported frozen-hyperparameter schema")
    if selection.get("method") != method or frozen.get("method") != method:
        raise UserError("frozen method does not match order evaluation")
    current_commit = git("rev-parse", "HEAD")
    current_spec = sha256(SPEC_PATH)
    for label, document in (("FINAL_SELECTION", selection),
                            ("FROZEN_HPARAMETERS", frozen)):
        checks = (
            ("git_commit", current_commit),
            ("spec_sha256", current_spec),
            ("split", spec["split"]),
        )
        for key, expected in checks:
            if document.get(key) != expected:
                raise UserError(
                    "{} mismatch for {}: expected {!r}, got {!r}".format(
                        label, key, expected, document.get(key)
                    )
                )
        _validate_exact_setting_keys(document, settings, label)
    try:
        final_manifest = json.loads(
            (batch_root / "stages/final/stage_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        control_manifest = json.loads(
            (batch_root / "stages/final_controls/stage_manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(
            "frozen selection lacks completed final-stage provenance: {}".format(
                error
            )
        )
    batch_id = final_manifest.get("batch_id")
    if (not isinstance(batch_id, str) or not batch_id
            or control_manifest.get("batch_id") != batch_id):
        raise UserError("final and final_controls batch provenance disagrees")
    authenticated_selection, authenticated_frozen = _derive_final_documents(
        batch_root, batch_id, method, settings, spec
    )
    if _canonical(selection) != _canonical(authenticated_selection):
        raise UserError(
            "FINAL_SELECTION is not authenticated by completed final evidence"
        )
    if _canonical(frozen) != _canonical(authenticated_frozen):
        raise UserError(
            "FROZEN_HPARAMETERS is not authenticated by completed final evidence"
        )
    output = {}
    for setting in settings:
        try:
            selected = selection["settings"][setting]
            parameters = frozen["settings"][setting]
        except KeyError:
            raise UserError("missing frozen selection for {}".format(setting))
        if _canonical(selected.get("frozen_parameters")) != _canonical(parameters):
            raise UserError("final selection and frozen parameters disagree for {}".format(
                setting
            ))
        output[setting] = {
            "winner_run_tag": selected["winner_run_tag"],
            "parameters": parameters,
        }
    return output


def _order_parameters(method, frozen, order_seed):
    if type(order_seed) is not int or order_seed not in (0, 1, 2):
        raise UserError("order_seed must be an exact integer in [0, 1, 2]")
    parameters = dict(frozen)
    if method == "feedtta":
        parameters["action_seed"] = order_seed
        parameters["sgr_seed"] = order_seed
    # EAM replay sampling intentionally consumes the runner's global model
    # seed, which run_source_eval.sh derives from --order-seed.
    return parameters


def _stage_candidates(method, stage, settings, spec, batch_root):
    promotion_records = []
    if stage == "smoke":
        return {
            setting: [_candidate(anchor_for(method, setting, spec))]
            for setting in settings
        }, promotion_records
    if stage in ("controls", "final_controls"):
        return _source_candidates(method, settings, spec), promotion_records
    if stage == "stage1":
        return {
            setting: _ensure_anchor([
                _candidate(point) for point in stage1_points(method, setting, spec)
            ], method, setting, spec)
            for setting in settings
        }, promotion_records

    if stage == "orders":
        frozen = _load_frozen_selection(batch_root, method, settings, spec)
        candidates = {}
        for setting in settings:
            winner = frozen[setting]
            candidates[setting] = [
                _candidate(
                    _order_parameters(method, winner["parameters"], order_seed),
                    (winner["winner_run_tag"],),
                    order_seed=order_seed,
                )
                for order_seed in spec["final_order_seeds"]
            ]
        return candidates, promotion_records

    previous = previous_search_stage(method, stage)
    prior = load_stage_results(
        batch_root / "stages" / previous, spec, allow_partial=True
    )
    prior_by_setting = {
        setting: [item for item in prior if item["setting"] == setting]
        for setting in settings
    }
    controls = load_stage_results(batch_root / "stages" / "controls", spec)
    source_by_setting = {item["setting"]: item for item in controls}
    selected_by_setting = {}
    for setting in settings:
        selected, record = rank_and_promote(
            prior_by_setting[setting], source_by_setting.get(setting), method,
            setting, promotion_limit(method, stage, spec), spec,
            force_anchor=True,
        )
        promotion_records.append(record)
        selected_by_setting[setting] = selected

    if stage in ("stage2", "stage3"):
        return {
            setting: expand_stage(
                method, stage, setting, selected_by_setting[setting], spec
            ) for setting in settings
        }, promotion_records
    if stage == "final":
        return {
            setting: [
                _candidate(item["parameters"], (item["run_tag"],))
                for item in selected_by_setting[setting]
            ] for setting in settings
        }, promotion_records
    raise UserError("unknown stage {}".format(stage))


def build_jobs(args, spec, stage_dir, stage=None, candidates_by_setting=None):
    stage = stage or ("smoke" if getattr(args, "smoke", False)
                      else getattr(args, "stage", "stage1"))
    settings = list(getattr(args, "settings", None) or spec["settings"])
    unknown = set(settings).difference(spec["settings"])
    if unknown:
        raise UserError("unknown settings: {}".format(", ".join(sorted(unknown))))
    if candidates_by_setting is None:
        if stage == "smoke":
            candidates_by_setting = {
                setting: [_candidate(anchor_for(args.method, setting, spec))]
                for setting in settings
            }
        elif stage == "stage1":
            candidates_by_setting = {
                setting: _ensure_anchor([
                    _candidate(point)
                    for point in stage1_points(args.method, setting, spec)
                ], args.method, setting, spec)
                for setting in settings
            }
        else:
            raise UserError("candidates are required for stage {}".format(stage))
    episodes = resolve_stage_episodes(
        stage, getattr(args, "episodes", None), spec
    )
    gpu = str(getattr(args, "gpu", 0))

    jobs = []
    max_points = max(len(points) for points in candidates_by_setting.values())
    ordinal = 0
    for point_index in range(max_points):
        for setting in settings:
            points = candidates_by_setting[setting]
            if point_index >= len(points):
                continue
            candidate = points[point_index]
            candidate_order_seed = candidate.get("order_seed")
            if stage == "orders":
                if (type(candidate_order_seed) is not int
                        or candidate_order_seed not in spec["final_order_seeds"]):
                    raise UserError(
                        "orders candidates require one of the declared order seeds"
                    )
            elif candidate_order_seed is not None:
                raise UserError("only orders candidates may set order_seed")
            parameters = candidate["parameters"]
            config_method = candidate.get("config_method", args.method)
            config_identity = {
                "method": config_method,
                "parameters": parameters,
                "order_seed": candidate.get("order_seed"),
            }
            digest = point_tag(config_identity)
            # Tuning outputs share one global result namespace.  Keep the
            # search method in every newly planned tag even when the executed
            # config is method-independent Source; persisted plans retain
            # their recorded legacy tags when resumed.
            base_run_tag = "{}-{}-{}-{:04d}-{}-{}".format(
                args.batch_id, args.method, stage, ordinal, setting, digest
            )
            job_dir = Path(stage_dir) / "jobs" / setting / base_run_tag
            config_path = job_dir / "parameters.json"
            result_root = tuning_result_root(
                args.method, args.batch_id, stage, setting, base_run_tag,
                result_namespace=args.method,
            )
            command = [
                str(RUNNER), setting, "val_seen", gpu,
                "--run-tag", base_run_tag, "--tta-config", str(config_path),
                "--result-root", str(result_root),
            ]
            if episodes > 0:
                command += ["--episode-limit", str(episodes)]
            order_seed = candidate_order_seed
            if order_seed is not None:
                command += ["--order-seed", str(order_seed)]
            job = {
                "batch_id": args.batch_id,
                "ordinal": ordinal,
                "base_run_tag": base_run_tag,
                "run_tag": base_run_tag,
                "attempt": 0,
                "setting": setting,
                "model": SETTING_MODEL[setting],
                "family": "continuous" if setting in CONTINUOUS_SETTINGS else "discrete",
                "benchmark": (
                    "reverie" if setting.endswith("reverie")
                    else "r2r-ce" if setting.endswith("r2r-ce") else "r2r"
                ),
                "search_method": args.method,
                "config_method": config_method,
                "result_layout": RESULT_LAYOUT,
                "result_namespace": args.method,
                "stage": stage,
                "episodes": episodes,
                "order_seed": order_seed,
                "parameters": parameters,
                "parent_run_tags": candidate.get("parent_run_tags", []),
                "config_path": str(config_path),
                "job_dir": str(job_dir),
                "result_root": str(result_root),
                "command": command,
            }
            jobs.append(job)
            ordinal += 1
    return jobs


def _write_job(job):
    job_dir = Path(job["job_dir"])
    job_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "navtta.vln_tta_job.v1",
        "method": job["config_method"],
        "search_method": job["search_method"],
        "stage": job["stage"],
        "episodes": job["episodes"],
        "parameters": job["parameters"],
    }
    if job.get("result_layout") is not None:
        config.update({
            "batch_id": job.get("batch_id"),
            "setting": job.get("setting"),
            "run_tag": job.get("run_tag"),
            "result_layout": job.get("result_layout"),
            "result_namespace": job.get("result_namespace"),
        })
    if job["order_seed"] is not None:
        config["order_seed"] = job["order_seed"]
    atomic_json(job["config_path"], config)
    atomic_json(job_dir / "job.json", job)


def write_plan(stage_dir, jobs):
    for job in jobs:
        _write_job(job)
    fields = [
        "ordinal", "run_tag", "setting", "model", "family", "benchmark",
        "search_method", "config_method", "stage", "episodes", "order_seed",
        "result_layout", "result_namespace", "config_path", "job_dir",
        "result_root", "command",
    ]
    with (Path(stage_dir) / "grid.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            row = {key: job.get(key) for key in fields}
            row["command"] = json.dumps(row["command"])
            writer.writerow(row)


def load_jobs(stage_dir):
    jobs_root = Path(stage_dir) / "jobs"
    paths = sorted(
        list(jobs_root.glob("*/job.json"))
        + list(jobs_root.glob("*/*/job.json"))
    )
    if not paths:
        raise UserError("stage has no persisted jobs: {}".format(stage_dir))
    jobs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return sorted(jobs, key=lambda job: int(job["ordinal"]))


def cgroup_memory_gib():
    for candidate in (Path("/sys/fs/cgroup/memory.current"),
                      Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            return int(candidate.read_text().strip()) / 1024 ** 3
        except (OSError, ValueError):
            pass
    return 0.0


def gpu_stats():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0].split(",")
        return int(output[0].strip()), int(output[1].strip())
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return 0, 0


def append_resource(path, running):
    exists = Path(path).exists()
    gpu_memory, gpu_util = gpu_stats()
    with Path(path).open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow([
                "unix_time", "gpu_memory_mib", "gpu_util_percent",
                "cgroup_memory_gib", "running_jobs", "running_run_tags",
            ])
        writer.writerow([
            "{:.3f}".format(time.time()), gpu_memory, gpu_util,
            "{:.3f}".format(cgroup_memory_gib()), len(running),
            "|".join(sorted(
                active["job"]["run_tag"] for active in running.values()
            )),
        ])


def summarize_smoke_resources(stage_dir, jobs):
    resource_path = Path(stage_dir) / "resource.csv"
    if not resource_path.is_file():
        raise UserError("smoke stage has no resource samples")
    by_tag = {job["run_tag"]: [] for job in jobs}
    with resource_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            tags = [tag for tag in row.get("running_run_tags", "").split("|") if tag]
            if len(tags) == 1 and tags[0] in by_tag:
                by_tag[tags[0]].append(row)
    profiles = {}
    for job in jobs:
        rows = by_tag[job["run_tag"]]
        if not rows:
            raise UserError(
                "smoke job has no isolated resource samples: {}".format(
                    job["run_tag"]
                )
            )
        profiles[job["setting"]] = {
            "run_tag": job["run_tag"],
            "samples": len(rows),
            "peak_gpu_memory_mib": max(int(row["gpu_memory_mib"]) for row in rows),
            "peak_cgroup_memory_gib": max(
                float(row["cgroup_memory_gib"]) for row in rows
            ),
            "peak_gpu_util_percent": max(
                int(row["gpu_util_percent"]) for row in rows
            ),
        }
    value = {
        "schema": "navtta.vln_tta_smoke_resource_profile.v1",
        "measurement": "one isolated scheduler worker at a time",
        "settings": profiles,
    }
    atomic_json(Path(stage_dir) / "RESOURCE_PROFILE.json", value)
    return value


def write_summary(stage_dir, jobs, spec):
    summary_path = Path(stage_dir) / "SUMMARY.json"
    prior_promotion_ready = None
    if summary_path.is_file():
        try:
            prior_promotion_ready = json.loads(
                summary_path.read_text(encoding="utf-8")
            ).get("promotion_ready")
        except (OSError, json.JSONDecodeError):
            pass
    results, errors = [], []
    for job in jobs:
        exit_path = Path(job["job_dir"]) / "exitcode"
        if not exit_path.is_file():
            continue
        try:
            code = int(exit_path.read_text().strip())
        except ValueError:
            errors.append({"run_tag": job["run_tag"], "error": "invalid exitcode"})
            continue
        if code != 0:
            errors.append({"run_tag": job["run_tag"], "exit_code": code})
            continue
        try:
            result = parse_metrics(job, spec)
            atomic_json(Path(job["job_dir"]) / "metrics.json", result)
            results.append(result)
        except Exception as error:  # Invalid evidence is a failed job.
            errors.append({"run_tag": job["run_tag"], "error": str(error)})
    terminal = len(results) + len(errors) == len(jobs)
    atomic_json(summary_path, {
        "schema": "navtta.vln_tta_stage_summary.v1",
        "planned": len(jobs),
        "validated": len(results),
        "errors": errors,
        "terminal": terminal,
        "promotion_ready": prior_promotion_ready,
        "complete": len(results) == len(jobs) and not errors,
    })
    metric_names = sorted({key for result in results for key in result["metrics"]})
    with (Path(stage_dir) / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = [
            "run_tag", "setting", "search_method", "config_method", "stage",
            "order_seed",
        ] + metric_names
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fields if key not in metric_names}
            row.update(result["metrics"])
            writer.writerow(row)
    return results, errors


def load_stage_results(stage_dir, spec, allow_partial=False):
    jobs = load_jobs(stage_dir)
    results, errors = write_summary(stage_dir, jobs, spec)
    terminal = len(results) + len(errors) == len(jobs)
    if not terminal or (not allow_partial and errors):
        raise UserError("prerequisite stage is incomplete: {}".format(stage_dir))
    return results


def screening_promotion_preview(batch_root, batch_id, stage, method, settings,
                                results, spec):
    _validate_persisted_stage(
        Path(batch_root) / "stages" / "controls",
        batch_id,
        method, settings, spec, "controls",
    )
    _validate_persisted_stage(
        Path(batch_root) / "stages" / stage,
        batch_id,
        method, settings, spec, stage,
    )
    stages = workflow_stages(method)
    try:
        next_stage = stages[stages.index(stage) + 1]
    except (ValueError, IndexError):
        raise UserError("{} is not a screening stage for {}".format(stage, method))
    # The full-val matched Source stage is an execution barrier, not a search
    # promotion.  Finalists still come directly from the last 256-episode
    # screening stage and are ranked against the screening Source control.
    if next_stage == "final_controls":
        next_stage = "final"
    controls = load_stage_results(
        Path(batch_root) / "stages" / "controls", spec
    )
    source_by_setting = {item["setting"]: item for item in controls}
    records = []
    for setting in settings:
        candidates = [item for item in results if item["setting"] == setting]
        _, record = rank_and_promote(
            candidates, source_by_setting.get(setting), method, setting,
            promotion_limit(method, next_stage, spec), spec,
            force_anchor=True,
        )
        records.append(record)
    value = {
        "schema": "navtta.vln_tta_promotion_preview.v1",
        "from_stage": stage,
        "to_stage": next_stage,
        "settings": records,
    }
    atomic_json(Path(batch_root) / "stages" / stage / "promotion_preview.json", value)
    summary_path = Path(batch_root) / "stages" / stage / "SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["promotion_ready"] = True
    atomic_json(summary_path, summary)
    return value


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def worker_main(job_path):
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_dir = Path(job["job_dir"])
    console_path = job_dir / "console.log"
    exit_path = job_dir / "exitcode"
    state_path = job_dir / "worker_state.json"
    stop_signal = {"value": None}
    child = {"process": None}

    def forward(signum, frame):
        del frame
        stop_signal["value"] = signum
        process = child["process"]
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    atomic_json(state_path, {
        "status": "starting", "worker_pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    with console_path.open("ab", buffering=0) as console:
        process = subprocess.Popen(
            job["command"], cwd=str(REPO_ROOT), stdout=console,
            stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
        )
        child["process"] = process
        atomic_json(state_path, {
            "status": "running", "worker_pid": os.getpid(),
            "runner_pid": process.pid,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        code = process.wait()
    if stop_signal["value"] is not None and code == 0:
        code = 128 + int(stop_signal["value"])
    atomic_text(exit_path, str(code) + "\n")
    atomic_json(state_path, {
        "status": "finished", "worker_pid": os.getpid(),
        "runner_pid": child["process"].pid, "exit_code": code,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return code


def _bump_attempt(job):
    job_dir = Path(job["job_dir"])
    attempt_root = job_dir / "attempts" / "attempt-{:02d}".format(
        int(job.get("attempt", 0))
    )
    attempt_root.mkdir(parents=True, exist_ok=False)
    old_job = dict(job)
    atomic_json(attempt_root / "job.json", old_job)
    for name in ("console.log", "exitcode", "metrics.json", "worker_state.json"):
        path = job_dir / name
        if path.exists():
            path.rename(attempt_root / name)
    archived = {}
    old_result_root = Path(job["result_root"])
    if old_result_root.exists():
        destination = attempt_root / "result_root"
        old_result_root.rename(destination)
        archived["result_root"] = str(destination)
    data_version = (
        "v1.3-unified" if job["setting"] in CONTINUOUS_SETTINGS else "native"
    )
    formal_dir = REPO_ROOT / "vln/results/runs" / (
        "{}-{}-val_seen-{}".format(
            job["run_tag"], job["setting"], data_version
        )
    )
    if formal_dir.exists():
        destination = attempt_root / "formal_run_manifest"
        formal_dir.rename(destination)
        archived["formal_run_manifest"] = str(destination)
    atomic_json(attempt_root / "archived_evidence.json", archived)
    job["attempt"] = int(job.get("attempt", 0)) + 1
    job["run_tag"] = "{}-retry{}".format(job["base_run_tag"], job["attempt"])
    if job.get("result_layout") == RESULT_LAYOUT:
        job["result_root"] = str(tuning_result_root(
            job["search_method"], job["batch_id"], job["stage"],
            job["setting"], job["run_tag"],
            result_namespace=job.get("result_namespace"),
        ))
    else:
        job["result_root"] = str(
            TUNING_ROOT / job["run_tag"] / job["setting"] / "val_seen"
        )
    command = list(job["command"])
    command[command.index("--run-tag") + 1] = job["run_tag"]
    if "--result-root" in command:
        command[command.index("--result-root") + 1] = job["result_root"]
    elif job.get("result_layout") == RESULT_LAYOUT:
        command += ["--result-root", job["result_root"]]
    job["command"] = command
    _write_job(job)


def _worker_pid(job):
    state_path = Path(job["job_dir"]) / "worker_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pid = int(state["worker_pid"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return pid if state.get("status") in ("starting", "running") else None


def _progress(stage_dir, jobs, running, pending):
    succeeded = failed = 0
    for job in jobs:
        path = Path(job["job_dir"]) / "exitcode"
        if not path.is_file():
            continue
        try:
            code = int(path.read_text().strip())
        except ValueError:
            code = 255
        if code == 0:
            succeeded += 1
        else:
            failed += 1
    value = {
        "planned": len(jobs), "pending": len(pending),
        "running": len(running), "succeeded": succeeded, "failed": failed,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "running_jobs": [active["job"]["run_tag"] for active in running.values()],
    }
    atomic_json(Path(stage_dir) / "progress.json", value)
    return value


def run_batch(args, stage_dir, jobs, spec):
    stage = jobs[0]["stage"] if jobs else Path(stage_dir).name
    screening_stage = stage in ("stage1", "stage2", "stage3")
    effective_fail_fast = bool(args.fail_fast and not screening_stage)
    max_workers = 1 if stage == "smoke" else args.max_workers
    max_per_model = 1 if stage == "smoke" else args.max_per_model
    max_discrete_workers = 1 if stage == "smoke" else args.max_discrete_workers
    max_continuous_workers = 1 if stage == "smoke" else args.max_continuous_workers
    if args.retry_failed:
        ordered = workflow_stages(args.method, include_orders=True)
        stage_index = ordered.index(stage)
        downstream = Path(stage_dir).parent
        existing_downstream = [
            name for name in ordered[stage_index + 1:]
            if (downstream / name / "stage_manifest.json").is_file()
        ]
        if existing_downstream:
            raise UserError(
                "cannot retry {} after downstream plans exist: {}".format(
                    stage, ", ".join(existing_downstream)
                )
            )
    running = {}
    pending = []
    terminal_failure = False
    for job in jobs:
        exit_path = Path(job["job_dir"]) / "exitcode"
        if exit_path.is_file():
            try:
                code = int(exit_path.read_text().strip())
            except ValueError:
                code = 255
            if code == 0:
                try:
                    parse_metrics(job, spec)
                    continue
                except Exception:
                    code = 255
            if args.retry_failed:
                _bump_attempt(job)
                pending.append(job)
            else:
                terminal_failure = True
            continue
        pid = _worker_pid(job)
        if pid is not None and process_alive(pid):
            running[pid] = {"process": None, "job": job, "external": True}
        elif pid is not None:
            if args.retry_failed:
                _bump_attempt(job)
                pending.append(job)
            else:
                terminal_failure = True
        else:
            pending.append(job)

    stop_requested = False

    def request_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    scheduler_log = Path(stage_dir) / "scheduler.log"
    resource_log = Path(stage_dir) / "resource.csv"
    last_resource = 0.0
    resource_blocked_since = None
    launch_blocked = terminal_failure and effective_fail_fast
    with scheduler_log.open("a", encoding="utf-8", buffering=1) as log:
        def record(message):
            log.write("{} {}\n".format(
                time.strftime("%Y-%m-%dT%H:%M:%S%z"), message
            ))

        record("scheduler_start jobs={} workers={} per_model={} resume={}".format(
            len(jobs), max_workers, max_per_model, int(args.resume)
        ))
        while pending or running:
            for pid, active in list(running.items()):
                exit_path = Path(active["job"]["job_dir"]) / "exitcode"
                process = active["process"]
                if exit_path.is_file():
                    try:
                        code = int(exit_path.read_text().strip())
                    except ValueError:
                        code = 255
                    if process is not None:
                        process.poll()
                    record("finish run_tag={} exit={}".format(
                        active["job"]["run_tag"], code
                    ))
                    if code != 0:
                        terminal_failure = True
                        launch_blocked = launch_blocked or effective_fail_fast
                    del running[pid]
                elif not process_alive(pid):
                    record("orphaned_worker run_tag={} pid={}".format(
                        active["job"]["run_tag"], pid
                    ))
                    terminal_failure = True
                    launch_blocked = launch_blocked or effective_fail_fast
                    del running[pid]

            if stop_requested:
                launch_blocked = True
                terminal_failure = True
                for pid in list(running):
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                record("stop_requested")

            now = time.time()
            if now - last_resource >= 5.0:
                append_resource(resource_log, running)
                _progress(stage_dir, jobs, running, pending)
                last_resource = now

            model_counts = {}
            family_counts = {"discrete": 0, "continuous": 0}
            for active in running.values():
                model = active["job"]["model"]
                family = active["job"]["family"]
                model_counts[model] = model_counts.get(model, 0) + 1
                family_counts[family] += 1

            launched = False
            if (len(running) < max_workers and pending and
                    not stop_requested and not launch_blocked):
                gpu_memory, _ = gpu_stats()
                memory = cgroup_memory_gib()
                resources_ok = (
                    gpu_memory <= args.max_gpu_memory_mib
                    and memory <= args.max_memory_gib
                )
                if resources_ok:
                    resource_blocked_since = None
                    selected = None
                    for index, job in enumerate(pending):
                        family_limit = (
                            max_continuous_workers
                            if job["family"] == "continuous"
                            else max_discrete_workers
                        )
                        if (model_counts.get(job["model"], 0) < max_per_model
                                and family_counts[job["family"]] < family_limit):
                            selected = index
                            break
                    if selected is not None:
                        job = pending.pop(selected)
                        process = subprocess.Popen(
                            [sys.executable, str(Path(__file__).resolve()),
                             "--worker-job", str(Path(job["job_dir"]) / "job.json")],
                            cwd=str(REPO_ROOT), start_new_session=True,
                        )
                        running[process.pid] = {
                            "process": process, "job": job, "external": False,
                        }
                        record("launch worker_pid={} run_tag={}".format(
                            process.pid, job["run_tag"]
                        ))
                        launched = True
                        time.sleep(args.launch_stagger)
                else:
                    if resource_blocked_since is None:
                        resource_blocked_since = now
                        record("resource_wait gpu_mib={} memory_gib={:.3f}".format(
                            gpu_memory, memory
                        ))
                    if (not running and now - resource_blocked_since
                            >= args.resource_wait_timeout):
                        raise UserError(
                            "resource thresholds blocked all launches for {} seconds"
                            .format(args.resource_wait_timeout)
                        )
            if launch_blocked and not running:
                break
            if not launched:
                time.sleep(1.0)
        record("scheduler_finish failed={} pending={}".format(
            int(terminal_failure), len(pending)
        ))
    _progress(stage_dir, jobs, running, pending)
    results, errors = write_summary(stage_dir, jobs, spec)
    terminal = len(results) + len(errors) == len(jobs)
    if stop_requested or not terminal:
        raise UserError("stage completed with failed, invalid, or pending jobs")
    if screening_stage:
        screening_promotion_preview(
            Path(stage_dir).parent.parent, args.batch_id, stage, args.method,
            args.settings or spec["settings"], results, spec,
        )
    elif terminal_failure or errors:
        raise UserError("strict stage completed with failed or invalid jobs")
    if stage == "smoke":
        summarize_smoke_resources(stage_dir, jobs)
    return results


def _stage_manifest(args, stage, jobs, spec):
    return {
        "schema": "navtta.vln_tta_search_stage.v1",
        "batch_id": args.batch_id,
        "method": args.method,
        "stage": stage,
        "episodes": jobs[0]["episodes"] if jobs else None,
        "job_count": len(jobs),
        "git_commit": git("rev-parse", "HEAD"),
        "spec_path": str(SPEC_PATH),
        "spec_sha256": sha256(SPEC_PATH),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "settings": args.settings or spec["settings"],
        "final_order_seeds": spec["final_order_seeds"],
        "result_layout": RESULT_LAYOUT,
        "resource_limits": {
            "max_workers": args.max_workers,
            "max_per_model": args.max_per_model,
            "max_discrete_workers": args.max_discrete_workers,
            "max_continuous_workers": args.max_continuous_workers,
            "max_gpu_memory_mib": args.max_gpu_memory_mib,
            "max_memory_gib": args.max_memory_gib,
        },
    }


def ensure_batch_manifest(args, spec, batch_root):
    path = Path(batch_root) / "batch.json"
    expected = {
        "schema": "navtta.vln_tta_search_batch.v1",
        "batch_id": args.batch_id,
        "method": args.method,
        "split": spec["split"],
        "settings": list(args.settings or spec["settings"]),
        "primary_order_seed": spec["primary_order_seed"],
        "git_commit": git("rev-parse", "HEAD"),
        "spec_path": str(SPEC_PATH),
        "spec_sha256": sha256(SPEC_PATH),
        "result_layout": RESULT_LAYOUT,
    }
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserError("invalid batch manifest {}: {}".format(path, error))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise UserError(
                    "batch manifest {} mismatch for {}".format(path, key)
                )
        return existing
    document = dict(expected)
    document["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(path, document)
    return document


def ensure_stage_plan(args, spec, batch_root, stage):
    stage_dir = Path(batch_root) / "stages" / stage
    manifest_path = stage_dir / "stage_manifest.json"
    settings = list(args.settings or spec["settings"])
    resolve_stage_episodes(stage, getattr(args, "episodes", None), spec)
    if manifest_path.is_file():
        if not args.resume:
            raise UserError("stage exists; use --resume: {}".format(stage_dir))
        _, jobs = _validate_persisted_stage(
            stage_dir, args.batch_id, args.method, settings, spec, stage
        )
        _validate_stage_prerequisites(
            batch_root, args.batch_id, args.method, stage, settings, spec
        )
        return stage_dir, jobs
    if stage_dir.exists() and any(stage_dir.iterdir()):
        raise UserError("stage directory is nonempty without a manifest")
    _validate_stage_prerequisites(
        batch_root, args.batch_id, args.method, stage, settings, spec
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    candidates, promotion = _stage_candidates(
        args.method, stage, settings, spec, batch_root
    )
    jobs = build_jobs(args, spec, stage_dir, stage, candidates)
    _validate_jobs(
        jobs, args.batch_id, args.method, settings, stage, spec,
        expected_job_count=len(jobs),
    )
    write_plan(stage_dir, jobs)
    if promotion:
        atomic_json(stage_dir / "promotion.json", {
            "schema": "navtta.vln_tta_promotion.v1",
            "from_stage": previous_search_stage(args.method, stage),
            "to_stage": stage,
            "settings": promotion,
        })
    atomic_json(manifest_path, _stage_manifest(args, stage, jobs, spec))
    return stage_dir, jobs


def _derive_final_documents(batch_root, batch_id, method, settings, spec):
    batch_root = Path(batch_root)
    _validate_persisted_stage(
        batch_root / "stages" / "final_controls",
        batch_id,
        method, settings, spec, "final_controls",
    )
    _validate_persisted_stage(
        batch_root / "stages" / "final",
        batch_id,
        method, settings, spec, "final",
    )
    finalists = load_stage_results(batch_root / "stages" / "final", spec)
    controls = load_stage_results(
        batch_root / "stages" / "final_controls", spec
    )
    expected_finalists = int(
        spec["protocol"]["full_val_seen_finalists_per_setting"]
    )
    output = {
        "schema": "navtta.vln_tta_final_selection.v1",
        "method": method,
        "split": spec["split"],
        "finalist_stage": "final",
        "matched_source_stage": "final_controls",
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": sha256(SPEC_PATH),
        "settings": {},
    }
    for setting in settings:
        setting_finalists = [
            item for item in finalists if item["setting"] == setting
        ]
        setting_controls = [
            item for item in controls if item["setting"] == setting
        ]
        if len(setting_finalists) != expected_finalists:
            raise UserError(
                "{} has {} full-val finalists; expected {}".format(
                    setting, len(setting_finalists), expected_finalists
                )
            )
        if len(setting_controls) != 1:
            raise UserError(
                "{} has {} full-val Source controls; expected 1".format(
                    setting, len(setting_controls)
                )
            )
        selected, record = rank_and_promote(
            setting_finalists, setting_controls[0], method, setting, 1, spec,
            force_anchor=False,
        )
        winner = selected[0]
        output["settings"][setting] = {
            "winner_run_tag": winner["run_tag"],
            "source_run_tag": setting_controls[0]["run_tag"],
            "frozen_parameters": winner["parameters"],
            "winner_metrics": winner["metrics"],
            "source_metrics": setting_controls[0]["metrics"],
            "selection": record,
        }
    frozen_output = {
        "schema": "navtta.vln_tta_frozen_hparams.v1",
        "method": method,
        "split": spec["split"],
        "generated_from": "FINAL_SELECTION.json",
        "git_commit": output["git_commit"],
        "spec_sha256": output["spec_sha256"],
        "settings": {
            setting: output["settings"][setting]["frozen_parameters"]
            for setting in settings
        },
    }
    return output, frozen_output


def summarize_final(batch_root, batch_id, method, settings, spec):
    batch_root = Path(batch_root)
    output, frozen_output = _derive_final_documents(
        batch_root, batch_id, method, settings, spec
    )
    _write_immutable_documents((
        (batch_root / "FINAL_SELECTION.json", output),
        (batch_root / "FROZEN_HPARAMETERS.json", frozen_output),
    ))
    return output


def summarize_orders(batch_root, batch_id, method, settings, spec):
    _validate_persisted_stage(
        Path(batch_root) / "stages" / "orders",
        batch_id,
        method, settings, spec, "orders",
    )
    frozen = _load_frozen_selection(batch_root, method, settings, spec)
    results = load_stage_results(Path(batch_root) / "stages" / "orders", spec)
    output = {
        "schema": "navtta.vln_tta_order_robustness.v1",
        "method": method,
        "split": spec["split"],
        "git_commit": git("rev-parse", "HEAD"),
        "spec_sha256": sha256(SPEC_PATH),
        "required_order_seeds": spec["final_order_seeds"],
        "settings": {},
    }
    for setting in settings:
        values = [item for item in results if item["setting"] == setting]
        if any(
                type(item.get("order_seed")) is not int
                for item in values):
            raise UserError("{} contains a non-integer order seed".format(setting))
        values.sort(key=lambda item: item["order_seed"])
        seeds = [item["order_seed"] for item in values]
        if seeds != spec["final_order_seeds"]:
            raise UserError("{} does not have exactly order seeds {}".format(
                setting, spec["final_order_seeds"]
            ))
        for item in values:
            expected = _order_parameters(
                method, frozen[setting]["parameters"], item["order_seed"]
            )
            if _canonical(item["parameters"]) != _canonical(expected):
                raise UserError(
                    "order run changed frozen parameters for {} seed {}".format(
                        setting, item["order_seed"]
                    )
                )
        metrics = sorted(set.intersection(*(
            set(item["metrics"]) for item in values
        )))
        output["settings"][setting] = {
            "frozen_parameters": frozen[setting]["parameters"],
            "runs": [{
                "run_tag": item["run_tag"], "order_seed": item["order_seed"],
                "metrics": item["metrics"],
            } for item in values],
            "aggregate": {
                metric: {
                    "mean": statistics.fmean(item["metrics"][metric] for item in values),
                    "sample_std": statistics.stdev(
                        item["metrics"][metric] for item in values
                    ),
                } for metric in metrics
            },
        }
    _write_immutable_documents((
        (Path(batch_root) / "ORDER_ROBUSTNESS.json", output),
    ))
    return output


def runner_supports_order_seed():
    try:
        return "--order-seed" in RUNNER.read_text(encoding="utf-8")
    except OSError:
        return False


def execute_method(args, spec):
    batch_root = LOG_ROOT / args.method / args.batch_id
    batch_root.mkdir(parents=True, exist_ok=True)
    ensure_batch_manifest(args, spec, batch_root)
    stages = (
        workflow_stages(args.method, include_orders=args.with_orders)
        if args.stage == "all" else (args.stage,)
    )
    if args.stage == "all" and args.episodes is not None:
        raise UserError("--episodes cannot override the complete workflow")
    settings = args.settings or spec["settings"]
    for stage in stages:
        if stage == "orders" and not runner_supports_order_seed():
            raise UserError(
                "run_source_eval.sh lacks --order-seed support; refusing to fake "
                "the required three-order evaluation"
            )
        stage_dir, jobs = ensure_stage_plan(args, spec, batch_root, stage)
        if args.dry_run:
            for job in jobs:
                print(subprocess.list2cmdline(job["command"]))
            print("stage={} jobs={}".format(stage, len(jobs)))
            if len(stages) > 1:
                raise UserError(
                    "--dry-run all cannot synthesize data-dependent promotions; "
                    "inspect one stage at a time"
                )
            continue
        run_batch(args, stage_dir, jobs, spec)
        if stage == "final":
            summarize_final(
                batch_root, args.batch_id, args.method, settings, spec
            )
    if stages[-1] == "orders" and not args.dry_run:
        summarize_orders(
            batch_root, args.batch_id, args.method, settings, spec
        )


def campaign_status(method, batch_id, watch=False):
    methods = METHODS if method == "all" else (method,)
    while True:
        snapshot = {"batch_id": batch_id, "methods": {}}
        terminal = True
        for item in methods:
            root = LOG_ROOT / item / batch_id / "stages"
            stage_values = {}
            if root.is_dir():
                for stage_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                    progress = stage_dir / "progress.json"
                    if progress.is_file():
                        value = json.loads(progress.read_text(encoding="utf-8"))
                    else:
                        try:
                            jobs = load_jobs(stage_dir)
                            value = _progress(stage_dir, jobs, {}, jobs)
                        except UserError:
                            continue
                    stage_values[stage_dir.name] = value
                    terminal = terminal and (
                        value["succeeded"] + value["failed"] == value["planned"]
                    )
            else:
                terminal = False
            snapshot["methods"][item] = stage_values
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        if not watch or terminal:
            return
        time.sleep(10)


def parse_args(argv=None):
    defaults = load_spec()["scheduler_defaults"]
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS + ("all",))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--settings", nargs="+")
    parser.add_argument(
        "--stage", choices=(
            "smoke", "controls", "stage1", "stage2", "stage3",
            "final_controls", "final", "orders", "all",
        ), default="all",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--with-orders", action="store_true",
        help="after canonical finalist selection, also run order seeds 0, 1, 2",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="compatibility alias for --stage smoke")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=defaults["max_workers"])
    parser.add_argument("--max-per-model", type=int,
                        default=defaults["max_per_model"])
    parser.add_argument("--max-discrete-workers", type=int,
                        default=defaults["max_discrete_workers"])
    parser.add_argument("--max-continuous-workers", type=int,
                        default=defaults["max_continuous_workers"])
    parser.add_argument("--max-gpu-memory-mib", type=int,
                        default=defaults["max_gpu_memory_mib_before_launch"])
    parser.add_argument("--max-memory-gib", type=float,
                        default=defaults["max_cgroup_memory_gib_before_launch"])
    parser.add_argument("--launch-stagger", type=float,
                        default=defaults["launch_stagger_seconds"])
    parser.add_argument("--resource-wait-timeout", type=float,
                        default=defaults["resource_wait_timeout_seconds"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.set_defaults(fail_fast=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        if args.stage not in ("all", "smoke"):
            parser.error("--smoke conflicts with --stage")
        args.stage = "smoke"
    if args.with_orders and args.stage != "all":
        parser.error("--with-orders requires --stage all")
    if not re.match(r"^[A-Za-z0-9._-]+$", args.batch_id):
        parser.error("invalid batch id")
    if args.episodes is not None and (args.episodes == 0 or args.episodes < -1):
        parser.error("episodes must be -1 or positive")
    if (args.stage in STRICT_FULL_STAGES and args.episodes is not None
            and args.episodes != -1):
        parser.error(
            "{} is a strict full-val stage and requires --episodes -1".format(
                args.stage
            )
        )
    if min(args.max_workers, args.max_per_model, args.max_discrete_workers,
           args.max_continuous_workers) < 1:
        parser.error("worker limits must be positive")
    if args.gpu < 0 or args.launch_stagger < 0 or args.resource_wait_timeout <= 0:
        parser.error("invalid resource option")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.status or args.watch:
        campaign_status(args.method, args.batch_id, watch=args.watch)
        return
    if git("status", "--porcelain", "--untracked-files=no") and not args.dry_run:
        raise UserError("tracked worktree must be clean before launch")
    spec = load_spec()
    if args.method == "all":
        if args.stage != "all":
            raise UserError("method=all requires --stage all")
        for method in METHODS:
            child = argparse.Namespace(**vars(args))
            child.method = method
            execute_method(child, spec)
    else:
        execute_method(args, spec)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker-job":
        raise SystemExit(worker_main(sys.argv[2]))
    try:
        main()
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
