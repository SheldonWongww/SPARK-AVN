#!/usr/bin/env python3
"""Validate one VLN TTA job config and emit baseline-specific CLI tokens."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


VLN_ROOT = Path(__file__).resolve().parents[1]
_PROVIDER_PATH = VLN_ROOT / "navtta_vln" / "reverie_llm_feedback.py"
_PROVIDER_SPEC = importlib.util.spec_from_file_location(
    "_navtta_reverie_llm_feedback_contract", str(_PROVIDER_PATH)
)
_PROVIDER_MODULE = importlib.util.module_from_spec(_PROVIDER_SPEC)
_PROVIDER_SPEC.loader.exec_module(_PROVIDER_MODULE)
LLM_FEEDBACK_MODEL_ID = _PROVIDER_MODULE.MODEL_ID
LLM_FEEDBACK_MODEL_REVISION = _PROVIDER_MODULE.PINNED_MODEL_REVISION
LLM_FEEDBACK_WEIGHTS_SHA256 = _PROVIDER_MODULE.PINNED_WEIGHTS_SHA256
LLM_FEEDBACK_BUNDLE_SHA256 = _PROVIDER_MODULE.PINNED_BUNDLE_SHA256
PROMPT_BUNDLE_SHA256 = _PROVIDER_MODULE.PROMPT_BUNDLE_SHA256
validate_loopback_url = _PROVIDER_MODULE.validate_loopback_url

DISCRETE_SETTINGS = {
    "duet-r2r", "duet-reverie", "hamt-r2r", "hamt-reverie",
    "goat-r2r", "goat-reverie",
}
CONTINUOUS_SETTINGS = {"etpnav-r2r-ce", "bevbert-r2r-ce"}
METHODS = {"source", "tent", "fstta", "eam", "feedtta", "atena", "idea"}
FEEDTTA_SCOPE_PROFILES = {
    "configured_prefixes", "paper_full", "last_crossmodal", "action_head",
}


COMMON = {
    "lr", "norm_scope", "last_k_ln", "optimizer", "momentum", "beta1",
    "beta2", "weight_decay", "max_grad_norm", "update_interval",
    "max_updates_per_episode", "episodic", "action_selection", "action_seed",
    "matched_feedtta_source", "diagnostics_expected_episodes",
}
METHOD_KEYS = {
    "source": {
        "action_selection", "action_seed", "matched_feedtta_source",
    },
    "tent": set(),
    "fstta": {
        "lr_fast", "lr_slow", "m", "n", "q", "rho", "tau", "a", "b",
        "fast_grad_mode", "use_fast_lr_scaler", "use_slow", "slow_optimizer",
        "slow_momentum", "reset_slow_optimizer_each_window",
        "reset_optimizer_each_episode", "reset_var_hist_each_episode",
        "eigen_eps",
    },
    "eam": {"lr", "confidence_scale", "memory_size", "batch_size",
            "update_interval"},
    "feedtta": {"lr", "p", "alpha", "sgr_seed", "sgr_mode", "gamma",
                "normalize_gradient", "optimizer_eps", "action_seed",
                "scope_profile", "feedback_provider", "llm_feedback_url",
                "llm_feedback_model_id", "llm_feedback_revision",
                "llm_feedback_weights_sha256", "llm_feedback_bundle_sha256",
                "llm_feedback_prompt_sha256", "llm_feedback_token_file",
                "llm_feedback_cache_dir", "llm_feedback_transcript_path",
                "llm_feedback_timeout_seconds", "llm_feedback_abort_on_failure"},
    "atena": {"lr_query", "lr_self", "mix_lambda", "query_threshold",
              "self_loss_weight", "update_scope"},
    "idea": {"lr", "prompt_length", "k_max", "lambda", "tau", "fisher_beta",
             "opt_steps", "use_fisher", "ridge", "prompt_layers",
             "source_stats_path", "source_stats_sha256",
             "source_trajectories", "collect_source_stats",
             "source_stats_output", "source_checkpoint_sha256",
             "source_dataset", "source_dataset_version", "source_split",
             "collection_policy"},
}


def _load(path):
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("TTA config must be a JSON object")
    schema = document.get("schema")
    audit_zero_update = bool(document.get("audit_zero_update", False))
    audit_control = bool(document.get("audit_control", False))
    audit_expected_episodes = document.get("episodes") if (
        audit_zero_update or audit_control
    ) else None
    if (audit_zero_update or audit_control) and schema != (
        "navtta.vln_tta_adapter_parity_job.v1"
    ):
        raise ValueError(
            "zero-update audit requires the adapter-parity job schema"
        )
    if (
        schema == "navtta.vln_tta_adapter_parity_job.v1"
        and document.get("namespace") != "adapter_parity_audit"
    ):
        raise ValueError("adapter-parity job has an invalid namespace")
    if (
        schema != "navtta.vln_tta_adapter_parity_job.v1"
        and document.get("namespace") == "adapter_parity_audit"
    ):
        raise ValueError(
            "ordinary TTA schema cannot claim the adapter-parity namespace"
        )
    order_seed = None
    if schema == "navtta.vln_tta_adapter_parity_job.v1":
        order_seed = document.get("order_seed")
        if type(order_seed) is not int or order_seed != 0:
            raise ValueError(
                "adapter-parity config order_seed must be the exact integer 0"
            )
    elif (schema == "navtta.vln_tta_job.v1"
          and document.get("stage") == "orders"):
        order_seed = document.get("order_seed")
        if type(order_seed) is not int or order_seed not in (0, 1, 2, 3):
            raise ValueError(
                "orders config order_seed must be an exact integer in "
                "[0, 1, 2, 3]"
            )
    elif "order_seed" in document:
        raise ValueError(
            "ordinary non-orders config must omit order_seed entirely"
        )
    method = str(document.get("method", "")).lower()
    if method not in METHODS:
        raise ValueError("invalid or missing TTA method: {!r}".format(method))
    parameters = document.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("TTA parameters must be a JSON object")
    unknown = set(parameters).difference(COMMON | METHOD_KEYS[method])
    if unknown:
        raise ValueError("unknown {} parameters: {}".format(
            method, ", ".join(sorted(unknown))))
    if method == "idea":
        collecting = parameters.get("collect_source_stats", False)
        namespace = document.get("namespace", "tuning")
        if collecting:
            if (
                schema != "navtta.vln_tta_job.v1"
                or namespace != "idea_source_statistics"
                or document.get("stage") != "source_statistics"
            ):
                raise ValueError(
                    "IDEA source collection requires the ordinary job schema, "
                    "namespace=idea_source_statistics, and stage=source_statistics"
                )
            if parameters.get("action_selection", "argmax") != "argmax":
                raise ValueError(
                    "IDEA source collection must follow the Source policy's "
                    "native argmax actions"
                )
        elif namespace == "idea_source_statistics":
            raise ValueError(
                "namespace=idea_source_statistics requires collect_source_stats=true"
            )
        required = (
            ("source_stats_output", "source_checkpoint_sha256",
             "source_dataset", "source_dataset_version", "source_split",
             "collection_policy")
            if collecting else ("source_stats_path", "source_stats_sha256")
        )
        for key in required:
            if not isinstance(parameters.get(key), str) or not parameters[key]:
                raise ValueError("IDEA requires {}".format(key))
        if collecting and any(parameters.get(key) for key in (
            "source_stats_path", "source_stats_sha256"
        )):
            raise ValueError(
                "IDEA source collection cannot also consume a source artifact"
            )
        digest = parameters.get(
            "source_checkpoint_sha256" if collecting else "source_stats_sha256"
        )
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF"
                                    for char in digest):
            raise ValueError("IDEA source/checkpoint digest must be a SHA256")
        if collecting:
            if not parameters["source_stats_output"].endswith(".json"):
                raise ValueError("formal IDEA source collection output must be JSON")
            if parameters["source_split"] != "train":
                raise ValueError("IDEA source collection requires a training split")
            if parameters["collection_policy"] != (
                "frozen_source_argmax_rollout"
            ):
                raise ValueError(
                    "IDEA source collection requires the declared frozen "
                    "Source argmax rollout policy"
                )
        source_trajectories = parameters.get("source_trajectories", 128)
        if type(source_trajectories) is not int or source_trajectories != 128:
            raise ValueError("IDEA source statistics require 128 trajectories")
        opt_steps = parameters.get("opt_steps", 50)
        if type(opt_steps) is not int or opt_steps != 50:
            raise ValueError("canonical IDEA requires opt_steps=50")
    for key in ("action_seed", "sgr_seed"):
        if key in parameters and type(parameters[key]) is not int:
            raise ValueError("{} must be an exact integer".format(key))
    if (schema == "navtta.vln_tta_job.v1"
            and document.get("stage") == "orders" and method == "feedtta"):
        if parameters.get("sgr_seed") != order_seed:
            raise ValueError(
                "FeedTTA orders config sgr_seed must equal order_seed"
            )
        if (
            parameters.get("action_selection", "argmax") == "sample"
            and parameters.get("action_seed") != order_seed
        ):
            raise ValueError(
                "sampled FeedTTA orders config action_seed must equal order_seed"
            )
        if (
            parameters.get("action_selection", "argmax") != "sample"
            and "action_seed" in parameters
        ):
            raise ValueError(
                "native-argmax FeedTTA orders config must omit action_seed"
            )
    for key in (
        "matched_feedtta_source", "reset_var_hist_each_episode",
        "collect_source_stats",
    ):
        if key in parameters and type(parameters[key]) is not bool:
            raise ValueError("{} must be an exact boolean".format(key))
    if (
        "sgr_mode" in parameters
        and parameters["sgr_mode"] not in ("paper_main", "appendix_b1")
    ):
        raise ValueError("invalid FeedTTA sgr_mode")
    feedback_provider = parameters.get("feedback_provider", "task_evaluator")
    if feedback_provider not in ("task_evaluator", "qwen2_vl_2b_v1"):
        raise ValueError("invalid FeedTTA feedback_provider")
    if feedback_provider == "qwen2_vl_2b_v1":
        required = (
            "llm_feedback_url",
            "llm_feedback_model_id",
            "llm_feedback_revision",
            "llm_feedback_weights_sha256",
            "llm_feedback_bundle_sha256",
            "llm_feedback_prompt_sha256",
            "llm_feedback_token_file",
            "llm_feedback_cache_dir",
            "llm_feedback_transcript_path",
        )
        for key in required:
            if not isinstance(parameters.get(key), str) or not parameters[key]:
                raise ValueError("FeedTTA-LLM requires {}".format(key))
        if parameters["llm_feedback_model_id"] != LLM_FEEDBACK_MODEL_ID:
            raise ValueError("FeedTTA-LLM model id is not pinned")
        validate_loopback_url(parameters["llm_feedback_url"])
        revision = parameters["llm_feedback_revision"].lower()
        if len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise ValueError("FeedTTA-LLM revision must be a commit SHA")
        if revision != LLM_FEEDBACK_MODEL_REVISION:
            raise ValueError("FeedTTA-LLM revision is not code-pinned")
        for key in (
            "llm_feedback_weights_sha256", "llm_feedback_bundle_sha256",
            "llm_feedback_prompt_sha256"
        ):
            digest = parameters[key].lower()
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError("{} must be a SHA256".format(key))
        if parameters["llm_feedback_weights_sha256"].lower() != (
            LLM_FEEDBACK_WEIGHTS_SHA256
        ):
            raise ValueError("FeedTTA-LLM weights digest is not code-pinned")
        if parameters["llm_feedback_bundle_sha256"].lower() != (
            LLM_FEEDBACK_BUNDLE_SHA256
        ):
            raise ValueError("FeedTTA-LLM bundle digest is not code-pinned")
        if parameters["llm_feedback_prompt_sha256"].lower() != (
            PROMPT_BUNDLE_SHA256
        ):
            raise ValueError("FeedTTA-LLM prompt bundle is not code-pinned")
        if not os.path.isabs(parameters["llm_feedback_cache_dir"]):
            raise ValueError("FeedTTA-LLM cache directory must be absolute")
        for key in (
            "llm_feedback_token_file", "llm_feedback_transcript_path"
        ):
            if not os.path.isabs(parameters[key]):
                raise ValueError("{} must be absolute".format(key))
        abort_on_failure = parameters.get(
            "llm_feedback_abort_on_failure", True
        )
        if type(abort_on_failure) is not bool:
            raise ValueError(
                "llm_feedback_abort_on_failure must be an exact boolean"
            )
        if not abort_on_failure:
            raise ValueError(
                "formal FeedTTA-LLM requires abort_on_failure=true"
            )
        timeout = parameters.get("llm_feedback_timeout_seconds", 120.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise ValueError("FeedTTA-LLM timeout must be finite and positive")
    elif any(key.startswith("llm_feedback_") for key in parameters):
        raise ValueError(
            "llm_feedback_* parameters require feedback_provider=qwen2_vl_2b_v1"
        )
    if method == "source":
        sampled_source = parameters.get("action_selection") == "sample"
        matched_source = parameters.get("matched_feedtta_source", False)
        if matched_source and not sampled_source:
            raise ValueError(
                "matched FeedTTA Source control requires action_selection=sample"
            )
        if sampled_source:
            # Preserve legacy sampled-Source configs while making their role
            # explicit in the translated command and formal run manifest.
            parameters["matched_feedtta_source"] = True
    elif parameters.get("matched_feedtta_source", False):
        raise ValueError(
            "matched_feedtta_source is valid only for method=source"
        )
    if method == "atena" and parameters.get(
        "update_scope", "replay_reachable_high_level_navigation"
    ) != "replay_reachable_high_level_navigation":
        raise ValueError(
            "ATENA cannot claim an unreplayed full policy"
        )
    if schema == "navtta.vln_tta_job.v1":
        # Legacy v1 manifests contain both reproduction and explicitly named
        # ablation jobs.  Preserve their replayability here; new campaign
        # runners enforce their canonical profiles before materialization.
        if method == "fstta" and parameters.get(
            "reset_var_hist_each_episode", False
        ):
            raise ValueError(
                "formal paper FSTTA must not reset variance history each episode"
            )
    for key, value in parameters.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite parameter {}".format(key))
    expected_diagnostics = parameters.get("diagnostics_expected_episodes")
    if expected_diagnostics is not None and (
        isinstance(expected_diagnostics, bool)
        or not isinstance(expected_diagnostics, int)
        or expected_diagnostics <= 0
    ):
        raise ValueError("diagnostics_expected_episodes must be a positive integer")
    if audit_zero_update and method == "source":
        raise ValueError("Source controls cannot claim adapter audit mode")
    if audit_control and method != "source":
        raise ValueError("adapter audit controls must use method=source")
    if (
        schema == "navtta.vln_tta_adapter_parity_job.v1"
        and audit_control == audit_zero_update
    ):
        raise ValueError(
            "adapter-parity jobs require exactly one of audit_control or "
            "audit_zero_update"
        )
    if (audit_zero_update or audit_control) and (
        isinstance(audit_expected_episodes, bool)
        or not isinstance(audit_expected_episodes, int)
        or audit_expected_episodes <= 0
    ):
        raise ValueError("adapter audit requires a positive episode count")
    return (
        method, parameters, audit_zero_update, audit_control,
        audit_expected_episodes,
    )


def _scalar(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _continuous(
    method, params, diagnostics, audit_zero_update=False, audit_control=False,
    audit_expected_episodes=None,
):
    if (
        method == "feedtta"
        and params.get("scope_profile", "paper_full")
        not in FEEDTTA_SCOPE_PROFILES
    ):
        raise ValueError("invalid continuous FeedTTA scope_profile")
    tokens = ["TTA.METHOD", method,
              "TTA.DIAGNOSTICS_FILE", diagnostics]
    if audit_zero_update:
        tokens.extend(["TTA.AUDIT_ZERO_UPDATE", "True"])
    if audit_control:
        tokens.extend(["TTA.AUDIT_CONTROL", "True"])
    if audit_zero_update or audit_control:
        tokens.extend([
            "TTA.AUDIT_EXPECTED_EPISODES", str(audit_expected_episodes)
        ])
    common_map = {
        "lr": "TTA.LR", "norm_scope": "TTA.NORM_SCOPE",
        "last_k_ln": "TTA.LAST_K_LN", "optimizer": "TTA.OPTIMIZER",
        "momentum": "TTA.MOMENTUM", "beta1": "TTA.BETA1",
        "beta2": "TTA.BETA2", "weight_decay": "TTA.WEIGHT_DECAY",
        "max_grad_norm": "TTA.MAX_GRAD_NORM",
        "update_interval": "TTA.UPDATE_INTERVAL",
        "max_updates_per_episode": "TTA.MAX_UPDATES_PER_EPISODE",
        "episodic": "TTA.EPISODIC",
        "action_selection": "TTA.ACTION_SELECTION",
        "action_seed": "TTA.ACTION_SEED",
        "matched_feedtta_source": "TTA.MATCHED_FEEDTTA_SOURCE",
    }
    prefix = {
        "fstta": "TTA.FSTTA.", "eam": "TTA.EAM.",
        "feedtta": "TTA.FEEDTTA.", "atena": "TTA.ATENA.",
        "idea": "TTA.IDEA.",
    }.get(method)
    method_map = {
        "lr": "LR", "lr_fast": "TTA.LR", "lr_slow": "LR_SLOW", "m": "M", "n": "N",
        "q": "Q", "rho": "RHO", "tau": "TAU", "a": "A", "b": "B",
        "fast_grad_mode": "FAST_GRAD_MODE",
        "use_fast_lr_scaler": "USE_FAST_LR_SCALER", "use_slow": "USE_SLOW",
        "slow_optimizer": "SLOW_OPTIMIZER", "slow_momentum": "SLOW_MOMENTUM",
        "reset_slow_optimizer_each_window": "RESET_SLOW_OPTIMIZER_EACH_WINDOW",
        "reset_optimizer_each_episode": "RESET_OPTIMIZER_EACH_EPISODE",
        "reset_var_hist_each_episode": "RESET_VAR_HIST_EACH_EPISODE",
        "eigen_eps": "EIGEN_EPS", "confidence_scale": "CONFIDENCE_SCALE",
        "memory_size": "MEMORY_SIZE", "batch_size": "BATCH_SIZE",
        "update_interval": "UPDATE_INTERVAL",
        "p": "P", "alpha": "ALPHA", "sgr_seed": "SGR_SEED",
        "sgr_mode": "SGR_MODE",
        "gamma": "GAMMA", "normalize_gradient": "NORMALIZE_GRADIENT",
        "optimizer_eps": "EPS", "lr_query": "LR_QUERY", "lr_self": "LR_SELF",
        "scope_profile": "SCOPE_PROFILE",
        "mix_lambda": "MIX_LAMBDA", "query_threshold": "QUERY_THRESHOLD",
        "self_loss_weight": "SELF_LOSS_WEIGHT",
        "update_scope": "TASK_UPDATE_SCOPE",
        # IDEA (TTA.IDEA.*). ``tau`` reuses the FSTTA-shared "TAU" suffix name.
        "prompt_length": "PROMPT_LENGTH", "k_max": "K_MAX", "lambda": "LAMBDA",
        "fisher_beta": "FISHER_BETA", "opt_steps": "OPT_STEPS",
        "use_fisher": "USE_FISHER", "ridge": "RIDGE",
        "prompt_layers": "PROMPT_LAYERS",
        "source_stats_path": "SOURCE_STATS_PATH",
        "source_stats_sha256": "SOURCE_STATS_SHA256",
        "source_trajectories": "SOURCE_TRAJECTORIES",
        "collect_source_stats": "COLLECT_SOURCE_STATS",
        "source_stats_output": "SOURCE_STATS_OUTPUT",
        "source_checkpoint_sha256": "SOURCE_CHECKPOINT_SHA256",
        "source_dataset": "SOURCE_DATASET",
        "source_dataset_version": "SOURCE_DATASET_VERSION",
        "source_split": "SOURCE_SPLIT",
        "collection_policy": "COLLECTION_POLICY",
    }
    for key, value in params.items():
        method_specific = (
            (method in ("eam", "feedtta", "idea") and key == "lr")
            or (method == "eam" and key == "update_interval")
        )
        if key in common_map and not method_specific:
            config_key = common_map[key]
        else:
            suffix = method_map.get(key)
            if suffix is None:
                continue
            config_key = suffix if suffix.startswith("TTA.") else prefix + suffix
        tokens.extend([config_key, _scalar(value)])
    return tokens


def _discrete(
    method, params, diagnostics, audit_zero_update=False, audit_control=False,
    audit_expected_episodes=None,
):
    tokens = ["--tta_method", method, "--tta_diagnostics", diagnostics]
    if audit_zero_update:
        tokens.append("--tta_audit_zero_update")
    if audit_control:
        tokens.append("--tta_audit_control")
    if audit_zero_update or audit_control:
        tokens.extend([
            "--tta_audit_expected_episodes", str(audit_expected_episodes)
        ])
    common_map = {
        "lr": "--tta_lr", "norm_scope": "--tta_norm_scope",
        "last_k_ln": "--tta_last_k_ln", "optimizer": "--tta_optimizer",
        "momentum": "--tta_momentum", "beta1": "--tta_beta1",
        "beta2": "--tta_beta2", "weight_decay": "--tta_weight_decay",
        "max_grad_norm": "--tta_max_grad_norm",
        "update_interval": "--tta_update_interval",
        "max_updates_per_episode": "--tta_max_updates_per_episode",
        "action_selection": "--tta_action_selection",
        "action_seed": "--tta_action_seed",
        "matched_feedtta_source": "--tta_matched_feedtta_source",
        "diagnostics_expected_episodes": "--tta_diagnostics_expected_episodes",
    }
    method_map = {
        "lr_fast": "--tta_lr", "lr_slow": "--tta_fstta_lr_slow",
        "m": "--tta_fstta_m", "n": "--tta_fstta_n",
        "q": "--tta_fstta_q", "rho": "--tta_fstta_rho",
        "tau": "--tta_idea_tau" if method == "idea" else "--tta_fstta_tau",
        "a": "--tta_fstta_a",
        "b": "--tta_fstta_b", "fast_grad_mode": "--tta_fstta_fast_grad_mode",
        "slow_optimizer": "--tta_fstta_slow_optimizer",
        "slow_momentum": "--tta_fstta_slow_momentum",
        "eigen_eps": "--tta_fstta_eigen_eps", "lr": {
            "eam": "--tta_eam_lr", "feedtta": "--tta_feedtta_lr",
            "idea": "--tta_idea_lr",
        }.get(method, "--tta_lr"),
        "confidence_scale": "--tta_eam_confidence_scale",
        "memory_size": "--tta_eam_memory_size",
        "batch_size": "--tta_eam_batch_size",
        "update_interval": "--tta_eam_update_interval",
        "p": "--tta_feedtta_p", "alpha": "--tta_feedtta_alpha",
        "sgr_seed": "--tta_feedtta_sgr_seed",
        "sgr_mode": "--tta_feedtta_sgr_mode",
        "gamma": "--tta_feedtta_gamma",
        "scope_profile": "--tta_feedtta_scope_profile",
        "feedback_provider": "--tta_feedback_provider",
        "llm_feedback_url": "--tta_llm_feedback_url",
        "llm_feedback_model_id": "--tta_llm_feedback_model_id",
        "llm_feedback_revision": "--tta_llm_feedback_revision",
        "llm_feedback_weights_sha256": "--tta_llm_feedback_weights_sha256",
        "llm_feedback_bundle_sha256": "--tta_llm_feedback_bundle_sha256",
        "llm_feedback_prompt_sha256": "--tta_llm_feedback_prompt_sha256",
        "llm_feedback_token_file": "--tta_llm_feedback_token_file",
        "llm_feedback_cache_dir": "--tta_llm_feedback_cache_dir",
        "llm_feedback_transcript_path": "--tta_llm_feedback_transcript_path",
        "llm_feedback_timeout_seconds": "--tta_llm_feedback_timeout_seconds",
        "optimizer_eps": "--tta_feedtta_eps", "lr_query": "--tta_atena_lr_query",
        "lr_self": "--tta_atena_lr_self", "mix_lambda": "--tta_atena_mix_lambda",
        "query_threshold": "--tta_atena_query_threshold",
        "self_loss_weight": "--tta_atena_self_loss_weight",
        "update_scope": "--tta_atena_update_scope",
        # IDEA discrete CLI (--tta_idea_*). ``lr`` and ``tau`` are handled by
        # the method-conditional entries above.
        "prompt_length": "--tta_idea_prompt_length",
        "k_max": "--tta_idea_k_max", "lambda": "--tta_idea_lambda",
        "fisher_beta": "--tta_idea_fisher_beta",
        "opt_steps": "--tta_idea_opt_steps", "ridge": "--tta_idea_ridge",
        "prompt_layers": "--tta_idea_prompt_layers",
        "source_stats_path": "--tta_idea_source_stats_path",
        "source_stats_sha256": "--tta_idea_source_stats_sha256",
        "source_trajectories": "--tta_idea_source_trajectories",
        "source_stats_output": "--tta_idea_source_stats_output",
        "source_checkpoint_sha256": "--tta_idea_source_checkpoint_sha256",
        "source_dataset": "--tta_idea_source_dataset",
        "source_dataset_version": "--tta_idea_source_dataset_version",
        "source_split": "--tta_idea_source_split",
        "collection_policy": "--tta_idea_collection_policy",
    }
    true_flags = {
        "episodic": "--tta_episodic",
        "normalize_gradient": "--tta_feedtta_normalize_gradient",
        "collect_source_stats": "--tta_idea_collect_source_stats",
        "reset_slow_optimizer_each_window":
            "--tta_fstta_reset_slow_optimizer_each_window",
        "reset_var_hist_each_episode":
            "--tta_fstta_reset_var_hist_each_episode",
    }
    inverse_flags = {
        "use_fast_lr_scaler": "--tta_fstta_no_fast_lr_scaler",
        "use_slow": "--tta_fstta_no_slow",
        "reset_optimizer_each_episode":
            "--tta_fstta_no_reset_optimizer_each_episode",
        "use_fisher": "--tta_idea_no_fisher",
    }
    for key, value in params.items():
        if key == "matched_feedtta_source":
            if bool(value):
                tokens.append("--tta_matched_feedtta_source")
            continue
        if key in true_flags:
            if bool(value):
                tokens.append(true_flags[key])
            continue
        if key in inverse_flags:
            if not bool(value):
                tokens.append(inverse_flags[key])
            continue
        method_specific = (
            (method in ("eam", "feedtta", "idea") and key == "lr")
            or (method == "eam" and key == "update_interval")
        )
        option = (
            method_map.get(key)
            if method_specific or key not in common_map
            else common_map.get(key)
        )
        if option is not None:
            tokens.extend([option, _scalar(value)])
    return tokens


def _validate_setting(setting, method, params):
    feedback_provider = params.get("feedback_provider", "task_evaluator")
    if feedback_provider == "qwen2_vl_2b_v1" and setting != "goat-reverie":
        raise ValueError(
            "qwen2_vl_2b_v1 is restricted to the goat-reverie setting"
        )
    if (
        setting in CONTINUOUS_SETTINGS
        and "diagnostics_expected_episodes" in params
    ):
        raise ValueError(
            "diagnostics_expected_episodes is supported only by discrete VLN"
        )
    if setting not in DISCRETE_SETTINGS | CONTINUOUS_SETTINGS:
        raise ValueError("TTA search does not support setting {!r}".format(setting))


def translate(setting, config_path, diagnostics):
    (
        method, params, audit_zero_update, audit_control,
        audit_expected_episodes,
    ) = _load(config_path)
    _validate_setting(setting, method, params)
    feedback_provider = params.get("feedback_provider", "task_evaluator")
    if feedback_provider == "qwen2_vl_2b_v1":
        expected_transcript = os.path.join(
            os.path.dirname(os.path.abspath(diagnostics)),
            "llm_feedback_transcript.ndjson",
        )
        if os.path.abspath(params["llm_feedback_transcript_path"]) != (
            expected_transcript
        ):
            raise ValueError(
                "FeedTTA-LLM transcript must be the current job evidence file {}"
                .format(expected_transcript)
            )
    if setting in DISCRETE_SETTINGS:
        return method, _discrete(
            method, params, diagnostics,
            audit_zero_update=audit_zero_update,
            audit_control=audit_control,
            audit_expected_episodes=audit_expected_episodes,
        )
    if setting in CONTINUOUS_SETTINGS:
        return method, _continuous(
            method, params, diagnostics,
            audit_zero_update=audit_zero_update,
            audit_control=audit_control,
            audit_expected_episodes=audit_expected_episodes,
        )
    raise AssertionError("validated setting has no translator")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--print-method", action="store_true")
    parser.add_argument("--print-namespace", action="store_true")
    parser.add_argument("--nul", action="store_true")
    args = parser.parse_args()
    if args.print_method or args.print_namespace:
        method, params, _, _, _ = _load(os.path.abspath(args.config))
        _validate_setting(args.setting, method, params)
    else:
        method, tokens = translate(
            args.setting, os.path.abspath(args.config),
            os.path.abspath(args.diagnostics),
        )
    if args.print_method:
        print(method)
        return
    if args.print_namespace:
        with open(args.config, "r", encoding="utf-8") as stream:
            document = json.load(stream)
        print(document.get("namespace", "tuning"))
        return
    if args.nul:
        sys.stdout.buffer.write(b"\0".join(token.encode("utf-8") for token in tokens))
        if tokens:
            sys.stdout.buffer.write(b"\0")
    else:
        json.dump(tokens, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
