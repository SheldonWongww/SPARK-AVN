#!/usr/bin/env python3
"""Validate one VLN TTA job config and emit baseline-specific CLI tokens."""

import argparse
import json
import math
import os
import sys


DISCRETE_SETTINGS = {
    "duet-r2r", "duet-reverie", "hamt-r2r", "hamt-reverie",
    "goat-r2r", "goat-reverie",
}
CONTINUOUS_SETTINGS = {"etpnav-r2r-ce", "bevbert-r2r-ce"}
METHODS = {"source", "tent", "fstta", "eam", "feedtta", "atena"}


COMMON = {
    "lr", "norm_scope", "last_k_ln", "optimizer", "momentum", "beta1",
    "beta2", "weight_decay", "max_grad_norm", "update_interval",
    "max_updates_per_episode", "episodic", "action_selection", "action_seed",
}
METHOD_KEYS = {
    "source": {"action_selection", "action_seed"},
    "tent": set(),
    "fstta": {
        "lr_fast", "lr_slow", "m", "n", "q", "rho", "tau", "a", "b",
        "fast_grad_mode", "use_fast_lr_scaler", "use_slow", "slow_optimizer",
        "slow_momentum", "reset_slow_optimizer_each_window",
        "reset_optimizer_each_episode", "eigen_eps",
    },
    "eam": {"lr", "confidence_scale", "memory_size", "batch_size",
            "update_interval"},
    "feedtta": {"lr", "p", "alpha", "sgr_seed", "gamma",
                "normalize_gradient", "optimizer_eps", "action_seed"},
    "atena": {"lr_query", "lr_self", "mix_lambda", "query_threshold",
              "self_loss_weight"},
}


def _load(path):
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("TTA config must be a JSON object")
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
    for key, value in parameters.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite parameter {}".format(key))
    return method, parameters


def _scalar(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _continuous(method, params, diagnostics):
    tokens = ["TTA.METHOD", method,
              "TTA.DIAGNOSTICS_FILE", diagnostics]
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
    }
    prefix = {
        "fstta": "TTA.FSTTA.", "eam": "TTA.EAM.",
        "feedtta": "TTA.FEEDTTA.", "atena": "TTA.ATENA.",
    }.get(method)
    method_map = {
        "lr": "LR", "lr_fast": "TTA.LR", "lr_slow": "LR_SLOW", "m": "M", "n": "N",
        "q": "Q", "rho": "RHO", "tau": "TAU", "a": "A", "b": "B",
        "fast_grad_mode": "FAST_GRAD_MODE",
        "use_fast_lr_scaler": "USE_FAST_LR_SCALER", "use_slow": "USE_SLOW",
        "slow_optimizer": "SLOW_OPTIMIZER", "slow_momentum": "SLOW_MOMENTUM",
        "reset_slow_optimizer_each_window": "RESET_SLOW_OPTIMIZER_EACH_WINDOW",
        "reset_optimizer_each_episode": "RESET_OPTIMIZER_EACH_EPISODE",
        "eigen_eps": "EIGEN_EPS", "confidence_scale": "CONFIDENCE_SCALE",
        "memory_size": "MEMORY_SIZE", "batch_size": "BATCH_SIZE",
        "update_interval": "UPDATE_INTERVAL",
        "p": "P", "alpha": "ALPHA", "sgr_seed": "SGR_SEED",
        "gamma": "GAMMA", "normalize_gradient": "NORMALIZE_GRADIENT",
        "optimizer_eps": "EPS", "lr_query": "LR_QUERY", "lr_self": "LR_SELF",
        "mix_lambda": "MIX_LAMBDA", "query_threshold": "QUERY_THRESHOLD",
        "self_loss_weight": "SELF_LOSS_WEIGHT",
    }
    for key, value in params.items():
        method_specific = (
            (method in ("eam", "feedtta") and key == "lr")
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


def _discrete(method, params, diagnostics):
    tokens = ["--tta_method", method, "--tta_diagnostics", diagnostics]
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
    }
    method_map = {
        "lr_fast": "--tta_lr", "lr_slow": "--tta_fstta_lr_slow",
        "m": "--tta_fstta_m", "n": "--tta_fstta_n",
        "q": "--tta_fstta_q", "rho": "--tta_fstta_rho",
        "tau": "--tta_fstta_tau", "a": "--tta_fstta_a",
        "b": "--tta_fstta_b", "fast_grad_mode": "--tta_fstta_fast_grad_mode",
        "slow_optimizer": "--tta_fstta_slow_optimizer",
        "slow_momentum": "--tta_fstta_slow_momentum",
        "eigen_eps": "--tta_fstta_eigen_eps", "lr": {
            "eam": "--tta_eam_lr", "feedtta": "--tta_feedtta_lr"
        }.get(method, "--tta_lr"),
        "confidence_scale": "--tta_eam_confidence_scale",
        "memory_size": "--tta_eam_memory_size",
        "batch_size": "--tta_eam_batch_size",
        "update_interval": "--tta_eam_update_interval",
        "p": "--tta_feedtta_p", "alpha": "--tta_feedtta_alpha",
        "sgr_seed": "--tta_feedtta_sgr_seed", "gamma": "--tta_feedtta_gamma",
        "optimizer_eps": "--tta_feedtta_eps", "lr_query": "--tta_atena_lr_query",
        "lr_self": "--tta_atena_lr_self", "mix_lambda": "--tta_atena_mix_lambda",
        "query_threshold": "--tta_atena_query_threshold",
        "self_loss_weight": "--tta_atena_self_loss_weight",
    }
    true_flags = {
        "episodic": "--tta_episodic",
        "normalize_gradient": "--tta_feedtta_normalize_gradient",
        "reset_slow_optimizer_each_window":
            "--tta_fstta_reset_slow_optimizer_each_window",
    }
    inverse_flags = {
        "use_fast_lr_scaler": "--tta_fstta_no_fast_lr_scaler",
        "use_slow": "--tta_fstta_no_slow",
        "reset_optimizer_each_episode":
            "--tta_fstta_no_reset_optimizer_each_episode",
    }
    for key, value in params.items():
        if key in true_flags:
            if bool(value):
                tokens.append(true_flags[key])
            continue
        if key in inverse_flags:
            if not bool(value):
                tokens.append(inverse_flags[key])
            continue
        method_specific = (
            (method in ("eam", "feedtta") and key == "lr")
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


def translate(setting, config_path, diagnostics):
    method, params = _load(config_path)
    if setting in DISCRETE_SETTINGS:
        return method, _discrete(method, params, diagnostics)
    if setting in CONTINUOUS_SETTINGS:
        return method, _continuous(method, params, diagnostics)
    raise ValueError("TTA search does not support setting {!r}".format(setting))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--print-method", action="store_true")
    parser.add_argument("--nul", action="store_true")
    args = parser.parse_args()
    method, tokens = translate(
        args.setting, os.path.abspath(args.config), os.path.abspath(args.diagnostics)
    )
    if args.print_method:
        print(method)
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
