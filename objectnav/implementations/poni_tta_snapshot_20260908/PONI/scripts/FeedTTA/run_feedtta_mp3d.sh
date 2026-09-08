#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PONI_ROOT="${PONI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

FEEDTTA_LR="${FEEDTTA_LR:-5e-6}"
FEEDTTA_P="${FEEDTTA_P:-0.05}"
FEEDTTA_ALPHA="${FEEDTTA_ALPHA:--0.2}"
FEEDTTA_SGR_SEED="${FEEDTTA_SGR_SEED:-0}"
FEEDTTA_GAMMA="${FEEDTTA_GAMMA:-0.99}"
FEEDTTA_NORMALIZE_GRADIENT="${FEEDTTA_NORMALIZE_GRADIENT:-false}"
FEEDTTA_TRAINABLE_PREFIXES="${FEEDTTA_TRAINABLE_PREFIXES:-deconv4,agant0,final_conv,final_deconv_custom}"
FEEDTTA_OPTIMIZER="${FEEDTTA_OPTIMIZER:-Adam}"
FEEDTTA_MOMENTUM="${FEEDTTA_MOMENTUM:-0.9}"
FEEDTTA_BETA1="${FEEDTTA_BETA1:-0.9}"
FEEDTTA_BETA2="${FEEDTTA_BETA2:-0.999}"
FEEDTTA_WEIGHT_DECAY="${FEEDTTA_WEIGHT_DECAY:-0.0}"
FEEDTTA_EPS="${FEEDTTA_EPS:-1e-5}"
FEEDTTA_MAX_GRAD_NORM="${FEEDTTA_MAX_GRAD_NORM:-0.0}"

METHOD_NAME=FeedTTA METHOD_SLUG=feedtta
EVAL_SCRIPT="$SCRIPT_DIR/eval_poni_feedtta.py"
MONITOR_SCRIPT="$SCRIPT_DIR/monitor_feedtta.py"
DIAGNOSTICS_FLAG=--feedtta-diagnostics
DEFAULT_EXPERIMENT_ROOT="$PONI_ROOT/experiments/FeedTTA/mp3d_poni_seed_123"
METHOD_ARGS=(--feedtta-lr "$FEEDTTA_LR" --feedtta-p "$FEEDTTA_P" --feedtta-alpha "$FEEDTTA_ALPHA" --feedtta-sgr-seed "$FEEDTTA_SGR_SEED" --feedtta-gamma "$FEEDTTA_GAMMA" --feedtta-normalize-gradient "$FEEDTTA_NORMALIZE_GRADIENT" --feedtta-trainable-prefixes "$FEEDTTA_TRAINABLE_PREFIXES" --feedtta-optimizer "$FEEDTTA_OPTIMIZER" --feedtta-momentum "$FEEDTTA_MOMENTUM" --feedtta-beta1 "$FEEDTTA_BETA1" --feedtta-beta2 "$FEEDTTA_BETA2" --feedtta-weight-decay "$FEEDTTA_WEIGHT_DECAY" --feedtta-eps "$FEEDTTA_EPS" --feedtta-max-grad-norm "$FEEDTTA_MAX_GRAD_NORM")
MANIFEST_TTA_ARGS=(--tta "lr=$FEEDTTA_LR" --tta "p=$FEEDTTA_P" --tta "alpha=$FEEDTTA_ALPHA" --tta "sgr_seed=$FEEDTTA_SGR_SEED" --tta "gamma=$FEEDTTA_GAMMA" --tta "normalize_gradient=$FEEDTTA_NORMALIZE_GRADIENT" --tta "trainable_prefixes=$FEEDTTA_TRAINABLE_PREFIXES" --tta "optimizer=$FEEDTTA_OPTIMIZER" --tta "momentum=$FEEDTTA_MOMENTUM" --tta "beta1=$FEEDTTA_BETA1" --tta "beta2=$FEEDTTA_BETA2" --tta "weight_decay=$FEEDTTA_WEIGHT_DECAY" --tta "eps=$FEEDTTA_EPS" --tta "max_grad_norm=$FEEDTTA_MAX_GRAD_NORM")
source "$PONI_ROOT/scripts/tta_mp3d_common.sh"

