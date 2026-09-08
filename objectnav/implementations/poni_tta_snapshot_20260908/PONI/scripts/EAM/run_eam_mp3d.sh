#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PONI_ROOT="${PONI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

EAM_LR="${EAM_LR:-1e-5}"
EAM_CONFIDENCE_SCALE="${EAM_CONFIDENCE_SCALE:-0.4}"
EAM_AUX_WEIGHT="${EAM_AUX_WEIGHT:-0.5}"
EAM_MEMORY_SIZE="${EAM_MEMORY_SIZE:-32}"
EAM_BATCH_SIZE="${EAM_BATCH_SIZE:-8}"
EAM_UPDATE_INTERVAL="${EAM_UPDATE_INTERVAL:-1}"
EAM_INFERENCE_INTERVAL="${EAM_INFERENCE_INTERVAL:-$EAM_UPDATE_INTERVAL}"
EAM_MAX_RELIABLE_PIXELS="${EAM_MAX_RELIABLE_PIXELS:--1}"
EAM_DEPLOYMENT_SCOPE="${EAM_DEPLOYMENT_SCOPE:-selected}"
EAM_PIXEL_SELECTION="${EAM_PIXEL_SELECTION:-low_entropy}"
EAM_TRAINABLE_PREFIXES="${EAM_TRAINABLE_PREFIXES:-deconv4,agant0,final_conv,final_deconv_custom}"
EAM_OPTIMIZER="${EAM_OPTIMIZER:-Adam}"
EAM_MOMENTUM="${EAM_MOMENTUM:-0.9}"
EAM_BETA1="${EAM_BETA1:-0.9}"
EAM_BETA2="${EAM_BETA2:-0.999}"
EAM_WEIGHT_DECAY="${EAM_WEIGHT_DECAY:-0.0}"
EAM_MAX_GRAD_NORM="${EAM_MAX_GRAD_NORM:-0.0}"

METHOD_NAME=EAM METHOD_SLUG=eam
EVAL_SCRIPT="$SCRIPT_DIR/eval_poni_eam.py"
MONITOR_SCRIPT="$SCRIPT_DIR/monitor_eam.py"
DIAGNOSTICS_FLAG=--eam-diagnostics
DEFAULT_EXPERIMENT_ROOT="$PONI_ROOT/experiments/EAM/mp3d_poni_seed_123"
METHOD_ARGS=(--eam-lr "$EAM_LR" --eam-confidence-scale "$EAM_CONFIDENCE_SCALE" --eam-aux-weight "$EAM_AUX_WEIGHT" --eam-memory-size "$EAM_MEMORY_SIZE" --eam-batch-size "$EAM_BATCH_SIZE" --eam-update-interval "$EAM_UPDATE_INTERVAL" --eam-inference-interval "$EAM_INFERENCE_INTERVAL" --eam-max-reliable-pixels "$EAM_MAX_RELIABLE_PIXELS" --eam-deployment-scope "$EAM_DEPLOYMENT_SCOPE" --eam-pixel-selection "$EAM_PIXEL_SELECTION" --eam-trainable-prefixes "$EAM_TRAINABLE_PREFIXES" --eam-optimizer "$EAM_OPTIMIZER" --eam-momentum "$EAM_MOMENTUM" --eam-beta1 "$EAM_BETA1" --eam-beta2 "$EAM_BETA2" --eam-weight-decay "$EAM_WEIGHT_DECAY" --eam-max-grad-norm "$EAM_MAX_GRAD_NORM")
MANIFEST_TTA_ARGS=(--tta "lr=$EAM_LR" --tta "confidence_scale=$EAM_CONFIDENCE_SCALE" --tta "aux_weight=$EAM_AUX_WEIGHT" --tta "memory_size=$EAM_MEMORY_SIZE" --tta "batch_size=$EAM_BATCH_SIZE" --tta "update_interval=$EAM_UPDATE_INTERVAL" --tta "inference_interval=$EAM_INFERENCE_INTERVAL" --tta "max_reliable_pixels=$EAM_MAX_RELIABLE_PIXELS" --tta "deployment_scope=$EAM_DEPLOYMENT_SCOPE" --tta "pixel_selection=$EAM_PIXEL_SELECTION" --tta "trainable_prefixes=$EAM_TRAINABLE_PREFIXES" --tta "optimizer=$EAM_OPTIMIZER" --tta "momentum=$EAM_MOMENTUM" --tta "beta1=$EAM_BETA1" --tta "beta2=$EAM_BETA2" --tta "weight_decay=$EAM_WEIGHT_DECAY" --tta "max_grad_norm=$EAM_MAX_GRAD_NORM")
source "$PONI_ROOT/scripts/tta_mp3d_common.sh"
