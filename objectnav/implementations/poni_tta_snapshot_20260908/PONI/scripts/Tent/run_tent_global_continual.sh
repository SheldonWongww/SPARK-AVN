#!/usr/bin/env bash
set -Eeuo pipefail
METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PONI_ROOT="${PONI_ROOT:-$(cd "$METHOD_DIR/../.." && pwd)}"
TTA_LR="${TTA_LR:-1e-8}"; TTA_STEPS="${TTA_STEPS:-1}"; TTA_EPISODIC="${TTA_EPISODIC:-false}"
if [[ "$TTA_EPISODIC" != "false" ]]; then
    echo "global-continual Tent requires TTA_EPISODIC=false" >&2
    exit 2
fi
TTA_RESET_BN_STATS="${TTA_RESET_BN_STATS:-false}"; TTA_NORM_SCOPE="${TTA_NORM_SCOPE:-bn}"
TTA_ENTROPY_MODE="${TTA_ENTROPY_MODE:-all}"; TTA_MIN_ADAPTATION_PIXELS="${TTA_MIN_ADAPTATION_PIXELS:-128}"
TTA_OPTIMIZER="${TTA_OPTIMIZER:-Adam}"; TTA_MOMENTUM="${TTA_MOMENTUM:-0.9}"
TTA_BETA1="${TTA_BETA1:-0.9}"; TTA_BETA2="${TTA_BETA2:-0.999}"; TTA_WEIGHT_DECAY="${TTA_WEIGHT_DECAY:-0.0}"
TTA_UPDATE_INTERVAL="${TTA_UPDATE_INTERVAL:-16}"; TTA_MAX_UPDATES_PER_EPISODE="${TTA_MAX_UPDATES_PER_EPISODE:-32}"; TTA_MAX_GRAD_NORM="${TTA_MAX_GRAD_NORM:-1.0}"
METHOD_NAME=Tent; METHOD_SLUG=tent; EVAL_SCRIPT="$METHOD_DIR/eval_poni_tent.py"; DIAGNOSTICS_FLAG=--tta-diagnostics
DEFAULT_EXPERIMENT_ROOT="$PONI_ROOT/experiments/global_continual/Tent"
METHOD_ARGS=(--tta-lr "$TTA_LR" --tta-steps "$TTA_STEPS" --tta-episodic "$TTA_EPISODIC" --tta-reset-bn-stats "$TTA_RESET_BN_STATS" --tta-norm-scope "$TTA_NORM_SCOPE" --tta-norm-prefixes "" --tta-entropy-mode "$TTA_ENTROPY_MODE" --tta-min-adaptation-pixels "$TTA_MIN_ADAPTATION_PIXELS" --tta-optimizer "$TTA_OPTIMIZER" --tta-momentum "$TTA_MOMENTUM" --tta-beta1 "$TTA_BETA1" --tta-beta2 "$TTA_BETA2" --tta-weight-decay "$TTA_WEIGHT_DECAY" --tta-update-interval "$TTA_UPDATE_INTERVAL" --tta-max-updates-per-episode "$TTA_MAX_UPDATES_PER_EPISODE" --tta-max-grad-norm "$TTA_MAX_GRAD_NORM")
TTA_ARGS=(--tta "lr=$TTA_LR" --tta "steps=$TTA_STEPS" --tta "episodic=$TTA_EPISODIC" --tta "reset_bn_stats=$TTA_RESET_BN_STATS" --tta "norm_scope=bn_all" --tta "entropy_mode=$TTA_ENTROPY_MODE" --tta "min_adaptation_pixels=$TTA_MIN_ADAPTATION_PIXELS" --tta "optimizer=$TTA_OPTIMIZER" --tta "momentum=$TTA_MOMENTUM" --tta "beta1=$TTA_BETA1" --tta "beta2=$TTA_BETA2" --tta "weight_decay=$TTA_WEIGHT_DECAY" --tta "update_interval=$TTA_UPDATE_INTERVAL" --tta "max_updates_per_episode=$TTA_MAX_UPDATES_PER_EPISODE" --tta "max_grad_norm=$TTA_MAX_GRAD_NORM")
source "$PONI_ROOT/scripts/global_continual/run_common.sh"
