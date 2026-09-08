#!/usr/bin/env bash
set -Eeuo pipefail
METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PONI_ROOT="${PONI_ROOT:-$(cd "$METHOD_DIR/../.." && pwd)}"
ATENA_LR_QUERY="${ATENA_LR_QUERY:-1e-8}"; ATENA_LR_SELF="${ATENA_LR_SELF:-1e-9}"; ATENA_MIX_LAMBDA="${ATENA_MIX_LAMBDA:-0.5}"; ATENA_QUERY_THRESHOLD="${ATENA_QUERY_THRESHOLD:-0.01}"; ATENA_SELF_LOSS_WEIGHT="${ATENA_SELF_LOSS_WEIGHT:-0.1}"
ATENA_PARAM_SCOPE="${ATENA_PARAM_SCOPE:-bn}"; ATENA_OPTIMIZER="${ATENA_OPTIMIZER:-AdamW}"; ATENA_MOMENTUM="${ATENA_MOMENTUM:-0.9}"; ATENA_BETA1="${ATENA_BETA1:-0.9}"; ATENA_BETA2="${ATENA_BETA2:-0.999}"; ATENA_WEIGHT_DECAY="${ATENA_WEIGHT_DECAY:-0.01}"; ATENA_MAX_GRAD_NORM="${ATENA_MAX_GRAD_NORM:-1.0}"
METHOD_NAME=ATENA; METHOD_SLUG=atena; EVAL_SCRIPT="$METHOD_DIR/eval_poni_atena.py"; DIAGNOSTICS_FLAG=--atena-diagnostics
DEFAULT_EXPERIMENT_ROOT="$PONI_ROOT/experiments/global_continual/ATENA"
METHOD_ARGS=(--atena-lr-query "$ATENA_LR_QUERY" --atena-lr-self "$ATENA_LR_SELF" --atena-mix-lambda "$ATENA_MIX_LAMBDA" --atena-query-threshold "$ATENA_QUERY_THRESHOLD" --atena-self-loss-weight "$ATENA_SELF_LOSS_WEIGHT" --atena-param-scope "$ATENA_PARAM_SCOPE" --atena-optimizer "$ATENA_OPTIMIZER" --atena-momentum "$ATENA_MOMENTUM" --atena-beta1 "$ATENA_BETA1" --atena-beta2 "$ATENA_BETA2" --atena-weight-decay "$ATENA_WEIGHT_DECAY" --atena-max-grad-norm "$ATENA_MAX_GRAD_NORM")
TTA_ARGS=(--tta "lr_query=$ATENA_LR_QUERY" --tta "lr_self=$ATENA_LR_SELF" --tta "mix_lambda=$ATENA_MIX_LAMBDA" --tta "query_threshold=$ATENA_QUERY_THRESHOLD" --tta "self_loss_weight=$ATENA_SELF_LOSS_WEIGHT" --tta "param_scope=$ATENA_PARAM_SCOPE" --tta "optimizer=$ATENA_OPTIMIZER" --tta "momentum=$ATENA_MOMENTUM" --tta "beta1=$ATENA_BETA1" --tta "beta2=$ATENA_BETA2" --tta "weight_decay=$ATENA_WEIGHT_DECAY" --tta "max_grad_norm=$ATENA_MAX_GRAD_NORM")
source "$PONI_ROOT/scripts/global_continual/run_common.sh"
