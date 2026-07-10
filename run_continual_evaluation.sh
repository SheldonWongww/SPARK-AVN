#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=""
EXP_CONFIG="cfgs/cl_exp/single_source/spark_avn/val.yaml"
GPU="0"
SEED=""
EVAL_MODE="final"
EXTRA_OPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"; shift 2 ;;
        --exp-config)
            EXP_CONFIG="$2"; shift 2 ;;
        --gpu)
            GPU="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --eval-mode)
            EVAL_MODE="$2"; shift 2 ;;
        --)
            shift
            EXTRA_OPTS=("$@")
            break ;;
        -h|--help)
            echo "Usage: $0 --model-dir DIR [--exp-config PATH] [--gpu ID] [--seed N] [--eval-mode final|matrix] [-- EXTRA_OPTS...]"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

if [[ -z "${MODEL_DIR}" ]]; then
    echo "--model-dir is required"
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Model directory not found: ${MODEL_DIR}"
    exit 1
fi

if [[ ! -f "${EXP_CONFIG}" ]]; then
    echo "Config file not found: ${EXP_CONFIG}"
    exit 1
fi

CMD=(
    python run.py
    --run-type eval-continual
    --exp-config "${EXP_CONFIG}"
    --model-dir "${MODEL_DIR}"
    --continual-eval-mode "${EVAL_MODE}"
)

if [[ -n "${SEED}" ]]; then
    CMD+=(--seed "${SEED}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" "${EXTRA_OPTS[@]}"
