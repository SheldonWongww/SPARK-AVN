#!/usr/bin/env bash
set -euo pipefail

SEED=1
EXP_CONFIG="cfgs/cl_exp/single_source/spark_avn/train.yaml"
MODEL_DIR=""
GPUS="0"
OVERWRITE=0
EXTRA_OPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            SEED="$2"; shift 2 ;;
        --exp-config)
            EXP_CONFIG="$2"; shift 2 ;;
        --model-dir)
            MODEL_DIR="$2"; shift 2 ;;
        --gpus)
            GPUS="$2"; shift 2 ;;
        --overwrite)
            OVERWRITE=1; shift ;;
        --)
            shift
            EXTRA_OPTS=("$@")
            break ;;
        -h|--help)
            echo "Usage: $0 [--seed N] [--exp-config PATH] [--model-dir DIR] [--gpus IDS] [--overwrite] [-- EXTRA_OPTS...]"
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

if [[ ! -f "${EXP_CONFIG}" ]]; then
    echo "Config file not found: ${EXP_CONFIG}"
    exit 1
fi

if [[ -z "${MODEL_DIR}" ]]; then
    if [[ "${EXP_CONFIG}" == *"multi_source"* ]]; then
        SOURCE_TYPE="multi_source"
    else
        SOURCE_TYPE="single_source"
    fi

    if [[ "${EXP_CONFIG}" == *"finetune"* ]]; then
        METHOD="finetune"
    elif [[ "${EXP_CONFIG}" == *"spark_avn"* ]]; then
        METHOD="spark_avn"
    else
        METHOD="continual"
    fi

    MODEL_DIR="data/results/${METHOD}/${SOURCE_TYPE}/seed_${SEED}"
fi

mkdir -p "${MODEL_DIR}"

RUN_ARGS=(
    --run-type train
    --exp-config "${EXP_CONFIG}"
    --model-dir "${MODEL_DIR}"
    --seed "${SEED}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
    RUN_ARGS+=(--overwrite)
fi

CUDA_VISIBLE_DEVICES="${GPUS}" python run.py \
    "${RUN_ARGS[@]}" \
    "${EXTRA_OPTS[@]}"
