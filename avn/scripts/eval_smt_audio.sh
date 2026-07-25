#!/usr/bin/env bash
set -euo pipefail

SOURCE_SETTING="${1:?usage: $0 <single_source|multi_source> <source|tent|fstta|eam|feedtta|atena> <seed> [CONFIG OVERRIDES ...]}"
METHOD="${2:?missing method}"
SEED="${3:?missing seed}"
shift 3

case "${SOURCE_SETTING}" in
    single_source) CHECKPOINT_NAME="single_best_val.pth" ;;
    multi_source) CHECKPOINT_NAME="multi_best_val.pth" ;;
    *) printf 'invalid source setting: %s\n' "${SOURCE_SETTING}" >&2; exit 2 ;;
esac

case "${METHOD}" in
    source) CONFIG_METHOD="none" ;;
    tent|fstta|eam|feedtta|atena) CONFIG_METHOD="${METHOD}" ;;
    *) printf 'invalid method: %s\n' "${METHOD}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINE_ROOT="${REPO_ROOT}/avn/baselines/smt_audio"
CONFIG="ss_baselines/savi/config/tta_avn/${SOURCE_SETTING}/smt_audio_tta_test.yaml"
CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/smt_audio/${CHECKPOINT_NAME}"
DATASET="${REPO_ROOT}/avn/data/datasets/tta_test/${SOURCE_SETTING}/mp3d/v1/val/val.json.gz"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'uncommitted')"
RUN_TAG="${NAVTTA_RUN_TAG:-}"
if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'invalid NAVTTA_RUN_TAG: %s\n' "${RUN_TAG}" >&2
    exit 2
fi
RUN_TAG_SUFFIX="${RUN_TAG:+-${RUN_TAG}}"
RUN_ID="avn-mp3d-smt_audio-${METHOD}-${SOURCE_SETTING}-seed${SEED}${RUN_TAG_SUFFIX}-$(date -u +%Y%m%dT%H%M%SZ)-${COMMIT}"
RUN_DIR="${REPO_ROOT}/avn/results/runs/${RUN_ID}"
STREAM_ORDER_SHA256="${NAVTTA_STREAM_ORDER_SHA256:-}"
STREAM_CONTENT_SHA256="${NAVTTA_STREAM_CONTENT_SHA256:-}"

test -f "${CHECKPOINT}" || { printf 'missing checkpoint: %s\n' "${CHECKPOINT}" >&2; exit 1; }
test -f "${DATASET}" || { printf 'missing dataset: %s\n' "${DATASET}" >&2; exit 1; }
if [[ -z "${STREAM_ORDER_SHA256}" || -z "${STREAM_CONTENT_SHA256}" ]]; then
    fingerprints="$(python3 "${REPO_ROOT}/avn/scripts/fingerprint_episode_stream.py" \
        --dataset "${DATASET}" --seed "${SEED}")"
    read -r STREAM_ORDER_SHA256 STREAM_CONTENT_SHA256 <<< "${fingerprints}"
fi
[[ "${STREAM_ORDER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { printf 'invalid stream-order SHA256\n' >&2; exit 1; }
[[ "${STREAM_CONTENT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { printf 'invalid stream-content SHA256\n' >&2; exit 1; }
mkdir -p "${REPO_ROOT}/avn/results/runs"
mkdir "${RUN_DIR}" || { printf 'run directory already exists: %s\n' "${RUN_DIR}" >&2; exit 1; }
mkdir -p "${RUN_DIR}/raw/model"

python3 "${REPO_ROOT}/tools/create_run_manifest.py" \
    --output "${RUN_DIR}/manifest.json" \
    --run-id "${RUN_ID}" --task avn --benchmark mp3d \
    --model smt_audio --method "${METHOD}" --run-tag "${RUN_TAG}" \
    --source-setting "${SOURCE_SETTING}" \
    --seed "${SEED}" --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" --dataset-version v1 \
    --stream-order-sha256 "${STREAM_ORDER_SHA256}" \
    --stream-content-sha256 "${STREAM_CONTENT_SHA256}" \
    --extra "$@"

cd "${BASELINE_ROOT}"
export PYTHONPATH="${REPO_ROOT}/core:${BASELINE_ROOT}:${PYTHONPATH:-}"
set +e
python3 ss_baselines/savi/run.py \
    --run-type eval \
    --exp-config "${CONFIG}" \
    --model-dir "${RUN_DIR}/raw/model" \
    "$@" \
    EVAL_CKPT_PATH_DIR "${CHECKPOINT}" \
    SEED "${SEED}" \
    TASK_CONFIG.SEED "${SEED}" \
    TTA.METHOD "${CONFIG_METHOD}"
status=$?
set -e
python3 "${REPO_ROOT}/tools/finalize_run_manifest.py" \
    --manifest "${RUN_DIR}/manifest.json" --exit-code "${status}"
exit "${status}"
