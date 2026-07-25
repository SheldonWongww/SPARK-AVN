#!/usr/bin/env bash
set -euo pipefail

SOURCE_SETTING="${1:?usage: $0 <single_source|multi_source> source <seed> [CONFIG OVERRIDES ...]}"
METHOD="${2:?missing method; SAVi currently supports source evaluation here}"
SEED="${3:?missing seed}"
shift 3

case "${SOURCE_SETTING}" in
    single_source) CHECKPOINT_NAME="single_best_val.pth" ;;
    multi_source) CHECKPOINT_NAME="multi_best_val.pth" ;;
    *) printf 'invalid source setting: %s\n' "${SOURCE_SETTING}" >&2; exit 2 ;;
esac

if [[ "${METHOD}" != "source" ]]; then
    printf 'eval_savi.sh supports source only; got: %s\n' "${METHOD}" >&2
    exit 2
fi

if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    printf 'seed must be a nonnegative integer: %s\n' "${SEED}" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINE_ROOT="${REPO_ROOT}/avn/baselines/smt_audio"
CONFIG="ss_baselines/savi/config/tta_avn/${SOURCE_SETTING}/savi_test.yaml"
CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/savi/${CHECKPOINT_NAME}"
DATASET="${REPO_ROOT}/avn/data/datasets/tta_test/${SOURCE_SETTING}/mp3d/v1/val/val.json.gz"
DESCRIPTOR="${BASELINE_ROOT}/data/pretrained_weights/semantic_audionav/savi/best_val.pth"
LABEL_PREDICTOR="${BASELINE_ROOT}/data/pretrained_weights/semantic_audionav/savi/label_predictor.pth"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'uncommitted')"
RUN_TAG="${NAVTTA_RUN_TAG:-}"

if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'invalid NAVTTA_RUN_TAG: %s\n' "${RUN_TAG}" >&2
    exit 2
fi

RUN_TAG_SUFFIX="${RUN_TAG:+-${RUN_TAG}}"
RUN_ID="avn-mp3d-savi-source-${SOURCE_SETTING}-seed${SEED}${RUN_TAG_SUFFIX}-$(date -u +%Y%m%dT%H%M%SZ)-${COMMIT}"
RUN_DIR="${REPO_ROOT}/avn/results/runs/${RUN_ID}"
STREAM_ORDER_SHA256="${NAVTTA_STREAM_ORDER_SHA256:-}"
STREAM_CONTENT_SHA256="${NAVTTA_STREAM_CONTENT_SHA256:-}"

for required_file in \
    "${BASELINE_ROOT}/${CONFIG}" "${CHECKPOINT}" "${DATASET}" \
    "${DESCRIPTOR}" "${LABEL_PREDICTOR}"; do
    test -f "${required_file}" || {
        printf 'missing SAVi evaluation dependency: %s\n' "${required_file}" >&2
        exit 1
    }
done
if [[ -z "${STREAM_ORDER_SHA256}" || -z "${STREAM_CONTENT_SHA256}" ]]; then
    fingerprints="$(python3 "${REPO_ROOT}/avn/scripts/fingerprint_episode_stream.py" \
        --dataset "${DATASET}" --seed "${SEED}")"
    read -r STREAM_ORDER_SHA256 STREAM_CONTENT_SHA256 <<< "${fingerprints}"
fi
[[ "${STREAM_ORDER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { printf 'invalid stream-order SHA256\n' >&2; exit 1; }
[[ "${STREAM_CONTENT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { printf 'invalid stream-content SHA256\n' >&2; exit 1; }
mkdir -p "${REPO_ROOT}/avn/results/runs"
mkdir "${RUN_DIR}" || {
    printf 'run directory already exists: %s\n' "${RUN_DIR}" >&2
    exit 1
}
mkdir -p "${RUN_DIR}/raw/model"

python3 "${REPO_ROOT}/tools/create_run_manifest.py" \
    --output "${RUN_DIR}/manifest.json" \
    --run-id "${RUN_ID}" --task avn --benchmark mp3d \
    --model savi --method source --run-tag "${RUN_TAG}" \
    --source-setting "${SOURCE_SETTING}" \
    --seed "${SEED}" --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
    --aux-checkpoint "descriptor=${DESCRIPTOR}" \
    --aux-checkpoint "label_predictor=${LABEL_PREDICTOR}" \
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
    EVAL.SPLIT val \
    EVAL.USE_CKPT_CONFIG False \
    EVAL.ACTION_SELECTION sample \
    NUM_PROCESSES 1 \
    TTA.METHOD none \
    SEED "${SEED}" \
    TASK_CONFIG.SEED "${SEED}"
status=$?
set -e
python3 "${REPO_ROOT}/tools/finalize_run_manifest.py" \
    --manifest "${RUN_DIR}/manifest.json" --exit-code "${status}"
exit "${status}"
