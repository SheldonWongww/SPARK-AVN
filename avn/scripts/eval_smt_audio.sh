#!/usr/bin/env bash
set -euo pipefail

SOURCE_SETTING="${1:?usage: $0 <single_source|multi_source> <source|tent|fstta|eam|feedtta|atena|idea> <seed> [CONFIG OVERRIDES ...]}"
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
    tent|fstta|eam|feedtta|atena|idea) CONFIG_METHOD="${METHOD}" ;;
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
EPISODE_COUNT=2000
IDEA_SOURCE_MANIFEST=""

overrides=("$@")
if (( ${#overrides[@]} % 2 != 0 )); then
    printf 'config overrides must be KEY VALUE pairs\n' >&2
    exit 2
fi
for ((index = 0; index + 1 < ${#overrides[@]}; index += 2)); do
    key="${overrides[$index]}"
    value="${overrides[$((index + 1))]}"
    case "${key}" in
        TASK_CONFIG.DATASET.*|TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.*|\
        BASE_TASK_CONFIG_PATH)
            printf 'unsupported stream-defining override: %s\n' "${key}" >&2
            exit 2
            ;;
    esac
    if [[ "${key}" == "EVAL.USE_CKPT_CONFIG" && "${value}" != "False" ]]; then
        printf 'EVAL.USE_CKPT_CONFIG must remain False\n' >&2
        exit 2
    fi
    if [[ "${key}" == "NUM_PROCESSES" && "${value}" != "1" ]]; then
        printf 'manifested SMT+Audio TTA requires NUM_PROCESSES=1\n' >&2
        exit 2
    fi
    if [[ "${key}" == "EVAL.SPLIT" && "${value}" != "val" ]]; then
        printf 'SMT+Audio TTA evaluation is pinned to EVAL.SPLIT=val\n' >&2
        exit 2
    fi
    if [[ "${key}" == "TEST_EPISODE_COUNT" ]]; then
        EPISODE_COUNT="${value}"
    fi
    if [[ "${key}" == "TTA.IDEA.SOURCE_EPISODE_MANIFEST" ]]; then
        IDEA_SOURCE_MANIFEST="${value}"
    fi
done
[[ "${EPISODE_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'invalid TEST_EPISODE_COUNT: %s\n' "${EPISODE_COUNT}" >&2
    exit 2
}
[[ ${EPISODE_COUNT} -le 2000 ]] || {
    printf 'TEST_EPISODE_COUNT exceeds the configured 2000-episode stream\n' >&2
    exit 2
}

test -f "${CHECKPOINT}" || { printf 'missing checkpoint: %s\n' "${CHECKPOINT}" >&2; exit 1; }
test -f "${DATASET}" || { printf 'missing dataset: %s\n' "${DATASET}" >&2; exit 1; }
fingerprints="$(python3 "${REPO_ROOT}/avn/scripts/fingerprint_episode_stream.py" \
    --dataset "${DATASET}" --seed "${SEED}" --episode-count "${EPISODE_COUNT}")"
read -r CURRENT_STREAM_ORDER_SHA256 CURRENT_STREAM_CONTENT_SHA256 <<< "${fingerprints}"
if [[ -n "${STREAM_ORDER_SHA256}" && \
      "${STREAM_ORDER_SHA256}" != "${CURRENT_STREAM_ORDER_SHA256}" ]]; then
    printf 'launcher and runtime stream-order SHA256 differ\n' >&2
    exit 1
fi
if [[ -n "${STREAM_CONTENT_SHA256}" && \
      "${STREAM_CONTENT_SHA256}" != "${CURRENT_STREAM_CONTENT_SHA256}" ]]; then
    printf 'launcher and runtime stream-content SHA256 differ\n' >&2
    exit 1
fi
STREAM_ORDER_SHA256="${CURRENT_STREAM_ORDER_SHA256}"
STREAM_CONTENT_SHA256="${CURRENT_STREAM_CONTENT_SHA256}"
[[ "${STREAM_ORDER_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { printf 'invalid stream-order SHA256\n' >&2; exit 1; }
[[ "${STREAM_CONTENT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { printf 'invalid stream-content SHA256\n' >&2; exit 1; }
mkdir -p "${REPO_ROOT}/avn/results/runs"
mkdir "${RUN_DIR}" || { printf 'run directory already exists: %s\n' "${RUN_DIR}" >&2; exit 1; }
mkdir -p "${RUN_DIR}/raw/model"

manifest_options=()
if [[ "${METHOD}" == "idea" ]]; then
    test -f "${IDEA_SOURCE_MANIFEST}" || {
        printf 'missing IDEA source episode manifest: %s\n' "${IDEA_SOURCE_MANIFEST}" >&2
        exit 1
    }
    manifest_options+=(--asset-manifest "${IDEA_SOURCE_MANIFEST}")
fi

python3 "${REPO_ROOT}/tools/create_run_manifest.py" \
    --output "${RUN_DIR}/manifest.json" \
    --run-id "${RUN_ID}" --task avn --benchmark mp3d \
    --model smt_audio --method "${METHOD}" --run-tag "${RUN_TAG}" \
    --source-setting "${SOURCE_SETTING}" \
    --seed "${SEED}" --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
    --dataset "${DATASET}" --dataset-version v1 \
    --stream-order-sha256 "${STREAM_ORDER_SHA256}" \
    --stream-content-sha256 "${STREAM_CONTENT_SHA256}" \
    "${manifest_options[@]}" \
    --extra "$@"

cd "${BASELINE_ROOT}"
export PYTHONPATH="${REPO_ROOT}/avn:${REPO_ROOT}/core:${BASELINE_ROOT}:${PYTHONPATH:-}"
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
artifacts=()
STATS="${RUN_DIR}/raw/model/tb/val_stats_${SEED}.json"
DIAGNOSTICS="${RUN_DIR}/raw/model/tb/tta_diagnostics_${SEED}.json"
[[ -f "${STATS}" ]] && artifacts+=(--artifact "episode_metrics=${STATS}")
[[ -f "${DIAGNOSTICS}" ]] && artifacts+=(--artifact "tta_diagnostics=${DIAGNOSTICS}")
python3 "${REPO_ROOT}/tools/finalize_run_manifest.py" \
    --manifest "${RUN_DIR}/manifest.json" --exit-code "${status}" \
    "${artifacts[@]}"
exit "${status}"
