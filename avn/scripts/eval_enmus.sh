#!/usr/bin/env bash
set -euo pipefail

SOURCE_SETTING="${1:?usage: $0 <single_source|multi_source> <source|tent|fstta|eam|feedtta|atena|idea> <seed> [CONFIG OVERRIDES ...]}"
METHOD="${2:?missing method}"
SEED="${3:?missing seed}"
shift 3

case "${SOURCE_SETTING}" in
    single_source) CHECKPOINT_NAME="single_source_best_val.pth" ;;
    multi_source) CHECKPOINT_NAME="multi_source_best_val.pth" ;;
    *) printf 'invalid source setting: %s\n' "${SOURCE_SETTING}" >&2; exit 2 ;;
esac

case "${METHOD}" in
    source) CONFIG_METHOD="none" ;;
    tent|fstta|eam|feedtta|atena|idea) CONFIG_METHOD="${METHOD}" ;;
    *) printf 'invalid method: %s\n' "${METHOD}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINE_ROOT="${REPO_ROOT}/avn/baselines/enmus"
SOUNDSPACES_ROOT="${REPO_ROOT}/avn/baselines/smt_audio"
CONFIG="sen_baselines/enmus/config/${SOURCE_SETTING}/enmus_tta_test.yaml"
CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/enmus/${CHECKPOINT_NAME}"
EVAL_SPLIT="${NAVTTA_EVAL_SPLIT:-val}"
case "${EVAL_SPLIT}" in
    train|val) ;;
    *) printf 'invalid NAVTTA_EVAL_SPLIT: %s\n' "${EVAL_SPLIT}" >&2; exit 2 ;;
esac
DATASET="${REPO_ROOT}/avn/data/datasets/tta_test/${SOURCE_SETTING}/mp3d/v1/${EVAL_SPLIT}/${EVAL_SPLIT}.json.gz"
AUDIO_ENCODER="${BASELINE_ROOT}/data/pretrained_weights/semantic_audionav/enmus/audio_encoder_best_val.pth"
VISUAL_ENCODER="${BASELINE_ROOT}/data/pretrained_weights/semantic_audionav/enmus/visual_encoder_best_val.pth"
SELD_ENCODER="${BASELINE_ROOT}/data/pretrained_weights/semantic_audionav/enmus/seld_crnn_best_val.h5"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'uncommitted')"
RUN_TAG="${NAVTTA_RUN_TAG:-}"
if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'invalid NAVTTA_RUN_TAG: %s\n' "${RUN_TAG}" >&2
    exit 2
fi
RUN_TAG_SUFFIX="${RUN_TAG:+-${RUN_TAG}}"
RUN_ID="avn-mp3d-enmus-${METHOD}-${SOURCE_SETTING}-seed${SEED}${RUN_TAG_SUFFIX}-$(date -u +%Y%m%dT%H%M%SZ)-${COMMIT}"
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
split_override_found=0
for ((index = 0; index + 1 < ${#overrides[@]}; index += 2)); do
    case "${overrides[$index]}" in
        TASK_CONFIG.DATASET.*|TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.*|\
        BASE_TASK_CONFIG_PATH)
            printf 'unsupported stream-defining override: %s\n' \
                "${overrides[$index]}" >&2
            exit 2
            ;;
    esac
    if [[ "${overrides[$index]}" == "EVAL.USE_CKPT_CONFIG" && \
          "${overrides[$((index + 1))]}" != "False" ]]; then
        printf 'EVAL.USE_CKPT_CONFIG must remain False\n' >&2
        exit 2
    fi
    if [[ "${overrides[$index]}" == "NUM_PROCESSES" && \
          "${overrides[$((index + 1))]}" != "1" ]]; then
        printf 'manifested ENMuS evaluation requires NUM_PROCESSES=1\n' >&2
        exit 2
    fi
    if [[ "${overrides[$index]}" == "EVAL.SPLIT" ]]; then
        [[ "${overrides[$((index + 1))]}" == "${EVAL_SPLIT}" ]] || {
            printf 'EVAL.SPLIT override does not match the manifested dataset split\n' >&2
            exit 2
        }
        split_override_found=1
    fi
    if [[ "${overrides[$index]}" == "TEST_EPISODE_COUNT" ]]; then
        EPISODE_COUNT="${overrides[$((index + 1))]}"
    fi
    if [[ "${overrides[$index]}" == "TTA.IDEA.SOURCE_EPISODE_MANIFEST" ]]; then
        IDEA_SOURCE_MANIFEST="${overrides[$((index + 1))]}"
    fi
done
if [[ -n "${NAVTTA_EVAL_SPLIT:-}" ]]; then
    [[ ${split_override_found} -eq 1 ]] || {
        printf 'NAVTTA_EVAL_SPLIT requires a matching EVAL.SPLIT override\n' >&2
        exit 2
    }
fi
[[ "${EPISODE_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'invalid TEST_EPISODE_COUNT: %s\n' "${EPISODE_COUNT}" >&2
    exit 2
}
[[ ${EPISODE_COUNT} -le 2000 ]] || {
    printf 'TEST_EPISODE_COUNT exceeds the configured 2000-episode stream\n' >&2
    exit 2
}

for required_file in \
    "${CHECKPOINT}" "${DATASET}" "${AUDIO_ENCODER}" \
    "${VISUAL_ENCODER}" "${SELD_ENCODER}"; do
    test -f "${required_file}" || { printf 'missing ENMuS dependency: %s\n' "${required_file}" >&2; exit 1; }
done
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
    --model enmus --method "${METHOD}" --run-tag "${RUN_TAG}" \
    --source-setting "${SOURCE_SETTING}" \
    --seed "${SEED}" --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
    --aux-checkpoint "audio_encoder=${AUDIO_ENCODER}" \
    --aux-checkpoint "visual_encoder=${VISUAL_ENCODER}" \
    --aux-checkpoint "seld_encoder=${SELD_ENCODER}" \
    --dataset "${DATASET}" --dataset-version v1 \
    --stream-order-sha256 "${STREAM_ORDER_SHA256}" \
    --stream-content-sha256 "${STREAM_CONTENT_SHA256}" \
    "${manifest_options[@]}" \
    --extra "$@"

cd "${BASELINE_ROOT}"
export PYTHONPATH="${REPO_ROOT}/avn:${REPO_ROOT}/core:${BASELINE_ROOT}:${SOUNDSPACES_ROOT}:${PYTHONPATH:-}"
set +e
python3 sen_baselines/enmus/run.py \
    --run-type eval \
    --decoder-type MSMT \
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
STATS="${RUN_DIR}/raw/model/tb/${EVAL_SPLIT}_stats_${SEED}.json"
DIAGNOSTICS="${RUN_DIR}/raw/model/tb/tta_diagnostics_${SEED}.json"
PROTOCOL="${RUN_DIR}/raw/model/tb/eval_protocol_${SEED}.json"
AUDIO_SCHEDULE="${RUN_DIR}/raw/model/tb/audio_schedule_${SEED}.json"
[[ -f "${STATS}" ]] && artifacts+=(--artifact "episode_metrics=${STATS}")
[[ -f "${DIAGNOSTICS}" ]] && artifacts+=(--artifact "tta_diagnostics=${DIAGNOSTICS}")
[[ -f "${PROTOCOL}" ]] && artifacts+=(--artifact "eval_protocol=${PROTOCOL}")
[[ -f "${AUDIO_SCHEDULE}" ]] && artifacts+=(--artifact "audio_schedule=${AUDIO_SCHEDULE}")
python3 "${REPO_ROOT}/tools/finalize_run_manifest.py" \
    --manifest "${RUN_DIR}/manifest.json" --exit-code "${status}" \
    "${artifacts[@]}"
exit "${status}"
