#!/usr/bin/env bash
set -euo pipefail

SOURCE_SETTING="${1:?usage: $0 <single_source|multi_source> <source|tent|fstta|eam|feedtta|atena> <seed> [CONFIG OVERRIDES ...]}"
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
    tent|fstta|eam|feedtta|atena) CONFIG_METHOD="${METHOD}" ;;
    *) printf 'invalid method: %s\n' "${METHOD}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINE_ROOT="${REPO_ROOT}/avn/baselines/enmus"
SOUNDSPACES_ROOT="${REPO_ROOT}/avn/baselines/smt_audio"
CONFIG="sen_baselines/enmus/config/${SOURCE_SETTING}/enmus_tta_test.yaml"
CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/enmus/${CHECKPOINT_NAME}"
COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'uncommitted')"
RUN_TAG="${NAVTTA_RUN_TAG:-}"
if [[ -n "${RUN_TAG}" && ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'invalid NAVTTA_RUN_TAG: %s\n' "${RUN_TAG}" >&2
    exit 2
fi
RUN_TAG_SUFFIX="${RUN_TAG:+-${RUN_TAG}}"
RUN_ID="avn-mp3d-enmus-${METHOD}-${SOURCE_SETTING}-seed${SEED}${RUN_TAG_SUFFIX}-$(date -u +%Y%m%dT%H%M%SZ)-${COMMIT}"
RUN_DIR="${REPO_ROOT}/avn/results/runs/${RUN_ID}"

test -f "${CHECKPOINT}" || { printf 'missing checkpoint: %s\n' "${CHECKPOINT}" >&2; exit 1; }
mkdir -p "${RUN_DIR}/raw/model"

python3 "${REPO_ROOT}/tools/create_run_manifest.py" \
    --output "${RUN_DIR}/manifest.json" \
    --run-id "${RUN_ID}" --task avn --benchmark mp3d \
    --model enmus --method "${METHOD}" --run-tag "${RUN_TAG}" \
    --source-setting "${SOURCE_SETTING}" \
    --seed "${SEED}" --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --extra "$@"

cd "${BASELINE_ROOT}"
export PYTHONPATH="${REPO_ROOT}/core:${BASELINE_ROOT}:${SOUNDSPACES_ROOT}:${PYTHONPATH:-}"
python3 sen_baselines/enmus/run.py \
    --run-type eval \
    --decoder-type MSMT \
    --exp-config "${CONFIG}" \
    --model-dir "${RUN_DIR}/raw/model" \
    EVAL_CKPT_PATH_DIR "${CHECKPOINT}" \
    TTA.METHOD "${CONFIG_METHOD}" \
    SEED "${SEED}" \
    TASK_CONFIG.SEED "${SEED}" \
    "$@"
