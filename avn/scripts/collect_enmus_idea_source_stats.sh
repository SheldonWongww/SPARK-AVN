#!/usr/bin/env bash
set -euo pipefail

SOURCE_SETTING="${1:?usage: $0 <single_source|multi_source> [seed]}"
SEED="${2:-0}"
case "${SOURCE_SETTING}" in
  single_source) CHECKPOINT_NAME="single_source_best_val.pth" ;;
  multi_source) CHECKPOINT_NAME="multi_source_best_val.pth" ;;
  *) printf 'invalid source setting: %s\n' "${SOURCE_SETTING}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASELINE_ROOT="${REPO_ROOT}/avn/baselines/enmus"
SOUNDSPACES_ROOT="${REPO_ROOT}/avn/baselines/smt_audio"
DATASET="${REPO_ROOT}/avn/data/datasets/train/${SOURCE_SETTING}/mp3d/v1/train/train.json.gz"
CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/enmus/${CHECKPOINT_NAME}"
AUDIO_ENCODER="${BASELINE_ROOT}/data/pretrained_weights/semantic_audionav/enmus/audio_encoder_best_val.pth"
VISUAL_ENCODER="${BASELINE_ROOT}/data/pretrained_weights/semantic_audionav/enmus/visual_encoder_best_val.pth"
SELD_ENCODER="${BASELINE_ROOT}/data/pretrained_weights/semantic_audionav/enmus/seld_crnn_best_val.h5"
MANIFEST="${REPO_ROOT}/avn/manifests/idea_source/enmus_${SOURCE_SETTING}_sample_seed${SEED}.json"
OUTPUT="${REPO_ROOT}/avn/results/idea_source_statistics/enmus_${SOURCE_SETTING}_sample_seed${SEED}.json"
CONFIG="sen_baselines/enmus/config/${SOURCE_SETTING}/enmus_tta_test.yaml,sen_baselines/enmus/config/${SOURCE_SETTING}/enmus_idea_source_stats.yaml"

for required in "${DATASET}" "${CHECKPOINT}" "${AUDIO_ENCODER}" "${VISUAL_ENCODER}" "${SELD_ENCODER}"; do
  test -f "${required}" || { printf 'missing ENMuS source dependency: %s\n' "${required}" >&2; exit 1; }
done
mkdir -p "$(dirname "${MANIFEST}")" "$(dirname "${OUTPUT}")"
if [[ ! -f "${MANIFEST}" ]]; then
  python3 "${REPO_ROOT}/avn/scripts/build_idea_source_manifest.py" \
    --model enmus --source-setting "${SOURCE_SETTING}" \
    --dataset "${DATASET}" --checkpoint "${CHECKPOINT}" \
    --seed "${SEED}" --output "${MANIFEST}"
fi
MANIFEST_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${MANIFEST}")"

cd "${BASELINE_ROOT}"
export PYTHONPATH="${REPO_ROOT}/avn:${REPO_ROOT}/core:${BASELINE_ROOT}:${SOUNDSPACES_ROOT}:${PYTHONPATH:-}"
python3 sen_baselines/enmus/run.py --run-type eval --decoder-type MSMT \
  --exp-config "${CONFIG}" \
  --model-dir "${REPO_ROOT}/avn/results/idea_source_statistics/enmus_${SOURCE_SETTING}_sample_runtime" \
  EVAL_CKPT_PATH_DIR "${CHECKPOINT}" SEED "${SEED}" TASK_CONFIG.SEED "${SEED}" \
  EVAL.ACTION_SELECTION sample \
  TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE True \
  TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE False \
  TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE False \
  TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE -1 \
  TTA.IDEA.SOURCE_COLLECTION_OUTPUT "${OUTPUT}" \
  TTA.IDEA.SOURCE_EPISODE_MANIFEST "${MANIFEST}" \
  TTA.IDEA.SOURCE_EPISODE_MANIFEST_SHA256 "${MANIFEST_SHA}" \
  TASK_CONFIG.DATASET.IDEA_SOURCE_EPISODE_MANIFEST "${MANIFEST}" \
  TASK_CONFIG.DATASET.IDEA_SOURCE_EPISODE_MANIFEST_SHA256 "${MANIFEST_SHA}"
