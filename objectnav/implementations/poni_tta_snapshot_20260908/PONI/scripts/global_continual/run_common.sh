#!/usr/bin/env bash

# Method wrapper defines METHOD_NAME, METHOD_SLUG, EVAL_SCRIPT,
# DIAGNOSTICS_FLAG, DEFAULT_EXPERIMENT_ROOT, METHOD_ARGS[], TTA_ARGS[].
set -Eeuo pipefail
: "${METHOD_NAME:?}" "${METHOD_SLUG:?}" "${EVAL_SCRIPT:?}"
: "${DIAGNOSTICS_FLAG:?}" "${DEFAULT_EXPERIMENT_ROOT:?}"

PONI_ROOT="${PONI_ROOT:-$(cd "$METHOD_DIR/../.." && pwd)}"
NAVTTA_ROOT="${NAVTTA_ROOT:-$(cd "$PONI_ROOT/../NavTTA" && pwd)}"
PYTHON="${PYTHON:-/home/king/miniconda3/envs/poni-mp3d/bin/python}"
GPU_ID="${1:-0}"
MODEL_PATH="${MODEL_PATH:-$PONI_ROOT/pretrained_models/mp3d_models/poni_seed_123.ckpt}"
EXPERIMENT_ROOT="${SAVE_ROOT:-$DEFAULT_EXPERIMENT_ROOT}"
if [[ "$EXPERIMENT_ROOT" != /* ]]; then EXPERIMENT_ROOT="$PONI_ROOT/$EXPERIMENT_ROOT"; fi
mkdir -p "$EXPERIMENT_ROOT"
EXPERIMENT_ROOT="$(cd "$EXPERIMENT_ROOT" && pwd)"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')_pid$$}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid RUN_ID" >&2; exit 1
fi
RUN_ROOT="$EXPERIMENT_ROOT/runs/$RUN_ID"
if [[ -e "$RUN_ROOT/run_manifest.json" ]]; then
    echo "Global-continual resume is unsupported; choose a new RUN_ID" >&2
    exit 2
fi
mkdir -p "$RUN_ROOT"

DATASET_ROOT="${GLOBAL_TTA_DATASET_ROOT:-$PONI_ROOT/experiments/global_continual_dataset}"
SPLIT="${GLOBAL_TTA_SPLIT:-val_tta_all}"
mkdir -p "$DATASET_ROOT"
# Full runs build the canonical 2195-episode stream here. Hyperparameter
# searches provide a prebuilt, shorter scene-spanning stream and explicitly
# skip this builder so it cannot be overwritten by the full dataset.
if [[ "${GLOBAL_TTA_SKIP_DATASET_BUILD:-false}" != "true" ]]; then
    if [[ "$SPLIT" != "val_tta_all" ]]; then
        echo "Noncanonical split requires GLOBAL_TTA_SKIP_DATASET_BUILD=true" >&2
        exit 2
    fi
    # All method launchers share this generated split. Serialize generation so
    # parallel startup cannot race on the gzip index or atomic manifest temp.
    (
        flock -x 9
        "$PYTHON" "$PONI_ROOT/scripts/global_continual/build_val_tta_all.py" \
            --output-root "$DATASET_ROOT" --split "$SPLIT" >/dev/null
    ) 9>"$DATASET_ROOT/.build.lock"
fi
STREAM_MANIFEST="$DATASET_ROOT/$SPLIT/STREAM_MANIFEST.json"
DATA_PATH="$DATASET_ROOT/{split}/{split}.json.gz"
if [[ ! -f "$STREAM_MANIFEST" || ! -f "$DATASET_ROOT/$SPLIT/$SPLIT.json.gz" ]]; then
    echo "Missing prebuilt global-continual dataset for split $SPLIT" >&2
    exit 2
fi
TOTAL_EPISODES="$($PYTHON - "$STREAM_MANIFEST" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["total_episodes"]))
PY
)"

export PONI_ROOT NAVTTA_ROOT
export PYTHONPATH="$PONI_ROOT:$PONI_ROOT/scripts:$PONI_ROOT/dependencies/astar_pycpp:$NAVTTA_ROOT/core${PYTHONPATH:+:$PYTHONPATH}"
export MAGNUM_LOG=quiet GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mkdir -p "$RUN_ROOT/tb" "$MPLCONFIGDIR"

MANIFEST="$RUN_ROOT/run_manifest.json"
"$PYTHON" "$PONI_ROOT/scripts/global_continual/run_manifest.py" start \
    --path "$MANIFEST" --method "$METHOD_SLUG" --run-id "$RUN_ID" \
    --launcher-pid "$$" --poni-root "$PONI_ROOT" \
    --navtta-root "$NAVTTA_ROOT" --save-root "$RUN_ROOT" \
    --model-path "$MODEL_PATH" --gpu-id "$GPU_ID" \
    --stream-manifest "$STREAM_MANIFEST" "${TTA_ARGS[@]}"

finalized=0
finalize() {
    if (( finalized == 0 )); then
        "$PYTHON" "$PONI_ROOT/scripts/global_continual/run_manifest.py" finish \
            --path "$MANIFEST" --status "$1" || true
        finalized=1
    fi
}
on_exit() {
    local code=$?
    if (( finalized == 0 )); then (( code == 0 )) && finalize complete || finalize failed; fi
}
on_signal() { finalize interrupted; exit 130; }
trap on_exit EXIT
trap on_signal INT TERM

echo "PONI + $METHOD_NAME global continual: $TOTAL_EPISODES episodes, GPU=$GPU_ID"
echo "Run: $RUN_ROOT"
echo "Protocol: one process, one environment, 11 ordered scenes, no reset"
echo "Monitor: $PYTHON $PONI_ROOT/scripts/global_continual/monitor.py --save-root $EXPERIMENT_ROOT"

cd "$PONI_ROOT/hlab"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$EVAL_SCRIPT" \
    --run-type eval \
    --exp-config "$PONI_ROOT/hlab/transfer_configs/transfer_objectnav_mp3d.yaml" \
    "${METHOD_ARGS[@]}" "$DIAGNOSTICS_FLAG" "$RUN_ROOT/${METHOD_SLUG}_diagnostics.json" \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH" \
    TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE False \
    TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE True \
    TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES -1 \
    TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS -1 \
    EVAL.SPLIT "$SPLIT" TEST_EPISODE_COUNT -1 TASK_CONFIG.SEED 100 \
    NUM_ENVIRONMENTS 1 TENSORBOARD_DIR "$RUN_ROOT/tb" \
    LOG_FILE "$RUN_ROOT/run.log" GLOBAL_AGENT.name PFExp \
    GLOBAL_AGENT.smart_local_boundaries True GLOBAL_AGENT.num_local_steps 1 \
    PF_EXP_POLICY.pf_model_path "$MODEL_PATH" \
    PF_EXP_POLICY.pf_masking_opt unexplored \
    >"$RUN_ROOT/console.log" 2>&1

"$PYTHON" "$PONI_ROOT/hlab/merge_results.py" \
    --path_format "$RUN_ROOT/tb/stats.json" | tee "$RUN_ROOT/merged_results.txt"
finalize complete
