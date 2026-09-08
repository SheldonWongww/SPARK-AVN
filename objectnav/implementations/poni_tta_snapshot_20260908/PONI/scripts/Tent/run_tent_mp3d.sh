#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PONI_ROOT="${PONI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
NAVTTA_ROOT="${NAVTTA_ROOT:-$(cd "$PONI_ROOT/../NavTTA" && pwd)}"
PYTHON="${PYTHON:-/home/king/miniconda3/envs/poni-mp3d/bin/python}"
GPU_ID="${1:-0}"
MAX_JOBS="${MAX_JOBS:-1}"
PARTS="${PARTS:-0 1 2 3 4 5 6 7 8 9 10}"
TEST_EPISODE_COUNT="${TEST_EPISODE_COUNT:--1}"
MODEL_PATH="${MODEL_PATH:-$PONI_ROOT/pretrained_models/mp3d_models/poni_seed_123.ckpt}"
EXPERIMENT_ROOT="${SAVE_ROOT:-$PONI_ROOT/experiments/Tent/mp3d_poni_seed_123}"
if [[ "$EXPERIMENT_ROOT" != /* ]]; then
    EXPERIMENT_ROOT="$PONI_ROOT/$EXPERIMENT_ROOT"
fi
mkdir -p "$EXPERIMENT_ROOT"
EXPERIMENT_ROOT="$(cd "$EXPERIMENT_ROOT" && pwd)"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')_pid$$}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_ID may contain only letters, numbers, dot, underscore, and dash" >&2
    exit 1
fi
SAVE_ROOT="$EXPERIMENT_ROOT/runs/$RUN_ID"
DATA_PATH="../data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz"

# NavTTA Tent defaults.  All values remain environment-overridable for a later
# hyperparameter search without changing the experiment code.
TTA_LR="${TTA_LR:-1e-6}"
TTA_STEPS="${TTA_STEPS:-1}"
TTA_EPISODIC="${TTA_EPISODIC:-false}"
TTA_RESET_BN_STATS="${TTA_RESET_BN_STATS:-true}"
TTA_NORM_SCOPE="${TTA_NORM_SCOPE:-bn}"
TTA_NORM_PREFIXES="${TTA_NORM_PREFIXES:-}"
TTA_ENTROPY_MODE="${TTA_ENTROPY_MODE:-all}"
TTA_MIN_ADAPTATION_PIXELS="${TTA_MIN_ADAPTATION_PIXELS:-128}"
TTA_OPTIMIZER="${TTA_OPTIMIZER:-Adam}"
TTA_MOMENTUM="${TTA_MOMENTUM:-0.9}"
TTA_BETA1="${TTA_BETA1:-0.9}"
TTA_BETA2="${TTA_BETA2:-0.999}"
TTA_WEIGHT_DECAY="${TTA_WEIGHT_DECAY:-0.0}"
TTA_UPDATE_INTERVAL="${TTA_UPDATE_INTERVAL:-1}"
TTA_MAX_UPDATES_PER_EPISODE="${TTA_MAX_UPDATES_PER_EPISODE:--1}"
TTA_MAX_GRAD_NORM="${TTA_MAX_GRAD_NORM:-1.0}"

export PONI_ROOT NAVTTA_ROOT
export PYTHONPATH="$PONI_ROOT:$PONI_ROOT/dependencies/astar_pycpp:$NAVTTA_ROOT/core${PYTHONPATH:+:$PYTHONPATH}"
export MAGNUM_LOG=quiet
export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SAVE_ROOT/matplotlib}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mkdir -p "$SAVE_ROOT" "$MPLCONFIGDIR"

for required in \
    "$PYTHON" \
    "$NAVTTA_ROOT/core/navtta_core/tta/tta_core.py" \
    "$PONI_ROOT/pretrained_models/rednet_mp3d.pth" \
    "$MODEL_PATH" \
    "$PONI_ROOT/dependencies/astar_pycpp/astar.so"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

RUN_MANIFEST="$SAVE_ROOT/run_manifest.json"
"$PYTHON" "$SCRIPT_DIR/run_manifest.py" start \
    --path "$RUN_MANIFEST" \
    --save-root "$SAVE_ROOT" \
    --poni-root "$PONI_ROOT" \
    --navtta-root "$NAVTTA_ROOT" \
    --model-path "$MODEL_PATH" \
    --run-id "$RUN_ID" \
    --launcher-pid "$$" \
    --gpu-id "$GPU_ID" \
    --max-jobs "$MAX_JOBS" \
    --parts "$PARTS" \
    --test-episode-count "$TEST_EPISODE_COUNT" \
    --tta "lr=$TTA_LR" \
    --tta "steps=$TTA_STEPS" \
    --tta "episodic=$TTA_EPISODIC" \
    --tta "reset_bn_stats=$TTA_RESET_BN_STATS" \
    --tta "norm_scope=$TTA_NORM_SCOPE" \
    --tta "norm_prefixes=$TTA_NORM_PREFIXES" \
    --tta "entropy_mode=$TTA_ENTROPY_MODE" \
    --tta "min_adaptation_pixels=$TTA_MIN_ADAPTATION_PIXELS" \
    --tta "optimizer=$TTA_OPTIMIZER" \
    --tta "momentum=$TTA_MOMENTUM" \
    --tta "beta1=$TTA_BETA1" \
    --tta "beta2=$TTA_BETA2" \
    --tta "weight_decay=$TTA_WEIGHT_DECAY" \
    --tta "update_interval=$TTA_UPDATE_INTERVAL" \
    --tta "max_updates_per_episode=$TTA_MAX_UPDATES_PER_EPISODE" \
    --tta "max_grad_norm=$TTA_MAX_GRAD_NORM"

manifest_finalized=0

finalize_manifest() {
    local status="$1"
    if (( manifest_finalized == 0 )); then
        "$PYTHON" "$SCRIPT_DIR/run_manifest.py" finish \
            --path "$RUN_MANIFEST" --status "$status" || true
        manifest_finalized=1
    fi
}

finalize_on_exit() {
    local status=$?
    if (( manifest_finalized == 0 )); then
        if (( status == 0 )); then
            finalize_manifest complete
        else
            finalize_manifest failed
        fi
    fi
}
trap finalize_on_exit EXIT

stats_complete() {
    local part_id="$1"
    local stats_path="$SAVE_ROOT/tb_seed_100_val_part_${part_id}/stats.json"
    local episodes_path="$PONI_ROOT/data/datasets/objectnav/mp3d/v1/val_parts/val_part_${part_id}/content"

    "$PYTHON" - "$stats_path" "$episodes_path" "$TEST_EPISODE_COUNT" <<'PY'
import gzip
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
content_dir = Path(sys.argv[2])
requested = int(sys.argv[3])
if not stats_path.is_file():
    raise SystemExit(1)

with stats_path.open() as handle:
    completed = len(json.load(handle))

available = 0
for path in content_dir.glob("*.json.gz"):
    with gzip.open(path, "rt") as handle:
        available += len(json.load(handle).get("episodes", []))

expected = available if requested < 0 else min(requested, available)
raise SystemExit(0 if completed >= expected else 1)
PY
}

run_part() {
    local part_id="$1"
    local val_part="val_part_${part_id}"
    local tb_dir="$SAVE_ROOT/tb_seed_100_${val_part}"
    local log_file="$SAVE_ROOT/logs_seed_100_${val_part}.txt"
    local console_file="$SAVE_ROOT/console_seed_100_${val_part}.log"
    local diagnostics_file="$SAVE_ROOT/tent_diagnostics_seed_100_${val_part}.json"

    if stats_complete "$part_id"; then
        echo "[$(date '+%F %T')] $val_part already complete; skipping"
        return 0
    fi

    mkdir -p "$tb_dir"
    echo "[$(date '+%F %T')] starting $val_part on GPU $GPU_ID"
    if CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/eval_poni_tent.py" \
        --run-type eval \
        --exp-config "$PONI_ROOT/hlab/transfer_configs/transfer_objectnav_mp3d.yaml" \
        --tta-lr "$TTA_LR" \
        --tta-steps "$TTA_STEPS" \
        --tta-episodic "$TTA_EPISODIC" \
        --tta-reset-bn-stats "$TTA_RESET_BN_STATS" \
        --tta-norm-scope "$TTA_NORM_SCOPE" \
        --tta-norm-prefixes "$TTA_NORM_PREFIXES" \
        --tta-entropy-mode "$TTA_ENTROPY_MODE" \
        --tta-min-adaptation-pixels "$TTA_MIN_ADAPTATION_PIXELS" \
        --tta-optimizer "$TTA_OPTIMIZER" \
        --tta-momentum "$TTA_MOMENTUM" \
        --tta-beta1 "$TTA_BETA1" \
        --tta-beta2 "$TTA_BETA2" \
        --tta-weight-decay "$TTA_WEIGHT_DECAY" \
        --tta-update-interval "$TTA_UPDATE_INTERVAL" \
        --tta-max-updates-per-episode "$TTA_MAX_UPDATES_PER_EPISODE" \
        --tta-max-grad-norm "$TTA_MAX_GRAD_NORM" \
        --tta-diagnostics "$diagnostics_file" \
        TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH" \
        EVAL.SPLIT "$val_part" \
        TEST_EPISODE_COUNT "$TEST_EPISODE_COUNT" \
        TASK_CONFIG.SEED 100 \
        TENSORBOARD_DIR "$tb_dir" \
        LOG_FILE "$log_file" \
        GLOBAL_AGENT.name PFExp \
        GLOBAL_AGENT.smart_local_boundaries True \
        GLOBAL_AGENT.num_local_steps 1 \
        PF_EXP_POLICY.pf_model_path "$MODEL_PATH" \
        PF_EXP_POLICY.pf_masking_opt unexplored \
        >"$console_file" 2>&1; then
        echo "[$(date '+%F %T')] completed $val_part"
    else
        local status=$?
        echo "[$(date '+%F %T')] FAILED $val_part (exit $status); see $console_file" >&2
        return "$status"
    fi
}

terminate_children() {
    local child
    for child in $(jobs -pr); do
        kill "$child" 2>/dev/null || true
    done
}

handle_interrupt() {
    terminate_children
    finalize_manifest interrupted
    exit 130
}
trap handle_interrupt INT TERM

cd "$PONI_ROOT/hlab"
echo "PONI + Tent MP3D evaluation: GPU=$GPU_ID, concurrency=$MAX_JOBS, parts=[$PARTS]"
echo "Run ID: $RUN_ID"
echo "Experiment root: $EXPERIMENT_ROOT"
echo "Run outputs: $SAVE_ROOT"
echo "Monitor: $PYTHON $SCRIPT_DIR/monitor_tent.py --save-root $EXPERIMENT_ROOT"

failed=0
for part_id in $PARTS; do
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        if ! wait -n; then
            failed=1
        fi
    done
    run_part "$part_id" &
done

while (( $(jobs -rp | wc -l) > 0 )); do
    if ! wait -n; then
        failed=1
    fi
done

if (( failed != 0 )); then
    echo "One or more splits failed. Re-run after inspecting console logs." >&2
    exit 1
fi

RESULTS_FILE="$SAVE_ROOT/merged_results.txt"
"$PYTHON" "$PONI_ROOT/hlab/merge_results.py" \
    --path_format "$SAVE_ROOT/tb_seed_100_val_part_*/stats.json" \
    | tee "$RESULTS_FILE"
finalize_manifest complete
echo "Evaluation complete. Merged metrics: $RESULTS_FILE"
