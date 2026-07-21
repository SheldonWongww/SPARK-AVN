#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Launch the Tent update-interval study with fixed LR=1e-7 and intervals 2,3,4.
Each interval runs in its own detached screen session and, by default, on a
separate GPU.

Usage:
  bash avn/scripts/run_tent_interval_grid.sh \
    <model> <source_setting> <norm_scope> [options]

Arguments:
  model             smt_audio | enmus
  source_setting    single_source | multi_source
  norm_scope        ln | last_k_ln | first_ln | last_ln

Options:
  --seed N          Evaluation seed (default: 0)
  --gpus LIST       Three physical GPU ids (default: 0,1,2)
  --episodes N      Episodes per run (default: 2000)
  --batch-id ID     Safe unique batch id (default: current UTC timestamp)
  --dry-run         Validate and print the three jobs without launching
  -h, --help        Show this help

Examples:
  bash avn/scripts/run_tent_interval_grid.sh \
    smt_audio single_source last_ln

  bash avn/scripts/run_tent_interval_grid.sh \
    enmus multi_source last_k_ln --gpus 0,1,2 --dry-run

Run this command from the activated model environment. The interval=1 control
is intentionally not repeated; reuse the matching LR=1e-7 result from the
previous learning-rate x LayerNorm grid.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

case "$1" in
    -h|--help)
        usage
        exit 0
        ;;
esac

[[ $# -ge 3 ]] || \
    die "model, source_setting, and norm_scope are required; use --help"

MODEL="$1"
SOURCE_SETTING="$2"
NORM_SCOPE="$3"
shift 3

SEED=0
GPU_CSV="0,1,2"
EPISODES=2000
BATCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DRY_RUN=0
LEARNING_RATE="1e-7"
INTERVALS=(2 3 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            [[ $# -ge 2 ]] || die "--seed requires a value"
            SEED="$2"
            shift 2
            ;;
        --gpus)
            [[ $# -ge 2 ]] || die "--gpus requires a value"
            GPU_CSV="$2"
            shift 2
            ;;
        --episodes)
            [[ $# -ge 2 ]] || die "--episodes requires a value"
            EPISODES="$2"
            shift 2
            ;;
        --batch-id)
            [[ $# -ge 2 ]] || die "--batch-id requires a value"
            BATCH_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "$MODEL" in
    smt_audio)
        RUNNER_REL="avn/scripts/eval_smt_audio.sh"
        case "$SOURCE_SETTING" in
            single_source) CHECKPOINT_NAME="single_best_val.pth" ;;
            multi_source) CHECKPOINT_NAME="multi_best_val.pth" ;;
            *) die "source_setting must be single_source or multi_source" ;;
        esac
        CHECKPOINT_REL="avn/checkpoints/source/smt_audio/${CHECKPOINT_NAME}"
        ;;
    enmus)
        RUNNER_REL="avn/scripts/eval_enmus.sh"
        case "$SOURCE_SETTING" in
            single_source) CHECKPOINT_NAME="single_source_best_val.pth" ;;
            multi_source) CHECKPOINT_NAME="multi_source_best_val.pth" ;;
            *) die "source_setting must be single_source or multi_source" ;;
        esac
        CHECKPOINT_REL="avn/checkpoints/source/enmus/${CHECKPOINT_NAME}"
        ;;
    *)
        die "model must be smt_audio or enmus"
        ;;
esac

case "$NORM_SCOPE" in
    ln|last_k_ln|first_ln|last_ln) ;;
    *) die "norm_scope must be ln, last_k_ln, first_ln, or last_ln" ;;
esac

[[ "$SEED" =~ ^[0-9]+$ ]] || die "seed must be a nonnegative integer"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "episodes must be a positive integer"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "batch-id may contain only letters, numbers, dot, underscore, and hyphen"

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
[[ ${#GPUS[@]} -eq 3 ]] || die "--gpus must contain exactly three GPU ids"
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id: $gpu"
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/${RUNNER_REL}"
CHECKPOINT="${REPO_ROOT}/${CHECKPOINT_REL}"
DATASET_FILE="${REPO_ROOT}/avn/data/datasets/tta_test/${SOURCE_SETTING}/mp3d/v1/val/val.json.gz"
LOG_ROOT="${REPO_ROOT}/avn/results/logs/tent_interval_grid/${BATCH_ID}/${MODEL}/${SOURCE_SETTING}/${NORM_SCOPE}"

[[ -f "$RUNNER" ]] || die "runner is missing: $RUNNER"

if [[ $DRY_RUN -eq 0 ]]; then
    command -v screen >/dev/null 2>&1 || die "GNU screen is not installed"
    command -v python3 >/dev/null 2>&1 || die "python3 is not available in PATH"
    [[ -f "$CHECKPOINT" ]] || die "checkpoint is missing: $CHECKPOINT"
    [[ -f "$DATASET_FILE" ]] || die "TTA dataset is missing: $DATASET_FILE"
fi

printf 'Tent update-interval grid\n'
printf '  repository: %s\n' "$REPO_ROOT"
printf '  model:      %s\n' "$MODEL"
printf '  source:     %s\n' "$SOURCE_SETTING"
printf '  scope:      %s\n' "$NORM_SCOPE"
printf '  LR:         %s (fixed)\n' "$LEARNING_RATE"
printf '  intervals:  2,3,4\n'
printf '  seed:       %s\n' "$SEED"
printf '  episodes:   %s\n' "$EPISODES"
printf '  GPUs:       %s\n' "$GPU_CSV"
printf '  batch:      %s\n' "$BATCH_ID"
printf '  logs:       %s\n' "$LOG_ROOT"

if [[ $DRY_RUN -eq 0 ]]; then
    for index in 0 1 2; do
        interval="${INTERVALS[$index]}"
        session="tent_interval_${MODEL}_${SOURCE_SETTING}_${NORM_SCOPE}_i${interval}_${BATCH_ID}"
        if screen -ls 2>/dev/null | grep -Fq ".${session}"; then
            die "screen session already exists: $session"
        fi
    done
fi

for index in 0 1 2; do
    GPU="${GPUS[$index]}"
    UPDATE_INTERVAL="${INTERVALS[$index]}"
    SESSION="tent_interval_${MODEL}_${SOURCE_SETTING}_${NORM_SCOPE}_i${UPDATE_INTERVAL}_${BATCH_ID}"
    LOG_DIR="${LOG_ROOT}/interval_${UPDATE_INTERVAL}"
    START_DELAY=$((index * 4))

    if [[ $DRY_RUN -eq 1 ]]; then
        printf '\n[screen %s] GPU=%s start_delay=%ss\n' \
            "$SESSION" "$GPU" "$START_DELAY"
        printf '  %s %s tent %s TTA.LR %s TTA.NORM_SCOPE %s TTA.UPDATE_INTERVAL %s TEST_EPISODE_COUNT %s\n' \
            "$RUNNER_REL" "$SOURCE_SETTING" "$SEED" "$LEARNING_RATE" \
            "$NORM_SCOPE" "$UPDATE_INTERVAL" "$EPISODES"
        continue
    fi

    mkdir -p "$LOG_DIR"
    screen -dmS "$SESSION" env \
        REPO_ROOT="$REPO_ROOT" \
        RUNNER="$RUNNER" \
        MODEL="$MODEL" \
        SOURCE_SETTING="$SOURCE_SETTING" \
        NORM_SCOPE="$NORM_SCOPE" \
        LEARNING_RATE="$LEARNING_RATE" \
        UPDATE_INTERVAL="$UPDATE_INTERVAL" \
        SEED="$SEED" \
        GPU="$GPU" \
        EPISODES="$EPISODES" \
        LOG_DIR="$LOG_DIR" \
        START_DELAY="$START_DELAY" \
        bash -c '
            set -uo pipefail
            cd "$REPO_ROOT" || exit 1
            mkdir -p "$LOG_DIR"

            # tee keeps the screen useful for live inspection while retaining
            # the complete console output in an ignored local log file.
            exec > >(tee -a "$LOG_DIR/interval_${UPDATE_INTERVAL}.log") 2>&1

            printf "started_at=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf "model=%s source=%s scope=%s interval=%s lr=%s gpu=%s\n" \
                "$MODEL" "$SOURCE_SETTING" "$NORM_SCOPE" \
                "$UPDATE_INTERVAL" "$LEARNING_RATE" "$GPU"
            printf "initial_delay=%s\n" "$START_DELAY"
            sleep "$START_DELAY"

            CUDA_DEVICE_ORDER=PCI_BUS_ID \
            CUDA_VISIBLE_DEVICES="$GPU" \
            TF_FORCE_GPU_ALLOW_GROWTH=true \
            PYTHONUNBUFFERED=1 \
            OMP_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            bash "$RUNNER" "$SOURCE_SETTING" tent "$SEED" \
                TTA.LR "$LEARNING_RATE" \
                TTA.NORM_SCOPE "$NORM_SCOPE" \
                TTA.LAST_K_LN 4 \
                TTA.EPISODIC False \
                TTA.STEPS 1 \
                TTA.UPDATE_INTERVAL "$UPDATE_INTERVAL" \
                TEST_EPISODE_COUNT "$EPISODES"

            status=$?
            printf "%s\n" "$status" >"$LOG_DIR/exitcode"
            {
                printf "completed_at=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                printf "exitcode=%s\n" "$status"
            } >"$LOG_DIR/ALL_DONE"
            printf "completed_at=%s exitcode=%s\n" \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status"
            exit "$status"
        '

    printf '  started screen=%s GPU=%s interval=%s\n' \
        "$SESSION" "$GPU" "$UPDATE_INTERVAL"
done

if [[ $DRY_RUN -eq 1 ]]; then
    printf '\nDry run complete: 3 jobs, no screen sessions created.\n'
else
    printf '\nAll three interval-study screen sessions were created.\n'
    printf 'Monitor sessions: screen -ls\n'
    printf 'Monitor GPUs:     watch -n 2 nvidia-smi\n'
    printf 'Logs:             %s\n' "$LOG_ROOT"
    printf 'The matching interval=1 result must be taken from the earlier grid.\n'
fi
