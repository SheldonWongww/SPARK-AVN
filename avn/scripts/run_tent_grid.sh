#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Launch the 3-learning-rate x 4-LayerNorm-scope Tent grid in four detached
screen sessions. Each GPU runs the three learning rates concurrently.

Usage:
  bash avn/scripts/run_tent_grid.sh <model> <source_setting> [options]

Arguments:
  model             smt_audio | enmus
  source_setting    single_source | multi_source

Options:
  --seed N          Evaluation seed (default: 0)
  --gpus LIST       Four physical GPU ids (default: 0,1,2,3)
  --lrs LIST        Three comma-separated learning rates
                    (default: 1e-5,1e-6,1e-7)
  --episodes N      Episodes per run (default: 2000)
  --batch-id ID     Safe unique batch id (default: current UTC timestamp)
  --dry-run         Validate arguments and print the 12 jobs without launching
  -h, --help        Show this help

Examples:
  bash avn/scripts/run_tent_grid.sh smt_audio single_source
  bash avn/scripts/run_tent_grid.sh enmus multi_source --seed 0
  bash avn/scripts/run_tent_grid.sh smt_audio single_source --episodes 1 --dry-run

Run this command from an activated model environment. Detached screen sessions
inherit the current PATH/Conda environment.
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

[[ $# -ge 2 ]] || die "model and source_setting are required; use --help"

MODEL="$1"
SOURCE_SETTING="$2"
shift 2

SEED=0
GPU_CSV="0,1,2,3"
LR_CSV="1e-5,1e-6,1e-7"
EPISODES=2000
BATCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DRY_RUN=0
LAUNCH_GAP=3

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
        --lrs)
            [[ $# -ge 2 ]] || die "--lrs requires a value"
            LR_CSV="$2"
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

[[ "$SEED" =~ ^[0-9]+$ ]] || die "seed must be a nonnegative integer"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "episodes must be a positive integer"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "batch-id may contain only letters, numbers, dot, underscore, and hyphen"

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
IFS=',' read -r -a LRS <<< "$LR_CSV"
SCOPES=("ln" "last_k_ln" "first_ln" "last_ln")

[[ ${#GPUS[@]} -eq 4 ]] || die "--gpus must contain exactly four GPU ids"
[[ ${#LRS[@]} -eq 3 ]] || die "--lrs must contain exactly three learning rates"

for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id: $gpu"
done

for lr in "${LRS[@]}"; do
    [[ "$lr" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] || \
        die "invalid learning rate: $lr"
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/${RUNNER_REL}"
CHECKPOINT="${REPO_ROOT}/${CHECKPOINT_REL}"
DATASET_FILE="${REPO_ROOT}/avn/data/datasets/tta_test/${SOURCE_SETTING}/mp3d/v1/val/val.json.gz"
LOG_ROOT="${REPO_ROOT}/avn/results/logs/tent_grid/${BATCH_ID}/${MODEL}/${SOURCE_SETTING}"

[[ -f "$RUNNER" ]] || die "runner is missing: $RUNNER"

if [[ $DRY_RUN -eq 0 ]]; then
    command -v screen >/dev/null 2>&1 || die "GNU screen is not installed"
    command -v python3 >/dev/null 2>&1 || die "python3 is not available in PATH"
    [[ -f "$CHECKPOINT" ]] || die "checkpoint is missing: $CHECKPOINT"
    [[ -f "$DATASET_FILE" ]] || die "TTA dataset is missing: $DATASET_FILE"
fi

printf 'Tent grid configuration\n'
printf '  repository: %s\n' "$REPO_ROOT"
printf '  model:      %s\n' "$MODEL"
printf '  source:     %s\n' "$SOURCE_SETTING"
printf '  seed:       %s\n' "$SEED"
printf '  episodes:   %s\n' "$EPISODES"
printf '  GPUs:       %s\n' "$GPU_CSV"
printf '  LRs:        %s\n' "$LR_CSV"
printf '  batch:      %s\n' "$BATCH_ID"
printf '  logs:       %s\n' "$LOG_ROOT"

# Refuse the complete launch before creating any partial batch when a session
# with the same deterministic name already exists.
if [[ $DRY_RUN -eq 0 ]]; then
    for index in 0 1 2 3; do
        session="tent_${MODEL}_${SOURCE_SETTING}_${SCOPES[$index]}_${BATCH_ID}"
        if screen -ls 2>/dev/null | grep -Fq ".${session}"; then
            die "screen session already exists: $session"
        fi
    done
fi

for index in 0 1 2 3; do
    GPU="${GPUS[$index]}"
    SCOPE="${SCOPES[$index]}"
    SESSION="tent_${MODEL}_${SOURCE_SETTING}_${SCOPE}_${BATCH_ID}"
    LOG_DIR="${LOG_ROOT}/${SCOPE}"

    # The evaluation run id has second-level timestamp precision and does not
    # include the grid hyperparameters. Stagger all twelve launches so no two
    # jobs can select the same result directory.
    START_DELAY=$((index * 12))

    if [[ $DRY_RUN -eq 1 ]]; then
        printf '\n[screen %s] GPU=%s scope=%s start_delay=%ss\n' \
            "$SESSION" "$GPU" "$SCOPE" "$START_DELAY"
        for lr in "${LRS[@]}"; do
            printf '  %s %s tent %s TTA.LR %s TTA.NORM_SCOPE %s TEST_EPISODE_COUNT %s\n' \
                "$RUNNER_REL" "$SOURCE_SETTING" "$SEED" "$lr" "$SCOPE" "$EPISODES"
        done
        continue
    fi

    mkdir -p "$LOG_DIR"
    screen -dmS "$SESSION" env \
        REPO_ROOT="$REPO_ROOT" \
        RUNNER="$RUNNER" \
        MODEL="$MODEL" \
        SOURCE_SETTING="$SOURCE_SETTING" \
        SEED="$SEED" \
        GPU="$GPU" \
        SCOPE="$SCOPE" \
        LR_CSV="$LR_CSV" \
        EPISODES="$EPISODES" \
        LOG_DIR="$LOG_DIR" \
        START_DELAY="$START_DELAY" \
        LAUNCH_GAP="$LAUNCH_GAP" \
        bash -c '
            set -u
            cd "$REPO_ROOT" || exit 1
            mkdir -p "$LOG_DIR"
            exec >"$LOG_DIR/launcher.log" 2>&1

            printf "started_at=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            printf "model=%s source=%s gpu=%s scope=%s seed=%s\n" \
                "$MODEL" "$SOURCE_SETTING" "$GPU" "$SCOPE" "$SEED"
            printf "initial_delay=%s\n" "$START_DELAY"
            sleep "$START_DELAY"

            IFS=, read -r -a LRS <<< "$LR_CSV"
            PIDS=()

            for LR in "${LRS[@]}"; do
                printf "launching_at=%s lr=%s\n" \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LR"

                (
                    CUDA_DEVICE_ORDER=PCI_BUS_ID \
                    CUDA_VISIBLE_DEVICES="$GPU" \
                    TF_FORCE_GPU_ALLOW_GROWTH=true \
                    PYTHONUNBUFFERED=1 \
                    OMP_NUM_THREADS=1 \
                    MKL_NUM_THREADS=1 \
                    bash "$RUNNER" "$SOURCE_SETTING" tent "$SEED" \
                        TTA.LR "$LR" \
                        TTA.NORM_SCOPE "$SCOPE" \
                        TTA.LAST_K_LN 4 \
                        TTA.EPISODIC False \
                        TTA.STEPS 1 \
                        TTA.UPDATE_INTERVAL 1 \
                        TEST_EPISODE_COUNT "$EPISODES" \
                        >"$LOG_DIR/lr_${LR}.log" 2>&1

                    status=$?
                    printf "%s\n" "$status" >"$LOG_DIR/lr_${LR}.exitcode"
                    exit "$status"
                ) &

                PIDS+=("$!")
                sleep "$LAUNCH_GAP"
            done

            FAILURES=0
            for PID in "${PIDS[@]}"; do
                if ! wait "$PID"; then
                    FAILURES=$((FAILURES + 1))
                fi
            done

            {
                printf "completed_at=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
                printf "failures=%s\n" "$FAILURES"
            } >"$LOG_DIR/ALL_DONE"

            exit "$FAILURES"
        '

    printf '  started screen=%s GPU=%s scope=%s\n' "$SESSION" "$GPU" "$SCOPE"
done

if [[ $DRY_RUN -eq 1 ]]; then
    printf '\nDry run complete: 12 jobs, no screen sessions created.\n'
else
    printf '\nAll four screen sessions were created.\n'
    printf 'Monitor sessions: screen -ls\n'
    printf 'Monitor GPUs:     watch -n 2 nvidia-smi\n'
    printf 'Logs:             %s\n' "$LOG_ROOT"
    printf 'Exit code 0 in each *.exitcode file means that run completed successfully.\n'
fi
