#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the canonical 120-job Tent core grid on single-source AVN.

Fixed Cartesian product:
  LR:              1e-8, 3e-8, 1e-7, 3e-7, 1e-6
  UPDATE_INTERVAL: 1, 2, 3, 4, 8, 16
  NORM_SCOPE:      first_ln, last_ln, last_k_ln, ln

Fixed controls:
  EPISODIC=False, STEPS=1, LAST_K_LN=4, Adam, weight_decay=0,
  max_grad_norm=1.0

Usage:
  bash avn/scripts/run_tent_core_grid.sh [options]

Options:
  --model MODEL        smt_audio|enmus (default: smt_audio)
  --seed N              Evaluation/order seed (default: 0)
  --gpus LIST           Four physical GPU ids (default: 0,1,2,3)
  --jobs-per-gpu N      Concurrent jobs per GPU (default: 4; maximum: 16)
  --episodes N          Episodes per job (default: 2000)
  --batch-id ID         Safe unique batch id (default: current UTC timestamp)
  --resume              Resume the named batch; skip jobs with exit code 0
  --dry-run             Print all 120 planned jobs without launching
  -h, --help            Show this help

Recommended detached launch:
  screen -dmS tent_core_grid \
    bash avn/scripts/run_tent_core_grid.sh --jobs-per-gpu 4

ENMuS convenience entry point:
  screen -dmS enmus_tent_core \
    bash avn/scripts/run_enmus_tent_core_grid.sh --jobs-per-gpu 4

Attach with:
  screen -d -r tent_core_grid

This is a development/tuning grid. Do not select the best configuration on the
same episode stream later used as the untouched final test table.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

MODEL="smt_audio"
SEED=0
GPU_CSV="0,1,2,3"
JOBS_PER_GPU=4
EPISODES=2000
BATCH_ID=""
RESUME=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || die "--model requires a value"
            MODEL="$2"
            shift 2
            ;;
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
        --jobs-per-gpu)
            [[ $# -ge 2 ]] || die "--jobs-per-gpu requires a value"
            JOBS_PER_GPU="$2"
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
        --resume)
            RESUME=1
            shift
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
    smt_audio|enmus) ;;
    *) die "model must be smt_audio or enmus: $MODEL" ;;
esac
if [[ -z "$BATCH_ID" ]]; then
    BATCH_ID="tent-core-${MODEL}-v1-seed${SEED}-$(date -u +%Y%m%dT%H%M%SZ)"
fi

[[ "$SEED" =~ ^[0-9]+$ ]] || die "seed must be a nonnegative integer"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "episodes must be a positive integer"
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || \
    die "jobs-per-gpu must be a positive integer"
[[ "$JOBS_PER_GPU" -le 16 ]] || die "jobs-per-gpu must not exceed 16"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "batch-id may contain only letters, numbers, dot, underscore, and hyphen"

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
[[ ${#GPUS[@]} -eq 4 ]] || die "--gpus must contain exactly four GPU ids"
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id: $gpu"
done

LRS=("1e-8" "3e-8" "1e-7" "3e-7" "1e-6")
UPDATE_INTERVALS=(1 2 3 4 8 16)
NORM_SCOPES=("first_ln" "last_ln" "last_k_ln" "ln")
EXPECTED_JOBS=$((${#LRS[@]} * ${#UPDATE_INTERVALS[@]} * ${#NORM_SCOPES[@]}))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
case "$MODEL" in
    smt_audio)
        RUNNER="${REPO_ROOT}/avn/scripts/eval_smt_audio.sh"
        CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/smt_audio/single_best_val.pth"
        ;;
    enmus)
        RUNNER="${REPO_ROOT}/avn/scripts/eval_enmus.sh"
        CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/enmus/single_source_best_val.pth"
        ;;
esac
DATASET_FILE="${REPO_ROOT}/avn/data/datasets/tta_test/single_source/mp3d/v1/val/val.json.gz"
LOG_ROOT="${REPO_ROOT}/avn/results/logs/tent_core_grid/${BATCH_ID}"
JOBS_ROOT="${LOG_ROOT}/jobs"

[[ -f "$RUNNER" ]] || die "runner is missing: $RUNNER"

printf 'Tent core Cartesian grid\n'
printf '  repository:       %s\n' "$REPO_ROOT"
printf '  model/source:     %s/single_source\n' "$MODEL"
printf '  LRs:              1e-8,3e-8,1e-7,3e-7,1e-6\n'
printf '  update intervals: 1,2,3,4,8,16\n'
printf '  norm scopes:      first_ln,last_ln,last_k_ln,ln\n'
printf '  jobs:             %s\n' "$EXPECTED_JOBS"
printf '  GPUs:             %s\n' "$GPU_CSV"
printf '  jobs/GPU:         %s\n' "$JOBS_PER_GPU"
printf '  seed:             %s\n' "$SEED"
printf '  episodes/job:     %s\n' "$EPISODES"
printf '  batch:            %s\n' "$BATCH_ID"
printf '  logs:             %s\n' "$LOG_ROOT"

lr_slug() {
    local value="$1"
    value="${value//./p}"
    value="${value//+/p}"
    value="${value//-/m}"
    printf '%s\n' "$value"
}

print_or_record_plan() {
    local output="$1"
    local job_index=0
    local gpu_index
    local gpu
    local scope
    local interval
    local lr
    local tag

    printf 'job_id,run_tag,gpu,norm_scope,lr,update_interval,seed,episodes\n' > "$output"
    for scope in "${NORM_SCOPES[@]}"; do
        for interval in "${UPDATE_INTERVALS[@]}"; do
            for lr in "${LRS[@]}"; do
                gpu_index=$((job_index % ${#GPUS[@]}))
                gpu="${GPUS[$gpu_index]}"
                tag="tentcore-${BATCH_ID}-j$(printf '%03d' "$job_index")-${scope}-lr$(lr_slug "$lr")-u${interval}"
                printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
                    "$job_index" "$tag" "$gpu" "$scope" "$lr" \
                    "$interval" "$SEED" "$EPISODES" >> "$output"
                job_index=$((job_index + 1))
            done
        done
    done
}

if [[ $DRY_RUN -eq 1 ]]; then
    DRY_PLAN="$(mktemp -t navtta_tent_core_grid.XXXXXX)"
    print_or_record_plan "$DRY_PLAN"
    cat "$DRY_PLAN"
    rm -f "$DRY_PLAN"
    printf '\nDry run complete: %s jobs, nothing launched.\n' "$EXPECTED_JOBS"
    exit 0
fi

command -v python3 >/dev/null 2>&1 || die "python3 is not available in PATH"
command -v tee >/dev/null 2>&1 || die "tee is not available in PATH"
command -v cmp >/dev/null 2>&1 || die "cmp is not available in PATH"
[[ -f "$CHECKPOINT" ]] || die "checkpoint is missing: $CHECKPOINT"
[[ -f "$DATASET_FILE" ]] || die "TTA dataset is missing: $DATASET_FILE"

if [[ -e "$LOG_ROOT" && $RESUME -eq 0 ]]; then
    die "batch already exists; choose another --batch-id or pass --resume: $LOG_ROOT"
fi
if [[ ! -d "$LOG_ROOT" && $RESUME -eq 1 ]]; then
    die "cannot resume a missing batch: $LOG_ROOT"
fi

mkdir -p "$JOBS_ROOT"
PLAN_FILE="${LOG_ROOT}/grid.csv"
if [[ $RESUME -eq 1 ]]; then
    [[ -f "$PLAN_FILE" ]] || die "resume batch is missing its grid plan: $PLAN_FILE"
    RESUME_PLAN="$(mktemp -t navtta_tent_core_resume.XXXXXX)"
    print_or_record_plan "$RESUME_PLAN"
    if ! cmp -s "$PLAN_FILE" "$RESUME_PLAN"; then
        rm -f "$RESUME_PLAN"
        die "resume arguments do not match the original grid plan"
    fi
    rm -f "$RESUME_PLAN"
else
    print_or_record_plan "$PLAN_FILE"
fi
SCHEDULER_LOG="${LOG_ROOT}/scheduler.log"
exec > >(tee -a "$SCHEDULER_LOG") 2>&1

printf 'scheduler_started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'python=%s\n' "$(command -v python3)"
printf 'resume=%s\n' "$RESUME"

GPU_ACTIVE=(0 0 0 0)
PIDS=()
PID_GPU_INDEX=()
PID_TAG=()
PID_JOB_DIR=()
LAUNCHED=0
SKIPPED=0
REAPED=0
REAP_FAILURES=0

total_active() {
    local total=0
    local count
    for count in "${GPU_ACTIVE[@]}"; do
        total=$((total + count))
    done
    printf '%s\n' "$total"
}

reap_finished() {
    local index
    local pid
    local gpu_index
    local tag
    local job_dir
    local status

    for ((index=0; index<${#PIDS[@]}; index++)); do
        pid="${PIDS[$index]}"
        [[ -n "$pid" ]] || continue
        job_dir="${PID_JOB_DIR[$index]}"

        if [[ -f "${job_dir}/exitcode" ]] || ! kill -0 "$pid" 2>/dev/null; then
            if wait "$pid"; then
                status=0
            else
                status=$?
            fi

            gpu_index="${PID_GPU_INDEX[$index]}"
            tag="${PID_TAG[$index]}"
            GPU_ACTIVE[$gpu_index]=$((GPU_ACTIVE[$gpu_index] - 1))
            PIDS[$index]=""
            REAPED=$((REAPED + 1))
            if [[ $status -ne 0 ]]; then
                REAP_FAILURES=$((REAP_FAILURES + 1))
            fi
            printf '%s finished tag=%s status=%s active=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$tag" "$status" \
                "$(total_active)"
        fi
    done
}

wait_for_gpu_slot() {
    local gpu_index="$1"
    while [[ ${GPU_ACTIVE[$gpu_index]} -ge $JOBS_PER_GPU ]]; do
        reap_finished
        if [[ ${GPU_ACTIVE[$gpu_index]} -ge $JOBS_PER_GPU ]]; then
            sleep 5
        fi
    done
}

launch_job() {
    local job_index="$1"
    local gpu_index="$2"
    local gpu="$3"
    local scope="$4"
    local lr="$5"
    local interval="$6"
    local tag="$7"
    local job_dir="${JOBS_ROOT}/${tag}"
    local prior_status=""
    local attempt
    local pid

    mkdir -p "$job_dir"
    if [[ -f "${job_dir}/exitcode" ]]; then
        prior_status="$(tr -d '[:space:]' < "${job_dir}/exitcode")"
        if [[ $RESUME -eq 1 && "$prior_status" == "0" ]]; then
            SKIPPED=$((SKIPPED + 1))
            printf '%s skip completed tag=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$tag"
            return
        fi
        attempt="$(date -u +%Y%m%dT%H%M%SZ)"
        mv "${job_dir}/exitcode" "${job_dir}/exitcode.previous.${attempt}"
    fi

    wait_for_gpu_slot "$gpu_index"
    {
        printf 'job_id=%s\n' "$job_index"
        printf 'run_tag=%s\n' "$tag"
        printf 'gpu=%s\n' "$gpu"
        printf 'model=%s\n' "$MODEL"
        printf 'norm_scope=%s\n' "$scope"
        printf 'lr=%s\n' "$lr"
        printf 'update_interval=%s\n' "$interval"
        printf 'seed=%s\n' "$SEED"
        printf 'episodes=%s\n' "$EPISODES"
    } > "${job_dir}/parameters.env"

    (
        set +e
        printf '\n===== attempt %s =====\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${job_dir}/console.log"
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="$gpu" \
        TF_FORCE_GPU_ALLOW_GROWTH=true \
        PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        NAVTTA_RUN_TAG="$tag" \
        bash "$RUNNER" single_source tent "$SEED" \
            TTA.LR "$lr" \
            TTA.NORM_SCOPE "$scope" \
            TTA.LAST_K_LN 4 \
            TTA.EPISODIC False \
            TTA.STEPS 1 \
            TTA.UPDATE_INTERVAL "$interval" \
            TTA.MAX_UPDATES_PER_EPISODE -1 \
            TTA.OPTIMIZER Adam \
            TTA.BETA1 0.9 \
            TTA.BETA2 0.999 \
            TTA.WEIGHT_DECAY 0.0 \
            TTA.MAX_GRAD_NORM 1.0 \
            TEST_EPISODE_COUNT "$EPISODES" \
            >> "${job_dir}/console.log" 2>&1
        status=$?
        printf '%s\n' "$status" > "${job_dir}/exitcode.tmp"
        mv "${job_dir}/exitcode.tmp" "${job_dir}/exitcode"
        exit "$status"
    ) &
    pid=$!

    PIDS+=("$pid")
    PID_GPU_INDEX+=("$gpu_index")
    PID_TAG+=("$tag")
    PID_JOB_DIR+=("$job_dir")
    GPU_ACTIVE[$gpu_index]=$((GPU_ACTIVE[$gpu_index] + 1))
    LAUNCHED=$((LAUNCHED + 1))
    printf '%s launched job=%s tag=%s gpu=%s active_on_gpu=%s total_active=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_index" "$tag" "$gpu" \
        "${GPU_ACTIVE[$gpu_index]}" "$(total_active)"
}

JOB_INDEX=0
for SCOPE in "${NORM_SCOPES[@]}"; do
    for UPDATE_INTERVAL in "${UPDATE_INTERVALS[@]}"; do
        for LR in "${LRS[@]}"; do
            GPU_INDEX=$((JOB_INDEX % ${#GPUS[@]}))
            GPU="${GPUS[$GPU_INDEX]}"
            RUN_TAG="tentcore-${BATCH_ID}-j$(printf '%03d' "$JOB_INDEX")-${SCOPE}-lr$(lr_slug "$LR")-u${UPDATE_INTERVAL}"
            launch_job "$JOB_INDEX" "$GPU_INDEX" "$GPU" "$SCOPE" "$LR" \
                "$UPDATE_INTERVAL" "$RUN_TAG"
            JOB_INDEX=$((JOB_INDEX + 1))
            reap_finished
        done
    done
done

while [[ $(total_active) -gt 0 ]]; do
    reap_finished
    [[ $(total_active) -eq 0 ]] || sleep 5
done

EXIT_FILES=$(find "$JOBS_ROOT" -type f -name exitcode | wc -l | tr -d ' ')
SUCCESSFUL=0
FAILED=0
while IFS= read -r EXIT_FILE; do
    STATUS="$(tr -d '[:space:]' < "$EXIT_FILE")"
    if [[ "$STATUS" == "0" ]]; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done < <(find "$JOBS_ROOT" -type f -name exitcode | sort)
MISSING=$((EXPECTED_JOBS - EXIT_FILES))

{
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'expected=%s\n' "$EXPECTED_JOBS"
    printf 'launched_this_invocation=%s\n' "$LAUNCHED"
    printf 'skipped_completed=%s\n' "$SKIPPED"
    printf 'successful=%s\n' "$SUCCESSFUL"
    printf 'failed=%s\n' "$FAILED"
    printf 'missing=%s\n' "$MISSING"
} | tee "${LOG_ROOT}/SUMMARY"

if [[ $FAILED -ne 0 || $MISSING -ne 0 ]]; then
    printf 'Grid incomplete; rerun with --batch-id %s --resume after inspection.\n' \
        "$BATCH_ID" >&2
    exit 1
fi

printf 'All %s %s Tent core-grid jobs completed successfully.\n' \
    "$EXPECTED_JOBS" "$MODEL"
