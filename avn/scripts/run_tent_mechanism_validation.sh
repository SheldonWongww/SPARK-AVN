#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Validate why Tent behaves differently in DUET and SMT+Audio.

The default 10-job grid contains:
  Source controls:
    source x {sample, argmax}
  Tent mechanism runs:
    {sample, argmax} x max updates/episode {5, 15, 30, unlimited}

Tent defaults target the known unstable setting so that recovery is visible:
  NORM_SCOPE=last_k_ln, LR=1e-6, UPDATE_INTERVAL=1,
  EPISODIC=False, Adam, max_grad_norm=1.0.

With UPDATE_INTERVAL=1, a cap of 15 means that only the first 15 actions in
each episode may update Tent. Model parameters still persist across episodes.

Usage:
  bash avn/scripts/run_tent_mechanism_validation.sh [options]

Options:
  --seed N              Evaluation/order seed (default: 0)
  --gpus LIST           Four physical GPU ids (default: 0,1,2,3)
  --jobs-per-gpu N      Concurrent jobs per GPU (default: 3; maximum: 16)
  --episodes N          Episodes per job (default: 2000)
  --lr VALUE            Tent learning rate (default: 1e-6)
  --scope SCOPE         first_ln|last_ln|last_k_ln|ln (default: last_k_ln)
  --update-interval N   Global action update interval (default: 1)
  --caps LIST           Per-episode update caps (default: 5,15,30,-1)
                        Use -1 for unlimited.
  --batch-id ID         Safe unique batch id (default: current UTC timestamp)
  --resume              Resume the named batch; skip jobs with exit code 0
  --dry-run             Print the planned jobs without launching
  -h, --help            Show this help

Recommended detached launch:
  screen -dmS tent_mechanism \
    bash avn/scripts/run_tent_mechanism_validation.sh --jobs-per-gpu 3

Attach with:
  screen -d -r tent_mechanism

This is a causal/mechanism experiment. Use a TTA-dev stream for conclusions;
do not select final-paper hyperparameters on an untouched test stream.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

SEED=0
GPU_CSV="0,1,2,3"
JOBS_PER_GPU=3
EPISODES=2000
LEARNING_RATE="1e-6"
NORM_SCOPE="last_k_ln"
UPDATE_INTERVAL=1
CAP_CSV="5,15,30,-1"
BATCH_ID=""
RESUME=0
DRY_RUN=0

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
        --lr)
            [[ $# -ge 2 ]] || die "--lr requires a value"
            LEARNING_RATE="$2"
            shift 2
            ;;
        --scope)
            [[ $# -ge 2 ]] || die "--scope requires a value"
            NORM_SCOPE="$2"
            shift 2
            ;;
        --update-interval)
            [[ $# -ge 2 ]] || die "--update-interval requires a value"
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        --caps)
            [[ $# -ge 2 ]] || die "--caps requires a value"
            CAP_CSV="$2"
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

if [[ -z "$BATCH_ID" ]]; then
    BATCH_ID="tent-mechanism-v1-seed${SEED}-$(date -u +%Y%m%dT%H%M%SZ)"
fi

[[ "$SEED" =~ ^[0-9]+$ ]] || die "seed must be a nonnegative integer"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || die "episodes must be positive"
[[ "$JOBS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || \
    die "jobs-per-gpu must be positive"
[[ "$JOBS_PER_GPU" -le 16 ]] || die "jobs-per-gpu must not exceed 16"
[[ "$UPDATE_INTERVAL" =~ ^[1-9][0-9]*$ ]] || \
    die "update interval must be positive"
[[ "$LEARNING_RATE" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || \
    die "invalid learning rate: $LEARNING_RATE"
[[ "$BATCH_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "batch-id may contain only letters, numbers, dot, underscore, and hyphen"

case "$NORM_SCOPE" in
    first_ln|last_ln|last_k_ln|ln) ;;
    *) die "invalid scope: $NORM_SCOPE" ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
[[ ${#GPUS[@]} -eq 4 ]] || die "--gpus must contain exactly four GPU ids"
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id: $gpu"
done

IFS=',' read -r -a UPDATE_CAPS <<< "$CAP_CSV"
[[ ${#UPDATE_CAPS[@]} -gt 0 ]] || die "--caps must not be empty"
for cap in "${UPDATE_CAPS[@]}"; do
    [[ "$cap" == "-1" || "$cap" =~ ^[0-9]+$ ]] || \
        die "invalid update cap: $cap"
done

ACTION_SELECTIONS=("sample" "argmax")
EXPECTED_JOBS=$((2 + ${#ACTION_SELECTIONS[@]} * ${#UPDATE_CAPS[@]}))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/avn/scripts/eval_smt_audio.sh"
CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/smt_audio/single_best_val.pth"
DATASET_FILE="${REPO_ROOT}/avn/data/datasets/tta_test/single_source/mp3d/v1/val/val.json.gz"
LOG_ROOT="${REPO_ROOT}/avn/results/logs/tent_mechanism_validation/${BATCH_ID}"
JOBS_ROOT="${LOG_ROOT}/jobs"

[[ -f "$RUNNER" ]] || die "runner is missing: $RUNNER"

cap_slug() {
    if [[ "$1" == "-1" ]]; then
        printf 'all\n'
    else
        printf '%s\n' "$1"
    fi
}

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
    local action_selection
    local cap
    local tag

    printf '%s\n' \
        'job_id,run_tag,gpu,method,action_selection,max_updates_per_episode,norm_scope,lr,update_interval,seed,episodes' \
        > "$output"

    for action_selection in "${ACTION_SELECTIONS[@]}"; do
        gpu_index=$((job_index % ${#GPUS[@]}))
        gpu="${GPUS[$gpu_index]}"
        tag="tentmech-${BATCH_ID}-j$(printf '%03d' "$job_index")-source-${action_selection}"
        printf '%s,%s,%s,source,%s,-1,%s,%s,%s,%s,%s\n' \
            "$job_index" "$tag" "$gpu" "$action_selection" \
            "$NORM_SCOPE" "$LEARNING_RATE" "$UPDATE_INTERVAL" \
            "$SEED" "$EPISODES" >> "$output"
        job_index=$((job_index + 1))
    done

    for action_selection in "${ACTION_SELECTIONS[@]}"; do
        for cap in "${UPDATE_CAPS[@]}"; do
            gpu_index=$((job_index % ${#GPUS[@]}))
            gpu="${GPUS[$gpu_index]}"
            tag="tentmech-${BATCH_ID}-j$(printf '%03d' "$job_index")-tent-${action_selection}-cap$(cap_slug "$cap")-lr$(lr_slug "$LEARNING_RATE")"
            printf '%s,%s,%s,tent,%s,%s,%s,%s,%s,%s,%s\n' \
                "$job_index" "$tag" "$gpu" "$action_selection" "$cap" \
                "$NORM_SCOPE" "$LEARNING_RATE" "$UPDATE_INTERVAL" \
                "$SEED" "$EPISODES" >> "$output"
            job_index=$((job_index + 1))
        done
    done
}

printf 'Tent mechanism validation grid\n'
printf '  repository:          %s\n' "$REPO_ROOT"
printf '  model/source:        smt_audio/single_source\n'
printf '  action selections:   sample,argmax\n'
printf '  update caps/episode: %s\n' "$CAP_CSV"
printf '  Tent scope/LR/u:     %s / %s / %s\n' \
    "$NORM_SCOPE" "$LEARNING_RATE" "$UPDATE_INTERVAL"
printf '  jobs:                %s\n' "$EXPECTED_JOBS"
printf '  GPUs:                %s\n' "$GPU_CSV"
printf '  jobs/GPU:            %s\n' "$JOBS_PER_GPU"
printf '  seed:                %s\n' "$SEED"
printf '  episodes/job:        %s\n' "$EPISODES"
printf '  batch:               %s\n' "$BATCH_ID"
printf '  logs:                %s\n' "$LOG_ROOT"

if [[ $DRY_RUN -eq 1 ]]; then
    DRY_PLAN="$(mktemp -t navtta_tent_mechanism.XXXXXX)"
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
    [[ -f "$PLAN_FILE" ]] || die "resume batch is missing grid.csv"
    RESUME_PLAN="$(mktemp -t navtta_tent_mechanism_resume.XXXXXX)"
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
    local method="$4"
    local action_selection="$5"
    local max_updates="$6"
    local scope="$7"
    local lr="$8"
    local interval="$9"
    local tag="${10}"
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
        printf 'method=%s\n' "$method"
        printf 'action_selection=%s\n' "$action_selection"
        printf 'max_updates_per_episode=%s\n' "$max_updates"
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
        bash "$RUNNER" single_source "$method" "$SEED" \
            EVAL.ACTION_SELECTION "$action_selection" \
            TTA.LR "$lr" \
            TTA.NORM_SCOPE "$scope" \
            TTA.LAST_K_LN 4 \
            TTA.EPISODIC False \
            TTA.STEPS 1 \
            TTA.UPDATE_INTERVAL "$interval" \
            TTA.MAX_UPDATES_PER_EPISODE "$max_updates" \
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
for ACTION_SELECTION in "${ACTION_SELECTIONS[@]}"; do
    GPU_INDEX=$((JOB_INDEX % ${#GPUS[@]}))
    GPU="${GPUS[$GPU_INDEX]}"
    RUN_TAG="tentmech-${BATCH_ID}-j$(printf '%03d' "$JOB_INDEX")-source-${ACTION_SELECTION}"
    launch_job "$JOB_INDEX" "$GPU_INDEX" "$GPU" source \
        "$ACTION_SELECTION" -1 "$NORM_SCOPE" "$LEARNING_RATE" \
        "$UPDATE_INTERVAL" "$RUN_TAG"
    JOB_INDEX=$((JOB_INDEX + 1))
    reap_finished
done

for ACTION_SELECTION in "${ACTION_SELECTIONS[@]}"; do
    for MAX_UPDATES in "${UPDATE_CAPS[@]}"; do
        GPU_INDEX=$((JOB_INDEX % ${#GPUS[@]}))
        GPU="${GPUS[$GPU_INDEX]}"
        RUN_TAG="tentmech-${BATCH_ID}-j$(printf '%03d' "$JOB_INDEX")-tent-${ACTION_SELECTION}-cap$(cap_slug "$MAX_UPDATES")-lr$(lr_slug "$LEARNING_RATE")"
        launch_job "$JOB_INDEX" "$GPU_INDEX" "$GPU" tent \
            "$ACTION_SELECTION" "$MAX_UPDATES" "$NORM_SCOPE" \
            "$LEARNING_RATE" "$UPDATE_INTERVAL" "$RUN_TAG"
        JOB_INDEX=$((JOB_INDEX + 1))
        reap_finished
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
    printf 'Grid incomplete; rerun with --batch-id %s --resume.\n' \
        "$BATCH_ID" >&2
    exit 1
fi

printf 'All %s Tent mechanism-validation jobs completed successfully.\n' \
    "$EXPECTED_JOBS"
