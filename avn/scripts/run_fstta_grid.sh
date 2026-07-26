#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the canonical 240-job FSTTA Cartesian grid on SMT+Audio single-source AVN.

Search space:
  fast LR:  1e-8, 1e-7, 3e-7, 1e-6
  fast M:   2, 3, 4, 8, 16
  slow LR:  1e-4, 3e-4, 1e-3, 3e-3
  slow N:   2, 4, 8

The unstable fast learning rates 1e-5 and 6e-5 are intentionally excluded.

Fixed protocol:
  SMT+Audio, single_source, sample actions, seed=0, 2000 episodes
  last_k_ln (K=4), EPISODIC=False, STEPS=1
  Q=0.1, RHO=0.95, TAU=0.7, A=0.9, B=1.1, USE_SLOW=True
  independent fast/slow AdamW, betas=(0.9, 0.99), weight_decay=0

Usage:
  bash avn/scripts/run_fstta_grid.sh [options]

Options:
  --seed N              Episode-order/evaluation seed (default: 0)
  --gpus LIST           Four distinct physical GPU ids (default: 0,1,2,3)
  --jobs-per-gpu N      Concurrent jobs per GPU (default: 4; maximum: 16)
  --episodes N          Episodes per job (default: 2000; maximum: 2000)
  --batch-id ID         Stable batch id (default: UTC timestamp)
  --resume              Resume a batch; skip validated completed jobs
  --allow-dirty         Permit tracked worktree changes (not recommended)
  --dry-run             Print all 240 planned jobs without launching
  -h, --help            Show this help

Recommended detached launch:
  screen -dmS fstta_grid \
    bash avn/scripts/run_fstta_grid.sh \
      --gpus 0,1,2,3 --jobs-per-gpu 4 \
      --batch-id fstta-smt-single-v1-seed0

Follow progress:
  tail -f avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0/scheduler.log

Resume the same immutable batch:
  bash avn/scripts/run_fstta_grid.sh \
    --gpus 0,1,2,3 --jobs-per-gpu 4 \
    --batch-id fstta-smt-single-v1-seed0 --resume
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

SEED=0
GPU_CSV="0,1,2,3"
JOBS_PER_GPU=4
EPISODES=2000
BATCH_ID=""
BATCH_ID_GIVEN=0
RESUME=0
ALLOW_DIRTY=0
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
        --batch-id)
            [[ $# -ge 2 ]] || die "--batch-id requires a value"
            BATCH_ID="$2"
            BATCH_ID_GIVEN=1
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --allow-dirty)
            ALLOW_DIRTY=1
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

[[ "${SEED}" =~ ^[0-9]+$ ]] || die "seed must be a nonnegative integer"
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || die "episodes must be positive"
[[ ${EPISODES} -le 2000 ]] || die "canonical stream contains only 2000 episodes"
[[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || \
    die "jobs-per-gpu must be positive"
[[ ${JOBS_PER_GPU} -le 16 ]] || die "jobs-per-gpu must not exceed 16"
if [[ ${RESUME} -eq 1 && ${BATCH_ID_GIVEN} -eq 0 ]]; then
    die "--resume requires --batch-id"
fi
if [[ -z "${BATCH_ID}" ]]; then
    BATCH_ID="fstta-smt-single-v1-seed${SEED}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "${BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "batch-id may contain only letters, numbers, dot, underscore, and hyphen"

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
[[ ${#GPUS[@]} -eq 4 ]] || die "--gpus must contain exactly four GPU ids"
for gpu in "${GPUS[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU id: ${gpu}"
done
for ((i = 0; i < ${#GPUS[@]}; i++)); do
    for ((j = i + 1; j < ${#GPUS[@]}; j++)); do
        [[ "${GPUS[$i]}" != "${GPUS[$j]}" ]] || \
            die "GPU ids must be distinct"
    done
done

FAST_LRS=("1e-8" "1e-7" "3e-7" "1e-6")
FAST_WINDOWS=(2 3 4 8 16)
SLOW_LRS=("1e-4" "3e-4" "1e-3" "3e-3")
SLOW_WINDOWS=(2 4 8)
EXPECTED_JOBS=$((${#FAST_LRS[@]} * ${#FAST_WINDOWS[@]} * \
    ${#SLOW_LRS[@]} * ${#SLOW_WINDOWS[@]}))
[[ ${EXPECTED_JOBS} -eq 240 ]] || die "internal grid-size error: ${EXPECTED_JOBS}"

# Frozen controls for this grid. Keep them explicit in every invocation so a
# future config-default change cannot silently change the experiment.
NORM_SCOPE="last_k_ln"
LAST_K_LN=4
EPISODIC="False"
STEPS=1
RESET_BN_STATS="True"
Q="0.1"
RHO="0.95"
TAU="0.7"
A="0.9"
B="1.1"
USE_SLOW="True"
OPTIMIZER="AdamW"
BETA1="0.9"
BETA2="0.99"
WEIGHT_DECAY="0.0"
MAX_GRAD_NORM="1.0"
RESET_OPTIMIZER_EACH_EPISODE="True"
EIGEN_EPS="1e-6"
ACTION_SELECTION="sample"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/avn/scripts/eval_smt_audio.sh"
MANIFEST_VALIDATOR="${REPO_ROOT}/tools/validate_run_manifest.py"
CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/smt_audio/single_best_val.pth"
DATASET="${REPO_ROOT}/avn/data/datasets/tta_test/single_source/mp3d/v1/val/val.json.gz"
LOG_ROOT="${REPO_ROOT}/avn/results/logs/fstta_grid/${BATCH_ID}"
JOBS_ROOT="${LOG_ROOT}/jobs"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf 'uncommitted')"

lr_slug() {
    local value="$1"
    value="${value//./p}"
    value="${value//+/p}"
    value="${value//-/m}"
    printf '%s\n' "${value}"
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print $1}'
    else
        shasum -a 256 "${path}" | awk '{print $1}'
    fi
}

job_tag() {
    local job_index="$1"
    local fast_lr="$2"
    local fast_window="$3"
    local slow_lr="$4"
    local slow_window="$5"
    printf 'fsttagrid-%s-j%03d-flr%s-m%s-slr%s-n%s\n' \
        "${BATCH_ID}" "${job_index}" "$(lr_slug "${fast_lr}")" \
        "${fast_window}" "$(lr_slug "${slow_lr}")" "${slow_window}"
}

write_plan() {
    local job_index=0
    local gpu_index
    local gpu
    local fast_lr
    local fast_window
    local slow_lr
    local slow_window
    local tag
    local fi
    local mi
    local si
    local ni

    printf '%s\n' \
        'job_id,run_tag,gpu,model,source_setting,method,action_selection,fast_lr,M,slow_lr,N,seed,episodes'
    for ((fi = 0; fi < ${#FAST_LRS[@]}; fi++)); do
        fast_lr="${FAST_LRS[$fi]}"
        for ((mi = 0; mi < ${#FAST_WINDOWS[@]}; mi++)); do
            fast_window="${FAST_WINDOWS[$mi]}"
            for ((si = 0; si < ${#SLOW_LRS[@]}; si++)); do
                slow_lr="${SLOW_LRS[$si]}"
                for ((ni = 0; ni < ${#SLOW_WINDOWS[@]}; ni++)); do
                    slow_window="${SLOW_WINDOWS[$ni]}"
                    # Mix all four hyperparameter indices into assignment.
                    # This keeps 60 jobs per GPU without permanently coupling
                    # a particular (slow_lr, N) pair to one physical device.
                    gpu_index=$(((fi + mi + si + ni) % ${#GPUS[@]}))
                    gpu="${GPUS[$gpu_index]}"
                    tag="$(job_tag "${job_index}" "${fast_lr}" \
                        "${fast_window}" "${slow_lr}" "${slow_window}")"
                    printf '%s,%s,%s,smt_audio,single_source,fstta,%s,%s,%s,%s,%s,%s,%s\n' \
                        "${job_index}" "${tag}" "${gpu}" \
                        "${ACTION_SELECTION}" "${fast_lr}" "${fast_window}" \
                        "${slow_lr}" "${slow_window}" "${SEED}" "${EPISODES}"
                    job_index=$((job_index + 1))
                done
            done
        done
    done
}

printf 'AVN FSTTA full Cartesian grid\n'
printf '  repository:       %s\n' "${REPO_ROOT}"
printf '  model/source:     smt_audio/single_source\n'
printf '  fast LRs:         1e-8,1e-7,3e-7,1e-6\n'
printf '  M:                2,3,4,8,16\n'
printf '  slow LRs:         1e-4,3e-4,1e-3,3e-3\n'
printf '  N:                2,4,8\n'
printf '  jobs:             %s\n' "${EXPECTED_JOBS}"
printf '  GPUs:             %s (mixed balanced map; 60 jobs/GPU)\n' "${GPU_CSV}"
printf '  jobs/GPU:         %s concurrent\n' "${JOBS_PER_GPU}"
printf '  seed:             %s\n' "${SEED}"
printf '  episodes/job:     %s\n' "${EPISODES}"
printf '  action selection: %s\n' "${ACTION_SELECTION}"
printf '  batch:            %s\n' "${BATCH_ID}"
printf '  logs:             %s\n' "${LOG_ROOT}"

if [[ ${DRY_RUN} -eq 1 ]]; then
    DRY_PLAN="$(mktemp "${TMPDIR:-/tmp}/navtta_fstta_grid.XXXXXX")"
    write_plan > "${DRY_PLAN}"
    cat "${DRY_PLAN}"
    PLAN_JOBS=$(($(wc -l < "${DRY_PLAN}") - 1))
    rm -f "${DRY_PLAN}"
    [[ ${PLAN_JOBS} -eq ${EXPECTED_JOBS} ]] || \
        die "dry-run plan has ${PLAN_JOBS} jobs, expected ${EXPECTED_JOBS}"
    printf '\nDry run complete: %s jobs, nothing launched.\n' "${EXPECTED_JOBS}"
    exit 0
fi

command -v python3 >/dev/null 2>&1 || die "python3 is unavailable"
command -v git >/dev/null 2>&1 || die "git is unavailable"
command -v tee >/dev/null 2>&1 || die "tee is unavailable"
command -v cmp >/dev/null 2>&1 || die "cmp is unavailable"
[[ -f "${RUNNER}" ]] || die "missing evaluation runner: ${RUNNER}"
[[ -f "${MANIFEST_VALIDATOR}" ]] || \
    die "missing run-manifest validator: ${MANIFEST_VALIDATOR}"
[[ -f "${CHECKPOINT}" ]] || die "missing source checkpoint: ${CHECKPOINT}"
[[ -f "${DATASET}" ]] || die "missing single-source TTA dataset: ${DATASET}"

if [[ ${ALLOW_DIRTY} -eq 0 ]]; then
    if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
        die "tracked worktree changes detected; commit them first or use --allow-dirty"
    fi
fi

fingerprints="$(
    python3 "${REPO_ROOT}/avn/scripts/fingerprint_episode_stream.py" \
        --dataset "${DATASET}" --seed "${SEED}"
)"
read -r STREAM_ORDER_SHA256 STREAM_CONTENT_SHA256 <<< "${fingerprints}"
for fingerprint in "${STREAM_ORDER_SHA256}" "${STREAM_CONTENT_SHA256}"; do
    [[ "${fingerprint}" =~ ^[0-9a-f]{64}$ ]] || \
        die "invalid episode-stream fingerprint: ${fingerprint}"
done
CHECKPOINT_SHA256="$(sha256_file "${CHECKPOINT}")"
[[ "${CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
    die "invalid checkpoint SHA256"

validate_job_artifacts() {
    local manifest_path="$1"
    local tag="$2"
    local slow_window="$3"

    [[ -f "${manifest_path}" ]] || return 1
    python3 "${MANIFEST_VALIDATOR}" \
        --manifest "${manifest_path}" \
        --run-tag "${tag}" \
        --model smt_audio \
        --method fstta \
        --source-setting single_source \
        --seed "${SEED}" \
        --git-commit "${GIT_COMMIT}" \
        --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
        --stream-order-sha256 "${STREAM_ORDER_SHA256}" \
        --stream-content-sha256 "${STREAM_CONTENT_SHA256}" >/dev/null || \
        return 1

    python3 - "${manifest_path}" "${SEED}" "${EPISODES}" \
        "${slow_window}" <<'PY'
import json
import os
import sys

manifest_path, seed, expected_text, slow_window_text = sys.argv[1:]
run_dir = os.path.dirname(os.path.abspath(manifest_path))
stats_path = os.path.join(
    run_dir, "raw", "model", "tb", "val_stats_{}.json".format(seed)
)
diagnostics_path = os.path.join(
    run_dir, "raw", "model", "tb", "tta_diagnostics_{}.json".format(seed)
)
with open(stats_path, "r", encoding="utf-8") as handle:
    stats = json.load(handle)
with open(diagnostics_path, "r", encoding="utf-8") as handle:
    diagnostics = json.load(handle)

expected = int(expected_text)
slow_window = int(slow_window_text)
if not isinstance(stats, dict) or len(stats) != expected:
    raise SystemExit(
        "episode statistics count is {}, expected {}".format(
            len(stats) if isinstance(stats, dict) else "non-dict", expected
        )
    )
if diagnostics.get("episodes") != expected:
    raise SystemExit("FSTTA diagnostic episode count mismatch")
if diagnostics.get("slow_optimizer") != "AdamW":
    raise SystemExit("FSTTA slow optimizer is not AdamW")
attempts = expected // slow_window
if diagnostics.get("slow_attempts") != attempts:
    raise SystemExit("FSTTA slow-attempt count mismatch")
if diagnostics.get("slow_pending_episodes") != expected % slow_window:
    raise SystemExit("FSTTA pending slow-window count mismatch")
completed_attempts = (
    diagnostics.get("slow_updates", 0)
    + diagnostics.get("slow_skipped_updates", 0)
)
if completed_attempts != attempts:
    raise SystemExit("FSTTA completed slow-attempt count mismatch")
PY
}

write_batch_spec() {
    printf 'batch_id=%s\n' "${BATCH_ID}"
    printf 'git_commit=%s\n' "${GIT_COMMIT}"
    printf 'model=smt_audio\n'
    printf 'source_setting=single_source\n'
    printf 'method=fstta\n'
    printf 'seed=%s\n' "${SEED}"
    printf 'episodes=%s\n' "${EPISODES}"
    printf 'gpus=%s\n' "${GPU_CSV}"
    printf 'jobs_per_gpu=%s\n' "${JOBS_PER_GPU}"
    printf 'action_selection=%s\n' "${ACTION_SELECTION}"
    printf 'num_processes=1\n'
    printf 'eval_use_ckpt_config=False\n'
    printf 'fast_lrs=1e-8,1e-7,3e-7,1e-6\n'
    printf 'fast_windows=2,3,4,8,16\n'
    printf 'slow_lrs=1e-4,3e-4,1e-3,3e-3\n'
    printf 'slow_windows=2,4,8\n'
    printf 'norm_scope=%s\n' "${NORM_SCOPE}"
    printf 'last_k_ln=%s\n' "${LAST_K_LN}"
    printf 'episodic=%s\n' "${EPISODIC}"
    printf 'steps=%s\n' "${STEPS}"
    printf 'reset_bn_stats=%s\n' "${RESET_BN_STATS}"
    printf 'q=%s\n' "${Q}"
    printf 'rho=%s\n' "${RHO}"
    printf 'tau=%s\n' "${TAU}"
    printf 'a=%s\n' "${A}"
    printf 'b=%s\n' "${B}"
    printf 'use_slow=%s\n' "${USE_SLOW}"
    printf 'optimizer=%s\n' "${OPTIMIZER}"
    printf 'beta1=%s\n' "${BETA1}"
    printf 'beta2=%s\n' "${BETA2}"
    printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
    printf 'max_grad_norm=%s\n' "${MAX_GRAD_NORM}"
    printf 'reset_optimizer_each_episode=%s\n' \
        "${RESET_OPTIMIZER_EACH_EPISODE}"
    printf 'eigen_eps=%s\n' "${EIGEN_EPS}"
    printf 'checkpoint_sha256=%s\n' "${CHECKPOINT_SHA256}"
    printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
    printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
}

if [[ ${RESUME} -eq 0 ]]; then
    [[ ! -e "${LOG_ROOT}" ]] || die "batch already exists: ${LOG_ROOT}"
    mkdir -p "${JOBS_ROOT}"
    write_batch_spec > "${LOG_ROOT}/batch.env"
    write_plan > "${LOG_ROOT}/grid.csv"
else
    [[ -d "${LOG_ROOT}" ]] || die "resume batch does not exist: ${LOG_ROOT}"
    [[ -f "${LOG_ROOT}/batch.env" ]] || die "resume batch is missing batch.env"
    [[ -f "${LOG_ROOT}/grid.csv" ]] || die "resume batch is missing grid.csv"
    RESUME_SPEC="$(mktemp "${TMPDIR:-/tmp}/navtta_fstta_spec.XXXXXX")"
    RESUME_PLAN="$(mktemp "${TMPDIR:-/tmp}/navtta_fstta_plan.XXXXXX")"
    write_batch_spec > "${RESUME_SPEC}"
    write_plan > "${RESUME_PLAN}"
    if ! cmp -s "${LOG_ROOT}/batch.env" "${RESUME_SPEC}"; then
        rm -f "${RESUME_SPEC}" "${RESUME_PLAN}"
        die "resume arguments or immutable inputs do not match batch.env"
    fi
    if ! cmp -s "${LOG_ROOT}/grid.csv" "${RESUME_PLAN}"; then
        rm -f "${RESUME_SPEC}" "${RESUME_PLAN}"
        die "resume arguments do not match grid.csv"
    fi
    rm -f "${RESUME_SPEC}" "${RESUME_PLAN}"
fi

LOCK_DIR="${LOG_ROOT}/.scheduler.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    die "batch is already running or has a stale lock: ${LOCK_DIR}"
fi
printf 'pid=%s\nhost=%s\nstarted_at=%s\n' \
    "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${LOCK_DIR}/owner"

GPU_ACTIVE=(0 0 0 0)
PIDS=()
PID_GPU_INDEX=()
PID_TAG=()
PID_JOB_DIR=()
LAUNCHED=0
SKIPPED=0
REAP_FAILURES=0
KEEP_LOCK=0
LOCK_REASON=""

terminate_process_tree() {
    local parent_pid="$1"
    local child_pid
    if command -v pgrep >/dev/null 2>&1; then
        for child_pid in $(pgrep -P "${parent_pid}" 2>/dev/null || true); do
            terminate_process_tree "${child_pid}"
        done
    fi
    kill -TERM "${parent_pid}" 2>/dev/null || true
}

active_worker_exists() {
    local pid
    for pid in "${PIDS[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

terminate_active_workers() {
    local index
    local pid
    for ((index = 0; index < ${#PIDS[@]}; index++)); do
        pid="${PIDS[$index]}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            terminate_process_tree "${pid}"
        fi
    done
    for ((index = 0; index < ${#PIDS[@]}; index++)); do
        pid="${PIDS[$index]}"
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
            PIDS[$index]=""
        fi
    done
}

cleanup_lock() {
    local scheduler_status=$?
    if [[ ${KEEP_LOCK} -eq 0 ]] && active_worker_exists; then
        KEEP_LOCK=1
        LOCK_REASON="unexpected_scheduler_exit_workers_terminated"
        terminate_active_workers
    fi
    if [[ ${KEEP_LOCK} -eq 1 ]]; then
        printf 'status=%s\nscheduler_exit_code=%s\n' \
            "${LOCK_REASON:-interrupted}" "${scheduler_status}" \
            >> "${LOCK_DIR}/owner"
        return
    fi
    rm -f "${LOCK_DIR}/owner"
    rmdir "${LOCK_DIR}" 2>/dev/null || true
}

handle_signal() {
    local status="$1"
    local reason="$2"
    KEEP_LOCK=1
    LOCK_REASON="${reason}"
    trap - HUP INT TERM
    terminate_active_workers
    printf 'scheduler interrupted; lock retained at %s\n' "${LOCK_DIR}" >&2
    exit "${status}"
}

trap cleanup_lock EXIT
trap 'handle_signal 129 signal_hup' HUP
trap 'handle_signal 130 signal_int' INT
trap 'handle_signal 143 signal_term' TERM

SCHEDULER_LOG="${LOG_ROOT}/scheduler.log"
exec > >(tee -a "${SCHEDULER_LOG}") 2>&1

printf 'scheduler_started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'git_commit=%s\n' "${GIT_COMMIT}"
printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
printf 'resume=%s\n' "${RESUME}"

total_active() {
    local total=0
    local count
    for count in "${GPU_ACTIVE[@]}"; do
        total=$((total + count))
    done
    printf '%s\n' "${total}"
}

reap_finished() {
    local index
    local pid
    local gpu_index
    local tag
    local job_dir
    local status

    for ((index = 0; index < ${#PIDS[@]}; index++)); do
        pid="${PIDS[$index]}"
        [[ -n "${pid}" ]] || continue
        job_dir="${PID_JOB_DIR[$index]}"
        if [[ -f "${job_dir}/exitcode" ]] || ! kill -0 "${pid}" 2>/dev/null; then
            if wait "${pid}"; then
                status=0
            else
                status=$?
            fi
            gpu_index="${PID_GPU_INDEX[$index]}"
            tag="${PID_TAG[$index]}"
            GPU_ACTIVE[$gpu_index]=$((GPU_ACTIVE[$gpu_index] - 1))
            PIDS[$index]=""
            if [[ ${status} -ne 0 ]]; then
                REAP_FAILURES=$((REAP_FAILURES + 1))
            fi
            printf '%s finished tag=%s status=%s active=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${tag}" "${status}" \
                "$(total_active)"
        fi
    done
}

wait_for_gpu_slot() {
    local gpu_index="$1"
    while [[ ${GPU_ACTIVE[$gpu_index]} -ge ${JOBS_PER_GPU} ]]; do
        reap_finished
        if [[ ${GPU_ACTIVE[$gpu_index]} -ge ${JOBS_PER_GPU} ]]; then
            sleep 5
        fi
    done
}

launch_job() {
    local job_index="$1"
    local gpu_index="$2"
    local gpu="$3"
    local fast_lr="$4"
    local fast_window="$5"
    local slow_lr="$6"
    local slow_window="$7"
    local tag="$8"
    local job_dir="${JOBS_ROOT}/${tag}"
    local previous_status=""
    local previous_validation=""
    local previous_manifest=""
    local attempt
    local pid

    mkdir -p "${job_dir}"
    if [[ -f "${job_dir}/exitcode" ]]; then
        previous_status="$(tr -d '[:space:]' < "${job_dir}/exitcode")"
        if [[ -f "${job_dir}/validation" ]]; then
            previous_validation="$(
                tr -d '[:space:]' < "${job_dir}/validation"
            )"
        fi
        if [[ -f "${job_dir}/manifest.path" ]]; then
            previous_manifest="$(
                tr -d '[:space:]' < "${job_dir}/manifest.path"
            )"
        fi
        if [[ ${RESUME} -eq 1 && "${previous_status}" == "0" && \
                "${previous_validation}" == "ok" && \
                -n "${previous_manifest}" ]] && \
                validate_job_artifacts "${previous_manifest}" "${tag}" \
                    "${slow_window}" >/dev/null 2>&1; then
                SKIPPED=$((SKIPPED + 1))
                printf '%s skip validated tag=%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${tag}"
                return
        fi
    fi
    if [[ -f "${job_dir}/exitcode" || -f "${job_dir}/runner_exitcode" || \
            -f "${job_dir}/validation" || -f "${job_dir}/console.log" || \
            -f "${job_dir}/manifest.path" ]]; then
        attempt="$(date -u +%Y%m%dT%H%M%SZ)"
        [[ ! -f "${job_dir}/exitcode" ]] || \
            mv "${job_dir}/exitcode" \
                "${job_dir}/exitcode.previous.${attempt}"
        [[ ! -f "${job_dir}/runner_exitcode" ]] || \
            mv "${job_dir}/runner_exitcode" \
                "${job_dir}/runner_exitcode.previous.${attempt}"
        [[ ! -f "${job_dir}/validation" ]] || \
            mv "${job_dir}/validation" \
                "${job_dir}/validation.previous.${attempt}"
        [[ ! -f "${job_dir}/console.log" ]] || \
            mv "${job_dir}/console.log" \
                "${job_dir}/console.previous.${attempt}.log"
        [[ ! -f "${job_dir}/manifest.path" ]] || \
            mv "${job_dir}/manifest.path" \
                "${job_dir}/manifest.previous.${attempt}.path"
    fi

    wait_for_gpu_slot "${gpu_index}"
    {
        printf 'job_id=%s\n' "${job_index}"
        printf 'run_tag=%s\n' "${tag}"
        printf 'gpu=%s\n' "${gpu}"
        printf 'model=smt_audio\n'
        printf 'source_setting=single_source\n'
        printf 'method=fstta\n'
        printf 'action_selection=%s\n' "${ACTION_SELECTION}"
        printf 'fast_lr=%s\n' "${fast_lr}"
        printf 'M=%s\n' "${fast_window}"
        printf 'slow_lr=%s\n' "${slow_lr}"
        printf 'N=%s\n' "${slow_window}"
        printf 'seed=%s\n' "${SEED}"
        printf 'episodes=%s\n' "${EPISODES}"
        printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
        printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
    } > "${job_dir}/parameters.env"

    (
        set +e
        printf '\n===== attempt %s =====\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${job_dir}/console.log"
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export PYTHONUNBUFFERED=1
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NAVTTA_RUN_TAG="${tag}"
        export NAVTTA_STREAM_ORDER_SHA256="${STREAM_ORDER_SHA256}"
        export NAVTTA_STREAM_CONTENT_SHA256="${STREAM_CONTENT_SHA256}"
        bash "${RUNNER}" single_source fstta "${SEED}" \
            NUM_PROCESSES 1 \
            EVAL.USE_CKPT_CONFIG False \
            EVAL.ACTION_SELECTION "${ACTION_SELECTION}" \
            TTA.LR "${fast_lr}" \
            TTA.NORM_SCOPE "${NORM_SCOPE}" \
            TTA.LAST_K_LN "${LAST_K_LN}" \
            TTA.EPISODIC "${EPISODIC}" \
            TTA.STEPS "${STEPS}" \
            TTA.RESET_BN_STATS "${RESET_BN_STATS}" \
            TTA.MAX_GRAD_NORM "${MAX_GRAD_NORM}" \
            TTA.FSTTA.M "${fast_window}" \
            TTA.FSTTA.N "${slow_window}" \
            TTA.FSTTA.Q "${Q}" \
            TTA.FSTTA.LR_SLOW "${slow_lr}" \
            TTA.FSTTA.RHO "${RHO}" \
            TTA.FSTTA.TAU "${TAU}" \
            TTA.FSTTA.A "${A}" \
            TTA.FSTTA.B "${B}" \
            TTA.FSTTA.USE_SLOW "${USE_SLOW}" \
            TTA.FSTTA.OPTIMIZER "${OPTIMIZER}" \
            TTA.FSTTA.BETA1 "${BETA1}" \
            TTA.FSTTA.BETA2 "${BETA2}" \
            TTA.FSTTA.WEIGHT_DECAY "${WEIGHT_DECAY}" \
            TTA.FSTTA.RESET_OPTIMIZER_EACH_EPISODE \
                "${RESET_OPTIMIZER_EACH_EPISODE}" \
            TTA.FSTTA.EIGEN_EPS "${EIGEN_EPS}" \
            TEST_EPISODE_COUNT "${EPISODES}" \
            >> "${job_dir}/console.log" 2>&1
        status=$?
        composite_status="${status}"

        manifest_path="$(
            tr '\r' '\n' < "${job_dir}/console.log" | \
                sed -n '/\/manifest[.]json$/p' | tail -n 1
        )"
        if [[ -n "${manifest_path}" && -f "${manifest_path}" ]]; then
            printf '%s\n' "${manifest_path}" > "${job_dir}/manifest.path.tmp"
            mv "${job_dir}/manifest.path.tmp" "${job_dir}/manifest.path"
        fi

        validation="failed"
        if [[ ${status} -eq 0 && -n "${manifest_path}" ]] && \
                validate_job_artifacts "${manifest_path}" "${tag}" \
                    "${slow_window}" >> "${job_dir}/console.log" 2>&1; then
            validation="ok"
        elif [[ ${status} -eq 0 ]]; then
            composite_status=90
            printf 'artifact validation failed\n' >> "${job_dir}/console.log"
        fi

        printf '%s\n' "${status}" > "${job_dir}/runner_exitcode.tmp"
        mv "${job_dir}/runner_exitcode.tmp" "${job_dir}/runner_exitcode"
        printf '%s\n' "${validation}" > "${job_dir}/validation.tmp"
        mv "${job_dir}/validation.tmp" "${job_dir}/validation"
        printf '%s\n' "${composite_status}" > "${job_dir}/exitcode.tmp"
        mv "${job_dir}/exitcode.tmp" "${job_dir}/exitcode"
        exit "${composite_status}"
    ) &
    pid=$!

    PIDS+=("${pid}")
    PID_GPU_INDEX+=("${gpu_index}")
    PID_TAG+=("${tag}")
    PID_JOB_DIR+=("${job_dir}")
    GPU_ACTIVE[$gpu_index]=$((GPU_ACTIVE[$gpu_index] + 1))
    LAUNCHED=$((LAUNCHED + 1))
    printf '%s launched job=%s tag=%s gpu=%s active_on_gpu=%s total_active=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${job_index}" "${tag}" \
        "${gpu}" "${GPU_ACTIVE[$gpu_index]}" "$(total_active)"
}

JOB_INDEX=0
for ((FI = 0; FI < ${#FAST_LRS[@]}; FI++)); do
    FAST_LR="${FAST_LRS[$FI]}"
    for ((MI = 0; MI < ${#FAST_WINDOWS[@]}; MI++)); do
        FAST_WINDOW="${FAST_WINDOWS[$MI]}"
        for ((SI = 0; SI < ${#SLOW_LRS[@]}; SI++)); do
            SLOW_LR="${SLOW_LRS[$SI]}"
            for ((NI = 0; NI < ${#SLOW_WINDOWS[@]}; NI++)); do
                SLOW_WINDOW="${SLOW_WINDOWS[$NI]}"
                GPU_INDEX=$(((FI + MI + SI + NI) % ${#GPUS[@]}))
                GPU="${GPUS[$GPU_INDEX]}"
                RUN_TAG="$(job_tag "${JOB_INDEX}" "${FAST_LR}" \
                    "${FAST_WINDOW}" "${SLOW_LR}" "${SLOW_WINDOW}")"
                launch_job "${JOB_INDEX}" "${GPU_INDEX}" "${GPU}" \
                    "${FAST_LR}" "${FAST_WINDOW}" "${SLOW_LR}" \
                    "${SLOW_WINDOW}" "${RUN_TAG}"
                JOB_INDEX=$((JOB_INDEX + 1))
                reap_finished
            done
        done
    done
done

while [[ $(total_active) -gt 0 ]]; do
    reap_finished
    [[ $(total_active) -eq 0 ]] || sleep 5
done

EXIT_FILES=$(find "${JOBS_ROOT}" -type f -name exitcode | wc -l | tr -d ' ')
SUCCESSFUL=0
FAILED=0
while IFS= read -r exit_file; do
    status="$(tr -d '[:space:]' < "${exit_file}")"
    if [[ "${status}" == "0" ]]; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done < <(find "${JOBS_ROOT}" -type f -name exitcode | sort)
MISSING=$((EXPECTED_JOBS - EXIT_FILES))
MANIFEST_POINTERS=$(find "${JOBS_ROOT}" -type f -name manifest.path | wc -l | tr -d ' ')
VALIDATED=0
while IFS= read -r validation_file; do
    validation_status="$(tr -d '[:space:]' < "${validation_file}")"
    if [[ "${validation_status}" == "ok" ]]; then
        VALIDATED=$((VALIDATED + 1))
    fi
done < <(find "${JOBS_ROOT}" -type f -name validation | sort)

{
    printf 'batch_id=%s\n' "${BATCH_ID}"
    printf 'git_commit=%s\n' "${GIT_COMMIT}"
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'expected=%s\n' "${EXPECTED_JOBS}"
    printf 'launched_this_invocation=%s\n' "${LAUNCHED}"
    printf 'skipped_completed=%s\n' "${SKIPPED}"
    printf 'successful=%s\n' "${SUCCESSFUL}"
    printf 'failed=%s\n' "${FAILED}"
    printf 'missing=%s\n' "${MISSING}"
    printf 'manifest_pointers=%s\n' "${MANIFEST_POINTERS}"
    printf 'validated=%s\n' "${VALIDATED}"
    printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
    printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
} | tee "${LOG_ROOT}/SUMMARY"

if [[ ${REAP_FAILURES} -ne 0 || ${FAILED} -ne 0 || ${MISSING} -ne 0 || \
        ${MANIFEST_POINTERS} -ne ${EXPECTED_JOBS} || \
        ${VALIDATED} -ne ${EXPECTED_JOBS} ]]; then
    printf 'Grid incomplete; inspect logs and resume with --batch-id %s --resume.\n' \
        "${BATCH_ID}" >&2
    exit 1
fi

printf 'All %s SMT+Audio single-source FSTTA jobs completed successfully.\n' \
    "${EXPECTED_JOBS}"
