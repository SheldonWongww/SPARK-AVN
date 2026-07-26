#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the two canonical multi-source Tent jobs used by the AVN main table.

Fixed jobs:
  GPU 0: SMT+Audio, multi_source
  GPU 1: ENMuS,     multi_source

Fixed Tent configuration:
  NORM_SCOPE=ln, LR=1e-8, UPDATE_INTERVAL=1, EPISODIC=False, STEPS=1

Usage:
  bash avn/scripts/run_tent_main_multi_source.sh [options]

Options:
  --seed N          Fixed episode-order/evaluation seed (default: 0)
  --gpus LIST       SMT+Audio,ENMuS physical GPU ids (default: 0,1)
  --episodes N      Episodes per job (default: 2000; maximum: 2000)
  --batch-id ID     Stable batch id (default: UTC timestamp)
  --resume          Resume an explicit batch id; skip exit-code-0 jobs
  --allow-dirty     Permit tracked worktree changes (not for formal results)
  --dry-run         Print the exact two-job plan without launching
  -h, --help        Show this help

Recommended detached launch:
  screen -dmS tent_main_multi \
    bash avn/scripts/run_tent_main_multi_source.sh \
      --gpus 0,1 --seed 0 --batch-id tent-main-multi-v1-seed0

Attach:
  screen -d -r tent_main_multi

Follow progress from another terminal:
  tail -f avn/results/logs/tent_main_multi/tent-main-multi-v1-seed0/scheduler.log

This script intentionally does not rerun or modify the selected single-source
Tent runs retained in the earlier hyperparameter-search batches.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

SEED=0
GPU_CSV="0,1"
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
if [[ ${RESUME} -eq 1 && ${BATCH_ID_GIVEN} -eq 0 ]]; then
    die "--resume requires --batch-id"
fi
if [[ -z "${BATCH_ID}" ]]; then
    BATCH_ID="tent-main-multi-seed${SEED}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "${BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid batch id"

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
[[ ${#GPUS[@]} -eq 2 ]] || die "--gpus must contain exactly two GPU ids"
for gpu in "${GPUS[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU id: ${gpu}"
done
[[ "${GPUS[0]}" != "${GPUS[1]}" ]] || die "GPU ids must be distinct"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="${REPO_ROOT}/avn/results/logs/tent_main_multi/${BATCH_ID}"
JOBS_ROOT="${LOG_ROOT}/jobs"
DATASET="${REPO_ROOT}/avn/data/datasets/tta_test/multi_source/mp3d/v1/val/val.json.gz"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf 'uncommitted')"
MODELS=("smt_audio" "enmus")
EXPECTED_JOBS=2

# These values are the frozen AVN main-table Tent configuration.
TENT_LR="1e-8"
TENT_NORM_SCOPE="ln"
TENT_UPDATE_INTERVAL=1
TENT_EPISODIC="False"
TENT_STEPS=1

runner_for_model() {
    case "$1" in
        smt_audio) printf '%s\n' "${REPO_ROOT}/avn/scripts/eval_smt_audio.sh" ;;
        enmus) printf '%s\n' "${REPO_ROOT}/avn/scripts/eval_enmus.sh" ;;
        *) die "unknown model: $1" ;;
    esac
}

checkpoint_for_model() {
    case "$1" in
        smt_audio)
            printf '%s\n' \
                "${REPO_ROOT}/avn/checkpoints/source/smt_audio/multi_best_val.pth"
            ;;
        enmus)
            printf '%s\n' \
                "${REPO_ROOT}/avn/checkpoints/source/enmus/multi_source_best_val.pth"
            ;;
        *) die "unknown model: $1" ;;
    esac
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print $1}'
    else
        shasum -a 256 "${path}" | awk '{print $1}'
    fi
}

run_tag_for_model() {
    local model="$1"
    local job_index
    case "${model}" in
        smt_audio) job_index=0 ;;
        enmus) job_index=1 ;;
        *) die "unknown model: ${model}" ;;
    esac
    printf '%s-j%02d-%s-multi_source\n' \
        "${BATCH_ID}" "${job_index}" "${model}"
}

print_command() {
    local model="$1"
    local gpu="$2"
    local runner="$3"
    local tag="$4"
    printf '  CUDA_VISIBLE_DEVICES=%s NAVTTA_RUN_TAG=%s bash %s multi_source tent %s' \
        "${gpu}" "${tag}" "${runner}" "${SEED}"
    printf ' TTA.LR %s TTA.NORM_SCOPE %s TTA.LAST_K_LN 4' \
        "${TENT_LR}" "${TENT_NORM_SCOPE}"
    printf ' TTA.EPISODIC %s TTA.STEPS %s TTA.UPDATE_INTERVAL %s' \
        "${TENT_EPISODIC}" "${TENT_STEPS}" "${TENT_UPDATE_INTERVAL}"
    printf ' TTA.MAX_UPDATES_PER_EPISODE -1 TTA.OPTIMIZER Adam'
    printf ' TTA.BETA1 0.9 TTA.BETA2 0.999 TTA.WEIGHT_DECAY 0.0'
    printf ' TTA.MAX_GRAD_NORM 1.0 TEST_EPISODE_COUNT %s' "${EPISODES}"
    if [[ "${model}" == "smt_audio" ]]; then
        printf ' EVAL.ACTION_SELECTION sample'
    fi
    printf '\n'
}

printf 'AVN Tent main-table multi-source evaluation\n'
printf '  repository:       %s\n' "${REPO_ROOT}"
printf '  batch id:         %s\n' "${BATCH_ID}"
printf '  seed/order:       %s\n' "${SEED}"
printf '  episodes/job:     %s\n' "${EPISODES}"
printf '  models:           SMT+Audio, ENMuS\n'
printf '  source setting:   multi_source\n'
printf '  Tent:             ln, lr=1e-8, interval=1, episodic=False, steps=1\n'
printf '  GPU/model map:    %s=SMT+Audio, %s=ENMuS\n' "${GPUS[0]}" "${GPUS[1]}"
printf '  Git commit:       %s\n' "${GIT_COMMIT}"
printf '  logs:             %s\n' "${LOG_ROOT}"
if [[ ${EPISODES} -ne 2000 ]]; then
    printf '  warning:          smoke run only; not valid for the main table\n'
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
    for ((i = 0; i < ${#MODELS[@]}; i++)); do
        model="${MODELS[$i]}"
        runner="$(runner_for_model "${model}")"
        tag="$(run_tag_for_model "${model}")"
        printf '  job=%02d gpu=%s model=%s\n' "${i}" "${GPUS[$i]}" "${model}"
        print_command "${model}" "${GPUS[$i]}" "${runner}" "${tag}"
    done
    printf 'Dry run only; no files created and no jobs launched.\n'
    exit 0
fi

command -v python3 >/dev/null 2>&1 || die "python3 is unavailable"
command -v git >/dev/null 2>&1 || die "git is unavailable"
command -v tee >/dev/null 2>&1 || die "tee is unavailable"
[[ -f "${DATASET}" ]] || die "missing multi-source TTA dataset: ${DATASET}"

for model in "${MODELS[@]}"; do
    runner="$(runner_for_model "${model}")"
    checkpoint="$(checkpoint_for_model "${model}")"
    [[ -f "${runner}" ]] || die "missing evaluation runner: ${runner}"
    [[ -f "${checkpoint}" ]] || die "missing source checkpoint: ${checkpoint}"
done
for auxiliary in \
    "${REPO_ROOT}/avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/audio_encoder_best_val.pth" \
    "${REPO_ROOT}/avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/visual_encoder_best_val.pth" \
    "${REPO_ROOT}/avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/seld_crnn_best_val.h5"; do
    [[ -f "${auxiliary}" ]] || die "missing ENMuS auxiliary checkpoint: ${auxiliary}"
done

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
SMT_AUDIO_CHECKPOINT_SHA256="$(sha256_file "$(checkpoint_for_model smt_audio)")"
ENMUS_CHECKPOINT_SHA256="$(sha256_file "$(checkpoint_for_model enmus)")"

checkpoint_sha_for_model() {
    case "$1" in
        smt_audio) printf '%s\n' "${SMT_AUDIO_CHECKPOINT_SHA256}" ;;
        enmus) printf '%s\n' "${ENMUS_CHECKPOINT_SHA256}" ;;
        *) die "unknown model: $1" ;;
    esac
}

write_batch_spec() {
    printf 'batch_id=%s\n' "${BATCH_ID}"
    printf 'git_commit=%s\n' "${GIT_COMMIT}"
    printf 'seed=%s\n' "${SEED}"
    printf 'episodes=%s\n' "${EPISODES}"
    printf 'gpus=%s\n' "${GPU_CSV}"
    printf 'source_setting=multi_source\n'
    printf 'method=tent\n'
    printf 'norm_scope=%s\n' "${TENT_NORM_SCOPE}"
    printf 'lr=%s\n' "${TENT_LR}"
    printf 'update_interval=%s\n' "${TENT_UPDATE_INTERVAL}"
    printf 'episodic=%s\n' "${TENT_EPISODIC}"
    printf 'steps=%s\n' "${TENT_STEPS}"
    printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
    printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
    printf 'smt_audio_checkpoint_sha256=%s\n' "${SMT_AUDIO_CHECKPOINT_SHA256}"
    printf 'enmus_checkpoint_sha256=%s\n' "${ENMUS_CHECKPOINT_SHA256}"
}

write_plan() {
    local model
    local tag
    local checkpoint_sha
    printf 'job_id,run_tag,gpu,model,source_setting,method,seed,episodes,checkpoint_sha256,stream_order_sha256,stream_content_sha256\n'
    for ((i = 0; i < ${#MODELS[@]}; i++)); do
        model="${MODELS[$i]}"
        tag="$(run_tag_for_model "${model}")"
        checkpoint_sha="$(checkpoint_sha_for_model "${model}")"
        printf '%s,%s,%s,%s,multi_source,tent,%s,%s,%s,%s,%s\n' \
            "${i}" "${tag}" "${GPUS[$i]}" "${model}" "${SEED}" \
            "${EPISODES}" "${checkpoint_sha}" "${STREAM_ORDER_SHA256}" \
            "${STREAM_CONTENT_SHA256}"
    done
}

if [[ ${RESUME} -eq 0 ]]; then
    [[ ! -e "${LOG_ROOT}" ]] || die "batch already exists: ${LOG_ROOT}"
    mkdir -p "${JOBS_ROOT}"
    write_batch_spec > "${LOG_ROOT}/batch.env"
    write_plan > "${LOG_ROOT}/plan.csv"
else
    [[ -d "${LOG_ROOT}" ]] || die "resume batch does not exist: ${LOG_ROOT}"
    [[ -f "${LOG_ROOT}/batch.env" ]] || die "resume batch is missing batch.env"
    [[ -f "${LOG_ROOT}/plan.csv" ]] || die "resume batch is missing plan.csv"
    [[ "$(write_batch_spec)" == "$(cat "${LOG_ROOT}/batch.env")" ]] || \
        die "resume arguments or immutable inputs do not match batch.env"
    [[ "$(write_plan)" == "$(cat "${LOG_ROOT}/plan.csv")" ]] || \
        die "resume plan does not match plan.csv"
fi

LOCK_DIR="${LOG_ROOT}/.scheduler.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    die "batch is already running or has a stale lock: ${LOCK_DIR}"
fi
printf 'pid=%s\nhost=%s\nstarted_at=%s\n' \
    "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${LOCK_DIR}/owner"

PIDS=()
PID_MODELS=()

cleanup() {
    local i
    local pid
    for ((i = 0; i < ${#PIDS[@]}; i++)); do
        pid="${PIDS[$i]}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
    rm -f "${LOCK_DIR}/owner"
    rmdir "${LOCK_DIR}" 2>/dev/null || true
}

handle_signal() {
    local status="$1"
    trap - HUP INT TERM
    exit "${status}"
}

trap cleanup EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

SCHEDULER_LOG="${LOG_ROOT}/scheduler.log"
exec > >(tee -a "${SCHEDULER_LOG}") 2>&1

printf 'scheduler_started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
printf 'resume=%s\n' "${RESUME}"

launch_job() {
    local model="$1"
    local gpu="$2"
    local runner="$3"
    local tag="$4"
    local job_dir="${JOBS_ROOT}/${model}"
    local previous_status=""
    local attempt
    local pid

    mkdir -p "${job_dir}"
    if [[ -f "${job_dir}/exitcode" ]]; then
        previous_status="$(tr -d '[:space:]' < "${job_dir}/exitcode")"
        if [[ ${RESUME} -eq 1 && "${previous_status}" == "0" ]]; then
            printf '%s skip completed model=%s tag=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${model}" "${tag}"
            return
        fi
        attempt="$(date -u +%Y%m%dT%H%M%SZ)"
        mv "${job_dir}/exitcode" "${job_dir}/exitcode.previous.${attempt}"
    fi
    attempt="$(date -u +%Y%m%dT%H%M%SZ)"
    if [[ -f "${job_dir}/console.log" ]]; then
        mv "${job_dir}/console.log" \
            "${job_dir}/console.previous.${attempt}.log"
    fi
    if [[ -f "${job_dir}/validation" ]]; then
        mv "${job_dir}/validation" \
            "${job_dir}/validation.previous.${attempt}"
    fi

    {
        printf 'model=%s\n' "${model}"
        printf 'gpu=%s\n' "${gpu}"
        printf 'run_tag=%s\n' "${tag}"
        printf 'source_setting=multi_source\n'
        printf 'method=tent\n'
        printf 'seed=%s\n' "${SEED}"
        printf 'episodes=%s\n' "${EPISODES}"
        printf 'norm_scope=%s\n' "${TENT_NORM_SCOPE}"
        printf 'lr=%s\n' "${TENT_LR}"
        printf 'update_interval=%s\n' "${TENT_UPDATE_INTERVAL}"
        printf 'episodic=%s\n' "${TENT_EPISODIC}"
        printf 'steps=%s\n' "${TENT_STEPS}"
    } > "${job_dir}/parameters.env"

    (
        set +e
        printf '\n===== attempt %s =====\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${job_dir}/console.log"
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export PYTHONUNBUFFERED=1
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NAVTTA_RUN_TAG="${tag}"
        export NAVTTA_STREAM_ORDER_SHA256="${STREAM_ORDER_SHA256}"
        export NAVTTA_STREAM_CONTENT_SHA256="${STREAM_CONTENT_SHA256}"
        if [[ "${model}" == "smt_audio" ]]; then
            bash "${runner}" multi_source tent "${SEED}" \
                TTA.LR "${TENT_LR}" \
                TTA.NORM_SCOPE "${TENT_NORM_SCOPE}" \
                TTA.LAST_K_LN 4 \
                TTA.EPISODIC "${TENT_EPISODIC}" \
                TTA.STEPS "${TENT_STEPS}" \
                TTA.UPDATE_INTERVAL "${TENT_UPDATE_INTERVAL}" \
                TTA.MAX_UPDATES_PER_EPISODE -1 \
                TTA.OPTIMIZER Adam \
                TTA.BETA1 0.9 \
                TTA.BETA2 0.999 \
                TTA.WEIGHT_DECAY 0.0 \
                TTA.MAX_GRAD_NORM 1.0 \
                TEST_EPISODE_COUNT "${EPISODES}" \
                EVAL.ACTION_SELECTION sample \
                >> "${job_dir}/console.log" 2>&1
        else
            bash "${runner}" multi_source tent "${SEED}" \
                TTA.LR "${TENT_LR}" \
                TTA.NORM_SCOPE "${TENT_NORM_SCOPE}" \
                TTA.LAST_K_LN 4 \
                TTA.EPISODIC "${TENT_EPISODIC}" \
                TTA.STEPS "${TENT_STEPS}" \
                TTA.UPDATE_INTERVAL "${TENT_UPDATE_INTERVAL}" \
                TTA.MAX_UPDATES_PER_EPISODE -1 \
                TTA.OPTIMIZER Adam \
                TTA.BETA1 0.9 \
                TTA.BETA2 0.999 \
                TTA.WEIGHT_DECAY 0.0 \
                TTA.MAX_GRAD_NORM 1.0 \
                TEST_EPISODE_COUNT "${EPISODES}" \
                >> "${job_dir}/console.log" 2>&1
        fi
        status=$?
        printf '%s\n' "${status}" > "${job_dir}/exitcode.tmp"
        mv "${job_dir}/exitcode.tmp" "${job_dir}/exitcode"
        exit "${status}"
    ) &
    pid=$!
    PIDS+=("${pid}")
    PID_MODELS+=("${model}")
    printf '%s launched model=%s gpu=%s pid=%s tag=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${model}" "${gpu}" "${pid}" "${tag}"
}

for ((i = 0; i < ${#MODELS[@]}; i++)); do
    model="${MODELS[$i]}"
    job_dir="${JOBS_ROOT}/${model}"
    if [[ ${RESUME} -eq 1 && -f "${job_dir}/exitcode" ]] && \
            [[ "$(tr -d '[:space:]' < "${job_dir}/exitcode")" == "0" ]]; then
        printf '%s skip completed model=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${model}"
        continue
    fi
    launch_job "${model}" "${GPUS[$i]}" \
        "$(runner_for_model "${model}")" "$(run_tag_for_model "${model}")"
done

FAILED_WAITS=0
for ((i = 0; i < ${#PIDS[@]}; i++)); do
    pid="${PIDS[$i]}"
    if wait "${pid}"; then
        printf '%s completed model=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PID_MODELS[$i]}"
    else
        FAILED_WAITS=$((FAILED_WAITS + 1))
        printf '%s failed model=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PID_MODELS[$i]}"
    fi
    PIDS[$i]=""
done

manifest_from_log() {
    local log_file="$1"
    tr '\r' '\n' < "${log_file}" | \
        sed -n '/\/manifest\.json$/p' | tail -n 1
}

extract_metric() {
    local metric="$1"
    local log_file="$2"
    tr '\r' '\n' < "${log_file}" | \
        sed -n "s/.*Average episode ${metric}:[[:space:]]*//p" | tail -n 1
}

is_number() {
    [[ "$1" =~ ^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]
}

write_summary_json() {
    local output="$1"
    local model="$2"
    local reward="$3"
    local dtg="$4"
    local ndtg="$5"
    local success="$6"
    local spl="$7"
    local softspl="$8"
    local na="$9"
    local sna="${10}"
    local sws="${11}"
    local canonical="false"
    [[ ${EPISODES} -eq 2000 ]] && canonical="true"
    {
        printf '{\n'
        printf '  "model": "%s",\n' "${model}"
        printf '  "method": "tent",\n'
        printf '  "source_setting": "multi_source",\n'
        printf '  "seed": %s,\n' "${SEED}"
        printf '  "episode_count": %s,\n' "${EPISODES}"
        printf '  "canonical_main_table_configuration": %s,\n' "${canonical}"
        printf '  "tta": {"norm_scope": "ln", "lr": 1e-8, "update_interval": 1, "episodic": false, "steps": 1},\n'
        printf '  "metrics": {\n'
        printf '    "reward": %s,\n' "${reward}"
        printf '    "distance_to_goal": %s,\n' "${dtg}"
        printf '    "normalized_distance_to_goal": %s,\n' "${ndtg}"
        printf '    "success": %s,\n' "${success}"
        printf '    "spl": %s,\n' "${spl}"
        printf '    "softspl": %s,\n' "${softspl}"
        printf '    "na": %s,\n' "${na}"
        printf '    "sna": %s,\n' "${sna}"
        printf '    "sws": %s\n' "${sws}"
        printf '  }\n'
        printf '}\n'
    } > "${output}.tmp"
    mv "${output}.tmp" "${output}"
}

METRICS_FILE="${LOG_ROOT}/metrics.csv"
printf 'model,source_setting,method,seed,episodes,exitcode,manifest,reward,distance_to_goal,normalized_distance_to_goal,success,spl,softspl,na,sna,sws\n' \
    > "${METRICS_FILE}"

SUCCEEDED=0
FAILED=0
for model in "${MODELS[@]}"; do
    job_dir="${JOBS_ROOT}/${model}"
    log_file="${job_dir}/console.log"
    status="missing"
    manifest=""
    manifest_record=""
    reward=""
    dtg=""
    ndtg=""
    success=""
    spl=""
    softspl=""
    na=""
    sna=""
    sws=""

    if [[ -f "${job_dir}/exitcode" ]]; then
        status="$(tr -d '[:space:]' < "${job_dir}/exitcode")"
    fi
    if [[ "${status}" == "0" && -f "${log_file}" ]]; then
        manifest="$(manifest_from_log "${log_file}")"
        reward="$(extract_metric reward "${log_file}")"
        dtg="$(extract_metric distance_to_goal "${log_file}")"
        ndtg="$(extract_metric normalized_distance_to_goal "${log_file}")"
        success="$(extract_metric success "${log_file}")"
        spl="$(extract_metric spl "${log_file}")"
        softspl="$(extract_metric softspl "${log_file}")"
        na="$(extract_metric na "${log_file}")"
        sna="$(extract_metric sna "${log_file}")"
        sws="$(extract_metric sws "${log_file}")"

        if [[ "${manifest}" != "${REPO_ROOT}"/avn/results/runs/*/manifest.json ]] || \
                [[ ! -f "${manifest}" ]]; then
            status="manifest_missing"
        elif ! python3 "${REPO_ROOT}/tools/validate_run_manifest.py" \
                --manifest "${manifest}" \
                --run-tag "$(run_tag_for_model "${model}")" \
                --model "${model}" \
                --method tent \
                --source-setting multi_source \
                --seed "${SEED}" \
                --git-commit "${GIT_COMMIT}" \
                --checkpoint-sha256 "$(checkpoint_sha_for_model "${model}")" \
                --stream-order-sha256 "${STREAM_ORDER_SHA256}" \
                --stream-content-sha256 "${STREAM_CONTENT_SHA256}" \
                >/dev/null; then
            status="manifest_invalid"
        else
            for value in "${reward}" "${dtg}" "${ndtg}" "${success}" \
                    "${spl}" "${softspl}" "${na}" "${sna}" "${sws}"; do
                if ! is_number "${value}"; then
                    status="metrics_missing"
                    break
                fi
            done
        fi
    fi

    if [[ "${status}" == "0" ]]; then
        run_dir="$(dirname "${manifest}")"
        diagnostics_source="${run_dir}/raw/model/tb/tta_diagnostics_${SEED}.json"
        if [[ -f "${diagnostics_source}" ]]; then
            cp "${diagnostics_source}" "${run_dir}/diagnostics.json"
            write_summary_json "${run_dir}/summary.json" "${model}" \
                "${reward}" "${dtg}" "${ndtg}" "${success}" "${spl}" \
                "${softspl}" "${na}" "${sna}" "${sws}"
        else
            status="diagnostics_missing"
        fi
    fi

    if [[ "${status}" == "0" ]]; then
        manifest_record="${manifest#"${REPO_ROOT}/"}"
        SUCCEEDED=$((SUCCEEDED + 1))
        printf 'ok\n' > "${job_dir}/validation"
    else
        FAILED=$((FAILED + 1))
        printf '%s\n' "${status}" > "${job_dir}/validation"
        manifest_record=""
        reward=""
        dtg=""
        ndtg=""
        success=""
        spl=""
        softspl=""
        na=""
        sna=""
        sws=""
    fi

    printf '%s,multi_source,tent,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${model}" "${SEED}" "${EPISODES}" "${status}" \
        "${manifest_record}" "${reward}" "${dtg}" "${ndtg}" "${success}" \
        "${spl}" "${softspl}" "${na}" "${sna}" "${sws}" \
        >> "${METRICS_FILE}"
done

{
    printf 'batch_id=%s\n' "${BATCH_ID}"
    printf 'git_commit=%s\n' "${GIT_COMMIT}"
    printf 'seed=%s\n' "${SEED}"
    printf 'episodes_per_job=%s\n' "${EPISODES}"
    printf 'planned=%s\n' "${EXPECTED_JOBS}"
    printf 'succeeded=%s\n' "${SUCCEEDED}"
    printf 'failed=%s\n' "${FAILED}"
    printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
    printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${LOG_ROOT}/SUMMARY"

printf 'Tent main-table multi-source batch finished: succeeded=%s failed=%s\n' \
    "${SUCCEEDED}" "${FAILED}"
printf 'Metrics: %s\n' "${METRICS_FILE}"
printf 'Logs:    %s\n' "${LOG_ROOT}"

if [[ ${FAILED_WAITS} -ne 0 || ${FAILED} -ne 0 ]]; then
    printf 'Batch incomplete; inspect logs and resume with --batch-id %s --resume.\n' \
        "${BATCH_ID}" >&2
    exit 1
fi

printf 'Both canonical multi-source Tent jobs completed successfully.\n'
