#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Re-evaluate all four AVN source checkpoints on the canonical TTA episode stream.

The fixed eight-job plan is:
  GPU 0: AV-Nav    x {single_source, multi_source}
  GPU 1: SAVi      x {single_source, multi_source}
  GPU 2: SMT+Audio x {single_source, multi_source}
  GPU 3: ENMuS     x {single_source, multi_source}

Each model keeps its original source-policy evaluation path. All jobs use the
same stream protocol as the Tent grids: 20 scenes x 100 episodes, one process,
seed-controlled global shuffle, and no TTA parameter update.

Usage:
  bash avn/scripts/run_source_reval.sh [options]

Options:
  --seed N          Episode-order and evaluation seed (default: 0)
  --gpus LIST       Four physical GPU ids in model order (default: 0,1,2,3)
  --episodes N      Episodes per job (default: 2000; canonical value: 2000)
  --batch-id ID     Explicit log batch id (default: UTC timestamp)
  --resume          Resume an explicit batch id and skip exit-code-0 jobs
  --allow-dirty     Permit a tracked dirty worktree (not recommended)
  --dry-run         Validate inputs and print the eight-job plan only
  -h, --help        Show this help

Recommended detached launch:
  screen -dmS avn_source_reval \
    bash avn/scripts/run_source_reval.sh --seed 0 --gpus 0,1,2,3 \
      --batch-id source-reval-v1-seed0

Attach:
  screen -d -r avn_source_reval

Follow all job completions from another terminal:
  tail -f avn/results/logs/source_reval/source-reval-v1-seed0/scheduler.log
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

SEED=0
GPU_CSV="0,1,2,3"
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
[[ ${EPISODES} -le 2000 ]] || die "canonical stream contains only 2000 unique episodes"
if [[ ${RESUME} -eq 1 && ${BATCH_ID_GIVEN} -eq 0 ]]; then
    die "--resume requires --batch-id"
fi
if [[ -z "${BATCH_ID}" ]]; then
    BATCH_ID="source-reval-seed${SEED}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "${BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid batch id"

IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
[[ ${#GPUS[@]} -eq 4 ]] || die "--gpus must contain exactly four GPU ids"
for ((i = 0; i < 4; i++)); do
    [[ "${GPUS[$i]}" =~ ^[0-9]+$ ]] || die "invalid GPU id: ${GPUS[$i]}"
    for ((j = i + 1; j < 4; j++)); do
        [[ "${GPUS[$i]}" != "${GPUS[$j]}" ]] || \
            die "GPU ids must be unique: ${GPUS[$i]}"
    done
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="${REPO_ROOT}/avn/results/logs/source_reval/${BATCH_ID}"
JOBS_ROOT="${LOG_ROOT}/jobs"
RESULT_ROOT="${REPO_ROOT}/avn/results/legacy/source_reval/${BATCH_ID}"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf 'uncommitted')"
MODELS=("av_nav" "savi" "smt_audio" "enmus")
SOURCE_SETTINGS=("single_source" "multi_source")
EXPECTED_JOBS=8

runner_for_model() {
    case "$1" in
        av_nav) printf '%s\n' "${REPO_ROOT}/avn/scripts/eval_av_nav.sh" ;;
        savi) printf '%s\n' "${REPO_ROOT}/avn/scripts/eval_savi.sh" ;;
        smt_audio) printf '%s\n' "${REPO_ROOT}/avn/scripts/eval_smt_audio.sh" ;;
        enmus) printf '%s\n' "${REPO_ROOT}/avn/scripts/eval_enmus.sh" ;;
        *) die "unknown model: $1" ;;
    esac
}

checkpoint_for_job() {
    local model="$1"
    local source_setting="$2"
    local filename
    case "${model}:${source_setting}" in
        av_nav:single_source|savi:single_source|smt_audio:single_source)
            filename="single_best_val.pth"
            ;;
        av_nav:multi_source|savi:multi_source|smt_audio:multi_source)
            filename="multi_best_val.pth"
            ;;
        enmus:single_source)
            filename="single_source_best_val.pth"
            ;;
        enmus:multi_source)
            filename="multi_source_best_val.pth"
            ;;
        *) die "unknown model/source pair: ${model}/${source_setting}" ;;
    esac
    printf '%s\n' "${REPO_ROOT}/avn/checkpoints/source/${model}/${filename}"
}

order_sha_for_setting() {
    case "$1" in
        single_source) printf '%s\n' "${SINGLE_ORDER_SHA256}" ;;
        multi_source) printf '%s\n' "${MULTI_ORDER_SHA256}" ;;
        *) die "unknown source setting: $1" ;;
    esac
}

content_sha_for_setting() {
    case "$1" in
        single_source) printf '%s\n' "${SINGLE_CONTENT_SHA256}" ;;
        multi_source) printf '%s\n' "${MULTI_CONTENT_SHA256}" ;;
        *) die "unknown source setting: $1" ;;
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

for model in "${MODELS[@]}"; do
    runner="$(runner_for_model "${model}")"
    [[ -f "${runner}" ]] || die "missing evaluation runner: ${runner}"
    for source_setting in "${SOURCE_SETTINGS[@]}"; do
        checkpoint="$(checkpoint_for_job "${model}" "${source_setting}")"
        [[ -f "${checkpoint}" ]] || die "missing checkpoint: ${checkpoint}"
    done
done
for source_setting in "${SOURCE_SETTINGS[@]}"; do
    dataset="${REPO_ROOT}/avn/data/datasets/tta_test/${source_setting}/mp3d/v1/val/val.json.gz"
    [[ -f "${dataset}" ]] || die "missing canonical TTA dataset: ${dataset}"
done
for auxiliary in \
    "${REPO_ROOT}/avn/baselines/smt_audio/data/pretrained_weights/semantic_audionav/savi/best_val.pth" \
    "${REPO_ROOT}/avn/baselines/smt_audio/data/pretrained_weights/semantic_audionav/savi/label_predictor.pth" \
    "${REPO_ROOT}/avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/audio_encoder_best_val.pth" \
    "${REPO_ROOT}/avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/visual_encoder_best_val.pth" \
    "${REPO_ROOT}/avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/seld_crnn_best_val.h5"; do
    [[ -f "${auxiliary}" ]] || die "missing auxiliary checkpoint: ${auxiliary}"
done

AV_NAV_SINGLE_SHA256="$(sha256_file "$(checkpoint_for_job av_nav single_source)")"
AV_NAV_MULTI_SHA256="$(sha256_file "$(checkpoint_for_job av_nav multi_source)")"
SAVI_SINGLE_SHA256="$(sha256_file "$(checkpoint_for_job savi single_source)")"
SAVI_MULTI_SHA256="$(sha256_file "$(checkpoint_for_job savi multi_source)")"
SMT_AUDIO_SINGLE_SHA256="$(sha256_file "$(checkpoint_for_job smt_audio single_source)")"
SMT_AUDIO_MULTI_SHA256="$(sha256_file "$(checkpoint_for_job smt_audio multi_source)")"
ENMUS_SINGLE_SHA256="$(sha256_file "$(checkpoint_for_job enmus single_source)")"
ENMUS_MULTI_SHA256="$(sha256_file "$(checkpoint_for_job enmus multi_source)")"
SAVI_DESCRIPTOR_SHA256="$(sha256_file "${REPO_ROOT}/avn/baselines/smt_audio/data/pretrained_weights/semantic_audionav/savi/best_val.pth")"
SAVI_LABEL_SHA256="$(sha256_file "${REPO_ROOT}/avn/baselines/smt_audio/data/pretrained_weights/semantic_audionav/savi/label_predictor.pth")"
ENMUS_AUDIO_SHA256="$(sha256_file "${REPO_ROOT}/avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/audio_encoder_best_val.pth")"
ENMUS_VISUAL_SHA256="$(sha256_file "${REPO_ROOT}/avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/visual_encoder_best_val.pth")"
ENMUS_SELD_SHA256="$(sha256_file "${REPO_ROOT}/avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/seld_crnn_best_val.h5")"

checkpoint_sha_for_job() {
    case "$1:$2" in
        av_nav:single_source) printf '%s\n' "${AV_NAV_SINGLE_SHA256}" ;;
        av_nav:multi_source) printf '%s\n' "${AV_NAV_MULTI_SHA256}" ;;
        savi:single_source) printf '%s\n' "${SAVI_SINGLE_SHA256}" ;;
        savi:multi_source) printf '%s\n' "${SAVI_MULTI_SHA256}" ;;
        smt_audio:single_source) printf '%s\n' "${SMT_AUDIO_SINGLE_SHA256}" ;;
        smt_audio:multi_source) printf '%s\n' "${SMT_AUDIO_MULTI_SHA256}" ;;
        enmus:single_source) printf '%s\n' "${ENMUS_SINGLE_SHA256}" ;;
        enmus:multi_source) printf '%s\n' "${ENMUS_MULTI_SHA256}" ;;
        *) die "unknown model/source pair: $1/$2" ;;
    esac
}

for baseline in smt_audio enmus; do
    for data_name in \
        datasets scene_datasets scene_observations metadata sounds binaural_rirs; do
        baseline_data="${REPO_ROOT}/avn/baselines/${baseline}/data/${data_name}"
        if [[ ! -d "${baseline_data}" ]]; then
            if [[ ${DRY_RUN} -eq 1 ]]; then
                printf 'warning: baseline data link is missing: %s\n' "${baseline_data}" >&2
            else
                die "baseline data link is missing: ${baseline_data}; run python3 avn/scripts/link_local_data.py"
            fi
        fi
    done
done

asset_dir_has_files() {
    local directory="$1"
    [[ -d "${directory}" ]] && \
        [[ -n "$(find -L "${directory}" -type f ! -name .gitkeep -print -quit 2>/dev/null)" ]]
}

check_asset_dir() {
    local label="$1"
    local directory="$2"
    if asset_dir_has_files "${directory}"; then
        return
    fi
    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf 'warning: local %s assets are unavailable: %s\n' "${label}" "${directory}" >&2
    else
        die "${label} assets are unavailable: ${directory}"
    fi
}

check_asset_dir "scene dataset" "${REPO_ROOT}/avn/data/scene_datasets"
check_asset_dir "rendered scene observation" "${REPO_ROOT}/avn/data/scene_observations"
check_asset_dir "audio metadata" "${REPO_ROOT}/avn/data/metadata"
check_asset_dir "source sound" "${REPO_ROOT}/avn/data/sounds"
check_asset_dir "binaural RIR" "${REPO_ROOT}/avn/data/binaural_rirs"

SINGLE_DATASET="${REPO_ROOT}/avn/data/datasets/tta_test/single_source/mp3d/v1/val/val.json.gz"
MULTI_DATASET="${REPO_ROOT}/avn/data/datasets/tta_test/multi_source/mp3d/v1/val/val.json.gz"
single_fingerprints="$(
    python3 "${REPO_ROOT}/avn/scripts/fingerprint_episode_stream.py" \
        --dataset "${SINGLE_DATASET}" --seed "${SEED}"
)"
multi_fingerprints="$(
    python3 "${REPO_ROOT}/avn/scripts/fingerprint_episode_stream.py" \
        --dataset "${MULTI_DATASET}" --seed "${SEED}"
)"
read -r SINGLE_ORDER_SHA256 SINGLE_CONTENT_SHA256 <<< "${single_fingerprints}"
read -r MULTI_ORDER_SHA256 MULTI_CONTENT_SHA256 <<< "${multi_fingerprints}"
for fingerprint in \
    "${SINGLE_ORDER_SHA256}" "${SINGLE_CONTENT_SHA256}" \
    "${MULTI_ORDER_SHA256}" "${MULTI_CONTENT_SHA256}"; do
    [[ "${fingerprint}" =~ ^[0-9a-f]{64}$ ]] || \
        die "invalid episode-stream fingerprint: ${fingerprint}"
done

if [[ ${DRY_RUN} -eq 0 && ${ALLOW_DIRTY} -eq 0 ]]; then
    if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
        die "tracked worktree changes detected; commit them first or use --allow-dirty"
    fi
    EXECUTION_FILES=(
        avn/scripts/run_source_reval.sh
        avn/scripts/eval_av_nav.sh
        avn/scripts/eval_savi.sh
        avn/scripts/eval_smt_audio.sh
        avn/scripts/eval_enmus.sh
        avn/scripts/fingerprint_episode_stream.py
        tools/create_run_manifest.py
        tools/finalize_run_manifest.py
        tools/validate_run_manifest.py
        core/navtta_core/experiment/episode_stream.py
    )
    for execution_file in "${EXECUTION_FILES[@]}"; do
        git -C "${REPO_ROOT}" ls-files --error-unmatch -- "${execution_file}" \
            >/dev/null 2>&1 || \
            die "execution file is not tracked by Git: ${execution_file}"
    done
fi

BATCH_SPEC="${LOG_ROOT}/batch.env"
spec_value() {
    local key="$1"
    sed -n "s/^${key}=//p" "${BATCH_SPEC}" | tail -n 1
}

require_spec_value() {
    local key="$1"
    local expected="$2"
    local actual
    actual="$(spec_value "${key}")"
    [[ "${actual}" == "${expected}" ]] || \
        die "resume mismatch for ${key}: old='${actual}' new='${expected}'"
}

validate_resume_spec() {
    [[ -f "${BATCH_SPEC}" ]] || die "resume batch is missing batch.env"
    require_spec_value batch_id "${BATCH_ID}"
    require_spec_value git_commit "${GIT_COMMIT}"
    require_spec_value seed "${SEED}"
    require_spec_value episodes "${EPISODES}"
    require_spec_value gpus "${GPU_CSV}"
    require_spec_value single_order_sha256 "${SINGLE_ORDER_SHA256}"
    require_spec_value single_content_sha256 "${SINGLE_CONTENT_SHA256}"
    require_spec_value multi_order_sha256 "${MULTI_ORDER_SHA256}"
    require_spec_value multi_content_sha256 "${MULTI_CONTENT_SHA256}"
    require_spec_value av_nav_single_sha256 "${AV_NAV_SINGLE_SHA256}"
    require_spec_value av_nav_multi_sha256 "${AV_NAV_MULTI_SHA256}"
    require_spec_value savi_single_sha256 "${SAVI_SINGLE_SHA256}"
    require_spec_value savi_multi_sha256 "${SAVI_MULTI_SHA256}"
    require_spec_value smt_audio_single_sha256 "${SMT_AUDIO_SINGLE_SHA256}"
    require_spec_value smt_audio_multi_sha256 "${SMT_AUDIO_MULTI_SHA256}"
    require_spec_value enmus_single_sha256 "${ENMUS_SINGLE_SHA256}"
    require_spec_value enmus_multi_sha256 "${ENMUS_MULTI_SHA256}"
    require_spec_value savi_descriptor_sha256 "${SAVI_DESCRIPTOR_SHA256}"
    require_spec_value savi_label_sha256 "${SAVI_LABEL_SHA256}"
    require_spec_value enmus_audio_sha256 "${ENMUS_AUDIO_SHA256}"
    require_spec_value enmus_visual_sha256 "${ENMUS_VISUAL_SHA256}"
    require_spec_value enmus_seld_sha256 "${ENMUS_SELD_SHA256}"
}

write_grid() {
    local job_id=0
    local model_index
    local model
    local gpu
    local source_setting
    local tag
    local checkpoint_sha256
    local order_sha256
    local content_sha256
    printf 'job_id,run_tag,gpu,model,source_setting,seed,episodes,git_commit,checkpoint_sha256,stream_order_sha256,stream_content_sha256\n'
    for ((model_index = 0; model_index < 4; model_index++)); do
        model="${MODELS[$model_index]}"
        gpu="${GPUS[$model_index]}"
        for source_setting in "${SOURCE_SETTINGS[@]}"; do
            tag="${BATCH_ID}-j$(printf '%02d' "${job_id}")-${model}-${source_setting}"
            checkpoint_sha256="$(checkpoint_sha_for_job "${model}" "${source_setting}")"
            order_sha256="$(order_sha_for_setting "${source_setting}")"
            content_sha256="$(content_sha_for_setting "${source_setting}")"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "${job_id}" "${tag}" "${gpu}" "${model}" "${source_setting}" \
                "${SEED}" "${EPISODES}" "${GIT_COMMIT}" \
                "${checkpoint_sha256}" "${order_sha256}" "${content_sha256}"
            job_id=$((job_id + 1))
        done
    done
}

validate_grid() {
    local grid_file="${LOG_ROOT}/grid.csv"
    local actual
    local expected
    [[ -f "${grid_file}" ]] || die "resume batch is missing grid.csv"
    actual="$(cat "${grid_file}")"
    expected="$(write_grid)"
    [[ "${actual}" == "${expected}" ]] || \
        die "resume grid.csv does not match the requested eight-job plan"
}

expected_tag_for_job() {
    local model="$1"
    local source_setting="$2"
    local model_index
    local source_offset
    case "${model}" in
        av_nav) model_index=0 ;;
        savi) model_index=1 ;;
        smt_audio) model_index=2 ;;
        enmus) model_index=3 ;;
        *) die "unknown model: ${model}" ;;
    esac
    case "${source_setting}" in
        single_source) source_offset=0 ;;
        multi_source) source_offset=1 ;;
        *) die "unknown source setting: ${source_setting}" ;;
    esac
    printf '%s-j%02d-%s-%s\n' \
        "${BATCH_ID}" "$((model_index * 2 + source_offset))" \
        "${model}" "${source_setting}"
}

manifest_from_log() {
    local log_file="$1"
    tr '\r' '\n' < "${log_file}" | \
        sed -n '/\/manifest\.json$/p' | tail -n 1
}

run_manifest_is_valid() {
    local manifest="$1"
    local model="$2"
    local source_setting="$3"
    local tag="$4"
    local checkpoint_sha256
    local order_sha256
    local content_sha256
    [[ "${manifest}" == "${REPO_ROOT}"/avn/results/runs/*/manifest.json ]] || \
        return 1
    [[ -f "${manifest}" ]] || return 1
    checkpoint_sha256="$(checkpoint_sha_for_job "${model}" "${source_setting}")"
    order_sha256="$(order_sha_for_setting "${source_setting}")"
    content_sha256="$(content_sha_for_setting "${source_setting}")"
    python3 "${REPO_ROOT}/tools/validate_run_manifest.py" \
        --manifest "${manifest}" \
        --run-tag "${tag}" \
        --model "${model}" \
        --method source \
        --source-setting "${source_setting}" \
        --seed "${SEED}" \
        --git-commit "${GIT_COMMIT}" \
        --checkpoint-sha256 "${checkpoint_sha256}" \
        --stream-order-sha256 "${order_sha256}" \
        --stream-content-sha256 "${content_sha256}"
}

if [[ ${DRY_RUN} -eq 1 ]]; then
    if [[ ${RESUME} -eq 1 ]]; then
        [[ -d "${LOG_ROOT}" ]] || die "resume batch does not exist: ${LOG_ROOT}"
        [[ ! -d "${LOG_ROOT}/.scheduler.lock" ]] || \
            die "resume batch is running or has a stale lock: ${LOG_ROOT}/.scheduler.lock"
        validate_resume_spec
        validate_grid
    else
        [[ ! -e "${LOG_ROOT}" ]] || die "batch already exists: ${LOG_ROOT}"
        [[ ! -e "${RESULT_ROOT}" ]] || die "result batch already exists: ${RESULT_ROOT}"
    fi
fi

printf 'AVN source checkpoint re-evaluation\n'
printf '  repository:       %s\n' "${REPO_ROOT}"
printf '  batch id:         %s\n' "${BATCH_ID}"
printf '  seed:             %s\n' "${SEED}"
printf '  episodes/job:     %s\n' "${EPISODES}"
printf '  stream:           20 scenes x 100 episodes, global shuffle\n'
printf '  single order SHA: %s\n' "${SINGLE_ORDER_SHA256}"
printf '  multi order SHA:  %s\n' "${MULTI_ORDER_SHA256}"
printf '  Git commit:       %s\n' "${GIT_COMMIT}"
printf '  GPU/model map:    %s=av_nav, %s=savi, %s=smt_audio, %s=enmus\n' \
    "${GPUS[0]}" "${GPUS[1]}" "${GPUS[2]}" "${GPUS[3]}"
printf '  jobs/GPU:         2 (single_source + multi_source)\n'
printf '  planned jobs:     %s\n' "${EXPECTED_JOBS}"
if [[ ${EPISODES} -ne 2000 ]]; then
    printf '  warning:          noncanonical smoke run; do not use as Source table\n'
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
    job_index=0
    for ((model_index = 0; model_index < 4; model_index++)); do
        model="${MODELS[$model_index]}"
        gpu="${GPUS[$model_index]}"
        runner="$(runner_for_model "${model}")"
        for source_setting in "${SOURCE_SETTINGS[@]}"; do
            tag="${BATCH_ID}-j$(printf '%02d' "${job_index}")-${model}-${source_setting}"
            printf '  job=%02d gpu=%s model=%s source=%s action=sample\n' \
                "${job_index}" "${gpu}" "${model}" "${source_setting}"
            if [[ "${model}" == "smt_audio" ]]; then
                printf '    CUDA_VISIBLE_DEVICES=%s NAVTTA_RUN_TAG=%s bash %s %s source %s TEST_EPISODE_COUNT %s EVAL.ACTION_SELECTION sample\n' \
                    "${gpu}" "${tag}" "${runner}" "${source_setting}" "${SEED}" "${EPISODES}"
            else
                printf '    CUDA_VISIBLE_DEVICES=%s NAVTTA_RUN_TAG=%s bash %s %s source %s TEST_EPISODE_COUNT %s\n' \
                    "${gpu}" "${tag}" "${runner}" "${source_setting}" "${SEED}" "${EPISODES}"
            fi
            job_index=$((job_index + 1))
        done
    done
    printf 'Dry run only; no jobs launched.\n'
    exit 0
fi

if [[ ${RESUME} -eq 0 ]]; then
    [[ ! -e "${LOG_ROOT}" ]] || die "batch already exists: ${LOG_ROOT}"
    [[ ! -e "${RESULT_ROOT}" ]] || die "result batch already exists: ${RESULT_ROOT}"
else
    [[ -d "${LOG_ROOT}" ]] || die "resume batch does not exist: ${LOG_ROOT}"
fi
mkdir -p "${JOBS_ROOT}" "${RESULT_ROOT}"

PIDS=()
PID_TAGS=()
KEEP_LOCK=0
LOCK_REASON=""
LOCK_DIR="${LOG_ROOT}/.scheduler.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    die "batch is already running or has a stale lock: ${LOCK_DIR}"
fi
printf 'pid=%s\nhost=%s\nstarted_at=%s\n' \
    "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${LOCK_DIR}/owner"
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
    local i
    local pid
    for ((i = 0; i < ${#PIDS[@]}; i++)); do
        pid="${PIDS[$i]}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            terminate_process_tree "${pid}"
        fi
    done
    for ((i = 0; i < ${#PIDS[@]}; i++)); do
        pid="${PIDS[$i]}"
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
            PIDS[$i]=""
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
    local exit_code="$1"
    local reason="$2"
    KEEP_LOCK=1
    LOCK_REASON="${reason}"
    trap - HUP INT TERM
    terminate_active_workers
    printf 'scheduler interrupted; lock retained at %s\n' "${LOCK_DIR}" >&2
    exit "${exit_code}"
}
trap cleanup_lock EXIT
trap 'handle_signal 129 signal_hup' HUP
trap 'handle_signal 130 signal_int' INT
trap 'handle_signal 143 signal_term' TERM

write_batch_spec() {
    {
        printf 'batch_id=%s\n' "${BATCH_ID}"
        printf 'git_commit=%s\n' "${GIT_COMMIT}"
        printf 'seed=%s\n' "${SEED}"
        printf 'episodes=%s\n' "${EPISODES}"
        printf 'gpus=%s\n' "${GPU_CSV}"
        printf 'single_order_sha256=%s\n' "${SINGLE_ORDER_SHA256}"
        printf 'single_content_sha256=%s\n' "${SINGLE_CONTENT_SHA256}"
        printf 'multi_order_sha256=%s\n' "${MULTI_ORDER_SHA256}"
        printf 'multi_content_sha256=%s\n' "${MULTI_CONTENT_SHA256}"
        printf 'av_nav_single_sha256=%s\n' "${AV_NAV_SINGLE_SHA256}"
        printf 'av_nav_multi_sha256=%s\n' "${AV_NAV_MULTI_SHA256}"
        printf 'savi_single_sha256=%s\n' "${SAVI_SINGLE_SHA256}"
        printf 'savi_multi_sha256=%s\n' "${SAVI_MULTI_SHA256}"
        printf 'smt_audio_single_sha256=%s\n' "${SMT_AUDIO_SINGLE_SHA256}"
        printf 'smt_audio_multi_sha256=%s\n' "${SMT_AUDIO_MULTI_SHA256}"
        printf 'enmus_single_sha256=%s\n' "${ENMUS_SINGLE_SHA256}"
        printf 'enmus_multi_sha256=%s\n' "${ENMUS_MULTI_SHA256}"
        printf 'savi_descriptor_sha256=%s\n' "${SAVI_DESCRIPTOR_SHA256}"
        printf 'savi_label_sha256=%s\n' "${SAVI_LABEL_SHA256}"
        printf 'enmus_audio_sha256=%s\n' "${ENMUS_AUDIO_SHA256}"
        printf 'enmus_visual_sha256=%s\n' "${ENMUS_VISUAL_SHA256}"
        printf 'enmus_seld_sha256=%s\n' "${ENMUS_SELD_SHA256}"
        printf 'provenance_status=legacy_checkpoint_provenance_incomplete\n'
    } > "${BATCH_SPEC}"
}

if [[ ${RESUME} -eq 1 ]]; then
    validate_resume_spec
else
    write_batch_spec
fi

if [[ ${RESUME} -eq 0 ]]; then
    write_grid > "${LOG_ROOT}/grid.csv"
else
    validate_grid
fi
cp "${BATCH_SPEC}" "${RESULT_ROOT}/batch.env"
cp "${LOG_ROOT}/grid.csv" "${RESULT_ROOT}/grid.csv"

SCHEDULER_LOG="${LOG_ROOT}/scheduler.log"
exec > >(tee -a "${SCHEDULER_LOG}") 2>&1

LAUNCHED=0
SKIPPED=0
job_index=0

launch_job() {
    local job_id="$1"
    local model="$2"
    local source_setting="$3"
    local gpu="$4"
    local runner="$5"
    local tag="$6"
    local job_dir="${JOBS_ROOT}/${model}/${source_setting}"
    local checkpoint_sha256
    local order_sha256
    local content_sha256
    local prior_status=""
    local prior_validation=""
    local prior_manifest=""
    local attempt

    mkdir -p "${job_dir}"
    checkpoint_sha256="$(checkpoint_sha_for_job "${model}" "${source_setting}")"
    order_sha256="$(order_sha_for_setting "${source_setting}")"
    content_sha256="$(content_sha_for_setting "${source_setting}")"
    if [[ -f "${job_dir}/exitcode" ]]; then
        prior_status="$(tr -d '[:space:]' < "${job_dir}/exitcode")"
        if [[ -f "${job_dir}/validation" ]]; then
            prior_validation="$(tr -d '[:space:]' < "${job_dir}/validation")"
        fi
        if [[ ${RESUME} -eq 1 && "${prior_status}" == "0" && \
                "${prior_validation}" == "ok" && \
                -f "${job_dir}/console.log" ]]; then
            prior_manifest="$(manifest_from_log "${job_dir}/console.log")"
            if run_manifest_is_valid "${prior_manifest}" "${model}" \
                    "${source_setting}" "${tag}" >/dev/null; then
                SKIPPED=$((SKIPPED + 1))
                printf '%s skip completed tag=%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${tag}"
                return
            fi
            printf '%s rerun tag=%s because its completed manifest is invalid\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${tag}"
        fi
        attempt="$(date -u +%Y%m%dT%H%M%SZ)"
        mv "${job_dir}/exitcode" "${job_dir}/exitcode.previous.${attempt}"
        if [[ -f "${job_dir}/validation" ]]; then
            mv "${job_dir}/validation" "${job_dir}/validation.previous.${attempt}"
        fi
    fi

    {
        printf 'job_id=%s\n' "${job_id}"
        printf 'run_tag=%s\n' "${tag}"
        printf 'gpu=%s\n' "${gpu}"
        printf 'model=%s\n' "${model}"
        printf 'source_setting=%s\n' "${source_setting}"
        printf 'method=source\n'
        printf 'action_protocol=original_sample\n'
        printf 'seed=%s\n' "${SEED}"
        printf 'episodes=%s\n' "${EPISODES}"
        printf 'tta_episodes_per_scene=100\n'
        printf 'tta_expected_scenes=20\n'
        printf 'tta_global_shuffle=true\n'
        printf 'stream_order_sha256=%s\n' "${order_sha256}"
        printf 'stream_content_sha256=%s\n' "${content_sha256}"
        printf 'git_commit=%s\n' "${GIT_COMMIT}"
        printf 'checkpoint_sha256=%s\n' "${checkpoint_sha256}"
    } > "${job_dir}/parameters.env"

    (
        set +e
        printf '\n===== attempt %s =====\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${job_dir}/console.log"

        extra_overrides=(TEST_EPISODE_COUNT "${EPISODES}")
        if [[ "${model}" == "smt_audio" ]]; then
            extra_overrides+=(EVAL.ACTION_SELECTION sample)
        fi

        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        TF_FORCE_GPU_ALLOW_GROWTH=true \
        PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        NAVTTA_RUN_TAG="${tag}" \
        NAVTTA_STREAM_ORDER_SHA256="${order_sha256}" \
        NAVTTA_STREAM_CONTENT_SHA256="${content_sha256}" \
        bash "${runner}" "${source_setting}" source "${SEED}" \
            "${extra_overrides[@]}" \
            >> "${job_dir}/console.log" 2>&1
        status=$?
        printf '%s\n' "${status}" > "${job_dir}/exitcode.tmp"
        mv "${job_dir}/exitcode.tmp" "${job_dir}/exitcode"
        exit "${status}"
    ) &

    PIDS+=("$!")
    PID_TAGS+=("${tag}")
    LAUNCHED=$((LAUNCHED + 1))
    printf '%s launched job=%02d tag=%s gpu=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${job_id}" "${tag}" "${gpu}"
}

for ((model_index = 0; model_index < 4; model_index++)); do
    model="${MODELS[$model_index]}"
    gpu="${GPUS[$model_index]}"
    runner="$(runner_for_model "${model}")"
    for source_setting in "${SOURCE_SETTINGS[@]}"; do
        tag="${BATCH_ID}-j$(printf '%02d' "${job_index}")-${model}-${source_setting}"
        launch_job "${job_index}" "${model}" "${source_setting}" \
            "${gpu}" "${runner}" "${tag}"
        job_index=$((job_index + 1))
    done
done

FAILED_WAITS=0
for ((i = 0; i < ${#PIDS[@]}; i++)); do
    pid="${PIDS[$i]}"
    if wait "${pid}"; then
        printf '%s completed tag=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PID_TAGS[$i]}"
    else
        FAILED_WAITS=$((FAILED_WAITS + 1))
        printf '%s failed tag=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${PID_TAGS[$i]}"
    fi
    PIDS[$i]=""
done

extract_metric() {
    local metric="$1"
    local log_file="$2"
    tr '\r' '\n' < "${log_file}" | \
        sed -n "s/.*Average episode ${metric}:[[:space:]]*//p" | tail -n 1
}

write_validation() {
    local job_dir="$1"
    local value="$2"
    printf '%s\n' "${value}" > "${job_dir}/validation.tmp"
    mv "${job_dir}/validation.tmp" "${job_dir}/validation"
}

METRICS_FILE="${RESULT_ROOT}/metrics.csv"
RESULT_MANIFEST_ROOT="${RESULT_ROOT}/manifests"
mkdir -p "${RESULT_MANIFEST_ROOT}"
printf 'model,source_setting,gpu,exitcode,manifest,reward,distance_to_goal,normalized_distance_to_goal,success,spl,softspl,na,sna,sws\n' \
    > "${METRICS_FILE}"

SUCCEEDED=0
FAILED=0
MISSING=0
for ((model_index = 0; model_index < 4; model_index++)); do
    model="${MODELS[$model_index]}"
    gpu="${GPUS[$model_index]}"
    for source_setting in "${SOURCE_SETTINGS[@]}"; do
        job_dir="${JOBS_ROOT}/${model}/${source_setting}"
        if [[ ! -f "${job_dir}/exitcode" ]]; then
            MISSING=$((MISSING + 1))
            printf '%s,%s,%s,missing,,,,,,,,,,\n' \
                "${model}" "${source_setting}" "${gpu}" >> "${METRICS_FILE}"
            continue
        fi

        status="$(tr -d '[:space:]' < "${job_dir}/exitcode")"
        log_file="${job_dir}/console.log"
        tag="$(expected_tag_for_job "${model}" "${source_setting}")"
        if [[ ! -f "${log_file}" ]]; then
            FAILED=$((FAILED + 1))
            write_validation "${job_dir}" log_missing
            printf '%s,%s,%s,log_missing,,,,,,,,,,\n' \
                "${model}" "${source_setting}" "${gpu}" >> "${METRICS_FILE}"
            continue
        fi

        manifest="$(manifest_from_log "${log_file}")"
        manifest_record=""
        reward="$(extract_metric reward "${log_file}")"
        dtg="$(extract_metric distance_to_goal "${log_file}")"
        ndtg="$(extract_metric normalized_distance_to_goal "${log_file}")"
        success="$(extract_metric success "${log_file}")"
        spl="$(extract_metric spl "${log_file}")"
        softspl="$(extract_metric softspl "${log_file}")"
        na="$(extract_metric na "${log_file}")"
        sna="$(extract_metric sna "${log_file}")"
        sws="$(extract_metric sws "${log_file}")"
        row_status="${status}"
        if [[ "${status}" != "0" ]]; then
            FAILED=$((FAILED + 1))
            write_validation "${job_dir}" runner_failed
        elif [[ -z "${manifest}" || -z "${reward}" || -z "${dtg}" || \
                -z "${ndtg}" || -z "${success}" || -z "${spl}" || \
                -z "${softspl}" || -z "${na}" || -z "${sna}" || -z "${sws}" ]]; then
            row_status="metrics_missing"
            FAILED=$((FAILED + 1))
            write_validation "${job_dir}" metrics_missing
        elif ! run_manifest_is_valid "${manifest}" "${model}" \
                "${source_setting}" "${tag}" >/dev/null; then
            row_status="manifest_invalid"
            FAILED=$((FAILED + 1))
            write_validation "${job_dir}" manifest_invalid
        else
            result_manifest="${RESULT_MANIFEST_ROOT}/${model}-${source_setting}.json"
            cp "${manifest}" "${result_manifest}"
            manifest_record="${result_manifest#"${REPO_ROOT}/"}"
            SUCCEEDED=$((SUCCEEDED + 1))
            write_validation "${job_dir}" ok
        fi
        if [[ "${row_status}" != "0" ]]; then
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
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "${model}" "${source_setting}" "${gpu}" "${row_status}" \
            "${manifest_record}" "${reward}" "${dtg}" "${ndtg}" "${success}" \
            "${spl}" "${softspl}" "${na}" "${sna}" "${sws}" \
            >> "${METRICS_FILE}"
    done
done

{
    printf 'batch_id=%s\n' "${BATCH_ID}"
    printf 'seed=%s\n' "${SEED}"
    printf 'episodes_per_job=%s\n' "${EPISODES}"
    printf 'planned=%s\n' "${EXPECTED_JOBS}"
    printf 'launched=%s\n' "${LAUNCHED}"
    printf 'skipped=%s\n' "${SKIPPED}"
    printf 'succeeded=%s\n' "${SUCCEEDED}"
    printf 'failed=%s\n' "${FAILED}"
    printf 'missing=%s\n' "${MISSING}"
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\n' "${GIT_COMMIT}"
    printf 'single_order_sha256=%s\n' "${SINGLE_ORDER_SHA256}"
    printf 'single_content_sha256=%s\n' "${SINGLE_CONTENT_SHA256}"
    printf 'multi_order_sha256=%s\n' "${MULTI_ORDER_SHA256}"
    printf 'multi_content_sha256=%s\n' "${MULTI_CONTENT_SHA256}"
    printf 'provenance_status=legacy_checkpoint_provenance_incomplete\n'
} > "${RESULT_ROOT}/SUMMARY"

cp "${METRICS_FILE}" "${LOG_ROOT}/metrics.csv"
cp "${RESULT_ROOT}/SUMMARY" "${LOG_ROOT}/SUMMARY"

printf 'Source re-evaluation finished: succeeded=%s failed=%s missing=%s\n' \
    "${SUCCEEDED}" "${FAILED}" "${MISSING}"
printf 'Trackable legacy metrics: %s\n' "${METRICS_FILE}"
printf 'Raw logs: %s\n' "${LOG_ROOT}"

if [[ ${FAILED_WAITS} -ne 0 || ${FAILED} -ne 0 || ${MISSING} -ne 0 ]]; then
    printf 'Source re-evaluation is incomplete; inspect %s and resume with --batch-id %s --resume.\n' \
        "${LOG_ROOT}" "${BATCH_ID}" >&2
    exit 1
fi
