#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: vln/scripts/run_four_gpu_source_eval.sh [OPTIONS]

Run all nine VLN Source settings on four GPUs.  Full mode evaluates exactly
val_seen and val_unseen; it never runs the hidden-label test split.

Queue layout:
  queue 0: ETPNav R2R-CE
  queue 1: BEVBert R2R-CE
  queue 2: StreamVLN R2R-CE
  queue 3: DUET, HAMT, and GOAT on R2R and REVERIE

Options:
  --gpus LIST                 Four distinct GPU IDs (default: 0,1,2,3)
  --run-tag TAG               Batch tag (default: four-gpu-source-UTC)
  --ce-data-version VERSION   v1.3-unified (default) or v1.2-native
  --smoke                     Run two val_seen episodes per setting only
  --only-queue INDEX          Run only queue 0, 1, 2, or 3
  --skip-runtime-check        Skip verify_runtime_imports.sh
  --dry-run                   Print child commands without evaluation
  -h, --help                  Show this help

Launcher logs are written below:
  /data1/wxy/exp_data/NavTTA/vln/tmp/navtta-four-gpu-source/TAG/

Full Source results remain below vln/results/source/, using one distinct
derived tag per setting. Smoke results remain below vln/results/smoke/.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_option_value() {
    [[ "$#" -ge 2 && -n "$2" && "$2" != -* ]] || \
        die "$1 requires a value"
}

GPU_LIST=0,1,2,3
RUN_TAG=""
CE_DATA_VERSION=v1.3-unified
SMOKE=0
ONLY_QUEUE=""
SKIP_RUNTIME_CHECK=0
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --gpus)
            require_option_value "$@"
            GPU_LIST="$2"
            shift 2
            ;;
        --run-tag)
            require_option_value "$@"
            RUN_TAG="$2"
            shift 2
            ;;
        --ce-data-version)
            require_option_value "$@"
            CE_DATA_VERSION="$2"
            shift 2
            ;;
        --smoke)
            SMOKE=1
            shift
            ;;
        --only-queue)
            require_option_value "$@"
            ONLY_QUEUE="$2"
            shift 2
            ;;
        --skip-runtime-check)
            SKIP_RUNTIME_CHECK=1
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

case "${CE_DATA_VERSION}" in
    v1.3-unified|v1.2-native) ;;
    *) die "invalid CE data version: ${CE_DATA_VERSION}" ;;
esac
case "${ONLY_QUEUE}" in
    ''|0|1|2|3) ;;
    *) die "invalid queue: ${ONLY_QUEUE}" ;;
esac
# The launcher owns this choice.  Do not let a stale exported value alter the
# seven settings for which --ce-data-version is intentionally not forwarded.
unset NAVTTA_CE_DATA_VERSION

IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
[[ "${#GPUS[@]}" -eq 4 ]] || die "--gpus requires exactly four GPU IDs"
for index in "${!GPUS[@]}"; do
    gpu="${GPUS[${index}]}"
    [[ "${gpu}" =~ ^[0-9]+$ ]] || \
        die "GPU IDs must be non-negative integers: ${GPU_LIST}"
    for prior_index in "${!GPUS[@]}"; do
        [[ "${prior_index}" -lt "${index}" ]] || continue
        [[ "${GPUS[${prior_index}]}" != "${gpu}" ]] || \
            die "GPU IDs must be distinct: ${GPU_LIST}"
    done
done

if [[ -z "${RUN_TAG}" ]]; then
    if [[ "${SMOKE}" -eq 1 ]]; then
        RUN_TAG="four-gpu-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
    else
        RUN_TAG="four-gpu-source-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
fi
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid run tag: ${RUN_TAG}"

REPO_ROOT=/data1/wxy/code/NavTTA
VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln
ENV_ROOT="${VLN_ROOT}/envs"
TMP_ROOT="${VLN_ROOT}/tmp"
RUNNER="${REPO_ROOT}/vln/scripts/run_source_eval.sh"
RUNTIME_CHECK="${REPO_ROOT}/vln/scripts/verify_runtime_imports.sh"
SUMMARIZER="${REPO_ROOT}/vln/scripts/summarize_source_eval.py"
LOG_ROOT="${TMP_ROOT}/navtta-four-gpu-source/${RUN_TAG}"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)" || \
    die "cannot resolve the NavTTA Git commit"

[[ -x "${RUNNER}" ]] || die "missing source runner: ${RUNNER}"
[[ -x "${RUNTIME_CHECK}" ]] || die "missing runtime checker: ${RUNTIME_CHECK}"
[[ -f "${SUMMARIZER}" ]] || die "missing Source summarizer: ${SUMMARIZER}"
command -v setsid >/dev/null 2>&1 || die "setsid is required"
[[ ! -e "${LOG_ROOT}" ]] || die "launcher log directory exists: ${LOG_ROOT}"

if [[ "${SMOKE}" -eq 1 ]]; then
    RESULT_NAMESPACE=smoke
    SPLITS=(val_seen)
    MODE=smoke
else
    RESULT_NAMESPACE=source
    SPLITS=(val_seen val_unseen)
    MODE=full
fi

queue_settings() {
    case "$1" in
        0) printf '%s\n' etpnav-r2r-ce ;;
        1) printf '%s\n' bevbert-r2r-ce ;;
        2) printf '%s\n' streamvln-r2r-ce ;;
        3) printf '%s\n' \
            duet-r2r duet-reverie \
            hamt-r2r hamt-reverie \
            goat-r2r goat-reverie ;;
        *) die "invalid queue: $1" ;;
    esac
}

queue_selected() {
    [[ -z "${ONLY_QUEUE}" || "${ONLY_QUEUE}" == "$1" ]]
}

for queue in 0 1 2 3; do
    queue_selected "${queue}" || continue
    while IFS= read -r setting; do
        setting_tag="${RUN_TAG}-${setting}"
        result_base="${REPO_ROOT}/vln/results/${RESULT_NAMESPACE}/${setting_tag}/${setting}"
        [[ ! -e "${result_base}" ]] || \
            die "result path already exists: ${result_base}"
    done < <(queue_settings "${queue}")
done

mkdir -p "${LOG_ROOT%/*}"
mkdir "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/combined.log") 2>&1
record_early_exit() {
    local status="$?"
    printf '%s\n' "${status}" >"${LOG_ROOT}/exitcode"
}
trap record_early_exit EXIT

printf 'queue\tgpu\tsetting\tsplit\tsetting_tag\tprotocol\tmode\n' \
    >"${LOG_ROOT}/plan.tsv"
{
    printf 'key\tvalue\n'
    printf 'run_tag\t%s\n' "${RUN_TAG}"
    printf 'git_commit\t%s\n' "${GIT_COMMIT}"
    printf 'mode\t%s\n' "${MODE}"
    printf 'gpus\t%s\n' "${GPU_LIST}"
    printf 'ce_data_version\t%s\n' "${CE_DATA_VERSION}"
    printf 'started_at_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${LOG_ROOT}/metadata.tsv"
for queue in 0 1 2 3; do
    queue_selected "${queue}" || continue
    while IFS= read -r setting; do
        setting_tag="${RUN_TAG}-${setting}"
        protocol=native
        case "${setting}" in
            etpnav-r2r-ce|bevbert-r2r-ce) protocol="${CE_DATA_VERSION}" ;;
            streamvln-r2r-ce) protocol=v1.3-unified ;;
        esac
        for split in "${SPLITS[@]}"; do
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${queue}" "${GPUS[${queue}]}" "${setting}" "${split}" \
                "${setting_tag}" "${protocol}" "${MODE}" \
                >>"${LOG_ROOT}/plan.tsv"
        done
    done < <(queue_settings "${queue}")
done

printf 'Four-GPU VLN Source evaluation\n'
printf '  run tag:          %s\n' "${RUN_TAG}"
printf '  Git commit:       %s\n' "${GIT_COMMIT}"
printf '  mode:             %s\n' "${MODE}"
printf '  GPUs:             %s\n' "${GPU_LIST}"
printf '  CE data version:  %s\n' "${CE_DATA_VERSION}"
printf '  selected queues:  %s\n' "${ONLY_QUEUE:-0,1,2,3}"
printf '  launcher logs:    %s\n' "${LOG_ROOT}"
cat "${LOG_ROOT}/plan.tsv"

if [[ "${SKIP_RUNTIME_CHECK}" -eq 0 ]]; then
    "${RUNTIME_CHECK}"
fi

CURRENT_SETTING_WAIT_PID=""
CURRENT_SETTING_PGID=""
CURRENT_SETTING_PGID_FILE=""
CURRENT_SETTING_CHILD_PID=""
CURRENT_QUEUE_EXIT_FILE=""
ACTIVE_WORKER_PIDS=()
ACTIVE_QUEUE_IDS=()

wait_for_process_group_exit() {
    local pgid="$1"
    local attempt
    [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 0
    for attempt in {1..50}; do
        kill -0 -- "-${pgid}" 2>/dev/null || return 0
        sleep 0.1
    done
    return 1
}

load_current_pgid() {
    local attempt
    [[ -z "${CURRENT_SETTING_PGID}" ]] || return 0
    [[ -n "${CURRENT_SETTING_PGID_FILE}" ]] || return 1
    for attempt in {1..100}; do
        if [[ -s "${CURRENT_SETTING_PGID_FILE}" ]]; then
            IFS= read -r CURRENT_SETTING_PGID <"${CURRENT_SETTING_PGID_FILE}"
            [[ "${CURRENT_SETTING_PGID}" =~ ^[1-9][0-9]*$ ]] && return 0
            CURRENT_SETTING_PGID=""
            return 1
        fi
        sleep 0.02
    done
    load_pgid_from_wait_child
}

load_pgid_from_wait_child() {
    local child child_file candidate_pgid
    local -a children=()
    [[ "${CURRENT_SETTING_WAIT_PID}" =~ ^[1-9][0-9]*$ ]] || return 1
    child_file="/proc/${CURRENT_SETTING_WAIT_PID}/task/${CURRENT_SETTING_WAIT_PID}/children"
    [[ -r "${child_file}" ]] || return 1
    IFS=' ' read -r -a children <"${child_file}" || true
    for child in "${children[@]}"; do
        [[ "${child}" =~ ^[1-9][0-9]*$ ]] || continue
        CURRENT_SETTING_CHILD_PID="${child}"
        candidate_pgid="$(ps -o pgid= -p "${child}" 2>/dev/null)" || true
        candidate_pgid="${candidate_pgid//[[:space:]]/}"
        if [[ "${candidate_pgid}" == "${child}" ]]; then
            CURRENT_SETTING_PGID="${candidate_pgid}"
            return 0
        fi
    done
    return 1
}

terminate_current_setting() {
    local candidate_pgid
    local pgid=""
    load_current_pgid || true
    pgid="${CURRENT_SETTING_PGID}"
    if [[ -z "${pgid}" && -n "${CURRENT_SETTING_PGID_FILE}" && \
          -s "${CURRENT_SETTING_PGID_FILE}" ]]; then
        IFS= read -r pgid <"${CURRENT_SETTING_PGID_FILE}" || true
    fi
    if [[ -z "${pgid}" && \
          "${CURRENT_SETTING_WAIT_PID}" =~ ^[1-9][0-9]*$ ]]; then
        kill -STOP "${CURRENT_SETTING_WAIT_PID}" 2>/dev/null || true
        load_pgid_from_wait_child || true
        pgid="${CURRENT_SETTING_PGID}"
    fi
    if [[ -z "${pgid}" && \
          "${CURRENT_SETTING_CHILD_PID}" =~ ^[1-9][0-9]*$ ]]; then
        kill -STOP "${CURRENT_SETTING_CHILD_PID}" 2>/dev/null || true
        candidate_pgid="$(
            ps -o pgid= -p "${CURRENT_SETTING_CHILD_PID}" 2>/dev/null
        )" || true
        candidate_pgid="${candidate_pgid//[[:space:]]/}"
        if [[ "${candidate_pgid}" == "${CURRENT_SETTING_CHILD_PID}" ]]; then
            pgid="${candidate_pgid}"
        else
            kill -KILL "${CURRENT_SETTING_CHILD_PID}" 2>/dev/null || true
        fi
    fi
    if [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] && \
       kill -0 -- "-${pgid}" 2>/dev/null; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
        if ! wait_for_process_group_exit "${pgid}"; then
            kill -KILL -- "-${pgid}" 2>/dev/null || true
        fi
    fi
    if [[ "${CURRENT_SETTING_WAIT_PID}" =~ ^[1-9][0-9]*$ ]]; then
        kill -TERM "${CURRENT_SETTING_WAIT_PID}" 2>/dev/null || true
        wait "${CURRENT_SETTING_WAIT_PID}" 2>/dev/null || true
    fi
    [[ -z "${CURRENT_SETTING_PGID_FILE}" ]] || \
        rm -f -- "${CURRENT_SETTING_PGID_FILE}"
    CURRENT_SETTING_WAIT_PID=""
    CURRENT_SETTING_PGID=""
    CURRENT_SETTING_PGID_FILE=""
    CURRENT_SETTING_CHILD_PID=""
}

worker_cleanup() {
    local status="$1"
    trap - EXIT INT TERM
    if [[ -n "${CURRENT_QUEUE_EXIT_FILE}" ]]; then
        printf '%s\n' "${status}" >"${CURRENT_QUEUE_EXIT_FILE}"
    fi
    terminate_current_setting
    exit "${status}"
}

parent_cleanup() {
    local status="$1"
    local pid
    trap - EXIT INT TERM
    for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    printf '%s\n' "${status}" >"${LOG_ROOT}/exitcode"
    exit "${status}"
}

trap 'parent_cleanup $?' EXIT
trap 'parent_cleanup 130' INT
trap 'parent_cleanup 143' TERM

run_one() {
    local queue="$1"
    local gpu="$2"
    local setting="$3"
    local split="$4"
    local setting_tag="${RUN_TAG}-${setting}"
    local -a command=(
        "${RUNNER}" "${setting}" "${split}" "${gpu}"
        --run-tag "${setting_tag}"
    )
    case "${setting}" in
        etpnav-r2r-ce|bevbert-r2r-ce)
            command+=(--ce-data-version "${CE_DATA_VERSION}")
            ;;
    esac
    [[ "${SMOKE}" -eq 0 ]] || command+=(--smoke-episodes 2)
    [[ "${DRY_RUN}" -eq 0 ]] || command+=(--dry-run)

    printf '[start] queue=%s gpu=%s setting=%s split=%s\n' \
        "${queue}" "${gpu}" "${setting}" "${split}"
    CURRENT_SETTING_PGID_FILE="${LOG_ROOT}/.queue-${queue}.pgid"
    CURRENT_SETTING_PGID=""
    CURRENT_SETTING_CHILD_PID=""
    setsid --fork --wait bash -c '
        pgid_file="$1"
        shift
        printf "%s\n" "$$" >"${pgid_file}" || exit 125
        exec "$@"
    ' navtta-four-gpu-setting "${CURRENT_SETTING_PGID_FILE}" "${command[@]}" &
    CURRENT_SETTING_WAIT_PID=$!
    if ! load_current_pgid; then
        printf '[failed:125] could not establish process-group identity\n' >&2
        terminate_current_setting
        return 125
    fi

    local status
    if wait "${CURRENT_SETTING_WAIT_PID}"; then
        status=0
    else
        status=$?
    fi
    # The waiter has been reaped.  Clear it before any process-group cleanup
    # so a recycled PID can never be signalled.
    CURRENT_SETTING_WAIT_PID=""
    if ! wait_for_process_group_exit "${CURRENT_SETTING_PGID}"; then
        terminate_current_setting
        [[ "${status}" -ne 0 ]] || status=1
    else
        rm -f -- "${CURRENT_SETTING_PGID_FILE}"
        CURRENT_SETTING_PGID=""
        CURRENT_SETTING_PGID_FILE=""
        CURRENT_SETTING_CHILD_PID=""
    fi
    if [[ "${status}" -ne 0 ]]; then
        printf '[failed:%s] queue=%s gpu=%s setting=%s split=%s\n' \
            "${status}" "${queue}" "${gpu}" "${setting}" "${split}" >&2
        return "${status}"
    fi
    printf '[passed] queue=%s gpu=%s setting=%s split=%s\n' \
        "${queue}" "${gpu}" "${setting}" "${split}"
}

run_queue() {
    local queue="$1"
    local gpu="$2"
    shift 2
    local setting split
    trap 'worker_cleanup $?' EXIT
    trap 'worker_cleanup 130' INT
    trap 'worker_cleanup 143' TERM
    for setting in "$@"; do
        for split in "${SPLITS[@]}"; do
            run_one "${queue}" "${gpu}" "${setting}" "${split}"
        done
    done
    trap - EXIT INT TERM
}

launch_queue() {
    local queue="$1"
    local gpu="${GPUS[${queue}]}"
    local -a settings=()
    mapfile -t settings < <(queue_settings "${queue}")
    (
        CURRENT_QUEUE_EXIT_FILE="${LOG_ROOT}/queue-${queue}.exitcode"
        run_queue "${queue}" "${gpu}" "${settings[@]}"
        printf '0\n' >"${CURRENT_QUEUE_EXIT_FILE}"
    ) > >(tee -a "${LOG_ROOT}/queue-${queue}.log") 2>&1 &
    ACTIVE_WORKER_PIDS+=("$!")
    ACTIVE_QUEUE_IDS+=("${queue}")
}

for queue in 0 1 2 3; do
    queue_selected "${queue}" || continue
    launch_queue "${queue}"
done

failed=0
printf 'queue\texit_code\n' >"${LOG_ROOT}/SUMMARY.tsv"
for index in "${!ACTIVE_WORKER_PIDS[@]}"; do
    pid="${ACTIVE_WORKER_PIDS[${index}]}"
    queue="${ACTIVE_QUEUE_IDS[${index}]}"
    if wait "${pid}"; then
        status=0
    else
        status=$?
        failed=1
    fi
    printf '%s\t%s\n' "${queue}" "${status}" >>"${LOG_ROOT}/SUMMARY.tsv"
    unset 'ACTIVE_WORKER_PIDS[index]'
    unset 'ACTIVE_QUEUE_IDS[index]'
done
ACTIVE_WORKER_PIDS=()
ACTIVE_QUEUE_IDS=()

[[ "${failed}" -eq 0 ]] || die "one or more GPU queues failed"

if [[ "${DRY_RUN}" -eq 0 && "${SMOKE}" -eq 0 && -z "${ONLY_QUEUE}" ]]; then
    "${ENV_ROOT}/duet/bin/python" "${SUMMARIZER}" \
        --run-tag "${RUN_TAG}" \
        --ce-data-version "${CE_DATA_VERSION}" \
        --output "${LOG_ROOT}/metrics.csv"
fi

printf 'Selected four-GPU Source evaluations completed and parsed.\n'
printf 'Run tag: %s\n' "${RUN_TAG}"
printf 'Launcher logs: %s\n' "${LOG_ROOT}"
if [[ "${SMOKE}" -eq 0 ]]; then
    printf 'Results use per-setting tags: %s-SETTING\n' "${RUN_TAG}"
    printf 'Inspect metrics.csv for Excel agreement; MISMATCH is not an execution failure.\n'
fi
