#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: vln/scripts/run_grouped_source_eval.sh [OPTIONS]

Run the nine VLN Source settings in three ordered resource groups:
  1. DUET, HAMT, GOAT (three parallel workers; R2R then REVERIE per worker)
  2. ETPNav, BEVBert (two parallel workers)
  3. StreamVLN (one worker)

Options:
  --gpu INDEX                 CUDA device index (default: 0)
  --run-tag TAG               Shared run tag (default: grouped-source-UTC)
  --split SPLIT               val_seen, val_unseen, test, or all (default: all)
  --ce-data-version VERSION   v1.3-unified or v1.2-native (default: v1.3-unified)
  --only-group INDEX          Run only resource group 1, 2, or 3
  --skip-runtime-check        Skip the offline runtime import check
  --dry-run                   Print/validate child commands without evaluation
  -h, --help                  Show this help

With --split all, every setting starts fresh processes in the canonical
val_seen -> val_unseen -> test order. Test emits submission trajectories and
is not scored locally.
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

GPU=0
RUN_TAG=""
SPLIT=all
CE_DATA_VERSION=v1.3-unified
SKIP_RUNTIME_CHECK=0
DRY_RUN=0
ONLY_GROUP=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --gpu)
            require_option_value "$@"
            GPU="$2"
            shift 2
            ;;
        --run-tag)
            require_option_value "$@"
            RUN_TAG="$2"
            shift 2
            ;;
        --split)
            require_option_value "$@"
            SPLIT="$2"
            shift 2
            ;;
        --ce-data-version)
            require_option_value "$@"
            CE_DATA_VERSION="$2"
            shift 2
            ;;
        --only-group)
            require_option_value "$@"
            ONLY_GROUP="$2"
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

[[ "${GPU}" =~ ^[0-9]+$ ]] || die "GPU index must be a non-negative integer"
case "${SPLIT}" in
    val_seen|val_unseen|test|all) ;;
    *) die "invalid split: ${SPLIT}" ;;
esac
case "${CE_DATA_VERSION}" in
    v1.3-unified|v1.2-native) ;;
    *) die "invalid CE data version: ${CE_DATA_VERSION}" ;;
esac
case "${ONLY_GROUP}" in
    ''|1|2|3) ;;
    *) die "invalid resource group: ${ONLY_GROUP}" ;;
esac
if [[ -z "${RUN_TAG}" ]]; then
    RUN_TAG="grouped-source-$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid run tag: ${RUN_TAG}"

REPO_ROOT=/data1/wxy/code/NavTTA
VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln
TMP_ROOT="${VLN_ROOT}/tmp"

RUNNER="${REPO_ROOT}/vln/scripts/run_source_eval.sh"
RUNTIME_CHECK="${REPO_ROOT}/vln/scripts/verify_runtime_imports.sh"
LOG_ROOT="${TMP_ROOT}/navtta-grouped-source/${RUN_TAG}"
ACTIVE_WORKER_PIDS=()
CURRENT_SETTING_WAIT_PID=""
CURRENT_SETTING_PGID=""
CURRENT_SETTING_PGID_FILE=""
CURRENT_SETTING_CHILD_PID=""
SOURCE_TAG_LOCK_FD=""
SOURCE_TAG_LOCK_ROOT="${TMP_ROOT}/navtta-source-tag-locks"
SOURCE_TAG_LOCK_FILE="${SOURCE_TAG_LOCK_ROOT}/${RUN_TAG}.lock"

[[ -x "${RUNNER}" ]] || die "missing source runner: ${RUNNER}"
command -v setsid >/dev/null 2>&1 || die "setsid is required for child cleanup"
[[ ! -e "${LOG_ROOT}" ]] || die "launcher log directory already exists: ${LOG_ROOT}"

claim_source_tag_lock() {
    command -v flock >/dev/null 2>&1 || die "flock is required for grouped runs"
    mkdir -p "${SOURCE_TAG_LOCK_ROOT}"
    exec {SOURCE_TAG_LOCK_FD}>"${SOURCE_TAG_LOCK_FILE}"
    if ! flock -n "${SOURCE_TAG_LOCK_FD}"; then
        die "source run tag is already active: ${RUN_TAG}"
    fi
    export NAVTTA_SOURCE_TAG_LOCKED="${RUN_TAG}"
    export NAVTTA_SOURCE_TAG_LOCK_FD="${SOURCE_TAG_LOCK_FD}"
}

check_run_tag_available() {
    [[ ! -e "${REPO_ROOT}/vln/results/source/${RUN_TAG}" ]] || \
        die "source result tag already exists: ${RUN_TAG}"
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
    claim_source_tag_lock
    check_run_tag_available
fi
mkdir -p "${LOG_ROOT%/*}"

printf 'Grouped VLN Source evaluation\n'
printf '  run tag:          %s\n' "${RUN_TAG}"
printf '  split:            %s\n' "${SPLIT}"
printf '  GPU:              %s\n' "${GPU}"
printf '  CE data version:  %s\n' "${CE_DATA_VERSION}"
printf '  resource groups:  %s\n' "${ONLY_GROUP:-1,2,3}"
printf '  launcher logs:    %s\n' "${LOG_ROOT}"

if [[ "${SKIP_RUNTIME_CHECK}" -eq 0 ]]; then
    "${RUNTIME_CHECK}"
fi
if [[ "${DRY_RUN}" -eq 0 ]]; then
    check_run_tag_available
fi
mkdir "${LOG_ROOT}" || die "could not create launcher log directory: ${LOG_ROOT}"

terminate_process_group() {
    local pgid="$1"
    [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 0
    if kill -0 -- "-${pgid}" 2>/dev/null; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
        if ! wait_for_process_group_exit "${pgid}"; then
            kill -KILL -- "-${pgid}" 2>/dev/null || true
        fi
    fi
}

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

load_current_setting_pgid() {
    local attempt
    [[ -z "${CURRENT_SETTING_PGID}" ]] || return 0
    [[ -n "${CURRENT_SETTING_PGID_FILE}" ]] || return 1
    for attempt in {1..50}; do
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
    local child
    local child_file
    local candidate_pgid
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
    load_current_setting_pgid || true
    if [[ -z "${CURRENT_SETTING_PGID}" && \
          "${CURRENT_SETTING_WAIT_PID}" =~ ^[1-9][0-9]*$ ]]; then
        kill -STOP "${CURRENT_SETTING_WAIT_PID}" 2>/dev/null || true
        load_pgid_from_wait_child || true
    fi
    if [[ -z "${CURRENT_SETTING_PGID}" && \
          "${CURRENT_SETTING_CHILD_PID}" =~ ^[1-9][0-9]*$ ]]; then
        kill -STOP "${CURRENT_SETTING_CHILD_PID}" 2>/dev/null || true
        candidate_pgid="$(
            ps -o pgid= -p "${CURRENT_SETTING_CHILD_PID}" 2>/dev/null
        )" || true
        candidate_pgid="${candidate_pgid//[[:space:]]/}"
        if [[ "${candidate_pgid}" == "${CURRENT_SETTING_CHILD_PID}" ]]; then
            CURRENT_SETTING_PGID="${candidate_pgid}"
        else
            kill -KILL "${CURRENT_SETTING_CHILD_PID}" 2>/dev/null || true
        fi
    fi
    terminate_process_group "${CURRENT_SETTING_PGID}"
    if [[ "${CURRENT_SETTING_WAIT_PID}" =~ ^[1-9][0-9]*$ ]]; then
        kill -CONT "${CURRENT_SETTING_WAIT_PID}" 2>/dev/null || true
        kill -TERM "${CURRENT_SETTING_WAIT_PID}" 2>/dev/null || true
        wait "${CURRENT_SETTING_WAIT_PID}" 2>/dev/null || true
    fi
    if [[ -n "${CURRENT_SETTING_PGID_FILE}" ]]; then
        rm -f -- "${CURRENT_SETTING_PGID_FILE}"
    fi
    CURRENT_SETTING_WAIT_PID=""
    CURRENT_SETTING_PGID=""
    CURRENT_SETTING_PGID_FILE=""
    CURRENT_SETTING_CHILD_PID=""
}

worker_cleanup() {
    local status="$1"
    trap - EXIT INT TERM
    terminate_current_setting
    exit "${status}"
}

parent_cleanup() {
    local status="$1"
    local pid
    local -a worker_pids=("${ACTIVE_WORKER_PIDS[@]}")
    trap - EXIT INT TERM
    terminate_current_setting
    while IFS= read -r pid; do
        worker_pids+=("${pid}")
    done < <(jobs -pr)
    for pid in "${worker_pids[@]}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    for pid in "${worker_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    ACTIVE_WORKER_PIDS=()
    exit "${status}"
}

trap 'parent_cleanup $?' EXIT
trap 'parent_cleanup 130' INT
trap 'parent_cleanup 143' TERM

run_setting() {
    local setting="$1"
    local log_path="${LOG_ROOT}/${setting}.log"
    local pgid_path="${LOG_ROOT}/.${setting}.pgid"
    local -a command=(
        "${RUNNER}" "${setting}" "${SPLIT}" "${GPU}"
        --run-tag "${RUN_TAG}"
    )
    case "${setting}" in
        etpnav-r2r-ce|bevbert-r2r-ce)
            command+=(--ce-data-version "${CE_DATA_VERSION}")
            ;;
    esac
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        command+=(--dry-run)
    fi

    printf '[start] %s -> %s\n' "${setting}" "${log_path}"
    CURRENT_SETTING_PGID=""
    CURRENT_SETTING_CHILD_PID=""
    CURRENT_SETTING_PGID_FILE="${pgid_path}"
    setsid --fork --wait bash -c '
        pgid_path="$1"
        shift
        printf "%s\n" "$$" >"${pgid_path}" || exit 125
        exec "$@"
    ' navtta-grouped-setting "${pgid_path}" "${command[@]}" \
        >"${log_path}" 2>&1 &
    CURRENT_SETTING_WAIT_PID=$!
    if ! load_current_setting_pgid; then
        printf '[failed:125] %s could not establish its process-group identity\n' \
            "${setting}" >&2
        terminate_current_setting
        tail -n 40 "${log_path}" >&2 || true
        return 125
    fi
    local status
    if wait "${CURRENT_SETTING_WAIT_PID}"; then
        status=0
    else
        status=$?
    fi
    if ! wait_for_process_group_exit "${CURRENT_SETTING_PGID}"; then
        printf '[failed] %s left processes in PGID %s; terminating them\n' \
            "${setting}" "${CURRENT_SETTING_PGID}" >&2
        terminate_process_group "${CURRENT_SETTING_PGID}"
        [[ "${status}" -ne 0 ]] || status=1
    fi
    rm -f -- "${CURRENT_SETTING_PGID_FILE}"
    CURRENT_SETTING_WAIT_PID=""
    CURRENT_SETTING_PGID=""
    CURRENT_SETTING_PGID_FILE=""
    CURRENT_SETTING_CHILD_PID=""
    if [[ "${status}" -eq 0 ]]; then
        printf '[passed] %s\n' "${setting}"
    else
        printf '[failed:%d] %s; tail of %s follows\n' \
            "${status}" "${setting}" "${log_path}" >&2
        tail -n 40 "${log_path}" >&2 || true
        return "${status}"
    fi
}

run_worker() {
    local setting
    trap 'worker_cleanup $?' EXIT
    trap 'worker_cleanup 130' INT
    trap 'worker_cleanup 143' TERM
    for setting in "$@"; do
        run_setting "${setting}"
    done
    trap - EXIT INT TERM
}

wait_group() {
    local label="$1"
    shift
    local failed=0
    local pid
    for pid in "$@"; do
        if ! wait "${pid}"; then
            failed=1
        fi
        local -a remaining=()
        local active_pid
        for active_pid in "${ACTIVE_WORKER_PIDS[@]}"; do
            [[ "${active_pid}" == "${pid}" ]] || remaining+=("${active_pid}")
        done
        ACTIVE_WORKER_PIDS=("${remaining[@]}")
    done
    [[ "${failed}" -eq 0 ]] || die "${label} failed; later groups were not started"
    printf '[group passed] %s\n' "${label}"
}

if [[ -z "${ONLY_GROUP}" || "${ONLY_GROUP}" == 1 ]]; then
    printf '\n[group 1/3] DUET, HAMT, GOAT\n'
    (run_worker duet-r2r duet-reverie) &
    duet_pid=$!
    ACTIVE_WORKER_PIDS+=("${duet_pid}")
    (run_worker hamt-r2r hamt-reverie) &
    hamt_pid=$!
    ACTIVE_WORKER_PIDS+=("${hamt_pid}")
    (run_worker goat-r2r goat-reverie) &
    goat_pid=$!
    ACTIVE_WORKER_PIDS+=("${goat_pid}")
    wait_group "group 1 (DUET/HAMT/GOAT)" \
        "${duet_pid}" "${hamt_pid}" "${goat_pid}"
fi

if [[ -z "${ONLY_GROUP}" || "${ONLY_GROUP}" == 2 ]]; then
    printf '\n[group 2/3] ETPNav, BEVBert\n'
    (run_worker etpnav-r2r-ce) &
    etpnav_pid=$!
    ACTIVE_WORKER_PIDS+=("${etpnav_pid}")
    (run_worker bevbert-r2r-ce) &
    bevbert_pid=$!
    ACTIVE_WORKER_PIDS+=("${bevbert_pid}")
    wait_group "group 2 (ETPNav/BEVBert)" \
        "${etpnav_pid}" "${bevbert_pid}"
fi

if [[ -z "${ONLY_GROUP}" || "${ONLY_GROUP}" == 3 ]]; then
    printf '\n[group 3/3] StreamVLN\n'
    run_setting streamvln-r2r-ce
    printf '[group passed] group 3 (StreamVLN)\n'
fi

printf '\nSelected grouped Source evaluations passed.\n'
printf 'Run tag: %s\n' "${RUN_TAG}"
printf 'Results: %s\n' "${REPO_ROOT}/vln/results/source/${RUN_TAG}"
printf 'Launcher logs: %s\n' "${LOG_ROOT}"
