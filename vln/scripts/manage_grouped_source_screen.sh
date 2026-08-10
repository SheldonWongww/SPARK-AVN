#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  vln/scripts/manage_grouped_source_screen.sh start RUN_TAG [RUNNER_OPTIONS]
  vln/scripts/manage_grouped_source_screen.sh status RUN_TAG
  vln/scripts/manage_grouped_source_screen.sh logs RUN_TAG [--follow] [--setting SETTING]
  vln/scripts/manage_grouped_source_screen.sh attach RUN_TAG [--multi]
  vln/scripts/manage_grouped_source_screen.sh stop RUN_TAG [--timeout SECONDS]

Manage one detached GNU screen session around the grouped VLN Source runner.
RUNNER_OPTIONS are forwarded to run_grouped_source_eval.sh; do not repeat
--run-tag. A run tag identifies one immutable attempt and cannot be reused.

Examples:
  TAG=grouped-source-20260810T120000Z
  vln/scripts/manage_grouped_source_screen.sh start "$TAG" --gpu 0
  vln/scripts/manage_grouped_source_screen.sh status "$TAG"
  vln/scripts/manage_grouped_source_screen.sh logs "$TAG" --follow
  vln/scripts/manage_grouped_source_screen.sh attach "$TAG"
  vln/scripts/manage_grouped_source_screen.sh stop "$TAG"

Detach from an attached screen with Ctrl-a d. `attach --multi` uses screen -x
and keeps any existing attachment connected.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

validate_tag() {
    [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid run tag: $1"
    [[ "${#1}" -le 96 ]] || die "run tag is too long for a screen session"
}

ACTION="${1:-}"
case "${ACTION}" in
    -h|--help)
        usage
        exit 0
        ;;
    start|status|logs|attach|stop|_run) ;;
    '')
        usage >&2
        exit 2
        ;;
    *) die "unknown action: ${ACTION}" ;;
esac
shift

RUN_TAG="${1:-}"
[[ -n "${RUN_TAG}" ]] || die "${ACTION} requires RUN_TAG"
validate_tag "${RUN_TAG}"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${REPO_ROOT}" in
    /root/autodl-tmp/*) ;;
    *) die "refusing to run outside /root/autodl-tmp: ${REPO_ROOT}" ;;
esac

SCRIPT_PATH="${REPO_ROOT}/vln/scripts/manage_grouped_source_screen.sh"
RUNNER="${REPO_ROOT}/vln/scripts/run_grouped_source_eval.sh"
SESSION="vln_source_${RUN_TAG}"
CONTROL_ROOT="${REPO_ROOT}/vln/results/logs/grouped_source_screen/${RUN_TAG}"
SCREEN_LOG="${CONTROL_ROOT}/screen.log"
OWNER_FILE="${CONTROL_ROOT}/owner"
EXITCODE_FILE="${CONTROL_ROOT}/exitcode"
SUMMARY_FILE="${CONTROL_ROOT}/SUMMARY"
COMMAND_FILE="${CONTROL_ROOT}/command.txt"
STOP_REQUEST_DIR="${CONTROL_ROOT}/stop.requested"
GROUP_LOG_ROOT="/root/autodl-tmp/tmp/navtta-grouped-source/${RUN_TAG}"

[[ -x "${RUNNER}" ]] || die "missing grouped runner: ${RUNNER}"

screen_session_exists() {
    local token rest
    while read -r token rest; do
        if [[ "${token}" == *.* && "${token#*.}" == "${SESSION}" && \
              "${rest}" != *Dead* ]]; then
            return 0
        fi
    done < <(screen -ls 2>/dev/null || true)
    return 1
}

dead_screen_socket_exists() {
    local token rest
    while read -r token rest; do
        if [[ "${token}" == *.* && "${token#*.}" == "${SESSION}" && \
              "${rest}" == *Dead* ]]; then
            return 0
        fi
    done < <(screen -ls 2>/dev/null || true)
    return 1
}

current_boot_id() {
    [[ -r /proc/sys/kernel/random/boot_id ]] || return 1
    IFS= read -r REPLY </proc/sys/kernel/random/boot_id
    printf '%s\n' "${REPLY}"
}

process_starttime() {
    local pid="$1"
    [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "/proc/${pid}/stat" ]] || return 1
    awk '{print $22}' "/proc/${pid}/stat"
}

process_group_id() {
    local pid="$1"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
    ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]'
}

process_state() {
    local pid="$1"
    [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "/proc/${pid}/stat" ]] || return 1
    awk '{print $3}' "/proc/${pid}/stat"
}

process_is_running() {
    local state
    state="$(process_state "$1" 2>/dev/null || true)"
    [[ -n "${state}" && "${state}" != "Z" ]]
}

sweep_runner_process_group() {
    local pgid="$1"
    local attempt
    [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 0
    kill -0 -- "-${pgid}" 2>/dev/null || return 0
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for attempt in {1..50}; do
        kill -0 -- "-${pgid}" 2>/dev/null || return 0
        sleep 0.1
    done
    kill -KILL -- "-${pgid}" 2>/dev/null || true
}

process_cmdline_matches_runner() {
    local pid="$1"
    local command_line
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    [[ " ${command_line} " == *" ${RUNNER} "* ]]
}

owner_value() {
    local key="$1"
    [[ -f "${OWNER_FILE}" ]] || return 1
    awk -F= -v key="${key}" '$1 == key {print substr($0, length(key) + 2); exit}' \
        "${OWNER_FILE}"
}

runner_identity_is_live() {
    local pid expected actual owner_host owner_boot_id
    pid="$(owner_value runner_pid 2>/dev/null || true)"
    expected="$(owner_value runner_starttime 2>/dev/null || true)"
    owner_host="$(owner_value host 2>/dev/null || true)"
    owner_boot_id="$(owner_value boot_id 2>/dev/null || true)"
    [[ "${pid}" =~ ^[1-9][0-9]*$ && "${expected}" =~ ^[0-9]+$ ]] || return 1
    [[ "${owner_host}" == "$(hostname)" ]] || return 1
    [[ "${owner_boot_id}" == "$(current_boot_id)" ]] || return 1
    actual="$(process_starttime "${pid}" 2>/dev/null || true)"
    [[ "${actual}" == "${expected}" ]] || return 1
    process_is_running "${pid}" || return 1
    [[ "$(process_group_id "${pid}" 2>/dev/null || true)" == "${pid}" ]] || return 1
    process_cmdline_matches_runner "${pid}"
}

atomic_write() {
    local destination="$1"
    local value="$2"
    local temporary="${destination}.tmp.$$"
    printf '%s\n' "${value}" >"${temporary}"
    mv -f "${temporary}" "${destination}"
}

write_completion() {
    local status="$1"
    local completed_at
    completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    atomic_write "${EXITCODE_FILE}" "${status}"
    atomic_write "${SUMMARY_FILE}" \
        "completed_at=${completed_at} exitcode=${status} run_tag=${RUN_TAG}"
    printf 'completed_at=%s exitcode=%s\n' "${completed_at}" "${status}"
}

run_inside_screen() {
    local runner_pid="" runner_start manager_start status attempt ready=0
    local -a runner_args=(--run-tag "${RUN_TAG}" "$@")
    mkdir -p "${CONTROL_ROOT}"
    exec > >(tee -a "${SCREEN_LOG}") 2>&1
    manager_start="$(process_starttime "$$")"
    printf 'started_at=%s run_tag=%s session=%s git_commit=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RUN_TAG}" "${SESSION}" \
        "$(git -C "${REPO_ROOT}" rev-parse HEAD)"

    forward_signal() {
        local signal_status="$1"
        local attempt
        local fallback_pid
        trap - HUP INT TERM
        if [[ ! "${runner_pid}" =~ ^[1-9][0-9]*$ ]]; then
            while IFS= read -r fallback_pid; do
                if [[ "${fallback_pid}" =~ ^[1-9][0-9]*$ ]]; then
                    runner_pid="${fallback_pid}"
                    break
                fi
            done < <(jobs -pr)
        fi
        if [[ "${runner_pid}" =~ ^[1-9][0-9]*$ ]]; then
            kill -TERM "${runner_pid}" 2>/dev/null || true
            for attempt in {1..60}; do
                process_is_running "${runner_pid}" || break
                sleep 0.5
            done
            if process_is_running "${runner_pid}"; then
                kill -KILL "${runner_pid}" 2>/dev/null || true
            fi
            wait "${runner_pid}" 2>/dev/null || true
            sweep_runner_process_group "${runner_pid}"
        fi
        write_completion "${signal_status}"
        exit "${signal_status}"
    }
    trap 'forward_signal 129' HUP
    trap 'forward_signal 130' INT
    trap 'forward_signal 143' TERM

    set +e
    python3 -c \
        'import os, sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
        "${RUNNER}" "${runner_args[@]}" &
    runner_pid=$!
    for attempt in {1..50}; do
        if [[ "$(process_group_id "${runner_pid}" 2>/dev/null || true)" == "${runner_pid}" ]] && \
           process_cmdline_matches_runner "${runner_pid}"; then
            ready=1
            break
        fi
        if ! kill -0 "${runner_pid}" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if [[ "${ready}" -ne 1 ]]; then
        kill -TERM "${runner_pid}" 2>/dev/null || true
        wait "${runner_pid}"
        status=$?
        sweep_runner_process_group "${runner_pid}"
        [[ "${status}" -ne 0 ]] || status=125
        write_completion "${status}"
        exit "${status}"
    fi

    runner_start="$(process_starttime "${runner_pid}")"
    {
        printf 'manager_pid=%s\n' "$$"
        printf 'manager_starttime=%s\n' "${manager_start}"
        printf 'runner_pid=%s\n' "${runner_pid}"
        printf 'runner_starttime=%s\n' "${runner_start}"
        printf 'host=%s\n' "$(hostname)"
        printf 'boot_id=%s\n' "$(current_boot_id)"
        printf 'session=%s\n' "${SESSION}"
    } >"${OWNER_FILE}.tmp.$$"
    mv -f "${OWNER_FILE}.tmp.$$" "${OWNER_FILE}"

    wait "${runner_pid}"
    status=$?
    sweep_runner_process_group "${runner_pid}"
    trap - HUP INT TERM
    write_completion "${status}"
    exit "${status}"
}

start_screen() {
    local option early_completed=0
    local -a runner_options=("$@")
    command -v screen >/dev/null 2>&1 || die "GNU screen is not installed"
    for option in "${runner_options[@]}"; do
        [[ "${option}" != "--run-tag" ]] || die "RUN_TAG is positional; do not pass --run-tag"
    done
    [[ ! -e "${CONTROL_ROOT}" ]] || die "screen control directory already exists: ${CONTROL_ROOT}"
    screen_session_exists && die "screen session already exists: ${SESSION}"
    dead_screen_socket_exists && \
        die "dead screen socket exists for ${SESSION}; inspect it and run screen -wipe"

    mkdir -p "${CONTROL_ROOT}"
    {
        printf '%q ' "${RUNNER}" --run-tag "${RUN_TAG}" "${runner_options[@]}"
        printf '\n'
    } >"${COMMAND_FILE}"

    screen -dmS "${SESSION}" \
        "${SCRIPT_PATH}" _run "${RUN_TAG}" "${runner_options[@]}"

    local attempt
    for attempt in {1..50}; do
        if [[ -f "${OWNER_FILE}" || -f "${EXITCODE_FILE}" ]]; then
            break
        fi
        sleep 0.1
    done
    if [[ ! -f "${OWNER_FILE}" && ! -f "${EXITCODE_FILE}" ]]; then
        die "screen did not publish startup metadata; inspect ${SCREEN_LOG}"
    fi
    if [[ -f "${EXITCODE_FILE}" ]]; then
        local early_status
        IFS= read -r early_status <"${EXITCODE_FILE}"
        if [[ "${early_status}" != "0" ]]; then
            die "grouped runner exited during startup with status ${early_status}; inspect ${SCREEN_LOG}"
        fi
        early_completed=1
    fi

    if [[ "${early_completed}" -eq 1 ]]; then
        printf 'Grouped VLN Source evaluation completed during startup (exitcode=0).\n'
    else
        printf 'Started grouped VLN Source evaluation in screen.\n'
    fi
    printf '  run tag: %s\n' "${RUN_TAG}"
    printf '  session: %s\n' "${SESSION}"
    printf '  status:  %s status %s\n' "${SCRIPT_PATH}" "${RUN_TAG}"
    printf '  attach:  %s attach %s\n' "${SCRIPT_PATH}" "${RUN_TAG}"
    printf '  logs:    %s logs %s --follow\n' "${SCRIPT_PATH}" "${RUN_TAG}"
}

show_status() {
    local exitcode="" state screen_live=no
    if screen_session_exists; then
        screen_live=yes
    fi
    if [[ -f "${EXITCODE_FILE}" ]]; then
        IFS= read -r exitcode <"${EXITCODE_FILE}"
        if [[ "${exitcode}" == "0" ]]; then
            state=succeeded
        else
            state="failed(exitcode=${exitcode})"
        fi
    elif runner_identity_is_live && [[ "${screen_live}" == "yes" ]]; then
        if [[ -d "${STOP_REQUEST_DIR}" ]]; then
            state=stopping
        else
            state=running
        fi
    elif runner_identity_is_live; then
        state=orphaned_runner
    elif [[ "${screen_live}" == "yes" ]]; then
        state=starting_or_stopping
    elif dead_screen_socket_exists; then
        state=stale_dead_screen_socket
    elif [[ -e "${CONTROL_ROOT}" ]]; then
        state=stale_or_interrupted
    else
        state=not_found
    fi
    printf 'run_tag=%s\nstate=%s\nsession=%s\n' "${RUN_TAG}" "${state}" "${SESSION}"
    printf 'screen_live=%s\n' "${screen_live}"
    printf 'screen_log=%s\ngroup_logs=%s\n' "${SCREEN_LOG}" "${GROUP_LOG_ROOT}"
    [[ "${state}" != "not_found" ]]
}

show_logs() {
    local follow=0 setting="" log_path
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --follow) follow=1; shift ;;
            --setting)
                [[ "$#" -ge 2 && "$2" != -* ]] || die "--setting requires a value"
                setting="$2"
                shift 2
                ;;
            *) die "unknown logs option: $1" ;;
        esac
    done
    if [[ -n "${setting}" ]]; then
        case "${setting}" in
            duet-r2r|duet-reverie|hamt-r2r|hamt-reverie|goat-r2r|goat-reverie|\
            etpnav-r2r-ce|bevbert-r2r-ce|streamvln-r2r-ce) ;;
            *) die "invalid setting: ${setting}" ;;
        esac
        log_path="${GROUP_LOG_ROOT}/${setting}.log"
    else
        log_path="${SCREEN_LOG}"
    fi
    [[ -f "${log_path}" ]] || die "log is not available yet: ${log_path}"
    if [[ "${follow}" -eq 1 ]]; then
        exec tail -F "${log_path}"
    fi
    tail -n 80 "${log_path}"
}

attach_screen() {
    local multi=0
    if [[ "${1:-}" == "--multi" ]]; then
        multi=1
        shift
    fi
    [[ "$#" -eq 0 ]] || die "unknown attach option: $1"
    screen_session_exists || die "screen session is not running: ${SESSION}"
    if [[ "${multi}" -eq 1 ]]; then
        exec screen -x "${SESSION}"
    fi
    exec screen -d -r "${SESSION}"
}

stop_screen() {
    local timeout=30 runner_pid attempt owner_host owner_boot_id send_signal=0 swept=0
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --timeout)
                [[ "$#" -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || \
                    die "--timeout requires positive seconds"
                timeout="$2"
                shift 2
                ;;
            *) die "unknown stop option: $1" ;;
        esac
    done
    runner_pid="$(owner_value runner_pid 2>/dev/null || true)"
    owner_host="$(owner_value host 2>/dev/null || true)"
    owner_boot_id="$(owner_value boot_id 2>/dev/null || true)"
    [[ "${owner_host}" == "$(hostname)" ]] || \
        die "owner host does not match this machine"
    [[ "${owner_boot_id}" == "$(current_boot_id)" ]] || \
        die "owner boot identity does not match this boot"
    if ! runner_identity_is_live; then
        if [[ -f "${EXITCODE_FILE}" ]]; then
            printf 'run already completed\n'
            show_status
            return 0
        fi
        die "validated runner process is not available; inspect status and logs"
    fi

    if mkdir "${STOP_REQUEST_DIR}" 2>/dev/null; then
        send_signal=1
    fi
    if [[ "${send_signal}" -eq 1 ]]; then
        kill -TERM "${runner_pid}"
        printf 'sent TERM to grouped runner pid=%s; waiting up to %ss\n' \
            "${runner_pid}" "${timeout}"
    else
        printf 'stop was already requested; waiting up to %ss for completion\n' \
            "${timeout}"
    fi
    for ((attempt = 0; attempt < timeout * 2; attempt++)); do
        if [[ -f "${EXITCODE_FILE}" ]]; then
            show_status
            return 0
        fi
        if [[ "${swept}" -eq 0 ]] && ! runner_identity_is_live; then
            sweep_runner_process_group "${runner_pid}"
            swept=1
        fi
        sleep 0.5
    done
    die "runner is still stopping; do not kill screen directly; inspect ${SCREEN_LOG}"
}

case "${ACTION}" in
    _run) run_inside_screen "$@" ;;
    start) start_screen "$@" ;;
    status) [[ "$#" -eq 0 ]] || die "status takes no options"; show_status ;;
    logs) show_logs "$@" ;;
    attach) attach_screen "$@" ;;
    stop) stop_screen "$@" ;;
esac
