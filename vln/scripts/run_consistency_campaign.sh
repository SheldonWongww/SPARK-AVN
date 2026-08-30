#!/usr/bin/env bash
# Staged VLN TTA campaign for one 32 GiB GPU.
#
# Per cell: val_unseen search (seeds 1/2/3) -> freeze -> winner-only
# val_seen retention (seeds 1/2/3) -> report.  A failed command stops the
# campaign.  R2R-CE completes every ETPNav cell before starting BEVBert.
set -euo pipefail

REPO_ROOT=/data1/wxy/code/NavTTA
VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln
ENV_ROOT="${VLN_ROOT}/envs"
PY="${PYTHON:-${ENV_ROOT}/duet/bin/python}"
RUN_TAG="${RUN_TAG:-consistency-v2}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/vln/results/tuning/consistency_v2}"
GPU="${GPU:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
# IDEA remains opt-in because its SHA256-pinned Source-statistics artifacts are
# intentionally untracked and therefore are not present on every checkout.
# A prepared experiment host can include ``idea`` in CAMPAIGN_METHODS; the
# Python runner fails closed if any local artifact or provenance binding is
# missing or changed.
CAMPAIGN_METHODS="${CAMPAIGN_METHODS:-tent fstta eam feedtta atena}"
# Optional whitespace-separated allowlist of model/benchmark settings.  An
# empty value keeps the historical behavior (all settings for each requested
# benchmark).  This is intentionally an allowlist rather than a positional
# "start from" flag so interrupted campaigns can resume without touching
# already completed cells.
CAMPAIGN_SETTINGS="${CAMPAIGN_SETTINGS:-}"
R2R_CE_SPEC="${R2R_CE_SPEC:-${REPO_ROOT}/vln/experiments/r2r_ce_consistency_search_v2.json}"
LOG_DIR="${OUT_ROOT}/_campaign_logs"
RUNNER="${REPO_ROOT}/vln/scripts/run_consistency_hparam_search.py"

mkdir -p "${LOG_DIR}"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${LOG_DIR}/campaign.log"; }

setting_requested() {
    local candidate="$1"
    [[ ${#REQUESTED_SETTINGS[@]} -eq 0 ]] && return 0
    local requested
    for requested in "${REQUESTED_SETTINGS[@]}"; do
        [[ "${requested}" == "${candidate}" ]] && return 0
    done
    return 1
}

run_stage() {
    local benchmark="$1"
    local setting="$2"
    local method="$3"
    local stage="$4"
    local spec="$5"
    local logfile="${LOG_DIR}/${benchmark}-${setting}-${method}.log"
    local args=(
        "${PY}" "${RUNNER}"
        --spec "${spec}" --run-tag "${RUN_TAG}" --out-dir "${OUT_ROOT}"
        --gpu "${GPU}" --settings "${setting}" --methods "${method}"
        --stage "${stage}"
    )
    log "START ${benchmark} ${setting} ${method} ${stage}"
    if ! "${args[@]}" >> "${logfile}" 2>&1; then
        log "FAILED ${benchmark} ${setting} ${method} ${stage}; see ${logfile}"
        return 1
    fi
    log "DONE ${benchmark} ${setting} ${method} ${stage}"
}

preflight_cell() {
    local benchmark="$1"
    local setting="$2"
    local method="$3"
    local spec="$4"
    local logfile="${LOG_DIR}/${benchmark}-${setting}-${method}-preflight.log"
    log "PREFLIGHT ${benchmark} ${setting} ${method}"
    "${PY}" "${RUNNER}" \
        --spec "${spec}" --run-tag "${RUN_TAG}" --out-dir "${OUT_ROOT}" \
        --gpu "${GPU}" --settings "${setting}" --methods "${method}" \
        --stage search --dry-run --launcher-preflight \
        >> "${logfile}" 2>&1
}

run_benchmark() {
    local benchmark="$1"
    local spec
    local -a settings
    case "${benchmark}" in
        r2r)
            spec="${REPO_ROOT}/vln/experiments/r2r_consistency_search_v2.json"
            settings=(duet-r2r hamt-r2r goat-r2r)
            ;;
        reverie)
            spec="${REPO_ROOT}/vln/experiments/reverie_consistency_search_v2.json"
            settings=(duet-reverie hamt-reverie goat-reverie)
            ;;
        r2r-ce)
            spec="${R2R_CE_SPEC}"
            settings=(etpnav-r2r-ce bevbert-r2r-ce)
            ;;
        *)
            log "FAILED unknown benchmark ${benchmark}"
            return 1
            ;;
    esac

    local -a filtered_settings=()
    local setting
    for setting in "${settings[@]}"; do
        if setting_requested "${setting}"; then
            filtered_settings+=("${setting}")
        fi
    done
    if [[ ${#filtered_settings[@]} -eq 0 ]]; then
        log "SKIP ${benchmark}; no requested settings"
        return 0
    fi
    settings=("${filtered_settings[@]}")

    local -a methods
    read -r -a methods <<< "${CAMPAIGN_METHODS}"
    local method stage
    # Settings are the outer barrier.  This is required for R2R-CE and is kept
    # for the discrete benchmarks to make phase ownership unambiguous.
    for setting in "${settings[@]}"; do
        for method in "${methods[@]}"; do
            if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
                preflight_cell \
                    "${benchmark}" "${setting}" "${method}" "${spec}"
                continue
            fi
            for stage in search select retention report; do
                run_stage \
                    "${benchmark}" "${setting}" "${method}" "${stage}" \
                    "${spec}"
            done
        done
    done
}

BENCHMARKS=("$@")
[[ ${#BENCHMARKS[@]} -gt 0 ]] || BENCHMARKS=(r2r reverie r2r-ce)
[[ "${GPU}" =~ ^[0-9]+$ ]] || { printf 'error: GPU must be a nonnegative integer\n' >&2; exit 2; }
case "${PREFLIGHT_ONLY}" in
    0|1) ;;
    *) printf 'error: PREFLIGHT_ONLY must be 0 or 1\n' >&2; exit 2 ;;
esac
read -r -a REQUESTED_METHODS <<< "${CAMPAIGN_METHODS}"
[[ ${#REQUESTED_METHODS[@]} -gt 0 ]] || { printf 'error: CAMPAIGN_METHODS is empty\n' >&2; exit 2; }
SEEN_METHODS=" "
for method in "${REQUESTED_METHODS[@]}"; do
    case "${method}" in
        tent|fstta|eam|feedtta|atena|idea) ;;
        *) printf 'error: unknown campaign method: %s\n' "${method}" >&2; exit 2 ;;
    esac
    case "${SEEN_METHODS}" in
        *" ${method} "*)
            printf 'error: duplicate campaign method: %s\n' "${method}" >&2
            exit 2
            ;;
    esac
    SEEN_METHODS="${SEEN_METHODS}${method} "
done
SEEN_BENCHMARKS=" "
for benchmark in "${BENCHMARKS[@]}"; do
    case "${benchmark}" in
        r2r|reverie|r2r-ce) ;;
        *) printf 'error: unknown benchmark: %s\n' "${benchmark}" >&2; exit 2 ;;
    esac
    case "${SEEN_BENCHMARKS}" in
        *" ${benchmark} "*)
            printf 'error: duplicate benchmark: %s\n' "${benchmark}" >&2
            exit 2
            ;;
    esac
    SEEN_BENCHMARKS="${SEEN_BENCHMARKS}${benchmark} "
done
declare -a REQUESTED_SETTINGS=()
if [[ -n "${CAMPAIGN_SETTINGS}" ]]; then
    read -r -a REQUESTED_SETTINGS <<< "${CAMPAIGN_SETTINGS}"
fi
SEEN_SETTINGS=" "
for setting in "${REQUESTED_SETTINGS[@]}"; do
    case "${setting}" in
        duet-r2r|hamt-r2r|goat-r2r)
            setting_benchmark="r2r"
            ;;
        duet-reverie|hamt-reverie|goat-reverie)
            setting_benchmark="reverie"
            ;;
        etpnav-r2r-ce|bevbert-r2r-ce)
            setting_benchmark="r2r-ce"
            ;;
        *)
            printf 'error: unknown campaign setting: %s\n' "${setting}" >&2
            exit 2
            ;;
    esac
    case "${SEEN_SETTINGS}" in
        *" ${setting} "*)
            printf 'error: duplicate campaign setting: %s\n' "${setting}" >&2
            exit 2
            ;;
    esac
    case "${SEEN_BENCHMARKS}" in
        *" ${setting_benchmark} "*) ;;
        *)
            printf 'error: campaign setting %s requires benchmark %s\n' \
                "${setting}" "${setting_benchmark}" >&2
            exit 2
            ;;
    esac
    SEEN_SETTINGS="${SEEN_SETTINGS}${setting} "
done
[[ -x "${PY}" ]] || { printf 'error: missing campaign Python: %s\n' "${PY}" >&2; exit 2; }
if [[ "${PREFLIGHT_ONLY}" == "0" ]]; then
    if ! CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -c \
        'import torch; assert torch.cuda.is_available(), "no CUDA device"'; then
        printf 'error: GPU preflight failed; no experiment was launched\n' >&2
        exit 2
    fi
fi
DISPLAY_SETTINGS="${CAMPAIGN_SETTINGS:-all}"
log "CAMPAIGN START benchmarks=${BENCHMARKS[*]} settings=${DISPLAY_SETTINGS} methods=${CAMPAIGN_METHODS} preflight_only=${PREFLIGHT_ONLY}"
for benchmark in "${BENCHMARKS[@]}"; do
    run_benchmark "${benchmark}"
done
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    log "CAMPAIGN PREFLIGHT COMPLETE; no simulator was started"
else
    log "CAMPAIGN COMPLETE"
fi
