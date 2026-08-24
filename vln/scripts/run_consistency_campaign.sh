#!/usr/bin/env bash
# Fail-closed staged VLN TTA campaign for one 32 GiB GPU.
#
# Per cell: val_unseen search (seeds 1/2/3) -> freeze -> winner-only
# val_seen retention (seeds 1/2/3) -> report.  A failed command stops the
# campaign.  R2R-CE completes every ETPNav cell before starting BEVBert.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-/root/autodl-tmp/conda/envs/duet/bin/python}"
RUN_TAG="${RUN_TAG:-consistency-v2}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/vln/results/tuning/consistency_v2}"
GPU="${GPU:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
# IDEA is intentionally not in the default set while its precomputed Source
# statistics have no pinned artifacts. Requesting CAMPAIGN_METHODS=idea fails
# closed in the Python runner instead of launching the warmup approximation.
CAMPAIGN_METHODS="${CAMPAIGN_METHODS:-tent fstta eam feedtta atena}"
LOG_DIR="${OUT_ROOT}/_campaign_logs"
RUNNER="${REPO_ROOT}/vln/scripts/run_consistency_hparam_search.py"

mkdir -p "${LOG_DIR}"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${LOG_DIR}/campaign.log"; }

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
            spec="${REPO_ROOT}/vln/experiments/r2r_ce_consistency_search_v2.json"
            settings=(etpnav-r2r-ce bevbert-r2r-ce)
            ;;
        *)
            log "FAILED unknown benchmark ${benchmark}"
            return 1
            ;;
    esac

    local -a methods
    read -r -a methods <<< "${CAMPAIGN_METHODS}"
    local setting method stage
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
[[ -x "${PY}" ]] || { printf 'error: missing campaign Python: %s\n' "${PY}" >&2; exit 2; }
if [[ "${PREFLIGHT_ONLY}" == "0" ]]; then
    if ! CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" -c \
        'import torch; assert torch.cuda.is_available(), "no CUDA device"'; then
        printf 'error: GPU preflight failed; no experiment was launched\n' >&2
        exit 2
    fi
fi
log "CAMPAIGN START benchmarks=${BENCHMARKS[*]} methods=${CAMPAIGN_METHODS} preflight_only=${PREFLIGHT_ONLY}"
for benchmark in "${BENCHMARKS[@]}"; do
    run_benchmark "${benchmark}"
done
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    log "CAMPAIGN PREFLIGHT COMPLETE; no simulator was started"
else
    log "CAMPAIGN COMPLETE"
fi
