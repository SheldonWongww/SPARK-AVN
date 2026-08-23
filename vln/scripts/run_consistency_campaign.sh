#!/usr/bin/env bash
# Unattended cross-split consistency hyperparameter search campaign.
#
# Runs the phases sequentially (one benchmark's one (model,method) at a time),
# reusing the existing formal Source evals as the selection reference.  Each
# phase's candidates run at the spec's same-(model,method) concurrency.  After a
# benchmark's phases finish it writes selected_config.json per cell.
#
# Designed to be launched once under nohup on the AutoDL server; every phase
# appends to a single campaign log so progress can be tailed remotely.
#
# Usage:
#   bash vln/scripts/run_consistency_campaign.sh [BENCHMARK ...]
#     BENCHMARK in {r2r, reverie, r2r-ce}; default runs all three in order.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/root/autodl-tmp/conda/envs/duet/bin/python"   # stdlib-only runner
RUN_TAG="consistency-v1"
OUT_ROOT="${REPO_ROOT}/vln/results/tuning/consistency_v1"
LOG_DIR="${OUT_ROOT}/_campaign_logs"
mkdir -p "${LOG_DIR}"
RUNNER="${REPO_ROOT}/vln/scripts/run_consistency_hparam_search.py"

# Existing formal Source roots (already on disk; verified to match checkpoints
# used by run_source_eval.sh).  HAMT-r2r uses the e2e Source, the checkpoint the
# launcher actually loads.
SOURCE_GROUPED="${REPO_ROOT}/vln/results/source/grouped-source-20260810T080743Z"
SOURCE_HAMT_R2R="${REPO_ROOT}/vln/results/source/hamt-r2r-e2e-source-20260810T152328Z"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${LOG_DIR}/campaign.log"; }

# Build a per-benchmark source root symlinked into the layout the selector
# expects: <source_root>/<setting>/<split>/console.log
prepare_source_root() {
    local benchmark="$1"; shift
    local settings=("$@")
    local sroot="${OUT_ROOT}/_source/${benchmark}"
    mkdir -p "${sroot}"
    for setting in "${settings[@]}"; do
        local src="${SOURCE_GROUPED}/${setting}"
        [[ "${setting}" == "hamt-r2r" ]] && src="${SOURCE_HAMT_R2R}/hamt-r2r"
        ln -sfn "${src}" "${sroot}/${setting}"
    done
    echo "${sroot}"
}

run_benchmark() {
    local benchmark="$1"
    local spec settings
    case "${benchmark}" in
        r2r)
            spec="${REPO_ROOT}/vln/experiments/r2r_consistency_search_v1.json"
            settings=(duet-r2r hamt-r2r goat-r2r) ;;
        reverie)
            spec="${REPO_ROOT}/vln/experiments/reverie_consistency_search_v1.json"
            settings=(duet-reverie hamt-reverie goat-reverie) ;;
        r2r-ce)
            spec="${REPO_ROOT}/vln/experiments/r2r_ce_consistency_search_v1.json"
            settings=(etpnav-r2r-ce bevbert-r2r-ce) ;;
        *) log "unknown benchmark ${benchmark}"; return 1 ;;
    esac
    local sroot
    sroot="$(prepare_source_root "${benchmark}" "${settings[@]}")"

    local methods=(tent fstta eam feedtta atena idea)
    for setting in "${settings[@]}"; do
        for method in "${methods[@]}"; do
            log "START ${benchmark} ${setting} ${method}"
            "${PY}" "${RUNNER}" \
                --spec "${spec}" --run-tag "${RUN_TAG}" --out-dir "${OUT_ROOT}" \
                --settings "${setting}" --methods "${method}" \
                >> "${LOG_DIR}/${benchmark}-${setting}-${method}.log" 2>&1
            log "DONE  ${benchmark} ${setting} ${method} (exit $?)"
        done
    done

    log "SELECT ${benchmark}"
    "${PY}" "${RUNNER}" --spec "${spec}" --run-tag "${RUN_TAG}" \
        --out-dir "${OUT_ROOT}" --source-root "${sroot}" --select-only \
        >> "${LOG_DIR}/${benchmark}-select.log" 2>&1
    log "SELECTED ${benchmark} -> ${OUT_ROOT}/<setting>/<method>/selected_config.json"
}

BENCHMARKS=("$@")
[[ ${#BENCHMARKS[@]} -gt 0 ]] || BENCHMARKS=(r2r reverie r2r-ce)
log "CAMPAIGN START benchmarks=${BENCHMARKS[*]}"
for benchmark in "${BENCHMARKS[@]}"; do
    run_benchmark "${benchmark}"
done
log "CAMPAIGN COMPLETE"
