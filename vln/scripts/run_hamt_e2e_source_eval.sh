#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: vln/scripts/run_hamt_e2e_source_eval.sh [--gpu INDEX]
                                                [--run-tag TAG]
                                                [--dry-run]

Run the released final HAMT checkpoints on R2R and REVERIE.  Each benchmark
uses the canonical val_seen -> val_unseen -> test order, and every split starts
from its source checkpoint in a fresh process.  The R2R setting uses the
vitbase-finetune-e2e checkpoint and matching r2r.e2e.ft.22k visual features;
REVERIE uses its released checkpoint and matching R2R-e2e visual features.

This is a foreground runner.  Invoke it inside GNU screen for a persistent
background run.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

GPU=0
RUN_TAG="hamt-e2e-source-$(date -u +%Y%m%dT%H%M%SZ)"
DRY_RUN=0
GPU_SET=0
TAG_SET=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --gpu)
            [[ "$#" -ge 2 && "$2" != --* ]] || die "--gpu requires a value"
            [[ "${GPU_SET}" -eq 0 ]] || die "--gpu specified more than once"
            [[ "$2" =~ ^[0-9]+$ ]] || die "GPU index must be a nonnegative integer"
            GPU="$2"
            GPU_SET=1
            shift 2
            ;;
        --run-tag)
            [[ "$#" -ge 2 && "$2" != --* ]] || die "--run-tag requires a value"
            [[ "${TAG_SET}" -eq 0 ]] || die "--run-tag specified more than once"
            RUN_TAG="$2"
            TAG_SET=1
            shift 2
            ;;
        --dry-run)
            [[ "${DRY_RUN}" -eq 0 ]] || die "--dry-run specified more than once"
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

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid run tag: ${RUN_TAG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${REPO_ROOT}" in
    /root/autodl-tmp/*) ;;
    *) die "refusing to run outside /root/autodl-tmp: ${REPO_ROOT}" ;;
esac

SOURCE_RUNNER="${REPO_ROOT}/vln/scripts/run_source_eval.sh"
for setting in hamt-r2r hamt-reverie; do
    command=(
        "${SOURCE_RUNNER}" "${setting}" all "${GPU}"
        --run-tag "${RUN_TAG}"
    )
    [[ "${DRY_RUN}" -eq 0 ]] || command+=(--dry-run)
    printf '\n[HAMT e2e] %s\ncommand:' "${setting}"
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}"
done

printf '\nHAMT e2e Source evaluation completed: %s\n' "${RUN_TAG}"
