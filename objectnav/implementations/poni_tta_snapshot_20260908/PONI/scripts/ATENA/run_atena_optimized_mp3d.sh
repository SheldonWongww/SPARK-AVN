#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PONI_ROOT="${PONI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export SAVE_ROOT="${SAVE_ROOT:-$PONI_ROOT/experiments/ATENA/optimized}"
export ATENA_LR_QUERY="${ATENA_LR_QUERY:-1e-8}"
export ATENA_LR_SELF="${ATENA_LR_SELF:-1e-9}"
export ATENA_QUERY_THRESHOLD="${ATENA_QUERY_THRESHOLD:-0.01}"
export ATENA_PARAM_SCOPE="${ATENA_PARAM_SCOPE:-bn}"
export ATENA_MAX_GRAD_NORM="${ATENA_MAX_GRAD_NORM:-1.0}"
exec "$SCRIPT_DIR/run_atena_mp3d.sh" "$@"

