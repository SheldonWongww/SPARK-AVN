#!/usr/bin/env bash

# Conservative RedNet profile selected after the default run showed persistent
# fast/slow drift and lower Success/SPL than Source on the first three parts.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PONI_ROOT="${PONI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

export SAVE_ROOT="${SAVE_ROOT:-$PONI_ROOT/experiments/FSTTA/optimized}"
export FSTTA_LR_FAST="${FSTTA_LR_FAST:-3e-7}"
export FSTTA_LR_SLOW="${FSTTA_LR_SLOW:-1e-5}"
export FSTTA_M="${FSTTA_M:-8}"
export FSTTA_N="${FSTTA_N:-8}"
export FSTTA_EPISODIC="${FSTTA_EPISODIC:-false}"
export FSTTA_USE_SLOW="${FSTTA_USE_SLOW:-true}"
export FSTTA_RESET_BN_STATS="${FSTTA_RESET_BN_STATS:-false}"
export FSTTA_MAX_GRAD_NORM="${FSTTA_MAX_GRAD_NORM:-1.0}"
export FSTTA_RESET_OPTIMIZER_EACH_EPISODE="${FSTTA_RESET_OPTIMIZER_EACH_EPISODE:-true}"

exec "$SCRIPT_DIR/run_fstta_mp3d.sh" "$@"

