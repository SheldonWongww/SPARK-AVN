#!/usr/bin/env bash

# Late-decoder BN + foreground/valid-depth entropy profile.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PONI_ROOT="${PONI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

export SAVE_ROOT="${SAVE_ROOT:-$PONI_ROOT/experiments/FSTTA/structural}"
export FSTTA_LR_FAST="${FSTTA_LR_FAST:-3e-7}"
export FSTTA_LR_SLOW="${FSTTA_LR_SLOW:-1e-5}"
export FSTTA_M="${FSTTA_M:-8}"
export FSTTA_N="${FSTTA_N:-8}"
export FSTTA_EPISODIC="${FSTTA_EPISODIC:-false}"
export FSTTA_USE_SLOW="${FSTTA_USE_SLOW:-true}"
export FSTTA_RESET_BN_STATS="${FSTTA_RESET_BN_STATS:-false}"
export FSTTA_NORM_PREFIXES="${FSTTA_NORM_PREFIXES:-deconv3,deconv4,agant1,agant0,final_conv}"
export FSTTA_ENTROPY_MODE="${FSTTA_ENTROPY_MODE:-foreground_valid_depth}"
export FSTTA_MIN_ADAPTATION_PIXELS="${FSTTA_MIN_ADAPTATION_PIXELS:-128}"
exec "$SCRIPT_DIR/run_fstta_mp3d.sh" "$@"

