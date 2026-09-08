#!/usr/bin/env bash

# Late-decoder BN + foreground/valid-depth entropy profile.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PONI_ROOT="${PONI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

export SAVE_ROOT="${SAVE_ROOT:-$PONI_ROOT/experiments/Tent/structural}"
export TTA_LR="${TTA_LR:-3e-7}"
export TTA_EPISODIC="${TTA_EPISODIC:-true}"
export TTA_RESET_BN_STATS="${TTA_RESET_BN_STATS:-false}"
export TTA_NORM_PREFIXES="${TTA_NORM_PREFIXES:-deconv3,deconv4,agant1,agant0,final_conv}"
export TTA_ENTROPY_MODE="${TTA_ENTROPY_MODE:-foreground_valid_depth}"
export TTA_MIN_ADAPTATION_PIXELS="${TTA_MIN_ADAPTATION_PIXELS:-128}"
export TTA_UPDATE_INTERVAL="${TTA_UPDATE_INTERVAL:-4}"
export TTA_MAX_UPDATES_PER_EPISODE="${TTA_MAX_UPDATES_PER_EPISODE:-32}"
exec "$SCRIPT_DIR/run_tent_mp3d.sh" "$@"

