#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PONI_ROOT="${PONI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export SAVE_ROOT="${SAVE_ROOT:-$PONI_ROOT/experiments/EAM/optimized}"
export EAM_LR="${EAM_LR:-1e-8}"
export EAM_UPDATE_INTERVAL="${EAM_UPDATE_INTERVAL:-64}"
export EAM_MAX_RELIABLE_PIXELS="${EAM_MAX_RELIABLE_PIXELS:-2048}"
export EAM_MAX_GRAD_NORM="${EAM_MAX_GRAD_NORM:-1.0}"
exec "$SCRIPT_DIR/run_eam_mp3d.sh" "$@"

