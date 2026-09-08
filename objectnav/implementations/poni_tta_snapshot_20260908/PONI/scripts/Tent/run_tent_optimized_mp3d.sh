#!/usr/bin/env bash

# Conservative RedNet profile selected after the default continual run showed
# very low entropy, 2.5-3.1% BN drift, and lower navigation metrics than Source.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PONI_ROOT="${PONI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

export SAVE_ROOT="${SAVE_ROOT:-$PONI_ROOT/experiments/Tent/optimized}"
export TTA_LR="${TTA_LR:-3e-7}"
# Keep the same continual cross-episode protocol as the default experiment.
# Episodic reset remains available as an explicit ablation via the environment.
export TTA_EPISODIC="${TTA_EPISODIC:-false}"
export TTA_RESET_BN_STATS="${TTA_RESET_BN_STATS:-false}"
export TTA_UPDATE_INTERVAL="${TTA_UPDATE_INTERVAL:-4}"
export TTA_MAX_UPDATES_PER_EPISODE="${TTA_MAX_UPDATES_PER_EPISODE:-32}"
export TTA_MAX_GRAD_NORM="${TTA_MAX_GRAD_NORM:-1.0}"

exec "$SCRIPT_DIR/run_tent_mp3d.sh" "$@"
