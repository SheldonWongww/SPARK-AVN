#!/bin/bash
#
# Run all TTA-AVN experiments: 3 models x 2 source types
# Usage: bash scripts/run_all_tta_avn.sh <stage>
#   stage: train | eval | test | train_eval | full
#
# Example:
#   bash scripts/run_all_tta_avn.sh train       # Train all 6 experiments
#   bash scripts/run_all_tta_avn.sh eval        # Validate all 6 experiments
#   bash scripts/run_all_tta_avn.sh test        # Zero-shot eval all 6 experiments
#   bash scripts/run_all_tta_avn.sh full        # Full pipeline for all

set -e

STAGE=${1:? "Usage: $0 <stage>  (train|eval|test|train_eval|full)"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=("av_nav" "savi" "av_wan")
SOURCES=("single_source" "multi_source")

for model in "${MODELS[@]}"; do
    for source in "${SOURCES[@]}"; do
        echo ""
        echo "############################################################"
        echo "# ${model} / ${source} / ${STAGE}"
        echo "############################################################"
        echo ""
        bash "${SCRIPT_DIR}/run_tta_avn.sh" "${model}" "${source}" "${STAGE}"
    done
done

echo ""
echo "============================================"
echo "All experiments completed!"
echo "============================================"
