#!/bin/bash
#
# TTA-AVN Training, Validation, and Zero-shot Evaluation Pipeline
# Usage: bash scripts/run_tta_avn.sh <model> <source_type> <stage>
#   model:       av_nav | savi | av_wan
#   source_type: single_source | multi_source
#   stage:       train | eval | test | train_eval | full
#
# Examples:
#   bash scripts/run_tta_avn.sh av_nav single_source train       # Train av_nav on single source
#   bash scripts/run_tta_avn.sh av_nav single_source eval        # Validate on 60-scene val split (600 eps)
#   bash scripts/run_tta_avn.sh av_nav single_source test        # Zero-shot eval on 20 test scenes (best ckpt)
#   bash scripts/run_tta_avn.sh av_nav single_source full        # Train -> Eval -> Test pipeline
#   bash scripts/run_tta_avn.sh savi single_source train         # SAVi 2-stage: pretrain + finetune
#   bash scripts/run_tta_avn.sh savi single_source eval          # Validate SAVi (finetune stage)
#
# Hardware: 2 GPUs, DDPPO with GLOO backend
# Hyperparameters: NUM_PROCESSES=12, NUM_UPDATES=30000, CHECKPOINT_INTERVAL=100
#   Single source lr=1.5e-5, Multi source lr=7.5e-6

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export MASTER_PORT=${MASTER_PORT:-8640}

set -e

MODEL=${1:?  "Usage: $0 <model> <source_type> <stage>"}
SOURCE=${2:?  "Usage: $0 <model> <source_type> <stage>"}
STAGE=${3:?   "Usage: $0 <model> <source_type> <stage>"}
NUM_GPUS=${NUM_GPUS:-2}

# --- Model directories ---
case "${MODEL}" in
    av_nav)
        RUN_SCRIPT="ss_baselines/av_nav/run.py"
        TRAIN_CONFIG="ss_baselines/av_nav/config/tta_avn/${SOURCE}/train.yaml"
        EVAL_CONFIG="ss_baselines/av_nav/config/tta_avn/${SOURCE}/eval.yaml"
        TEST_CONFIG="ss_baselines/av_nav/config/tta_avn/${SOURCE}/test.yaml"
        MODEL_DIR="data/models/av_nav_tta_${SOURCE}"
        ;;
    savi)
        RUN_SCRIPT="ss_baselines/savi/run.py"
        PRETRAIN_CONFIG="ss_baselines/savi/config/tta_avn/${SOURCE}/savi_pretraining.yaml"
        TRAIN_CONFIG="ss_baselines/savi/config/tta_avn/${SOURCE}/savi.yaml"
        EVAL_CONFIG="ss_baselines/savi/config/tta_avn/${SOURCE}/savi_eval.yaml"
        TEST_CONFIG="ss_baselines/savi/config/tta_avn/${SOURCE}/savi_test.yaml"
        PRETRAIN_DIR="data/models/savi_tta_${SOURCE}_pretrain"
        MODEL_DIR="data/models/savi_tta_${SOURCE}"
        ;;
    av_wan)
        RUN_SCRIPT="ss_baselines/av_wan/run.py"
        TRAIN_CONFIG="ss_baselines/av_wan/config/tta_avn/${SOURCE}/train.yaml"
        EVAL_CONFIG="ss_baselines/av_wan/config/tta_avn/${SOURCE}/eval.yaml"
        TEST_CONFIG="ss_baselines/av_wan/config/tta_avn/${SOURCE}/test.yaml"
        MODEL_DIR="data/models/av_wan_tta_${SOURCE}"
        ;;
    *)
        echo "Error: Unknown model '${MODEL}'. Choose from: av_nav, savi, av_wan"
        exit 1
        ;;
esac

# --- Functions ---

do_train() {
    echo "=========================================="
    echo " Training ${MODEL} (${SOURCE})"
    echo "=========================================="

    if [ "${MODEL}" = "savi" ]; then
        # Stage 1: Pretraining (memory_size=1, freeze_encoders=False)
        echo "[Stage 1/2] SAVi Pretraining..."
        python -u -m torch.distributed.launch \
            --use_env \
            --nproc_per_node ${NUM_GPUS} \
            --master_port ${MASTER_PORT} \
            ${RUN_SCRIPT} \
            --exp-config ${PRETRAIN_CONFIG} \
            --model-dir ${PRETRAIN_DIR}

        # Find best pretraining checkpoint
        echo "[Stage 1/2] Finding best pretraining checkpoint..."
        BEST_PRETRAIN_CKPT=$(python -c "
import os, sys
sys.path.insert(0, '.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from ss_baselines.savi.run import find_best_ckpt_idx
idx = find_best_ckpt_idx('${PRETRAIN_DIR}/tb', max_step=30000)
if idx == -1:
    # Fallback: use the last checkpoint
    ckpts = sorted([f for f in os.listdir('${PRETRAIN_DIR}/data') if f.startswith('ckpt.')])
    if ckpts:
        idx = int(ckpts[-1].split('.')[1])
    else:
        print('ERROR: No checkpoints found', file=sys.stderr)
        sys.exit(1)
print(idx)
")
        PRETRAIN_CKPT_PATH="${PRETRAIN_DIR}/data/ckpt.${BEST_PRETRAIN_CKPT}.pth"
        echo "[Stage 1/2] Best pretraining checkpoint: ${PRETRAIN_CKPT_PATH}"

        # Stage 2: Fine-tuning (memory_size=150, freeze_encoders=True, load pretrained)
        echo "[Stage 2/2] SAVi Fine-tuning with pretrained weights..."
        python -u -m torch.distributed.launch \
            --use_env \
            --nproc_per_node ${NUM_GPUS} \
            --master_port ${MASTER_PORT} \
            ${RUN_SCRIPT} \
            --exp-config ${TRAIN_CONFIG} \
            --model-dir ${MODEL_DIR} \
            RL.DDPPO.pretrained_weights "${PRETRAIN_CKPT_PATH}"
    else
        # Single-stage training for av_nav and av_wan
        python -u -m torch.distributed.launch \
            --use_env \
            --nproc_per_node ${NUM_GPUS} \
            --master_port ${MASTER_PORT} \
            ${RUN_SCRIPT} \
            --exp-config ${TRAIN_CONFIG} \
            --model-dir ${MODEL_DIR}
    fi

    echo "Training complete. Model saved to: ${MODEL_DIR}"
}

do_eval() {
    echo "=========================================="
    echo " Validating ${MODEL} (${SOURCE})"
    echo " 60-scene val split, 600 episodes"
    echo "=========================================="

    python ${RUN_SCRIPT} \
        --run-type eval \
        --exp-config ${EVAL_CONFIG} \
        --model-dir ${MODEL_DIR}

    echo "Validation complete. Check ${MODEL_DIR}/tb for results."
}

do_test() {
    echo "=========================================="
    echo " Zero-shot Eval ${MODEL} (${SOURCE})"
    echo " 20 test scenes, best checkpoint"
    echo "=========================================="

    python ${RUN_SCRIPT} \
        --run-type eval \
        --exp-config ${TEST_CONFIG} \
        --model-dir ${MODEL_DIR} \
        --eval-best

    echo "Zero-shot evaluation complete."
}

# --- Main ---

case "${STAGE}" in
    train)
        do_train
        ;;
    eval)
        do_eval
        ;;
    test)
        do_test
        ;;
    train_eval)
        do_train
        do_eval
        ;;
    full)
        do_train
        do_eval
        do_test
        ;;
    *)
        echo "Error: Unknown stage '${STAGE}'. Choose from: train, eval, test, train_eval, full"
        exit 1
        ;;
esac
