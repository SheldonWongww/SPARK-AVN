#!/usr/bin/env bash
set -euo pipefail

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export MKL_NUM_THREADS=1
fi

REPO_ROOT=/data1/wxy/code/NavTTA
VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln
ENV_ROOT="${VLN_ROOT}/envs"
MATTERSIM_NATIVE_LIB="${ENV_ROOT}/mattersim-native/lib"
CACHE_ROOT="${VLN_ROOT}/cache"
HOME_ROOT="${VLN_ROOT}/home"
XDG_CACHE_ROOT="${CACHE_ROOT}/xdg"

mkdir -p "${HOME_ROOT}" "${XDG_CACHE_ROOT}"
export HOME="${HOME_ROOT}"
export XDG_CACHE_HOME="${XDG_CACHE_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="${CACHE_ROOT}/nltk_data"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SIM_BUILD="${REPO_ROOT}/vln/data/simulators/Matterport3DSimulator/build"
"${REPO_ROOT}/vln/scripts/build_mattersim.sh" --check

(
    prefix="${ENV_ROOT}/streamvln"
    export PATH="${prefix}/bin:${PATH}"
    export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export HF_HOME="${CACHE_ROOT}/huggingface"
    export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
    export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
    export STREAMVLN_MODEL_ROOT="${REPO_ROOT}/vln/checkpoints/streamvln/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
    cd "${REPO_ROOT}/vln/baselines/streamvln"
    export PYTHONPATH="${PWD}:${PWD}/streamvln${PYTHONPATH:+:${PYTHONPATH}}"
    "${prefix}/bin/python" - <<'PY'
import os

import decord
import flash_attn
import habitat
import navtta_core
import torch
from model.stream_video_vln import StreamVLNForCausalLM
from transformers import AutoConfig, AutoTokenizer

root = os.environ["STREAMVLN_MODEL_ROOT"]
AutoConfig.from_pretrained(root, local_files_only=True)
AutoTokenizer.from_pretrained(root, local_files_only=True)
print("streamvln imports/tokenizer passed", torch.__version__, flash_attn.__version__, decord.__version__)
PY
)

for setting in duet hamt goat; do
    (
        prefix="${ENV_ROOT}/${setting}"
        export PATH="${prefix}/bin:${PATH}"
        export LD_LIBRARY_PATH="${SIM_BUILD}:${MATTERSIM_NATIVE_LIB}:${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        case "${setting}" in
            duet)
                source_root="${REPO_ROOT}/vln/baselines/duet/map_nav_src"
                export HF_HOME="${REPO_ROOT}/vln/checkpoints/duet/.hf_cache"
                export HF_HUB_CACHE="${HF_HOME}/hub"
                export TRANSFORMERS_CACHE="${HF_HOME}/hub"
                ;;
            hamt)
                source_root="${REPO_ROOT}/vln/baselines/hamt/finetune_src"
                export HF_HOME="${CACHE_ROOT}/transformers/hamt"
                export HF_HUB_CACHE="${CACHE_ROOT}/transformers/hamt"
                export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers/hamt"
                ;;
            goat)
                source_root="${REPO_ROOT}/vln/baselines/goat/map_nav_src"
                ;;
        esac
        cd "${source_root}"
        export PYTHONPATH="${SIM_BUILD}:${source_root}${PYTHONPATH:+:${PYTHONPATH}}"
        SETTING="${setting}" "${prefix}/bin/python" - <<'PY'
import os
import sys
import MatterSim
import navtta_core
import torch

setting = os.environ["SETTING"]
sim = MatterSim.Simulator()
sim.setRenderingEnabled(False)
if setting == "duet":
    import r2r.main_nav
    import reverie.main_nav_obj
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
elif setting == "hamt":
    import r2r.main
    import reverie.main_navref
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
else:
    import spacy
    from transformers import AutoTokenizer
    spacy.load("en_core_web_sm")
    AutoTokenizer.from_pretrained("../datasets/pretrained/roberta", local_files_only=True)
print(setting, "imports/simulator/tokenizer passed", torch.__version__)
PY
        if [[ "${setting}" == goat ]]; then
            PYTHONPATH="${PYTHONPATH}:${source_root}/r2r" \
                "${prefix}/bin/python" -c \
                'import sys; output = sys.argv.pop(1); sys.argv += ["--mode", "valid", "--root_dir", "../datasets", "--output_dir", output, "--name", "offline_smoke"]; import r2r.main_nav; print("goat R2R entry import passed")' \
                "${CACHE_ROOT}/runtime_imports/goat/r2r"
            PYTHONPATH="${PYTHONPATH}:${source_root}/reverie" \
                "${prefix}/bin/python" -c \
                'import sys; output = sys.argv.pop(1); sys.argv += ["--mode", "valid", "--root_dir", "../datasets", "--output_dir", output, "--name", "offline_smoke", "--features", "clip768", "--obj_features", "vitbase"]; import reverie.main_nav_obj; print("goat REVERIE entry import passed")' \
                "${CACHE_ROOT}/runtime_imports/goat/reverie"
        fi
        "${prefix}/bin/python" -m pip check
    )
done

for setting in etpnav bevbert; do
    (
        prefix="${ENV_ROOT}/vlnce017"
        export PATH="${prefix}/bin:${PATH}"
        export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export HF_HOME="${CACHE_ROOT}/huggingface"
        export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
        export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
        export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
        export NAVTTA_CLIP_CACHE="${CACHE_ROOT}/clip"
        if [[ "${setting}" == etpnav ]]; then
            source_root="${REPO_ROOT}/vln/baselines/etpnav"
            trainer=SS-ETP
        else
            source_root="${REPO_ROOT}/vln/baselines/bevbert/bevbert_ce"
            trainer=SS-BEV
        fi
        cd "${source_root}"
        TRAINER="${trainer}" "${prefix}/bin/python" - <<'PY'
import os
import clip
import habitat
import habitat_extensions
import navtta_core
import torch
import torch_scatter
import vlnce_baselines
from habitat_baselines.common.baseline_registry import baseline_registry

name = os.environ["TRAINER"]
assert baseline_registry.get_trainer(name) is not None, name
print(name, "imports/registry passed", torch.__version__, getattr(habitat, "__version__", "unknown"))
PY
    )
done

"${ENV_ROOT}/vlnce017/bin/python" -m pip check
printf 'All VLN CPU/offline runtime imports passed\n'
