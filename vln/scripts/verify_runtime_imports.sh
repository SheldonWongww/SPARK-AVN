#!/usr/bin/env bash
set -euo pipefail

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export MKL_NUM_THREADS=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${REPO_ROOT}" in
    /root/autodl-tmp/*) ;;
    *) printf 'error: refusing to run outside /root/autodl-tmp\n' >&2; exit 2 ;;
esac

export HOME=/root/autodl-tmp
export XDG_CACHE_HOME=/root/autodl-tmp/.cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA=/root/autodl-tmp/cache/nltk_data
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SIM_BUILD="${REPO_ROOT}/vln/data/simulators/Matterport3DSimulator/build"

(
    prefix=/root/autodl-tmp/conda/envs/streamvln
    export PATH="${prefix}/bin:${PATH}"
    export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export HF_HOME=/root/autodl-tmp/cache/huggingface
    export HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface/hub
    export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface/transformers
    cd "${REPO_ROOT}/vln/baselines/streamvln"
    export PYTHONPATH="${PWD}:${PWD}/streamvln${PYTHONPATH:+:${PYTHONPATH}}"
    "${prefix}/bin/python" - <<'PY'
import decord
import flash_attn
import habitat
import navtta_core
import torch
from model.stream_video_vln import StreamVLNForCausalLM
from transformers import AutoConfig, AutoTokenizer

root = "/root/autodl-tmp/code/NavTTA/vln/checkpoints/streamvln/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
AutoConfig.from_pretrained(root, local_files_only=True)
AutoTokenizer.from_pretrained(root, local_files_only=True)
print("streamvln imports/tokenizer passed", torch.__version__, flash_attn.__version__, decord.__version__)
PY
)

for setting in duet hamt goat; do
    (
        prefix="/root/autodl-tmp/conda/envs/${setting}"
        export PATH="${prefix}/bin:${PATH}"
        export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        case "${setting}" in
            duet)
                source_root="${REPO_ROOT}/vln/baselines/duet/map_nav_src"
                export HF_HOME="${REPO_ROOT}/vln/checkpoints/duet/.hf_cache"
                export HF_HUB_CACHE="${HF_HOME}/hub"
                export TRANSFORMERS_CACHE="${HF_HOME}/hub"
                ;;
            hamt)
                source_root="${REPO_ROOT}/vln/baselines/hamt/finetune_src"
                export HF_HOME=/root/autodl-tmp/cache/transformers/hamt
                export HF_HUB_CACHE=/root/autodl-tmp/cache/transformers/hamt
                export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/transformers/hamt
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
                'import sys; sys.argv += ["--mode", "valid", "--root_dir", "../datasets", "--output_dir", "/root/autodl-tmp/cache/runtime_imports/goat/r2r", "--name", "offline_smoke"]; import r2r.main_nav; print("goat R2R entry import passed")'
            PYTHONPATH="${PYTHONPATH}:${source_root}/reverie" \
                "${prefix}/bin/python" -c \
                'import sys; sys.argv += ["--mode", "valid", "--root_dir", "../datasets", "--output_dir", "/root/autodl-tmp/cache/runtime_imports/goat/reverie", "--name", "offline_smoke", "--features", "clip768", "--obj_features", "vitbase"]; import reverie.main_nav_obj; print("goat REVERIE entry import passed")'
        fi
        "${prefix}/bin/python" -m pip check
    )
done

for setting in etpnav bevbert; do
    (
        prefix=/root/autodl-tmp/conda/envs/vlnce017
        export PATH="${prefix}/bin:${PATH}"
        export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export HF_HOME=/root/autodl-tmp/cache/huggingface
        export HF_HUB_CACHE=/root/autodl-tmp/cache/huggingface/hub
        export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface/transformers
        export HF_DATASETS_CACHE=/root/autodl-tmp/cache/huggingface/datasets
        export NAVTTA_CLIP_CACHE=/root/autodl-tmp/cache/clip
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

/root/autodl-tmp/conda/envs/vlnce017/bin/python -m pip check
printf 'All VLN CPU/offline runtime imports passed\n'
