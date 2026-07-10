# SPARK-AVN: Task-Free Continual Audio-Visual Navigation

This repository provides the C-AVN benchmark and the implementation of SPARK-AVN for task-free continual audio-visual navigation.

## Data

This repository includes the C-AVN episode definitions under `data/datasets/`.
It also includes the ENMuS pretrained weights under `data/pretrained_weights/enmus/`.
Other resources should be prepared separately:

- Matterport3D scene meshes
- cached scene observations
- SoundSpaces binaural room impulse responses
- sound event and noise audio files

Expected data layout:

```text
data/
|-- binaural_rirs/
|   `-- mp3d/
|-- datasets/
|   |-- base_datasets/
|   |   |-- single_source/mp3d/v1/{train,val}/
|   |   `-- multi_source/mp3d/v1/{train,val}/
|   `-- cl_datasets/
|       |-- single_source/mp3d/v1/{train,val}/
|       `-- multi_source/mp3d/v1/{train,val}/
|-- metadata/
|   `-- mp3d/
|-- pretrained_weights/
|   `-- enmus/
|       |-- audio_encoder_best_val.pth
|       |-- enmus_best_val.pth
|       |-- visual_encoder_best_val.pth
|       |-- seld_crnn_best_val.h5
|       |-- single_source_best_val.pth
|       `-- multi_source_best_val.pth
|-- scene_datasets/
|   `-- mp3d/
|-- scene_observations/
|   `-- mp3d/
`-- sounds/
    `-- sound_event_splits/
        |-- noise/
        `-- sound_event/
```

Follow the official instructions of Matterport3D, SoundSpaces, and ENMuS to obtain these resources.

## Environment

Create the conda environment:

```bash
conda create -n cavn python=3.9 cmake=3.14.0 -y
conda activate cavn
```

Install Habitat-Sim v0.2.2 with audio support:

```bash
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout RLRAudioPropagationUpdate
python setup.py install --headless --audio
```

Install Habitat-Lab v0.2.2:

```bash
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout v0.2.2
pip install -e .
```

Install SoundSpaces:

```bash
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
pip install -e .
```

Patch `sound-spaces/soundspaces/tasks/nav.py` as in ENMuS:

```python
return np.array(
    [-agent_position_xyz[2], agent_position_xyz[0], agent_heading[0], ep_time],
    dtype=np.float32,
)
```

Install this project's Python dependencies:

```bash
cd /path/to/cavn
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Project Structure

```text
cavn/
|-- cavn/
|   |-- algo/
|   |-- cl_method/
|   |-- common/
|   |-- engin/
|   |-- env/
|   `-- model/
|-- cfgs/
|   |-- av_nav/
|   |-- cl_exp/
|   `-- pretrain/
|-- data/
|   `-- datasets/
|-- scripts/
|-- run.py
|-- run_continual_training.sh
`-- run_continual_evaluation.sh
```

## Pretraining

Train the single-source base policy:

```bash
python run.py \
    --run-type train \
    --exp-config cfgs/pretrain/single_source/train.yaml \
    --model-dir data/results/pretrain/single_source/seed_1 \
    --seed 1 \
    --overwrite
```

Evaluate the single-source base policy:

```bash
python run.py \
    --run-type eval \
    --exp-config cfgs/pretrain/single_source/val.yaml \
    --model-dir data/results/pretrain/single_source/seed_1 \
    --eval-interval 10
```

Train the multi-source base policy:

```bash
python run.py \
    --run-type train \
    --exp-config cfgs/pretrain/multi_source/train.yaml \
    --model-dir data/results/pretrain/multi_source/seed_1 \
    --seed 1 \
    --overwrite
```

Evaluate the multi-source base policy:

```bash
python run.py \
    --run-type eval \
    --exp-config cfgs/pretrain/multi_source/val.yaml \
    --model-dir data/results/pretrain/multi_source/seed_1 \
    --eval-interval 10
```

## Continual Learning

### SPARK-AVN

Train SPARK-AVN on single-source C-AVN:

```bash
bash run_continual_training.sh \
    --exp-config cfgs/cl_exp/single_source/spark_avn/train.yaml \
    --model-dir data/results/spark_avn/single_source/seed_1 \
    --seed 1 \
    --gpus 0 \
    --overwrite

for i in $(seq 2 20); do
    bash run_continual_training.sh \
        --exp-config cfgs/cl_exp/single_source/spark_avn/train.yaml \
        --model-dir data/results/spark_avn/single_source/seed_1 \
        --seed 1 \
        --gpus 0
done
```

Evaluate SPARK-AVN:

```bash
bash run_continual_evaluation.sh \
    --exp-config cfgs/cl_exp/single_source/spark_avn/val.yaml \
    --model-dir data/results/spark_avn/single_source/seed_1 \
    --eval-mode matrix \
    --gpu 0
```

Use `cfgs/cl_exp/multi_source/spark_avn/train.yaml` and `cfgs/cl_exp/multi_source/spark_avn/val.yaml` for the multi-source setting.

### Finetune Lower Bound

```bash
bash run_continual_training.sh \
    --exp-config cfgs/cl_exp/single_source/finetune/train.yaml \
    --model-dir data/results/finetune/single_source/seed_1 \
    --seed 1 \
    --gpus 0 \
    --overwrite
```

Use the corresponding `multi_source/finetune` config for the multi-source setting.

### Joint Upper Bound

```bash
python run.py \
    --run-type train \
    --exp-config cfgs/cl_exp/single_source/joint/train.yaml \
    --model-dir data/results/joint/single_source/seed_1 \
    --seed 1 \
    --overwrite
```

Use the corresponding `multi_source/joint` config for the multi-source setting.

## Metrics

Evaluate the pretrained policy on incremental domains:

```bash
python run.py \
    --run-type eval-pretrained \
    --exp-config cfgs/cl_exp/single_source/eval_pretrained_on_incremental.yaml \
    --model-dir data/results/pretrained_eval/single_source/seed_1 \
    --seed 1
```

Compute AP, F, FT, and BT:

```bash
python scripts/compute_cl_metrics_json.py \
    --eval-json data/results/spark_avn/single_source/seed_1/cl_eval_state.json \
    --base-json data/results/pretrained_eval/single_source/seed_1/pretrained_performance_single_source.json \
    --output data/results/spark_avn/single_source/seed_1/cl_metrics_summary.txt
```

## License

See `LICENSE`.
