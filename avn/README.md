# Audio-Visual Navigation

AVN is the active task line. The first formal comparison uses SMT+Audio and ENMuS with Source, Tent, FSTTA, EAM, FeedTTA, and ATENA under single-source and multi-source evaluation.

Current structure:

- `baselines/`: SMT+Audio and ENMuS working code, including their own configs and dependency files
- `data/`: datasets and simulator assets, preserving upstream directory names
- `checkpoints/`: local source weights and tracked provenance manifests
- `experiments/`: comparison definition, run registry, and method-specific pre-run reviews
- `results/`: generated runs plus retained legacy results
- `scripts/`: evaluation entry points and local data-link setup

Prepare baseline-local links on each machine:

```bash
python3 avn/scripts/link_local_data.py
```

Example evaluation entry points:

```bash
bash avn/scripts/eval_smt_audio.sh single_source source 0
bash avn/scripts/eval_enmus.sh multi_source tent 0
```

Re-evaluate the four pretrained Source baselines on exactly the same
seed-0, 20-scene x 100-episode streams used by the Tent searches:

```bash
bash avn/scripts/run_source_reval.sh --dry-run --seed 0 --gpus 0,1,2,3
screen -dmS avn_source_reval \
  bash avn/scripts/run_source_reval.sh --seed 0 --gpus 0,1,2,3 \
    --batch-id source-reval-v1-seed0
screen -d -r avn_source_reval
```

The GPU order is AV-Nav, SAVi, SMT+Audio, and ENMuS. Each GPU runs the
single-source and multi-source Source evaluations concurrently. Raw batch logs
are written under `avn/results/logs/source_reval/<batch-id>/`; the compact
`metrics.csv`, validated run manifests, stream fingerprints, and batch
specification are retained under
`avn/results/legacy/source_reval/<batch-id>/`. They remain legacy results until
the imported source-checkpoint provenance is completed.
If the scheduler is interrupted by HUP/INT/TERM, it deliberately retains the
batch's `.scheduler.lock`; verify that no worker remains before removing that
lock and resuming the same `--batch-id` with `--resume`.

Launch the complete Tent grid for one model and source setting (three learning
rates by four LayerNorm scopes across four detached `screen` sessions):

```bash
bash avn/scripts/run_tent_grid.sh smt_audio single_source
bash avn/scripts/run_tent_grid.sh enmus multi_source
```

Use `bash avn/scripts/run_tent_grid.sh --help` for GPU, seed, learning-rate,
episode-count, batch-id, and dry-run options. Run it from the corresponding
activated model environment; every detached session inherits that environment.

Queue additional grids without oversubscribing already-busy GPUs:

```bash
screen -dmS tent_remaining_queue \
  bash avn/scripts/queue_tent_grids.sh --wait-batch latest \
    smt_audio:multi_source enmus:single_source enmus:multi_source
```

The supervisor waits for all 12 runs in each batch to exit successfully before
launching the next batch. Queue progress remains visible inside its `screen`
session and is also written under `results/logs/tent_grid/`.

After selecting a LayerNorm scope, isolate Tent update-frequency effects with
fixed `LR=1e-7` and `UPDATE_INTERVAL={2,3,4}`:

```bash
bash avn/scripts/run_tent_interval_grid.sh \
  smt_audio single_source last_ln
```

The three runs use GPUs 0, 1, and 2 by default. Reuse the matching interval-1
run from the earlier Tent grid as the control; use `--dry-run` to inspect all
commands before launch.

Run the canonical comprehensive Tent development grid on SMT+Audio
single-source (5 learning rates x 6 update intervals x 4 LayerNorm scopes =
120 jobs):

```bash
screen -dmS tent_core_grid \
  bash avn/scripts/run_tent_core_grid.sh --jobs-per-gpu 4
```

The scheduler uses four GPUs, enforces a per-GPU concurrency limit, assigns a
unique run tag to every job, and records separate logs and exit codes. Inspect
the full matrix first with `--dry-run`. To resume an interrupted batch, reuse
its explicit batch id with `--resume`, but only after confirming that its old
scheduler and worker processes are no longer running.

Run the complete SMT+Audio single-source FSTTA grid (4 fast learning rates x
5 fast windows x 4 slow learning rates x 3 slow windows = 240 jobs):

```bash
bash avn/scripts/run_fstta_grid.sh --dry-run \
  --batch-id fstta-smt-single-v1-seed0
screen -dmS fstta_grid \
  bash avn/scripts/run_fstta_grid.sh \
    --gpus 0,1,2,3 --jobs-per-gpu 4 \
    --batch-id fstta-smt-single-v1-seed0
```

The fast learning-rate axis is `1e-8,1e-7,3e-7,1e-6`; `1e-5` and `6e-5` are
deliberately excluded. FSTTA uses separate FAST and SLOW AdamW optimizers, and
the SLOW moments persist across its episode windows. Logs are written under
`avn/results/logs/fstta_grid/<batch-id>/`. Resume an interrupted batch with the
same arguments plus `--resume` after ensuring no old worker is still running.
Signal interruption deliberately retains `.scheduler.lock`; inspect its owner,
then remove `owner` and the empty lock directory before resuming.

Run all four FSTTA mechanism-exploration suites on SMT+Audio single-source
(92 jobs total):

```bash
python3 avn/scripts/run_fstta_explorations.py all --gpus 0,1,2,3 --jobs-per-gpu 5
```

The selectable suites are `slow_boundary` (40 jobs), `fast_geometry` (24),
`slow_optimizer` (12), and `q_scaler` (16); `all` schedules all 92 jobs. Logs
are written under `avn/results/logs/fstta_exploration/<batch-id>/`.

Before launching the 24-job fixed-scope EAM intensity grid, validate the
Transformer-plus-action-head adaptation path with a short single-GPU smoke
run:

```bash
bash avn/scripts/run_eam_intensity_grid.sh --dry-run \
  --gpus 3 --jobs-per-gpu 1 --episodes 2 --smoke \
  --batch-id eam-smoke-full-scope-v2-seed0
bash avn/scripts/run_eam_intensity_grid.sh \
  --gpus 3 --jobs-per-gpu 1 --episodes 2 --smoke \
  --batch-id eam-smoke-full-scope-v2-seed0
```

After the smoke run passes, omit `--smoke` and restore `--episodes 2000` for
the complete `6 learning rates x 4 update intervals` grid. The sole adaptation
scope is `full_transformer_plus_head`; pre-Transformer encoders remain frozen.
`--gpus` accepts any non-empty list of distinct physical GPU ids, so GPU 3 can
be isolated while another method occupies GPUs 0--2.
The launcher checks only tracked worktree changes; expected datasets,
checkpoints, run manifests, and result files do not block launch. Preflight
failures are saved under `avn/results/logs/eam_intensity_grid/*.preflight.log`.

Run the joint weak-update EAM boundary grid on SMT+Audio and ENMuS
single-source (3 learning rates x 3 update intervals x 2 models = 18 jobs):

```bash
python3 avn/scripts/run_eam_boundary_grid.py --dry-run \
  --gpus 0,1,2,3 --jobs-per-gpu 2 \
  --batch-id eam-boundary-joint-v1-seed0
screen -dmS eam_boundary_joint \
  python3 avn/scripts/run_eam_boundary_grid.py \
    --gpus 0,1,2,3 --jobs-per-gpu 2 \
    --batch-id eam-boundary-joint-v1-seed0
screen -d -r eam_boundary_joint
```

The boundary is `LR={3e-9,1e-8,3e-8}` x
`UPDATE_INTERVAL={32,64,128}` for each model. Full runs are locked to the
canonical single-source val stream, seed 0, and 2000 episodes. Exactly four
distinct GPU ids are required, and `--jobs-per-gpu` is the combined per-GPU
limit across both models. Logs remain separate under
`avn/results/logs/eam_boundary_grid/<batch-id>/<model>/jobs/`. Every validated
job also copies its aggregate metrics into that job directory's `metrics.json`;
the batch root contains the combined `metrics.csv`. A full launch requires a
clean tracked worktree. `--allow-dirty` is accepted only with `--smoke`.

Resume an interrupted batch with exactly the same immutable arguments:

```bash
python3 avn/scripts/run_eam_boundary_grid.py \
  --gpus 0,1,2,3 --jobs-per-gpu 2 \
  --batch-id eam-boundary-joint-v1-seed0 --resume
```

For a short pathway check, add `--smoke --episodes 2`; smoke mode selects one
configuration per model. It does not replace the fixed 18-job search plan.

FeedTTA uses a two-stage, joint SMT+Audio/ENMuS single-source search. Stage 1
calibrates six learning rates by four discount factors (48 jobs):

```bash
python3 avn/scripts/run_feedtta_stage1.py --dry-run \
  --gpus 0,1,2,3 --jobs-per-gpu 2 \
  --batch-id feedtta-stage1-v1-seed0
screen -dmS avn_feedtta_stage1 \
  python3 avn/scripts/run_feedtta_stage1.py \
    --gpus 0,1,2,3 --jobs-per-gpu 2 \
    --batch-id feedtta-stage1-v1-seed0
```

After reviewing each model's Stage-1 winner, Stage 2 holds those model-specific
`LR/gamma` values fixed and runs the official 35-point `p/alpha` grid plus four
mechanism controls per model (78 jobs):

```bash
python3 avn/scripts/run_feedtta_stage2.py --dry-run \
  --stage1-batch-id feedtta-stage1-v1-seed0 \
  --smt-audio-lr 3e-8 --smt-audio-gamma 0.95 \
  --enmus-lr 3e-7 --enmus-gamma 1.0 \
  --gpus 0,1,2,3 --jobs-per-gpu 2 \
  --batch-id feedtta-stage2-v1-seed0
```

These are the reviewed Stage-1 choices: SMT+Audio job 5 and provisional ENMuS
job 39. Stage 2 retains the known ENMuS parameter-count warning while preserving
complete navigation metrics; other validation failures remain fatal. Both
schedulers use one combined per-GPU quota, require a clean tracked worktree for
full runs, revalidate artifacts on resume, and write separate logs and combined metrics under
`avn/results/logs/feedtta_stage{1,2}/`. See
[`experiments/FEEDTTA_EXPERIMENT_PLAN.md`](experiments/FEEDTTA_EXPERIMENT_PLAN.md)
for the fixed protocol, paper ambiguities, selection rule, controls, and smoke
commands.

ATENA uses the official greedy-action protocol, so its batch first includes a
matched Source-argmax control for each model. The complete single-source search
contains 144 ATENA points/model plus those two controls (290 jobs total):

```bash
python3 avn/scripts/run_atena_grid.py --dry-run \
  --gpus 0,1,2,3 --jobs-per-gpu 1 \
  --batch-id atena-grid-v1-seed0
```

Before the full launch, run the released DUET-R2R anchor for 1, 5, and 20
episodes using `--smoke --episodes N` and distinct batch ids. The server should
choose `--jobs-per-gpu` only after the 20-episode run confirms safe GPU/CPU
memory. ATENA stores detached trajectories on CPU and reconstructs the exact
episode gradient one step at a time, avoiding the official implementation's
episode-length GPU graph growth. Full logs and validated metrics are written to
`avn/results/logs/atena_grid/<batch-id>/`. See
[`experiments/ATENA_PRE_RUN_REVIEW.md`](experiments/ATENA_PRE_RUN_REVIEW.md)
for the alignment audit and selection constraints.

After freezing the complete FAST/SLOW candidate from exploration job 35, run
the three remaining FSTTA main-table jobs concurrently:

```bash
python3 avn/scripts/run_fstta_main.py --dry-run \
  --gpus 0,1,2 --seed 0 --batch-id fstta-main-v1-seed0
screen -dmS fstta_main \
  python3 avn/scripts/run_fstta_main.py \
    --gpus 0,1,2 --seed 0 --batch-id fstta-main-v1-seed0
screen -d -r fstta_main
```

The fixed plan is SMT+Audio multi-source, ENMuS single-source, and ENMuS
multi-source. It uses the same configuration for all three jobs:
`last_k_ln=4`, FAST `LR=3e-7, M=16`, SLOW `LR=1e-4, N=32, q=0.1`,
concordant gradients, FAST LR scaling, and persistent AdamW SLOW state. The
runner records immutable batch inputs, validates each run manifest, and writes
aggregate metrics under `avn/results/logs/fstta_main/<batch-id>/`. It does not
rerun the SMT+Audio single-source development result; that result still needs
a final-commit confirmation run before its provisional marker can be removed.

ENMuS uses a separately calibrated FSTTA configuration. Run its frozen
multi-source revalidation with the single-source grid's selected job 0:

```bash
python3 avn/scripts/run_fstta_enmus_multi.py --dry-run \
  --gpus 0 --jobs-per-gpu 1 \
  --batch-id fstta-enmus-multi-v1-seed0
screen -dmS fstta_enmus_multi \
  python3 avn/scripts/run_fstta_enmus_multi.py \
    --gpus 0 --jobs-per-gpu 1 \
    --batch-id fstta-enmus-multi-v1-seed0
screen -d -r fstta_enmus_multi
```

This one-job runner fixes FAST `LR=1e-8, M=16`, SLOW
`LR=1e-5, N=32, q=0.1`, concordant gradients, FAST LR scaling, persistent
AdamW SLOW state, the canonical multi-source val stream, seed 0, and 2000
episodes. It also pins the source and auxiliary checkpoint hashes and rejects
tracked worktree changes.

After freezing the Tent configuration, run the two multi-source main-table
jobs concurrently on SMT+Audio and ENMuS:

```bash
bash avn/scripts/run_tent_main_multi_source.sh --dry-run \
  --gpus 0,1 --seed 0 --batch-id tent-main-multi-v1-seed0
screen -dmS tent_main_multi \
  bash avn/scripts/run_tent_main_multi_source.sh \
    --gpus 0,1 --seed 0 --batch-id tent-main-multi-v1-seed0
screen -d -r tent_main_multi
```

This runner fixes `NORM_SCOPE=ln`, `LR=1e-8`, `UPDATE_INTERVAL=1`,
`EPISODIC=False`, and `STEPS=1`. It only evaluates `multi_source`; selected
single-source results remain in their original hyperparameter-search batches.
Raw scheduler/job logs and `metrics.csv` are written under
`avn/results/logs/tent_main_multi/<batch-id>/`. Each evaluation also creates a
run manifest, compact `summary.json`, and diagnostics under
`avn/results/runs/<run-id>/`.

Baseline-specific environments are documented inside each baseline; no second
environment abstraction is maintained at the AVN root.
