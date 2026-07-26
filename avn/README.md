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
