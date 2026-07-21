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

Baseline-specific environments are documented inside each baseline; no second
environment abstraction is maintained at the AVN root.
