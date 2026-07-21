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

Baseline-specific environments are documented inside each baseline; no second
environment abstraction is maintained at the AVN root.
