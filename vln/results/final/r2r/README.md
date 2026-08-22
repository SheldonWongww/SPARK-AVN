# R2R final selection registry

> **Status (2026-08-22): superseded as the primary cross-split estimate.**
> The tables in this directory use the canonical order seed 0, whose
> scene-blocked continual stream is much more correlated on `val_unseen` than
> on `val_seen`.  Keep these files as immutable stress-test evidence; do not
> cite their `val_unseen` column as the final R2R main result.  The replacement
> protocol is registered in
> [`R2R_CROSS_SPLIT_ROBUST_REEVALUATION_V1_PLAN.md`](../../../experiments/R2R_CROSS_SPLIT_ROBUST_REEVALUATION_V1_PLAN.md).

This directory freezes the best `val_seen`, order-seed-0 R2R configuration for
Source and the `3 models × 5 TTA methods` matrix. It is the transfer source for
REVERIE, not a publication-final estimate: the same split selected the
hyperparameters, and independent split/order robustness has not been measured.

## Frozen results

Values are `SR / SPL` in percentage points.

| Model | Source | Tent | FSTTA | EAM | FeedTTA | ATENA |
|---|---:|---:|---:|---:|---:|---:|
| DUET | 78.84 / 72.88 | 78.94 / 73.08 | 79.14 / 73.10 | 80.31 / 74.75 | 79.33 / 73.88 | 80.41 / 76.30 |
| HAMT | 75.61 / 72.18 | 76.30 / 72.88 | 76.40 / 72.97 | 76.69 / 73.40 | 76.00 / 72.66 | 77.47 / 74.16 |
| GOAT | 84.82 / 80.05 | 84.92 / 80.32 | 84.92 / 80.21 | 85.31 / 80.55 | 84.92 / 80.11 | 84.92 / 80.24 |

Tent, FSTTA, and EAM are unsupervised TTA. FeedTTA and ATENA consume binary
episode navigation feedback and are reported separately by supervision type in
the machine-readable registry.

`selected_winners.json` is the reviewed selection input. The five formerly
pending EAM/ATENA cells completed in
`vln-r2r-eam-atena-formal-confirm-v1-seed0`; their canonical formal manifests
and SR/SPL values are now pinned, so `selection_status=complete`.

`registry.json` was generated only after all 18 Source/TTA entries had
completed canonical formal manifests. The builder remains fail-closed and
exits nonzero before writing anything if a future selection contains a pending
entry:

```bash
python3 vln/scripts/build_r2r_final_registry.py
```

The command atomically creates `registry.json`. Validate it and all pinned
provenance with:

```bash
python3 vln/scripts/build_r2r_final_registry.py --validate
python3 vln/tests/test_r2r_final_registry.py
```

The generated schema is `navtta.vln_r2r_final_registry.v1`. Every
`records.<r2r-setting>.<method>` entry includes effective parameters, run tag,
SR/SPL metrics, delta from that model's Source result, supervision category,
checkpoint digest, canonical formal-manifest path/digest, and immutable run
identity. Only manifests under `vln/results/runs/` are accepted. Raw result
artifacts remain untracked; their path, size, and SHA256 metadata are checked
inside each immutable manifest.

## Two-split benchmark result

After the frozen `val_unseen` campaign has completed and been downloaded, the
offline builder combines its 15 TTA jobs with the three reuse-only Source
controls and the frozen `val_seen` registry:

```bash
python3 vln/scripts/build_r2r_benchmark_results.py \
  --batch-root vln/results/logs/r2r/frozen_val_unseen/vln-r2r-val-unseen-frozen-eval-v1-seed0 \
  --check-only

python3 vln/scripts/build_r2r_benchmark_results.py \
  --batch-root vln/results/logs/r2r/frozen_val_unseen/vln-r2r-val-unseen-frozen-eval-v1-seed0
```

The download must retain the complete batch directory (including every
attempt's `metrics.json` and `parameters.json`) and the 15 corresponding
formal manifests under `vln/results/runs/`. Raw tuning outputs are not needed
by this builder: it verifies the metric and diagnostics hashes recorded by
`metrics.json` against the immutable formal manifest.

The only generated files are `benchmark_results.json` and
`BENCHMARK_RESULTS.md` in this directory. The builder never launches a run or
modifies downloaded evidence. It rejects incomplete matrices, Source
execution, parameter drift from the `val_seen` selection, unauthenticated
metrics, and any checkpoint, dataset, episode-order, commit, or immutable-run
identity mismatch. Use `--validate` after generation to rebuild in memory and
compare both files with their evidence.
