# R2R `val_unseen` frozen evaluation

This campaign is an independent-split evaluation of the completed R2R
`val_seen`, canonical-order-seed-0 selection registry. It is not another
hyperparameter search. Every one of the 15 model-method cells is evaluated and
reported; no result on `val_unseen` may replace or reselect a frozen
configuration.

## Immutable inputs

- Selection registry: `vln/results/final/r2r/registry.json`
- Search specification: `r2r_val_unseen_frozen_eval_v1.json`
- Source reuse ledger:
  `vln/manifests/r2r_val_unseen_reused_source_controls.json`
- DUET/HAMT episode order:
  `vln/manifests/episode_order/r2r_duet_hamt/val_unseen.json`
- GOAT episode order:
  `vln/manifests/episode_order/r2r_goat/val_unseen.json`

Both model-family manifests contain 2,349 episodes with the same ordered-ID
digest:

```text
bfaa25c07a8755e67585b1cde3c99833377005ea70017107457761e8ad82a390
```

The scientific protocol calls this canonical order seed 0. The existing base
runner deliberately reserves the `--order-seed` option for complete
`val_seen` robustness runs, so the `val_unseen` commands must **not** pass
`--order-seed 0`. Seed 0 is enforced instead by the tracked specification,
the canonical manifest digest, the model seed recorded in each formal run
manifest, and the absence of any noncanonical order argument.

## Matrix and scheduling

The campaign executes exactly 15 TTA jobs and zero Source jobs:

```text
DUET: Tent, FSTTA, EAM, FeedTTA, ATENA (parallel, maximum 5)
  barrier
HAMT: Tent, FSTTA, EAM, FeedTTA, ATENA (parallel, maximum 5)
  barrier
GOAT: Tent, FSTTA, EAM, FeedTTA, ATENA (parallel, maximum 5)
```

The runner reads each parameter object verbatim from the pinned final registry
and verifies the selected run tag and formal-manifest digest declared by the
specification. Generated TTA configs contain no `order_seed`, episode prefix,
or Source/control job. FeedTTA and ATENA retain their binary-navigation-success
supervision category; the result validator requires the corrected R2R eager
and lazy evaluator endpoints respectively.

## Source reuse and recovered HAMT evidence

Source execution is forbidden. Existing `val_unseen` controls are:

| Model | Source SR / SPL | Evidence state |
|---|---:|---|
| DUET | 71.52 / 60.41 | Formal manifest recovered and tracked |
| HAMT E2E | 66.24 / 61.51 | Formal manifest recovered and tracked |
| GOAT | 78.12 / 67.58 | Formal manifest recovered and tracked |

The HAMT formal manifest was initially absent locally. The runner's fail-closed
preflight rejected formal execution before it created a batch or started a
worker. The already-produced manifest has now been recovered from the
experiment server without rerunning Source:

```text
vln/results/runs/hamt-r2r-e2e-source-20260810T152328Z-hamt-r2r-val_unseen-native/manifest.json
```

Its SHA256 is `89b08dd6...78d3c3`; its immutable identity, checkpoint digest
`cb3c37b3...b9dd657`, dataset/order digests, and existing `valid.txt` hash all
validate. All three Source records are now reuse-ready. The fail-closed path
remains covered by a synthetic missing-manifest unit test.

## Commands

Review the immutable plan with:

```bash
python3 vln/scripts/run_r2r_val_unseen_frozen_eval.py \
  --plan-only --print-commands
```

Formal execution uses the complete reuse-only ledger and still requires a
clean committed tree plus explicit review confirmation:

```bash
python3 vln/scripts/run_r2r_val_unseen_frozen_eval.py \
  --confirm-reviewed
```

Resume and retry use immutable batch identity and a new `-retryN` run tag:

```bash
python3 vln/scripts/run_r2r_val_unseen_frozen_eval.py \
  --resume --confirm-reviewed
python3 vln/scripts/run_r2r_val_unseen_frozen_eval.py \
  --resume --retry-failed --confirm-reviewed
```

Formal launch also requires a clean tracked worktree. Raw tuning outputs and
logs stay ignored; completed formal manifests remain trackable under
`vln/results/runs/`.
