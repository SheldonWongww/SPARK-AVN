# REVERIE post-search evaluation plan

This plan starts only after
`vln-reverie-val-seen-small-hparam-search-v1-seed0` has completed and its
entire scheduler batch, selected full-run result roots (including each
manifest-authenticated `valid.txt`), and canonical formal manifests have been
downloaded. It does not authorize another Source run.

## 1. Freeze the `val_seen` winners offline

The small search selects by full-split `RGSPL`, then `RGS`, `SPL`, and `SR`,
subject to the configured Source-SR floor. Its final selection already chooses
between every authenticated new full candidate and the authenticated transfer
incumbent. Build the portable 3-model x 5-method registry directly from that
output:

```bash
python3 vln/scripts/build_reverie_final_registry.py \
  --from-search \
  vln/results/logs/reverie/hparam_search/vln-reverie-val-seen-small-hparam-search-v1-seed0/FINAL_SELECTION.json
```

This writes `vln/results/final/reverie/selected_winners.json` and
`vln/results/final/reverie/registry.json`. The builder is offline and
fail-closed: it requires all 15 winner manifests under `vln/results/runs/`,
recomputes immutable identities, checks checkpoint and episode-order digests,
and verifies each four-metric Source delta. It never launches an experiment.

Review the generated files, then materialize the immutable downstream plan:

```bash
python3 vln/scripts/prepare_reverie_val_unseen_frozen_eval.py
```

Commit the two final-registry files and the generated
`vln/experiments/reverie_val_unseen_frozen_eval_v1.json` before any formal
launch. The execution runner rejects an untracked plan or dirty tracked tree.

## 2. Run frozen `val_unseen`

The complete split contains 3,521 episodes. All runs use the canonical tracked
order whose SHA256 is
`67c909272166bb3d8e7ee612d597de190f9218a6342da6357b0c7aea8832baca`;
the model/action seed is 0 and no order-changing CLI flag is permitted.

The batch is exactly 15 TTA jobs and zero Source jobs. Existing Source results
are pinned by `vln/manifests/reverie_val_unseen_reused_source_controls.json`.
Execution is model-major with strict barriers:

1. DUET: five methods, at most two concurrent jobs.
2. HAMT: five methods, at most two concurrent jobs.
3. GOAT: five methods, at most four concurrent jobs.

The scheduler also applies per-method memory reservations and a 29,000 MiB
aggregate cap, so these are upper bounds rather than unconditional launch
counts.

```bash
python3 vln/scripts/run_reverie_val_unseen_frozen_eval.py \
  --plan-only --print-commands

python3 vln/scripts/run_reverie_val_unseen_frozen_eval.py \
  --confirm-reviewed
```

Use `--status` for observation and `--resume --retry-failed` for a reviewed
retry. A retry receives a new run tag. Completion requires 15 authenticated
formal manifests, 15 diagnostics files, and exactly one full-split metric row
(`SR`, `SPL`, `RGS`, `RGSPL`) per job. There is no selection or promotion on
`val_unseen`.

## 3. Generate native hidden-test submissions

REVERIE's native `test` split contains 6,292 episodes and hides ground truth.
It can generate submissions but cannot produce local metrics or support local
selection/ranking. The three existing Source submissions are reused from
`vln/manifests/reverie_test_reused_source_submissions.json`; Source is not run
again.

Tent, FSTTA, and EAM need no episode label, so the runner can generate nine
frozen TTA submissions (three models x three methods):

```bash
python3 vln/scripts/run_reverie_val_unseen_frozen_eval.py \
  --test-submissions --batch-id vln-reverie-test-submissions-v1-seed0 \
  --plan-only --print-commands

python3 vln/scripts/run_reverie_val_unseen_frozen_eval.py \
  --test-submissions --batch-id vln-reverie-test-submissions-v1-seed0 \
  --confirm-reviewed
```

The test path validates the exact 6,292-ID canonical order and emits only
submission metadata; its summary has `local_metrics: null`. FeedTTA and ATENA
are `N/A` and fail closed because the hidden server does not expose legal
per-episode online binary navigation-success feedback. They must not be run by
deriving labels from simulator distance, grounding success, or post-hoc test
scores. Enabling them would require a separately reviewed, legitimate online
feedback interface and a new protocol/specification.

## 4. Completion criteria

- `val_seen`: frozen registry contains Source plus all 15 selected TTA cells.
- `val_unseen`: three reused Source records plus all 15 frozen TTA results.
- `test`: three reused Source submissions plus nine unsupervised-TTA
  submissions; FeedTTA and ATENA remain explicitly `N/A`.
- No hidden-test metric is reported, and no `val_unseen` or test result changes
  a frozen hyperparameter.
