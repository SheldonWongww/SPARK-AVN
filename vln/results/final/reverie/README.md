# REVERIE final `val_seen` selection

This directory freezes the best canonical-order, seed-0 `val_seen`
configuration for Source and the `3 models x 5 TTA methods` matrix. The
hyperparameters were selected only on `val_seen`; downstream `val_unseen` and
hidden-test runs must not change them.

Values below are `SR / SPL / RGS / RGSPL` in percentage points.

| Model | Source | Tent | FSTTA | EAM | FeedTTA | ATENA |
|---|---:|---:|---:|---:|---:|---:|
| DUET | 71.75 / 63.94 / 57.41 / 51.14 | 71.26 / 64.53 / 56.92 / 51.73 | 71.40 / 64.30 / 57.13 / 51.56 | 72.17 / 66.17 / 58.68 / 53.97 | 74.14 / 65.02 / 60.15 / 52.57 | 75.33 / 68.35 / 61.07 / 55.55 |
| HAMT | 43.29 / 40.19 / 27.20 / 25.18 | 43.71 / 40.88 / 27.48 / 25.61 | 45.40 / 42.49 / 28.74 / 27.00 | 44.91 / 42.18 / 28.95 / 27.26 | 45.12 / 41.65 / 27.76 / 25.61 | 54.18 / 49.02 / 34.08 / 31.15 |
| GOAT | 80.74 / 73.44 / 64.93 / 58.82 | 81.17 / 74.69 / 64.93 / 59.49 | 80.89 / 73.60 / 65.00 / 58.89 | 80.82 / 74.55 / 64.72 / 59.49 | 80.89 / 73.70 / 65.35 / 59.31 | 86.79 / 77.03 / 70.63 / 62.36 |

Tent, FSTTA, and EAM are unsupervised. FeedTTA and ATENA consume binary
episode-level navigation-success feedback and must be reported as a separate
supervision category.

The small REVERIE search replaced the prior transferred winner for DUET–Tent,
HAMT–FSTTA, and all four targeted GOAT cells. It retained the prior winner for
DUET–FSTTA and HAMT–FeedTTA. The machine-readable registry contains the exact
parameters, Source deltas, formal-manifest hashes, checkpoint identity, and
selection provenance for all 18 Source/TTA records.

Validate and materialize the frozen downstream protocol with:

```bash
python3 vln/scripts/build_reverie_final_registry.py --validate
python3 vln/scripts/prepare_reverie_val_unseen_frozen_eval.py --check-only
```

The native REVERIE `test` split has hidden ground truth. Source plus
Tent/FSTTA/EAM can produce submissions, but FeedTTA and ATENA are `N/A` there
unless a legitimate online binary-success feedback interface is provided.
