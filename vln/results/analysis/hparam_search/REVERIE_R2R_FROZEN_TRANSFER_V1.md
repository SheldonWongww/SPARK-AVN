# REVERIE R2R-frozen transfer results

## Scope

- Benchmark/split: discrete REVERIE `val_seen`
- Episodes: 1,423; seed: 0; action selection: argmax
- TTA batch: `vln-reverie-r2r-frozen-transfer-v1-seed0`
- TTA commit: `4e4cbd1429bb2065ba564bb931318aebf0aa326f`
- Hyperparameters were frozen from the corresponding R2R winners; REVERIE was not used for tuning.
- Source controls are the previously completed formal runs recorded in
  `vln/manifests/reverie_reused_source_controls.json`; Source was not rerun.
- Tent, FSTTA, and EAM are unsupervised TTA. FeedTTA consumes binary feedback
  after every episode; ATENA uses lazy binary-feedback queries. Their results
  should therefore be compared within the corresponding supervision setting.

All values below are percentages. Parentheses report the absolute percentage-
point change from the matched Source run for the same navigation model.

## Results

| Model | Method | SR | SPL | RGS | RGSPL |
|---|---|---:|---:|---:|---:|
| DUET | Source | 71.75 | 63.94 | 57.41 | 51.14 |
| DUET | Tent | 71.19 (-0.56) | 64.31 (+0.37) | 57.06 (-0.35) | 51.66 (+0.52) |
| DUET | FSTTA | 71.40 (-0.35) | 64.30 (+0.36) | 57.13 (-0.28) | 51.56 (+0.42) |
| DUET | EAM | 72.17 (+0.42) | 66.17 (+2.23) | 58.68 (+1.27) | 53.97 (+2.83) |
| DUET | FeedTTA | 74.14 (+2.39) | 65.02 (+1.08) | 60.15 (+2.74) | 52.57 (+1.43) |
| DUET | ATENA | **75.33 (+3.58)** | **68.35 (+4.41)** | **61.07 (+3.66)** | **55.55 (+4.41)** |
| HAMT | Source | 43.29 | 40.19 | 27.20 | 25.18 |
| HAMT | Tent | 43.71 (+0.42) | 40.88 (+0.69) | 27.48 (+0.28) | 25.61 (+0.43) |
| HAMT | FSTTA | 43.15 (-0.14) | 40.24 (+0.05) | 26.42 (-0.78) | 24.82 (-0.36) |
| HAMT | EAM | 44.91 (+1.62) | 42.18 (+1.99) | 28.95 (+1.75) | 27.26 (+2.08) |
| HAMT | FeedTTA | 45.12 (+1.83) | 41.65 (+1.46) | 27.76 (+0.56) | 25.61 (+0.43) |
| HAMT | ATENA | **54.18 (+10.89)** | **49.02 (+8.83)** | **34.08 (+6.88)** | **31.15 (+5.97)** |
| GOAT | Source | 80.74 | 73.44 | 64.93 | 58.82 |
| GOAT | Tent | 81.03 (+0.29) | 74.10 (+0.66) | 64.86 (-0.07) | 59.05 (+0.23) |
| GOAT | FSTTA | 80.74 (+0.00) | 73.51 (+0.07) | 64.86 (-0.07) | 58.80 (-0.02) |
| GOAT | EAM | 80.82 (+0.08) | 74.55 (+1.11) | 64.72 (-0.21) | **59.49 (+0.67)** |
| GOAT | FeedTTA | 80.74 (+0.00) | 73.62 (+0.18) | 65.00 (+0.07) | 59.01 (+0.19) |
| GOAT | ATENA | **81.80 (+1.06)** | **74.92 (+1.48)** | **65.14 (+0.21)** | 59.40 (+0.58) |

## Interpretation

- ATENA is strongest overall. It is best on all four metrics for DUET and
  HAMT, with especially large HAMT gains, and is best on SR/SPL/RGS for GOAT.
- EAM is the most consistently useful unsupervised method. It improves SPL and
  RGSPL on all three models and gives the best GOAT RGSPL.
- FeedTTA clearly improves DUET and HAMT, but is almost neutral on GOAT.
- Tent gives small efficiency gains but little or negative change in raw
  success/grounding on DUET and GOAT.
- FSTTA transfers poorly: HAMT and GOAT RGSPL are below Source, and the other
  gains are marginal. This is the clearest candidate for implementation or
  hyperparameter follow-up rather than a strong REVERIE conclusion.
- On the primary grounding-aware metric RGSPL, 13 of 15 model-method cells are
  above Source; the exceptions are HAMT-FSTTA (-0.36 pp) and GOAT-FSTTA
  (-0.02 pp).

These are single-seed `val_seen` transfer results, so small changes near zero
should not be treated as statistically established improvements.

## Local evidence

- Scheduler evidence and per-job launcher logs:
  `vln/results/logs/reverie/frozen_transfer/vln-reverie-r2r-frozen-transfer-v1-seed0/`
- Raw per-run outputs and TTA diagnostics:
  `vln/results/tuning/reverie/frozen_transfer/vln-reverie-r2r-frozen-transfer-v1-seed0/`
- Formal manifests:
  `vln/results/runs/vln-reverie-r2r-frozen-transfer-v1-seed0-*/manifest.json`

The batch summary reports 15/15 completed, zero failed, zero invalid, and zero
orphaned jobs. All 15 recorded exit codes are zero.
