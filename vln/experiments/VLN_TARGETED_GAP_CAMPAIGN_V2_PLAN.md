# VLN targeted gap campaign v2

## Purpose

Run only the 16 currently blank, non-StreamVLN, non-`OURS` VLN cells. The
candidate matrix, datasets, checkpoints, seeds, metrics, and queue assignment
remain pinned by the superseded v1 design. Version 2 changes one operational
decision: existing Source results are reporting references, not a launch or
winner-selection dependency.

## Executed matrix

- 1,024 complete `val_unseen` candidate runs, seed 0: 64 candidates for
  each of the 16 cells.
- One winner per each of the 16 cells, selected only from `val_unseen`.
- 16 fresh `val_seen` runs using the frozen winners.
- Five REVERIE test submissions in a later, separate phase.
- StreamVLN and `OURS` remain excluded.

Each GPU owns four fixed cell queues. A cell runs one candidate at a time;
different cells assigned to the same GPU may run concurrently. Development
job counts on GPUs 0, 1, 2, and 3 are 256 each.

ETPNav and BEVBert continue to use R2R-CE v1.2-native. Their existing Source
results may be compared after the campaign, but Source values and ledgers are
not read by candidate selection and are not frozen into winner evidence.
The three IDEA cells consume the three reviewed, Git-tracked offline
source-training-statistics JSON files at the paths pinned by the base spec.
The launcher verifies their content before it creates a campaign batch.

## Low-learning-rate search spaces

Every cell expands an ordered Cartesian grid with exactly 64 candidates. The
non-LR axes are restricted to dimensions already exercised in the imported
AutoDL analysis; model code and method semantics are unchanged.

| Method/grid | Learning-rate axis | Other axis |
|---|---|---|
| EAM (REVERIE/R2R/R2R-CE) | `5e-8,1e-7,2e-7,4e-7,6e-7,8e-7,1e-6,2e-6` | 8 replay profiles over memory/batch, confidence and interval |
| FeedTTA (REVERIE/R2R) | `1e-7,2e-7,4e-7,6e-7,8e-7,1e-6,2e-6,4e-6` | 8 previously used gamma/SGR profiles |
| IDEA (REVERIE/R2R) | `1e-6,3e-6,1e-5,3e-5,1e-4,2e-4,4e-4,8e-4` | `tau=0.3..1.0` in steps of `0.1` |
| Tent (R2R) | `1e-8,3e-8,1e-7,3e-7,6e-7,1e-6,2e-6,5e-6` | all/last-4/last-9/last-15 LayerNorm, interval 1/2 |
| FSTTA (R2R) | 8 fast/slow pairs from `6e-7/1e-6` to `6e-5/1e-4` | 8 `(M,N)` window profiles |
| ATENA (R2R) | 8 query/self pairs from `2.5e-8/3.125e-9` to `4e-7/5e-8` | threshold `0.1/0.15/0.2/0.25` and lambda `0/0.1` |

## One-command validation campaign

```bash
cd /data1/wxy/code/NavTTA

PYTHON=/data1/wxy/exp_data/NavTTA/vln/envs/duet/bin/python

"$PYTHON" vln/scripts/run_targeted_gap_campaign.py plan \
  --spec vln/experiments/vln_targeted_gap_campaign_v2.json \
  --batch-id vln-targeted-gap-campaign-v2-seed0 \
  --gpus 0,1,2,3

"$PYTHON" vln/scripts/run_targeted_gap_campaign.py run \
  --spec vln/experiments/vln_targeted_gap_campaign_v2.json \
  --batch-id vln-targeted-gap-campaign-v2-seed0 \
  --gpus 0,1,2,3
```

`run` executes `search`, `freeze`, and `val-seen` sequentially. It does not
run REVERIE test because the FeedTTA-LLM submission requires a separately
started and verified Qwen provider. The detailed stage commands remain
available for diagnosis and exact resume.

## Selection and leakage boundary

- `val_unseen` is the only hyperparameter-selection input.
- `val_seen` is evaluated only after all 16 winners are frozen.
- Test/leaderboard values never alter a winner or search grid.
- Every candidate and split starts from a fresh model, optimizer, memory, and
  RNG state.
- Source results do not affect ranking or freeze.

Administrative evidence is written to
`vln/results/logs/targeted_gap/vln-targeted-gap-campaign-v2-seed0/`; model
outputs are written to
`vln/results/tuning/targeted_gap/vln-targeted-gap-campaign-v2-seed0/`.
