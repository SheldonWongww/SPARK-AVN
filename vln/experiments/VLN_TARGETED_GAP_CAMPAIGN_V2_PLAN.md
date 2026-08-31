# VLN targeted gap campaign v2

## Purpose

Run only the 16 currently blank, non-StreamVLN, non-`OURS` VLN cells. The
candidate matrix, datasets, checkpoints, seeds, metrics, and queue assignment
remain pinned by the superseded v1 design. Version 2 changes one operational
decision: existing Source results are reporting references, not a launch or
winner-selection dependency.

## Executed matrix

- 55 complete `val_unseen` candidate runs, seed 0.
- One winner per each of the 16 cells, selected only from `val_unseen`.
- 16 fresh `val_seen` runs using the frozen winners.
- Five REVERIE test submissions in a later, separate phase.
- StreamVLN and `OURS` remain excluded.

Each GPU owns four fixed cell queues. A cell runs one candidate at a time;
different cells assigned to the same GPU may run concurrently. Development
job counts on GPUs 0, 1, 2, and 3 are respectively 15, 15, 13, and 12.

ETPNav and BEVBert continue to use R2R-CE v1.2-native. Their existing Source
results may be compared after the campaign, but Source values and ledgers are
not read by candidate selection and are not frozen into winner evidence.

## One-command validation campaign

```bash
cd /data1/wxy/code/NavTTA

python3 vln/scripts/run_targeted_gap_campaign.py plan \
  --spec vln/experiments/vln_targeted_gap_campaign_v2.json \
  --batch-id vln-targeted-gap-campaign-v2-seed0 \
  --gpus 0,1,2,3

python3 vln/scripts/run_targeted_gap_campaign.py run \
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
