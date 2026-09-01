# AVN blank-row low-learning-rate search v1

## Scope

This campaign fills only the six currently blank workbook rows: FeedTTA,
ATENA, and IDEA for SMT+Audio and ENMuS. Single-source and multi-source are
selected independently, giving 12 cells. Tent, FSTTA, EAM, Source, and `OURS`
are not rerun.

Each cell has 64 complete 2,000-episode validation candidates. The two model
campaigns therefore contain 384 jobs each and 768 jobs in total. Every
candidate starts from its model/source checkpoint with seed 0 and native
sampled actions. FeedTTA and ATENA remain explicitly feedback-supervised;
IDEA remains unsupervised.

## Search spaces

All adaptation learning rates are strictly below `1e-6`.

| Method | Cartesian grid | Candidates/cell |
|---|---|---:|
| FeedTTA | 8 LR (`1e-10` through `3e-7`) × 4 gamma (`.90/.95/.99/1.0`) × 2 reviewed SGR profiles | 64 |
| ATENA | 8 query/self LR pairs (`1e-9/3e-10` through `3e-7/3e-8`) × 4 thresholds (`.25/.5/.75/1.0`) × 2 lambdas (`.25/.5`) | 64 |
| IDEA | 8 LR (`1e-10` through `3e-7`) × 8 tau (`.3` through `1.0`) | 64 |

The ranges are based on the imported AutoDL analysis and the paused
SMT+Audio batch. FeedTTA performed best around `3e-8` on SMT+Audio and had a
model-specific ENMuS point at `3e-7`; rates at or above `1e-6` degraded the
aggregate results. ATENA degraded sharply as query LR increased, particularly
on multi-source. The previous IDEA range started at `3e-4`, so it is moved
entirely into the sub-`1e-6` regime while retaining the existing tau axis and
all other implemented settings.

## Execution

Run the models in their respective environments. Each method is assigned one
GPU lane and finishes single-source before starting multi-source.

```bash
python3 avn/scripts/run_avn_blank_val_search.py smt_audio --dry-run \
  --batch-id avn-smt-audio-blank-val-v1-seed0
python3 avn/scripts/run_avn_blank_val_search.py enmus --dry-run \
  --batch-id avn-enmus-blank-val-v1-seed0
```

For a full run, remove `--dry-run`. The ENMuS launcher prepares and validates
its two IDEA source-statistics artifacts before launching IDEA target jobs.
Results are written below `avn/results/logs/blank_val_search/<model>/<batch>`;
formal run manifests remain below `avn/results/runs/`.

