# AVN status

Current phase: implementation recovery and reproducibility validation.

- Source checkpoints for SMT+Audio and ENMuS are locally available but require provenance manifests.
- Tent, FSTTA, EAM, and FeedTTA prototypes exist.
- ATENA has a paper/official-aligned prototype, but AVN execution is deliberately
  gated by `experiments/ATENA_PRE_RUN_REVIEW.md`; it must be reassessed before
  any smoke test or formal run.
- No TTA result is considered formal until a run manifest and reproducible summary are produced.
- Legacy ENMuS metrics are retained under `results/legacy/` and excluded from the main table.
