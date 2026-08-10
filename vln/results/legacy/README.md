# Legacy VLN reference results

Files in this directory are reference values transcribed from upstream papers
or official model cards. They are **not** measurements produced by this
workspace and must not be used as formal NavTTA results.

`upstream_published_metrics.json` uses one record per model/benchmark variant:

- `benchmark` fixes the task protocol and dataset variant;
- `model_variant` and `checkpoint_reference` describe the upstream artifact
  that the published row is intended to represent;
- `citation` identifies an immutable paper version and an exact table label or
  official model-card statement;
- `metrics` preserves the split names and values as reported upstream; and
- `qualification` records unresolved artifact mappings or other caveats.

Rate metrics are stored in percentage points, while `TL` and `NE` are in
meters. Missing splits or metrics are omitted rather than inferred. In
particular, discrete R2R/REVERIE results and continuous R2R-CE results are
different protocols and must never be placed in the same comparison column.

A formal result belongs outside `results/legacy/` and requires a run manifest
tied to the top-level Git commit, local configuration, checkpoint and dataset
digests, seed, episode-order manifest, and hardware.
