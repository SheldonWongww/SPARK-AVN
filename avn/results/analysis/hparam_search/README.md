# AVN hyperparameter-search analyses

This directory contains compact, Git-trackable analyses derived from local AVN
search evidence. Raw scheduler logs, episode traces, checkpoints, videos, and
TensorBoard data remain under ignored result directories and must not be
committed.

## Curated method reports

- [Tent](TENT_HYPERPARAMETER_SEARCH_REPORT.md)
- [FSTTA](FSTTA_HYPERPARAMETER_SEARCH_REPORT.md)
- [EAM](EAM_HYPERPARAMETER_SEARCH_REPORT.md)
- [FeedTTA](FEEDTTA_HYPERPARAMETER_SEARCH_REPORT.md)

## Per-model reports

Use `avn/scripts/export_hparam_analysis.py` to convert one or more compact
`metrics.csv` files into deterministic reports at:

```text
avn/results/analysis/hparam_search/by_model/<model>/<method>.md
```

Example:

```bash
python3 avn/scripts/export_hparam_analysis.py \
  avn/results/logs/eam_boundary_grid/eam-boundary-joint-v1-seed0/metrics.csv \
  --source-metrics \
  avn/results/logs/source_reval/source-reval-v1-seed0/metrics.csv
```

The exporter includes only aggregate SR/SPL, hyperparameters, validation
labels, run tags, and evidence hashes. It never follows paths to raw evidence
or model assets. Generated reports should be reviewed before committing.
Rows with nonzero launcher status are excluded by default. For a reviewed
provisional batch whose navigation metrics are complete, pass
`--include-nonzero-status`; the generated report preserves both runner status
and validation labels rather than presenting those rows as validated.
