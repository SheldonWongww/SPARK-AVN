# AVN results

Evaluation scripts create `runs/<run_id>/` on demand. Commit `manifest.json`,
`summary.json`, compact metrics, and diagnostics. Raw logs, videos, TensorBoard
data, and large episode traces stay under a local `raw/` subdirectory.

Curated hyperparameter-search analyses live in
`analysis/hparam_search/`. Method-wide reports and deterministic per-model ×
method Markdown exports in `analysis/hparam_search/by_model/` are intended for
Git. Generate the latter from compact local `metrics.csv` evidence with:

```bash
python3 avn/scripts/export_hparam_analysis.py \
  avn/results/logs/<experiment>/<batch>/metrics.csv
```

The exporter records input hashes but does not copy raw logs, absolute local
paths, episode traces, or model assets into the report.

`legacy/` contains pre-refactor numbers whose provenance is incomplete and which must not be used directly in the main comparison table.
