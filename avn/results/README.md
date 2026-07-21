# AVN results

Evaluation scripts create `runs/<run_id>/` on demand. Commit `manifest.json`,
`summary.json`, compact metrics, and diagnostics. Raw logs, videos, TensorBoard
data, and large episode traces stay under a local `raw/` subdirectory.

`legacy/` contains pre-refactor numbers whose provenance is incomplete and which must not be used directly in the main comparison table.
