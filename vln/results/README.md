# VLN results

Formal compact results belong in `runs/<run-id>/` with a manifest tied to the
Git commit, configuration, checkpoint digest, dataset version, order seed, and
hardware. Raw hyperparameter-search outputs and consoles remain local and are
ignored by Git.

Staged TTA search evidence uses these parallel trees:

```text
logs/hparam_search/<method>/<batch>/
  batch.json
  stages/<stage>/
    stage_manifest.json  grid.csv  metrics.csv  SUMMARY.json
    jobs/<setting>/<run-tag>/

tuning/<method>/<batch>/<stage>/<setting>/<run-tag>/val_seen/
```

The first completed Tent/FSTTA campaign predates this layout. Its raw outputs
are migrated without changing run identities. Legacy Source outputs referenced
by both searches live in `tuning/_shared/`, and migration ledgers live in
`tuning/_migrations/`.

Do not add raw logs, checkpoints, predictions, videos, TensorBoard files, or
datasets to Git. Incomplete-provenance metrics belong under `legacy/` and must
not be used in formal comparison tables.
