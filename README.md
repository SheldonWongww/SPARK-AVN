# NavTTA

NavTTA is a research workspace for test-time adaptation (TTA) in three embodied-navigation tasks:

- `avn/`: audio-visual navigation (active experiments)
- `vln/`: vision-language navigation (active pre-evaluation preparation)
- `objectnav/`: object navigation (isolated until benchmark selection is complete)

AVN is the active experiment line.  VLN was explicitly activated for isolated
environment, asset, and evaluation-protocol preparation; it does not share
task-specific dependencies with AVN.  ObjectNav remains a lightweight task
record until benchmark selection.  Task-agnostic TTA and run utilities live in
`core/`; untouched upstream repositories live under `references/repos/`.

## Repository policy

- Create directories when they become necessary; do not pre-build empty task scaffolding.
- Git tracks code, configurations, manifests, compact results, and documentation.
- Datasets, pretrained weights, training checkpoints, raw logs, videos, and TensorBoard files remain local.
- Every formal experiment must write a manifest containing the Git commit, configuration, checkpoint digest, dataset version, seed, and hardware.
- Code in `references/repos/` is read-only reference material. Research modifications belong in the corresponding task directory.

Current top-level structure:

- `avn/`: active AVN experiments
- `vln/`: isolated VLN preparation and future experiments
- `objectnav/`: isolated task record
- `core/`: shared TTA implementation and tests
- `references/`: pinned upstream source repositories
- `docs/`: research progress and literature notes
- `tools/`: small repository-level utilities
