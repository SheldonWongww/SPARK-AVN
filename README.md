# NavTTA

NavTTA is a research workspace for test-time adaptation (TTA) in three embodied-navigation tasks:

- `avn/`: audio-visual navigation (the active task line)
- `vln/`: vision-language navigation (isolated until its experiment phase starts)
- `objectnav/`: object navigation (isolated until benchmark selection is complete)

Only AVN is active. Its code, data, checkpoints, experiment definitions, and
results live directly under `avn/`. VLN and ObjectNav remain as lightweight
task records until their experiment phases begin. Task-agnostic TTA and run
utilities live in `core/`; untouched upstream repositories live under
`references/repos/`.

## Repository policy

- Create directories when they become necessary; do not pre-build empty task scaffolding.
- Git tracks code, configurations, manifests, compact results, and documentation.
- Datasets, pretrained weights, training checkpoints, raw logs, videos, and TensorBoard files remain local.
- Every formal experiment must write a manifest containing the Git commit, configuration, checkpoint digest, dataset version, seed, and hardware.
- Code in `references/repos/` is read-only reference material. Research modifications belong in the corresponding task directory.

Current top-level structure:

- `avn/`: active AVN work
- `vln/`, `objectnav/`: isolated task records
- `core/`: shared TTA implementation and tests
- `references/`: pinned upstream source repositories
- `docs/`: research progress and literature notes
- `tools/`: small repository-level utilities
