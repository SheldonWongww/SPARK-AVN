# NavTTA workspace rules

## Scope and ownership

- Keep VLN, AVN, and ObjectNav code, environments, data, checkpoints, experiments, and results inside their respective top-level task directories.
- Put only task-agnostic TTA algorithms and experiment utilities in `core/`.
- Treat `references/repos/` as read-only upstream material. Never implement research changes there.
- Do not copy a task-specific Habitat or simulator dependency into `core/`.

## Assets and results

- Never add datasets, scene assets, audio/RIR files, checkpoint binaries, videos, TensorBoard files, or raw logs to Git.
- Record dataset/checkpoint provenance and SHA256 hashes in the task's tracked `manifests/` directory.
- A formal result requires a run manifest tied to the top-level Git commit, configuration, checkpoint digest, dataset version, seed, and hardware.
- Keep legacy or incomplete-provenance metrics under `results/legacy/`; do not use them in formal tables.

## Current research phase

- AVN is active, initially using SMT+Audio and ENMuS with Source, Tent, FSTTA, EAM, FeedTTA, and ATENA.
- VLN and ObjectNav are isolated. Do not activate or refactor their baselines without an explicit task decision.
- Preserve the supervision distinction between unsupervised TTA and methods that consume binary episode feedback.

## Development

- Keep active source files LF-normalized.
- Preserve upstream licenses and record upstream URLs and pinned commits.
- Run `python3 tools/verify_layout.py` after structural changes.
- Install the shared package into each task environment with `python -m pip install -e core`.
