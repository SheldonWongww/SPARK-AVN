# Object Navigation

ObjectNav is intentionally isolated during the AVN phase. The current plan uses
the Habitat ObjectNav 2022 `objectnav_hm3d_v1` protocol with PIRLNav and ZSON.
Their upstream repositories remain under `references/repos/navigation/objectnav/`.
Data, working code, checkpoints, and result directories will be created here
only when ObjectNav experiments actually begin.

## Imported PONI TTA archive

A local legacy archive containing the complete MP3D PONI Source record, the
final FeedTTA, EAM, and ATENA runs that exceeded it, and the best completed
global-continual Tent and FSTTA runs is available under
`results/legacy/poni_mp3d_tta_20260908/`, with its
task-specific code snapshot under `implementations/poni_tta_snapshot_20260908/`
and provenance under `manifests/poni_tta_archive_20260908.json`.

This import does not activate a new formal ObjectNav protocol. Compact metrics,
manifests, and diagnostics are tracked as legacy evidence; raw logs and
TensorBoard events remain local and must not be committed.
