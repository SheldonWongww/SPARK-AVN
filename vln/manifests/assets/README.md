# VLN evaluation asset manifest

`eval_assets.json` records the exact external binaries used to prepare the six
VLN baselines for `val_seen`, `val_unseen`, and submission-only `test` runs.
Training annotations and training-only MP3D scene extraction are intentionally
outside this manifest.

Paths without a leading slash are relative to the top-level NavTTA checkout.
Absolute cache paths are still under `/root/autodl-tmp`; launchers point to
them explicitly so evaluation never writes to `/root/.cache`.

The SHA256 values were computed from the installed files, not copied blindly
from download pages. Large PyTorch archives were inspected with a restricted
pickle reader that never materialized tensor storage. HDF5 files were opened
read-only and only their metadata, shapes, dtypes, attributes, and key coverage
were inspected.

The active HAMT R2R evaluation uses the final upstream
`vitbase-finetune-e2e/best_val_unseen` checkpoint together with
`pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5`.  The previously used
`vitbase-finetune` fixed-feature checkpoint is retained only for provenance of
the superseded 2026-08-10 run and must not be paired with the paper's e2e
headline metrics.  Both released checkpoints use 50 history positions, so the
launcher sets `--max_action_steps 50` explicitly.

The official GOAT checkpoints retain auxiliary CFP-era modules that are not
instantiated by the validation policy graph.  Strict loading narrowly
allowlists only those known unused prefixes; every policy key remains required
and every other unexpected key is rejected.  Both GOAT settings passed the
two-episode GPU lifecycle smoke under tag
`gpu-smoke-20260810T023726Z-goatfix2`; its metrics are non-formal.
