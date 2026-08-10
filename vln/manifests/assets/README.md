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

The HAMT R2R checkpoint is intentionally named
`hamt_r2r_fixed_feature_checkpoint`: it is the upstream `vitbase-finetune`
checkpoint and must not be paired with the paper's final e2e headline metrics.
Its history position embedding uses 50 positions (`--max_action_steps 50`) to
match the released checkpoint.

The official GOAT checkpoints retain auxiliary CFP-era modules that are not
instantiated by the validation policy graph.  Strict loading narrowly
allowlists only those known unused prefixes; every policy key remains required
and every other unexpected key is rejected.  Both GOAT settings passed the
two-episode GPU lifecycle smoke under tag
`gpu-smoke-20260810T023726Z-goatfix2`; its metrics are non-formal.
