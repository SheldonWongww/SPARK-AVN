# VLN status

Current phase: pre-evaluation preparation; no new formal evaluation result has
been claimed.

- Discrete settings: DUET, HAMT, and GOAT on R2R and REVERIE.
- Existing completed R2R-CE TTA campaigns used unified v1.3 episode starts;
  those results remain tied to that frozen protocol.  The newly selected
  paper-native campaign instead uses ETPNav/BEVBert v1.2 and StreamVLN v1.3.
  Existing ETPNav/BEVBert v1.3 results must not be reused as matched controls
  for a future v1.2 TTA campaign.
- Paper-native Source reproduction is deliberately separate: ETPNav uses the
  released `ckpt.iter12000.pth` on v1.2, BEVBert uses `ckpt.iter9600.pth` on
  v1.2, and the updated StreamVLN benchmark checkpoint uses v1.3.  These runs
  check each model against its own paper but are not one same-stream
  cross-model comparison.  R2R/R2R-CE SR and SPL parity is judged after
  decimal half-up rounding to integer percentage points.  The rerunner keeps
  NavTTA's canonical single-rank execution, so `paper-native` identifies the
  released checkpoint/data pairing rather than unpublished upstream scheduling
  or RNG state.
- The current four-GPU validation campaign is Source-only: model seed 0 and
  canonical episode-order seed 0, with no TTA method or hyperparameter search.
  Future TTA/search work starts only after this Source gate and must match each
  model's selected dataset version and seed-0 stream.
- Six official source repositories are pinned in
  `manifests/upstream_repositories.json` and exported into `baselines/` as
  active, Git-tracked snapshots.
- Canonical manifests fix `val_seen -> val_unseen -> test`, then normalized
  scene ID and natural episode ID.  Stateful online runs use one rank, one
  environment, batch size one, exact episode counts, and a fresh process per
  split.
- Public test goals remain hidden.  Test runs emit submission trajectories and
  are never scored locally.
- Published paper metrics are retained under `results/legacy/`; they are not
  formal NavTTA reruns.
- CPU/offline dependency, asset, and entry-point checks are recorded in the
  task manifests.  All nine model/task settings also passed a two-episode
  `val_seen` GPU lifecycle smoke.  Successful outputs use tags
  `gpu-smoke-20260810T023726Z`, `gpu-smoke-20260810T023726Z-fix1` for
  HAMT-R2R, and `gpu-smoke-20260810T023726Z-goatfix2` for GOAT R2R/REVERIE.
  Smoke metrics are lifecycle-only diagnostics, not formal results.
- ETPNav and BEVBert also passed a one-episode, no-ground-truth `test`
  submission-path lifecycle check under
  `gpu-test-inference-20260810T035454Z`; both generated trajectories passed the
  R2R-CE submission validator.  These partial outputs are not leaderboard
  submissions or formal results.
