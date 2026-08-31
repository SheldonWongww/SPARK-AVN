# VLN status

Current phase: pre-evaluation preparation; no new formal evaluation result has
been claimed.

- The targeted TTA implementation covers exactly 16 workbook gaps: 55 full
  seed-0 `val_unseen` search jobs, one campaign-wide freeze, 16 fresh
  `val_seen` winner evaluations, and five REVERIE submission-only test jobs.
  Four immutable cell queues are assigned to each of GPUs 0--3; candidates
  inside one cell are serial. StreamVLN and `OURS` remain excluded.
- The v1 targeted contract is superseded. The active v2 execution overlay
  directly runs the 16-cell TTA campaign; existing Source results are used
  only for post-hoc reporting and do not block search, freeze, or `val_seen`.
- REVERIE FeedTTA-LLM test transfer uses the pinned Qwen2-VL-2B model only
  after a same-commit CUDA/float16 provider plus real 36-view headless
  MatterSim render preflight. `PRECHECK.json` and its nested artifact graph are
  bound to the stage plan, job identity, and formal manifest. It is reported
  separately as pseudo-feedback, never as exact-feedback FeedTTA or IDEA.
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
- Source reproduction remains useful for checking published-number parity but
  is independent of the targeted TTA search. The TTA jobs still use model seed
  0, canonical episode-order seed 0, and each model's selected data version.
- The historical BEVBert Source record under
  `grouped-source-20260810T080743Z` used unified v1.3 starts, so it is not the
  paper-native v1.2 control requested for the current comparison. BEVBert has
  one released CE checkpoint (`ckpt.iter9600.pth`) shared by both validation
  splits; a focused verifier now authenticates it, CLIP ViT-B/16, its two
  frozen perception sidecars, and both v1.2 annotation/ground-truth streams
  before the replacement run.
- A StreamVLN full run exposed a corrupted
  `1pXnuDYAj8r_semantic.ply` at the first episode of the second canonical
  scene. The partial run is invalid. Tracked tools now restore that complete
  scene from the manifest-pinned official MP3D archive and fail before model
  loading if its GLB or semantic PLY layout is malformed. A fresh-tag rerun is
  still required after server-side repair.
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
