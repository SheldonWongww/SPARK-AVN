# BEVBert and StreamVLN Source recovery contract v1

Status: `blocked_pending_server_asset_repair_and_rerun`
Owner/date: NavTTA / 2026-08-30

## 1. Identity and evidence

| Field | Value |
|---|---|
| Contract ID | `bevbert-streamvln-source-recovery-v1` |
| Category | Source evaluation; no test-time adaptation |
| BEVBert paper/upstream | *BEVBert: Multimodal Map Pre-training for Language-guided Navigation*, ICCV 2023; `https://github.com/MarSaKi/VLN-BEVBert.git` at `ee40002e9eb75ca6d587e8746990e91b0463f5ca` |
| StreamVLN paper/upstream | *StreamVLN*; `https://github.com/InternRobotics/StreamVLN.git` at `e48f6ff7e9201d93aae003e8f64b04c00cec13bc` |
| Local read-only references | `references/repos/navigation/vln/VLN-BEVBert`, `references/repos/navigation/vln/StreamVLN` |
| Active ports | `vln/baselines/bevbert`, `vln/baselines/streamvln` |
| License/permission | Recorded in `vln/manifests/upstream_repositories.json`; upstream license is not declared for these two snapshots and direct-author permission is recorded |

This contract recovers two interrupted or protocol-mismatched Source cells. It
does not authorize a TTA run or a hyperparameter search.

## 2. Paper / official / NavTTA decision ledger

| Semantic item | Paper / official execution | NavTTA execution | Classification and consequence |
|---|---|---|---|
| BEVBert checkpoint | Released CE `ckpt/ckpt.iter9600.pth` for evaluation and inference | `vln/checkpoints/bevbert/ckpt.iter9600.pth`, SHA256 `70dfdfff153f9d54888215e492dae6bff69b674616f32c94a4d51538a67cc0e8` | Exact released checkpoint. The same file is used for `val_seen` and `val_unseen`; there are no split-specific weights. |
| BEVBert fixed perception weights | Released waypoint predictor plus DDPPO depth encoder | Waypoint SHA256 `09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f`; depth encoder SHA256 `a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad` | These are shared frozen sidecars, not split-specific navigation checkpoints; both files must match. |
| BEVBert CLIP encoder | OpenAI CLIP ViT-B/16 | SHA256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f` | Shared frozen image encoder; the recovery preflight checks the runtime cache file explicitly. |
| BEVBert dataset | Released config names `R2R_VLNCE_v1-2_preprocessed_BERTidx` | `--ce-data-version v1.2-native`; `val_seen` SHA256 `83b231674ae3a18cb02b3d65847377e978f45cc35682f807c03becc38b09410c`, `val_unseen` SHA256 `7db0a39bef374cf5ce986473f8b35c9721f962c6c1a9af069257aa710df16365` | Paper-native data pairing. Existing `v1.3-unified` Source results are a different stream and cannot be compared to the v1.2 paper/VLN-TTA row. |
| BEVBert evaluator ground truth | Released v1.2 preprocessed `val_seen_gt` and `val_unseen_gt` | SHA256 `2c1df3c1f857b5f974192ea60da5d7dc71ccdd61cdfbdcd004258d5c57b2d0d0` and `1b9497dc1aa6ab68073976f9678ac42a154b393edd9d0b0450d0ac90841f684c` | Exact evaluator inputs; both are checked before the rerun. |
| BEVBert process topology | Official shell uses four ranks and four environments | One rank, one environment, batch size one | Deliberate NavTTA stream-control mapping. Aggregate parity is judged at the papers' integer SR/SPL precision, not trajectory identity. |
| BEVBert action/controller | `IL.back_algo control`, sliding enabled | Same | Exact. |
| StreamVLN checkpoint | Released `StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` | Pinned local snapshot; index SHA256 `d1f86714849892ee46d17fd82a90b9f2690e0083d2b2339ccd827673d4c5c2e3` | Exact released snapshot identity. |
| StreamVLN dataset | Public R2R-CE v1.3 | `r2r_vlnce_v1_3`, one rank/environment | Exact dataset version with controlled order. |
| StreamVLN scene assets | Official Habitat MP3D archive | Archive size `16085306031`, SHA256 `470948ee78ff4d6dc4c4870395d1e790a0f15c72065d1f62cdd229b2bcbe0d36` | The server's `1pXnuDYAj8r_semantic.ply` must be restored from this archive before rerun. Disabling semantic loading is not accepted for this recovery run. |
| Episode order | Upstream order/process scheduling is not published | Canonical scene-blocked order seed 0 | Controlled NavTTA Source contract requested by the user. Source is non-adaptive; the ordering is nevertheless recorded. |
| Metric comparison | R2R-CE papers report integer percentages | Preserve raw metrics and compare SR/SPL with decimal `ROUND_HALF_UP` | Match decision uses integer SR/SPL; NE/OSR/TL remain diagnostic. |

The v1.2 and v1.3 validation manifests contain the same episode IDs and order,
but their annotation file hashes differ because v1.3 changes episode start
rotations and oracle paths. Therefore matching IDs or counts does not make the
two protocols interchangeable.

## 3. Ownership boundary

| Component | Location | Constraint |
|---|---|---|
| Model/runtime adapters | `vln/baselines/bevbert`, `vln/baselines/streamvln` | No edits under `references/repos/` |
| Launch and validation | `vln/scripts/` | No cross-task simulator dependency in `core/` |
| Dataset/checkpoint provenance | `vln/manifests/` | Assets remain untracked |
| Run outputs | `vln/results/source/` | Raw logs/results remain Git-ignored |

VLN is explicitly activated by the user's 2026-08-30 Source-recovery request.

## 4. Deployment semantics

- Evaluation unit: one episode; one rank and one environment.
- Lifecycle: fresh process for each split, in `val_seen`, then `val_unseen`.
- Source update policy: no optimizer, parameter, persistent-buffer, replay, or
  teacher update.
- Episode reset: simulator and model navigation state reset per episode.
- Action selection: the model-native deterministic evaluation path; BEVBert
  uses its released high-level waypoint policy and low-level controller;
  StreamVLN uses greedy generation (`do_sample=False`, one beam).
- Seeds: model seed 0 and canonical episode-order seed 0.
- Evaluator: each baseline's task-native R2R-CE metrics. Test is excluded from
  this rerun because public test labels are hidden.
- Allowed observations: RGB/depth, instruction, GPS/compass, and model-native
  map state. Source does not consume evaluator success or distance fields.

## 5. Replay contract

N/A — Source evaluation performs no adaptation and has no replay memory.

## 6. Parity gates

| Gate | Required evidence | Status |
|---|---|---|
| BEVBert asset identity | Frozen size/SHA256 checks pass for the checkpoint, CLIP, waypoint/depth sidecars, annotations, and evaluator ground truth | Pending server check |
| BEVBert dataset identity | Console command contains the v1.2 path; dataset/order hashes match | Pending rerun |
| StreamVLN asset integrity | Static GLB/PLY validation, official archive digest, native one-scene load | Pending repair |
| Complete evaluation | 778 `val_seen` and 1839 `val_unseen` episodes for each model | Pending rerun |
| Metric parity | BEVBert SR/SPL round to `68/60` and `59/50` | Pending rerun |
| Output isolation | Fresh run tag and paths below `vln/results/source/` | Required |

## 7. Deviations, gates, and handoff

| Issue | Evidence | Scientific impact | Resolution |
|---|---|---|---|
| Historical BEVBert result was `v1.3-unified` | Recorded Source was approximately `68.38/59.92` (`val_seen`) and `58.24/48.26` (`val_unseen`) | It cannot validate the v1.2 VLN-TTA/paper row | Rerun `ckpt.iter9600.pth` with explicit `v1.2-native` |
| StreamVLN native abort at `1pXnuDYAj8r` | Semantic PLY is truncated/padded; the first face byte is 64 | Incomplete result; no aggregate metric is valid | Restore the entire scene bundle from the verified official archive, validate, and restart with a fresh tag |
| Official BEVBert uses four ranks | NavTTA uses one canonical stream | Possible runtime-level numerical divergence, but not a checkpoint/data ambiguity | Preserve controlled single-rank protocol and judge reported SR/SPL precision |

Allowed next stage: asset repair, focused scene preflight, then the two-model
Source rerun. The failed StreamVLN partial output must not be resumed or merged.
