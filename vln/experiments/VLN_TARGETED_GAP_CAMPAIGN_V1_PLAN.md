# VLN targeted gap campaign v1

> Superseded by `vln_targeted_gap_campaign_v2.json` before any TTA job was
> launched. The Source-promotion workflow below is retained as historical
> audit documentation and is no longer a prerequisite for the 16-cell search.
> Use the direct `run` command documented in `vln/README.md`.

## Status and handoff

The version-1 design contract, before its one permitted lifecycle transition,
is
[`vln_targeted_gap_campaign_v1.json`](vln_targeted_gap_campaign_v1.json),
SHA256 `fc0f0ba695457fd927c7e97c3b07fd21686483befb906d15eb681fe4ba505801`.
`create-successor` later changes only its lifecycle fields and records the
resulting predecessor digest inside v2. This plan is a pre-execution contract,
not evidence that any run occurred. It does
not supersede the broader consistency-search specs: it defines a new targeted
campaign over an explicit subset of their cells and candidate anchors.

Campaign launch remains blocked pending a clean, committed implementation
review and authenticated
native-v1.2 matched Source manifests for ETPNav and BEVBert. The older
consistency launcher must not be used to approximate this contract. The
legacy R2R `val_seen` Source ledger must also be reauthenticated so that each
selected metric artifact digest and size is present in its formal run
manifest; a tracked ledger that merely points at a derived JSON file is not
sufficient. The
FeedTTA-LLM provider identity is contract-ready and content-addressed, but its
model directory, contract file, loopback service health, headless renderer,
and licensed raw RGB must be reverified by a same-commit `PRECHECK.json` on
the target server. The three pinned IDEA
source-training-statistics artifacts are intentionally untracked and must also
be staged and digest-verified there before an IDEA queue can start.

## Research question

The campaign asks whether small grids around already tracked, paper-anchored
settings can close sixteen remaining model-method gaps. Every candidate is
selected on the complete canonical `val_unseen` stream with episode-order seed
0 and model seed 0. Exactly one winner per cell is then frozen. Only after one
campaign-wide freeze artifact binds all sixteen winners may fresh processes
evaluate those winners on complete canonical `val_seen`, again with seed 0.

`val_seen` is evaluation in this campaign. It cannot influence a winner,
break a tie, trigger a grid extension, or revise a feedback provider. This
reverses the role used by older VLN searches intentionally; split identity and
split role are recorded separately in the JSON.

## Workbook-derived gap inventory

The scope is derived from
[`docs/NavTTA_benchmark_results.xlsx`](../../docs/NavTTA_benchmark_results.xlsx),
SHA256 `fe9149277ecc9505df32b5fb2e61f4bd54170e72ebea5585ae257911ca663bac`.
Across the three VLN sheets there are 63 non-Source method rows before
exclusions. Applying the fixed precedence removes the seven-row StreamVLN
block, eight remaining `OURS` rows, and 32 rows containing at least one prior
metric. The 16 fully blank eligible rows are:

- REVERIE rows 6, 14, 22, 23, and 25: HAMT EAM; DUET EAM; GOAT EAM,
  FeedTTA, and IDEA.
- R2R rows 4, 5, 7, 8, 9, 20, 22, 23, and 25: HAMT Tent, FSTTA,
  FeedTTA, ATENA, and IDEA; GOAT Tent, EAM, FeedTTA, and IDEA.
- R2R-CE rows 6 and 14: ETPNav EAM and BEVBert EAM.

For eligibility, the complete metric spans must be blank: `B:M` for REVERIE,
`B:I` for R2R, and `B:K` for R2R-CE. The workbook is scope evidence only;
none of its populated metrics are selection inputs.

## Exact matrix and queues

There are 16 cells and 55 development jobs. Each cell is a serial queue. Queue
IDs are immutable, there is no work stealing, and `gpu = queue_id mod 4` gives
exactly four queues to every GPU.

| Queue | GPU | Cell | Candidates |
|---:|---:|---|---:|
| 0 | 0 | HAMT–REVERIE EAM | 3 |
| 1 | 1 | DUET–REVERIE EAM | 3 |
| 2 | 2 | GOAT–REVERIE EAM | 3 |
| 3 | 3 | GOAT–REVERIE FeedTTA | 3 |
| 4 | 0 | GOAT–REVERIE IDEA | 4 |
| 5 | 1 | HAMT–R2R Tent | 4 |
| 6 | 2 | HAMT–R2R FSTTA | 3 |
| 7 | 3 | HAMT–R2R FeedTTA | 3 |
| 8 | 0 | HAMT–R2R ATENA | 5 |
| 9 | 1 | HAMT–R2R IDEA | 4 |
| 10 | 2 | GOAT–R2R Tent | 4 |
| 11 | 3 | GOAT–R2R EAM | 3 |
| 12 | 0 | GOAT–R2R FeedTTA | 3 |
| 13 | 1 | GOAT–R2R IDEA | 4 |
| 14 | 2 | ETPNav–R2R-CE EAM | 3 |
| 15 | 3 | BEVBert–R2R-CE EAM | 3 |

The per-GPU development totals are 15, 15, 13, and 12 jobs for GPUs 0–3.
The user has explicitly approved the fixed four-queue allocation for the
available hardware. The launcher must identify all four requested GPUs and
hard-cap navigation workers at those four declared queues per GPU; it may not
add a fifth navigation worker, steal work, migrate a queue, or change a
candidate to improve utilization. The Qwen service is not active during
search or `val_seen`; during the five-job REVERIE test phase GPU 0 has only two
navigation jobs, so the separate provider remains within the same declared
process envelope.

The grids copy the explicit candidates and bases from the tracked REVERIE/R2R
consistency v2 specs and R2R-CE consistency v3 spec. Each JSON candidate names
its provenance. Every grid is below the ten-candidate limit; the actual maximum
is five. Seven of the ten reusable grids vary learning-rate fields only, the
two IDEA grids vary learning rate and `tau`, and ATENA varies paired learning
rates plus one threshold and one mixture probe. The sole predeclared reduction
is HAMT–R2R ATENA: its `delta_0p2` point is omitted before execution. Restoring
that point after metrics are visible is forbidden.

## Data, seeds, horizon, and action protocol

Discrete R2R and REVERIE use their native tracked annotations and native
argmax action protocol. ETPNav and BEVBert use paper-native R2R-CE v1.2—not
the unified v1.3 stream. The physical streams, annotation digests, order-file
digests, and checkpoint digests are all pinned in the spec.

| Benchmark | Development (`val_unseen`) | Evaluation (`val_seen`) | Model seed | Order seed |
|---|---:|---:|---:|---:|
| REVERIE | 3,521 episodes | 1,423 episodes | 0 | 0 |
| R2R | 2,349 episodes | 1,021 episodes | 0 | 0 |
| R2R-CE v1.2-native | 1,839 episodes | 778 episodes | 0 | 0 |

The action policy is deterministic argmax and evaluation augmentation is
disabled. Each candidate and split starts a fresh process with the baseline
model seed fixed to 0; replay and other stochastic method operations inherit
that process seed, while FeedTTA additionally fixes its SGR seed to 0. The
implementation does not claim independently hash-derived RNG streams. A fresh
complete checkpoint, optimizer, memory, prompt library, and RNG state is
required for every candidate and every split. Development state never flows
into `val_seen` or `test`.

The horizon is the full split. Update dose is a separate contract: Tent and
FSTTA fast updates trigger per policy forward; FSTTA slow updates trigger every
four completed episodes; EAM performs one eight-sample replay update per step
after its seven-step warm-up; FeedTTA and ATENA have at most one episode-end
optimizer attempt; IDEA performs 50 prompt-only steps for each new-domain
event and never writes base-policy parameters. IDEA records prompt optimizer
attempts/updates plus parameter-name, tensor-version, and content hashes; the
expensive full-policy after-hash is deferred to the declared final episode.
Accepted updates, action steps,
trainable parameter names, parameter drift, and each method's native schedule
diagnostics are required. A full-split adaptive run with no trainable scope or
no accepted update is invalid rather than a successful Source-like result.

## Source controls

The existing content-addressed, seed-0 discrete Source controls are reused
only after matching checkpoint, annotation, physical split, episode order,
model seed, action protocol, evaluator, and horizon. Any mismatch fails closed.

ETPNav and BEVBert must receive new matched Source runs on native R2R-CE v1.2
for both `val_unseen` and `val_seen`. The tracked unified-v1.3 Source manifests
are listed under `forbidden_reuse` because v1.2 and v1.3 have different start
rotations. Source runs are prerequisites and are not silently counted among
the 16 TTA queues.

Promotion is fail-closed and has four required setting/split records:
ETPNav and BEVBert on each of `val_unseen` and `val_seen`. First run each with
the pinned v1.2-native annotations, released checkpoint, model seed 0,
canonical order seed 0, native argmax action protocol, full split horizon, and
the same evaluator that the matching TTA cell will use. Then authenticate each
formal run manifest and metric artifact, verify every matching dimension, and
write one tracked aggregate Source ledger per split under `vln/manifests/`.
Only after a result audit accepts both complete ledgers may a successor spec
supersede this contract and replace both `rerun_required` entries with `reuse`
bindings containing the ledger paths and SHA256 values. Candidate, partial,
or unified-v1.3 ledgers cannot be promoted, and this v1 file must not be
silently edited after execution evidence exists.

On the clean target-server checkout, the producing workflow is:

```bash
BATCH="r2r-ce-v12-source-$(date -u +%Y%m%dT%H%M%SZ)"
python3 vln/scripts/run_r2r_ce_v12_source_controls.py plan \
  --batch-id "$BATCH" --gpus 0,1
python3 vln/scripts/run_r2r_ce_v12_source_controls.py run \
  --batch-id "$BATCH" --gpus 0,1 \
  --vln-root /data1/wxy/exp_data/NavTTA/vln
python3 vln/scripts/run_r2r_ce_v12_source_controls.py status \
  --batch-id "$BATCH" --json
python3 vln/scripts/run_r2r_ce_v12_source_controls.py finalize \
  --batch-id "$BATCH"
```

`finalize` emits content-addressed candidate ledgers with
`candidate_status=review_required_not_yet_promoted`; it does not authorize the
campaign. After independent review, promote the complete ledgers to the
suggested tracked paths
`vln/manifests/r2r_ce_v1_2_val_unseen_source_controls.json` and
`vln/manifests/r2r_ce_v1_2_val_seen_source_controls.json`, then bind their
actual digests in the successor campaign spec.

The auditable promotion is a separate, explicit workflow. First add and
commit the four generated formal Source manifests under `vln/results/runs/`;
raw logs and metric artifacts remain outside Git but must stay present at
their recorded paths. Then run:

```bash
REVIEWER="wxy"
REVIEWED_AT="$(date --iso-8601=seconds)"

python3 vln/scripts/promote_targeted_gap_source_controls.py review-ce \
  --batch-id "$BATCH" \
  --reviewed-by "$REVIEWER" --reviewed-at "$REVIEWED_AT" \
  --confirm-independent-review

python3 vln/scripts/promote_targeted_gap_source_controls.py audit-discrete \
  --reviewed-by "$REVIEWER" --reviewed-at "$REVIEWED_AT" \
  --confirm-independent-review
```

`review-ce` recomputes native aggregates from the canonical per-episode
streams and revalidates every formal-manifest identity before writing the two
final-schema ledgers. `audit-discrete` writes new ledger copies and never
modifies the historical files. If a formal manifest's native aggregate is not
available, it exits without writing any discrete ledger and lists each
setting/split whose Source evidence must be retrieved or rerun.

Review and commit `vln/manifests/source_candidates/`,
`vln/manifests/audited_source_controls/`, and the two promoted native-v1.2
ledgers. Only then create the successor; this final command calls the campaign
runner's all-six-Source validator before changing either spec file:

```bash
python3 vln/scripts/promote_targeted_gap_source_controls.py create-successor \
  --reviewed-by "$REVIEWER" --reviewed-at "$REVIEWED_AT" \
  --confirm-v1-supersession
```

The command writes deterministic v2 content for the supplied review identity,
marks v1 `superseded`, and binds the resulting v1 digest, current campaign
runner digest, and all six reviewed Source-ledger digests. The v2 spec remains
non-launchable until both lifecycle files are reviewed, added to Git, and
committed; the runner independently checks those tracked identities again.

## Selection and freeze

REVERIE ranks by RGSPL, RGS, SPL, SR, lower parameter drift, fewer accepted
updates, then candidate ID. R2R and R2R-CE rank by SPL, SR, lower parameter
drift, fewer accepted updates, then candidate ID. The full REVERIE metric
ordering is part of the frozen selection rule rather than merely a reporting
list. A non-positive Source delta does not license removing a candidate or
changing the grid; the best valid candidate is frozen and the negative finding
is retained. Source deltas are aggregate descriptive comparisons: the reused
discrete Source evidence has no authenticated per-episode sidecar, so this
campaign does not claim paired Source–TTA episode analysis. Every TTA candidate
and frozen evaluation must still preserve its own ordered per-episode metrics.

An algorithmic failure is retained as evidence and blocks the campaign-wide
freeze; it is not silently omitted, assigned a synthetic score, or retried.
Only an independently classified infrastructure failure may be retried under
the identical job identity. Partial-cell or partial-campaign selection is
forbidden.

The single `FROZEN.json` must bind the campaign spec digest, all 55 development
run-manifest digests, all 16 resolved winner-config digests, every matched
Source-manifest digest, the selection-code digest, and the top-level Git
commit. No `val_seen` or hidden-test information is allowed in that artifact's
selection inputs. Evaluation consists of exactly 16 fresh winner runs.

Unsupervised cells (Tent, FSTTA, EAM, IDEA) and feedback-supervised cells
(FeedTTA, ATENA) must be reported separately. IDEA is a prompt/library method
using pinned offline source-training statistics; it is not an LLM feedback
provider.

## REVERIE hidden-test submissions and GOAT FeedTTA deviation

After the campaign-wide freeze and all 16 `val_seen` evaluations terminate,
the five targeted REVERIE cells produce submission-only hidden-test runs. The
HAMT–EAM, DUET–EAM, GOAT–EAM, and GOAT–IDEA winners run normally as their
unsupervised methods. The GOAT–FeedTTA cell is the sole exception described
below. All five start fresh from their source checkpoint and reuse the frozen
navigation configuration; no `val_seen` state is transferred.

Ordinary FeedTTA receives exact evaluator binary navigation success after the
submitted trajectory endpoint on validation splits. REVERIE test labels are
hidden, so ordinary FeedTTA is illegal there.

The fifth hidden-test submission is separately named
`FeedTTA-LLM` (`goat-reverie-feedtta-llm-qwen2-vl-2b-v1`). It reuses the
frozen GOAT–REVERIE FeedTTA navigation hyperparameters without reselection but
replaces evaluator truth with pseudo-feedback from
`Qwen/Qwen2-VL-2B-Instruct`. It is therefore a protocol transfer and must not
be reported as FeedTTA, exact feedback, or IDEA.

The intended provider follows a two-stage contract:

1. Extract a simple target-goal phrase from the instruction.
2. Given that phrase and a 36-view RGB panorama captured at the final
   submitted/reranked endpoint, return exactly `Yes` or `No`.

Only deploy-time instruction, submitted trajectory identifiers, and the
explicitly captured final panorama are allowed. Evaluator internals,
`gt_trajs`, `_eval_item`, distance, reward, oracle actions, object ground truth,
goal viewpoint, and leaderboard feedback are forbidden. Precomputed CLIP
features or object logits cannot silently substitute for the panorama. The
label arrives after episode *n* ends and can affect only episode *n+1* onward.

The full 6,292-episode test stream permits at most 6,292 pseudo-labels and
12,584 provider requests (two per episode). The provider is frozen to
`Qwen/Qwen2-VL-2B-Instruct` revision
`895c3a49bc3fa70a340399125c650a463535e71c`, aggregate weights SHA256
`4faeb74ee719f9c35d7fa254d7cc2d7131e4b4ada7d0340615c66b22d5e16fc5`,
full model-bundle SHA256
`cf7dd27d27987b7ae71458529e6a72e3dcc3db6e91fb7298577e03f88e238a7e`,
stage prompt SHA256 values
`62d6978269100d5129fa2ac612e83e033603bcba50009ab711124fee1c22f0e4`
and `f023385c8c153b87aa20b28ea446ab5ef6a55e632d21fb1f13a48289a7f4d81f`,
and prompt-bundle SHA256
`9d608193a2cd444688c6f507973ab8da68940f4b11439d329195cfd2804e0ed9`.
The content-addressed runtime contract is
`vln/manifests/models/qwen2_vl_2b_instruct.json` (SHA256
`f48aa543d41992e647007af9cc4c2e3d4d926923baf6deb452a64b956234dc4d`),
with server model path
`/data1/wxy/exp_data/NavTTA/vln/models/Qwen2-VL-2B-Instruct` and loopback
service `http://127.0.0.1:8765`. These are immutable scientific inputs;
formal preflight must recheck the contract file digest, local model directory,
weights, and service health. The four unsupervised submissions remain subject
to the campaign-wide phase barrier and ordinary runtime preflight. Timeout,
malformed, missing, or duplicate output must fail closed
with no update and no truth fallback. Test output is submission-only: no local
metric, response, or leaderboard score may tune the provider or select the
navigation configuration. Each transcript label record must preserve the
model, revision, weights digest, both prompt/request/response digests,
two-stage latency values, parsed label, cache identity/hit, token counts, and
cost.

The executable provider gate is
`vln/scripts/verify_reverie_llm_feedback_preflight.sh`. It must pass before the
FeedTTA-LLM submission is launched. Its evidence binds a live CUDA/float16
service health response, a real 36-view headless MatterSim render, the exact
raw `matterport_skybox_images/*_skybox_small.jpg`, connectivity inputs,
rendering extension/CMake cache, generated panorama, and SHA256 values. The
ordinary navigation MatterSim build has EGL and OSMesa disabled and therefore
fails this gate; a separate pinned headless build plus licensed raw RGB is an
explicit deployment prerequisite, not something the campaign downloads or
rebuilds implicitly. The combined `reverie-test` command validates this gate
before launching any of the five submissions. The resulting PRECHECK path,
digest, renderer path/backend/module/cache digests, render evidence, panorama,
and provider-smoke evidence are bound into `STAGE_PLAN.json`, the FeedTTA-LLM
job identity, `job.json`, and its formal run manifest. The launcher strips the
render-build override from the four non-LLM submission processes.

## Executable lifecycle

The v1 contract is intentionally blocked. After the Source promotion workflow
creates, reviews, tracks, and commits the active v2 successor, every formal
command must name that successor explicitly:

```bash
SPEC=vln/experiments/vln_targeted_gap_campaign_v2.json
BATCH=vln-targeted-gap-campaign-v2-seed0

python3 vln/scripts/run_targeted_gap_campaign.py plan \
  --spec "$SPEC" --batch-id "$BATCH" --gpus 0,1,2,3
python3 vln/scripts/run_targeted_gap_campaign.py search \
  --spec "$SPEC" --batch-id "$BATCH" --gpus 0,1,2,3
python3 vln/scripts/run_targeted_gap_campaign.py freeze \
  --spec "$SPEC" --batch-id "$BATCH" --gpus 0,1,2,3
python3 vln/scripts/run_targeted_gap_campaign.py val-seen \
  --spec "$SPEC" --batch-id "$BATCH" --gpus 0,1,2,3
python3 vln/scripts/run_targeted_gap_campaign.py reverie-test \
  --spec "$SPEC" --batch-id "$BATCH" --gpus 0,1,2,3
```

Each phase is a separate global barrier. `search` executes the 55 development
jobs; `freeze` writes exactly one sixteen-winner artifact; `val-seen` runs the
sixteen fresh frozen configurations; and `reverie-test` writes five
submission-only trajectory files. A resume repeats the same stage command
with `--resume`. Failed, invalid, or orphaned attempts remain terminal unless
they are independently classified as infrastructure failures and the operator
also supplies `--retry-failed --retry-reason TEXT`.

## Design-contract mapping

This is a task-local executable schema consumed by
`vln/scripts/run_targeted_gap_campaign.py`. It deliberately differs from the
canonical template and from the legacy consistency-search schemas; the
experiment-designer checklist maps as follows:

| Canonical concern | Task-local field |
|---|---|
| lifecycle and supersession | `status`, `supersedes`, `superseded_by` |
| purpose and hypothesis | `purpose` |
| exact workbook-derived scope/matrix | `scope`, `gap_inventory`, `cells` |
| split roles and phase barrier | `protocol.phase_order`, `protocol.global_barriers` |
| horizon and update dose | `protocol.horizons`, `protocol.update_dose` |
| seeds and RNG derivation | `protocol.randomness` |
| data/checkpoint/order identity | `data_bindings` |
| matched Source policy | `source_control` |
| per-method supervision | `supervision_contracts`, each cell's `supervision` |
| candidates and provenance | `candidate_grids` |
| selection and freeze | `selection`, `freeze` |
| leakage and stop rules | `leakage_controls`, `stopping` |
| manifests and diagnostics | `outputs` |
| operational assignment | `schedule` |
| hidden-test provider | `hidden_test_transfer` |

The focused stdlib test validates these mappings, all referenced tracked
digests, recomputes the exact 16-cell matrix from the pinned workbook, checks
the 4×4 queue allocation and 55-job count, enforces the ten-candidate ceiling
(actual maximum five) and learning-rate-majority rule, and verifies split,
seed, version, Source-promotion, supervision, and FeedTTA-LLM contracts. Once
the missing Source provenance is supplied and target-server preflight passes,
an operator must review the generated v2 successor again before launch; smoke
or development outputs are never formal results.
