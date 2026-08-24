# VLN TTA consistency-search protocol (v2)

This protocol supersedes the earlier dual-split/no-regression proposal for the
three historical `*consistency_search_v1.json` specifications. Those files stay
unchanged for result replay; the corrected protocol is encoded only in the
three `*consistency_search_v2.json` specifications.

## Split roles and order control

`val_unseen` is an explicitly disclosed **development and selection split**.
It is not a hidden test result and must be labelled as such in tables and
claims. Every candidate is evaluated on each complete globally shuffled stream
with order seeds 1, 2, and 3. Canonical scene-blocked seed 0 is excluded from
selection.

After selection is frozen, only the winner is run on `val_seen`, again on
global-shuffle seeds 1, 2, and 3. Those runs are a retention/sanity report and
cannot trigger reselection. Test submissions are outside this workflow.

Source uses the authenticated deterministic argmax aggregate for the relevant
split. Reordering does not change that aggregate. Candidate deltas are computed
against this fixed Source value. The runner consumes the tracked per-setting
`*_reused_source_controls.json` ledgers directly; it does not infer provenance
from a shared results symlink (notably, HAMT-R2R uses a different source batch).

## Selection

For each candidate and metric, compute the mean and sample standard deviation
over order seeds 1, 2, and 3. The robust score is

`LCB = mean(candidate - Source) - sample_std(candidate - Source)`.

R2R and R2R-CE rank by SPL LCB, then SR LCB. REVERIE ranks by RGSPL LCB,
then RGS LCB. Remaining ties prefer lower mean relative parameter drift, then
fewer mean updates, then lexicographically smaller candidate ID. There is no
positive-result filter, no no-regression gate, and no paper-default fallback:
the best registered candidate is selected even when every Source-relative LCB
is negative, and that fact is reported.

## Candidate budget

Each method contains one explicit `paper_anchor`; no implicit Cartesian product
is allowed.

- Tent: four learning rates.
- FSTTA: paired fast/slow learning rates at common scales 0.1, 0.3, and 1.0 of
  `(6e-4, 1e-3)`; R2R-CE also includes scale 0.03. The formal paper anchor
  fixes `rho=.95, tau=.7, a=.9, b=1.1`, M=3, N=4, q=.1, and the last four
  LayerNorms. Eq. 6 variance history spans the complete test stream
  (`reset_var_hist_each_episode=false`). The released-code rollout-reset
  behavior and `.9/.5/.5/1.5` band are ablations, not the paper anchor.
- EAM: learning rates `{1e-6, 3e-6, 1e-5}` with confidence scale 0.4, memory
  32, batch 8, and update interval 1 fixed.
- FeedTTA: learning rates `{2e-6, 5e-6, 1e-5}`, `paper_full` scope, and the
  benchmark-specific reported `(p, alpha)` fixed. The formal VLN port keeps
  each evaluator's target-native argmax action rule; paper policy sampling is
  an optional, explicitly labelled ablation. The SGR RNG seed equals the order
  seed; native-argmax jobs have no action-sampling seed, and
  `sgr_mode=paper_main` is explicit. R2R/R2R-CE use `alpha=+0.1`;
  REVERIE uses `alpha=-0.2`. Main results must be labelled as the task-adapted
  `FeedTTA-argmax` port, not as an unbiased on-policy reproduction. They compare
  to the ordinary argmax Source; only the optional policy-sampling ablation
  requires a same-seed sampled Source control.
- ATENA: six explicit points: official benchmark anchor, paired learning-rate
  half/double variants, two positive delta variants, and one positive lambda
  variant. Delta 0 and lambda 0 are forbidden.
- IDEA: the four `lr x tau` points use paper `O=50` in search and retention.
  There is no reduced-O screening surrogate.

IDEA's paper states that source statistics are estimated from 128 randomly
selected source-training trajectories, but it does not specify whether those
trajectories are teacher-forced or policy rollouts.  This benchmark therefore
freezes one reproducible interpretation: a SHA256-ranked subset of 128 real
training examples is traversed by the frozen source policy with each target
model's native argmax action rule.  Artifacts and bindings record this as
`collection_policy=frozen_source_argmax_rollout`; it is a disclosed protocol
choice, not a claim that the unspecified rollout detail exactly matches the
paper.

## Fail-closed evidence contract

A selection or report is produced only when every planned seed is complete.
Each result must have a successful formal manifest with immutable identity,
checkpoint, dataset, order-manifest, metric-artifact, configuration, and
diagnostics digests consistent with the plan. Diagnostics must account for the
complete stream and preserve the distinction between unsupervised methods and
FeedTTA/ATENA binary episode feedback. Any failed, missing, duplicate, stale,
or unauthenticated job aborts the stage.

ATENA additionally requires exact full-episode replay diagnostics and honest
task-level evidence for the replay-reachable high-level policy scope; it must
not call that limited scope a full end-to-end policy update. IDEA remains prelaunch-blocked until precomputed
Source-statistics artifacts are available per model, SHA256-pinned, accepted by
the launcher, and attested in run diagnostics. StreamVLN is not added: the
current launcher rejects shuffled order seeds for it and the TTA translator has
no StreamVLN adapter. These are explicit blocked capabilities, not skipped
results.

## Scheduling

One `(model, method)` phase runs at a time. R2R uses the measured per-method
caps recorded in its spec; REVERIE uses one worker per `(model, method)`;
R2R-CE uses three workers and a strict ETPNav-before-BEVBert model barrier.
IDEA remains one worker until calibrated. A nonzero worker exit stops the phase
and campaign.

For each cell the campaign executes:

1. `--stage search`
2. `--stage select`
3. `--stage retention`
4. `--stage report`

The first default phase is R2R / DUET / Tent. Use `--dry-run` on a launch stage
to emit the exact commands without invoking a simulator.
