# NavTTA original-method implementation gates

Use this checklist for a method proposed by the research workflow. Existing published methods belong to `$navtta-method-reproducer` until their fidelity contract is resolved.

## Contract gate

- Name the mechanism and version independently from an aspirational paper title.
- Link the approved hypothesis card, closest-prior-art analysis, expected causal chain, rival explanation, falsifier, and stopping rule.
- Declare inputs available at decision time, objective, update target, action/update order, optimizer, state, memory, reset, randomness, supervision, and compute budget.
- Mark borrowed components and licenses. A combination of known components remains a combination until novelty is established.

## Code-surface gate

Inspect current code instead of relying on this list as exhaustive:

| Layer | Typical surfaces | Required evidence |
|---|---|---|
| Shared algorithm | `core/navtta_core/tta/` | no task/simulator imports; unit tests |
| Discrete VLN | `vln/navtta_vln/discrete_tta.py` | native action mask, STOP, feedback and replay semantics |
| Continuous VLN | `vln/navtta_vln/continuous_tta.py` | waypoint/action hierarchy, long-horizon dose and replay scope |
| AVN | `avn/baselines/`, `avn/scripts/`, `avn/experiments/` | audio/visual state, recurrent lifecycle and task feedback |
| Configuration | task CLI/YACS translators, experiment specs | one canonical parameter meaning and explicit defaults |
| Evidence | task tests, diagnostics, manifests | effective config, counters, hashes and provenance |

Use the sibling reproducer's read-only `scan_method_surface.py` to find likely surfaces, then trace executed control flow manually. Do not duplicate task-specific dependencies in `core/`.

## Behavioral gate

- Invalid actions are excluded before loss computation and selection.
- The current action is chosen before any update derived from it.
- Post-episode feedback affects only later episodes.
- Independent streams start from the same complete Source state.
- Optimizer, teacher/anchor, replay, RNG, recurrent/map state, and dynamic heads follow explicit reset rules.
- “Full policy” means gradients reach every claimed component; otherwise use the narrower replay-reachable label.
- No-op controls record attempted writes while producing zero actual writes and zero full-state drift.

## Test matrix

Require, as applicable:

1. formula and reduction tests on synthetic tensors;
2. legal-action mask and variable action-count tests;
3. action-before-update and delayed-feedback tests;
4. episode and stream reset tests;
5. deterministic RNG and replay reconstruction tests;
6. trainable-name/count and gradient-reachability tests;
7. zero-write matched-Source parity;
8. one short lifecycle smoke per authorized task/model family;
9. diagnostics schema and formal-manifest integration tests.

Do not use a development or held-out metric to repair an implementation test. Once behavior gates pass, return to the experiment designer to freeze any revised protocol before running comparative jobs.
