# NavTTA evidence map and recurring failure modes

Use this map to locate current evidence. Treat paths as discovery anchors: verify the active commit, status, and supersession metadata before drawing a conclusion.

## Implementation evidence

- Shared task-agnostic adapters: `core/navtta_core/tta/tta_core.py` and adjacent `core/tests/`.
- VLN task glue: `vln/navtta_vln/discrete_tta.py` and `vln/navtta_vln/continuous_tta.py`.
- AVN task glue and runners: `avn/baselines/`, `avn/scripts/`, and `avn/experiments/`.
- Upstream evidence: `references/catalog.json` and pinned, read-only `references/repos/` checkouts.
- Task authorization and provenance policy: repository `AGENTS.md`, then each task's `README.md` and `STATUS.md`.

Never assume prose status files are current enough to establish a running campaign. Ask `$navtta-experiment-operator` for live state and `$navtta-result-auditor` for evidence eligibility.

## Experiment and result evidence

- Active and historical designs: `<task>/experiments/`; inspect `active`, `superseded`, and `superseded_by` language instead of selecting by filename date alone.
- Compact analyses: `<task>/results/analysis/`; trace every number back to a run or manifest.
- Formal run identities: `<task>/results/runs/` plus the repository-level manifest tools under `tools/`.
- Legacy or incomplete-provenance evidence: `<task>/results/legacy/`; use it for hypotheses only, never as a formal table row.
- Literature corpus and method-selection notes: `docs/literature/tta_embodied_navigation_cvpr2027/`.

## Recurring failure modes to test first

1. **Action-protocol mismatch.** Sampling versus target-native argmax can dominate the apparent adaptation effect. Require a matched Source action protocol.
2. **State-lifetime drift.** Per-action, per-episode, and whole-stream statistics or optimizer state are not interchangeable. Trace every reset boundary.
3. **Horizon-dependent dose.** A longer or more correlated stream changes update count and drift even when learning rate is unchanged. Analyze `LR x accepted updates x scope x horizon/window`.
4. **No-op selection.** Zero updates, unreachable windows, unchanged action traces, or an ineffective trainable scope can look stable and win a low-drift tie-break. They do not prove adaptation.
5. **Replay mismatch.** A named full-policy update may reach only a high-level decision subgraph; replay may omit recurrent, map, mask, or stochastic state.
6. **Feedback mismatch.** Binary success is supervision. Verify evaluator endpoint, eager versus lazy access, timing, query count, and whether hidden-test feedback is legally available.
7. **Single-order overfitting.** Scene-blocked order and stream length can change continual-TTA behavior. Prefer paired multi-order evidence and report mean, variation, worst case, update count, and drift.
8. **Metric trade-off.** An SR-first winner may reduce SPL or another task metric. Preserve Pareto trade-offs and exact success counts.
9. **Process-success confusion.** Exit code zero, parsed metrics, runner validation, and formal provenance are separate evidence levels.
10. **Protocol-change novelty.** Fixing an implementation, changing action selection, adding a better control, or porting a method to another task is not automatically a new algorithm.

## Candidate research axes

Use these only as hypothesis seeds, not novelty claims:

- horizon- or action-normalized adaptation budgets;
- drift budgets, source anchoring, rollback, and failure-loop interruption;
- uncertainty calibrated for changing legal-action sets;
- multimodal and temporal reliability beyond raw entropy;
- replay-reachable rather than nominal trainable scope;
- feedback gain versus query-cost Pareto behavior;
- stability across stream orders and task horizons.

For each axis, search primary literature and official code, name the closest mechanism, propose at least one rival explanation, and design the smallest experiment that produces different predictions.
