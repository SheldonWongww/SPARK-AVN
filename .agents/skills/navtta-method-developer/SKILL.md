---
name: navtta-method-developer
description: Implement an approved original or project-defined navigation test-time adaptation method in NavTTA. Use when turning a reviewed hypothesis and method contract into task-agnostic core code, authorized AVN/VLN/ObjectNav adapters, configuration surfaces, diagnostics, and tests. Do not use for open-ended ideation, published-method fidelity review, experiment selection, or server operation.
---

# NavTTA Method Developer

Implement the smallest code change that realizes an approved mechanism without changing its scientific contract.

## Require inputs

1. Resolve the repository with `git rev-parse --show-toplevel` and read its `AGENTS.md` plus the authorized task documentation.
2. Require a versioned hypothesis card with an `advance` verdict, a nearest-prior-art or reproduction dossier when relevant, and an experiment design identifying observable predictions and diagnostics.
3. Confirm the exact task/model scope. Do not activate a task because code or old results exist.
4. Stop and return unresolved choices when the update signal, supervision class, action timing, trainable scope, state lifetime, or reset boundary is ambiguous.

Read [references/implementation-gates.md](references/implementation-gates.md) before editing.

## Implement by ownership layer

- Put only simulator-independent adaptation state, objectives, optimizers, and diagnostics in `core/`.
- Put action-space mapping, model feature extraction, evaluator feedback, replay callbacks, and simulator lifecycle in the authorized task directory.
- Never edit `references/repos/`; use pinned upstream code as evidence only.
- Extend existing adapters and configuration translators instead of creating a second hidden implementation path.
- Preserve Source behavior. Do not modify a baseline, evaluator, metric, episode order, or checkpoint-loading rule merely to improve the proposed method.

## Preserve the mechanism

Implement the approved signal, causal timing, update target, memory, reset, and budget exactly. Keep an action independent of the update generated from that same action. Treat binary success or any target outcome as feedback supervision and expose its timing and query count.

Record diagnostics that can falsify the mechanism: eligible observations, attempted and accepted updates, skipped reasons, gradient/update norms, parameter drift and hashes, action changes, memory/replay coverage, feedback queries, and method-specific state. A result metric alone is insufficient.

If implementation constraints require a semantic change, stop, version the method contract, and send the change back to `$navtta-research-mentor` and `$navtta-experiment-designer`; do not silently redefine the method in code.

## Verify before handoff

1. Run focused unit tests for equations, masks, causal timing, state/reset, RNG, and parameter selection.
2. Run zero-write or zero-learning-rate controls through the complete adapter path and require matched Source actions, trajectories, metrics, and full-model hashes where deterministic.
3. Test replay reachability and gradient coverage for every claimed trainable component.
4. Run only the smallest authorized smoke needed to prove lifecycle correctness; do not tune during implementation.
5. Run task tests, core tests affected by the change, and `python3 tools/verify_layout.py`.

Return changed paths, the method-contract version, exact tests and outcomes, known deviations, diagnostics coverage, the resulting commit requirement, and the next allowed designer/operator stage. Do not launch a search or claim effectiveness under this skill.
