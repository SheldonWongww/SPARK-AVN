---
name: navtta-method-reproducer
description: Reproduce, audit, or port a test-time adaptation method into NavTTA from a paper and official implementation. Use for method provenance, paper-versus-code alignment, navigation task ports, trainable-scope mapping, action or feedback semantics, replay equivalence, Source parity, and zero-write audits in AVN, VLN, or ObjectNav.
---

# NavTTA Method Reproducer

Build an auditable semantic port, not a name-compatible approximation. Treat the paper, pinned official code, and NavTTA port as three separately evidenced layers.

## Establish the contract

1. Resolve the repository with `git rev-parse --show-toplevel`, then read its `AGENTS.md` and the target task's `README.md` and `STATUS.md`.
2. Confirm that the requested task is authorized by the current user request and repository policy. Do not infer activation from the mere presence of code, results, or an old chat summary.
3. Read [references/reproduction-contract-template.md](references/reproduction-contract-template.md) completely and create a filled contract in the target task's tracked experiment area.
4. Pin the paper identity, official repository URL and commit, license, and local read-only reference path. Record ambiguities instead of silently choosing an interpretation.
5. Extract the paper contract first, the official execution contract second, and the proposed port contract third. For every divergence, state the evidence, reason, expected effect, and whether the result remains method-native or becomes a named variant.

Use the read-only scanner to locate likely implementation surfaces before tracing control flow:

```bash
python3 .agents/skills/navtta-method-reproducer/scripts/scan_method_surface.py \
  references/repos/<upstream> --format text
python3 .agents/skills/navtta-method-reproducer/scripts/scan_method_surface.py \
  <task>/<implementation> --format json
```

The scanner only reads source text and writes its report to stdout. Treat its matches as leads, not proof; inspect every relevant call path.

## Respect repository boundaries

- Keep simulator, Habitat, model, dataset, and task adapters inside `avn/`, `vln/`, or `objectnav/`.
- Put only dependency-free, task-agnostic adaptation logic and experiment utilities in `core/`.
- Never edit `references/repos/`. Preserve upstream licenses, URLs, and pinned commits.
- Keep datasets, checkpoints, raw logs, videos, and TensorBoard artifacts out of Git. Put provenance and SHA256 records in the task's `manifests/` areas.

## Preserve semantics

- Define the action space, invalid-action mask, STOP behavior, action selection rule, and every RNG stream. Compare only with a Source control using the same action protocol.
- Enforce causality: an action comes from the pre-update policy; an update affects only later actions. Episode feedback arrives after termination and affects only later episodes.
- Classify all environment-derived signals. Unsupervised methods must not read hidden success, reward, distance, oracle action, or evaluator-only fields. Mark any method consuming binary episode outcome as feedback-supervised.
- Define persistent and reset state independently: model weights, optimizer, teachers/anchors, replay, recurrent/map state, prediction heads, and counters.
- For replay, specify the exact payload and prove either exact deterministic reconstruction or explicitly weaken the claim. Compare replayed logits/features/loss/gradients under a stated tolerance.

## Gate the port

Require these gates before smoke tests or searches:

1. Unit parity for equations, masks, reductions, schedules, and parameter selection.
2. Official-anchor parity on a small deterministic fixture when the upstream can run.
3. Matched Source parity for checkpoint, data, episode order, model seed, action rule, evaluator, and horizon.
4. Zero-write parity with the complete adapter path active: intercept optimizer and direct writes, require zero parameter/buffer mutation by hash, and compare per-step actions, trajectories, and metrics to matched Source.
5. Replay parity, including recurrent state, masks, stochastic layers, mutable buffers, and update-time reconstruction.

Do not dismiss a mismatch as numeric noise. Localize it to protocol, RNG consumption, mutation, task mapping, or an implementation defect. Stop and record a blocked contract if equivalence cannot be supported.

## Deliverables

Return the filled reproduction contract, implementation and config paths, focused tests, scanner findings, parity evidence, unresolved deviations, and the exact next allowed experiment stage. Run relevant tests and `python3 tools/verify_layout.py` after structural repository changes.
