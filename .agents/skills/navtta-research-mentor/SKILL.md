---
name: navtta-research-mentor
description: Generate, challenge, and refine evidence-grounded research ideas for navigation test-time adaptation (NavTTA). Use when Codex must assess a proposed mechanism, synthesize literature/code/results into a falsifiable hypothesis, distinguish novelty from a port or protocol change, or decide whether an AVN, VLN, or ObjectNav idea merits reproduction and experiments. Produces a claim ledger, mechanism matrix, contradiction log, and hypothesis card while checking confirmation bias, split leakage, supervision mismatch, and unsupported novelty claims.
---

# NavTTA Research Mentor

Turn a research question into a defensible decision and a falsifiable handoff. Treat an idea as a hypothesis to stress-test, not a conclusion to defend.

## Load the references

- Read [references/evidence-protocol.md](references/evidence-protocol.md) before evaluating claims or novelty.
- Read [references/navtta-evidence-map.md](references/navtta-evidence-map.md) when the question depends on existing implementations, experiments, or known failure modes.
- Read [references/templates.md](references/templates.md) and use all four templates for a full idea review. Use only the requested template for a narrowly scoped update.

## Establish the decision boundary

1. Restate the exact decision, task, model, shift, adaptation timing, and evidence cutoff.
2. Confirm which task is authorized by the current request and repository policy. Do not infer activation from code or historical results, and do not carry an old task decision into a new scope silently.
3. Declare what information is available at adaptation time and classify the supervision before comparing methods.
4. Separate verified facts, cited prior claims, local observations, inferences, and untested hypotheses.
5. Mark missing primary evidence as unknown. Do not fill gaps with plausible detail.

## Build the evidence package

1. **Populate the claim ledger.** Make every consequential statement atomic and attach an exact source locator or an explicit verification test.
2. **Construct the mechanism matrix.** Compare the idea with Source, the closest baselines, and the nearest prior art along signal, update target, timing, memory, reset, supervision, assumptions, and claimed difference.
3. **Maintain the contradiction log.** Search deliberately for evidence that weakens the preferred mechanism, including null results, architecture incompatibilities, alternate explanations, and protocol mismatches.
4. **Write the hypothesis card.** State one causal mechanism, discriminating predictions, controls, falsifiers, split policy, supervision class, metrics, and stop criteria.

Do not collapse conflicting evidence into a narrative average. Preserve each conflict until a source check or experiment resolves it.

## Apply the four integrity gates

- **Confirmation bias:** Require a credible rival explanation and a test whose outcomes distinguish it from the proposed mechanism.
- **Split leakage:** Account for every data-dependent choice and statistic by split. Never tune, select, or construct source statistics from the final evaluation stream unless the declared online protocol explicitly permits that information.
- **Supervision mismatch:** Keep fully unsupervised TTA separate from methods using binary episode feedback, task labels, oracle success, or equivalent target feedback. State when feedback arrives and which future decisions it can affect.
- **False novelty:** Compare mechanisms rather than names. Label the contribution as a new algorithm, navigation adaptation, protocol, analysis, or combination; downgrade it to a port or recombination when evidence supports only that claim.

Treat a failed gate as a required revision, not a wording problem.

## Reach a decision

Assign exactly one verdict:

- `advance`: evidence and a discriminating experiment justify the next stage;
- `revise`: the mechanism is plausible but the hypothesis or protocol is underspecified;
- `reject`: existing evidence or an unavoidable mismatch defeats the central claim;
- `blocked`: a named primary source, artifact, or task decision is unavailable.

Report the strongest supporting evidence, strongest contradiction, decisive unknown, and cheapest valid next test. Calibrate confidence to evidence quality and coverage.

## Hand off

Return the completed artifact package with source locators and unresolved items. For a workflow handoff:

- send prior-method parity questions to `$navtta-method-reproducer`;
- send an accepted hypothesis card to `$navtta-experiment-designer`;
- send completed run artifacts, never provisional impressions, to `$navtta-result-auditor`.

Do not implement methods, launch experiments, or certify results under this skill.
