---
name: navtta-research-workflow
description: Route NavTTA research work across specialist skills for idea refinement, published-method reproduction, original-method implementation, experiment design and operation, and result auditing. Use when a navigation test-time adaptation request is ambiguous, spans multiple research stages, needs the next responsible specialist selected, or requires a traceable handoff from evidence and hypotheses through formal results. This skill coordinates artifacts and gates only; it does not duplicate specialist research logic.
---

# NavTTA Research Workflow

Select the smallest capable specialist, preserve its artifacts, and route the result to the next accountable stage. Do not perform a specialist's work inside this router.

## Route the request

| User intent or current state | Invoke | Required output artifact |
|---|---|---|
| Generate, challenge, or refine an idea; assess mechanism or novelty | `$navtta-research-mentor` | evidence package with claim ledger, mechanism matrix, contradiction log, and hypothesis card |
| Reproduce an existing paper or implementation; resolve paper/code parity | `$navtta-method-reproducer` | reproduction dossier with provenance, method contract, parity evidence, and unresolved deviations |
| Turn a hypothesis or reproduction target into a valid evaluation | `$navtta-experiment-designer` | frozen experiment specification with protocol, comparison matrix, gates, and resource plan |
| Implement an approved original or project-defined mechanism | `$navtta-method-developer` | implementation record with contract version, code surfaces, diagnostics, tests, and deviations |
| Launch, monitor, resume, or diagnose approved runs | `$navtta-experiment-operator` | execution record linking run manifests, statuses, logs, failures, and produced artifacts |
| Verify provenance, comparability, statistics, or publication eligibility | `$navtta-result-auditor` | audit report with per-run eligibility and an evidence-backed verdict |

Apply these routing rules:

1. Honor an explicit specialist request and route directly when its required inputs exist.
2. Start an underspecified research idea with the mentor.
3. Route claims of fidelity to an existing method through the reproducer before designing comparative experiments.
4. Route any run request without an approved experiment specification to the designer first.
5. Route implementation of a reviewed original mechanism to the developer; keep published-method fidelity work in the reproducer.
6. Route every result intended for a formal table or scientific claim to the auditor.
7. Route a conceptual failure back to the mentor, a parity failure to the reproducer, an implementation failure to the developer, a protocol failure to the designer, and an execution failure to the operator.

If a required sibling skill is unavailable, report that dependency instead of reconstructing its procedure here.

## Preserve stage gates

Use the normal path only as far as the request requires:

`mentor -> reproducer when prior-art parity is material -> designer -> developer for original methods -> operator -> auditor`

Skip a stage only when its required artifact already exists and is identified by an exact locator. Never treat a chat summary as a substitute for a versioned artifact when the specialist requires one.

Stop at a gate when the next stage would require an unresolved scientific choice, missing provenance, new task activation, changed supervision class, or authorization to spend compute. Return the choice to the user or responsible specialist.

Do not call provisional metrics formal, merge feedback-dependent and fully unsupervised methods, or promote an unaudited run into a research conclusion. Leave those judgments to the responsible specialist.

## Use the handoff envelope

Require each stage to return this compact envelope alongside its domain artifact:

```markdown
## NavTTA handoff
- Request ID / objective:
- Completed stage and specialist:
- Task / model / method scope:
- Repository commit and relevant working-tree state:
- Input artifact locators and versions:
- Output artifact locators and versions:
- Dataset, checkpoint, and upstream provenance locators:
- Declared supervision class and feedback timing:
- Split roles, stream/reset boundary, and allowed test-time information:
- Locked decisions:
- Open assumptions, deviations, and blockers:
- Requested next stage and decision:
```

Use `unknown` rather than inventing a value. Preserve prior envelopes or link them; do not silently rewrite locked decisions. When a value changes, record who changed it, why, and which downstream artifacts became stale.

## Check handoff readiness

Before invoking the next specialist, verify only the interface contract:

- **Mentor to reproducer/designer:** include the verdict, hypothesis card, evidence locators, and unresolved contradictions.
- **Reproducer to designer:** include the pinned upstream identity, reproduction status, deviations, and parity evidence.
- **Designer to developer:** include the frozen mechanism contract, required diagnostics, authorized tasks, and behavior gates.
- **Developer to designer/operator:** include the implementation record, tests, deviations, and commit; return semantic changes to the designer before launch.
- **Designer to operator:** include the approved executable specification locator, immutable run identity inputs, launch gate, and stop conditions.
- **Operator to auditor:** include run manifest locators, configuration and digest bindings, status/failure record, and raw result locators.
- **Auditor onward:** include eligibility, exclusions, unresolved defects, and the exact claims the evidence permits.

Do not inspect or reinterpret the specialist artifact beyond determining whether these named fields are present. Send incomplete artifacts back to the producing stage with the missing fields listed.

## Report the route

State the chosen specialist, why it owns the next decision, the exact inputs supplied, the expected handoff artifact, and the next gate. For multi-stage requests, update this routing summary after each handoff rather than promising downstream completion prematurely.
