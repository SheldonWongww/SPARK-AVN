# NavTTA core

This package contains only task-agnostic TTA algorithms and experiment utilities. It must not import Habitat, Matterport3D Simulator, SoundSpaces, or a task-specific policy.

Install it into each baseline environment from the repository root:

```bash
python -m pip install -e core
```

`build_adapter` also accepts the explicit `AUDIT_ZERO_UPDATE=True` evidence
mode.  It keeps native adapter control flow active while intercepting optimizer
and direct parameter writes, then reports attempted/suppressed write counts,
zero actual updates, selected-parameter drift, and exact before/after hashes
for the complete deployed model state (all parameters and buffers).  EAM hashes
both source and auxiliary branches; FSTTA and ATENA additionally hash the slow
anchor and dynamically constructed prediction head.  Task runners must expose
this only through a dedicated audit schema;
it is not a hyperparameter-search option or a substitute for a Source run.
