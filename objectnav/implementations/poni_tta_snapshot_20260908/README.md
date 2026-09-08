# PONI TTA implementation snapshot

This source-only snapshot contains the runtime files needed by the retained
FeedTTA, ATENA, EAM, FSTTA, and Tent runs: their method directories,
global-continual stream builder/runner/monitor utilities, shared TTA helpers,
PONI integration edits, transfer configuration, result merger, and environment
requirement files.

Hyperparameter-search drivers and records, multi-method launchers, old campaign
scripts, bytecode, datasets, scenes, environments, and checkpoints are not
included.

The snapshot is not a complete PONI checkout. Start from PONI commit
`30682c2bdcd820eec8f72043b2579eb045d547bf`, review/apply the files under
`PONI/`, and use the task-agnostic implementation in
`NavTTA/core/navtta_core/tta/tta_core.py`.
