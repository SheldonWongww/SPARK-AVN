# Research skill repository audit

Audit date: 2026-07-24

These repositories are retained only as read-only upstream references under
`references/repos/research/`. They are Git-ignored and are not installed as
active Codex skills.

## `phd-skills`

- Upstream: <https://github.com/fcakyon/phd-skills.git>
- Audited commit: `8d642d3e114ee1d1e4d000f918d71e9bf0453dc2`
- License: MIT
- Static validation: all shipped Shell scripts pass `bash -n`; JSON manifests
  parse successfully; no symlinks were found.
- Useful components: paper reproduction, experiment design, evidence-first
  debugging, same-step run comparison, and paper/code verification.
- Restrictions: do not enable the complete plugin unchanged. Its optional
  hooks can read `.env`, send notifications through ntfy/Slack/email, compile
  LaTeX automatically, and save transcript-derived state under the user's home
  directory. The `launch` skill also contains destructive cleanup examples.
  Review and port individual methodology skills before activation.

## `academic-research-skills-codex`

- Upstream: <https://github.com/Imbad0202/academic-research-skills-codex.git>
- Audited commit: `f8d6b061efe98564a3f554c917fce66dcef6ca54`
- License: CC BY-NC 4.0
- Static validation: repository quality gates passed; Python files compile;
  Shell scripts pass `bash -n`; JSON manifests parse successfully; no symlinks
  or direct `shell=True`, `os.system`, `eval`, or `exec` calls were found in
  the installable suite.
- Test limitation: the full upstream pytest suite was not rerun because the
  current system Python does not have `pytest`; no dependency was installed
  merely for this read-only audit.
- Safety posture: the Codex hook pack and cross-model calls are disabled by
  default. External manuscript upload requires explicit provider selection and
  user consent. The vendored Claude update checker and live cross-model smoke
  test can access the network but are not part of default Codex execution.
- Restrictions: keep the optional full-runtime profile, hooks, cross-model
  verification, and update checker disabled unless they receive a separate
  task-specific review. Respect the non-commercial license.

## Decision

Both repositories are acceptable as inactive reference material. Neither is
approved for wholesale activation in NavTTA. Any future installation should
select only the needed skills, preserve NavTTA's `AGENTS.md` rules, and add
NavTTA-specific run-manifest and data-provenance requirements.
