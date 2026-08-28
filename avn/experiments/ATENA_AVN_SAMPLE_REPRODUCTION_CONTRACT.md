# ATENA-AVN(sample) reproduction contract

## 1. Identity and evidence

| Field | Value |
|---|---|
| Contract ID and status | `atena-avn-sample-port-v1` / `draft` (task smoke pending) |
| Method and category | `ATENA-AVN(sample)` / feedback-supervised with self-generated labels on non-query episodes |
| Paper | Heeju Ko et al., *Active Test-time Vision-Language Navigation*, NeurIPS 2025, arXiv:2506.06630 |
| Paper artifact | Public identifier `arXiv:2506.06630`; method summary is retained in `docs/literature/具身导航-TTA.md` |
| Official repository | <https://github.com/kuai-lab/NeurIPS25_att_vln.git> |
| Official pinned revision | `122b56990d46fea2aab7cc270f30cb2424cfe95f` |
| License | Apache-2.0; retained at `references/repos/tta/vln/NeurIPS25_att_vln/LICENSE` |
| Local upstream reference | `references/repos/tta/vln/NeurIPS25_att_vln` (read-only) |
| NavTTA target | AVN SMT+Audio now; the same task adapter surface is present for ENMuS |
| Port implementation | `core/navtta_core/tta/tta_core.py`; SMT+Audio and ENMuS trainer/config bindings under `avn/baselines/` |
| Contract owner/date | NavTTA / 2026-08-28 |

No secondary implementation or author clarification is used. The official
repository and paper are the two method authorities. The AVN sampled-action
choice is a declared task port, not an upstream claim.

## 2. Paper / official / port decision ledger

| Semantic item | Paper claim | Official code at pinned revision | NavTTA port | Classification | Decision and consequence |
|---|---|---|---|---|---|
| Objective | Eq. 1--4 form a convex mixture of policy and one-hot pseudo expert, then minimize its episode mean entropy on success and maximize it on failure | `ATENA_DUET/map_nav_src/r2r/agent_atena.py:492-502`; signed loss at `agent_base_atena.py:152-170` | `_mixture_entropy` and signed episode-end replay in `ATENAAdapter` | exact under the selected action protocol | Preserve the equation and episode mean reduction |
| Query/self-label | Eq. 5 queries external feedback when mean policy entropy exceeds the threshold; otherwise Eq. 6 supplies the self label | `agent_base_atena.py:153-166` | `ATENAAdapter.episode_end` | exact | Only queried episodes read `episode_stats["success"]`; non-query episodes use the self head |
| Self loss | Eq. 7--8 add binary self-prediction loss to MEO | `agent_base_atena.py:169-170` | BCE-with-logits weighted by `SELF_LOSS_WEIGHT` | exact mechanism; searched weight | Weight is declared per candidate |
| Optimizer | A fresh AdamW is constructed at every episode end with query/self learning rate | `agent_base_atena.py:158-168` | `_build_episode_optimizer` is called once per completed episode | exact | Adam moments do not persist across episodes |
| Update timing | One update after the episode outcome/self-label is chosen | `agent_base_atena.py:172-174` | `episode_end` performs the sole optimizer step | exact | The completed episode cannot be changed retroactively; updates affect later episodes only |
| Persistence | Updated policy is used for subsequent episodes | official test loop does not restore source weights between episodes | `EPISODIC=False`; reset only at a new run | exact | Continual stream adaptation |
| Official action | Evaluation invokes `feedback='argmax'`; the pseudo expert is one-hot of `a_t` | `main_nav_atena.py:188`; `agent_atena.py:433-476,496-500` | shared default `policy_argmax` remains available | exact baseline | The official protocol is preserved for audit and is not silently renamed |
| AVN action | N/A | N/A | `sample_from_policy`; one-hot uses the same sampled action tensor passed unchanged to `envs.step` | explicit deviation | Report only as **ATENA-AVN(sample)**. It reinforces/penalizes behavior that actually generated AVN feedback, never the policy argmax proxy |
| Parameter scope | Official optimizer covers `vln_bert.parameters()` including the self head | `agent_base_atena.py:159-165` | replay-reachable non-critic actor parameters plus the self head | task mapping | Value-only critic and graph-disconnected heads are excluded; effective names/counts must be logged |
| Episode graph | Official implementation retains the episode computation graph | accumulated `m_entropy` is backpropagated at episode end | detached CPU snapshots with deterministic one-step replay | resource-preserving implementation change | Claimed equivalent only when replay reachability and feature error checks pass |
| Feedback | Binary success is read only for high-entropy queried episodes | `agent_base_atena.py:158-166` | AVN `episode_stats["success"]` after termination | task mapping | This is feedback-supervised, not unsupervised TTA |

Precedence is paper for equations and supervision, then official code for
execution details. The sampled AVN behavior deliberately overrides only the
official argmax action contract and is therefore named as a variant.

## 3. Ownership boundary

| Component | Location | Reason | Forbidden dependency/write |
|---|---|---|---|
| Task-agnostic ATENA | `core/navtta_core/tta/tta_core.py` | Objective, query gate, self head, replay and diagnostics are simulator-independent | No Habitat, SMT+Audio or ENMuS imports |
| AVN binding | `avn/baselines/smt_audio/.../ppo_trainer.py`, `avn/baselines/enmus/.../ddppo_enmus_trainer.py` | Supplies policy inputs, sampled action and delayed AVN success | No edits under `references/repos/` |
| AVN configs/launcher | `avn/baselines/*/.../config`, `avn/scripts/run_smt_audio_val_search.py` | Explicitly opts into `sample_from_policy` | No cross-task assets |
| Run provenance | `avn/results/runs/` and batch artifacts under `avn/results/logs/` | Runtime-generated evidence | No raw logs or checkpoints in Git |

AVN is active under the repository policy. VLN and ObjectNav are not changed
by this contract.

## 4. Deployment semantics

### Stream and lifecycle

- Evaluation unit: one sequential AVN environment; batch size one.
- Causal order: observe -> frozen/current policy forward -> categorical sample
  -> cache that exact action -> execute the same integer action -> receive
  terminal feedback -> update once at episode end.
- Weights persist across episodes; policy, optimizer, self head and trajectory
  state start fresh for every candidate and source setting.
- The episode optimizer is reconstructed each episode, matching official code.
- Recurrent state, external memory and simulator state reset through the native
  episode boundary; they are not carried between candidates.

### Action contract

- Action space: task-native fixed discrete AVN action space. No candidate mask
  or VLN waypoint remapping is introduced by ATENA.
- STOP and terminal handling remain task-native.
- Selection: `torch.distributions.Categorical(...).sample()` through
  `ATENAAdapter.select_action` when the AVN config explicitly sets
  `ACTION_SELECTION_PROTOCOL=sample_from_policy`.
- The tensor passed to `ATENAAdapter.adapt(..., action=actions)` is the same
  tensor converted to integer IDs and passed to `envs.step`; replay stores a
  detached clone of it. The pseudo expert is never recomputed with argmax in
  this mode.
- The shared/default ATENA protocol remains `policy_argmax`; mismatching an AVN
  task action mode and ATENA protocol fails closed in both model trainers.
- The current development campaign uses seed 0 in a fresh process. Its reused
  sampled Source metrics have incomplete provenance, so results remain
  `validation-selected` until matched Source manifests are validated.

### Supervision and feedback contract

- Supervision: feedback-supervised on queried episodes; pseudo-label
  self-training on non-query episodes.
- Allowed inputs: policy observations/state and, after termination, AVN binary
  `success` only when the entropy gate queries it.
- Forbidden inputs: distance-to-goal, reward, shortest path, oracle action and
  any future episode outcome.
- Feedback affects only the update after the completed episode and therefore
  only later actions/episodes.
- Query budget is threshold-driven and unbounded by a separate count in the
  current paper-aligned implementation; query rate is reported.

### Parameters and mutable state

| State group | Selector | Mutable? | Reset boundary | Audit |
|---|---|---:|---|---|
| Deployed actor policy | replay-reachable `actor_critic` parameters except top-level `critic.*` | yes | new run | selected names/count and full-state digest |
| Value critic | top-level `critic.*` | no | N/A | exclusion count |
| Self-prediction head | ATENA-created MLP | yes | new run | parameter count and state digest |
| Episode optimizer | fresh AdamW at episode end | yes | every episode | optimizer type/current LR |
| Trajectory replay | detached inputs, features and executed actions | yes | every episode | step/storage counts and replay error |

## 5. Replay contract

- Unit/payload: every action step stores observations, recurrent state,
  previous action, masks, external memory/masks, policy feature and the exact
  environment action.
- Payload is cloned and detached to CPU; the model stays in eval mode.
- At episode end each step is moved back to the device and passed through the
  same task-provided policy callback.
- The replay objective uses the stored action, not a newly sampled or argmax
  action. Gradients are accumulated across steps and divided by trajectory
  length before one optimizer step.
- The first replayed feature in every episode must match its online feature at
  max absolute error `<=1e-5`; otherwise the run fails.
- Replay reachability is computed from real policy inputs before the first
  adaptation and removes graph-disconnected candidates from the optimizer.
- This is exact eval-mode replay within the declared reachable actor scope; it
  is not claimed to reproduce modules outside that graph.

## 6. Parity evidence

| Gate | Fixture and command | Expected invariant | Evidence | Result |
|---|---|---|---|---|
| Formula/action unit | `python -m unittest core.tests.test_tta_core.TTACoreTest.test_atena_task_native_sampling_uses_executed_action` | a non-argmax executed sample is stored, replayed and named as pseudo expert | test output | local environment pending PyTorch; syntax checked |
| Official action default | ATENA core unit suite | omitted protocol selects argmax and rejects a different executed action | `core/tests/test_tta_core.py` | previously passing; rerun required in server env |
| Task binding | `python -m unittest discover -s avn/tests` | configs require AVN sample and scheduler validates sampled-action diagnostics | test output | local stdlib tests pass; GPU path pending |
| Matched Source | full sampled Source manifest for the identical stream | checkpoint/data/order/model seed/action/evaluator/horizon all match | run manifest | pending; existing metrics are provisional only |
| Zero-write | real adapter path with write suppression | identical full-state hashes, trajectories and actions versus Source | run artifacts | pending |
| Replay parity | 20-episode single-source smoke | replay feature error `<=1e-5`, reachable scope nonempty, all actions accounted for | diagnostics/manifest | pending school-server GPU smoke |

## 7. Deviations, gates, and handoff

| Issue | Evidence | Scientific impact | Resolution/variant label | Owner |
|---|---|---|---|---|
| AVN samples while official VLN is greedy | official `main_nav_atena.py:188`; AVN evaluation configs | changes trajectories and pseudo-expert distribution | `ATENA-AVN(sample)`; never pool with official-argmax runs | NavTTA |
| Step replay replaces retained graph | official episode graph versus `ATENAAdapter.episode_end` | equivalent only under deterministic replay | fail at `1e-5`; log reachability and replay diagnostics | NavTTA |
| AVN scope excludes critic and unreachable heads | task actor contains modules absent from the executed navigation graph | prevents meaningless parameters from entering optimizer | report exact names/counts; do not claim full end-to-end policy | NavTTA |
| Existing sampled Source lacks complete provenance | active search specification | prevents formal matched comparison | label outputs `validation-selected`; produce authenticated Source before a formal table | NavTTA |

- Allowed next stage: local tests, then a short school-server GPU smoke. The
  2000-episode development search is allowed only if the smoke gates pass.
- Required diagnostics: action steps, sampled-action steps, sample/argmax match
  rate, query rate, external/self feedback counts, update count, effective
  parameter names/count, drift, replay reachability, replay feature error and
  adaptation time.
- Formal-run prerequisites: clean pinned commit; dataset/checkpoint/stream
  hashes; action protocol and seeds; hardware; validated run and matched Source
  manifests.
- Remaining non-equivalences: sampled instead of greedy actions, AVN action and
  success semantics, and step replay instead of retained graphs.
