"""Runtime feedback bridge for episode-supervised RedNet TTA methods.

The original PONI evaluator is left untouched.  This module wraps its VectorEnv
after construction and forwards terminal ``info['success']`` through a lazy
provider.  ATENA therefore reads oracle feedback only for queried episodes,
while FeedTTA reads it for every completed episode.
"""


def _success_from_info(info):
    if isinstance(info, dict):
        if "success" in info:
            return float(info["success"])
        for value in info.values():
            try:
                return _success_from_info(value)
            except KeyError:
                pass
    raise KeyError("terminal Habitat info does not contain success")


class _FeedbackVectorEnvProxy:
    def __init__(self, envs, evaluator):
        self._envs = envs
        self._evaluator = evaluator

    def __getattr__(self, name):
        return getattr(self._envs, name)

    def step(self, *args, **kwargs):
        outputs = self._envs.step(*args, **kwargs)
        agent = getattr(self._evaluator, "global_agent", None)
        if agent is not None and hasattr(agent, "on_episode_feedback"):
            for output in outputs:
                _, _, done, info = output
                if done:
                    # Capture this exact info object.  The provider is invoked
                    # lazily so ATENA's non-query path never reads the oracle.
                    def provider(terminal_info=info):
                        return {"success": _success_from_info(terminal_info)}

                    agent.on_episode_feedback(provider)
        return outputs


def install_feedback_env_hook(transfer_evaluator_module):
    """Patch TransferEvaluator._init_envs once for this Python process."""

    evaluator_class = transfer_evaluator_module.TransferEvaluator
    if getattr(evaluator_class, "_tta_feedback_hook_installed", False):
        return
    original_init_envs = evaluator_class._init_envs

    def init_envs_with_feedback(self, config=None):
        original_init_envs(self, config)
        self.envs = _FeedbackVectorEnvProxy(self.envs, self)

    evaluator_class._init_envs = init_envs_with_feedback
    evaluator_class._tta_feedback_hook_installed = True

