"""PONI GlobalAgent with exact terminal feedback hooks for FeedTTA."""

from global_agent import GlobalAgent


class FeedTTAGlobalAgent(GlobalAgent):
    def __init__(self, cfg, device):
        if int(cfg.NUM_ENVIRONMENTS) != 1:
            raise ValueError("PONI + FeedTTA requires NUM_ENVIRONMENTS=1")
        super().__init__(cfg, device)
        self._feedtta_episode_active = False

    def act(self, batched_obs, agent_states, steps, g_masks, l_masks):
        if not bool(l_masks.reshape(-1)[0].item()):
            if self._feedtta_episode_active:
                raise RuntimeError("new episode started before FeedTTA feedback")
            self.sem_seg_model.episode_start()
            self._feedtta_episode_active = True
        return super().act(batched_obs, agent_states, steps, g_masks, l_masks)

    def on_episode_feedback(self, provider):
        if not self._feedtta_episode_active:
            raise RuntimeError("FeedTTA received feedback without an active episode")
        self.sem_seg_model.episode_end(provider)
        self._feedtta_episode_active = False

    def close(self):
        self.sem_seg_model.write_feedtta_diagnostics()
        super().close()

