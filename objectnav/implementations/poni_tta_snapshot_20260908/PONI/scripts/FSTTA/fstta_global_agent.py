"""Episode-aware PONI GlobalAgent shim for FSTTA."""

from global_agent import GlobalAgent


class FSTTAGlobalAgent(GlobalAgent):
    """Forward ordered Habitat episode boundaries to FSTTA."""

    def __init__(self, cfg, device):
        if int(cfg.NUM_ENVIRONMENTS) != 1:
            raise ValueError(
                "PONI + FSTTA requires NUM_ENVIRONMENTS=1 so fast/slow state "
                "follows one unambiguous episode stream"
            )
        super().__init__(cfg, device)
        self._fstta_episode_active = False

    def act(self, batched_obs, agent_states, steps, g_masks, l_masks):
        is_episode_start = not bool(l_masks.reshape(-1)[0].item())
        if is_episode_start:
            if self._fstta_episode_active:
                self.sem_seg_model.episode_end()
            self.sem_seg_model.episode_start()
            self._fstta_episode_active = True
        return super().act(batched_obs, agent_states, steps, g_masks, l_masks)

    def close(self):
        if self._fstta_episode_active:
            self.sem_seg_model.episode_end()
            self._fstta_episode_active = False
        self.sem_seg_model.write_fstta_diagnostics()
        super().close()

