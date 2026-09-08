"""Episode-aware PONI GlobalAgent for EAM."""

from global_agent import GlobalAgent


class EAMGlobalAgent(GlobalAgent):
    def __init__(self, cfg, device):
        if int(cfg.NUM_ENVIRONMENTS) != 1:
            raise ValueError("PONI + EAM requires NUM_ENVIRONMENTS=1")
        super().__init__(cfg, device)
        self._eam_episode_active = False

    def act(self, batched_obs, agent_states, steps, g_masks, l_masks):
        if not bool(l_masks.reshape(-1)[0].item()):
            if self._eam_episode_active:
                self.sem_seg_model.episode_end()
            self.sem_seg_model.episode_start()
            self._eam_episode_active = True
        return super().act(batched_obs, agent_states, steps, g_masks, l_masks)

    def close(self):
        if self._eam_episode_active:
            self.sem_seg_model.episode_end()
            self._eam_episode_active = False
        self.sem_seg_model.write_eam_diagnostics()
        super().close()

