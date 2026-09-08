"""Episode-aware PONI GlobalAgent shim for Tent."""

from global_agent import GlobalAgent


class TentGlobalAgent(GlobalAgent):
    """Forward Habitat episode boundaries to the RedNet Tent adapter.

    NavTTA's sequential adapters assume one environment per process.  This is
    also PONI's default evaluation setting and the launcher parallelizes whole
    validation partitions as separate processes.
    """

    def __init__(self, cfg, device):
        if int(cfg.NUM_ENVIRONMENTS) != 1:
            raise ValueError(
                "PONI + Tent requires NUM_ENVIRONMENTS=1 so adaptation follows "
                "one unambiguous episode stream"
            )
        super().__init__(cfg, device)
        self._tent_episode_active = False

    def act(self, batched_obs, agent_states, steps, g_masks, l_masks):
        # TransferEvaluator sets l_masks=False on the first step after reset.
        # Close the previous episode before starting the next one.
        is_episode_start = not bool(l_masks.reshape(-1)[0].item())
        if is_episode_start:
            if self._tent_episode_active:
                self.sem_seg_model.episode_end()
            self.sem_seg_model.episode_start()
            self._tent_episode_active = True
        return super().act(
            batched_obs, agent_states, steps, g_masks, l_masks
        )

    def close(self):
        if self._tent_episode_active:
            self.sem_seg_model.episode_end()
            self._tent_episode_active = False
        self.sem_seg_model.write_tent_diagnostics()
        super().close()

