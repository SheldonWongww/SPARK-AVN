# cavn/model/spark_avn/rollout_storage_spark_avn.py
import torch
from collections import defaultdict

from cavn.model.rollout_storage_multi_len import (
    RolloutStorageMultiLen,
    ExternalMemoryMultiLen,
)


class RolloutStorageSparkAVN(RolloutStorageMultiLen):
    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        use_external_memory,
        external_memory_size,
        external_memory_capacity,
        external_memory_dim,
        router_pool_size,
        num_recurrent_layers=1,
    ):
        super().__init__(
            num_steps=num_steps,
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
            recurrent_hidden_state_size=recurrent_hidden_state_size,
            use_external_memory=use_external_memory,
            external_memory_size=external_memory_size,
            external_memory_capacity=external_memory_capacity,
            external_memory_dim=external_memory_dim,
            num_recurrent_layers=num_recurrent_layers,
        )
        self.router_alpha = torch.zeros(
            num_steps,
            num_envs,
            router_pool_size,
            dtype=torch.float32,
        )

    def to(self, device):
        super().to(device)
        self.router_alpha = self.router_alpha.to(device)
        return self

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        not_done_masks,
        em_features,
        router_alpha,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(observations[sensor])
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(not_done_masks)
        self.router_alpha[self.step].copy_(router_alpha)
        if self.use_external_memory:
            self.em.insert(em_features, not_done_masks, self.step + 1)
            self.em_masks[self.step + 1].copy_(self.em.masks)

        self.step = self.step + 1

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            router_alpha_batch = []

            if self.use_external_memory:
                em_store_batch = []
                em_masks_batch = []
            else:
                em_store_batch = None
                em_masks_batch = None

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )
                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )
                adv_targ.append(advantages[: self.step, ind])
                router_alpha_batch.append(self.router_alpha[: self.step, ind])

                if self.use_external_memory:
                    temp_memory_list = []
                    temp_memory_masks_list = []
                    for step in range(self.step):
                        temp_memory = self.external_memory(step).unsqueeze(1)
                        temp_memory_list.append(temp_memory)
                        temp_masks = self.external_memory_masks(step).unsqueeze(0)
                        temp_memory_masks_list.append(temp_masks)

                    em_store = torch.cat(temp_memory_list, dim=1)[:, :, ind]
                    em_masks = torch.cat(temp_memory_masks_list, dim=0)[:, ind]
                    em_store_batch.append(em_store)
                    em_masks_batch.append(em_masks)

            T, N = self.step, num_envs_per_batch
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(observations_batch[sensor], 1)

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)
            router_alpha_batch = torch.stack(router_alpha_batch, 1)

            if self.use_external_memory:
                em_store_batch = torch.stack(em_store_batch, 2)
                em_masks_batch = torch.stack(em_masks_batch, 1)

            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )
            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)
            router_alpha_batch = self._flatten_helper(T, N, router_alpha_batch)

            if self.use_external_memory:
                em_store_batch = em_store_batch.view(-1, T * N, self.em_dim)
                em_masks_batch = self._flatten_helper(T, N, em_masks_batch)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                em_store_batch,
                em_masks_batch,
                router_alpha_batch,
            )
