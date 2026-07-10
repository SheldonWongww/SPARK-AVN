from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEnvRouter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        pool_size: int,
        prototype_bank_size: int,
        temperature: float = 1.0,
        ema_momentum: float = 0.95,
        eval_warmup_steps: int = 10,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.pool_size = pool_size
        self.prototype_bank_size = prototype_bank_size
        self.temperature = max(float(temperature), 1e-6)
        self.ema_momentum = ema_momentum
        self.eval_warmup_steps = max(int(eval_warmup_steps), 0)

        self.register_buffer(
            "rollout_ema_context",
            torch.zeros(1, feature_dim),
            persistent=False,
        )
        self.register_buffer(
            "rollout_ema_initialized",
            torch.zeros(1, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "eval_warmup_sum",
            torch.zeros(1, feature_dim),
            persistent=False,
        )
        self.register_buffer(
            "eval_warmup_counts",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "eval_step_counts",
            torch.zeros(1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "eval_ema_context",
            torch.zeros(1, feature_dim),
            persistent=False,
        )
        self.register_buffer(
            "eval_ema_initialized",
            torch.zeros(1, dtype=torch.bool),
            persistent=False,
        )

    def ensure_num_envs(self, num_envs: int, device: torch.device, dtype: torch.dtype):
        current = self.rollout_ema_context.size(0)
        if current == num_envs and self.rollout_ema_context.device == device:
            self.rollout_ema_context = self.rollout_ema_context.to(device=device, dtype=dtype)
            self.eval_warmup_sum = self.eval_warmup_sum.to(device=device, dtype=dtype)
            self.eval_ema_context = self.eval_ema_context.to(device=device, dtype=dtype)
            self.rollout_ema_initialized = self.rollout_ema_initialized.to(device=device)
            self.eval_warmup_counts = self.eval_warmup_counts.to(device=device)
            self.eval_step_counts = self.eval_step_counts.to(device=device)
            self.eval_ema_initialized = self.eval_ema_initialized.to(device=device)
            return

        self.rollout_ema_context = torch.zeros(num_envs, self.feature_dim, device=device, dtype=dtype)
        self.rollout_ema_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.eval_warmup_sum = torch.zeros(num_envs, self.feature_dim, device=device, dtype=dtype)
        self.eval_warmup_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.eval_step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.eval_ema_context = torch.zeros(num_envs, self.feature_dim, device=device, dtype=dtype)
        self.eval_ema_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset_all(self):
        self.rollout_ema_context.zero_()
        self.rollout_ema_initialized.zero_()
        self.eval_warmup_sum.zero_()
        self.eval_warmup_counts.zero_()
        self.eval_step_counts.zero_()
        self.eval_ema_context.zero_()
        self.eval_ema_initialized.zero_()

    def pause_envs(self, envs_to_pause):
        if envs_to_pause is None or len(envs_to_pause) == 0:
            return
        keep_indices = [
            idx
            for idx in range(self.rollout_ema_context.size(0))
            if idx not in set(envs_to_pause)
        ]
        if len(keep_indices) == self.rollout_ema_context.size(0):
            return
        if len(keep_indices) == 0:
            self.reset_all()
            return

        keep_tensor = torch.tensor(
            keep_indices,
            device=self.rollout_ema_context.device,
            dtype=torch.long,
        )
        self.rollout_ema_context = self.rollout_ema_context.index_select(0, keep_tensor)
        self.rollout_ema_initialized = self.rollout_ema_initialized.index_select(0, keep_tensor)
        self.eval_warmup_sum = self.eval_warmup_sum.index_select(0, keep_tensor)
        self.eval_warmup_counts = self.eval_warmup_counts.index_select(0, keep_tensor)
        self.eval_step_counts = self.eval_step_counts.index_select(0, keep_tensor)
        self.eval_ema_context = self.eval_ema_context.index_select(0, keep_tensor)
        self.eval_ema_initialized = self.eval_ema_initialized.index_select(0, keep_tensor)

    def _reset_done_envs(self, not_done_masks: torch.Tensor):
        done = not_done_masks.view(-1) <= 0.0
        if not done.any():
            return

        self.rollout_ema_context[done] = 0.0
        self.rollout_ema_initialized[done] = False
        self.eval_warmup_sum[done] = 0.0
        self.eval_warmup_counts[done] = 0
        self.eval_step_counts[done] = 0
        self.eval_ema_context[done] = 0.0
        self.eval_ema_initialized[done] = False

    def _compute_alpha_from_query(
        self,
        query: torch.Tensor,
        pool_prototypes: torch.Tensor,
        pool_prototype_counts: torch.Tensor,
        active_count: int,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        device = query.device
        dtype = query.dtype

        valid_prototypes = pool_prototypes[:active_count].to(device=device, dtype=dtype)
        valid_prototypes = F.normalize(valid_prototypes, p=2, dim=-1)
        valid_counts = pool_prototype_counts[:active_count].to(device=device)

        sims = torch.einsum("bd,skd->bsk", query, valid_prototypes)
        prototype_mask = (
            torch.arange(self.prototype_bank_size, device=device)
            .view(1, 1, -1)
            .expand(batch_size, active_count, -1)
        ) < valid_counts.view(1, active_count, 1)
        sims = sims.masked_fill(~prototype_mask, float("-inf"))

        slot_scores = sims.max(dim=-1).values
        alpha_valid = F.softmax(slot_scores / self.temperature, dim=-1)

        alpha = torch.zeros(batch_size, self.pool_size, device=device, dtype=dtype)
        alpha[:, :active_count] = alpha_valid
        return alpha

    def _forward_rollout(
        self,
        anchor_features: torch.Tensor,
        pool_prototypes: torch.Tensor,
        pool_prototype_counts: torch.Tensor,
        active_count: int,
        not_done_masks: Optional[torch.Tensor],
        is_rollout: bool,
    ) -> torch.Tensor:
        features = anchor_features.detach()
        if is_rollout:
            if not_done_masks is not None:
                self._reset_done_envs(not_done_masks.to(device=features.device))
            initialized = self.rollout_ema_initialized
            if initialized.any():
                self.rollout_ema_context[initialized] = (
                    self.ema_momentum * self.rollout_ema_context[initialized]
                    + (1.0 - self.ema_momentum) * features[initialized]
                )
            if (~initialized).any():
                self.rollout_ema_context[~initialized] = features[~initialized]
                self.rollout_ema_initialized[~initialized] = True

        context = features.clone()
        if self.rollout_ema_initialized.any():
            context[self.rollout_ema_initialized] = self.rollout_ema_context[self.rollout_ema_initialized]
        context = F.normalize(context, p=2, dim=-1)
        return self._compute_alpha_from_query(
            context,
            pool_prototypes,
            pool_prototype_counts,
            active_count,
        )

    def _forward_eval(
        self,
        anchor_features: torch.Tensor,
        pool_prototypes: torch.Tensor,
        pool_prototype_counts: torch.Tensor,
        active_count: int,
        not_done_masks: Optional[torch.Tensor],
        is_rollout: bool,
    ) -> torch.Tensor:
        device = anchor_features.device
        dtype = anchor_features.dtype
        batch_size = anchor_features.size(0)
        normalized_features = F.normalize(anchor_features.detach(), p=2, dim=-1)

        if not_done_masks is not None:
            self._reset_done_envs(not_done_masks.to(device=device))

        if self.eval_warmup_steps <= 0:
            if is_rollout:
                initialized = self.eval_ema_initialized
                if initialized.any():
                    self.eval_ema_context[initialized] = (
                        self.ema_momentum * self.eval_ema_context[initialized]
                        + (1.0 - self.ema_momentum) * normalized_features[initialized]
                    )
                if (~initialized).any():
                    self.eval_ema_context[~initialized] = normalized_features[~initialized]
                    self.eval_ema_initialized[~initialized] = True
                self.eval_step_counts += 1

            query = normalized_features.clone()
            if self.eval_ema_initialized.any():
                query[self.eval_ema_initialized] = self.eval_ema_context[self.eval_ema_initialized]
            query = F.normalize(query, p=2, dim=-1)
            return self._compute_alpha_from_query(
                query,
                pool_prototypes,
                pool_prototype_counts,
                active_count,
            )

        alpha = torch.zeros(batch_size, self.pool_size, device=device, dtype=dtype)
        step_counts = self.eval_step_counts.clone()

        warm_mask = step_counts < self.eval_warmup_steps
        query_mask = step_counts == self.eval_warmup_steps
        post_mask = step_counts > self.eval_warmup_steps

        if warm_mask.any():
            if is_rollout:
                self.eval_warmup_sum[warm_mask] += normalized_features[warm_mask]
                self.eval_warmup_counts[warm_mask] += 1
                self.eval_step_counts[warm_mask] += 1

        if query_mask.any():
            warm_counts = self.eval_warmup_counts[query_mask].clamp_min(1).to(dtype=dtype).unsqueeze(-1)
            q_warm = self.eval_warmup_sum[query_mask] / warm_counts
            q_warm = F.normalize(q_warm, p=2, dim=-1)
            alpha[query_mask] = self._compute_alpha_from_query(
                q_warm,
                pool_prototypes,
                pool_prototype_counts,
                active_count,
            )

            if is_rollout:
                e_next = (
                    self.ema_momentum * q_warm
                    + (1.0 - self.ema_momentum) * normalized_features[query_mask]
                )
                self.eval_ema_context[query_mask] = e_next
                self.eval_ema_initialized[query_mask] = True
                self.eval_step_counts[query_mask] += 1

        if post_mask.any():
            if is_rollout:
                initialized = self.eval_ema_initialized[post_mask]
                current_context = self.eval_ema_context[post_mask]
                if initialized.any():
                    current_context[initialized] = (
                        self.ema_momentum * current_context[initialized]
                        + (1.0 - self.ema_momentum) * normalized_features[post_mask][initialized]
                    )
                if (~initialized).any():
                    current_context[~initialized] = normalized_features[post_mask][~initialized]
                self.eval_ema_context[post_mask] = current_context
                self.eval_ema_initialized[post_mask] = True
                self.eval_step_counts[post_mask] += 1

            query = normalized_features[post_mask].clone()
            initialized = self.eval_ema_initialized[post_mask]
            if initialized.any():
                query[initialized] = self.eval_ema_context[post_mask][initialized]
            query = F.normalize(query, p=2, dim=-1)
            alpha[post_mask] = self._compute_alpha_from_query(
                query,
                pool_prototypes,
                pool_prototype_counts,
                active_count,
            )

        return alpha

    def forward(
        self,
        anchor_features: torch.Tensor,
        pool_prototypes: torch.Tensor,
        pool_prototype_counts: torch.Tensor,
        active_count: int,
        not_done_masks: Optional[torch.Tensor] = None,
        is_rollout: bool = True,
        use_current_task: bool = True,
    ) -> torch.Tensor:
        batch_size = anchor_features.size(0)
        device = anchor_features.device
        dtype = anchor_features.dtype
        self.ensure_num_envs(batch_size, device=device, dtype=dtype)

        if active_count <= 0:
            if (not_done_masks is not None) and is_rollout:
                self._reset_done_envs(not_done_masks.to(device=device))
            return torch.zeros(batch_size, self.pool_size, device=device, dtype=dtype)

        if use_current_task:
            return self._forward_rollout(
                anchor_features=anchor_features,
                pool_prototypes=pool_prototypes,
                pool_prototype_counts=pool_prototype_counts,
                active_count=active_count,
                not_done_masks=not_done_masks,
                is_rollout=is_rollout,
            )

        return self._forward_eval(
            anchor_features=anchor_features,
            pool_prototypes=pool_prototypes,
            pool_prototype_counts=pool_prototype_counts,
            active_count=active_count,
            not_done_masks=not_done_masks,
            is_rollout=is_rollout,
        )
