import copy
import logging
from typing import Dict, Optional, List

import torch
import torch.nn as nn

from cavn.model.policy import AudioNavMSMTPolicyWithGD
from cavn.model.spark_avn.fuse_layers import (
    FuseLinear,
    FuseConv2d,
    FuseConv1d,
    FuseLayerNorm,
)
from cavn.model.spark_avn.fuse_attention import (
    FuseMultiheadAttention,
    FusePoolingAttention,
)
from cavn.model.spark_avn.router import VisualEnvRouter
from cavn.model.spark_avn.knowledge_pool import AdaptiveKnowledgePool
from cavn.model.spark_avn.prototype_bank import (
    PROTOTYPE_BANK_SIZE,
    MAX_KMEANS_ITERS,
    TaskRoutingSummary,
    routing_compatibility,
)
from cavn.model.transformer_decoder import PoolingAttention


class AudioNavMSMTPolicy_SparkAVN(AudioNavMSMTPolicyWithGD):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_size=128,
        actor_critic_pretrained_path: str = "-1",
        pool_size: int = 5,
        router_ema_momentum: float = 0.95,
        router_temperature: float = 1.0,
        eval_routing_mode: str = "top1",
        prototype_bank_size: int = PROTOTYPE_BANK_SIZE,
        eval_warmup_steps: int = 10,
        max_kmeans_iters: int = MAX_KMEANS_ITERS,
        merge_w_value: float = 0.5,
        merge_w_hhi: float = 0.25,
        **kwargs,
    ):
        requested_use_pretrained = kwargs.get("use_pretrained", False)
        audio_pretrained_path = kwargs.get("audio_pretrained_path", "")
        visual_pretrained_path = kwargs.get("visual_pretrained_path", "")
        kwargs["use_pretrained"] = False

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            **kwargs,
        )

        if actor_critic_pretrained_path not in ["-1", "", None]:
            ckpt_dict = torch.load(
                actor_critic_pretrained_path, map_location="cpu", weights_only=False
            )
            state_dict = ckpt_dict.get("state_dict", ckpt_dict)
            mapped_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("actor_critic."):
                    mapped_state_dict[key[len("actor_critic."):]] = value
                else:
                    mapped_state_dict[key] = value
            missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)
            if len(missing) > 0:
                logging.info(
                    "[SPARK-AVN] missing keys while loading actor_critic checkpoint: %d",
                    len(missing),
                )
            if len(unexpected) > 0:
                logging.info(
                    "[SPARK-AVN] unexpected keys while loading actor_critic checkpoint: %d",
                    len(unexpected),
                )
        elif requested_use_pretrained:
            try:
                self.net.pretrained_initialization(
                    audio_pretrained_path,
                    visual_pretrained_path,
                )
            except Exception as error:
                logging.warning(
                    "[SPARK-AVN] fallback pretrained_initialization failed: %s", error
                )

        self._freeze_all_base_parameters()

        self.pool_size = pool_size
        self.freeze_goal_action_pose = True
        self.fuse_layer_norm = True
        self.fuse_mha = True
        self.fuse_mha_out_proj = True
        self.fuse_conv1d = True
        self.mha_batched_alpha_mode = "loop"
        self.eval_routing_mode = str(eval_routing_mode).lower()
        self._task_active = False

        self.g_task = nn.Parameter(torch.ones(1), requires_grad=True)

        self.anchor_visual_encoder = copy.deepcopy(self.net.visual_encoder)
        for param in self.anchor_visual_encoder.parameters():
            param.requires_grad = False
        self.anchor_visual_encoder.eval()

        visual_dim = self.net.visual_encoder.feature_dims
        self.prototype_dim = visual_dim
        self.prototype_bank_size = int(prototype_bank_size)
        self.eval_warmup_steps = int(eval_warmup_steps)
        self.max_kmeans_iters = int(max_kmeans_iters)
        self.merge_w_value = float(merge_w_value)
        self.merge_w_hhi = float(merge_w_hhi)
        self.router_temperature = max(float(router_temperature), 1e-6)
        self.router = VisualEnvRouter(
            feature_dim=visual_dim,
            pool_size=pool_size,
            prototype_bank_size=self.prototype_bank_size,
            temperature=router_temperature,
            ema_momentum=router_ema_momentum,
            eval_warmup_steps=self.eval_warmup_steps,
        )

        self.register_buffer(
            "pool_prototypes",
            torch.zeros(pool_size, self.prototype_bank_size, visual_dim),
        )
        self.register_buffer(
            "pool_prototype_counts",
            torch.zeros(pool_size, dtype=torch.long),
        )
        self.register_buffer("pool_counters", torch.zeros(pool_size, dtype=torch.long))
        self.register_buffer("pool_active_count", torch.zeros(1, dtype=torch.long))

        if self.freeze_goal_action_pose:
            self._freeze_invariant_modules()

        self.fuse_layers: Dict[str, nn.Module] = {}
        self._build_fused_adaptation_modules()
        self._set_runtime_router(alpha=None, use_current_task=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor_visual_encoder.eval()
        if self.freeze_goal_action_pose:
            if hasattr(self.net, "action_encoder") and self.net._use_action_encoding:
                self.net.action_encoder.eval()
            if hasattr(self.net, "goal_descriptor") and self.net._use_goal_descriptor:
                self.net.goal_descriptor.eval()
                if hasattr(self.net, "goal_downsample"):
                    self.net.goal_downsample.eval()
            if hasattr(self.net, "smt_state_encoder") and hasattr(self.net.smt_state_encoder, "pose_encoder"):
                self.net.smt_state_encoder.pose_encoder.eval()
        return self

    def _freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _freeze_all_base_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def _freeze_invariant_modules(self):
        if hasattr(self.net, "action_encoder") and self.net._use_action_encoding:
            self._freeze_module(self.net.action_encoder)
        if hasattr(self.net, "goal_descriptor") and self.net._use_goal_descriptor:
            self._freeze_module(self.net.goal_descriptor)
            if hasattr(self.net, "goal_downsample"):
                self._freeze_module(self.net.goal_downsample)
        if hasattr(self.net, "smt_state_encoder") and hasattr(self.net.smt_state_encoder, "pose_encoder"):
            self._freeze_module(self.net.smt_state_encoder.pose_encoder)

    def _register_attention_fuse_layers(self, full_name: str, module: nn.Module):
        if hasattr(module, "in_proj") and isinstance(module.in_proj, FuseLinear):
            self.fuse_layers[f"{full_name}.in_proj"] = module.in_proj
        if hasattr(module, "out_proj") and isinstance(module.out_proj, FuseLinear):
            self.fuse_layers[f"{full_name}.out_proj"] = module.out_proj

    def _replace_modules_with_fused_layers(
        self,
        module: nn.Module,
        prefix: str,
        excluded_prefixes: Optional[List[str]] = None,
    ):
        excluded_prefixes = excluded_prefixes or []
        for child_name, child_module in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if any(full_name.startswith(ex) for ex in excluded_prefixes):
                continue
            if (
                not self.fuse_mha_out_proj
                and child_name == "out_proj"
                and isinstance(module, (FuseMultiheadAttention, FusePoolingAttention))
            ):
                continue

            current_child = child_module
            replaced = False

            if self.fuse_mha and isinstance(child_module, PoolingAttention):
                fused_attn = FusePoolingAttention.from_pooling_attention(
                    child_module,
                    pool_size=self.pool_size,
                    fuse_out_proj=self.fuse_mha_out_proj,
                    batched_alpha_mode=self.mha_batched_alpha_mode,
                )
                setattr(module, child_name, fused_attn)
                current_child = fused_attn
                self._register_attention_fuse_layers(full_name, fused_attn)
                replaced = True
            elif self.fuse_mha and isinstance(child_module, nn.MultiheadAttention):
                fused_attn = FuseMultiheadAttention.from_multihead_attention(
                    child_module,
                    pool_size=self.pool_size,
                    fuse_out_proj=self.fuse_mha_out_proj,
                    batched_alpha_mode=self.mha_batched_alpha_mode,
                )
                setattr(module, child_name, fused_attn)
                current_child = fused_attn
                self._register_attention_fuse_layers(full_name, fused_attn)
                replaced = True
            elif isinstance(child_module, nn.Linear):
                fused = FuseLinear.from_linear(child_module, self.pool_size)
                setattr(module, child_name, fused)
                self.fuse_layers[full_name] = fused
                current_child = fused
                replaced = True
            elif isinstance(child_module, nn.Conv2d):
                fused = FuseConv2d.from_conv2d(child_module, self.pool_size)
                setattr(module, child_name, fused)
                self.fuse_layers[full_name] = fused
                current_child = fused
                replaced = True
            elif self.fuse_conv1d and isinstance(child_module, nn.Conv1d):
                fused = FuseConv1d.from_conv1d(child_module, self.pool_size)
                setattr(module, child_name, fused)
                self.fuse_layers[full_name] = fused
                current_child = fused
                replaced = True
            elif self.fuse_layer_norm and isinstance(child_module, nn.LayerNorm):
                if child_module.elementwise_affine:
                    fused = FuseLayerNorm.from_layer_norm(child_module, self.pool_size)
                    setattr(module, child_name, fused)
                    self.fuse_layers[full_name] = fused
                    current_child = fused
                    replaced = True

            if replaced:
                self._replace_modules_with_fused_layers(
                    current_child,
                    full_name,
                    excluded_prefixes=excluded_prefixes,
                )
                continue

            self._replace_modules_with_fused_layers(
                current_child,
                full_name,
                excluded_prefixes=excluded_prefixes,
            )

    def _build_fused_adaptation_modules(self):
        # These are the modules that remain plastic in SPARK-AVN.
        self._replace_modules_with_fused_layers(self.net.visual_encoder, "net.visual_encoder")

        self._replace_modules_with_fused_layers(self.net.goal_encoder, "net.goal_encoder")
        if hasattr(self.net, "downsample"):
            if isinstance(self.net.downsample, nn.Linear):
                fused = FuseLinear.from_linear(self.net.downsample, self.pool_size)
                self.net.downsample = fused
                self.fuse_layers["net.downsample"] = fused
            else:
                self._replace_modules_with_fused_layers(self.net.downsample, "net.downsample")

        excluded = []
        if self.freeze_goal_action_pose:
            excluded = ["net.smt_state_encoder.pose_encoder"]
        self._replace_modules_with_fused_layers(
            self.net.smt_state_encoder,
            "net.smt_state_encoder",
            excluded_prefixes=excluded,
        )

        if isinstance(self.action_distribution.linear, nn.Linear):
            fused_actor = FuseLinear.from_linear(
                self.action_distribution.linear, self.pool_size
            )
            self.action_distribution.linear = fused_actor
            self.fuse_layers["action_distribution.linear"] = fused_actor
        if isinstance(self.critic.fc, nn.Linear):
            fused_critic = FuseLinear.from_linear(self.critic.fc, self.pool_size)
            self.critic.fc = fused_critic
            self.fuse_layers["critic.fc"] = fused_critic

    def _set_runtime_router(
        self, alpha: Optional[torch.Tensor], use_current_task: bool
    ):
        active_count = int(self.pool_active_count.item())
        runtime_alpha = alpha if active_count > 0 else None
        for layer in self.fuse_layers.values():
            layer.set_runtime_context(
                alpha=runtime_alpha,
                use_current_task=use_current_task,
                g_task=self.g_task,
                active_count=active_count,
            )

    @property
    def active_pool_count(self) -> int:
        return int(self.pool_active_count.item())

    def get_anchor_features(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            features = self.anchor_visual_encoder(observations)
        return features

    def _compute_router_alpha(
        self,
        observations: Dict[str, torch.Tensor],
        masks: Optional[torch.Tensor],
        is_rollout: bool,
        use_current_task: bool,
        router_alpha_override: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = next(iter(observations.values())).shape[0]
        device = next(self.parameters()).device
        dtype = self.pool_prototypes.dtype

        if router_alpha_override is not None:
            alpha = router_alpha_override.to(device=device, dtype=dtype)
            if (not is_rollout) and alpha.dim() == 2:
                alpha = alpha.mean(dim=0)
            return alpha

        if self.active_pool_count <= 0:
            return torch.zeros(batch_size, self.pool_size, device=device, dtype=dtype)

        anchor_features = self.get_anchor_features(observations)
        alpha = self.router(
            anchor_features=anchor_features,
            pool_prototypes=self.pool_prototypes,
            pool_prototype_counts=self.pool_prototype_counts,
            active_count=self.active_pool_count,
            not_done_masks=masks,
            is_rollout=is_rollout,
            use_current_task=use_current_task,
        )
        if not use_current_task:
            if self.eval_routing_mode == "dense":
                pass
            elif self.eval_routing_mode == "top1":
                alpha = self._top1_alpha(alpha, self.active_pool_count)
            else:
                raise ValueError(
                    f"Unsupported eval_routing_mode: {self.eval_routing_mode}. "
                    "Expected one of: dense, top1"
                )
        return alpha

    def _top1_alpha(self, alpha: torch.Tensor, active_count: int) -> torch.Tensor:
        input_is_vector = alpha.dim() == 1
        if input_is_vector:
            alpha = alpha.unsqueeze(0)

        out = torch.zeros_like(alpha)
        if active_count <= 0:
            return out.squeeze(0) if input_is_vector else out

        active_count = min(active_count, self.pool_size)
        for b_idx in range(alpha.size(0)):
            probs = alpha[b_idx, :active_count]
            if probs.numel() == 0 or probs.sum() <= 0:
                continue

            top1_idx = int(torch.argmax(probs).item())
            out[b_idx, top1_idx] = 1.0

        return out.squeeze(0) if input_is_vector else out

    def start_new_task(self, num_envs: int):
        self._task_active = True
        with torch.no_grad():
            self.g_task.fill_(1.0)
        for layer in self.fuse_layers.values():
            layer.reset_current_vector()
        self.router.ensure_num_envs(
            num_envs,
            device=next(self.parameters()).device,
            dtype=self.pool_prototypes.dtype,
        )
        self.router.reset_all()

    def disable_current_task(self):
        self._task_active = False
        for layer in self.fuse_layers.values():
            layer.reset_current_vector()

    def pause_env_states(self, envs_to_pause):
        self.router.pause_envs(envs_to_pause)

    def _get_current_value_entry(self):
        entry = {}
        for layer_name, layer in self.fuse_layers.items():
            entry[layer_name] = layer.get_current_vector()
        return entry

    def _compute_commit_alpha(
        self,
        summary: TaskRoutingSummary,
        active_count: int,
    ) -> Optional[torch.Tensor]:
        if active_count <= 0:
            return None

        scores = []
        for slot in range(active_count):
            score = routing_compatibility(
                summary.prototype_bank,
                summary.prototype_count,
                self.pool_prototypes[slot],
                int(self.pool_prototype_counts[slot].item()),
            )
            scores.append(score)

        score_tensor = torch.tensor(
            scores,
            device=self.pool_prototypes.device,
            dtype=self.pool_prototypes.dtype,
        )
        return torch.softmax(score_tensor / self.router_temperature, dim=0)

    def _get_committed_value_entry(self, summary: TaskRoutingSummary, active_count: int):
        alpha_global = self._compute_commit_alpha(summary, active_count)
        g_scalar = float(self.g_task.item())
        entry = {}
        for layer_name, layer in self.fuse_layers.items():
            entry[layer_name] = layer.get_committed_vector(
                alpha_global=alpha_global,
                g_scalar=g_scalar,
                active_count=active_count,
            )
        return entry

    def _get_pool_value_entries(self, active_count: int):
        values = []
        for slot in range(active_count):
            entry = {}
            for layer_name, layer in self.fuse_layers.items():
                entry[layer_name] = layer.get_pool_vector(slot)
            values.append(entry)
        return values

    def _set_pool_value_entries(self, values, active_count: int):
        for layer in self.fuse_layers.values():
            for slot in range(self.pool_size):
                layer.clear_pool_slot(slot)

        for slot in range(active_count):
            value_entry = values[slot]
            for layer_name, layer in self.fuse_layers.items():
                layer.set_pool_vector(
                    slot=slot,
                    weight=value_entry[layer_name]["weight"],
                    bias=value_entry[layer_name]["bias"],
                )

    def _get_pool_prototype_entries(self, active_count: int):
        prototype_banks = []
        prototype_counts = []
        for slot in range(active_count):
            prototype_banks.append(self.pool_prototypes[slot].detach().clone())
            prototype_counts.append(int(self.pool_prototype_counts[slot].item()))
        return prototype_banks, prototype_counts

    def _set_pool_prototype_entries(
        self,
        prototype_banks,
        prototype_counts,
        active_count: int,
    ):
        with torch.no_grad():
            self.pool_prototypes.zero_()
            self.pool_prototype_counts.zero_()
            for slot in range(active_count):
                self.pool_prototypes[slot].copy_(
                    prototype_banks[slot].to(
                        device=self.pool_prototypes.device,
                        dtype=self.pool_prototypes.dtype,
                    )
                )
                self.pool_prototype_counts[slot] = int(prototype_counts[slot])

    def commit_current_task_to_pool(self, summary: TaskRoutingSummary):
        summary = summary.to(
            device=self.pool_prototypes.device,
            dtype=self.pool_prototypes.dtype,
        )
        if int(summary.prototype_count) <= 0:
            raise ValueError("TaskRoutingSummary must contain at least one valid prototype")

        active_count = self.active_pool_count
        existing_prototype_banks, existing_prototype_counts = self._get_pool_prototype_entries(active_count)
        existing_counters = [
            int(self.pool_counters[idx].item()) for idx in range(active_count)
        ]
        existing_values = self._get_pool_value_entries(active_count)
        committed_value = self._get_committed_value_entry(summary, active_count)

        pool = AdaptiveKnowledgePool(
            self.pool_size,
            prototype_bank_size=self.prototype_bank_size,
            max_kmeans_iters=self.max_kmeans_iters,
            merge_w_value=self.merge_w_value,
            merge_w_hhi=self.merge_w_hhi,
        )
        pool.load(
            existing_prototype_banks,
            existing_prototype_counts,
            existing_counters,
            existing_values,
        )
        pool.add(
            summary.prototype_bank,
            summary.prototype_count,
            committed_value,
            counter=1,
        )
        merged_prototype_banks, merged_prototype_counts, merged_counters, merged_values = pool.state()

        with torch.no_grad():
            self.pool_counters.zero_()
            for idx, count in enumerate(merged_counters):
                self.pool_counters[idx] = int(count)
            self.pool_active_count[0] = len(merged_counters)

        self._set_pool_prototype_entries(
            merged_prototype_banks,
            merged_prototype_counts,
            len(merged_prototype_banks),
        )
        self._set_pool_value_entries(merged_values, len(merged_values))
        self.disable_current_task()

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        ext_memory,
        ext_memory_masks,
        deterministic=False,
        is_rollout: bool = True,
        use_current_task: bool = False,
        router_alpha_override: Optional[torch.Tensor] = None,
    ):
        alpha = self._compute_router_alpha(
            observations=observations,
            masks=masks,
            is_rollout=is_rollout,
            use_current_task=use_current_task,
            router_alpha_override=router_alpha_override,
        )
        self._set_runtime_router(
            alpha=alpha,
            use_current_task=bool(use_current_task and self._task_active),
        )

        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            ext_memory,
            ext_memory_masks,
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_probs(action)

        return (
            value,
            action,
            action_log_probs,
            rnn_hidden_states,
            ext_memory_feats,
            alpha.detach(),
        )

    def get_value(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        ext_memory,
        ext_memory_masks,
        is_rollout: bool = False,
        use_current_task: bool = False,
        router_alpha_override: Optional[torch.Tensor] = None,
    ):
        alpha = self._compute_router_alpha(
            observations=observations,
            masks=masks,
            is_rollout=is_rollout,
            use_current_task=use_current_task,
            router_alpha_override=router_alpha_override,
        )
        self._set_runtime_router(
            alpha=alpha,
            use_current_task=bool(use_current_task and self._task_active),
        )
        features, _, _ = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            ext_memory,
            ext_memory_masks,
        )
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        ext_memory,
        ext_memory_masks,
        is_rollout: bool = False,
        use_current_task: bool = False,
        router_alpha_override: Optional[torch.Tensor] = None,
    ):
        alpha = self._compute_router_alpha(
            observations=observations,
            masks=masks,
            is_rollout=is_rollout,
            use_current_task=use_current_task,
            router_alpha_override=router_alpha_override,
        )
        self._set_runtime_router(
            alpha=alpha,
            use_current_task=bool(use_current_task and self._task_active),
        )

        features, rnn_hidden_states, ext_memory_feats = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            ext_memory,
            ext_memory_masks,
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            ext_memory_feats,
        )
