# cavn/engin/base_ddppo_trainer.py

import contextlib
import os
import random
import gc
import time
import logging
import json
import torch
import torch.distributed as distrib
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from collections import defaultdict, deque
from gym import spaces
from tqdm import tqdm
from numpy.linalg import norm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.savi.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from ss_baselines.savi.ddppo.algo.ddppo import DDPPO
from ss_baselines.savi.ppo.ppo_trainer import PPOTrainer

from cavn.common.utils import (
    batch_obs, linear_decay, generate_video, NpEncoder,
    observations_to_image, plot_top_down_map, resize_observation
)
from cavn.model.policy import AudioNavMSMTPolicyWithGD
from cavn.model.rollout_storage_multi_len import RolloutStorageMultiLen, ExternalMemoryMultiLen


@baseline_registry.register_trainer(name="base_ddppo")
class BaseDDPPOTrainer(PPOTrainer):
    """Base DDPPO trainer without continual learning capabilities."""
    
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None):
        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            config = interrupted_state["config"]
        super().__init__(config)
        self.current_domain_id = -1
        self._static_smt_encoder = False

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        logger.add_filehandler(self.config.LOG_FILE)
        
        action_space = self.envs.action_spaces[0]
        self.action_space = action_space

        if ppo_cfg.policy_type == 'msmt':
            smt_cfg = ppo_cfg.SCENE_MEMORY_TRANSFORMER
            belief_cfg = ppo_cfg.BELIEF_PREDICTOR
            seld_cfg = ppo_cfg.SELD_ENCODER
            
            self.actor_critic = AudioNavMSMTPolicyWithGD(
                observation_space=self.envs.observation_spaces[0],
                action_space=self.envs.action_spaces[0],
                hidden_size=smt_cfg.hidden_size,
                nhead=smt_cfg.nhead,
                num_encoder_layers=smt_cfg.num_encoder_layers,
                num_decoder_layers=smt_cfg.num_decoder_layers,
                dropout=smt_cfg.dropout,
                activation=smt_cfg.activation,
                use_pretrained=smt_cfg.use_pretrained,
                audio_pretrained_path=smt_cfg.audio_pretrained_path,
                visual_pretrained_path=smt_cfg.visual_pretrained_path,
                seld_pretrained_path=smt_cfg.seld_pretrained_path,
                pretraining=smt_cfg.pretraining,
                use_belief_encoding=smt_cfg.use_belief_encoding,
                use_belief_as_goal=ppo_cfg.use_belief_predictor,
                use_label_belief=belief_cfg.use_label_belief,
                use_location_belief=belief_cfg.use_location_belief,
                normalize_category_distribution=belief_cfg.normalize_category_distribution,
                use_category_input=True,
                use_goal_descriptor=smt_cfg.use_goal_descriptor,
                norm_first=ppo_cfg.norm_first,
                decoder_type=smt_cfg.decoder_type,
                gd_encoder_type=seld_cfg.gd_encoder_type,
                use_downsample=seld_cfg.use_downsample,
            )

            if smt_cfg.freeze_encoders:
                self._static_smt_encoder = True
                self.actor_critic.net.freeze_encoders()
        else:
            raise ValueError(f'Policy type {ppo_cfg.policy_type} is not defined!')

        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

        if smt_cfg.actor_critic_pretrained_path not in ["-1", ""]:
            ckpt_dict = self.load_checkpoint(smt_cfg.actor_critic_pretrained_path, map_location="cpu", weights_only=False)
            self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
            self.actor_critic = self.agent.actor_critic

    def _collect_rollout_step(self, rollouts, current_episode_reward, running_episode_stats):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            external_memory = None
            external_memory_masks = None
            if self.config.RL.PPO.use_external_memory:
                external_memory = rollouts.external_memory(rollouts.step)
                external_memory_masks = rollouts.external_memory_masks(rollouts.step)
            
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                external_memory_features
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks,
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        
        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=current_episode_reward.device
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )
            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards.to(device=self.device),
            masks.to(device=self.device),
            external_memory_features,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            external_memory = None
            external_memory_masks = None
            if ppo_cfg.use_external_memory:
                external_memory = rollouts.external_memory(rollouts.step)
                external_memory_masks = rollouts.external_memory_masks(rollouts.step)

            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks,
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
        rollouts.after_update()

        self.agent.optimizer.zero_grad(set_to_none=True)

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def save_checkpoint(self, file_name: str, extra_state=None) -> None:
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        
        checkpoint["optim_state"] = self.agent.optimizer.state_dict()
        
        if hasattr(self, 'lr_scheduler'):
            checkpoint["lr_sched_state"] = self.lr_scheduler.state_dict()
        
        if hasattr(self, "cl_method") and self.cl_method is not None:
            checkpoint["cl_method_type"] = str(self.config.CL_METHOD.TYPE)
            checkpoint["cl_method_state"] = self.cl_method.state_dict()
        
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, map_location="cpu", weights_only=False) -> Dict:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        domain_id: Optional[int] = None,
        progress_desc: Optional[str] = None,
    ) -> Dict:
        """Evaluate a checkpoint on validation episodes."""
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu", weights_only=False)
        
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()
            
        ppo_cfg = config.RL.PPO
        
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        
        # Override CONTENT_SCENES if domain_id is specified
        if domain_id is not None:
            domain_scenes = self._get_domain_scenes(domain_id)
            config.TASK_CONFIG.DATASET.CONTENT_SCENES = domain_scenes
        
        # CRITICAL: Disable pretrained weight loading during evaluation
        # We only want to load the checkpoint weights, not the pretrained weights
        original_pretrained_path = None
        if hasattr(ppo_cfg, 'SCENE_MEMORY_TRANSFORMER'):
            if hasattr(ppo_cfg.SCENE_MEMORY_TRANSFORMER, 'actor_critic_pretrained_path'):
                original_pretrained_path = ppo_cfg.SCENE_MEMORY_TRANSFORMER.actor_critic_pretrained_path
                ppo_cfg.SCENE_MEMORY_TRANSFORMER.actor_critic_pretrained_path = "-1"
            
        config.freeze()

        logger.info(f"Evaluating on domain {domain_id if domain_id is not None else 'all'}")
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        
        observation_space = self.envs.observation_spaces[0]
        self._setup_actor_critic_agent(ppo_cfg, observation_space)
        
        # Load checkpoint weights (this is the trained model, not pretrained)
        self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
        self.actor_critic = self.agent.actor_critic
        
        # Restore original config (for integrity)
        if original_pretrained_path is not None:
            config.defrost()
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.actor_critic_pretrained_path = original_pretrained_path
            config.freeze()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            1,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )

        if ppo_cfg.use_external_memory:
            test_em = ExternalMemoryMultiLen(
                self.config.NUM_PROCESSES,
                ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                self.actor_critic.net.memory_dim,
                is_mapping=False,
            )
            test_em.to(self.device)
        else:
            test_em = None

        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()

        self.actor_critic.eval()
        
        t = tqdm(total=self.config.TEST_EPISODE_COUNT, desc=progress_desc)
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            if ppo_cfg.use_external_memory:
                em_memory = test_em.memory[:, 0]
                if test_em.idx >= test_em.capacity:
                    em_memory = em_memory[test_em.idx-test_em.capacity:test_em.idx]
                else:
                    em_memory = torch.cat([em_memory[test_em.idx-test_em.capacity:], em_memory[:test_em.idx]], dim=0)

                em_masks = test_em.masks
                if test_em.idx >= test_em.capacity:
                    em_masks = em_masks[:, test_em.idx-test_em.capacity:test_em.idx]
                else:
                    em_masks = torch.cat([em_masks[:, test_em.idx-test_em.capacity:], em_masks[:, :test_em.idx]], dim=1)
            
            with torch.no_grad():
                act_outputs = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    em_memory if ppo_cfg.use_external_memory else None,
                    em_masks if ppo_cfg.use_external_memory else None,
                    deterministic=False
                )
                _, actions, _, test_recurrent_hidden_states, test_em_features = act_outputs[:5]

                prev_actions.copy_(actions)

            actions = [a[0].item() for a in actions]
            outputs = self.envs.step(actions)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            
            if ppo_cfg.use_external_memory:
                test_em.insert(test_em_features, not_done_masks)

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            
            for i in range(self.envs.num_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["success"] = infos[i]["success"]
                    episode_stats["spl"] = infos[i]["spl"]
                    episode_stats["distance_to_goal"] = infos[i]["distance_to_goal"]
                    episode_stats["normalized_distance_to_goal"] = infos[i]["normalized_distance_to_goal"]
                    
                    # Handle potentially missing metrics gracefully
                    # Some metrics may use abbreviated keys (e.g., "na" instead of "num_action")
                    episode_stats["num_action"] = infos[i].get("num_action", infos[i].get("na", 0))
                    episode_stats["success_weighted_by_num_action"] = infos[i].get(
                        "success_weighted_by_num_action", 
                        infos[i].get("sna", 0)
                    )
                    episode_stats["success_when_silent"] = infos[i].get(
                        "success_when_silent",
                        infos[i].get("sws", 0)
                    )
                    
                    current_episode_reward[i] = 0
                    
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    t.update()

            # Initialize rgb_frames list with None for each environment
            # Must create list with current num_envs BEFORE any pausing occurs
            rgb_frames = [None] * self.envs.num_envs

            if len(envs_to_pause) > 0 and hasattr(self.actor_critic, "pause_env_states"):
                self.actor_critic.pause_env_states(envs_to_pause)
            
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                test_em,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                test_em,
                None  # descriptor_pred_gt: should be None, not empty list
            )

        aggregated_stats = dict()
        for stat_key in ["reward", "success", "spl", "distance_to_goal", 
                        "normalized_distance_to_goal", "num_action", 
                        "success_weighted_by_num_action", "success_when_silent"]:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )
        
        num_episodes = len(stats_episodes)
        
        for stat_key, stat_val in aggregated_stats.items():
            logger.info(f"Average {stat_key}: {stat_val:.4f} over {num_episodes} episodes")

        # Write evaluation metrics to TensorBoard (following ENMuS convention)
        # Use checkpoint_index as the x-axis value
        if writer is not None:
            # Write each metric under "val/" prefix for validation metrics
            writer.add_scalar("val/reward", aggregated_stats["reward"], checkpoint_index)
            writer.add_scalar("val/success", aggregated_stats["success"], checkpoint_index)
            writer.add_scalar("val/spl", aggregated_stats["spl"], checkpoint_index)
            writer.add_scalar("val/distance_to_goal", aggregated_stats["distance_to_goal"], checkpoint_index)
            writer.add_scalar("val/normalized_distance_to_goal", aggregated_stats["normalized_distance_to_goal"], checkpoint_index)
            writer.add_scalar("val/num_action", aggregated_stats["num_action"], checkpoint_index)
            writer.add_scalar("val/success_weighted_by_num_action", aggregated_stats["success_weighted_by_num_action"], checkpoint_index)
            writer.add_scalar("val/success_when_silent", aggregated_stats["success_when_silent"], checkpoint_index)
            
            logger.info(f"Evaluation metrics written to TensorBoard under 'val/' prefix")

        self.envs.close()

        return aggregated_stats

    def _get_domain_scenes(self, domain_id: int):
        """Get the scene list for a specific domain."""
        # This should match the domain definitions in your data preparation
        domain_scenes_mapping = {
            0: ['q9vSo1VnCiC'],
            1: ['ac26ZMwG7aT'],
            2: ['PuKPg4mmafe'],
            3: ['r47D5H71a5s'],
            4: ['jh4fc5c5qoQ'],
            5: ['S9hNv5qa7GM'],
            6: ['QUCTc6BB5sX'],
            7: ['VLzqgDo317F'],
            8: ['5ZKStnWn8Zo'],
            9: ['EDJbREhghzL'],
            10: ['qoiz87JEwZ2'],
            11: ['VVfe2KiqLaN'],
            12: ['vyrNrziPKCB'],
            13: ['b8cTxDM8gDG'],
            14: ['e9zR4mvMWw7'],
            15: ['kEZ7cmS4wCh'],
            16: ['V2XKFyX4ASd'],
            17: ['gTV8FGcVJC9'],
            18: ['oLBMNvg9in8'],
            19: ['gYvKGZ5eRqb']
        }
        return domain_scenes_mapping.get(domain_id, [])
    
    def train(self) -> None:
        """Main training loop for standard (non-continual) DDPPO training."""
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if not distrib.is_initialized() and self.world_size > 1:
            distrib.init_process_group(
                backend=self.config.RL.DDPPO.distrib_backend, init_method="env://"
            )
        
        if self.world_size > 1:
            add_signal_handlers()

        # Setup distributed synchronization directory
        if self.world_size > 1:
            tmp_store_dir = os.path.join(self.config.TENSORBOARD_DIR, "ddp_sync")
            if self.world_rank == 0:
                os.makedirs(tmp_store_dir, exist_ok=True)
            distrib.barrier()

            file_store = distrib.FileStore(os.path.join(tmp_store_dir, "sync_file"), self.world_size)
            num_rollouts_done_store = distrib.PrefixStore("rollout_tracker", file_store)
            num_rollouts_done_store.set("num_done", "0")
        else:
            num_rollouts_done_store = None

        # Configure for multi-GPU setup
        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        self.config.TASK_CONFIG.SEED += (self.world_rank * self.config.NUM_PROCESSES)
        self.config.freeze()

        # Set random seeds
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Construct environments
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        
        ppo_cfg = self.config.RL.PPO
        
        # Create checkpoint folder
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER) and self.world_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        # Setup actor-critic agent
        self._setup_actor_critic_agent(ppo_cfg)
        
        # Initialize distributed training
        if self.world_size > 1:
            self.agent.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                f"agent number of trainable parameters: {sum(param.numel() for param in self.agent.parameters() if param.requires_grad)}"
            )
            logger.info(f"config: {self.config}")

        # Initialize environment
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        obs_space = self.envs.observation_spaces[0]
        memory_dim = self.actor_critic.net.memory_dim if ppo_cfg.use_external_memory else None

        rollouts = RolloutStorageMultiLen(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.action_space,
            ppo_cfg.hidden_size,
            ppo_cfg.use_external_memory,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size + ppo_cfg.num_steps,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
            memory_dim,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        batch = None
        observations = None

        # Training statistics
        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(lambda: deque(maxlen=ppo_cfg.reward_window_size))

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        # Learning rate scheduler
        self.lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        # Try to resume from checkpoint
        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(interrupted_state["optim_state"])
            self.lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=30)
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            # Main training loop
            for update in range(start_update, self.config.NUM_UPDATES):
                # Learning rate and clip decay
                if ppo_cfg.use_linear_lr_decay:
                    self.lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(update, self.config.NUM_UPDATES)

                # Handle interrupts
                if EXIT.is_set():
                    self.envs.close()
                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time, pth_time=pth_time, count_steps=count_steps,
                            count_checkpoints=count_checkpoints, start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        state_dict = dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=self.lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        )
                        save_interrupted_state(state_dict)
                    requeue_job()
                    return

                # Collect rollouts
                count_steps_delta = 0
                self.agent.eval()
                if hasattr(self, '_static_smt_encoder') and self._static_smt_encoder:
                    self.actor_critic.net.set_eval_encoders()
                
                for step in range(ppo_cfg.num_steps):
                    (delta_pth_time, delta_env_time, delta_steps) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    if self.world_size > 1:
                        if (step >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD and
                            int(num_rollouts_done_store.get("num_done")) > (self.config.RL.DDPPO.sync_frac * self.world_size)):
                            break

                if self.world_size > 1:
                    num_rollouts_done_store.add("num_done", 1)

                # Update agent
                self.agent.train()
                if hasattr(self, '_static_smt_encoder') and self._static_smt_encoder:
                    self.actor_critic.net.set_eval_encoders()

                (delta_pth_time, value_loss, action_loss, dist_entropy) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                # Collect statistics
                stats_ordering = list(sorted(running_episode_stats.keys()))
                stats = torch.stack([running_episode_stats[k] for k in stats_ordering], 0)
                if self.world_size > 1:
                    distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone().detach())

                stats = torch.tensor([value_loss, action_loss, dist_entropy, count_steps_delta], device=self.device)
                if self.world_size > 1:
                    distrib.all_reduce(stats)
                count_steps += stats[3].item()

                # Logging (only rank 0)
                if self.world_rank == 0:
                    if self.world_size > 1:
                        num_rollouts_done_store.set("num_done", "0")

                    losses = [stats[i].item() / self.world_size for i in range(3)]
                    
                    deltas = {
                        k: (v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item()
                        for k, v in window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    # Tensorboard logging
                    writer.add_scalar("Metrics/reward", deltas["reward"] / deltas["count"], count_steps)

                    metrics = {k: v / deltas["count"] for k, v in deltas.items() if k not in {"reward", "count"}}
                    if len(metrics) > 0:
                        for metric, value in metrics.items():
                            writer.add_scalar(f"Metrics/{metric}", value, count_steps)

                    writer.add_scalar("Policy/value_loss", losses[0], count_steps)
                    writer.add_scalar("Policy/policy_loss", losses[1], count_steps)
                    writer.add_scalar("Policy/entropy_loss", losses[2], count_steps)
                    writer.add_scalar('Policy/learning_rate', self.lr_scheduler.get_lr()[0], count_steps)

                    # Console logging in ENMuS format
                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        fps = count_steps / ((time.time() - t_start) + prev_time)
                        logger.info(f"update: {update}\tfps: {fps:.3f}\t")
                        logger.info(
                            f"update: {update}\tenv-time: {env_time:.3f}s\tpth-time: {pth_time:.3f}s\t"
                            f"frames: {count_steps:.1f}"
                        )
                        
                        metric_strs = []
                        for k, v in deltas.items():
                            if k != "count":
                                metric_strs.append(f"{k}: {v/deltas['count']:.3f}")
                        
                        logger.info(
                            f"Average window size: {len(window_episode_stats['count'])}  {'  '.join(metric_strs)}"
                        )

                    # Save checkpoint
                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(f"ckpt.{count_checkpoints}.pth", dict(step=count_steps))
                        count_checkpoints += 1
                
                if update % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            self.envs.close()
