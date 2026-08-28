# ddppo_enmus_trainer.py
import contextlib
import os
import random
import time
import logging
import json
import re
import torch
import torch.distributed as distrib
import torch.nn as nn
import numpy as np
import traceback

from typing import Dict
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
from sen_baselines.common.utils import batch_obs, linear_decay, generate_video, NpEncoder
from ss_baselines.common.utils import observations_to_image, plot_top_down_map, resize_observation
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

from sen_baselines.enmus.ppo.msmt_policy import AudioNavMSMTPolicyWithGD
from sen_baselines.enmus.models.rollout_storage_multi_len import RolloutStorageMultiLen, ExternalMemoryMultiLen


@baseline_registry.register_trainer(name="ddppo_enmus")
class DDPPOTrainer(PPOTrainer):
    
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None):
        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            config = interrupted_state["config"]

        super().__init__(config)

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        logger.add_filehandler(self.config.LOG_FILE)
        
        action_space = self.envs.action_spaces[0]
        self.action_space = action_space

        has_distractor_sound = self.config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND

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

        if self.config.RL.DDPPO.pretrained:
            pretrained_state = torch.load(self.config.RL.DDPPO.pretrained_weights, map_location="cpu")
            
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder" not in k and
                       "actor_critic.net.smt_state_encoder" not in k
                },
                strict=False
            )
            
            self.actor_critic.net.visual_encoder.rgb_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.rgb_encoder."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder.rgb_encoder." in k
                },
            )
            
            self.actor_critic.net.visual_encoder.depth_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.depth_encoder."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder.depth_encoder." in k
                },
            )

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

        if smt_cfg.actor_critic_pretrained_path != -1:
            print("use pretrained model for actor critic ", smt_cfg.actor_critic_pretrained_path)
            ckpt_dict = self.load_checkpoint(smt_cfg.actor_critic_pretrained_path, map_location="cpu")
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.actor_critic = self.agent.actor_critic

    def train(self) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        if not distrib.is_initialized():
            distrib.init_process_group(
                backend=self.config.RL.DDPPO.distrib_backend, init_method="env://"
            )
        
        add_signal_handlers()

        tmp_store_dir = os.path.join(self.config.TENSORBOARD_DIR, "ddp_sync")
        if self.world_rank == 0:
            os.makedirs(tmp_store_dir, exist_ok=True)
        distrib.barrier()

        file_store = distrib.FileStore(os.path.join(tmp_store_dir, "sync_file"), self.world_size)
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", file_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_PROCESSES
        )
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        
        if (
            not os.path.isdir(self.config.CHECKPOINT_FOLDER)
            and self.world_rank == 0
        ):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        
        self.agent.init_distributed(find_unused_params=True)
        
        if ppo_cfg.use_belief_predictor and ppo_cfg.BELIEF_PREDICTOR.online_training:
            self.belief_predictor.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )
            if ppo_cfg.use_belief_predictor:
                logger.info(
                    "belief predictor number of trainable parameters: {}".format(
                        sum(
                            param.numel()
                            for param in self.belief_predictor.parameters()
                            if param.requires_grad
                        )
                    )
                )
            logger.info(f"config: {self.config}")

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        obs_space = self.envs.observation_spaces[0]
        if ppo_cfg.use_external_memory:
            memory_dim = self.actor_critic.net.memory_dim
        else:
            memory_dim = None

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

        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.update(batch, None)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        batch = None
        observations = None

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        self.lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        count_steps_start, count_checkpoints, start_update = self.try_to_resume_checkpoint()
        count_steps = count_steps_start

        if hasattr(self, '_saved_lr_scheduler_state') and self._saved_lr_scheduler_state is not None:
            self.lr_scheduler.load_state_dict(self._saved_lr_scheduler_state)
            print(f"Restored learning rate scheduler state from checkpoint")
            delattr(self, '_saved_lr_scheduler_state')

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            
            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.load_state_dict(interrupted_state["belief_predictor"])
            
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            
            self.lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    self.lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        state_dict = dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=self.lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        if self.config.RL.PPO.use_belief_predictor:
                            state_dict['belief_predictor'] = self.belief_predictor.state_dict()
                        save_interrupted_state(state_dict)

                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
                if self.config.RL.PPO.use_belief_predictor:
                    self.belief_predictor.eval()
                
                for step in range(ppo_cfg.num_steps):

                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break

                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()
                if self.config.RL.PPO.use_belief_predictor:
                    self.belief_predictor.train()
                    self.belief_predictor.set_eval_encoders()
                if self._static_smt_encoder:
                    self.actor_critic.net.set_eval_encoders()

                if ppo_cfg.use_belief_predictor and ppo_cfg.BELIEF_PREDICTOR.online_training:
                    location_predictor_loss, prediction_accuracy = self.train_belief_predictor(rollouts)
                else:
                    location_predictor_loss = 0
                    prediction_accuracy = 0
                
                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time

                stats_ordering = list(sorted(running_episode_stats.keys()))
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )
                distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [value_loss, action_loss, dist_entropy, location_predictor_loss, prediction_accuracy, count_steps_delta],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += stats[5].item()

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    losses = [
                        stats[0].item() / self.world_size,
                        stats[1].item() / self.world_size,
                        stats[2].item() / self.world_size,
                        stats[3].item() / self.world_size,
                        stats[4].item() / self.world_size,
                    ]
                    
                    deltas = {
                        k: (
                            (v[-1] - v[0]).sum().item()
                            if len(v) > 1
                            else v[0].sum().item()
                        )
                        for k, v in window_episode_stats.items()
                    }
                    deltas["count"] = max(deltas["count"], 1.0)

                    writer.add_scalar(
                        "Metrics/reward", deltas["reward"] / deltas["count"], count_steps
                    )

                    metrics = {
                        k: v / deltas["count"]
                        for k, v in deltas.items()
                        if k not in {"reward", "count"}
                    }
                    if len(metrics) > 0:
                        for metric, value in metrics.items():
                            writer.add_scalar(f"Metrics/{metric}", value, count_steps)

                    writer.add_scalar("Policy/value_loss", losses[0], count_steps)
                    writer.add_scalar("Policy/policy_loss", losses[1], count_steps)
                    writer.add_scalar("Policy/entropy_loss", losses[2], count_steps)
                    writer.add_scalar("Policy/predictor_loss", losses[3], count_steps)
                    writer.add_scalar("Policy/predictor_accuracy", losses[4], count_steps)
                    writer.add_scalar('Policy/learning_rate', self.lr_scheduler.get_lr()[0], count_steps)

                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                update,
                                (count_steps - count_steps_start)
                                / ((time.time() - t_start) + prev_time),
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(
                                update, env_time, pth_time, count_steps
                            )
                        )
                        
                        logger.info(
                            "Average window size: {}  {}".format(
                                len(window_episode_stats["count"]),
                                "  ".join(
                                    "{}: {:.3f}".format(k, v / deltas["count"])
                                    for k, v in deltas.items()
                                    if k != "count"
                                ),
                            )
                        )

                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth",
                            dict(step=count_steps),
                        )
                        count_checkpoints += 1

            self.envs.close()

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

        logging.debug('Reward: {}'.format(rewards[0]))

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

        if self.config.RL.PPO.use_belief_predictor:
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            self.belief_predictor.update(step_observation, dones)
            for sensor in [LocationBelief.cls_uuid, CategoryBelief.cls_uuid]:
                rollouts.observations[sensor][rollouts.step].copy_(step_observation[sensor])

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

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )


    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> Dict:
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()        
        ppo_cfg = config.RL.PPO
        action_selection = str(
            getattr(config.EVAL, "ACTION_SELECTION", "sample")
        ).lower()
        if action_selection not in ("sample", "argmax"):
            raise ValueError(
                "EVAL.ACTION_SELECTION must be sample or argmax; got {}"
                .format(action_selection)
            )
        
        config.defrost()
        # The original code used the misspelled key ``SPlIT`` and silently
        # kept evaluating the task-config's training split.
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        
        if self.config.DISPLAY_RESOLUTION != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH:
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = \
                self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        logger.info(f"env config: {config}")
        logging.info("[EVAL] action_selection=%s", action_selection)
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'] = spaces.Box(low=0, high=1, shape=(model_resolution,
                                                           model_resolution, 1), dtype=np.uint8)
            observation_space.spaces['rgb'] = spaces.Box(low=0, high=1, shape=(model_resolution,
                                                         model_resolution, 3), dtype=np.uint8)
        else:
            observation_space = self.envs.observation_spaces[0]
        
        self._setup_actor_critic_agent(ppo_cfg, observation_space)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        self.metric_uuids = []
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(metric_cfg.TYPE)
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        observations = self.envs.reset()
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            resize_observation(observations, model_resolution)
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        if self.actor_critic.net.num_recurrent_layers == -1:
            num_recurrent_layers = 1

        test_recurrent_hidden_states = torch.zeros(
            num_recurrent_layers,
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
        
        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.update(batch, None)

            descriptor_pred_gt = [[] for _ in range(self.config.NUM_PROCESSES)]
            for i in range(len(descriptor_pred_gt)):
                category_prediction = np.argmax(batch['category_belief'].cpu().numpy()[i])
                location_prediction = batch['location_belief'].cpu().numpy()[i]
                category_gt = np.argmax(batch['category'].cpu().numpy()[i])
                location_gt = batch['pointgoal_with_gps_compass'].cpu().numpy()[i]
                geodesic_distance = -1
                pair = (category_prediction, location_prediction, category_gt, location_gt, geodesic_distance)
                if 'view_point_goals' in observations[i]:
                    pair += (observations[i]['view_point_goals'],)
                descriptor_pred_gt[i].append(pair)

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        audios = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        self.actor_critic.eval()
        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.eval()

        # ---- Test-time adaptation (TTA) setup ----
        tta_adapter = None
        tta_cfg = getattr(self.config, "TTA", None)
        tta_method = str(
            getattr(tta_cfg, "METHOD", "none") if tta_cfg is not None else "none"
        ).lower()
        idea_cfg = getattr(tta_cfg, "IDEA", None) if tta_cfg is not None else None
        idea_source_collection = None
        source_model_state_sha256 = None
        source_collection_requested = bool(
            getattr(idea_cfg, "SOURCE_COLLECTION", False)
        ) if idea_cfg is not None else False
        if source_collection_requested:
            if bool(self.config.EVAL.USE_CKPT_CONFIG):
                raise ValueError(
                    "ENMuS IDEA source collection requires "
                    "EVAL.USE_CKPT_CONFIG=False"
                )
            if tta_method != "source":
                raise ValueError(
                    "ENMuS IDEA source collection requires TTA.METHOD=source"
                )
            if str(config.EVAL.SPLIT).lower() != "train":
                raise ValueError(
                    "ENMuS IDEA source collection requires EVAL.SPLIT=train"
                )
            if action_selection != "sample":
                raise ValueError(
                    "ENMuS IDEA source collection requires native sampled actions"
                )
            if self.envs.num_envs != 1:
                raise ValueError(
                    "ENMuS IDEA source collection requires NUM_PROCESSES=1"
                )
            if int(self.config.TEST_EPISODE_COUNT) != 128:
                raise ValueError(
                    "canonical ENMuS IDEA source collection requires exactly "
                    "128 episodes"
                )
            if (
                str(getattr(idea_cfg, "SOURCE_STATS_PATH", "") or "")
                or str(getattr(idea_cfg, "SOURCE_STATS_SHA256", "") or "")
            ):
                raise ValueError(
                    "source collection cannot consume an IDEA source artifact"
                )

            from navtta_avn.idea_source import (
                collection_provenance,
                file_sha256,
                load_source_manifest,
                stable_episode_id,
                verify_manifest_assets,
            )
            from navtta_core.tta import (
                SourceStatisticsCollectionSession,
                TransformerFusionProtocol,
                build_multiscale_memory_key_padding_mask,
                module_state_sha256,
            )

            dataset_path = str(config.TASK_CONFIG.DATASET.DATA_PATH).lower()
            if "multi_source" in dataset_path:
                source_setting = "multi_source"
            elif "single_source" in dataset_path:
                source_setting = "single_source"
            else:
                raise ValueError(
                    "cannot infer ENMuS source setting from DATA_PATH"
                )
            manifest_path = str(
                getattr(idea_cfg, "SOURCE_EPISODE_MANIFEST", "") or ""
            )
            manifest_sha256 = str(getattr(
                idea_cfg, "SOURCE_EPISODE_MANIFEST_SHA256", ""
            ) or "")
            task_manifest_path = str(getattr(
                config.TASK_CONFIG.DATASET, "IDEA_SOURCE_EPISODE_MANIFEST", ""
            ) or "")
            task_manifest_sha256 = str(getattr(
                config.TASK_CONFIG.DATASET,
                "IDEA_SOURCE_EPISODE_MANIFEST_SHA256",
                "",
            ) or "")
            if (
                manifest_path != task_manifest_path
                or manifest_sha256.lower() != task_manifest_sha256.lower()
            ):
                raise ValueError(
                    "TTA and task dataset must bind the same source manifest"
                )
            manifest, actual_manifest_sha256 = load_source_manifest(
                manifest_path,
                manifest_sha256,
                model="enmus",
                source_setting=source_setting,
            )
            dataset_index = config.TASK_CONFIG.DATASET.DATA_PATH.format(
                version=config.TASK_CONFIG.DATASET.VERSION,
                split=config.TASK_CONFIG.DATASET.SPLIT,
            )
            verify_manifest_assets(manifest, dataset_index, checkpoint_path)
            actual_checkpoint_sha256 = file_sha256(checkpoint_path)
            if manifest["checkpoint"]["sha256"] != actual_checkpoint_sha256:
                raise ValueError(
                    "loaded ENMuS checkpoint does not match source manifest"
                )

            actor_critic = self.actor_critic

            def _idea_source_forward_logits(policy_inputs):
                features, _, _ = actor_critic.net(
                    policy_inputs["observations"],
                    policy_inputs["rnn_hidden_states"],
                    policy_inputs["prev_actions"],
                    policy_inputs["masks"],
                    policy_inputs.get("ext_memory"),
                    policy_inputs.get("ext_memory_masks"),
                )
                return actor_critic.action_distribution(features).logits

            transformer = actor_critic.net.smt_state_encoder.transformer
            prompt_layers = int(getattr(idea_cfg, "PROMPT_LAYERS", 0))
            source_protocol = TransformerFusionProtocol.for_source_collection(
                transformer,
                _idea_source_forward_logits,
                feature_dim=transformer.d_model,
                num_layers=min(
                    prompt_layers or len(transformer.encoder.layers),
                    len(transformer.encoder.layers),
                ),
                pad_to_multiple=32,
                memory_key_padding_mask_builder=(
                    build_multiscale_memory_key_padding_mask
                ),
            )
            self.actor_critic.eval()
            self.actor_critic.requires_grad_(False)
            source_model_state_sha256 = module_state_sha256(self.actor_critic)
            provenance = collection_provenance(
                manifest,
                actual_manifest_sha256,
                source_model_state_sha256,
            )
            idea_source_collection = SourceStatisticsCollectionSession(
                source_protocol,
                str(getattr(
                    idea_cfg, "SOURCE_COLLECTION_OUTPUT", ""
                ) or ""),
                provenance,
                expected_trajectory_count=128,
            )
            logging.info(
                "[IDEA source] frozen sampled-action collection enabled: setting=%s "
                "manifest=%s order=%s",
                source_setting,
                actual_manifest_sha256,
                manifest["episode_order_sha256"],
            )
        elif tta_cfg is not None and tta_method not in ("none", "", "source"):
            if tta_method == "feedtta" and action_selection != "sample":
                raise ValueError(
                    "FeedTTA REINFORCE requires actions sampled from the policy; "
                    "set EVAL.ACTION_SELECTION=sample"
                )
            if tta_method == "idea" and action_selection != "sample":
                raise ValueError(
                    "AVN IDEA requires task-native sampled actions"
                )
            if tta_method == "atena" and not bool(
                getattr(getattr(tta_cfg, "ATENA", None), "PREFLIGHT_APPROVED", False)
            ):
                raise RuntimeError(
                    "AVN ATENA is intentionally gated. Complete "
                    "avn/experiments/ATENA_PRE_RUN_REVIEW.md and set "
                    "TTA.ATENA.PREFLIGHT_APPROVED=True only for the reviewed run."
                )
            if tta_method == "atena":
                configured_protocol = str(getattr(
                    getattr(tta_cfg, "ATENA", None),
                    "ACTION_SELECTION_PROTOCOL",
                    "policy_argmax",
                )).lower()
                expected_protocol = (
                    "sample_from_policy"
                    if action_selection == "sample"
                    else "policy_argmax"
                )
                if configured_protocol != expected_protocol:
                    raise ValueError(
                        "ATENA action protocol mismatch: EVAL.ACTION_SELECTION={} "
                        "requires TTA.ATENA.ACTION_SELECTION_PROTOCOL={}"
                        .format(action_selection, expected_protocol)
                    )
            if tta_method == "atena":
                atena_task_scope = str(getattr(
                    getattr(tta_cfg, "ATENA", None),
                    "TASK_UPDATE_SCOPE",
                    "replay_reachable_actor_navigation_policy",
                )).lower()
                if atena_task_scope != (
                    "replay_reachable_actor_navigation_policy"
                ):
                    raise ValueError(
                        "ENMuS ATENA can replay the actor navigation graph but "
                        "not an end-to-end full policy"
                    )
                for name, parameter in self.actor_critic.named_parameters():
                    if not name.startswith("critic."):
                        parameter.requires_grad_(True)
            if self.envs.num_envs != 1:
                raise ValueError(
                    "Sequential TTA requires NUM_PROCESSES=1; got {}"
                    .format(self.envs.num_envs)
                )
            from navtta_core.tta import build_adapter
            fusion_protocol = None
            if tta_method == "idea":
                # IDEA injects a soft prompt into the MSMT fusion transformer and
                # reads per-layer statistics.  The binding calls frozen
                # sub-modules only; the policy forward is never edited.
                from navtta_core.tta.fusion import (
                    TransformerFusionProtocol,
                    build_multiscale_memory_key_padding_mask,
                )
                from navtta_core.tta import module_state_sha256
                from navtta_avn.idea_source import file_sha256

                actor_critic = self.actor_critic

                def _idea_forward_logits(policy_inputs):
                    features, _, _ = actor_critic.net(
                        policy_inputs["observations"],
                        policy_inputs["rnn_hidden_states"],
                        policy_inputs["prev_actions"],
                        policy_inputs["masks"],
                        policy_inputs.get("ext_memory"),
                        policy_inputs.get("ext_memory_masks"),
                    )
                    return actor_critic.action_distribution(features).logits

                transformer = actor_critic.net.smt_state_encoder.transformer
                dataset_path = str(config.TASK_CONFIG.DATASET.DATA_PATH).lower()
                if "multi_source" in dataset_path:
                    source_setting = "multi_source"
                elif "single_source" in dataset_path:
                    source_setting = "single_source"
                else:
                    raise ValueError(
                        "cannot infer ENMuS source setting from DATA_PATH"
                    )
                source_manifest_sha256 = str(getattr(
                    idea_cfg, "SOURCE_EPISODE_MANIFEST_SHA256", ""
                ) or "").lower()
                if not re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256):
                    raise ValueError(
                        "IDEA evaluation requires the source episode-manifest SHA256"
                    )
                fusion_protocol = TransformerFusionProtocol(
                    transformer,
                    _idea_forward_logits,
                    feature_dim=transformer.d_model,
                    num_layers=min(
                        int(getattr(idea_cfg, "PROMPT_LAYERS", 0)) or
                        len(transformer.encoder.layers),
                        len(transformer.encoder.layers),
                    ),
                    source_stats_path=(
                        getattr(idea_cfg, "SOURCE_STATS_PATH", "") or None
                    ),
                    source_stats_sha256=(
                        getattr(idea_cfg, "SOURCE_STATS_SHA256", "") or None
                    ),
                    expected_source_trajectories=int(
                        getattr(idea_cfg, "SOURCE_TRAJECTORIES", 128)
                    ),
                    expected_source_provenance={
                        "checkpoint_sha256": file_sha256(checkpoint_path),
                        "model_state_sha256": module_state_sha256(actor_critic),
                        "episode_selection_manifest_sha256": (
                            source_manifest_sha256
                        ),
                        "action_selection": "sample",
                        "model": "enmus",
                        "source_setting": source_setting,
                    },
                    pad_to_multiple=32,
                    memory_key_padding_mask_builder=(
                        build_multiscale_memory_key_padding_mask
                    ),
                )
            tta_adapter = build_adapter(
                self.actor_critic, tta_cfg, fusion_protocol=fusion_protocol
            )
            if tta_method == "atena" and tta_adapter.diagnostics().get(
                "requires_task_trainability_wiring", False
            ):
                raise RuntimeError(
                    "ENMuS ATENA actor navigation parameters are not all "
                    "trainable after task-level wiring"
                )
            tta_adapter.episode_start()
            logging.info(
                "[TTA] enabled: method=%s, episodic=%s, lr=%s, scope=%s, last_k=%s",
                tta_cfg.METHOD,
                tta_cfg.EPISODIC,
                tta_adapter.current_lr,
                getattr(tta_adapter, "param_scope", tta_cfg.NORM_SCOPE),
                tta_cfg.LAST_K_LN,
            )

        tta_action_counts = [0 for _ in range(self.envs.action_spaces[0].n)]
        tta_valid_action_count = len(tta_action_counts)
        tta_max_prob_sum = 0.0
        tta_probability_steps = 0
        tta_window_stats = []
        tta_log_interval = int(
            getattr(tta_cfg, "LOG_INTERVAL_EPISODES", 50)
        ) if tta_cfg is not None else 50
        
        t = tqdm(total=self.config.TEST_EPISODE_COUNT)
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            if (
                idea_source_collection is not None
                and idea_source_collection.current_trajectory is None
            ):
                ordinal = idea_source_collection.accumulator.trajectory_count
                actual_id = stable_episode_id(current_episodes[0])
                expected_id = manifest["episodes"][ordinal]["trajectory_id"]
                if actual_id != expected_id:
                    raise RuntimeError(
                        "ENMuS IDEA source episode order mismatch: expected "
                        "{}, got {}".format(expected_id, actual_id)
                    )
                idea_source_collection.begin_trajectory(actual_id)

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
            
            if tta_adapter is None:
                if idea_source_collection is not None:
                    idea_source_collection.observe_step({
                        "observations": batch,
                        "rnn_hidden_states": test_recurrent_hidden_states,
                        "prev_actions": prev_actions,
                        "masks": not_done_masks,
                        "ext_memory": (
                            em_memory if ppo_cfg.use_external_memory else None
                        ),
                        "ext_memory_masks": (
                            em_masks if ppo_cfg.use_external_memory else None
                        ),
                    })
                with torch.no_grad():
                    _, actions, _, test_recurrent_hidden_states, test_em_features = self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        em_memory if ppo_cfg.use_external_memory else None,
                        em_masks if ppo_cfg.use_external_memory else None,
                        deterministic=(action_selection == "argmax")
                    )
                    prev_actions.copy_(actions)
            else:
                policy_inputs = {
                    "observations": batch,
                    "rnn_hidden_states": test_recurrent_hidden_states,
                    "prev_actions": prev_actions,
                    "masks": not_done_masks,
                    "ext_memory": (
                        em_memory if ppo_cfg.use_external_memory else None
                    ),
                    "ext_memory_masks": (
                        em_masks if ppo_cfg.use_external_memory else None
                    ),
                }
                valid_action_count = tta_valid_action_count
                if tta_method == "eam":
                    policy_inputs["valid_action_count"] = valid_action_count
                # EAM Algorithm 3 updates/samples its replay buffer before the
                # original and auxiliary policy inference for this action.
                tta_adapter.before_inference(
                    policy_inputs=policy_inputs,
                    **(
                        {"valid_action_count": valid_action_count}
                        if tta_method == "eam" else {}
                    )
                )
                with torch.set_grad_enabled(tta_adapter.requires_source_grad):
                    features, test_recurrent_hidden_states, test_em_features = self.actor_critic.net(
                        policy_inputs["observations"],
                        policy_inputs["rnn_hidden_states"],
                        policy_inputs["prev_actions"],
                        policy_inputs["masks"],
                        policy_inputs["ext_memory"],
                        policy_inputs["ext_memory_masks"],
                    )
                    source_distribution = self.actor_critic.action_distribution(features)
                    action_logits = tta_adapter.prepare_action(
                        source_distribution.logits,
                        features=features,
                        policy_inputs=policy_inputs,
                        **(
                            {"valid_action_count": valid_action_count}
                            if tta_method == "eam" else {}
                        )
                    )
                    distribution = source_distribution.__class__(logits=action_logits)
                with torch.no_grad():
                    actions = tta_adapter.select_action(distribution)
                    tta_max_prob_sum += distribution.probs.max(dim=-1)[0].mean().item()
                    tta_probability_steps += 1
                tta_adapter.adapt(
                    source_distribution.logits,
                    action=actions,
                    features=features,
                    policy_inputs=policy_inputs,
                )
                test_recurrent_hidden_states = test_recurrent_hidden_states.detach()
                test_em_features = test_em_features.detach()
                prev_actions.copy_(actions)

            actions = [a[0].item() for a in actions]
            if tta_adapter is not None or idea_source_collection is not None:
                for action in actions:
                    tta_action_counts[action] += 1
            outputs = self.envs.step(actions)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]     

            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    pred = None
                    if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in observations[i]:
                        for observation in observations[i]['intermediate']:
                            frame = observations_to_image(observation, infos[i], pred=pred)
                            rgb_frames[i].append(frame)
                        del observations[i]['intermediate']

                    frame = observations_to_image(observations[i], infos[i], pred=pred)
                    rgb_frames[i].append(frame)
                    audios[i].append(observations[i]['audiogoal'])
                           
            if config.DISPLAY_RESOLUTION != model_resolution:
                resize_observation(observations, model_resolution)
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            
            if ppo_cfg.use_external_memory:
                test_em.insert(test_em_features, not_done_masks)
            
            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.update(batch, dones)

                for i in range(len(descriptor_pred_gt)):
                    category_prediction = np.argmax(batch['category_belief'].cpu().numpy()[i])
                    location_prediction = batch['location_belief'].cpu().numpy()[i]
                    category_gt = np.argmax(batch['category'].cpu().numpy()[i])
                    location_gt = batch['pointgoal_with_gps_compass'].cpu().numpy()[i]
                    if dones[i]:
                        geodesic_distance = -1
                    else:
                        geodesic_distance = infos[i]['distance_to_goal']
                    pair = (category_prediction, location_prediction, category_gt, location_gt, geodesic_distance)
                    if 'view_point_goals' in observations[i]:
                        pair += (observations[i]['view_point_goals'],)
                    descriptor_pred_gt[i].append(pair)

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
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats['geodesic_distance'] = current_episodes[i].info['geodesic_distance']
                    episode_stats['euclidean_distance'] = norm(np.array(current_episodes[i].goals[0].position) -
                                                               np.array(current_episodes[i].start_position))
                    episode_stats['audio_duration'] = int(current_episodes[i].duration)
                    episode_stats['gt_na'] = int(current_episodes[i].info['num_action'])
                    
                    if self.config.RL.PPO.use_belief_predictor:
                        episode_stats['gt_na'] = int(current_episodes[i].info['num_action'])
                        episode_stats['descriptor_pred_gt'] = descriptor_pred_gt[i][:-1]
                        descriptor_pred_gt[i] = [descriptor_pred_gt[i][-1]]
                    
                    logging.debug(episode_stats)
                    current_episode_reward[i] = 0
                    
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    t.update()

                    if idea_source_collection is not None:
                        idea_source_collection.end_trajectory()
                    elif tta_adapter is not None:
                        tta_adapter.episode_end(episode_stats=episode_stats)
                        tta_window_stats.append(episode_stats)
                        if len(tta_window_stats) >= tta_log_interval:
                            window_summary = {}
                            for key in ("success", "spl", "softspl", "na"):
                                values = [entry[key] for entry in tta_window_stats if key in entry]
                                if values:
                                    window_summary[key] = sum(values) / len(values)
                            logging.info(
                                "[TTA] episodes=%d window=%s action_counts=%s "
                                "mean_max_prob=%.6f adapter=%s",
                                len(stats_episodes),
                                window_summary,
                                tta_action_counts,
                                tta_max_prob_sum / max(1, tta_probability_steps),
                                tta_adapter.diagnostics(),
                            )
                            tta_window_stats = []
                        tta_adapter.episode_start()

                    if len(self.config.VIDEO_OPTION) > 0:
                        fps = self.config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS \
                                    if self.config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE else 1
                        if 'sound' in current_episodes[i].info:
                            sound = current_episodes[i].info['sound']
                        else:
                            sound = current_episodes[i].sound_id.split('/')[1][:-4]
                        
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i][:-1],
                            scene_name=current_episodes[i].scene_id.split('/')[3],
                            sound=sound,
                            sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name='spl',
                            metric_value=infos[i]['spl'],
                            tb_writer=writer,
                            audios=audios[i][:-1],
                            fps=fps
                        )

                        rgb_frames[i] = []
                        audios[i] = []

                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        if self.config.RL.PPO.use_belief_predictor:
                            pred = episode_stats['descriptor_pred_gt'][-1]
                        else:
                            pred = None

                        top_down_map = plot_top_down_map(infos[i],
                                                         dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET,
                                                         pred=pred)

                        scene = current_episodes[i].scene_id.split('/')[3]
                        sound = current_episodes[i].sound_id.split('/')[1]
                        writer.add_image(f"{scene}_{current_episodes[i].episode_id}_{int(infos[i]['success']):d}_{infos[i]['spl']:.2f}",
                                         top_down_map,
                                         dataformats='WHC')
            if not self.config.RL.PPO.use_belief_predictor:
                descriptor_pred_gt = None

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
                descriptor_pred_gt
            )

        stats_file = os.path.join(config.TENSORBOARD_DIR,
                                  '{}_stats_{}.json'.format(config.EVAL.SPLIT, config.SEED))
        with open(stats_file, 'w') as fo:
            json.dump({','.join(key): value for key, value in stats_episodes.items()}, fo, cls=NpEncoder)

        if idea_source_collection is not None:
            if not idea_source_collection.complete:
                raise RuntimeError(
                    "IDEA source collection ended without exactly 128 "
                    "completed trajectories"
                )
            final_model_state_sha256 = module_state_sha256(self.actor_critic)
            if final_model_state_sha256 != source_model_state_sha256:
                raise RuntimeError(
                    "frozen source policy state changed during IDEA collection"
                )
            diagnostics = idea_source_collection.diagnostics()
            diagnostics["source_policy_frozen"] = True
            diagnostics["source_model_state_sha256"] = source_model_state_sha256
            diagnostics["action_selection"] = action_selection
            diagnostics["action_counts"] = tta_action_counts
        elif tta_adapter is not None:
            diagnostics = tta_adapter.diagnostics()
        else:
            diagnostics = None

        if diagnostics is not None:
            diagnostics["task_action_space_contract"] = (
                "fixed_discrete_all_actions_valid"
            )
            diagnostics["task_valid_action_count"] = tta_valid_action_count
            diagnostics["task_action_selection"] = action_selection
            diagnostics["tent_canonical_update_interval"] = (
                int(getattr(tta_cfg, "UPDATE_INTERVAL", 1)) == 1
                if tta_method == "tent" else None
            )
            diagnostics["fstta_reset_var_hist_each_episode"] = (
                bool(getattr(
                    getattr(tta_cfg, "FSTTA", None),
                    "RESET_VAR_HIST_EACH_EPISODE",
                    False,
                )) if tta_method == "fstta" else None
            )
            diagnostics["fstta_variance_history_profile"] = (
                (
                    "released_code_rollout_reset_ablation"
                    if diagnostics["fstta_reset_var_hist_each_episode"]
                    else "paper_eq6_test_stream_history"
                ) if tta_method == "fstta" else None
            )
            diagnostics["feedtta_sgr_mode"] = (
                str(getattr(
                    getattr(tta_cfg, "FEEDTTA", None),
                    "SGR_MODE",
                    "paper_main",
                )).lower() if tta_method == "feedtta" else None
            )
            diagnostics["feedtta_canonical_protocol"] = (
                action_selection == "sample"
                and diagnostics["feedtta_sgr_mode"] == "paper_main"
                if tta_method == "feedtta" else None
            )
            diagnostics["feedtta_requires_matched_sampled_source"] = (
                True if tta_method == "feedtta" else None
            )
            diagnostics["atena_task_update_scope"] = (
                "replay_reachable_actor_navigation_policy"
                if tta_method == "atena" else None
            )
            diagnostics["atena_exact_replay_within_declared_scope"] = (
                True if tta_method == "atena" else None
            )
            diagnostics["atena_full_end_to_end_policy_claimed"] = (
                False if tta_method == "atena" else None
            )
            diagnostics["action_counts"] = tta_action_counts
            if idea_source_collection is None:
                diagnostics["action_counts"] = tta_action_counts
                diagnostics["mean_max_action_probability"] = (
                    tta_max_prob_sum / max(1, tta_probability_steps)
                )
            diagnostics_file = os.path.join(
                config.TENSORBOARD_DIR,
                "tta_diagnostics_{}.json".format(config.SEED),
            )
            with open(diagnostics_file, "w") as file_handle:
                json.dump(diagnostics, file_handle, indent=2)

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            if stat_key in ['audio_duration', 'gt_na', 'descriptor_pred_gt', 'view_point_goals']:
                continue
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = aggregated_stats[metric_uuid] / num_episodes

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        for metric_uuid in self.metric_uuids:
            logger.info(
                f"Average episode {metric_uuid}: {episode_metrics_mean[metric_uuid]:.6f}"
            )

        if not config.EVAL.SPLIT.startswith('test'):
            writer.add_scalar("{}/reward".format(config.EVAL.SPLIT), episode_reward_mean, checkpoint_index)
            for metric_uuid in self.metric_uuids:
                writer.add_scalar(f"{config.EVAL.SPLIT}/{metric_uuid}", episode_metrics_mean[metric_uuid],
                                  checkpoint_index)

        self.envs.close()

        result = {
            'episode_reward_mean': episode_reward_mean
        }
        for metric_uuid in self.metric_uuids:
            result['episode_{}_mean'.format(metric_uuid)] = episode_metrics_mean[metric_uuid]

        return result

    def try_to_resume_checkpoint(self):
        import glob
        
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        if len(checkpoints) == 0:
            count_steps = 0
            count_checkpoints = 0
            start_update = 0
        else:
            last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
            checkpoint_path = last_ckpt
            
            ckpt_dict = self.load_checkpoint(checkpoint_path)
            
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            print(f"✓ Model weights loaded successfully")
            
            if self.config.RL.PPO.use_belief_predictor and "belief_predictor" in ckpt_dict:
                self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])
                print(f"✓ Belief predictor state loaded successfully")
            
            if "optim_state" in ckpt_dict:
                self.agent.optimizer.load_state_dict(ckpt_dict["optim_state"])
                print(f"✓ Optimizer state loaded successfully")
            elif "optimizer_state" in ckpt_dict:
                self.agent.optimizer.load_state_dict(ckpt_dict["optimizer_state"])
                print(f"✓ Optimizer state loaded successfully")
            else:
                print(f"⚠ Optimizer state not found in checkpoint")
            
            if "lr_sched_state" in ckpt_dict:
                self._saved_lr_scheduler_state = ckpt_dict["lr_sched_state"]
                print(f"✓ LR scheduler state saved for later restoration")
            else:
                print(f"⚠ LR scheduler state not found in checkpoint")
            
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = ckpt_dict["extra_state"]["step"]
            count_checkpoints = ckpt_id + 1
            start_update = ckpt_dict["config"].CHECKPOINT_INTERVAL * ckpt_id + 1
            
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")
            
            if not hasattr(self, '_saved_lr_scheduler_state'):
                self._saved_lr_scheduler_state = None

        return count_steps, count_checkpoints, start_update

    def save_checkpoint(
        self, file_name: str, extra_state=None
    ) -> None:
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        
        if self.config.RL.PPO.use_belief_predictor:
            checkpoint["belief_predictor"] = self.belief_predictor.state_dict()
        
        checkpoint["optim_state"] = self.agent.optimizer.state_dict()
        
        checkpoint["lr_sched_state"] = self.lr_scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )
