#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
from collections import deque, defaultdict
from typing import Dict, List, Any
import json
import random
import glob
import re

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm

from habitat import Config, logger
from ss_baselines.common.utils import observations_to_image
from ss_baselines.common.base_trainer import BaseRLTrainer
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorage
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    plot_top_down_map,
    resize_observation,
    NpEncoder
)
from ss_baselines.savi.ppo.policy import AudioNavBaselinePolicy, AudioNavSMTPolicy
from ss_baselines.savi.ppo.ppo import PPO
from ss_baselines.savi.ppo.slurm_utils import (
    EXIT,
    REQUEUE,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from ss_baselines.savi.models.rollout_storage import RolloutStorage, ExternalMemory
from ss_baselines.savi.models.belief_predictor import BeliefPredictor
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
from soundspaces.tasks.nav import LocationBelief, CategoryBelief, SpectrogramSensor


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@baseline_registry.register_trainer(name="savi")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None

        self._static_smt_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]

        if not ppo_cfg.use_external_memory:
            self.actor_critic = AudioNavBaselinePolicy(
                observation_space=observation_space,
                action_space=self.envs.action_spaces[0],
                hidden_size=ppo_cfg.hidden_size,
                goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
                extra_rgb=self.config.EXTRA_RGB
            )
        else:
            smt_cfg = ppo_cfg.SCENE_MEMORY_TRANSFORMER
            self.actor_critic = AudioNavSMTPolicy(
                observation_space=observation_space,
                action_space=self.envs.action_spaces[0],
                hidden_size=smt_cfg.hidden_size,
                nhead=smt_cfg.nhead,
                num_encoder_layers=smt_cfg.num_encoder_layers,
                num_decoder_layers=smt_cfg.num_decoder_layers,
                dropout=smt_cfg.dropout,
                activation=smt_cfg.activation,
                use_pretrained=smt_cfg.use_pretrained,
                pretrained_path=smt_cfg.pretrained_path,
                use_belief_as_goal=ppo_cfg.use_belief_predictor,
                use_label_belief=smt_cfg.use_label_belief,
                use_location_belief=smt_cfg.use_location_belief
            )

            if ppo_cfg.use_belief_predictor:
                belief_cfg = ppo_cfg.BELIEF_PREDICTOR
                smt = self.actor_critic.net.smt_state_encoder
                self.belief_predictor = BeliefPredictor(belief_cfg, self.device, smt._input_size, smt._pose_indices,
                                                        smt.hidden_state_size, self.envs.num_envs,
                                                        ).to(device=self.device)
                for param in self.belief_predictor.parameters():
                    param.requires_grad = False

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

        if self.config.RESUME:
            ckpt_dict = self.load_checkpoint('data/models/smt_with_pose/ckpt.400.pth', map_location="cpu")
            self.agent.actor_critic.net.visual_encoder.load_state_dict(self.search_dict(ckpt_dict, 'visual_encoder'))
            self.agent.actor_critic.net.goal_encoder.load_state_dict(self.search_dict(ckpt_dict, 'goal_encoder'))
            self.agent.actor_critic.net.action_encoder.load_state_dict(self.search_dict(ckpt_dict, 'action_encoder'))

        if ppo_cfg.use_external_memory and smt_cfg.freeze_encoders:
            self._static_smt_encoder = True
            self.actor_critic.net.freeze_encoders()

        self.actor_critic.to(self.device)

    @staticmethod
    def search_dict(ckpt_dict, encoder_name):
        encoder_dict = {}
        for key, value in ckpt_dict['state_dict'].items():
            if encoder_name in key:
                encoder_dict['.'.join(key.split('.')[3:])] = value

        return encoder_dict

    def save_checkpoint(
        self, file_name: str, extra_state=None
    ) -> None:
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if self.config.RL.PPO.use_belief_predictor:
            checkpoint["belief_predictor"] = self.belief_predictor.state_dict()
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def try_to_resume_checkpoint(self):
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        if len(checkpoints) == 0:
            count_steps = 0
            count_checkpoints = 0
            start_update = 0
        else:
            last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
            checkpoint_path = last_ckpt
            # Restore checkpoints to models
            ckpt_dict = self.load_checkpoint(checkpoint_path)
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = ckpt_dict["extra_state"]["step"]
            count_checkpoints = ckpt_id + 1
            start_update = ckpt_dict["config"].CHECKPOINT_INTERVAL * ckpt_id + 1
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

        return count_steps, count_checkpoints, start_update

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            external_memory = None
            external_memory_masks = None
            if self.config.RL.PPO.use_external_memory:
                external_memory = rollouts.external_memory[:, rollouts.step].contiguous()
                external_memory_masks = rollouts.external_memory_masks[rollouts.step]

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

    def train_belief_predictor(self, rollouts):
        bp = self.belief_predictor
        num_epoch = 5
        num_mini_batch = 1

        advantages = torch.zeros_like(rollouts.returns)
        value_loss_epoch = 0
        running_regressor_corrects = 0
        num_sample = 0

        for e in range(num_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    external_memory,
                    external_memory_masks,
                ) = sample

                bp.optimizer.zero_grad()

                inputs = obs_batch[SpectrogramSensor.cls_uuid].permute(0, 3, 1, 2)
                preds = bp.cnn_forward(obs_batch)

                masks = (torch.sum(torch.reshape(obs_batch[SpectrogramSensor.cls_uuid],
                        (obs_batch[SpectrogramSensor.cls_uuid].shape[0], -1)), dim=1, keepdim=True) != 0).float()
                gts = obs_batch[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
                transformed_gts = torch.stack([gts[:, 1], -gts[:, 0]], dim=1)
                masked_preds = masks.expand_as(preds) * preds
                masked_gts = masks.expand_as(transformed_gts) * transformed_gts
                loss = bp.regressor_criterion(masked_preds, masked_gts)

                bp.before_backward(loss)
                loss.backward()
                # self.after_backward(loss)

                bp.optimizer.step()
                value_loss_epoch += loss.item()

                rounded_preds = torch.round(preds)
                bitwise_close = torch.bitwise_and(torch.isclose(rounded_preds[:, 0], transformed_gts[:, 0]),
                                                  torch.isclose(rounded_preds[:, 1], transformed_gts[:, 1]))
                running_regressor_corrects += torch.sum(torch.bitwise_and(bitwise_close, masks.bool().squeeze(1)))
                num_sample += torch.sum(masks).item()

        value_loss_epoch /= num_epoch * num_mini_batch
        if num_sample == 0:
            prediction_accuracy = 0
        else:
            prediction_accuracy = running_regressor_corrects / num_sample

        return value_loss_epoch, prediction_accuracy

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            external_memory = None
            external_memory_masks = None
            if ppo_cfg.use_external_memory:
                external_memory = rollouts.external_memory[:, rollouts.step].contiguous()
                external_memory_masks = rollouts.external_memory_masks[rollouts.step]

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

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # add_signal_handlers()

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME), workers_ignore_signals=True
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        if ppo_cfg.use_external_memory:
            memory_dim = self.actor_critic.net.memory_dim
        else:
            memory_dim = None

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            ppo_cfg.use_external_memory,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size + ppo_cfg.num_steps,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
            memory_dim,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)
        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.update(batch, None)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor].view_as(rollouts.observations[sensor][0]))

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
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

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        interrupted_state = load_interrupted_state(model_dir=self.config.MODEL_DIR)
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optimizer_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_scheduler_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set():
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optimizer_state=self.agent.optimizer.state_dict(),
                                lr_scheduler_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            ),
                            model_dir=self.config.MODEL_DIR
                        )
                        requeue_job()
                    return

                for step in range(ppo_cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

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

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    # writer.add_scalars("metrics", metrics, count_steps)
                    for metric, value in metrics.items():
                        writer.add_scalar(f"Metrics/{metric}", value, count_steps)

                writer.add_scalar("Policy/value_loss", value_loss, count_steps)
                writer.add_scalar("Policy/policy_loss", action_loss, count_steps)
                writer.add_scalar("Policy/entropy_loss", dist_entropy, count_steps)
                writer.add_scalar('Policy/learning_rate', lr_scheduler.get_lr()[0], count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
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

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0
    ) -> Dict:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
            
        # Map location CPU is almost always better than mapping to a CUDA device.
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
                "EVAL.ACTION_SELECTION must be 'sample' or 'argmax'; got {}"
                .format(action_selection)
            )

        config.defrost()
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
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        logger.info(f"env config: {config}")
        logging.info("[EVAL] action_selection=%s", action_selection)
        # Force CUDA initialization before forking env workers (avoid SIGSEGV
        # in torch.cuda._lazy_init triggered by actor_critic.to(self.device)).
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.TORCH_GPU_ID)
            torch.cuda.init()
        self.envs = construct_envs(
            config, get_env_class(config.ENV_NAME)
        )
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces['depth'].shape = (model_resolution, model_resolution, 1)
            observation_space.spaces['rgb'].shape = (model_resolution, model_resolution, 3)
        else:
            observation_space = self.envs.observation_spaces[0]
        self._setup_actor_critic_agent(ppo_cfg, observation_space)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        if self.config.RL.PPO.use_belief_predictor and "belief_predictor" in ckpt_dict:
            self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])

        self.metric_uuids = []
        # get name of performance metric, e.g. "spl"
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(
                metric_cfg.TYPE
            )
            self.metric_uuids.append(measure_type(sim=None, task=None, config=None)._get_uuid())

        observations = self.envs.reset()
        if config.DISPLAY_RESOLUTION != model_resolution:
            obs_copy = resize_observation(observations, model_resolution)
        else:
            obs_copy = observations
        batch = batch_obs(obs_copy, self.device, skip_list=['view_point_goals', 'intermediate',
                                                            'oracle_action_sensor'])

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        if self.actor_critic.net.num_recurrent_layers == -1:
            num_recurrent_layers = 1
        else:
            num_recurrent_layers = self.actor_critic.net.num_recurrent_layers
        test_recurrent_hidden_states = torch.zeros(
            num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        if ppo_cfg.use_external_memory:
            test_em = ExternalMemory(
                self.config.NUM_PROCESSES,
                ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                self.actor_critic.net.memory_dim,
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
        stats_episodes = dict()  # dict of dicts that stores stats per episode
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
        ]  # type: List[List[np.ndarray]]
        audios = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        self.actor_critic.eval()
        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.eval()

        # IDEA source anchors are collected in a separate, fail-closed mode.
        # This is never enabled during target evaluation and never constructs
        # a TTA adapter.
        source_collection = None
        source_manifest = None
        source_model_state_sha256 = None
        tta_cfg = getattr(config, "TTA", None)
        tta_method = str(
            getattr(tta_cfg, "METHOD", "none") if tta_cfg is not None else "none"
        ).lower()
        idea_cfg = getattr(tta_cfg, "IDEA", None) if tta_cfg is not None else None
        idea_source_collection = bool(
            getattr(idea_cfg, "SOURCE_COLLECTION", False)
        )
        if idea_source_collection:
            from navtta_core.tta import (
                SourceStatisticsCollectionSession,
                TransformerFusionProtocol,
                module_state_sha256,
            )
            from navtta_avn.idea_source import (
                collection_provenance,
                load_source_manifest,
                verify_manifest_assets,
            )

            if bool(config.EVAL.USE_CKPT_CONFIG):
                raise ValueError(
                    "SMT+Audio IDEA source collection requires "
                    "EVAL.USE_CKPT_CONFIG=False"
                )
            if tta_method != "source":
                raise ValueError("IDEA source collection requires TTA.METHOD=source")
            if str(config.EVAL.SPLIT).lower() != "train":
                raise ValueError("IDEA source collection requires EVAL.SPLIT=train")
            if action_selection != "sample":
                raise ValueError("IDEA source collection requires native sampled actions")
            if int(config.NUM_PROCESSES) != 1 or int(config.TEST_EPISODE_COUNT) != 128:
                raise ValueError(
                    "IDEA source collection requires one environment and exactly 128 episodes"
                )
            if (
                str(getattr(idea_cfg, "SOURCE_STATS_PATH", "") or "")
                or str(getattr(idea_cfg, "SOURCE_STATS_SHA256", "") or "")
            ):
                raise ValueError(
                    "source collection cannot consume an IDEA source artifact"
                )
            dataset_path = str(config.TASK_CONFIG.DATASET.DATA_PATH).lower()
            if "multi_source" in dataset_path:
                source_setting = "multi_source"
            elif "single_source" in dataset_path:
                source_setting = "single_source"
            else:
                raise ValueError(
                    "cannot infer SMT+Audio source setting from DATA_PATH"
                )
            manifest_path = str(getattr(idea_cfg, "SOURCE_EPISODE_MANIFEST", ""))
            manifest_sha = str(getattr(
                idea_cfg, "SOURCE_EPISODE_MANIFEST_SHA256", ""
            ))
            source_manifest, manifest_sha = load_source_manifest(
                manifest_path,
                manifest_sha,
                model="smt_audio",
                source_setting=source_setting,
            )
            dataset_index = config.TASK_CONFIG.DATASET.DATA_PATH.format(
                version=config.TASK_CONFIG.DATASET.VERSION,
                split=config.TASK_CONFIG.DATASET.SPLIT,
            )
            verify_manifest_assets(
                source_manifest, dataset_index, checkpoint_path
            )
            task_manifest_path = str(getattr(
                config.TASK_CONFIG.DATASET,
                "IDEA_SOURCE_EPISODE_MANIFEST",
                "",
            ))
            task_manifest_sha = str(getattr(
                config.TASK_CONFIG.DATASET,
                "IDEA_SOURCE_EPISODE_MANIFEST_SHA256",
                "",
            ))
            if task_manifest_path != manifest_path or task_manifest_sha != manifest_sha:
                raise ValueError(
                    "IDEA trainer and dataset must consume the same source manifest"
                )
            for parameter in self.actor_critic.parameters():
                parameter.requires_grad_(False)
            source_model_state_sha256 = module_state_sha256(self.actor_critic)

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
            protocol = TransformerFusionProtocol.for_source_collection(
                transformer,
                _idea_source_forward_logits,
                feature_dim=transformer.d_model,
                num_layers=min(
                    int(getattr(idea_cfg, "PROMPT_LAYERS", 0))
                    or len(transformer.encoder.layers),
                    len(transformer.encoder.layers),
                ),
            )
            source_collection = SourceStatisticsCollectionSession(
                protocol,
                str(getattr(idea_cfg, "SOURCE_COLLECTION_OUTPUT", "")),
                collection_provenance(
                    source_manifest,
                    manifest_sha,
                    source_model_state_sha256,
                ),
                expected_trajectory_count=128,
            )

        # ---- Test-time adaptation (TTA) setup ----
        # Continual (no per-episode reset) online adaptation by default; each
        # adapter selects its declared parameter scope inside build_adapter().
        # Belief predictor stays frozen.
        tta_adapter = None
        if tta_cfg is not None and tta_method not in ("none", "", "source"):
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
                        "SMT+Audio ATENA can replay the actor navigation graph "
                        "but not an end-to-end full policy"
                    )
                # Core scope='all' respects existing requires_grad flags.  The
                # task binding must therefore enable the complete replayed
                # actor path explicitly while leaving the value-only critic out.
                for name, parameter in self.actor_critic.named_parameters():
                    if not name.startswith("critic."):
                        parameter.requires_grad_(True)
            from navtta_core.tta import build_adapter
            if self.envs.num_envs != 1:
                raise ValueError(
                    "Sequential TTA requires NUM_PROCESSES=1; got {}. "
                    "Use the TTA test yaml rather than a parallel source-eval yaml."
                    .format(self.envs.num_envs)
                )
            fusion_protocol = None
            if tta_method == "idea":
                # IDEA injects a soft prompt into the SMT fusion transformer and
                # reads per-layer statistics.  The binding calls frozen
                # sub-modules only; the policy forward is never edited.
                from navtta_core.tta import (
                    TransformerFusionProtocol,
                    module_state_sha256,
                )
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
                idea_cfg = getattr(tta_cfg, "IDEA", None)
                dataset_path = str(config.TASK_CONFIG.DATASET.DATA_PATH).lower()
                if "multi_source" in dataset_path:
                    source_setting = "multi_source"
                elif "single_source" in dataset_path:
                    source_setting = "single_source"
                else:
                    raise ValueError(
                        "cannot infer SMT+Audio source setting from DATA_PATH"
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
                        "model": "smt_audio",
                        "source_setting": source_setting,
                    },
                )
            tta_adapter = build_adapter(
                self.actor_critic, tta_cfg, fusion_protocol=fusion_protocol
            )
            if tta_method == "atena" and tta_adapter.diagnostics().get(
                "requires_task_trainability_wiring", False
            ):
                raise RuntimeError(
                    "SMT+Audio ATENA actor navigation parameters are not all "
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

            if source_collection is not None and source_collection.current_trajectory is None:
                from navtta_avn.idea_source import stable_episode_id
                ordinal = source_collection.accumulator.trajectory_count
                actual_id = stable_episode_id(current_episodes[0])
                expected_id = source_manifest["episodes"][ordinal]["trajectory_id"]
                if actual_id != expected_id:
                    raise RuntimeError(
                        "IDEA source episode order mismatch: expected {}, got {}"
                        .format(expected_id, actual_id)
                    )
                source_collection.begin_trajectory(actual_id)

            if source_collection is not None:
                policy_inputs = {
                    "observations": batch,
                    "rnn_hidden_states": test_recurrent_hidden_states,
                    "prev_actions": prev_actions,
                    "masks": not_done_masks,
                    "ext_memory": (
                        test_em.memory[:, 0]
                        if ppo_cfg.use_external_memory else None
                    ),
                    "ext_memory_masks": (
                        test_em.masks if ppo_cfg.use_external_memory else None
                    ),
                }
                source_collection.observe_step(policy_inputs)
                with torch.no_grad():
                    _, actions, _, test_recurrent_hidden_states, test_em_features = self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        test_em.memory[:, 0] if ppo_cfg.use_external_memory else None,
                        test_em.masks if ppo_cfg.use_external_memory else None,
                        # Source statistics must follow the same AVN behavior
                        # policy as target evaluation.  The guarded collection
                        # protocol above requires ``action_selection == sample``;
                        # the returned tensor is copied to ``prev_actions`` and
                        # passed unchanged to ``envs.step`` below.
                        deterministic=(action_selection == "argmax"),
                    )
                    prev_actions.copy_(actions)
            elif tta_adapter is None:
                with torch.no_grad():
                    _, actions, _, test_recurrent_hidden_states, test_em_features = self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        test_em.memory[:, 0] if ppo_cfg.use_external_memory else None,
                        test_em.masks if ppo_cfg.use_external_memory else None,
                        deterministic=(action_selection == "argmax")
                    )
                    prev_actions.copy_(actions)
            else:
                # The adapter receives an immutable description of the policy
                # forward. EAM replays it through both policy branches;
                # feedback methods use the selected action/features and update
                # only after the episode outcome is known.
                policy_inputs = {
                    "observations": batch,
                    "rnn_hidden_states": test_recurrent_hidden_states,
                    "prev_actions": prev_actions,
                    "masks": not_done_masks,
                    "ext_memory": (
                        test_em.memory[:, 0]
                        if ppo_cfg.use_external_memory else None
                    ),
                    "ext_memory_masks": (
                        test_em.masks if ppo_cfg.use_external_memory else None
                    ),
                }
                valid_action_count = tta_valid_action_count
                if tta_method == "eam":
                    # AVN has one fixed, unpadded discrete action space.
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
                    if action_selection == "argmax":
                        actions = distribution.probs.argmax(
                            dim=-1, keepdim=True
                        )
                    else:
                        actions = tta_adapter.select_action(distribution)
                    tta_max_prob_sum += distribution.probs.max(dim=-1)[0].mean().item()
                    tta_probability_steps += 1
                tta_adapter.adapt(
                    source_distribution.logits,
                    action=actions,
                    features=features,
                    policy_inputs=policy_inputs,
                )
                # detach states/memory carried to the next step (cut graph)
                test_recurrent_hidden_states = test_recurrent_hidden_states.detach()
                test_em_features = test_em_features.detach()
                prev_actions.copy_(actions)

            actions = [a[0].item() for a in actions]
            if tta_adapter is not None or source_collection is not None:
                for action in actions:
                    tta_action_counts[action] += 1
            outputs = self.envs.step(actions)

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            if config.DISPLAY_RESOLUTION != model_resolution:
                obs_copy = resize_observation(observations, model_resolution)
            else:
                obs_copy = observations
            batch = batch_obs(obs_copy, self.device, skip_list=['view_point_goals', 'intermediate',
                                                                'oracle_action_sensor'])

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )
            # Update external memory
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
            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    if self.config.RL.PPO.use_belief_predictor:
                        pred = descriptor_pred_gt[i][-1]
                    else:
                        pred = None
                    if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in observations[i]:
                        for observation in observations[i]['intermediate']:
                            frame = observations_to_image(observation, infos[i], pred=pred)
                            rgb_frames[i].append(frame)
                        del observations[i]['intermediate']

                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                           self.config.DISPLAY_RESOLUTION, 3))
                    frame = observations_to_image(observations[i], infos[i], pred=pred)
                    rgb_frames[i].append(frame)
                    audios[i].append(observations[i]['audiogoal'])

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(self.envs.num_envs):
                # pause envs which runs out of episodes
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
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
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    t.update()

                    if source_collection is not None:
                        artifact_sha = source_collection.end_trajectory()
                        if artifact_sha is not None:
                            logging.info(
                                "[IDEA] wrote frozen source statistics: %s sha256=%s",
                                source_collection.output_path,
                                artifact_sha,
                            )

                    if tta_adapter is not None:
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
                        # VectorEnv has already auto-reset the sole environment.
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

                        # observations has been reset but info has not
                        # to be consistent, do not use the last frame
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
                        sound = current_episodes[i].sound_id.split('/')[1][:-4]
                        writer.add_image(f"{config.EVAL.SPLIT}_{scene}_{current_episodes[i].episode_id}_{sound}/"
                                         f"{infos[i]['spl']}",
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

        # dump stats for each episode
        if source_collection is not None and not source_collection.complete:
            raise RuntimeError(
                "IDEA source collection ended before all 128 episodes completed"
            )
        if source_collection is not None:
            if module_state_sha256(self.actor_critic) != source_model_state_sha256:
                raise RuntimeError(
                    "frozen SMT+Audio source policy changed during collection"
                )
        stats_file = os.path.join(config.TENSORBOARD_DIR,
                                  '{}_stats_{}.json'.format(config.EVAL.SPLIT, config.SEED))
        with open(stats_file, 'w') as fo:
            json.dump({','.join(key): value for key, value in stats_episodes.items()}, fo, cls=NpEncoder)

        if source_collection is not None:
            diagnostics = source_collection.diagnostics()
            diagnostics["source_policy_frozen"] = True
            diagnostics["source_model_state_sha256"] = source_model_state_sha256
            diagnostics["action_selection"] = action_selection
        elif tta_adapter is not None:
            diagnostics = tta_adapter.diagnostics()
        else:
            diagnostics = None

        if diagnostics is not None:
            diagnostics["task_action_space_contract"] = (
                "fixed_discrete_all_actions_valid"
            )
            diagnostics["task_valid_action_count"] = tta_valid_action_count
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
            diagnostics["task_action_selection"] = action_selection
            if source_collection is None:
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


def compute_distance_to_pred(pred, sim):
    from habitat.utils.geometry_utils import quaternion_rotate_vector
    import networkx as nx

    current_position = sim.get_agent_state().position
    agent_state = sim.get_agent_state()
    source_position = agent_state.position
    source_rotation = agent_state.rotation

    rounded_pred = np.round(pred)
    direction_vector_agent = np.array([rounded_pred[1], 0, -rounded_pred[0]])
    direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)
    pred_goal_location = source_position + direction_vector.astype(np.float32)
    pred_goal_location[1] = source_position[1]

    try:
        if sim.position_encoding(pred_goal_location) not in sim._position_to_index_mapping:
            pred_goal_location = sim.find_nearest_graph_node(pred_goal_location)
        distance_to_target = sim.geodesic_distance(current_position, [pred_goal_location])
    except nx.exception.NetworkXNoPath:
        distance_to_target = -1
    return distance_to_target
