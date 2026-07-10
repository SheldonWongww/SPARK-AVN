# cavn/engin/continual_ddppo_trainer.py

import os
import time
import json
import random
import signal
import sys
import atexit
import torch
import torch.nn as nn
import torch.distributed as distrib
import numpy as np
import contextlib
from typing import Dict, List, Optional
from collections import defaultdict, deque
from tqdm import tqdm
from datetime import timedelta
from torch.optim.lr_scheduler import LambdaLR


from habitat import logger, Config
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.savi.ddppo.algo.ddp_utils import (
    EXIT, REQUEUE, add_signal_handlers,
    load_interrupted_state, requeue_job, save_interrupted_state,
)
from cavn.algo.ddppo_cl import DDPPO_CL
from cavn.algo.ddppo_spark_avn import DDPPO_SparkAVN

from cavn.engin.base_ddppo_trainer import BaseDDPPOTrainer
from cavn.common.utils import batch_obs, linear_decay
from cavn.cl_method import cl_method_registry
from cavn.model.policy import AudioNavMSMTPolicyWithGD
from cavn.model.rollout_storage_multi_len import RolloutStorageMultiLen
from cavn.model.spark_avn.policy import AudioNavMSMTPolicy_SparkAVN
from cavn.model.spark_avn.prototype_bank import (
    TaskRoutingSummary,
    build_task_routing_summary,
)
from cavn.model.spark_avn.rollout_storage_spark_avn import RolloutStorageSparkAVN

try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def cleanup_gpu_memory():
    """Clean up GPU memory and resources."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if is_main_process():
            logger.info("GPU memory cleaned up")

def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    if is_main_process():
        logger.info(f"Received signal {signum}. Cleaning up...")
    cleanup_gpu_memory()
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)


def _uses_spark_avn(config) -> bool:
    try:
        return str(config.CL_METHOD.TYPE).lower() == "spark_avn"
    except AttributeError:
        return False


@baseline_registry.register_trainer(name="continual_ddppo")
class ContinualDDPPOTrainer(BaseDDPPOTrainer):
    def __init__(self, config=None):
        if _uses_spark_avn(config):
            self._delegate = _SparkAVNContinualRunner(config)
            return

        self._delegate = None
        super().__init__(config)
        self.cl_method = None
        self.performance_matrix = defaultdict(lambda: defaultdict(dict))
        self.domain_order = None
        self.seen_domains = []
        self.current_domain_id = -1
        self._static_smt_encoder = False
        self.pretrained_performance = {}
        self._ddp_initialized = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        atexit.register(cleanup_gpu_memory)

    def __getattr__(self, name):
        delegate = self.__dict__.get("_delegate")
        if delegate is not None:
            return getattr(delegate, name)
        raise AttributeError(name)

    def _setup_actor_critic_agent(self, ppo_cfg, observation_space=None):
        """Set up the actor-critic agent for continual learning."""
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

        self.agent = DDPPO_CL(
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
        
        self._ddp_initialized = False

    def _setup_cl_method(self):
        """Set up the continual learning method."""
        cl_type = self.config.CL_METHOD.TYPE.lower()
        
        method_class = cl_method_registry.get(cl_type)
        
        if method_class is None:
            available_methods = cl_method_registry.list_methods()
            raise ValueError(
                f"Unknown CL method: {cl_type}. "
                f"Available methods: {available_methods}"
            )
        
        model_to_pass = self.actor_critic
        if isinstance(model_to_pass, nn.parallel.DistributedDataParallel):
            model_to_pass = model_to_pass.module
        
        self.cl_method = method_class(
            model=model_to_pass,
            agent_optimizer=self.agent.optimizer,
            config=self.config,
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
        )
        
        if hasattr(self.agent, 'set_cl_method'):
            self.agent.set_cl_method(self.cl_method)
        
        if is_main_process():
            logger.info(f"Using continual learning method: {cl_type}")
            
    def _train_on_domain(self, domain_id: int, num_updates: int, writer):
        """Train on one domain."""
        ppo_cfg = self.config.RL.PPO
        
        self.current_domain_id = domain_id
        
        if hasattr(self.cl_method, 'set_envs'):
            self.cl_method.set_envs(self.envs)
        
        task_id = domain_id
        
        if hasattr(self.agent, 'set_task_id'):
            self.agent.set_task_id(task_id)
        
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        
        rollouts = RolloutStorageMultiLen(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.action_space,
            ppo_cfg.hidden_size,
            ppo_cfg.use_external_memory,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size + ppo_cfg.num_steps,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
            self.actor_critic.net.memory_dim if ppo_cfg.use_external_memory else None,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)
        
        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])
        
        del batch
        del observations
        torch.cuda.empty_cache()
        
        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(lambda: deque(maxlen=ppo_cfg.reward_window_size))
        
        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, num_updates)
        )
        
        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0

        cleanup_interval = 50
        
        for update in range(num_updates):
            if self.world_size > 1 and EXIT.is_set():
                self.envs.close()
                if REQUEUE.is_set() and self.world_rank == 0:
                    requeue_job()
                return
            
            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()
            
            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * linear_decay(update, num_updates)
            
            count_steps_delta = 0
            self.agent.eval()

            if hasattr(self, '_static_smt_encoder') and self._static_smt_encoder:
                self.actor_critic.net.set_eval_encoders()
            
            for step in range(ppo_cfg.num_steps):
                self.cl_method.before_collect_rollout(step)
                
                delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                    rollouts, current_episode_reward, running_episode_stats
                )
                pth_time += delta_pth_time
                env_time += delta_env_time
                count_steps_delta += delta_steps
                
                self.cl_method.after_collect_rollout(rollouts, step)
            
            with torch.no_grad():
                last_observation = {k: v[-1] for k, v in rollouts.observations.items()}
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

            self.cl_method.on_rollout_complete(rollouts)
            
            self.agent.train()
            if hasattr(self, '_static_smt_encoder') and self._static_smt_encoder:
                self.actor_critic.net.set_eval_encoders()
            
            rollouts = self.cl_method.before_update(rollouts)
            
            delta_pth_time, value_loss, action_loss, dist_entropy, cl_loss_dict = self._update_agent_with_cl(
                rollouts, task_id
            )
            pth_time += delta_pth_time
            
            self.cl_method.after_update(
                rollouts=None,
                value_loss=value_loss,
                action_loss=action_loss,
                dist_entropy=dist_entropy,
                cl_losses=cl_loss_dict,
            )

            if update % cleanup_interval == 0:
                torch.cuda.empty_cache()
            
            stats_ordering = list(sorted(running_episode_stats.keys()))
            stats = torch.stack([running_episode_stats[k] for k in stats_ordering], 0)
            
            if self.world_size > 1:
                distrib.all_reduce(stats)
            
            for i, k in enumerate(stats_ordering):
                window_episode_stats[k].append(stats[i].clone())
            
            loss_stats = torch.tensor(
                [value_loss, action_loss, dist_entropy, count_steps_delta],
                device=self.device,
            )
            if self.world_size > 1:
                distrib.all_reduce(loss_stats)
            count_steps += loss_stats[3].item()
            
            # Logging
            if self.world_rank == 0:
                num_processes = self.world_size if self.world_size > 1 else 1
                losses = [loss_stats[i].item() / num_processes for i in range(3)]
                
                deltas = {
                    k: (v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item()
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)
                
                writer.add_scalar(f"Domain_{domain_id}/reward", deltas["reward"] / deltas["count"], count_steps)
                
                metrics = {k: v / deltas["count"] for k, v in deltas.items() if k not in {"reward", "count"}}
                for metric, value in metrics.items():
                    writer.add_scalar(f"Domain_{domain_id}/{metric}", value, count_steps)
                
                writer.add_scalar(f"Domain_{domain_id}/value_loss", losses[0], count_steps)
                writer.add_scalar(f"Domain_{domain_id}/policy_loss", losses[1], count_steps)
                writer.add_scalar(f"Domain_{domain_id}/entropy_loss", losses[2], count_steps)
                
                for loss_name, loss_value in cl_loss_dict.items():
                    writer.add_scalar(f"Domain_{domain_id}/CL_{loss_name}", loss_value, count_steps)
                
                cl_metrics = self.cl_method.get_additional_metrics()
                for metric_name, metric_value in cl_metrics.items():
                    writer.add_scalar(f"CL_Metrics/{metric_name}", metric_value, count_steps)
                
                writer.add_scalar(f"Domain_{domain_id}/learning_rate", lr_scheduler.get_lr()[0], count_steps)
                
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    fps = count_steps / (time.time() - t_start)
                    
                    if is_main_process():
                        logger.info(f"update: {update}\tfps: {fps:.3f}")
                        logger.info(
                            f"update: {update}\tenv-time: {env_time:.3f}s\tpth-time: {pth_time:.3f}s\t"
                            f"frames: {count_steps:.1f}"
                        )
                        
                        metric_strs = [f"{k}: {v/deltas['count']:.3f}" for k, v in deltas.items() if k != "count"]
                        logger.info(
                            f"Average window size: {len(window_episode_stats['count'])}  {'  '.join(metric_strs)}"
                        )
                        
                        if cl_loss_dict:
                            cl_loss_strs = [f"{k}: {v:.6f}" for k, v in cl_loss_dict.items()]
                            logger.info(f"CL Losses: {' '.join(cl_loss_strs)}")

                if update > 0 and update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt_domain_{domain_id}_{count_checkpoints}.pth",
                        dict(step=count_steps, domain_id=domain_id, update=update)
                    )
                    count_checkpoints += 1
        del rollouts
        del current_episode_reward
        del running_episode_stats
        del window_episode_stats
        torch.cuda.empty_cache()

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        """Collect one rollout step."""
        pth_time = 0.0
        env_time = 0.0
        ppo_cfg = self.config.RL.PPO

        t_sample_action = time.time()
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            external_memory = None
            external_memory_masks = None
            if ppo_cfg.use_external_memory:
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

    def _copy_rollouts_for_storage(self, rollouts):
        """Create a lightweight rollout copy for method-specific storage."""
        from types import SimpleNamespace
        
        rollouts_copy = SimpleNamespace()
        
        rollouts_copy.observations = {}
        for sensor in rollouts.observations:
            rollouts_copy.observations[sensor] = rollouts.observations[sensor].clone()
        
        rollouts_copy.actions = rollouts.actions.clone()
        rollouts_copy.action_log_probs = rollouts.action_log_probs.clone()
        rollouts_copy.value_preds = rollouts.value_preds.clone()
        rollouts_copy.rewards = rollouts.rewards.clone()
        rollouts_copy.masks = rollouts.masks.clone()
        rollouts_copy.recurrent_hidden_states = rollouts.recurrent_hidden_states.clone()
        rollouts_copy.prev_actions = rollouts.prev_actions.clone()
        rollouts_copy.returns = rollouts.returns.clone()
        
        rollouts_copy.step = rollouts.step
        
        return rollouts_copy
    
    def _get_domain_order(self, seed: int) -> List[int]:
        """Generate a random order of domains based on seed."""
        np.random.seed(seed)
        domain_ids = list(range(20))
        np.random.shuffle(domain_ids)
        return domain_ids
    
    def _get_domain_scenes(self, domain_id: int) -> List[str]:
        """Get the scene list for a specific domain."""
        domain_scenes_mapping = {
            0: ['q9vSo1VnCiC'], 1: ['ac26ZMwG7aT'], 2: ['PuKPg4mmafe'],
            3: ['r47D5H71a5s'], 4: ['jh4fc5c5qoQ'], 5: ['S9hNv5qa7GM'],
            6: ['QUCTc6BB5sX'], 7: ['VLzqgDo317F'], 8: ['5ZKStnWn8Zo'],
            9: ['EDJbREhghzL'], 10: ['qoiz87JEwZ2'], 11: ['VVfe2KiqLaN'],
            12: ['vyrNrziPKCB'], 13: ['b8cTxDM8gDG'], 14: ['e9zR4mvMWw7'],
            15: ['kEZ7cmS4wCh'], 16: ['V2XKFyX4ASd'], 17: ['gTV8FGcVJC9'],
            18: ['oLBMNvg9in8'], 19: ['gYvKGZ5eRqb']
        }
        return domain_scenes_mapping.get(domain_id, [])
    
    def _save_training_state(self, domain_id: int):
        """Save continual-training progress."""
        state_file = os.path.join(self.config.MODEL_DIR, "cl_training_state.json")
        
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
        else:
            state = {
                "domain_order": self.domain_order,
                "trained_domains": [],
                "seed": self.config.SEED
            }
        
        if domain_id not in state["trained_domains"]:
            state["trained_domains"].append(domain_id)
        
        state["last_checkpoint"] = os.path.join(
            self.config.CHECKPOINT_FOLDER,
            f"ckpt_domain_{domain_id}.pth"
        )
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_training_state(self) -> Dict:
        """Load continual-training progress."""
        state_file = os.path.join(self.config.MODEL_DIR, "cl_training_state.json")
        
        if not os.path.exists(state_file):
            return None
        
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        if is_main_process():
            logger.info(f"Loaded training state from {state_file}")
            logger.info(f"Trained domains: {state['trained_domains']}")
        
        return state
    
    def _get_next_domain_to_train(self) -> Optional[int]:
        """Return the next untrained domain."""
        state = self._load_training_state()
        
        if state is None:
            domain_order = self._get_domain_order(self.config.SEED)
            return domain_order[0]
        
        expected_order = self._get_domain_order(self.config.SEED)
        trained_domains = state["trained_domains"]
        
        for domain_id in expected_order:
            if domain_id not in trained_domains:
                return domain_id
        
        return None

    def train(self) -> None:
        """Train one domain per invocation."""
        if self._delegate is not None:
            return self._delegate.train()

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if not distrib.is_initialized() and self.world_size > 1:
            distrib.init_process_group(
                backend=self.config.RL.DDPPO.distrib_backend, 
                init_method="env://", 
                timeout=timedelta(minutes=20)
            )
        
        if self.world_size > 1:
            add_signal_handlers()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        self.config.TASK_CONFIG.SEED += (self.world_rank * self.config.NUM_PROCESSES)
        self.config.freeze()

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        next_domain_id = self._get_next_domain_to_train()
        
        if next_domain_id is None:
            if is_main_process():
                logger.info("All domains have been trained!")
            return
        
        self.current_domain_id = next_domain_id
        domain_order = self._get_domain_order(self.config.SEED)
        self.domain_order = domain_order
        
        cl_type = self.config.CL_METHOD.TYPE.lower() if hasattr(self.config, 'CL_METHOD') else 'finetune'
        
        domain_base_seed = self.config.SEED + next_domain_id * 1000
        random.seed(domain_base_seed + self.world_rank)
        np.random.seed(domain_base_seed + self.world_rank)
        torch.manual_seed(domain_base_seed + self.world_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(domain_base_seed + self.world_rank)
        
        if is_main_process():
            logger.info(f"=" * 80)
            logger.info(f"Domain order: {domain_order}")
            logger.info(f"Training Domain {next_domain_id} ({domain_order.index(next_domain_id) + 1}/{len(domain_order)})")
            logger.info(f"CL Method: {cl_type}")
            logger.info(f"=" * 80)
        
        state = self._load_training_state()
        if state is not None:
            self.seen_domains = state["trained_domains"]
        else:
            self.seen_domains = []
        
        self.config.defrost()
        domain_scenes = self._get_domain_scenes(next_domain_id)
        if len(domain_scenes) == 1 and self.config.NUM_PROCESSES > 1:
            domain_scenes = domain_scenes * self.config.NUM_PROCESSES
        self.config.TASK_CONFIG.DATASET.CONTENT_SCENES = domain_scenes
        self.config.freeze()
        
        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        
        ppo_cfg = self.config.RL.PPO
        
        cl_state_to_restore = None
        
        self._setup_actor_critic_agent(ppo_cfg)

        if state is None or len(self.seen_domains) == 0:
            pretrain_path = ppo_cfg.SCENE_MEMORY_TRANSFORMER.actor_critic_pretrained_path
            if pretrain_path != "-1" and os.path.exists(pretrain_path):
                if is_main_process():
                    logger.info(f"Loading pretrained weights from {pretrain_path}")
                ckpt_dict = self.load_checkpoint(pretrain_path, map_location="cpu")
                self.agent.load_state_dict(ckpt_dict["state_dict"])
                self.actor_critic = self.agent.actor_critic
                del ckpt_dict
                torch.cuda.empty_cache()
        else:
            last_checkpoint = state["last_checkpoint"]
            if os.path.exists(last_checkpoint):
                if is_main_process():
                    logger.info(f"Loading weights from: {last_checkpoint}")
                ckpt_dict = self.load_checkpoint(last_checkpoint, map_location="cpu")
                self.agent.load_state_dict(ckpt_dict["state_dict"])
                self.actor_critic = self.agent.actor_critic
                cl_state_to_restore = ckpt_dict.get("cl_method_state", None)
                del ckpt_dict
                torch.cuda.empty_cache()
            else:
                raise FileNotFoundError(f"Checkpoint not found: {last_checkpoint}")
        
        self.actor_critic = self.actor_critic.to(self.device)
        self.agent.actor_critic = self.actor_critic
        
        self._setup_cl_method()

        if cl_state_to_restore is not None and self.cl_method is not None:
            if is_main_process():
                logger.info("Restoring CL method state...")
            try:
                self.cl_method.load_state_dict(cl_state_to_restore)
                if is_main_process():
                    logger.info("CL method state restored.")
            except Exception as e:
                logger.warning(f"Failed to restore CL method state: {e}")
        
        self.seen_domains.append(next_domain_id)
        self.cl_method.on_task_start(next_domain_id)
        
        if self.world_size > 1 and not self._ddp_initialized:
            self.agent.init_distributed(find_unused_params=True)
            self._ddp_initialized = True
            self.actor_critic = self.agent.actor_critic
        
        if self.world_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        
        if is_main_process():
            logger.info(f"Trainable parameters: {sum(p.numel() for p in self.agent.parameters() if p.requires_grad)}")
        
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=30
        ) if self.world_rank == 0 else contextlib.nullcontext() as writer:
            
            current_updates = self.config.NUM_UPDATES_PER_DOMAIN
            
            self._train_on_domain(
                domain_id=next_domain_id,
                num_updates=current_updates,
                writer=writer
            )
            
            self.cl_method.on_task_end(next_domain_id)
            
            if self.world_rank == 0:
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_FOLDER,
                    f"ckpt_domain_{next_domain_id}.pth"
                )
                self.save_checkpoint(
                    f"ckpt_domain_{next_domain_id}.pth",
                    {"domain_id": next_domain_id, "trained_domains": self.seen_domains}
                )
                if is_main_process():
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
                
                self._save_training_state(next_domain_id)
            
            if self.world_size > 1:
                distrib.barrier()
        
        self.envs.close()
        
        if is_main_process():
            logger.info(f"Domain {next_domain_id} training completed!")

    def _save_eval_state(self, domain_id: int):
        """Save continual-evaluation progress."""
        state_file = os.path.join(self.config.MODEL_DIR, "cl_eval_state.json")
        
        state = {
            "domain_order": self.domain_order,
            "last_evaluated_domain": domain_id,
            "performance_matrix": {},
            "seed": self.config.SEED
        }
        
        for train_d, eval_dict in self.performance_matrix.items():
            state["performance_matrix"][str(train_d)] = {}
            for eval_d, metrics in eval_dict.items():
                state["performance_matrix"][str(train_d)][str(eval_d)] = metrics
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        if is_main_process():
            logger.info(f"Saved evaluation state to {state_file}")
    
    def _load_pretrained_performance(self):
        """Load pretrained-domain performance for transfer metrics."""
        pretrained_perf_path = os.path.join(
            "data",
            f"pretrained_performance_{'single' if not self.config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND else 'multi'}_source.json"
        )
        
        if os.path.exists(pretrained_perf_path):
            with open(pretrained_perf_path, 'r') as f:
                loaded_perf = json.load(f)
                self.pretrained_performance = {int(k): v for k, v in loaded_perf.items()}
            if is_main_process():
                logger.info(f"Loaded pretrained performance from {pretrained_perf_path}")
        else:
            if is_main_process():
                logger.warning(f"Pretrained performance file not found: {pretrained_perf_path}")
    
    def _compute_and_log_cl_metrics(self, domain_id: int, writer):
        """Compute and log continual-learning metrics."""
        metrics = self._compute_cl_metrics()
        
        if is_main_process():
            logger.info(f"=" * 80)
            logger.info(f"CL Metrics after Domain {domain_id}:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
                writer.add_scalar(f"CL_Metrics/{metric_name}", metric_value, domain_id)
            logger.info(f"=" * 80)

    def _update_agent_with_cl(self, rollouts, task_order):
        """Update the agent with CL-specific hooks."""
        t_update_model = time.time()
        
        value_loss, action_loss, dist_entropy, cl_loss_dict = self.agent.update_with_cl(
            rollouts,
            task_id=task_order,
        )
        rollouts.after_update()
        
        return (
            time.time() - t_update_model, 
            value_loss, 
            action_loss, 
            dist_entropy,
            cl_loss_dict
        )

    def _compute_cl_metrics(self) -> Dict[str, float]:
        """Compute AP, forgetting, forward transfer, and backward transfer."""
        if len(self.performance_matrix) < 2:
            return {}
        
        metrics = {}
        train_domains = sorted(self.performance_matrix.keys())
        N = len(train_domains)
        last_domain = train_domains[-1]
        
        metric_names = set()
        for eval_dict in self.performance_matrix.values():
            for domain_metrics in eval_dict.values():
                metric_names.update(domain_metrics.keys())
        
        for metric_name in metric_names:
            ap_values = []
            for eval_domain in train_domains:
                if eval_domain in self.performance_matrix[last_domain]:
                    perf = self.performance_matrix[last_domain][eval_domain].get(metric_name)
                    if perf is not None:
                        ap_values.append(perf)
            
            if ap_values:
                metrics[f"A_N_{metric_name}"] = np.mean(ap_values)
            
            fg_values = []
            for eval_domain in train_domains[:-1]:
                max_perf = float('-inf')
                for train_domain in train_domains[:-1]:
                    if eval_domain in self.performance_matrix.get(train_domain, {}):
                        perf = self.performance_matrix[train_domain][eval_domain].get(metric_name)
                        if perf is not None:
                            max_perf = max(max_perf, perf)
                
                if eval_domain in self.performance_matrix[last_domain]:
                    final_perf = self.performance_matrix[last_domain][eval_domain].get(metric_name)
                    if final_perf is not None and max_perf > float('-inf'):
                        forgetting = max(max_perf - final_perf, 0)
                        fg_values.append(forgetting)
            
            if fg_values:
                metrics[f"FG_{metric_name}"] = np.mean(fg_values)
            
            ft_values = []
            for i in range(1, N):
                current_domain = train_domains[i]
                prev_domain = train_domains[i - 1]
                
                if current_domain in self.performance_matrix.get(prev_domain, {}):
                    prev_perf = self.performance_matrix[prev_domain][current_domain].get(metric_name)
                    
                    if (prev_perf is not None and 
                        current_domain in self.pretrained_performance and
                        metric_name in self.pretrained_performance[current_domain]):
                        pretrain_perf = self.pretrained_performance[current_domain][metric_name]
                        transfer = prev_perf - pretrain_perf
                        ft_values.append(transfer)
            
            if ft_values:
                metrics[f"FT_{metric_name}"] = np.mean(ft_values)
            
            bt_values = []
            for train_domain in train_domains[:-1]:
                if train_domain in self.performance_matrix.get(train_domain, {}):
                    initial_perf = self.performance_matrix[train_domain][train_domain].get(metric_name)
                    
                    if (initial_perf is not None and 
                        train_domain in self.performance_matrix[last_domain]):
                        final_perf = self.performance_matrix[last_domain][train_domain].get(metric_name)
                        if final_perf is not None:
                            transfer = final_perf - initial_perf
                            bt_values.append(transfer)
            
            if bt_values:
                metrics[f"BT_{metric_name}"] = np.mean(bt_values)
        
        return metrics
    
    def _compute_incremental_ap(self, domain_id: int) -> Dict[str, float]:
        """Compute average performance after the current domain."""
        if domain_id not in self.performance_matrix:
            return {}
        
        train_domains = sorted(self.performance_matrix.keys())
        if domain_id not in train_domains:
            return {}
        
        idx = train_domains.index(domain_id)
        i = idx + 1
        
        seen_domains = train_domains[:i]
        
        incremental_ap = {}
        metric_names = set()
        
        for domain_metrics in self.performance_matrix[domain_id].values():
            metric_names.update(domain_metrics.keys())
        
        for metric_name in metric_names:
            ap_values = []
            for eval_domain in seen_domains:
                if eval_domain in self.performance_matrix[domain_id]:
                    perf = self.performance_matrix[domain_id][eval_domain].get(metric_name)
                    if perf is not None:
                        ap_values.append(perf)
            
            if ap_values:
                incremental_ap[f"A_{i}_{metric_name}"] = np.mean(ap_values)
        
        return incremental_ap
    
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
        domain_id: Optional[int] = None,
    ) -> Dict:
        """Evaluate one checkpoint."""
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()
            
        ppo_cfg = config.RL.PPO
        
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        
        if domain_id is not None:
            domain_scenes = self._get_domain_scenes(domain_id)
            if len(domain_scenes) == 1 and config.NUM_PROCESSES > 1:
                domain_scenes = domain_scenes * config.NUM_PROCESSES
            config.TASK_CONFIG.DATASET.CONTENT_SCENES = domain_scenes
        
        config.freeze()

        if is_main_process():
            logger.info(f"Evaluating checkpoint: {checkpoint_path}")
            logger.info(f"Domain: {domain_id}, Scenes: {config.TASK_CONFIG.DATASET.CONTENT_SCENES}")
        
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        
        observation_space = self.envs.observation_spaces[0]
        self._setup_actor_critic_agent(ppo_cfg, observation_space)
        
        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        
        del ckpt_dict
        torch.cuda.empty_cache()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            1, self.config.NUM_PROCESSES, ppo_cfg.hidden_size, device=self.device
        )

        if ppo_cfg.use_external_memory:
            from cavn.model.rollout_storage_multi_len import ExternalMemoryMultiLen
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

        prev_actions = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long)
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes = dict()

        self.actor_critic.eval()
        
        t = tqdm(total=self.config.TEST_EPISODE_COUNT)
        while len(stats_episodes) < self.config.TEST_EPISODE_COUNT and self.envs.num_envs > 0:
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
                _, actions, _, test_recurrent_hidden_states, test_em_features = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    em_memory if ppo_cfg.use_external_memory else None,
                    em_masks if ppo_cfg.use_external_memory else None,
                    deterministic=False
                )

                prev_actions.copy_(actions)

            actions = [a[0].item() for a in actions]
            outputs = self.envs.step(actions)

            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float, device=self.device,
            )
            
            if ppo_cfg.use_external_memory:
                test_em.insert(test_em_features, not_done_masks)

            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            
            for i in range(self.envs.num_envs):
                if (next_episodes[i].scene_id, next_episodes[i].episode_id) in stats_episodes:
                    envs_to_pause.append(i)

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["success"] = infos[i]["success"]
                    episode_stats["spl"] = infos[i]["spl"]
                    episode_stats["distance_to_goal"] = infos[i]["distance_to_goal"]
                    episode_stats["normalized_distance_to_goal"] = infos[i]["normalized_distance_to_goal"]
                    episode_stats["num_action"] = infos[i].get("num_action", infos[i].get("na", 0))
                    episode_stats["success_weighted_by_num_action"] = infos[i].get(
                        "success_weighted_by_num_action", infos[i].get("sna", 0)
                    )
                    episode_stats["success_when_silent"] = infos[i].get(
                        "success_when_silent", infos[i].get("sws", 0)
                    )
                    
                    current_episode_reward[i] = 0
                    
                    stats_episodes[(current_episodes[i].scene_id, current_episodes[i].episode_id)] = episode_stats
                    t.update()

            rgb_frames = [None] * self.envs.num_envs
            
            (
                self.envs, test_recurrent_hidden_states, not_done_masks, test_em,
                current_episode_reward, prev_actions, batch, rgb_frames,
            ) = self._pause_envs(
                envs_to_pause, self.envs, test_recurrent_hidden_states, not_done_masks,
                current_episode_reward, prev_actions, batch, rgb_frames, test_em, None
            )

        aggregated_stats = dict()
        for stat_key in ["reward", "success", "spl", "distance_to_goal",
                        "normalized_distance_to_goal", "num_action",
                        "success_weighted_by_num_action", "success_when_silent"]:
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])
        
        num_episodes = len(stats_episodes)
        
        for stat_key, stat_val in aggregated_stats.items():
            if is_main_process():
                logger.info(f"Average {stat_key}: {stat_val:.4f} over {num_episodes} episodes")

        self.envs.close()

        return aggregated_stats
    
    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        """Build the evaluation config from a checkpoint config."""
        config = self.config.clone()

        config.defrost()
        checkpoint_config.defrost()
        
        checkpoint_content_scenes = None
        if hasattr(checkpoint_config.TASK_CONFIG.DATASET, 'CONTENT_SCENES'):
            checkpoint_content_scenes = checkpoint_config.TASK_CONFIG.DATASET.CONTENT_SCENES
            checkpoint_config.TASK_CONFIG.DATASET.CONTENT_SCENES = []
        
        for cfg in [config, checkpoint_config]:
            if (hasattr(cfg.RL.PPO, 'SCENE_MEMORY_TRANSFORMER') and
                hasattr(cfg.RL.PPO.SCENE_MEMORY_TRANSFORMER, 'actor_critic_pretrained_path')):
                current_value = cfg.RL.PPO.SCENE_MEMORY_TRANSFORMER.actor_critic_pretrained_path
                if not isinstance(current_value, str):
                    cfg.RL.PPO.SCENE_MEMORY_TRANSFORMER.actor_critic_pretrained_path = str(current_value) if current_value != -1 else "-1"
        
        config.freeze()
        checkpoint_config.freeze()

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
        except KeyError:
            logger.warning("Saved config is outdated, using eval config only")
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.CONTENT_SCENES = []
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        if checkpoint_content_scenes is not None:
            checkpoint_config.defrost()
            checkpoint_config.TASK_CONFIG.DATASET.CONTENT_SCENES = checkpoint_content_scenes
            checkpoint_config.freeze()

        return config
    
    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        """Load a checkpoint and normalize legacy config values."""
        ckpt_dict = super().load_checkpoint(checkpoint_path, *args, **kwargs)
        
        if "config" in ckpt_dict:
            config = ckpt_dict["config"]
            if hasattr(config, "RL") and hasattr(config.RL, "PPO"):
                if hasattr(config.RL.PPO, "SCENE_MEMORY_TRANSFORMER"):
                    smt_cfg = config.RL.PPO.SCENE_MEMORY_TRANSFORMER
                    if hasattr(smt_cfg, "actor_critic_pretrained_path"):
                        current_value = smt_cfg.actor_critic_pretrained_path
                        
                        config.defrost()
                        if isinstance(current_value, int):
                            config.RL.PPO.SCENE_MEMORY_TRANSFORMER.actor_critic_pretrained_path = (
                                "-1" if current_value == -1 else str(current_value)
                            )
                        elif not isinstance(current_value, str):
                            config.RL.PPO.SCENE_MEMORY_TRANSFORMER.actor_critic_pretrained_path = str(current_value)
                        config.freeze()
        
        return ckpt_dict

    def _should_evaluate_domain_pair(self, train_domain: int, eval_domain: int) -> bool:
        """Return whether a train/eval domain pair still needs evaluation."""
        if train_domain in self.performance_matrix:
            if eval_domain in self.performance_matrix[train_domain]:
                return False
        return True

    def evaluate_continual(self, eval_mode: str = "final") -> None:
        """Evaluate continual checkpoints while reusing each loaded checkpoint."""
        if self._delegate is not None:
            return self._delegate.evaluate_continual(eval_mode=eval_mode)

        self.local_rank = 0
        self.world_rank = 0
        self.world_size = 1
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
        
        state = self._load_training_state()
        if state is None:
            logger.error("No training state found. Please train first.")
            return
        
        self.domain_order = state["domain_order"]
        trained_domains = state["trained_domains"]
        
        if len(trained_domains) == 0:
            logger.error("No domains have been trained yet.")
            return
        
        self._load_pretrained_performance()
        
        eval_state_file = os.path.join(self.config.MODEL_DIR, "cl_eval_state.json")
        if os.path.exists(eval_state_file):
            with open(eval_state_file, 'r') as f:
                eval_state = json.load(f)
                for train_d, eval_dict in eval_state.get("performance_matrix", {}).items():
                    for eval_d, metrics in eval_dict.items():
                        self.performance_matrix[int(train_d)][int(eval_d)] = metrics
            if is_main_process():
                logger.info(f"Loaded existing evaluation state from {eval_state_file}")
        
        logger.info(f"=" * 80)
        logger.info(f"Starting Continual Learning Evaluation")
        logger.info(f"Trained domains: {trained_domains}")
        logger.info(f"=" * 80)
        
        eval_tasks = []  # (train_domain_idx, train_domain, eval_domain)
        
        for idx, train_domain in enumerate(self.domain_order):
            if train_domain not in trained_domains:
                continue
            
            eval_domains = self.domain_order[:idx+1]
            if idx + 1 < len(self.domain_order):
                next_domain = self.domain_order[idx + 1]
                if next_domain in trained_domains:
                    eval_domains.append(next_domain)
            
            for eval_domain in eval_domains:
                if self._should_evaluate_domain_pair(train_domain, eval_domain):
                    eval_tasks.append((idx, train_domain, eval_domain))
        
        if not eval_tasks:
            logger.info("All evaluations already completed!")
            return
        
        logger.info(f"Total evaluation tasks: {len(eval_tasks)}")
        
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=30
        ) as writer:
            
            from collections import defaultdict
            tasks_by_train_domain = defaultdict(list)
            for idx, train_domain, eval_domain in eval_tasks:
                tasks_by_train_domain[train_domain].append((idx, eval_domain))
            
            for train_domain in self.domain_order:
                if train_domain not in tasks_by_train_domain:
                    continue
                
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_FOLDER,
                    f"ckpt_domain_{train_domain}.pth"
                )
                
                if not os.path.exists(checkpoint_path):
                    logger.warning(f"Checkpoint not found: {checkpoint_path}")
                    continue
                
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Processing checkpoint for Domain {train_domain}")
                logger.info(f"{'=' * 80}\n")
                
                logger.info(f"Loading checkpoint: {checkpoint_path}")
                ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
                
                eval_domains = [eval_d for _, eval_d in tasks_by_train_domain[train_domain]]
                
                for eval_domain in eval_domains:
                    logger.info(f">>> Evaluating on domain {eval_domain}")
                    
                    eval_stats = self._eval_checkpoint_on_domain(
                        ckpt_dict=ckpt_dict,
                        domain_id=eval_domain,
                    )
                    
                    self.performance_matrix[train_domain][eval_domain] = eval_stats
                    
                    logger.info(f">>> Domain {eval_domain}: Success={eval_stats['success']:.4f}, SPL={eval_stats['spl']:.4f}")
                    
                    self._save_eval_state(train_domain)
                
                del ckpt_dict
                torch.cuda.empty_cache()
                
                idx = self.domain_order.index(train_domain)
                if idx >= 1:
                    self._compute_and_log_cl_metrics(train_domain, writer)
                
                incremental_ap = self._compute_incremental_ap(train_domain)
                if incremental_ap and is_main_process():
                    logger.info(f"Incremental AP after domain {train_domain}:")
                    for metric_name, value in incremental_ap.items():
                        logger.info(f"  {metric_name}: {value:.4f}")
                        writer.add_scalar(f"CL_Metrics/{metric_name}", value, idx + 1)
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluation completed!")
        logger.info(f"{'=' * 80}\n")

    def _eval_checkpoint_on_domain(
        self,
        ckpt_dict: Dict,
        domain_id: Optional[int] = None,
    ) -> Dict:
        """Evaluate a checkpoint on one domain."""
        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()
            
        ppo_cfg = config.RL.PPO
        
        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        
        if domain_id is not None:
            domain_scenes = self._get_domain_scenes(domain_id)
            if len(domain_scenes) == 1 and config.NUM_PROCESSES > 1:
                domain_scenes = domain_scenes * config.NUM_PROCESSES
            config.TASK_CONFIG.DATASET.CONTENT_SCENES = domain_scenes
        
        config.freeze()

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        observation_space = self.envs.observation_spaces[0]
        
        if not hasattr(self, 'actor_critic') or self.actor_critic is None:
            self._setup_actor_critic_agent(ppo_cfg, observation_space)

        self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
        self.actor_critic = self.agent.actor_critic
        
        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            1, config.NUM_PROCESSES, ppo_cfg.hidden_size, device=self.device
        )

        if ppo_cfg.use_external_memory:
            from cavn.model.rollout_storage_multi_len import ExternalMemoryMultiLen
            test_em = ExternalMemoryMultiLen(
                config.NUM_PROCESSES,
                ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
                self.actor_critic.net.memory_dim,
                is_mapping=False,
            )
            test_em.to(self.device)
        else:
            test_em = None

        prev_actions = torch.zeros(config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long)
        not_done_masks = torch.zeros(config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes = dict()

        self.actor_critic.eval()

        pbar = tqdm(total=config.TEST_EPISODE_COUNT, desc=f"Domain {domain_id}", leave=False)
        
        while len(stats_episodes) < config.TEST_EPISODE_COUNT and self.envs.num_envs > 0:
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
                _, actions, _, test_recurrent_hidden_states, test_em_features = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    em_memory if ppo_cfg.use_external_memory else None,
                    em_masks if ppo_cfg.use_external_memory else None,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            actions = [a[0].item() for a in actions]
            outputs = self.envs.step(actions)

            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float, device=self.device,
            )
            
            if ppo_cfg.use_external_memory:
                test_em.insert(test_em_features, not_done_masks)

            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            
            for i in range(self.envs.num_envs):
                if (next_episodes[i].scene_id, next_episodes[i].episode_id) in stats_episodes:
                    envs_to_pause.append(i)

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["success"] = infos[i]["success"]
                    episode_stats["spl"] = infos[i]["spl"]
                    episode_stats["distance_to_goal"] = infos[i]["distance_to_goal"]
                    episode_stats["normalized_distance_to_goal"] = infos[i]["normalized_distance_to_goal"]
                    episode_stats["num_action"] = infos[i].get("num_action", infos[i].get("na", 0))
                    episode_stats["success_weighted_by_num_action"] = infos[i].get(
                        "success_weighted_by_num_action", infos[i].get("sna", 0)
                    )
                    episode_stats["success_when_silent"] = infos[i].get(
                        "success_when_silent", infos[i].get("sws", 0)
                    )
                    
                    current_episode_reward[i] = 0
                    
                    stats_episodes[(current_episodes[i].scene_id, current_episodes[i].episode_id)] = episode_stats
                    pbar.update(1)

            rgb_frames = [None] * self.envs.num_envs
            
            (
                self.envs, test_recurrent_hidden_states, not_done_masks, test_em,
                current_episode_reward, prev_actions, batch, rgb_frames,
            ) = self._pause_envs(
                envs_to_pause, self.envs, test_recurrent_hidden_states, not_done_masks,
                current_episode_reward, prev_actions, batch, rgb_frames, test_em, None
            )

        pbar.close()

        aggregated_stats = dict()
        for stat_key in ["reward", "success", "spl", "distance_to_goal",
                        "normalized_distance_to_goal", "num_action",
                        "success_weighted_by_num_action", "success_when_silent"]:
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])
        
        self.envs.close()
        
        del observations, batch, test_recurrent_hidden_states
        if test_em is not None:
            del test_em
        torch.cuda.empty_cache()

        return aggregated_stats

    def _get_pretrained_performance_path(self) -> str:
        """Return the pretrained-performance cache path."""
        is_multi_source = self.config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND
        suffix = "multi_source" if is_multi_source else "single_source"
        return os.path.join(self.config.MODEL_DIR, f"pretrained_performance_{suffix}.json")

    def _load_existing_pretrained_performance(self) -> Dict[int, Dict]:
        """Load cached pretrained-domain performance."""
        perf_path = self._get_pretrained_performance_path()
        if os.path.exists(perf_path):
            with open(perf_path, 'r') as f:
                loaded = json.load(f)
                return {int(k): v for k, v in loaded.items()}
        return {}

    def _save_pretrained_performance_incremental(self, domain_id: int, eval_stats: Dict):
        """Save pretrained-domain performance after each evaluated domain."""
        perf_path = self._get_pretrained_performance_path()
        
        pretrained_performance = self._load_existing_pretrained_performance()
        pretrained_performance[domain_id] = eval_stats
        
        os.makedirs(os.path.dirname(perf_path), exist_ok=True)
        
        temp_path = perf_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(pretrained_performance, f, indent=2)
        
        # Atomic replace keeps the cache valid if evaluation is interrupted.
        os.replace(temp_path, perf_path)
        
        if is_main_process():
            logger.info(f"Saved domain {domain_id} performance to: {perf_path}")
            logger.info(f"Total evaluated domains: {len(pretrained_performance)}/20")

    def eval_pretrained_on_all_domains(self, reverse_order: bool = False) -> None:
        """Evaluate the pretrained model with resumable per-domain caching."""
        if self._delegate is not None:
            raise RuntimeError("eval-pretrained is only supported by the finetune continual runner.")

        self.local_rank = 0
        self.world_rank = 0
        self.world_size = 1
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
        
        pretrained_performance = self._load_existing_pretrained_performance()
        evaluated_domains = set(pretrained_performance.keys())
        
        domain_order = list(range(20))
        if reverse_order:
            domain_order = list(reversed(domain_order))
        
        domains_to_eval = [d for d in domain_order if d not in evaluated_domains]
        
        if is_main_process():
            logger.info(f"Evaluating pretrained model on all domains")
            logger.info(f"Domain order: {domain_order}")
            if evaluated_domains:
                logger.info(f"Already evaluated domains: {sorted(evaluated_domains)}")
                logger.info(f"Remaining domains: {domains_to_eval}")
            logger.info(f"Total domains to evaluate: {len(domains_to_eval)}/20")
        
        if not domains_to_eval:
            if is_main_process():
                logger.info("All domains have been evaluated!")
                self._print_pretrained_statistics(pretrained_performance)
            return
        
        ppo_cfg = self.config.RL.PPO
        
        pretrain_path = ppo_cfg.SCENE_MEMORY_TRANSFORMER.actor_critic_pretrained_path
        if pretrain_path == "-1" or not os.path.exists(pretrain_path):
            logger.error(f"Pretrained model not found: {pretrain_path}")
            return
        
        if is_main_process():
            logger.info(f"Loading pretrained model from: {pretrain_path}")
        
        ckpt_dict = self.load_checkpoint(pretrain_path, map_location="cpu")
        
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=30
        ):
            
            for domain_id in domains_to_eval:
                if is_main_process():
                    logger.info(f"\n{'='*80}")
                    logger.info(f"Evaluating Domain {domain_id} ({domain_order.index(domain_id)+1}/{len(domain_order)})")
                    logger.info(f"{'='*80}\n")
                
                try:
                    eval_stats = self._eval_checkpoint_on_domain(
                        ckpt_dict=ckpt_dict,
                        domain_id=domain_id,
                    )
                    
                    if is_main_process():
                        self._save_pretrained_performance_incremental(domain_id, eval_stats)
                        logger.info(f"Domain {domain_id}: Success={eval_stats['success']:.4f}, SPL={eval_stats['spl']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating domain {domain_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if is_main_process():
            final_performance = self._load_existing_pretrained_performance()
            self._print_pretrained_statistics(final_performance)
        
        del ckpt_dict
        if hasattr(self, 'envs') and self.envs is not None:
            self.envs.close()
        torch.cuda.empty_cache()

    def _print_pretrained_statistics(self, pretrained_performance: Dict[int, Dict]):
        """Log aggregate pretrained-domain performance."""
        if not pretrained_performance:
            return
        
        perf_path = self._get_pretrained_performance_path()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Pretrained Performance Summary")
        logger.info(f"Saved to: {perf_path}")
        logger.info(f"{'='*80}")
        
        successes = [v['success'] for v in pretrained_performance.values()]
        spls = [v['spl'] for v in pretrained_performance.values()]
        
        logger.info(f"Evaluated domains: {len(pretrained_performance)}/20")
        logger.info(f"Average Success: {np.mean(successes):.4f} +/- {np.std(successes):.4f}")
        logger.info(f"Average SPL: {np.mean(spls):.4f} +/- {np.std(spls):.4f}")
        logger.info(f"{'='*80}\n")


class _SparkAVNContinualRunner(BaseDDPPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.cl_method = None
        self.domain_order = None
        self.current_domain_id = -1
        self.local_rank = 0
        self.world_rank = 0
        self.world_size = 1

    def _get_eval_result_filename(self) -> str:
        routing_mode = str(self.config.CL_METHOD.SPARK_AVN.EVAL_ROUTING_MODE).lower()
        safe_mode = routing_mode.replace("-", "_").replace(" ", "_")
        return f"spark_avn_eval_results_{safe_mode}.json"

    def _get_matrix_state_filename(self) -> str:
        tag = ""
        try:
            tag_val = self.config.CL_METHOD.SPARK_AVN.EVAL_TAG
            tag = str(tag_val).strip() if tag_val is not None else ""
        except AttributeError:
            tag = ""
        if tag:
            safe_tag = tag.replace("-", "_").replace(" ", "_")
            return f"cl_eval_state_{safe_tag}.json"
        return "cl_eval_state.json"

    def _get_policy_model(self):
        if isinstance(self.actor_critic, torch.nn.parallel.DistributedDataParallel):
            return self.actor_critic.module
        return self.actor_critic

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        logger.add_filehandler(self.config.LOG_FILE)
        action_space = self.envs.action_spaces[0]
        self.action_space = action_space

        smt_cfg = ppo_cfg.SCENE_MEMORY_TRANSFORMER
        belief_cfg = ppo_cfg.BELIEF_PREDICTOR
        seld_cfg = ppo_cfg.SELD_ENCODER
        spark_cfg = self.config.CL_METHOD.SPARK_AVN

        self.actor_critic = AudioNavMSMTPolicy_SparkAVN(
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
            actor_critic_pretrained_path=smt_cfg.actor_critic_pretrained_path,
            pool_size=spark_cfg.POOL_SIZE,
            router_ema_momentum=spark_cfg.ROUTER_EMA_MOMENTUM,
            router_temperature=spark_cfg.ROUTER_TEMPERATURE,
            eval_routing_mode=spark_cfg.EVAL_ROUTING_MODE,
            prototype_bank_size=spark_cfg.PROTOTYPE_BANK_SIZE,
            eval_warmup_steps=spark_cfg.EVAL_WARMUP_STEPS,
            max_kmeans_iters=spark_cfg.MAX_KMEANS_ITERS,
            merge_w_value=spark_cfg.MERGE_W_VALUE,
            merge_w_hhi=spark_cfg.MERGE_W_HHI,
        )

        self.actor_critic.to(self.device)
        self.agent = DDPPO_SparkAVN(
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

    def _setup_cl_method(self):
        method_class = cl_method_registry.get("spark_avn")
        if method_class is None:
            raise ValueError("CL method spark_avn is not registered")

        model_to_pass = self._get_policy_model()
        self.cl_method = method_class(
            model=model_to_pass,
            agent_optimizer=self.agent.optimizer,
            config=self.config,
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
        )

    def _training_state_path(self) -> str:
        return os.path.join(self.config.MODEL_DIR, "cl_training_state.json")

    def _load_training_state(self) -> Optional[Dict]:
        path = self._training_state_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _save_training_state(self, state: Dict):
        if self.world_rank != 0:
            return
        path = self._training_state_path()
        with open(path, "w", encoding="utf-8") as file:
            json.dump(state, file, indent=2)

    def _get_domain_order(self, seed: int) -> List[int]:
        np.random.seed(seed)
        domain_ids = list(range(20))
        np.random.shuffle(domain_ids)
        return domain_ids

    def _get_next_domain_to_train(self, state: Optional[Dict]) -> Optional[int]:
        if state is None:
            state = {
                "domain_order": self._get_domain_order(self.config.SEED),
                "trained_domains": [],
            }
        for domain_id in state["domain_order"]:
            if domain_id not in state["trained_domains"]:
                return domain_id
        return None

    def _broadcast_model_state(self):
        if self.world_size <= 1:
            return
        model = self._get_policy_model()
        distrib.barrier()
        for param in model.parameters():
            distrib.broadcast(param.data, src=0)
        for buffer in model.buffers():
            distrib.broadcast(buffer.data, src=0)
        distrib.barrier()

    def _collect_rollout_step(self, rollouts, current_episode_reward, running_episode_stats):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        with torch.no_grad():
            step_observation = {
                key: value[rollouts.step] for key, value in rollouts.observations.items()
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
                external_memory_features,
                router_alpha,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
                external_memory,
                external_memory_masks,
                deterministic=False,
                is_rollout=True,
                use_current_task=True,
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()
        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device).unsqueeze(1)
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks

        for key, value in self._extract_scalars_from_infos(infos).items():
            value = torch.tensor(value, dtype=torch.float, device=current_episode_reward.device).unsqueeze(1)
            if key not in running_episode_stats:
                running_episode_stats[key] = torch.zeros_like(running_episode_stats["count"])
            running_episode_stats[key] += (1 - masks) * value

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
            router_alpha.to(device=self.device),
        )

        pth_time += time.time() - t_update_stats
        return pth_time, env_time, self.envs.num_envs

    def _collect_update_centroid(self, rollouts) -> Optional[torch.Tensor]:
        policy_model = self._get_policy_model()
        if not hasattr(policy_model, "get_anchor_features"):
            return None

        visual_keys = ["rgb", "depth", "semantic", "semantic_object"]
        valid_keys = [key for key in visual_keys if key in rollouts.observations]
        if len(valid_keys) == 0:
            return None

        with torch.no_grad():
            feature_sum = None
            sample_count = 0
            for step_idx in range(rollouts.step):
                step_obs = {
                    key: rollouts.observations[key][step_idx]
                    for key in valid_keys
                }
                anchor_feats = policy_model.get_anchor_features(step_obs).detach()
                step_sum = anchor_feats.sum(dim=0)
                if feature_sum is None:
                    feature_sum = step_sum
                else:
                    feature_sum = feature_sum + step_sum
                sample_count += anchor_feats.size(0)

            if feature_sum is None or sample_count == 0:
                return None
            local_centroid = (feature_sum / float(sample_count)).detach()
        return local_centroid

    def _build_task_routing_summary(self) -> Optional[TaskRoutingSummary]:
        policy_model = self._get_policy_model()
        local_centroids = self.cl_method.drain_centroid_buffer()
        if local_centroids is None:
            local_centroids = torch.empty((0, policy_model.prototype_dim), dtype=torch.float32)

        if self.world_size > 1:
            gathered = [None for _ in range(self.world_size)] if self.world_rank == 0 else None
            distrib.gather_object(local_centroids.cpu(), gathered, dst=0)
            if self.world_rank != 0:
                return None
            all_centroids = [item for item in gathered if isinstance(item, torch.Tensor) and item.numel() > 0]
            if len(all_centroids) == 0:
                raise RuntimeError("No local centroids collected for task summary construction")
            centroid_tensor = torch.cat(all_centroids, dim=0)
        else:
            if local_centroids.numel() == 0:
                raise RuntimeError("No local centroids collected for task summary construction")
            centroid_tensor = local_centroids.cpu()

        return build_task_routing_summary(
            centroid_tensor,
            bank_size=policy_model.prototype_bank_size,
            max_iters=policy_model.max_kmeans_iters,
        )

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {key: value[-1] for key, value in rollouts.observations.items()}
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
                is_rollout=False,
                use_current_task=True,
            ).detach()

        rollouts.compute_returns(next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)
        value_loss, action_loss, dist_entropy = self.agent.update_with_router(rollouts)
        rollouts.after_update()
        self.agent.optimizer.zero_grad(set_to_none=True)

        return time.time() - t_update_model, value_loss, action_loss, dist_entropy

    def _train_on_domain(self, domain_id: int, num_updates: int, writer):
        ppo_cfg = self.config.RL.PPO
        spark_cfg = self.config.CL_METHOD.SPARK_AVN

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        rollouts = RolloutStorageSparkAVN(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.action_space,
            ppo_cfg.hidden_size,
            ppo_cfg.use_external_memory,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size + ppo_cfg.num_steps,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
            self.actor_critic.net.memory_dim if ppo_cfg.use_external_memory else None,
            router_pool_size=spark_cfg.POOL_SIZE,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        del batch
        del observations

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(lambda: deque(maxlen=ppo_cfg.reward_window_size))

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, num_updates),
        )

        count_steps = 0
        pth_time = 0.0
        env_time = 0.0
        t_start = time.time()

        for update in range(num_updates):
            if self.world_size > 1 and EXIT.is_set():
                self.envs.close()
                if REQUEUE.is_set() and self.world_rank == 0:
                    requeue_job()
                return

            if ppo_cfg.use_linear_lr_decay:
                lr_scheduler.step()

            if ppo_cfg.use_linear_clip_decay:
                self.agent.clip_param = ppo_cfg.clip_param * linear_decay(update, num_updates)

            count_steps_delta = 0
            self.agent.eval()
            for step in range(ppo_cfg.num_steps):
                delta_pth, delta_env, delta_steps = self._collect_rollout_step(
                    rollouts,
                    current_episode_reward,
                    running_episode_stats,
                )
                pth_time += delta_pth
                env_time += delta_env
                count_steps_delta += delta_steps

            local_centroid = self._collect_update_centroid(rollouts)
            self.cl_method.add_local_centroid(local_centroid)

            self.agent.train()
            delta_pth, value_loss, action_loss, dist_entropy = self._update_agent(ppo_cfg, rollouts)
            pth_time += delta_pth

            stats_ordering = list(sorted(running_episode_stats.keys()))
            stats = torch.stack([running_episode_stats[k] for k in stats_ordering], 0)
            if self.world_size > 1:
                distrib.all_reduce(stats)

            for idx, key in enumerate(stats_ordering):
                window_episode_stats[key].append(stats[idx].clone().detach())

            loss_stats = torch.tensor(
                [value_loss, action_loss, dist_entropy, count_steps_delta],
                device=self.device,
            )
            if self.world_size > 1:
                distrib.all_reduce(loss_stats)
            count_steps += loss_stats[3].item()

            if self.world_rank == 0 and writer is not None:
                num_processes = self.world_size if self.world_size > 1 else 1
                losses = [loss_stats[i].item() / num_processes for i in range(3)]
                deltas = {
                    k: (v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item()
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    f"Domain_{domain_id}/reward",
                    deltas["reward"] / deltas["count"],
                    count_steps,
                )
                metrics = {
                    k: (v / deltas["count"])
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                for metric_name, metric_value in metrics.items():
                    writer.add_scalar(
                        f"Domain_{domain_id}/{metric_name}",
                        metric_value,
                        count_steps,
                    )

                writer.add_scalar(f"Domain_{domain_id}/value_loss", losses[0], count_steps)
                writer.add_scalar(f"Domain_{domain_id}/policy_loss", losses[1], count_steps)
                writer.add_scalar(f"Domain_{domain_id}/entropy_loss", losses[2], count_steps)
                writer.add_scalar(
                    f"Domain_{domain_id}/g_task",
                    float(self._get_policy_model().g_task.item()),
                    count_steps,
                )
                writer.add_scalar(f"Domain_{domain_id}/learning_rate", lr_scheduler.get_lr()[0], count_steps)

                cl_metrics = self.cl_method.get_additional_metrics()
                for metric_name, metric_value in cl_metrics.items():
                    writer.add_scalar(f"CL_Metrics/{metric_name}", metric_value, count_steps)

                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    fps = count_steps / max(time.time() - t_start, 1e-6)
                    logger.info(f"update: {update}\tfps: {fps:.3f}")
                    logger.info(
                        f"update: {update}\tenv-time: {env_time:.3f}s\t"
                        f"pth-time: {pth_time:.3f}s\tframes: {count_steps:.1f}"
                    )
                    metric_strs = [
                        f"{metric_name}: {metric_value / deltas['count']:.3f}"
                        for metric_name, metric_value in deltas.items()
                        if metric_name != "count"
                    ]
                    logger.info(
                        f"Average window size: {len(window_episode_stats['count'])}  "
                        f"{'  '.join(metric_strs)}"
                    )

        return {
            "steps": count_steps,
            "env_time": env_time,
            "pth_time": pth_time,
        }

    def _load_resume_checkpoint_if_needed(self, state: Optional[Dict]):
        if state is None or len(state["trained_domains"]) == 0:
            return None
        last_domain = state["trained_domains"][-1]
        ckpt_path = os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt_domain_{last_domain}.pth")
        if not os.path.exists(ckpt_path):
            return None
        ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
        self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
        self.actor_critic = self.agent.actor_critic
        if self.cl_method is not None and "cl_method_state" in ckpt_dict:
            self.cl_method.load_state_dict(ckpt_dict["cl_method_state"])
        return ckpt_dict

    def train(self) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if not distrib.is_initialized() and self.world_size > 1:
            distrib.init_process_group(
                backend=self.config.RL.DDPPO.distrib_backend, init_method="env://"
            )
        if self.world_size > 1:
            add_signal_handlers()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        self.config.TASK_CONFIG.SEED += self.world_rank * self.config.NUM_PROCESSES
        self.config.freeze()

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        state = self._load_training_state()
        if state is None:
            state = {
                "domain_order": self._get_domain_order(self.config.SEED),
                "trained_domains": [],
            }
            self._save_training_state(state)

        self.domain_order = state["domain_order"]
        next_domain_id = self._get_next_domain_to_train(state)
        if next_domain_id is None:
            if self.world_rank == 0:
                logger.info("All domains have been trained.")
            return

        self.current_domain_id = next_domain_id
        domain_seed = self.config.SEED + next_domain_id * 1000
        random.seed(domain_seed + self.world_rank)
        np.random.seed(domain_seed + self.world_rank)
        torch.manual_seed(domain_seed + self.world_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(domain_seed + self.world_rank)

        self.config.defrost()
        domain_scenes = self._get_domain_scenes(next_domain_id)
        if len(domain_scenes) == 1 and self.config.NUM_PROCESSES > 1:
            domain_scenes = domain_scenes * self.config.NUM_PROCESSES
        self.config.TASK_CONFIG.DATASET.CONTENT_SCENES = domain_scenes
        self.config.freeze()

        if self.world_rank == 0:
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
            logger.info("Domain order: %s", self.domain_order)
            logger.info("Training domain %d", next_domain_id)

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        ppo_cfg = self.config.RL.PPO
        self._setup_actor_critic_agent(ppo_cfg)
        if self.world_size > 1:
            self.agent.init_distributed(find_unused_params=True)
        self._setup_cl_method()
        self._load_resume_checkpoint_if_needed(state)

        policy_model = self._get_policy_model()
        policy_model.start_new_task(self.envs.num_envs)
        self.cl_method.on_task_start(next_domain_id)

        if self.world_rank == 0:
            with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=30) as writer:
                self._train_on_domain(
                    domain_id=next_domain_id,
                    num_updates=self.config.NUM_UPDATES_PER_DOMAIN,
                    writer=writer,
                )
        else:
            self._train_on_domain(
                domain_id=next_domain_id,
                num_updates=self.config.NUM_UPDATES_PER_DOMAIN,
                writer=None,
            )

        task_summary = self._build_task_routing_summary()

        if self.world_rank == 0:
            if task_summary is None:
                raise RuntimeError("Rank0 failed to build TaskRoutingSummary")
            policy_model.commit_current_task_to_pool(task_summary)
            active_count = int(policy_model.pool_active_count.item())
            counters = [
                int(policy_model.pool_counters[idx].item())
                for idx in range(active_count)
            ]
            counters_str = ", ".join(
                [f"slot{idx}={count}" for idx, count in enumerate(counters)]
            )
            logger.info(
                "Domain %d pool counters (active=%d): %s",
                next_domain_id,
                active_count,
                counters_str if counters_str else "empty",
            )
        self._broadcast_model_state()

        self.cl_method.on_task_end(next_domain_id)

        if self.world_rank == 0:
            self.save_checkpoint(
                f"ckpt_domain_{next_domain_id}.pth",
                {
                    "domain_id": next_domain_id,
                    "trained_domains": state["trained_domains"] + [next_domain_id],
                },
            )
            if next_domain_id not in state["trained_domains"]:
                state["trained_domains"].append(next_domain_id)
            self._save_training_state(state)
            logger.info("Domain %d training completed.", next_domain_id)

        if self.world_size > 1:
            distrib.barrier()
        self.envs.close()

    def _load_matrix_eval_state(self) -> Dict[int, Dict[int, Dict]]:
        state_file = os.path.join(self.config.MODEL_DIR, self._get_matrix_state_filename())
        if not os.path.exists(state_file):
            return {}

        try:
            with open(state_file, "r", encoding="utf-8") as file:
                state = json.load(file)
        except Exception as error:
            logger.warning("Failed to load matrix eval state from %s: %s", state_file, error)
            return {}

        matrix: Dict[int, Dict[int, Dict]] = {}
        raw_matrix = state.get("performance_matrix", {})
        if not isinstance(raw_matrix, dict):
            return {}

        for train_domain_str, eval_dict in raw_matrix.items():
            try:
                train_domain = int(train_domain_str)
            except (TypeError, ValueError):
                continue
            matrix[train_domain] = {}
            if not isinstance(eval_dict, dict):
                continue
            for eval_domain_str, metrics in eval_dict.items():
                try:
                    eval_domain = int(eval_domain_str)
                except (TypeError, ValueError):
                    continue
                matrix[train_domain][eval_domain] = metrics

        logger.info("Loaded existing matrix evaluation state from %s", state_file)
        return matrix

    def _save_matrix_eval_state(
        self,
        domain_order: List[int],
        performance_matrix: Dict[int, Dict[int, Dict]],
        last_evaluated_domain: int,
    ) -> None:
        if self.world_rank != 0:
            return

        state_file = os.path.join(self.config.MODEL_DIR, self._get_matrix_state_filename())
        state = {
            "domain_order": domain_order,
            "last_evaluated_domain": last_evaluated_domain,
            "performance_matrix": {},
            "seed": self.config.SEED,
        }

        for train_domain, eval_dict in performance_matrix.items():
            state["performance_matrix"][str(train_domain)] = {}
            for eval_domain, metrics in eval_dict.items():
                state["performance_matrix"][str(train_domain)][str(eval_domain)] = metrics

        with open(state_file, "w", encoding="utf-8") as file:
            json.dump(state, file, indent=2)

    def _load_final_eval_results(self) -> Dict[int, Dict]:
        result_path = os.path.join(self.config.MODEL_DIR, self._get_eval_result_filename())
        if not os.path.exists(result_path):
            return {}

        try:
            with open(result_path, "r", encoding="utf-8") as file:
                raw_data = json.load(file)
        except Exception as error:
            logger.warning("Failed to load final eval results from %s: %s", result_path, error)
            return {}

        parsed: Dict[int, Dict] = {}
        if not isinstance(raw_data, dict):
            return {}

        for key, value in raw_data.items():
            if key == "next_domain":
                if isinstance(value, dict):
                    for domain_key, metrics in value.items():
                        try:
                            parsed[int(domain_key)] = metrics
                        except (TypeError, ValueError):
                            continue
                continue

            try:
                parsed[int(key)] = value
            except (TypeError, ValueError):
                continue

        return parsed

    def _bootstrap_last_row_from_final_eval(
        self,
        performance_matrix: Dict[int, Dict[int, Dict]],
        trained_domains: List[int],
        domain_order: List[int],
    ) -> int:
        if len(trained_domains) == 0:
            return 0

        final_results = self._load_final_eval_results()
        if len(final_results) == 0:
            return 0

        last_train_domain = trained_domains[-1]
        if last_train_domain not in domain_order:
            return 0

        trained_set = set(trained_domains)
        row_idx = domain_order.index(last_train_domain)
        eval_domains = domain_order[: row_idx + 1]
        if row_idx + 1 < len(domain_order) and domain_order[row_idx + 1] in trained_set:
            eval_domains = eval_domains + [domain_order[row_idx + 1]]

        if last_train_domain not in performance_matrix:
            performance_matrix[last_train_domain] = {}

        inserted = 0
        for eval_domain in eval_domains:
            if eval_domain in performance_matrix[last_train_domain]:
                continue
            if eval_domain not in final_results:
                continue
            performance_matrix[last_train_domain][eval_domain] = final_results[eval_domain]
            inserted += 1

        if inserted > 0:
            logger.info(
                "Bootstrapped %d matrix cells in last row (train domain=%d) from final eval results.",
                inserted,
                last_train_domain,
            )
        return inserted

    def _evaluate_final_mode(self, trained_domains: List[int], domain_order: List[int]) -> None:
        last_domain = trained_domains[-1]
        ckpt_path = os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt_domain_{last_domain}.pth")
        if not os.path.exists(ckpt_path):
            logger.error("Checkpoint not found: %s", ckpt_path)
            return

        output_path = os.path.join(self.config.MODEL_DIR, self._get_eval_result_filename())

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info("Resuming evaluation, already done: %s", list(results.keys()))
        else:
            results = {}

        all_eval_domains = list(trained_domains)
        if len(trained_domains) < len(domain_order):
            all_eval_domains.append(domain_order[len(trained_domains)])

        total = len(all_eval_domains)
        for i, eval_domain in enumerate(all_eval_domains):
            is_forward = (eval_domain not in trained_domains)
            result_key = "next_domain" if is_forward else str(eval_domain)

            if result_key in results:
                logger.info("Skipping domain %d (already evaluated)", eval_domain)
                continue

            logger.info("Evaluating final checkpoint on domain %d", eval_domain)
            metrics = self._eval_checkpoint(
                checkpoint_path=ckpt_path,
                writer=None,
                checkpoint_index=last_domain,
                domain_id=eval_domain,
                progress_desc=f"[{i+1}/{total}] domain={eval_domain}",
            )

            if is_forward:
                results[result_key] = {str(eval_domain): metrics}
            else:
                results[result_key] = metrics

            if self.world_rank == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                logger.info("Saved intermediate results to %s", output_path)

        logger.info("Final continual evaluation complete: %s", output_path)

    def _evaluate_matrix_mode(self, trained_domains: List[int], domain_order: List[int]) -> None:
        trained_set = set(trained_domains)
        performance_matrix = self._load_matrix_eval_state()
        bootstrapped = self._bootstrap_last_row_from_final_eval(
            performance_matrix=performance_matrix,
            trained_domains=trained_domains,
            domain_order=domain_order,
        )
        if bootstrapped > 0:
            self._save_matrix_eval_state(domain_order, performance_matrix, trained_domains[-1])
        trained_prefix = []
        for domain_id in domain_order:
            if domain_id not in trained_set:
                break
            trained_prefix.append(domain_id)
        total_rows = len(trained_prefix)
        col_pos_map = {domain_id: idx + 1 for idx, domain_id in enumerate(domain_order)}

        total_tasks = 0
        remaining_tasks = 0
        for idx, train_domain in enumerate(domain_order):
            if train_domain not in trained_set:
                break
            eval_domains = domain_order[: idx + 1]
            if idx + 1 < len(domain_order) and domain_order[idx + 1] in trained_set:
                eval_domains = eval_domains + [domain_order[idx + 1]]
            total_tasks += len(eval_domains)
            existing = performance_matrix.get(train_domain, {})
            for eval_domain in eval_domains:
                if eval_domain not in existing:
                    remaining_tasks += 1

        if total_tasks == 0:
            logger.warning("No available tasks for matrix evaluation.")
            return

        if remaining_tasks == 0:
            logger.info("Matrix evaluation already completed. Nothing to do.")
            self._save_matrix_eval_state(domain_order, performance_matrix, trained_domains[-1])
            return

        logger.info(
            "Matrix evaluation tasks: remaining %d / total %d (rows=%d)",
            remaining_tasks,
            total_tasks,
            total_rows,
        )
        completed_tasks = total_tasks - remaining_tasks

        for idx, train_domain in enumerate(domain_order):
            if train_domain not in trained_set:
                break

            ckpt_path = os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt_domain_{train_domain}.pth")
            if not os.path.exists(ckpt_path):
                logger.warning("Checkpoint not found, skip train domain %d: %s", train_domain, ckpt_path)
                continue

            eval_domains = domain_order[: idx + 1]
            if idx + 1 < len(domain_order) and domain_order[idx + 1] in trained_set:
                eval_domains = eval_domains + [domain_order[idx + 1]]
            row_total = len(eval_domains)
            row_done = 0
            for eval_domain in eval_domains:
                if eval_domain in performance_matrix.get(train_domain, {}):
                    row_done += 1

            logger.info(
                "[Matrix Eval] Start row %d/%d: train_domain=%d, row_total=%d, row_done=%d",
                idx + 1,
                total_rows,
                train_domain,
                row_total,
                row_done,
            )

            for eval_domain in eval_domains:
                existing = performance_matrix.get(train_domain, {})
                if eval_domain in existing:
                    continue

                completed_tasks += 1
                row_done += 1
                row_pos = idx + 1
                col_pos = col_pos_map.get(eval_domain, -1)
                logger.info(
                    "[Matrix Eval] Evaluate R[%d,%d] | task=%d/%d | row=%d/%d",
                    row_pos,
                    col_pos,
                    completed_tasks,
                    total_tasks,
                    row_done,
                    row_total,
                )
                logger.info(
                    "[Matrix Eval] Current task: checkpoint(domain=%d) -> eval(domain=%d)",
                    train_domain,
                    eval_domain,
                )
                progress_desc = f"R[{row_pos},{col_pos}] train={train_domain} eval={eval_domain}"
                metrics = self._eval_checkpoint(
                    checkpoint_path=ckpt_path,
                    writer=None,
                    checkpoint_index=train_domain,
                    domain_id=eval_domain,
                    progress_desc=progress_desc,
                )

                if train_domain not in performance_matrix:
                    performance_matrix[train_domain] = {}
                performance_matrix[train_domain][eval_domain] = metrics
                self._save_matrix_eval_state(domain_order, performance_matrix, train_domain)

            logger.info(
                "[Matrix Eval] Finished row %d/%d: train_domain=%d, row=%d/%d",
                idx + 1,
                total_rows,
                train_domain,
                row_done,
                row_total,
            )

        logger.info("Matrix evaluation completed. State saved to %s", os.path.join(self.config.MODEL_DIR, self._get_matrix_state_filename()))

    def evaluate_continual(self, eval_mode: str = "final") -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        state = self._load_training_state()
        if state is None or len(state.get("trained_domains", [])) == 0:
            logger.error("No trained domains found for continual evaluation.")
            return

        trained_domains = state["trained_domains"]
        domain_order = state["domain_order"]

        eval_mode = str(eval_mode).lower()
        if eval_mode not in {"final", "matrix"}:
            raise ValueError(f"Unsupported eval_mode: {eval_mode}, expected 'final' or 'matrix'")

        logger.info("SPARK-AVN continual evaluation mode: %s", eval_mode)
        if eval_mode == "matrix":
            self._evaluate_matrix_mode(trained_domains, domain_order)
        else:
            self._evaluate_final_mode(trained_domains, domain_order)
