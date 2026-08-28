#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union
import os
import logging
import shutil

import numpy as np

from habitat import get_config as get_task_config
from habitat.config import Config as CN
import habitat
from habitat.config.default import SIMULATOR_SENSOR

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointgoal.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "savi"
_C.ENV_NAME = "AudioNavRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.VISUALIZATION_OPTION = ["top_down_map"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.MODEL_DIR = 'data/models/output'
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.USE_VECENV = True
_C.USE_SYNC_VECENV = False
_C.EXTRA_RGB = False
_C.DEBUG = False
_C.USE_LAST_CKPT = False
_C.DISPLAY_RESOLUTION = 128
_C.CONTINUOUS = False
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# sample preserves the official AVN policy evaluation. argmax is available for
# controlled mechanism studies against deterministic VLN evaluation.
_C.EVAL.ACTION_SELECTION = "sample"
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.WITH_TIME_PENALTY = True
_C.RL.WITH_DISTANCE_REWARD = True
_C.RL.DISTANCE_REWARD_SCALE = 1.0
_C.RL.TIME_DIFF = False
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 2
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.use_normalized_advantage = False
_C.RL.PPO.policy_type = 'rnn'
_C.RL.PPO.use_external_memory = False
_C.RL.PPO.use_mlp_state_encoder = False
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER = CN()
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.memory_size = 300
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.hidden_size = 128
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.nhead = 8
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.num_encoder_layers = 1
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.num_decoder_layers = 1
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.dropout = 0.0
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.activation = 'relu'
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.use_pretrained = False
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.pretrained_path = ''
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.freeze_encoders = False
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.pretraining = False
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.use_action_encoding = True
_C.RL.PPO.SCENE_MEMORY_TRANSFORMER.use_belief_encoding = False
_C.RL.PPO.use_belief_predictor = False
_C.RL.PPO.BELIEF_PREDICTOR = CN()
_C.RL.PPO.BELIEF_PREDICTOR.online_training = False
_C.RL.PPO.BELIEF_PREDICTOR.lr = 1e-3
_C.RL.PPO.BELIEF_PREDICTOR.audio_only = False
_C.RL.PPO.BELIEF_PREDICTOR.train_encoder = False
_C.RL.PPO.BELIEF_PREDICTOR.normalize_category_distribution = False
_C.RL.PPO.BELIEF_PREDICTOR.use_label_belief = True
_C.RL.PPO.BELIEF_PREDICTOR.use_location_belief = True
# Use a frozen, author-pretrained goal descriptor (location predictor) instead
# of training it online with ground-truth pointgoal coordinates. When True the
# location predictor is loaded from `descriptor_weights` and frozen, so no
# ground-truth target leakage happens during navigation-policy training.
_C.RL.PPO.BELIEF_PREDICTOR.load_pretrained_descriptor = False
_C.RL.PPO.BELIEF_PREDICTOR.descriptor_weights = "data/pretrained_weights/semantic_audionav/savi/best_val.pth"
_C.RL.PPO.BELIEF_PREDICTOR.current_pred_only = False
_C.RL.PPO.BELIEF_PREDICTOR.weighting_factor = 0.5
# -----------------------------------------------------------------------------
# DECENTRALIZED DISTRIBUTED PROXIMAL POLICY OPTIMIZATION (DD-PPO)
# -----------------------------------------------------------------------------
_C.RL.DDPPO = CN()
_C.RL.DDPPO.sync_frac = 0.6
_C.RL.DDPPO.distrib_backend = "GLOO"
_C.RL.DDPPO.rnn_type = "LSTM"
_C.RL.DDPPO.num_recurrent_layers = 1
_C.RL.DDPPO.backbone = "resnet50"
_C.RL.DDPPO.pretrained_weights = ""
# Loads pretrained weights
_C.RL.DDPPO.pretrained = False
# Whether or not to reset the critic linear layer
_C.RL.DDPPO.reset_critic = True
# -----------------------------------------------------------------------------
# TEST-TIME ADAPTATION (TTA)
# -----------------------------------------------------------------------------
_C.TTA = CN()
# none | tent | fstta | eam | feedtta | atena
_C.TTA.METHOD = "none"
_C.TTA.MATCHED_FEEDTTA_SOURCE = False
_C.TTA.EPISODIC = False
# Conservative AVN default. Tune on a dedicated TTA-dev stream.
_C.TTA.LR = 1e-6
_C.TTA.STEPS = 1
# Tent-LN scope: ln (all) | last_k_ln | first_ln | last_ln
_C.TTA.NORM_SCOPE = "last_k_ln"
_C.TTA.LAST_K_LN = 4
_C.TTA.RESET_BN_STATS = True
_C.TTA.OPTIMIZER = "Adam"
_C.TTA.MOMENTUM = 0.9
_C.TTA.BETA1 = 0.9
_C.TTA.BETA2 = 0.999
_C.TTA.WEIGHT_DECAY = 0.0
_C.TTA.MAX_GRAD_NORM = 1.0
_C.TTA.UPDATE_INTERVAL = 1
# -1 keeps the original continual-Tent behavior. Nonnegative values cap the
# number of optimizer updates inside each episode without resetting the model.
_C.TTA.MAX_UPDATES_PER_EPISODE = -1
_C.TTA.LOG_INTERVAL_EPISODES = 50
_C.TTA.FSTTA = CN()
_C.TTA.FSTTA.M = 3
_C.TTA.FSTTA.N = 4
_C.TTA.FSTTA.Q = 0.1
_C.TTA.FSTTA.LR_SLOW = 1e-4
# Paper FAST-LR band: rho=0.95, tau=0.7, scale [0.9, 1.1].
_C.TTA.FSTTA.RHO = 0.95
_C.TTA.FSTTA.TAU = 0.7
_C.TTA.FSTTA.A = 0.9
_C.TTA.FSTTA.B = 1.1
_C.TTA.FSTTA.USE_SLOW = True
_C.TTA.FSTTA.FAST_GRAD_MODE = "concordant"
_C.TTA.FSTTA.USE_FAST_LR_SCALER = True
# FAST optimizer. The SLOW anchor has its own persistent optimizer below.
_C.TTA.FSTTA.OPTIMIZER = "AdamW"
_C.TTA.FSTTA.BETA1 = 0.9
_C.TTA.FSTTA.BETA2 = 0.99
_C.TTA.FSTTA.WEIGHT_DECAY = 0.0
# Empty/-1 preserve the historical behavior of inheriting the FAST optimizer
# and TTA.MOMENTUM. Exploration launchers set both fields explicitly.
_C.TTA.FSTTA.SLOW_OPTIMIZER = ""
_C.TTA.FSTTA.SLOW_MOMENTUM = -1.0
_C.TTA.FSTTA.RESET_OPTIMIZER_EACH_EPISODE = True
# Paper Eq. (6) keeps historical variance across the test stream.  ``True``
# remains available only as a released-code rollout-reset ablation.
_C.TTA.FSTTA.RESET_VAR_HIST_EACH_EPISODE = False
_C.TTA.FSTTA.RESET_SLOW_OPTIMIZER_EACH_WINDOW = False
_C.TTA.FSTTA.EIGEN_EPS = 1e-6
_C.TTA.EAM = CN()
_C.TTA.EAM.LR = 1e-5
_C.TTA.EAM.CONFIDENCE_SCALE = 0.4
# Replay units are online action-step snapshots.
_C.TTA.EAM.MEMORY_SIZE = 32
_C.TTA.EAM.BATCH_SIZE = 8
# Action-step interval; replay updates start only after |M| reaches batch K.
_C.TTA.EAM.UPDATE_INTERVAL = 1
# AVN mapping: freeze all pre-Transformer sensory/state encoders and update the
# auxiliary Transformer plus action-decision head.
_C.TTA.EAM.PARAM_SCOPE = "module_prefixes"
_C.TTA.EAM.TRAINABLE_PREFIXES = [
    "net.smt_state_encoder.transformer",
    "action_distribution",
]
_C.TTA.EAM.OPTIMIZER = "Adam"
_C.TTA.EAM.MOMENTUM = 0.9
_C.TTA.EAM.BETA1 = 0.9
_C.TTA.EAM.BETA2 = 0.999
_C.TTA.EAM.WEIGHT_DECAY = 0.0
# The paper does not apply gradient clipping; 0 disables it.
_C.TTA.EAM.MAX_GRAD_NORM = 0.0
_C.TTA.FEEDTTA = CN()
_C.TTA.FEEDTTA.LR = 5e-6
_C.TTA.FEEDTTA.P = 0.05
# True SGR anchor from REVERIE val-unseen. The paper's literal R2R value
# (+0.1) is gradient scaling rather than reversion; AVN must sweep both signs.
_C.TTA.FEEDTTA.ALPHA = -0.2
_C.TTA.FEEDTTA.SGR_SEED = 0
_C.TTA.FEEDTTA.SGR_MODE = "paper_main"
_C.TTA.FEEDTTA.GAMMA = 0.99
_C.TTA.FEEDTTA.ACTION_SELECTION_PROTOCOL = "sample_from_policy"
# Eq. (3) sums discounted trajectory gradients; it does not length-normalize.
_C.TTA.FEEDTTA.NORMALIZE_GRADIENT = False
# Freeze sensory encoders and adapt the fusion encoder plus action head.
_C.TTA.FEEDTTA.PARAM_SCOPE = "module_prefixes"
_C.TTA.FEEDTTA.TRAINABLE_PREFIXES = [
    "net.smt_state_encoder", "action_distribution"
]
_C.TTA.FEEDTTA.OPTIMIZER = "Adam"
_C.TTA.FEEDTTA.BETA1 = 0.9
_C.TTA.FEEDTTA.BETA2 = 0.999
_C.TTA.FEEDTTA.WEIGHT_DECAY = 0.0
_C.TTA.FEEDTTA.EPS = 1e-5
# The paper does not report gradient clipping; 0 disables it.
_C.TTA.FEEDTTA.MAX_GRAD_NORM = 0.0
_C.TTA.ATENA = CN()
# Keep ATENA blocked until the AVN-specific pre-run review is repeated.
_C.TTA.ATENA.PREFLIGHT_APPROVED = False
_C.TTA.ATENA.LR_QUERY = 1e-6
_C.TTA.ATENA.LR_SELF = 1e-7
_C.TTA.ATENA.MIX_LAMBDA = 0.5
# Paper/official ATENA applies the threshold directly to raw action entropy.
_C.TTA.ATENA.QUERY_THRESHOLD = 0.1
_C.TTA.ATENA.SELF_LOSS_WEIGHT = 0.1
_C.TTA.ATENA.PARAM_SCOPE = "all"
# Official VLN uses policy_argmax.  AVN launchers must opt in explicitly to
# sample_from_policy so the executed task-native action is the pseudo expert.
_C.TTA.ATENA.ACTION_SELECTION_PROTOCOL = "policy_argmax"
# AVN replays the actor navigation graph but excludes the value-only critic.
# A requested end-to-end/full-policy claim is rejected by the trainer.
_C.TTA.ATENA.TASK_UPDATE_SCOPE = (
    "replay_reachable_actor_navigation_policy"
)
_C.TTA.ATENA.OPTIMIZER = "AdamW"
_C.TTA.ATENA.BETA1 = 0.9
_C.TTA.ATENA.BETA2 = 0.999
_C.TTA.ATENA.WEIGHT_DECAY = 0.01
_C.TTA.ATENA.MAX_GRAD_NORM = 0.0
# IDEA (ICML 2026): training-free adaptation via a historical asset library.
# Requires a task-supplied IDEAFusionProtocol; the base policy is never updated.
_C.TTA.IDEA = CN()
_C.TTA.IDEA.PROMPT_LENGTH = 4
_C.TTA.IDEA.K_MAX = 32
_C.TTA.IDEA.LAMBDA = 0.4
_C.TTA.IDEA.TAU = 0.7
_C.TTA.IDEA.FISHER_BETA = 0.1
_C.TTA.IDEA.OPT_STEPS = 50
_C.TTA.IDEA.LR = 3e-3
_C.TTA.IDEA.OPTIMIZER = "AdamW"
_C.TTA.IDEA.MOMENTUM = 0.9
_C.TTA.IDEA.BETA1 = 0.9
_C.TTA.IDEA.BETA2 = 0.999
_C.TTA.IDEA.WEIGHT_DECAY = 0.0
_C.TTA.IDEA.USE_FISHER = True
_C.TTA.IDEA.RIDGE = 1e-4
_C.TTA.IDEA.MAX_GRAD_NORM = 0.0
_C.TTA.IDEA.PROMPT_INIT_STD = 0.02
_C.TTA.IDEA.SEED = 0
# Number of aligned fusion-transformer encoder layers (0 = use all encoder
# layers).  The SMT encoder is shallow, so the default keeps every layer.
_C.TTA.IDEA.PROMPT_LAYERS = 0
# Digest-pinned global moments collected from 128 source-training trajectories.
# IDEA evaluation fails closed while either field is empty.
_C.TTA.IDEA.SOURCE_STATS_PATH = ""
_C.TTA.IDEA.SOURCE_STATS_SHA256 = ""
_C.TTA.IDEA.SOURCE_TRAJECTORIES = 128
_C.TTA.IDEA.SOURCE_COLLECTION = False
_C.TTA.IDEA.SOURCE_COLLECTION_OUTPUT = ""
_C.TTA.IDEA.SOURCE_EPISODE_MANIFEST = ""
# Target evaluation also requires this digest so an artifact cannot be swapped
# across source selections; the manifest path itself is collection-only.
_C.TTA.IDEA.SOURCE_EPISODE_MANIFEST_SHA256 = ""
# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------
_TC = habitat.get_config()
_TC.defrost()
# -----------------------------------------------------------------------------
# AUDIOGOAL_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.AUDIOGOAL_SENSOR = CN()
_TC.TASK.AUDIOGOAL_SENSOR.TYPE = "AudioGoalSensor"
# -----------------------------------------------------------------------------
# SPECTROGRAM_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.SPECTROGRAM_SENSOR = CN()
_TC.TASK.SPECTROGRAM_SENSOR.TYPE = "SpectrogramSensor"
# -----------------------------------------------------------------------------
# soundspaces
# -----------------------------------------------------------------------------
_TC.SIMULATOR.GRID_SIZE = 0.5
_TC.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False
_TC.SIMULATOR.VIEW_CHANGE_FPS = 10
_TC.SIMULATOR.SCENE_DATASET = 'replica'
_TC.SIMULATOR.USE_RENDERED_OBSERVATIONS = True
_TC.SIMULATOR.SCENE_OBSERVATION_DIR = 'data/scene_observations'
_TC.SIMULATOR.AUDIO = CN()
_TC.SIMULATOR.AUDIO.SCENE = ""
_TC.SIMULATOR.AUDIO.EVERLASTING = True
_TC.SIMULATOR.AUDIO.BINAURAL_RIR_DIR = "data/binaural_rirs"
_TC.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = 44100
_TC.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = "data/sounds/1s_all"
_TC.SIMULATOR.AUDIO.METADATA_DIR = "data/metadata"
_TC.SIMULATOR.AUDIO.POINTS_FILE = 'points.txt'
_TC.SIMULATOR.AUDIO.GRAPH_FILE = 'graph.pkl'
_TC.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND = False
_TC.SIMULATOR.AUDIO.DISTRACTOR_SOUND_DIR = "data/sounds/1s_all"
_TC.SIMULATOR.AUDIO.DISTRACTOR_SOUND_DIR = 'data/sounds/1s_all_distractor'
# -----------------------------------------------------------------------------
# DistanceToGoal Measure
# -----------------------------------------------------------------------------
_TC.TASK.NORMALIZED_DISTANCE_TO_GOAL = CN()
_TC.TASK.NORMALIZED_DISTANCE_TO_GOAL.TYPE = "NormalizedDistanceToGoal"
# -----------------------------------------------------------------------------
# Dataset extension
# -----------------------------------------------------------------------------
_TC.DATASET.VERSION = 'v1'
_TC.DATASET.TTA_EPISODES_PER_SCENE = -1
_TC.DATASET.TTA_EXPECTED_SCENES = -1
_TC.DATASET.TTA_EPISODE_SEED = 0
_TC.DATASET.TTA_GLOBAL_SHUFFLE = False
_TC.DATASET.IDEA_SOURCE_EPISODE_MANIFEST = ""
_TC.DATASET.IDEA_SOURCE_EPISODE_MANIFEST_SHA256 = ""
# -----------------------------------------------------------------------------
# NumberOfAction Measure
# -----------------------------------------------------------------------------
_TC.TASK.NUM_ACTION = CN()
_TC.TASK.NUM_ACTION.TYPE = "NA"
_TC.TASK.SUCCESS_WEIGHTED_BY_NUM_ACTION = CN()
_TC.TASK.SUCCESS_WEIGHTED_BY_NUM_ACTION.TYPE = "SNA"
_TC.TASK.VIEW_POINT_GOALS = CN()
_TC.TASK.VIEW_POINT_GOALS.TYPE = "ViewPointGoals"
# -----------------------------------------------------------------------------
# Intensity estimated from ambisonic
# -----------------------------------------------------------------------------
_TC.TASK.CATEGORY = SIMULATOR_SENSOR.clone()
_TC.TASK.CATEGORY.TYPE = "Category"
_TC.TASK.CATEGORY_BELIEF = SIMULATOR_SENSOR.clone()
_TC.TASK.CATEGORY_BELIEF.TYPE = "CategoryBelief"
_TC.TASK.LOCATION_BELIEF = SIMULATOR_SENSOR.clone()
_TC.TASK.LOCATION_BELIEF.TYPE = "LocationBelief"
_TC.TASK.SUCCESS_WHEN_SILENT = CN()
_TC.TASK.SUCCESS_WHEN_SILENT.TYPE = "SWS"
# -----------------------------------------------------------------------------
# POSE SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.POSE_SENSOR = CN()
_TC.TASK.POSE_SENSOR.TYPE = "PoseSensor"
# -----------------------------------------------------------------------------
# SEMANTIC OBJECT SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.SEMANTIC_OBJECT_SENSOR = CN()
_TC.TASK.SEMANTIC_OBJECT_SENSOR.TYPE = "SemanticObjectSensor"
_TC.TASK.SEMANTIC_OBJECT_SENSOR.HEIGHT = 128
_TC.TASK.SEMANTIC_OBJECT_SENSOR.WIDTH = 128
_TC.TASK.SEMANTIC_OBJECT_SENSOR.HFOV = 90  # horizontal field of view in degrees
_TC.TASK.SEMANTIC_OBJECT_SENSOR.POSITION = [0, 1.25, 0]
_TC.TASK.SEMANTIC_OBJECT_SENSOR.ORIENTATION = [0.0, 0.0, 0.0]  # Euler's angles
_TC.TASK.SEMANTIC_OBJECT_SENSOR.CONVERT_TO_RGB = True
_TC.TASK.ORACLE_ACTION_SENSOR = CN()
_TC.TASK.ORACLE_ACTION_SENSOR.TYPE = "OracleActionSensor"


def merge_from_path(config, config_paths):
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)
    return config


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    model_dir: Optional[str] = None,
    run_type: Optional[str] = None,
    overwrite: bool = False
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
        model_dir: suffix for output dirs
        run_type: either train or eval
    """
    config = merge_from_path(_C.clone(), config_paths)
    config.TASK_CONFIG = get_task_config(config_paths=config.BASE_TASK_CONFIG_PATH)

    # config_name = os.path.basename(config_paths).split('.')[0]
    if model_dir is not None:
        config.MODEL_DIR = model_dir
    config.TENSORBOARD_DIR = os.path.join(config.MODEL_DIR, 'tb')
    config.CHECKPOINT_FOLDER = os.path.join(config.MODEL_DIR, 'data')
    config.VIDEO_DIR = os.path.join(config.MODEL_DIR, 'video_dir')
    config.LOG_FILE = os.path.join(config.MODEL_DIR, 'train.log')
    config.EVAL_CKPT_PATH_DIR = os.path.join(config.MODEL_DIR, 'data')

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    dirs = [config.VIDEO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
    if run_type == 'train':
        # check dirs
        if any([os.path.exists(d) for d in dirs]):
            for d in dirs:
                if os.path.exists(d):
                    logging.warning('{} exists'.format(d))
            # if overwrite or input('Output directory already exists! Overwrite the folder? (y/n)') == 'y':
            if overwrite:
                for d in dirs:
                    if os.path.exists(d):
                        shutil.rmtree(d)
    else:
        # Respect NUM_PROCESSES from the evaluation yaml.  In particular,
        # sequential TTA requires one environment so episode boundaries are
        # well-defined; source evaluation configs may still use more workers.
        config.defrost()
        # NOTE: do NOT hard-override TEST_EPISODE_COUNT here. The original repo
        # forced 500/1000, which silently ignored the value set in the
        # exp-config yaml. We respect the yaml value instead.
        if config.EVAL.SPLIT.startswith('val'):
            config.USE_SYNC_VECENV = True
        elif config.EVAL.SPLIT.startswith('test'):
            pass
        else:
            # TTA datasets (e.g. ENMuS-format) may name the evaluation split
            # 'train'. Don't hard-fail; fall back to the value already provided
            # by the exp-config yaml (TEST_EPISODE_COUNT) instead.
            logging.warning(
                "EVAL.SPLIT='{}' does not start with val/test; "
                "using TEST_EPISODE_COUNT from config ({}).".format(
                    config.EVAL.SPLIT, config.TEST_EPISODE_COUNT
                )
            )
        config.freeze()

    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.SIMULATOR.USE_SYNC_VECENV = config.USE_SYNC_VECENV
    config.TASK_CONFIG.freeze()
    config.freeze()
    return config


def get_task_config(
        config_paths: Optional[Union[List[str], str]] = None,
        opts: Optional[list] = None
) -> habitat.Config:
    config = _TC.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
