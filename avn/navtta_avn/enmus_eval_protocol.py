"""Resolved evaluation semantics and compact audio evidence for ENMuS."""

import json
from pathlib import Path

from .audio_schedule import (
    LEGACY_MODE, SEEDED_MODE, canonical_digest, episode_key,
)


PROFILE = "enmus_clavn_aligned_v1"
AUDIT_MEASURE = "AUDIO_SCHEDULE_AUDIT"
AUDIT_UUID = "audio_schedule_audit"

EXPECTED = {
    "EVAL.USE_CKPT_CONFIG": False,
    "EVAL.SPLIT": "val",
    "EVAL.ACTION_SELECTION": "sample",
    "NUM_PROCESSES": 1,
    "ENV_NAME": "AudioNavRLEnv",
    "CONTINUOUS": False,
    "USE_VECENV": True,
    "USE_SYNC_VECENV": False,
    "SENSORS": ["DEPTH_SENSOR", "RGB_SENSOR"],
    "TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS": 500,
    "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE": True,
    "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE": False,
    "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE": False,
    "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE": -1,
    "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES": -1,
    "TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS": -1,
    "TASK_CONFIG.SIMULATOR.TYPE": "SoundEventNavSim",
    "TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG": "v0",
    "TASK_CONFIG.SIMULATOR.SCENE_DATASET": "mp3d",
    "TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS": True,
    "TASK_CONFIG.SIMULATOR.GRID_SIZE": 1.0,
    "TASK_CONFIG.SIMULATOR.STEP_TIME": 1.0,
    "TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE": False,
    "TASK_CONFIG.SIMULATOR.RGB_SENSOR.TYPE": "HabitatSimRGBSensor",
    "TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH": 128,
    "TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT": 128,
    "TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.TYPE": "HabitatSimDepthSensor",
    "TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH": 128,
    "TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT": 128,
    # Cached depth frames still pass through HabitatSimDepthSensor's clipping
    # and normalization; these settings affect every policy observation.
    "TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH": 0.0,
    "TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH": 10.0,
    "TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH": True,
    "TASK_CONFIG.SIMULATOR.AUDIO.TYPE": "diff_gd",
    "TASK_CONFIG.SIMULATOR.AUDIO.EVERLASTING": False,
    "TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE": 16000,
    "TASK_CONFIG.SIMULATOR.AUDIO.CROSSFADE": False,
    "TASK_CONFIG.SIMULATOR.AUDIO.HAS_NOISE": False,
    "TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_MODE": SEEDED_MODE,
    "TASK_CONFIG.TASK.TYPE": "SoundEventNav",
    "TASK_CONFIG.TASK.POSSIBLE_ACTIONS": [
        "STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT",
    ],
    "TASK_CONFIG.TASK.SENSORS": [
        "SPECTROGRAM_SENSOR", "CATEGORY", "POSE_SENSOR", "POSE_SENSOR_GD",
    ],
    "TASK_CONFIG.TASK.GOAL_SENSOR_UUID": "spectrogram",
    "TASK_CONFIG.TASK.SUCCESS_DISTANCE": 1.0,
    "TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE": 1.0,
    "TASK_CONFIG.TASK.DISTANCE_TO_GOAL.DISTANCE_TO": "VIEW_POINTS",
    "TASK_CONFIG.TASK.SPECTROGRAM_SENSOR.TYPE": "SoundEventSpectrogramSensor",
    "TASK_CONFIG.TASK.CATEGORY.TYPE": "SoundEventCategory",
    "TASK_CONFIG.TASK.POSE_SENSOR.TYPE": "PoseSensor",
    "TASK_CONFIG.TASK.POSE_SENSOR_GD.TYPE": "PoseSensorGD",
    "TASK_CONFIG.TASK.MEASUREMENTS": [
        "DISTANCE_TO_GOAL", "NORMALIZED_DISTANCE_TO_GOAL", "SUCCESS", "SPL",
        "SOFT_SPL", "NUM_ACTION", "SUCCESS_WEIGHTED_BY_NUM_ACTION",
        "SUCCESS_WHEN_SILENT",
    ],
    "TASK_CONFIG.DATASET.TYPE": "SoundEventNav",
    "TASK_CONFIG.DATASET.SPLIT": "val",
    "TASK_CONFIG.DATASET.VERSION": "v1",
    "TASK_CONFIG.DATASET.TTA_EPISODES_PER_SCENE": 100,
    "TASK_CONFIG.DATASET.TTA_EXPECTED_SCENES": 20,
    "TASK_CONFIG.DATASET.TTA_GLOBAL_SHUFFLE": True,
    "RL.PPO.norm_first": False,
    "RL.PPO.policy_type": "msmt",
    "RL.PPO.hidden_size": 512,
    "RL.PPO.use_external_memory": True,
    "RL.PPO.use_belief_predictor": False,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.memory_size": 159,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.hidden_size": 256,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.nhead": 8,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.num_encoder_layers": 1,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.num_decoder_layers": 1,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.dropout": 0.0,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.activation": "relu",
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.decoder_type": "MSMT",
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.use_pretrained": True,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.pretraining": False,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.use_belief_encoding": False,
    "RL.PPO.SCENE_MEMORY_TRANSFORMER.use_goal_descriptor": True,
    "RL.PPO.SELD_ENCODER.gd_encoder_type": "CRNN",
    "RL.PPO.SELD_ENCODER.use_downsample": True,
    "RL.PPO.SELD_ENCODER.use_goal_similarity": False,
    "RL.DDPPO.pretrained": False,
    "RL.DDPPO.reset_critic": False,
}


def get_value(config, dotted_key, default=None):
    value = config
    for key in dotted_key.split("."):
        if isinstance(value, dict):
            if key not in value:
                return default
            value = value[key]
        else:
            if not hasattr(value, key):
                return default
            value = getattr(value, key)
    return value


def plain_config(value):
    if hasattr(value, "items"):
        return {str(key): plain_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_config(item) for item in value]
    return value


def build_eval_protocol(config):
    profile = str(get_value(config, "EVAL.PROTOCOL_PROFILE", "") or "")
    if profile not in ("", PROFILE):
        raise ValueError("unknown ENMuS evaluation profile: {}".format(profile))
    mode = str(get_value(config, "TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_MODE", LEGACY_MODE))
    if mode not in (LEGACY_MODE, SEEDED_MODE):
        raise ValueError("unknown audio SCHEDULE_MODE: {}".format(mode))
    seed = get_value(config, "TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_SEED", 0)
    if type(seed) is not int or seed < 0:
        raise ValueError("audio SCHEDULE_SEED must be a nonnegative integer")
    dataset_path = str(get_value(config, "TASK_CONFIG.DATASET.DATA_PATH", ""))
    settings = set(dataset_path.replace("\\", "/").split("/")) & {
        "single_source", "multi_source",
    }
    if len(settings) != 1:
        if profile or mode == SEEDED_MODE:
            raise ValueError("cannot identify ENMuS source setting from DATA_PATH")
        source_setting = (
            "multi_source" if get_value(config, "TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND", False)
            else "single_source"
        )
    else:
        source_setting = next(iter(settings))
    expected = dict(EXPECTED)
    expected["TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND"] = source_setting == "multi_source"
    expected["TASK_CONFIG.DATASET.TTA_EPISODE_SEED"] = get_value(config, "SEED")
    expected["TASK_CONFIG.SEED"] = get_value(config, "SEED")
    actual = {key: plain_config(get_value(config, key)) for key in expected}
    measurements = actual["TASK_CONFIG.TASK.MEASUREMENTS"]
    if isinstance(measurements, list):
        actual["TASK_CONFIG.TASK.MEASUREMENTS"] = [
            item for item in measurements if item != AUDIT_MEASURE
        ]
    if profile:
        errors = [
            "{}: expected {!r}, got {!r}".format(key, value, actual[key])
            for key, value in expected.items() if actual[key] != value
        ]
        count = get_value(config, "TEST_EPISODE_COUNT")
        if type(count) is not int or not 1 <= count <= 2000:
            errors.append("TEST_EPISODE_COUNT must be between 1 and 2000")
        if errors:
            raise ValueError("{} rejected resolved evaluation config:\n{}".format(
                PROFILE, "\n".join(errors),
            ))
    actual.update({
        "profile": profile,
        "source_setting": source_setting,
        "SEED": get_value(config, "SEED"),
        "TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_MODE": mode,
        "TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_SEED": seed,
    })
    return {
        "schema": "navtta.avn.enmus_eval_protocol.v1",
        "profile": profile,
        "source_setting": source_setting,
        "semantic_config": actual,
        "semantic_digest": canonical_digest(actual),
        "resolved_config": plain_config(config),
        "declared_deviations_from_clavn": (
            ["strict checkpoint loading"]
            + (["one environment; fixed globally shuffled 2000-episode stream"]
               if actual["NUM_PROCESSES"] == 1
               and actual["TASK_CONFIG.DATASET.TTA_EPISODES_PER_SCENE"] == 100
               and actual["TASK_CONFIG.DATASET.TTA_EXPECTED_SCENES"] == 20
               and actual["TASK_CONFIG.DATASET.TTA_GLOBAL_SHUFFLE"] is True else [])
            + (["independent episode-local audio RNG; not historical audio replay"]
               if mode == SEEDED_MODE else [])
        ),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_audio_schedule(path, protocol, episodes, complete=False):
    semantic = protocol["semantic_config"]
    order = list(episodes)
    payload = {
        "schema": "navtta.avn.audio_schedule.v1",
        "profile": protocol["profile"],
        "source_setting": protocol["source_setting"],
        "semantic_digest": protocol["semantic_digest"],
        "mode": semantic["TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_MODE"],
        "schedule_seed": semantic["TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_SEED"],
        "episode_count": len(episodes),
        "episode_order": order,
        "episode_order_sha256": canonical_digest(order),
        "episodes": episodes,
        "complete": bool(complete),
    }
    write_json(path, payload)
    return payload


def record_completed_episode(records, scene, episode_id, audit, strict=False):
    key = episode_key(scene, episode_id)
    if not isinstance(audit, dict) or episode_key(
        audit.get("scene_id"), audit.get("episode_id"),
    ) != key:
        raise ValueError("terminal audio schedule does not match episode {}".format(key))
    if strict and key in records:
        raise ValueError("duplicate completed audio episode: {}".format(key))
    if not audit.get("roles") or "target" not in audit["roles"]:
        raise ValueError("terminal audio schedule is missing target sound")
    expected_digest = canonical_digest({
        role: record["sha256"] for role, record in audit["roles"].items()
    })
    if audit.get("schedule_sha256") != expected_digest:
        raise ValueError("terminal audio schedule digest mismatch")
    records[key] = audit
