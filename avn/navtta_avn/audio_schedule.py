"""CL-AVN audio timing with an explicit, episode-local random stream."""

import hashlib
import json
import os


LEGACY_MODE = "legacy_global"
SEEDED_MODE = "episode_seeded_v1"
ROLES = ("target", "distractor", "noise")


def canonical_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def canonical_scene_id(scene):
    value = str(scene or "").replace("\\", "/").rstrip("/")
    if not value:
        raise ValueError("audio schedule requires an explicit scene id")
    return os.path.splitext(value.rsplit("/", 1)[-1])[0]


def episode_key(scene, episode_id):
    if episode_id is None or str(episode_id) == "":
        raise ValueError("audio schedule requires an explicit episode id")
    return "{}/{}".format(canonical_scene_id(scene), str(episode_id))


def schedule_seed(seed, source_setting, scene, episode_id, role):
    if type(seed) is not int or seed < 0:
        raise ValueError("audio SCHEDULE_SEED must be a nonnegative integer")
    if source_setting not in ("single_source", "multi_source"):
        raise ValueError("seeded audio schedule requires an explicit source setting")
    if role not in ROLES:
        raise ValueError("unknown audio source role: {}".format(role))
    identity = [SEEDED_MODE, seed, source_setting, episode_key(scene, episode_id), role]
    return int(canonical_digest(identity)[:8], 16)


def schedule_rng(mode, seed, source_setting, scene, episode_id, role):
    import numpy as np

    if mode == LEGACY_MODE:
        return np.random, None
    if mode != SEEDED_MODE:
        raise ValueError("unknown audio SCHEDULE_MODE: {}".format(mode))
    derived_seed = schedule_seed(seed, source_setting, scene, episode_id, role)
    return np.random.RandomState(derived_seed), derived_seed


def random_duration(mean, upper, lower, rng):
    sigma = min(abs(mean - upper), abs(mean - lower)) / 2
    duration = int(abs(rng.normal(mean, sigma)))
    if duration == 0:
        duration = 1
    elif duration > upper:
        duration = upper
    elif duration < lower:
        duration = lower
    return duration


def generate_sound_intervals(offset, duration, audio_length, interval_mean,
                             interval_upper_limit, interval_lower_limit, rng):
    intervals = [0] * offset
    if interval_mean == -1:
        intervals.extend([1] * duration)
    else:
        while len(intervals) < duration + offset and len(intervals) < 500:
            intervals.extend([1] * audio_length)
            intervals.extend([0] * random_duration(
                interval_mean, interval_upper_limit, interval_lower_limit, rng,
            ))
    intervals = intervals[:duration + offset]
    if len(intervals) < 500:
        intervals.extend([0] * (500 - len(intervals)))
    intervals.append(0)
    return intervals


def schedule_record(intervals, derived_seed, parameters):
    return {
        "sha256": hashlib.sha256(bytes(intervals)).hexdigest(),
        "length": len(intervals),
        "rng_seed": derived_seed,
        "active_steps": sum(intervals),
        "parameters": parameters,
    }


def episode_schedule_record(scene, episode_id, roles):
    return {
        "scene_id": canonical_scene_id(scene),
        "episode_id": str(episode_id),
        "schedule_sha256": canonical_digest({
            role: record["sha256"] for role, record in roles.items()
        }),
        "roles": dict(roles),
    }
