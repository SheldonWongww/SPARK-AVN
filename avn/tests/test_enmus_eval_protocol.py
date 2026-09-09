import copy
import json
import tempfile
import unittest
from pathlib import Path

from avn.navtta_avn.audio_schedule import (
    canonical_digest, canonical_scene_id, episode_schedule_record,
    generate_sound_intervals, random_duration, schedule_record, schedule_rng,
    schedule_seed,
)
from avn.navtta_avn.enmus_eval_protocol import (
    PROFILE, build_eval_protocol, record_completed_episode, write_audio_schedule,
)

try:
    import numpy as np
except ImportError:
    np = None


def aligned_config():
    return {
        "SEED": 0, "NUM_PROCESSES": 1, "TEST_EPISODE_COUNT": 2000,
        "ENV_NAME": "AudioNavRLEnv", "USE_VECENV": True, "USE_SYNC_VECENV": False,
        "CONTINUOUS": False,
        "SENSORS": ["DEPTH_SENSOR", "RGB_SENSOR"],
        "EVAL": {"PROTOCOL_PROFILE": PROFILE, "SPLIT": "val",
                 "USE_CKPT_CONFIG": False, "ACTION_SELECTION": "sample"},
        "TASK_CONFIG": {
            "SEED": 0,
            "ENVIRONMENT": {"MAX_EPISODE_STEPS": 500, "ITERATOR_OPTIONS": {
                "CYCLE": True, "SHUFFLE": False, "GROUP_BY_SCENE": False,
                "NUM_EPISODE_SAMPLE": -1, "MAX_SCENE_REPEAT_EPISODES": -1,
                "MAX_SCENE_REPEAT_STEPS": -1,
            }},
            "SIMULATOR": {
                "TYPE": "SoundEventNavSim", "ACTION_SPACE_CONFIG": "v0",
                "SCENE_DATASET": "mp3d", "USE_RENDERED_OBSERVATIONS": True,
                "GRID_SIZE": 1.0, "STEP_TIME": 1.0,
                "CONTINUOUS_VIEW_CHANGE": False,
                "RGB_SENSOR": {"TYPE": "HabitatSimRGBSensor", "WIDTH": 128, "HEIGHT": 128},
                "DEPTH_SENSOR": {"TYPE": "HabitatSimDepthSensor", "WIDTH": 128, "HEIGHT": 128,
                                 "MIN_DEPTH": 0.0, "MAX_DEPTH": 10.0, "NORMALIZE_DEPTH": True},
                "AUDIO": {"TYPE": "diff_gd", "EVERLASTING": False,
                          "RIR_SAMPLING_RATE": 16000, "CROSSFADE": False,
                          "HAS_NOISE": False, "HAS_DISTRACTOR_SOUND": False,
                          "SCHEDULE_MODE": "episode_seeded_v1", "SCHEDULE_SEED": 0},
            },
            "TASK": {
                "TYPE": "SoundEventNav", "GOAL_SENSOR_UUID": "spectrogram",
                "POSSIBLE_ACTIONS": ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"],
                "SENSORS": ["SPECTROGRAM_SENSOR", "CATEGORY", "POSE_SENSOR", "POSE_SENSOR_GD"],
                "SUCCESS_DISTANCE": 1.0, "SUCCESS": {"SUCCESS_DISTANCE": 1.0},
                "DISTANCE_TO_GOAL": {"DISTANCE_TO": "VIEW_POINTS"},
                "SPECTROGRAM_SENSOR": {"TYPE": "SoundEventSpectrogramSensor"},
                "CATEGORY": {"TYPE": "SoundEventCategory"},
                "POSE_SENSOR": {"TYPE": "PoseSensor"},
                "POSE_SENSOR_GD": {"TYPE": "PoseSensorGD"},
                "MEASUREMENTS": ["DISTANCE_TO_GOAL", "NORMALIZED_DISTANCE_TO_GOAL", "SUCCESS",
                                 "SPL", "SOFT_SPL", "NUM_ACTION", "SUCCESS_WEIGHTED_BY_NUM_ACTION",
                                 "SUCCESS_WHEN_SILENT"],
            },
            "DATASET": {
                "TYPE": "SoundEventNav", "SPLIT": "val", "VERSION": "v1",
                "DATA_PATH": "data/datasets/tta_test/single_source/mp3d/v1/val/val.json.gz",
                "TTA_EPISODES_PER_SCENE": 100, "TTA_EXPECTED_SCENES": 20,
                "TTA_GLOBAL_SHUFFLE": True, "TTA_EPISODE_SEED": 0,
            },
        },
        "RL": {"PPO": {
            "norm_first": False, "policy_type": "msmt", "hidden_size": 512,
            "use_external_memory": True, "use_belief_predictor": False,
            "SCENE_MEMORY_TRANSFORMER": {
                "memory_size": 159, "hidden_size": 256, "nhead": 8,
                "num_encoder_layers": 1, "num_decoder_layers": 1,
                "dropout": 0.0, "activation": "relu", "decoder_type": "MSMT",
                "use_pretrained": True, "pretraining": False,
                "use_belief_encoding": False, "use_goal_descriptor": True,
            },
            "SELD_ENCODER": {"gd_encoder_type": "CRNN", "use_downsample": True,
                             "use_goal_similarity": False},
        }, "DDPPO": {"pretrained": False, "reset_critic": False}},
    }


class EnmusEvalProtocolTest(unittest.TestCase):
    def test_explicit_profile_accepts_matching_single_and_multi(self):
        config = aligned_config()
        self.assertEqual(build_eval_protocol(config)["source_setting"], "single_source")
        config["TASK_CONFIG"]["DATASET"]["DATA_PATH"] = "/new/root/multi_source/mp3d/v1/val/val.json.gz"
        config["TASK_CONFIG"]["SIMULATOR"]["AUDIO"]["HAS_DISTRACTOR_SOUND"] = True
        self.assertEqual(build_eval_protocol(config)["source_setting"], "multi_source")

    def test_profile_rejects_semantic_mismatches(self):
        cases = [
            ("NUM_PROCESSES", 8), ("EVAL.ACTION_SELECTION", "argmax"),
            ("CONTINUOUS", True),
            ("TASK_CONFIG.TASK.POSSIBLE_ACTIONS", ["STOP", "TURN_LEFT", "MOVE_FORWARD", "TURN_RIGHT"]),
            ("EVAL.USE_CKPT_CONFIG", True), ("EVAL.SPLIT", "train"),
            ("TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS", 300),
            ("TASK_CONFIG.SIMULATOR.RGB_SENSOR.TYPE", "HabitatSimSemanticSensor"),
            ("TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.TYPE", "HabitatSimRGBSensor"),
            ("TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH", 0.1),
            ("TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH", 5.0),
            ("TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH", False),
            ("TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE", 22050),
            ("TASK_CONFIG.SIMULATOR.AUDIO.EVERLASTING", True),
            ("TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND", True),
            ("TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_MODE", "legacy_global"),
            ("TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE", 3.0),
            ("TASK_CONFIG.TASK.DISTANCE_TO_GOAL.DISTANCE_TO", "POINT"),
            ("TASK_CONFIG.TASK.POSE_SENSOR_GD.TYPE", "PoseSensor"),
            ("RL.PPO.SCENE_MEMORY_TRANSFORMER.memory_size", 150),
        ]
        for dotted_key, value in cases:
            with self.subTest(key=dotted_key):
                config = aligned_config()
                node = config
                keys = dotted_key.split(".")
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                with self.assertRaisesRegex(ValueError, keys[-1]):
                    build_eval_protocol(config)

    def test_legacy_config_without_new_fields_is_accepted(self):
        protocol = build_eval_protocol({"NUM_PROCESSES": 8})
        self.assertEqual(protocol["profile"], "")
        self.assertEqual(protocol["semantic_config"]["TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_MODE"], "legacy_global")
        with self.assertRaisesRegex(ValueError, "unknown ENMuS"):
            build_eval_protocol({"EVAL": {"PROTOCOL_PROFILE": "typo"}})

    def test_semantic_digest_matches_source_tta_and_smoke(self):
        source = aligned_config()
        target = copy.deepcopy(source)
        target["TTA"] = {"METHOD": "feedtta", "LR": 1e-8}
        target["TEST_EPISODE_COUNT"] = 20
        target["TASK_CONFIG"]["TASK"]["MEASUREMENTS"].append("AUDIO_SCHEDULE_AUDIT")
        target["TASK_CONFIG"]["DATASET"]["DATA_PATH"] = "/other/machine/single_source/mp3d/v1/val/val.json.gz"
        self.assertEqual(build_eval_protocol(source)["semantic_digest"], build_eval_protocol(target)["semantic_digest"])
        target["TASK_CONFIG"]["SIMULATOR"]["AUDIO"]["SCHEDULE_SEED"] = 1
        self.assertNotEqual(build_eval_protocol(source)["semantic_digest"], build_eval_protocol(target)["semantic_digest"])

    def test_terminal_identity_and_digests_reject_next_episode(self):
        target = schedule_record([1, 0, 0], 31, {})
        record = episode_schedule_record("/root/scene/scene.glb", "7", {"target": target})
        records = {}
        record_completed_episode(records, "scene", "7", record, strict=True)
        with self.assertRaisesRegex(ValueError, "does not match"):
            record_completed_episode({}, "scene", "8", record, strict=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            record_completed_episode(records, "scene", "7", record, strict=True)
        damaged = copy.deepcopy(record)
        damaged["roles"]["target"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            record_completed_episode({}, "scene", "7", damaged, strict=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio_schedule_0.json"
            write_audio_schedule(path, build_eval_protocol(aligned_config()), records, complete=True)
            result = json.loads(path.read_text())
        self.assertEqual(result["episode_count"], 1)
        self.assertEqual(result["episode_order"], ["scene/7"])
        self.assertEqual(result["episodes"]["scene/7"]["schedule_sha256"], record["schedule_sha256"])
        self.assertEqual(result["episode_order_sha256"], canonical_digest(["scene/7"]))

    def test_identity_is_path_independent_and_requires_episode_id(self):
        self.assertEqual(canonical_scene_id("/machine/data/scene/scene.glb"), "scene")
        args = (0, "single_source", "/machine/scene/scene.glb", "0", "target")
        self.assertEqual(schedule_seed(*args), schedule_seed(0, "single_source", "scene", "0", "target"))
        for episode_id in (None, ""):
            with self.assertRaisesRegex(ValueError, "episode id"):
                schedule_seed(0, "single_source", "scene", episode_id, "target")

    def test_original_abs_integer_clipping_order(self):
        class Draw:
            def __init__(self, value):
                self.value = value
            def normal(self, mean, sigma):
                return self.value
        self.assertEqual(random_duration(3, 8, 2, Draw(-3.9)), 3)
        self.assertEqual(random_duration(3, 8, 2, Draw(0.2)), 1)
        self.assertEqual(random_duration(3, 8, 2, Draw(20.2)), 8)
        self.assertEqual(random_duration(3, 8, 2, Draw(1.2)), 2)


@unittest.skipIf(np is None, "requires the task environment's NumPy")
class AudioScheduleNumpyTest(unittest.TestCase):
    PARAMETERS = (7, 410, 3, 8, 20, 1)

    @staticmethod
    def old_formula(parameters, rng):
        offset, duration, audio_length, mean, upper, lower = parameters
        result = [0] * offset
        if mean == -1:
            result.extend([1] * duration)
        else:
            while len(result) < duration + offset and len(result) < 500:
                result.extend([1] * audio_length)
                sigma = min(np.abs(mean - upper), np.abs(mean - lower)) / 2
                length = int(np.abs(rng.normal(mean, sigma)))
                if length == 0:
                    length = 1
                elif length > upper:
                    length = upper
                elif length < lower:
                    length = lower
                result.extend([0] * length)
        result = result[:duration + offset]
        if len(result) < 500:
            result.extend([0] * (500 - len(result)))
        result.extend([0])
        return result

    def test_exact_original_formula_and_rng_consumption(self):
        for parameters in (self.PARAMETERS, (0, 500, 2, -1, 0, 0), (2, 9, 1, 1, 1, 0), (4, 498, 7, 3, 9, 2)):
            with self.subTest(parameters=parameters):
                old_rng, new_rng = np.random.RandomState(19), np.random.RandomState(19)
                self.assertEqual(self.old_formula(parameters, old_rng), generate_sound_intervals(*parameters, new_rng))
                self.assertEqual(old_rng.normal(), new_rng.normal())

    def test_seeded_schedules_ignore_order_workers_and_global_rng(self):
        def run(order, workers):
            result = {}
            for index, episode in enumerate(order):
                np.random.seed(index % workers)
                np.random.normal(size=index + 1)
                state = np.random.get_state()
                rng, _ = schedule_rng("episode_seeded_v1", 0, "multi_source", "scene", episode, "target")
                result[episode] = generate_sound_intervals(*self.PARAMETERS, rng)
                after = np.random.get_state()
                self.assertEqual(state[0], after[0])
                np.testing.assert_array_equal(state[1], after[1])
                self.assertEqual(state[2:], after[2:])
            return result
        self.assertEqual(run(["1", "2", "3"], 1), run(["3", "1", "2"], 8))

    def test_identity_axes_and_roles_are_independent(self):
        identities = [
            (0, "single_source", "scene", "7", "target"),
            (1, "single_source", "scene", "7", "target"),
            (0, "multi_source", "scene", "7", "target"),
            (0, "single_source", "other", "7", "target"),
            (0, "single_source", "scene", "8", "target"),
            (0, "single_source", "scene", "7", "distractor"),
        ]
        digests = []
        for identity in identities:
            rng, derived = schedule_rng("episode_seeded_v1", *identity)
            intervals = generate_sound_intervals(*self.PARAMETERS, rng)
            digests.append(schedule_record(intervals, derived, {})["sha256"])
        self.assertEqual(len(set(digests)), len(identities))
        target_rng, _ = schedule_rng("episode_seeded_v1", *identities[0])
        distractor_rng, _ = schedule_rng("episode_seeded_v1", *identities[-1])
        generate_sound_intervals(*self.PARAMETERS, distractor_rng)
        self.assertEqual(schedule_record(generate_sound_intervals(*self.PARAMETERS, target_rng), None, {})["sha256"], digests[0])

    def test_legacy_mode_uses_original_global_stream(self):
        np.random.seed(23)
        expected = self.old_formula(self.PARAMETERS, np.random)
        next_value = np.random.normal()
        np.random.seed(23)
        rng, derived = schedule_rng("legacy_global", 0, "", "", "", "target")
        self.assertIs(rng, np.random)
        self.assertIsNone(derived)
        self.assertEqual(generate_sound_intervals(*self.PARAMETERS, rng), expected)
        self.assertEqual(np.random.normal(), next_value)

    def test_schedule_generation_leaves_torch_rng_unchanged(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires task environment's PyTorch")
        before = torch.random.get_rng_state().clone()
        rng, _ = schedule_rng("episode_seeded_v1", 0, "single_source", "scene", "7", "target")
        generate_sound_intervals(*self.PARAMETERS, rng)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
