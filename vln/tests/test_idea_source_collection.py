import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_module(
    "build_idea_source_order_manifests",
    "vln/scripts/build_idea_source_order_manifests.py",
)
launcher = _load_module(
    "collect_idea_source_stats",
    "vln/scripts/collect_idea_source_stats.py",
)


class IdeaSourceManifestTest(unittest.TestCase):
    def test_builder_selects_a_deterministic_train_subset_of_128(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "vln/data/duet/R2R/annotations/R2R_train_enc.json"
            dataset.parent.mkdir(parents=True)
            payload = [
                {
                    "path_id": index,
                    "scan": "scene-{}".format(index % 7),
                    "instructions": ["instruction {}".format(index)],
                }
                for index in range(160)
            ]
            encoded = json.dumps(payload).encode("utf-8")
            dataset.write_bytes(encoded)
            assets = root / "source_train_assets.json"
            assets.write_text(json.dumps({
                "assets": [{
                    "id": "duet_hamt_r2r_train_bert",
                    "paths": ["vln/data/duet/R2R/annotations/R2R_train_enc.json"],
                    "size": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }],
            }), encoding="utf-8")

            first = builder.build_manifest(
                "duet-r2r", repo_root=root, asset_manifest=assets
            )
            second = builder.build_manifest(
                "duet-r2r", repo_root=root, asset_manifest=assets
            )
            self.assertEqual(first, second)
            self.assertEqual(first["split"], "train")
            self.assertEqual(first["episode_count"], 128)
            self.assertEqual(first["selection"]["population_count"], 160)
            self.assertEqual(first["selection"]["sample_count"], 128)
            self.assertEqual(len({
                (item["scene_id"], item["episode_id"])
                for item in first["episodes"]
            }), 128)

    def test_builder_rejects_episode_ids_reused_across_scenes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "vln/data/duet/R2R/annotations/R2R_train_enc.json"
            dataset.parent.mkdir(parents=True)
            payload = [
                {"path_id": 7, "scan": "scene-a", "instructions": ["a"]},
                {"path_id": 7, "scan": "scene-b", "instructions": ["b"]},
            ]
            encoded = json.dumps(payload).encode("utf-8")
            dataset.write_bytes(encoded)
            assets = root / "source_train_assets.json"
            assets.write_text(json.dumps({
                "assets": [{
                    "id": "duet_hamt_r2r_train_bert",
                    "paths": [
                        "vln/data/duet/R2R/annotations/R2R_train_enc.json"
                    ],
                    "size": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }],
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "globally unique"):
                builder.build_manifest(
                    "duet-r2r", repo_root=root, asset_manifest=assets
                )


class IdeaSourceLauncherTest(unittest.TestCase):
    def test_prepare_collection_pins_checkpoint_dataset_and_argmax(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"frozen policy")
            results = root / "vln/results/idea_source_statistics"

            def fake_builder(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({
                    "schema": "navtta.episode_order.v1",
                    "benchmark": "r2r_discrete_duet_hamt",
                    "split": "train",
                    "episode_count": 128,
                    "dataset": {
                        "path": "vln/data/source.json",
                        "sha256": "a" * 64,
                    },
                }), encoding="utf-8")
                return mock.Mock(returncode=0)

            with mock.patch.object(launcher, "REPO_ROOT", root), \
                    mock.patch.object(launcher, "RESULTS_ROOT", results), \
                    mock.patch.dict(
                        launcher.CHECKPOINTS,
                        {"duet-r2r": "checkpoint.bin"},
                        clear=False,
                    ), \
                    mock.patch.object(launcher.subprocess, "run", fake_builder):
                record = launcher.prepare_collection("duet-r2r", "unit")

            config = json.loads(Path(record["config"]).read_text(encoding="utf-8"))
            parameters = config["parameters"]
            self.assertEqual(config["namespace"], "idea_source_statistics")
            self.assertEqual(config["stage"], "source_statistics")
            self.assertEqual(parameters["action_selection"], "argmax")
            self.assertEqual(
                parameters["collection_policy"],
                "frozen_source_argmax_rollout",
            )
            self.assertEqual(parameters["source_trajectories"], 128)
            self.assertEqual(
                parameters["source_checkpoint_sha256"],
                hashlib.sha256(b"frozen policy").hexdigest(),
            )
            self.assertEqual(parameters["source_dataset"], "vln/data/source.json")
            self.assertEqual(
                parameters["source_dataset_version"], "sha256:" + "a" * 64
            )
            self.assertIn("--source-order-manifest", record["command"])

    def test_apply_specs_requires_all_settings(self):
        with self.assertRaisesRegex(launcher.CollectionError, "all eight"):
            launcher.apply_bindings_to_specs({"duet-r2r": {}})


if __name__ == "__main__":
    unittest.main()
