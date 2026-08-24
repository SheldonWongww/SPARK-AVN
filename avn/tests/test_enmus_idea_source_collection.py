import ast
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from avn.navtta_avn.idea_source import file_sha256, load_source_manifest


ROOT = Path(__file__).resolve().parents[2]


def _builder():
    path = ROOT / "avn" / "scripts" / "build_idea_source_manifest.py"
    spec = importlib.util.spec_from_file_location("idea_source_builder_enmus", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnmusIdeaSourceCollectionTest(unittest.TestCase):
    def test_enmus_manifest_is_digest_pinned_and_has_128_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory) / "multi_source/mp3d/v1/train"
            (directory / "content").mkdir(parents=True)
            with gzip.open(directory / "train.json.gz", "wt") as stream:
                json.dump({
                    "content_scenes_path": "{data_path}/content/{scene}.json.gz",
                    "episodes": [],
                }, stream)
            episodes = [
                {
                    "scene_id": "data/mp3d/scene/scene.glb",
                    "episode_id": str(index),
                }
                for index in range(130)
            ]
            with gzip.open(directory / "content" / "scene.json.gz", "wt") as stream:
                json.dump({"episodes": episodes}, stream)
            checkpoint = directory / "enmus.pth"
            checkpoint.write_bytes(b"enmus checkpoint")
            payload = _builder().build_manifest(
                directory / "train.json.gz",
                checkpoint,
                "enmus",
                "multi_source",
            )
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            loaded, _ = load_source_manifest(
                manifest,
                file_sha256(manifest),
                model="enmus",
                source_setting="multi_source",
            )
            self.assertEqual(loaded["episode_count"], 128)
            self.assertEqual(loaded["dataset"]["split"], "train")
            self.assertEqual(
                loaded["selection"]["action_selection"], "argmax"
            )

    def test_enmus_trainer_has_fail_closed_collection_contract(self):
        trainer = ROOT / (
            "avn/baselines/enmus/sen_baselines/enmus/ddppo/"
            "ddppo_enmus_trainer.py"
        )
        source = trainer.read_text(encoding="utf-8")
        ast.parse(source)
        for required in (
            "SourceStatisticsCollectionSession",
            "TransformerFusionProtocol.for_source_collection",
            "verify_manifest_assets",
            "EVAL.SPLIT=train",
            "native argmax actions",
            "TEST_EPISODE_COUNT",
            "source policy state changed during IDEA collection",
            "expected_source_provenance={",
            '"model_state_sha256": module_state_sha256(actor_critic)',
            '"episode_selection_manifest_sha256": (',
        ):
            self.assertIn(required, source)

    def test_launcher_binds_the_same_manifest_to_trainer_and_dataset(self):
        launcher = (
            ROOT / "avn/scripts/collect_enmus_idea_source_stats.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("TTA.IDEA.SOURCE_EPISODE_MANIFEST", launcher)
        self.assertIn(
            "TASK_CONFIG.DATASET.IDEA_SOURCE_EPISODE_MANIFEST", launcher
        )
        self.assertIn("ITERATOR_OPTIONS.SHUFFLE False", launcher)


if __name__ == "__main__":
    unittest.main()
