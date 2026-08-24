import ast
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from avn.navtta_avn.idea_source import (
    collection_provenance,
    file_sha256,
    load_source_manifest,
    select_manifest_episodes,
    verify_manifest_assets,
)


def _load_builder():
    path = Path(__file__).parents[1] / "scripts" / "build_idea_source_manifest.py"
    spec = importlib.util.spec_from_file_location("idea_source_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Episode:
    def __init__(self, scene_id, episode_id):
        self.scene_id = "data/mp3d/{0}/{0}.glb".format(scene_id)
        self.episode_id = episode_id


class IdeaSourceManifestTest(unittest.TestCase):
    def test_build_validate_and_select_exact_128(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "single_source/mp3d/v1/train"
            (root / "content").mkdir(parents=True)
            with gzip.open(root / "train.json.gz", "wt") as stream:
                json.dump({
                    "content_scenes_path": "{data_path}/content/{scene}.json.gz",
                    "episodes": [],
                }, stream)
            episodes = [
                {
                    "scene_id": "data/mp3d/s{0}/s{0}.glb".format(index % 4),
                    "episode_id": str(index),
                }
                for index in range(140)
            ]
            with gzip.open(root / "content" / "episodes.json.gz", "wt") as stream:
                json.dump({"episodes": episodes}, stream)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"frozen checkpoint")
            payload = builder.build_manifest(
                root / "train.json.gz", checkpoint,
                "smt_audio", "single_source",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            manifest, _ = load_source_manifest(
                manifest_path,
                file_sha256(manifest_path),
                model="smt_audio",
                source_setting="single_source",
            )
            verify_manifest_assets(
                manifest, root / "train.json.gz", checkpoint
            )
            provenance = collection_provenance(
                manifest, file_sha256(manifest_path), "b" * 64
            )
            self.assertEqual(provenance["model_state_sha256"], "b" * 64)
            self.assertEqual(
                provenance["episode_selection_manifest_sha256"],
                file_sha256(manifest_path),
            )
            objects = [
                _Episode("s{}".format(index % 4), str(index))
                for index in range(140)
            ]
            self.assertEqual(
                len(select_manifest_episodes(objects, manifest)), 128
            )

    def test_builder_rejects_non_train_or_wrong_setting_path(self):
        builder = _load_builder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "single_source/mp3d/v1/val/val.json.gz"
            dataset.parent.mkdir(parents=True)
            with gzip.open(dataset, "wt") as stream:
                json.dump({"episodes": []}, stream)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaisesRegex(ValueError, "canonical AVN IDEA"):
                builder.build_manifest(
                    dataset, checkpoint, "smt_audio", "multi_source"
                )

    def test_missing_manifest_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            load_source_manifest("missing.json", "")

    def test_smt_trainer_emits_source_collection_diagnostics(self):
        trainer = Path(__file__).parents[1] / (
            "baselines/smt_audio/ss_baselines/savi/ppo/ppo_trainer.py"
        )
        source = trainer.read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn("source_setting=source_setting", source)
        self.assertIn("diagnostics = source_collection.diagnostics()", source)
        self.assertIn('diagnostics["source_policy_frozen"] = True', source)
        self.assertIn("expected_source_provenance={", source)
        self.assertIn('"model_state_sha256": module_state_sha256(actor_critic)', source)
        self.assertIn('"episode_selection_manifest_sha256": (', source)
        self.assertIn(
            "source collection cannot consume an IDEA source artifact",
            source,
        )


if __name__ == "__main__":
    unittest.main()
