import gzip
import json
import os
import tempfile
import unittest

from vln.scripts import verify_preflight

from navtta_core.experiment.episode_order import (
    build_episode_order_manifest,
    sha256_file,
)


class EpisodeManifestDatasetPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repo_root = self.temporary_directory.name

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _path(self, relative):
        path = os.path.join(self.repo_root, relative)
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        return path

    def _write_dataset(self, relative, payload):
        path = self._path(relative)
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream)
        return path

    def _write_manifest(self, relative, dataset_relative, episodes, source_id_field):
        dataset_path = os.path.join(self.repo_root, dataset_relative)
        manifest = build_episode_order_manifest(
            episodes,
            benchmark="synthetic-vln",
            split="val_seen",
            dataset_path=dataset_relative,
            dataset_sha256=sha256_file(dataset_path),
            source_id_field=source_id_field,
        )
        manifest_path = self._path(relative)
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream)
        return manifest_path, manifest

    def test_plain_json_root_list_matches_canonical_manifest(self):
        dataset_episodes = [
            {
                "path_id": 10,
                "scan": "scene-b",
                "instructions": ["first", "second"],
            },
            {"path_id": 2, "scan": "scene-b", "instructions": ["third"]},
            {
                "path_id": 3,
                "scene_id": "scenes/scene-a.glb",
                "instructions": ["fourth"],
            },
        ]
        manifest_episodes = [
            {"episode_id": "10_0", "scene_id": "scene-b"},
            {"episode_id": "10_1", "scene_id": "scene-b"},
            {"episode_id": "2_0", "scene_id": "scene-b"},
            {"episode_id": "3_0", "scene_id": "scene-a"},
        ]
        dataset_relative = "data/list.json"
        self._write_dataset(dataset_relative, dataset_episodes)
        manifest_path, _ = self._write_manifest(
            "orders/val_seen.json",
            dataset_relative,
            manifest_episodes,
            "instr_id",
        )

        errors = []
        verify_preflight.check_episode_manifest(
            self.repo_root, manifest_path, errors
        )

        self.assertEqual(errors, [])

    def test_gzip_json_root_dict_matches_canonical_manifest(self):
        episodes = [
            {"episode_id": 11, "scene_id": "data/scene-b/scene-b.glb"},
            {"episode_id": 2, "scene_id": "data/scene-a/scene-a.glb"},
        ]
        dataset_relative = "data/dict.json.gz"
        self._write_dataset(dataset_relative, {"episodes": episodes})
        manifest_path, _ = self._write_manifest(
            "orders/val_seen.json",
            dataset_relative,
            episodes,
            "episode_id",
        )

        errors = []
        verify_preflight.check_episode_manifest(
            self.repo_root, manifest_path, errors
        )

        self.assertEqual(errors, [])

    def test_dataset_episode_mismatch_is_rejected_after_matching_sha(self):
        dataset_episodes = [
            {"episode_id": "2", "scene_id": "scene-a.glb"},
            {"episode_id": "10", "scene_id": "scene-a.glb"},
        ]
        manifest_episodes = [
            {"episode_id": "2", "scene_id": "scene-a.glb"},
            {"episode_id": "11", "scene_id": "scene-a.glb"},
        ]
        dataset_relative = "data/mismatch.json"
        self._write_dataset(dataset_relative, {"episodes": dataset_episodes})
        manifest_path, _ = self._write_manifest(
            "orders/val_seen.json",
            dataset_relative,
            manifest_episodes,
            "episode_id",
        )

        errors = []
        verify_preflight.check_episode_manifest(
            self.repo_root, manifest_path, errors
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("dataset episode records mismatch", errors[0])
        self.assertIn("record 1", errors[0])

    def test_dataset_sha_mismatch_is_rejected_even_when_records_match(self):
        episodes = [{"episode_id": "1", "scene_id": "scene-a.glb"}]
        dataset_relative = "data/changed.json"
        dataset_path = self._write_dataset(
            dataset_relative, {"episodes": episodes}
        )
        manifest_path, _ = self._write_manifest(
            "orders/val_seen.json",
            dataset_relative,
            episodes,
            "episode_id",
        )
        with open(dataset_path, "a", encoding="utf-8") as stream:
            stream.write("\n")

        errors = []
        verify_preflight.check_episode_manifest(
            self.repo_root, manifest_path, errors
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("Dataset SHA256 mismatch", errors[0])

    def test_missing_dataset_is_rejected(self):
        dataset_relative = "data/missing.json"
        manifest = build_episode_order_manifest(
            [{"episode_id": "1", "scene_id": "scene-a.glb"}],
            benchmark="synthetic-vln",
            split="val_seen",
            dataset_path=dataset_relative,
            dataset_sha256="a" * 64,
            source_id_field="episode_id",
        )
        manifest_path = self._path("orders/val_seen.json")
        with open(manifest_path, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream)

        errors = []
        verify_preflight.check_episode_manifest(
            self.repo_root, manifest_path, errors
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("dataset is missing", errors[0])


if __name__ == "__main__":
    unittest.main()
