import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from vln.scripts import verify_bevbert_source_pairing as pairing


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BevbertSourcePairingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary_directory.name)
        self._build_valid_fixture()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_bytes(self, relative, content):
        path = self.repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _write_json(self, relative, document):
        path = self.repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def _write_gzip_json(self, relative, document):
        path = self.repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(document, stream)
        return path

    def _dataset_records(self, count, scene):
        return [
            {
                "episode_id": str(index),
                "scene_id": "data/scene_datasets/mp3d/{0}/{0}.glb".format(scene),
            }
            for index in range(count)
        ]

    def _asset_entry(
        self,
        asset_id,
        declared_path,
        source,
        required_for,
        actual_path=None,
        **extra
    ):
        actual_path = Path(actual_path or (self.repo_root / declared_path))
        entry = {
            "id": asset_id,
            "path": declared_path,
            "source": source,
            "required_for": required_for,
            "size": actual_path.stat().st_size,
            "sha256": _sha256(actual_path),
        }
        entry.update(extra)
        return entry

    def _build_valid_fixture(self):
        assets = []
        self.weight_specs = copy.deepcopy(pairing.WEIGHT_ASSETS)
        for index, (asset_id, spec) in enumerate(self.weight_specs.items()):
            path = self._write_bytes(
                spec["path"], (asset_id + str(index)).encode("utf-8")
            )
            spec["size"] = path.stat().st_size
            spec["sha256"] = _sha256(path)
            assets.append(
                self._asset_entry(
                    asset_id,
                    spec["path"],
                    spec["source"],
                    ["bevbert_ce"],
                )
            )

        self.clip_runtime_path = self._write_bytes(
            "runtime/clip/ViT-B-16.pt", b"synthetic-clip"
        )
        self.clip_spec = copy.deepcopy(pairing.CLIP_ASSET)
        self.clip_spec["size"] = self.clip_runtime_path.stat().st_size
        self.clip_spec["sha256"] = _sha256(self.clip_runtime_path)
        assets.append(
            self._asset_entry(
                self.clip_spec["id"],
                self.clip_spec["path"],
                self.clip_spec["source"],
                ["bevbert_ce"],
                actual_path=self.clip_runtime_path,
            )
        )

        self.split_specs = copy.deepcopy(pairing.SPLIT_SPECS)
        for split, spec in self.split_specs.items():
            raw_records = self._dataset_records(spec["episodes"], "scene-" + split)
            dataset_path = self._write_gzip_json(
                spec["path"], {"episodes": raw_records}
            )
            spec["size"] = dataset_path.stat().st_size
            spec["sha256"] = _sha256(dataset_path)
            canonical = pairing._canonical_records(raw_records, split)
            manifest = {
                "schema": pairing.MANIFEST_SCHEMA,
                "benchmark": pairing.PROTOCOL,
                "split": split,
                "split_ordinal": spec["ordinal"],
                "order_policy": pairing.ORDER_POLICY,
                "source_id_field": pairing.SOURCE_ID_FIELD,
                "dataset": {"path": spec["path"], "sha256": spec["sha256"]},
                "episode_count": spec["episodes"],
                "order_sha256": pairing._records_sha256(canonical),
                "episodes": canonical,
            }
            self._write_json(pairing.ORDER_MANIFEST_ROOT / (split + ".json"), manifest)
            assets.append(
                self._asset_entry(
                    spec["asset_id"],
                    spec["path"],
                    "etpnav_assets",
                    ["etpnav", "bevbert_ce"],
                    episodes=spec["episodes"],
                )
            )

            gt_spec = spec["gt"]
            gt_path = self._write_gzip_json(
                gt_spec["path"], {"split": split, "ground_truth": True}
            )
            gt_spec["size"] = gt_path.stat().st_size
            gt_spec["sha256"] = _sha256(gt_path)
            assets.append(
                self._asset_entry(
                    gt_spec["asset_id"],
                    gt_spec["path"],
                    gt_spec["source"],
                    ["etpnav", "bevbert_ce"],
                )
            )

        self.eval_assets_path = self._write_json(
            pairing.ASSET_MANIFEST,
            {"schema_version": 1, "assets": assets},
        )

    @contextmanager
    def _fixture_contract(self, split_specs=None):
        with mock.patch.object(pairing, "WEIGHT_ASSETS", self.weight_specs), mock.patch.object(
            pairing, "CLIP_ASSET", self.clip_spec
        ), mock.patch.object(
            pairing, "SPLIT_SPECS", split_specs or self.split_specs
        ):
            yield

    def _verify_fixture(self, split_specs=None):
        with self._fixture_contract(split_specs=split_specs):
            return pairing.verify_pairing(
                self.repo_root, clip_path=self.clip_runtime_path
            )

    def _read_json(self, relative):
        return json.loads((self.repo_root / relative).read_text(encoding="utf-8"))

    def _rewrite_eval_assets(self, transform):
        document = self._read_json(pairing.ASSET_MANIFEST)
        transform(document)
        self._write_json(pairing.ASSET_MANIFEST, document)

    def _mutate_same_size(self, path):
        path = Path(path)
        content = path.read_bytes()
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])

    def test_project_contract_freezes_every_critical_asset_identity(self):
        expected = {
            "bevbert_ce_checkpoint": (
                2551084857,
                "70dfdfff153f9d54888215e492dae6bff69b674616f32c94a4d51538a67cc0e8",
            ),
            "etpnav_waypoint_predictor": (
                201954133,
                "09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f",
            ),
            "ce_ddppo_depth_encoder": (
                49853716,
                "a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad",
            ),
            "openai_clip_vit_b16": (
                350837078,
                "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
            ),
            "etpnav_val_seen_bertidx": (
                758273,
                "83b231674ae3a18cb02b3d65847377e978f45cc35682f807c03becc38b09410c",
            ),
            "etpnav_val_unseen_bertidx": (
                928029,
                "7db0a39bef374cf5ce986473f8b35c9721f962c6c1a9af069257aa710df16365",
            ),
            "etpnav_val_seen_gt": (
                482513,
                "2c1df3c1f857b5f974192ea60da5d7dc71ccdd61cdfbdcd004258d5c57b2d0d0",
            ),
            "etpnav_val_unseen_gt": (
                1618489,
                "1b9497dc1aa6ab68073976f9678ac42a154b393edd9d0b0450d0ac90841f684c",
            ),
        }
        actual = {
            asset_id: (spec["size"], spec["sha256"])
            for asset_id, spec in pairing.WEIGHT_ASSETS.items()
        }
        actual[pairing.CLIP_ASSET["id"]] = (
            pairing.CLIP_ASSET["size"],
            pairing.CLIP_ASSET["sha256"],
        )
        for spec in pairing.SPLIT_SPECS.values():
            actual[spec["asset_id"]] = (spec["size"], spec["sha256"])
            actual[spec["gt"]["asset_id"]] = (
                spec["gt"]["size"],
                spec["gt"]["sha256"],
            )
        self.assertEqual(actual, expected)
        self.assertEqual(
            pairing.DEFAULT_CLIP_RUNTIME_PATH,
            Path("/data1/wxy/exp_data/NavTTA/vln/cache/clip/ViT-B-16.pt"),
        )

    def test_valid_pairing_verifies_one_checkpoint_for_both_v12_splits(self):
        report = self._verify_fixture()

        self.assertEqual(report["protocol"], pairing.PROTOCOL)
        self.assertEqual(report["shared_checkpoint"], "bevbert_ce_checkpoint")
        self.assertEqual(report["shared_splits"], ["val_seen", "val_unseen"])
        self.assertEqual(
            [item["episodes"] for item in report["splits"]], [778, 1839]
        )
        self.assertEqual(report["clip"]["path"], str(self.clip_runtime_path))
        self.assertEqual(
            [item["id"] for item in report["weights"]],
            list(pairing.WEIGHT_ASSETS),
        )
        self.assertEqual(
            [Path(item["gt_path"]).name for item in report["splits"]],
            ["val_seen_gt.json.gz", "val_unseen_gt.json.gz"],
        )

    def test_weight_size_mismatch_fails_before_accepting_pairing(self):
        checkpoint = self.repo_root / self.weight_specs[
            "bevbert_ce_checkpoint"
        ]["path"]
        with checkpoint.open("ab") as stream:
            stream.write(b"changed")

        with self.assertRaisesRegex(pairing.PairingError, "size mismatch"):
            self._verify_fixture()

    def test_weight_sha256_mismatch_with_same_size_fails(self):
        checkpoint = self.repo_root / self.weight_specs[
            "bevbert_ce_checkpoint"
        ]["path"]
        self._mutate_same_size(checkpoint)

        with self.assertRaisesRegex(pairing.PairingError, "SHA256 mismatch"):
            self._verify_fixture()

    def test_synchronized_manifest_and_file_tampering_still_fails(self):
        asset_id = "bevbert_ce_checkpoint"
        checkpoint = self.repo_root / self.weight_specs[asset_id]["path"]
        self._mutate_same_size(checkpoint)
        changed_sha256 = _sha256(checkpoint)

        def update_manifest(document):
            for asset in document["assets"]:
                if asset["id"] == asset_id:
                    asset["sha256"] = changed_sha256

        self._rewrite_eval_assets(update_manifest)
        with self.assertRaisesRegex(pairing.PairingError, "frozen SHA256 mismatch"):
            self._verify_fixture()

    def test_self_consistent_v13_dataset_path_is_rejected(self):
        split = "val_seen"
        old_path = self.split_specs[split]["path"]
        new_path = old_path.replace("v1-2", "v1-3")
        new_file = self.repo_root / new_path
        new_file.parent.mkdir(parents=True, exist_ok=True)
        new_file.write_bytes((self.repo_root / old_path).read_bytes())

        def change_asset(document):
            for asset in document["assets"]:
                if asset["id"] == self.split_specs[split]["asset_id"]:
                    asset["path"] = new_path

        self._rewrite_eval_assets(change_asset)
        manifest_path = pairing.ORDER_MANIFEST_ROOT / (split + ".json")
        manifest = self._read_json(manifest_path)
        manifest["dataset"]["path"] = new_path
        self._write_json(manifest_path, manifest)

        with self.assertRaisesRegex(pairing.PairingError, "path mismatch"):
            self._verify_fixture()

    def test_dataset_records_must_match_manifest_even_with_matching_file_hash(self):
        split = "val_seen"
        spec = self.split_specs[split]
        dataset_path = self.repo_root / spec["path"]
        with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
            dataset = json.load(stream)
        dataset["episodes"][0]["episode_id"] = "changed-id"
        with gzip.open(dataset_path, "wt", encoding="utf-8") as stream:
            json.dump(dataset, stream)
        digest = _sha256(dataset_path)
        size = dataset_path.stat().st_size

        def update_asset(document):
            for asset in document["assets"]:
                if asset["id"] == spec["asset_id"]:
                    asset["sha256"] = digest
                    asset["size"] = size

        self._rewrite_eval_assets(update_asset)
        manifest_path = pairing.ORDER_MANIFEST_ROOT / (split + ".json")
        manifest = self._read_json(manifest_path)
        manifest["dataset"]["sha256"] = digest
        self._write_json(manifest_path, manifest)
        changed_specs = copy.deepcopy(self.split_specs)
        changed_specs[split]["sha256"] = digest
        changed_specs[split]["size"] = size

        with self.assertRaisesRegex(
            pairing.PairingError, "dataset episodes do not match"
        ):
            self._verify_fixture(split_specs=changed_specs)

    def test_manifest_count_mismatch_is_rejected(self):
        path = pairing.ORDER_MANIFEST_ROOT / "val_unseen.json"
        manifest = self._read_json(path)
        manifest["episode_count"] -= 1
        self._write_json(path, manifest)

        with self.assertRaisesRegex(pairing.PairingError, "episode_count mismatch"):
            self._verify_fixture()

    def test_missing_clip_runtime_file_is_rejected(self):
        self.clip_runtime_path.unlink()
        with self.assertRaisesRegex(pairing.PairingError, "missing openai_clip_vit_b16"):
            self._verify_fixture()

    def test_clip_runtime_file_wrong_hash_is_rejected(self):
        self._mutate_same_size(self.clip_runtime_path)
        with self.assertRaisesRegex(
            pairing.PairingError, "openai_clip_vit_b16 SHA256 mismatch"
        ):
            self._verify_fixture()

    def test_missing_split_ground_truth_is_rejected(self):
        gt_spec = self.split_specs["val_seen"]["gt"]
        (self.repo_root / gt_spec["path"]).unlink()
        with self.assertRaisesRegex(
            pairing.PairingError, "missing etpnav_val_seen_gt"
        ):
            self._verify_fixture()

    def test_split_ground_truth_wrong_hash_is_rejected(self):
        gt_spec = self.split_specs["val_unseen"]["gt"]
        self._mutate_same_size(self.repo_root / gt_spec["path"])
        with self.assertRaisesRegex(
            pairing.PairingError, "etpnav_val_unseen_gt SHA256 mismatch"
        ):
            self._verify_fixture()

    def test_cli_returns_nonzero_and_concise_error(self):
        checkpoint = self.repo_root / self.weight_specs[
            "bevbert_ce_checkpoint"
        ]["path"]
        checkpoint.unlink()
        stdout = StringIO()
        stderr = StringIO()

        with self._fixture_contract(), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = pairing.main(
                [
                    "--repo-root",
                    str(self.repo_root),
                    "--clip-path",
                    str(self.clip_runtime_path),
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("BEVBERT SOURCE PAIRING FAILED", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_success_prints_protocol_and_all_dependency_evidence(self):
        stdout = StringIO()
        stderr = StringIO()

        with self._fixture_contract(), redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = pairing.main(
                [
                    "--repo-root",
                    str(self.repo_root),
                    "--clip-path",
                    str(self.clip_runtime_path),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        output = stdout.getvalue()
        self.assertIn("PASS weight=bevbert_ce_checkpoint", output)
        self.assertIn("PASS runtime_asset=openai_clip_vit_b16", output)
        self.assertIn("PASS split=val_seen episodes=778", output)
        self.assertIn("PASS split=val_unseen episodes=1839", output)
        self.assertIn("gt_sha256=", output)
        self.assertIn("protocol=" + pairing.PROTOCOL, output)


if __name__ == "__main__":
    unittest.main()
