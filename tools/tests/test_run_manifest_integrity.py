import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.run_manifest_identity import (
    IMMUTABLE_IDENTITY_FIELDS,
    IMMUTABLE_IDENTITY_SHA256_FIELD,
    immutable_identity_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CREATE = REPO_ROOT / "tools" / "create_run_manifest.py"
FINALIZE = REPO_ROOT / "tools" / "finalize_run_manifest.py"
VALIDATE = REPO_ROOT / "tools" / "validate_run_manifest.py"


class RunManifestIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.paths = {
            "checkpoint": self._write("checkpoint.bin", b"primary-checkpoint"),
            "auxiliary": self._write("auxiliary.bin", b"auxiliary-checkpoint"),
            "dataset": self._write("dataset.json", b'{"episodes":[]}\n'),
            "assets": self._write("assets.json", b'{"assets":[]}\n'),
            "environment": self._write("environment.json", b'{"env":{}}\n'),
            "episode_order": self._write("episode-order.json", b'{"episodes":[]}\n'),
        }
        self.manifest_path = self.root / "run" / "manifest.json"
        self._create_and_finalize()
        self.original_manifest = self._read_manifest()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def _run(self, command):
        return subprocess.run(
            list(map(str, command)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def _create_and_finalize(self):
        created = self._run(
            [
                sys.executable,
                CREATE,
                "--output",
                self.manifest_path,
                "--run-id",
                "synthetic-run-1",
                "--task",
                "avn",
                "--benchmark",
                "synthetic-benchmark",
                "--model",
                "synthetic-model",
                "--method",
                "source",
                "--run-tag",
                "synthetic-tag",
                "--source-setting",
                "synthetic-source",
                "--seed",
                "7",
                "--config",
                "configs/synthetic.yaml",
                "--checkpoint",
                self.paths["checkpoint"],
                "--aux-checkpoint",
                "encoder={}".format(self.paths["auxiliary"]),
                "--dataset",
                self.paths["dataset"],
                "--dataset-version",
                "synthetic-v1",
                "--asset-manifest",
                self.paths["assets"],
                "--environment-manifest",
                self.paths["environment"],
                "--episode-order-manifest",
                self.paths["episode_order"],
                "--stream-order-sha256",
                "1" * 64,
                "--stream-content-sha256",
                "2" * 64,
                "--extra",
                "OPTION_A",
                "value-a",
            ]
        )
        self.assertEqual(created.returncode, 0, created.stderr)
        finalized = self._run(
            [
                sys.executable,
                FINALIZE,
                "--manifest",
                self.manifest_path,
                "--exit-code",
                "0",
            ]
        )
        self.assertEqual(finalized.returncode, 0, finalized.stderr)

    def _read_manifest(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, document):
        self.manifest_path.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )

    def _validate(self, require_immutable_identity=False):
        manifest = self.original_manifest
        command = [
            sys.executable,
            VALIDATE,
            "--manifest",
            self.manifest_path,
            "--task",
            manifest["task"],
            "--benchmark",
            manifest["benchmark"],
            "--run-tag",
            manifest["run_tag"],
            "--model",
            manifest["model"],
            "--method",
            manifest["method"],
            "--source-setting",
            manifest["source_setting"],
            "--seed",
            manifest["seed"],
            "--git-commit",
            manifest["git_commit"],
            "--checkpoint-sha256",
            manifest["checkpoint"]["sha256"],
            "--stream-order-sha256",
            manifest["dataset"]["stream_order_sha256"],
            "--stream-content-sha256",
            manifest["dataset"]["stream_content_sha256"],
        ]
        if require_immutable_identity:
            command.append("--require-immutable-identity")
        return self._run(command)

    def test_creation_covers_complete_identity_and_file_metadata(self):
        manifest = self.original_manifest
        self.assertEqual(
            set(IMMUTABLE_IDENTITY_FIELDS),
            {
                "run_id",
                "task",
                "benchmark",
                "model",
                "method",
                "run_tag",
                "source_setting",
                "seed",
                "git_commit",
                "config",
                "config_overrides",
                "checkpoint",
                "auxiliary_checkpoints",
                "dataset",
                "pinned_manifests",
                "hardware",
                "started_at",
            },
        )
        self.assertEqual(
            manifest[IMMUTABLE_IDENTITY_SHA256_FIELD],
            immutable_identity_sha256(manifest),
        )
        self.assertEqual(
            set(manifest["pinned_manifests"]),
            {"assets", "environment", "episode_order"},
        )
        references = [manifest["checkpoint"], manifest["dataset"]]
        references.extend(manifest["auxiliary_checkpoints"])
        references.extend(manifest["pinned_manifests"].values())
        for reference in references:
            self.assertEqual(reference["size"], os.path.getsize(reference["path"]))
        self.assertEqual(
            self._validate(require_immutable_identity=True).returncode, 0
        )

    def test_every_declared_identity_field_changes_the_digest(self):
        manifest = self.original_manifest
        original_digest = immutable_identity_sha256(manifest)
        for field in IMMUTABLE_IDENTITY_FIELDS:
            with self.subTest(field=field):
                changed = copy.deepcopy(manifest)
                changed[field] = {"tampered": field}
                self.assertNotEqual(
                    immutable_identity_sha256(changed), original_digest
                )

    def test_validator_rejects_immutable_identity_tampering(self):
        manifest = self._read_manifest()
        manifest["config_overrides"].append("TAMPERED")
        self._write_manifest(manifest)

        completed = self._validate()

        self.assertEqual(completed.returncode, 1)
        self.assertIn("immutable identity SHA256 changed", completed.stderr)

    def test_validator_rejects_each_referenced_file_tampering(self):
        cases = (
            ("checkpoint", self.paths["checkpoint"]),
            ("auxiliary checkpoint encoder", self.paths["auxiliary"]),
            ("dataset", self.paths["dataset"]),
            ("pinned manifest assets", self.paths["assets"]),
            ("pinned manifest environment", self.paths["environment"]),
            ("pinned manifest episode_order", self.paths["episode_order"]),
        )
        for expected_label, path in cases:
            with self.subTest(reference=expected_label):
                original = path.read_bytes()
                replacement = bytes([original[0] ^ 1]) + original[1:]
                path.write_bytes(replacement)
                try:
                    completed = self._validate()
                finally:
                    path.write_bytes(original)
                self.assertEqual(completed.returncode, 1)
                self.assertIn(expected_label, completed.stderr)
                self.assertIn("SHA256 changed", completed.stderr)

    def test_legacy_manifest_without_identity_or_sizes_still_validates(self):
        manifest = self._read_manifest()
        manifest.pop(IMMUTABLE_IDENTITY_SHA256_FIELD)
        manifest["checkpoint"].pop("size")
        manifest["dataset"].pop("size")
        for auxiliary in manifest["auxiliary_checkpoints"]:
            auxiliary.pop("size")
        for pinned in manifest["pinned_manifests"].values():
            pinned.pop("size")
        manifest["pinned_manifests"].pop("environment")
        self._write_manifest(manifest)

        completed = self._validate()

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_strict_validator_rejects_deleted_identity_digest(self):
        manifest = self._read_manifest()
        manifest.pop(IMMUTABLE_IDENTITY_SHA256_FIELD)
        self._write_manifest(manifest)

        completed = self._validate(require_immutable_identity=True)

        self.assertEqual(completed.returncode, 1)
        self.assertIn("immutable identity SHA256 is required", completed.stderr)

    def test_legacy_manifest_size_metadata_is_enforced_when_present(self):
        manifest = self._read_manifest()
        manifest.pop(IMMUTABLE_IDENTITY_SHA256_FIELD)
        manifest["checkpoint"]["size"] += 1
        self._write_manifest(manifest)

        completed = self._validate()

        self.assertEqual(completed.returncode, 1)
        self.assertIn("checkpoint size changed", completed.stderr)


if __name__ == "__main__":
    unittest.main()
