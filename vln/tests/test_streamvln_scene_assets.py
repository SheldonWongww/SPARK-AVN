import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from vln.scripts import mp3d_scene_assets


SCENE = mp3d_scene_assets.DEFAULT_SCENE


def _write_glb(path, declared_length=None):
    json_chunk = b"{}  "
    actual_length = 12 + 8 + len(json_chunk)
    header_length = actual_length if declared_length is None else declared_length
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, header_length)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
    )


def _ply_header(vertex_count=3, face_count=1):
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex {}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face {}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).format(vertex_count, face_count).encode("ascii")


def _write_valid_ply(path):
    vertices = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    face = struct.pack("<Biii", 3, 0, 1, 2)
    path.write_bytes(_ply_header() + vertices + face)


def _write_scene(directory, scene=SCENE):
    directory.mkdir(parents=True)
    _write_glb(directory / (scene + ".glb"))
    (directory / (scene + ".navmesh")).write_bytes(b"synthetic-navmesh")
    (directory / (scene + ".house")).write_bytes(b"ASCII 1.1\n")
    _write_valid_ply(directory / (scene + "_semantic.ply"))


def _write_archive_and_manifest(repo_root):
    source_scene = repo_root / "fixture" / SCENE
    _write_scene(source_scene)
    archive_path = repo_root / "vln/data/.downloads/mp3d_habitat.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in mp3d_scene_assets.required_filenames(SCENE):
            archive.write(
                source_scene / filename,
                "mp3d/{}/{}".format(SCENE, filename),
            )

    manifest_path = repo_root / "vln/manifests/assets/eval_assets.json"
    manifest_path.parent.mkdir(parents=True)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "id": mp3d_scene_assets.ARCHIVE_ASSET_ID,
                        "path": "vln/data/.downloads/mp3d_habitat.zip",
                        "size": archive_path.stat().st_size,
                        "sha256": digest,
                        "source": "mp3d_habitat",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return archive_path, manifest_path, digest


class StreamVLNSceneAssetTest(unittest.TestCase):
    def test_project_manifest_pins_the_official_mp3d_archive(self):
        repo_root = Path(__file__).resolve().parents[2]
        spec = mp3d_scene_assets.load_archive_spec(repo_root)
        self.assertEqual(spec["expected_size"], 16085306031)
        self.assertEqual(
            spec["expected_sha256"],
            "470948ee78ff4d6dc4c4870395d1e790a0f15c72065d1f62cdd229b2bcbe0d36",
        )
        self.assertEqual(spec["source"], "mp3d_habitat")
        self.assertTrue(spec["archive_path"].endswith(
            "vln/data/.downloads/mp3d_habitat.zip"
        ))

    def test_scene_identifier_cannot_escape_the_scene_root(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_root = Path(directory) / "scenes"
            scene_root.mkdir()
            with self.assertRaisesRegex(
                mp3d_scene_assets.SceneAssetError, "invalid MP3D scene identifier"
            ):
                mp3d_scene_assets.repair_scene(
                    directory,
                    scene="../escape",
                    scene_root=scene_root,
                )
            self.assertFalse((Path(directory) / "escape").exists())

    def test_valid_scene_checks_glb_and_complete_ply_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            scene_dir = Path(directory) / SCENE
            _write_scene(scene_dir)

            report = mp3d_scene_assets.validate_scene_directory(scene_dir)

            self.assertEqual(report["scene"], SCENE)
            self.assertEqual(report["glb"]["version"], 2)
            self.assertEqual(report["semantic_ply"]["vertices"], 3)
            self.assertEqual(report["semantic_ply"]["faces"], 1)
            self.assertEqual(report["semantic_ply"]["first_face_size"], 3)

    def test_glb_declared_length_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.glb"
            _write_glb(path, declared_length=999)
            with self.assertRaisesRegex(
                mp3d_scene_assets.SceneAssetError, "declared length mismatch"
            ):
                mp3d_scene_assets.validate_glb(path)

    def test_truncated_vertex_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / (SCENE + "_semantic.ply")
            path.write_bytes(_ply_header(vertex_count=3) + b"\x00" * 8)
            with self.assertRaisesRegex(
                mp3d_scene_assets.SceneAssetError, "vertex payload is truncated"
            ):
                mp3d_scene_assets.validate_semantic_ply(path, scene=SCENE)

    def test_trailing_ply_padding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / (SCENE + "_semantic.ply")
            _write_valid_ply(path)
            with path.open("ab") as stream:
                stream.write(b"padding")
            with self.assertRaisesRegex(
                mp3d_scene_assets.SceneAssetError, "trailing/padded bytes"
            ):
                mp3d_scene_assets.validate_semantic_ply(path, scene=SCENE)

    def test_known_bad_size_and_first_face_byte_are_identified(self):
        self.assertEqual(mp3d_scene_assets.KNOWN_BAD_SEMANTIC_SIZE, 115277824)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / (SCENE + "_semantic.ply")
            header = _ply_header(vertex_count=1, face_count=1)
            with path.open("wb") as stream:
                stream.write(header)
                stream.write(struct.pack("<3f", 0.0, 0.0, 0.0))
                stream.write(bytes([mp3d_scene_assets.KNOWN_BAD_FIRST_FACE_SIZE]))
                stream.truncate(mp3d_scene_assets.KNOWN_BAD_SEMANTIC_SIZE)
            with self.assertRaisesRegex(
                mp3d_scene_assets.SceneAssetError,
                "known corrupt 1pXnuDYAj8r semantic PLY signature",
            ):
                mp3d_scene_assets.validate_semantic_ply(path, scene=SCENE)

    def test_archive_repair_validates_staging_and_retains_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            _, _, digest = _write_archive_and_manifest(repo_root)

            scene_root = repo_root / "vln/data/scene_datasets/mp3d"
            target = scene_root / SCENE
            target.mkdir(parents=True)
            marker = target / "corrupt-marker"
            marker.write_text("retain me", encoding="utf-8")

            report = mp3d_scene_assets.repair_scene(repo_root, scene=SCENE)

            self.assertFalse(report["dry_run"])
            self.assertEqual(report["archive"]["actual_sha256"], digest)
            self.assertTrue(Path(report["backup"]).is_dir())
            self.assertEqual(
                (Path(report["backup"]) / "corrupt-marker").read_text(encoding="utf-8"),
                "retain me",
            )
            self.assertFalse(marker.exists())
            installed = mp3d_scene_assets.validate_scene_directory(target)
            self.assertEqual(installed["semantic_ply"]["first_face_size"], 3)
            staged = [item for item in scene_root.iterdir() if ".repair-" in item.name]
            self.assertEqual(staged, [])

    def test_post_install_validation_failure_rolls_back_original_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            _write_archive_and_manifest(repo_root)
            scene_root = repo_root / "vln/data/scene_datasets/mp3d"
            target = scene_root / SCENE
            target.mkdir(parents=True)
            marker = target / "original-marker"
            marker.write_text("original", encoding="utf-8")
            real_validate = mp3d_scene_assets.validate_scene_directory

            def fail_only_after_install(path, scene=None):
                path = Path(path)
                if (
                    path.resolve() == target.resolve()
                    and (path / (SCENE + ".glb")).is_file()
                ):
                    raise mp3d_scene_assets.SceneAssetError(
                        "synthetic post-install validation failure"
                    )
                return real_validate(path, scene=scene)

            with mock.patch.object(
                mp3d_scene_assets,
                "validate_scene_directory",
                side_effect=fail_only_after_install,
            ):
                with self.assertRaisesRegex(
                    mp3d_scene_assets.SceneAssetError,
                    "synthetic post-install validation failure",
                ):
                    mp3d_scene_assets.repair_scene(repo_root, scene=SCENE)

            self.assertEqual(marker.read_text(encoding="utf-8"), "original")
            self.assertFalse((target / (SCENE + ".glb")).exists())
            backups = [item for item in scene_root.iterdir() if ".backup-" in item.name]
            self.assertEqual(backups, [])
            staging = [item for item in scene_root.iterdir() if ".repair-" in item.name]
            self.assertEqual(staging, [])

    def test_archive_digest_mismatch_never_touches_target(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            archive_path = repo_root / "archive.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("irrelevant", b"data")
            manifest_path = repo_root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "id": mp3d_scene_assets.ARCHIVE_ASSET_ID,
                                "path": str(archive_path),
                                "size": archive_path.stat().st_size,
                                "sha256": "0" * 64,
                                "source": "mp3d_habitat",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            scene_root = repo_root / "scenes"
            target = scene_root / SCENE
            target.mkdir(parents=True)
            marker = target / "marker"
            marker.write_text("unchanged", encoding="utf-8")

            with self.assertRaisesRegex(
                mp3d_scene_assets.SceneAssetError, "SHA256 mismatch"
            ):
                mp3d_scene_assets.repair_scene(
                    repo_root,
                    scene=SCENE,
                    scene_root=scene_root,
                    manifest_path=manifest_path,
                    archive_path=archive_path,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
            self.assertEqual(
                [item.name for item in scene_root.iterdir()],
                [SCENE],
            )


if __name__ == "__main__":
    unittest.main()
