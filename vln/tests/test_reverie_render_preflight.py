import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/verify_reverie_render_preflight.py"
WRAPPER = (
    REPO_ROOT / "vln/scripts/verify_reverie_llm_feedback_preflight.sh"
)
SPEC = importlib.util.spec_from_file_location(
    "_test_reverie_render_preflight", str(SCRIPT)
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReverieRenderPreflightTest(unittest.TestCase):
    @staticmethod
    def _build(root, egl="ON", osmesa="OFF"):
        build = root / "render-build"
        build.mkdir()
        (build / "CMakeCache.txt").write_text(
            "EGL_RENDERING:BOOL={}\nOSMESA_RENDERING:BOOL={}\n".format(
                egl, osmesa
            ),
            encoding="utf-8",
        )
        (build / "MatterSim.cpython-38-x86_64-linux-gnu.so").write_bytes(
            b"extension"
        )
        return build

    @staticmethod
    def _assets(root, include_rgb=True):
        from PIL import Image

        connectivity = root / "connectivity"
        connectivity.mkdir()
        (connectivity / "scans.txt").write_text("scan-a\n", encoding="utf-8")
        (connectivity / "scan-a_connectivity.json").write_text(
            json.dumps([{"image_id": "vp-1", "included": True}]),
            encoding="utf-8",
        )
        scans = root / "scans"
        rgb_dir = scans / "scan-a" / "matterport_skybox_images"
        rgb_dir.mkdir(parents=True)
        if include_rgb:
            Image.new("RGB", (12, 2), color=(1, 2, 3)).save(
                rgb_dir / "vp-1_skybox_small.jpg", format="JPEG"
            )
        return connectivity, scans

    def test_standard_navigation_build_is_rejected_as_non_headless(self):
        with tempfile.TemporaryDirectory() as directory:
            build = self._build(Path(directory), egl="OFF", osmesa="OFF")
            with self.assertRaisesRegex(
                MODULE.PreflightError, "standard NavTTA navigation build"
            ):
                MODULE.verify_headless_build(build)

    def test_source_asset_inventory_hashes_exact_raw_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connectivity, scans = self._assets(root)
            files, image = MODULE.verify_source_assets(
                connectivity, scans, "scan-a", "vp-1"
            )
            by_role = {item["role"]: item for item in files}
            raw = scans / "scan-a/matterport_skybox_images/vp-1_skybox_small.jpg"
            self.assertEqual(
                by_role["raw_rgb_cubemap"]["sha256"],
                hashlib.sha256(raw.read_bytes()).hexdigest(),
            )
            self.assertEqual(image["format"], "JPEG")
            self.assertEqual(image["width"], 12)

    def test_missing_raw_rgb_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connectivity, scans = self._assets(root, include_rgb=False)
            with self.assertRaisesRegex(MODULE.PreflightError, "raw_rgb_cubemap"):
                MODULE.verify_source_assets(
                    connectivity, scans, "scan-a", "vp-1"
                )

    def test_real_renderer_path_writes_hash_bound_evidence(self):
        from PIL import Image

        class Renderer(object):
            def __init__(self, connectivity_dir, scan_data_dir):
                self.paths = (connectivity_dir, scan_data_dir)

            def render(self, scan, viewpoint):
                image = Image.new("RGB", (1920, 360))
                image.putpixel((0, 0), (255, 1, 2))
                output = BytesIO()
                image.save(
                    output, format="PNG", optimize=False, compress_level=6
                )
                payload = output.getvalue()
                return payload, {
                    "view_count": 36,
                    "layout": "12_columns_x_3_elevation_rows",
                    "image_format": "PNG_RGB",
                    "camera_width": 640,
                    "camera_height": 480,
                    "tile_width": 160,
                    "tile_height": 120,
                    "vfov_degrees": 60.0,
                    "panorama_png_sha256": hashlib.sha256(payload).hexdigest(),
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = self._build(root)
            connectivity, scans = self._assets(root)
            output = root / "evidence"
            args = SimpleNamespace(
                mattersim_build=str(build),
                connectivity_dir=str(connectivity),
                scan_data_dir=str(scans),
                scan="scan-a",
                viewpoint="vp-1",
                output_dir=str(output),
                require_clean_git=False,
            )
            fake_imports = {
                "navtta_core": MODULE.file_evidence(
                    build / "CMakeCache.txt", "navtta_core_module"
                ),
                "mattersim": MODULE.file_evidence(
                    build / "MatterSim.cpython-38-x86_64-linux-gnu.so",
                    "loaded_mattersim_extension",
                ),
            }
            with mock.patch.object(
                MODULE, "verify_import_locations",
                return_value=fake_imports,
            ), mock.patch.object(
                MODULE.PROVIDER, "MatterSimPanoramaRenderer", Renderer
            ), mock.patch.object(
                MODULE, "repository_identity",
                return_value={
                    "root": str(REPO_ROOT), "git_commit": "a" * 40,
                    "tracked_tree_clean": True,
                },
            ):
                pointer = MODULE.run_preflight(args)
            evidence_path = Path(pointer["evidence_path"])
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "passed")
            self.assertEqual(evidence["headless_build"]["backend"], "egl")
            self.assertEqual(evidence["render"]["metadata"]["view_count"], 36)
            self.assertTrue(Path(pointer["panorama_path"]).is_file())
            self.assertEqual(
                pointer["evidence_sha256"],
                hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            )

    def test_formal_wrapper_binds_cuda_render_and_hash_evidence(self):
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("--device cuda:0 --dtype float16 --verify-only", source)
        self.assertIn("verify_reverie_render_preflight.py", source)
        self.assertIn("--require-clean-git", source)
        self.assertIn("provider_smoke_transcript.ndjson", source)
        self.assertIn(
            '"schema": "navtta.reverie_llm_feedback_preflight.v1"',
            source,
        )
        self.assertIn('"render_evidence": evidence(render_path)', source)
        self.assertIn(
            '"rendered_panorama": evidence(root / '
            '"render/endpoint_panorama.png")',
            source,
        )
        self.assertIn(
            "NAVTTA_REVERIE_RENDER_MATTERSIM_BUILD", source
        )
        self.assertIn("NAVTTA_REVERIE_LLM_PREFLIGHT", source)
        self.assertIn("sha256sum", source)
        completed = subprocess.run(
            ["bash", "-n", str(WRAPPER)],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
