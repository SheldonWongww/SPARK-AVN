import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "vln/navtta_vln/per_episode_metrics.py"
SPEC = importlib.util.spec_from_file_location(
    "navtta_per_episode_metrics", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PerEpisodeMetricsTest(unittest.TestCase):
    def _fixture(self, root):
        manifests = root / "orders"
        manifests.mkdir()
        manifest = {
            "schema": "navtta.episode_order.v1",
            "split": "val_unseen",
            "order_sha256": "a" * 64,
            "episodes": [
                {"episode_id": "7_0", "scene_id": "scan-a"},
                {"episode_id": "8_0", "scene_id": "scan-b"},
            ],
        }
        (manifests / "val_unseen.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )
        diagnostics = root / "result" / "tta_diagnostics.json"
        return SimpleNamespace(
            tta_diagnostics=str(diagnostics),
            episode_order_manifest=str(manifests),
        ), diagnostics

    def test_writes_canonical_ordered_native_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            args, diagnostics = self._fixture(Path(directory))
            raw = {
                "instr_id": ["7_0", "8_0"],
                "success": [True, False],
                "spl": [1.0, 0.25],
            }
            with mock.patch.dict(os.environ, {"NAVTTA_RUN_TAG": "run-1"}):
                output = MODULE.write_discrete_per_episode_metrics(
                    args, "hamt-r2r", "val_unseen", raw
                )
            self.assertEqual(
                output,
                (diagnostics.parent / "per_episode_metrics.json").resolve(),
            )
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["schema"], MODULE.SCHEMA)
            self.assertEqual(document["run_tag"], "run-1")
            self.assertEqual(document["episode_count"], 2)
            self.assertEqual(
                [item["episode_id"] for item in document["episodes"]],
                ["7_0", "8_0"],
            )
            self.assertEqual(document["episodes"][0]["metrics"]["success"], 1.0)

    def test_rejects_an_evaluator_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self._fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "does not match"):
                MODULE.write_discrete_per_episode_metrics(
                    args, "hamt-r2r", "val_unseen",
                    {"instr_id": ["8_0", "7_0"], "spl": [1.0, 1.0]},
                )

    def test_runtime_prefix_is_validated_and_identified_as_active_order(self):
        with tempfile.TemporaryDirectory() as directory:
            args, diagnostics = self._fixture(Path(directory))
            raw = {
                "instr_id": ["7_0"],
                "success": [True],
                "spl": [1.0],
            }
            with mock.patch.dict(
                os.environ,
                {"NAVTTA_SMOKE_EPISODES": "1"},
                clear=False,
            ):
                output = MODULE.write_discrete_per_episode_metrics(
                    args, "hamt-r2r", "val_unseen", raw
                )
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["episode_count"], 1)
            self.assertEqual(document["runtime_episode_limit"], 1)
            self.assertEqual(document["parent_order_sha256"], "a" * 64)
            self.assertEqual(
                document["order_sha256"],
                MODULE._records_sha256([{
                    "episode_id": "7_0", "scene_id": "scan-a"
                }]),
            )
            self.assertEqual(
                [row["episode_id"] for row in document["episodes"]],
                ["7_0"],
            )

    def test_runtime_prefix_rejects_full_manifest_evaluator_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self._fixture(Path(directory))
            with mock.patch.dict(
                os.environ,
                {"NAVTTA_SMOKE_EPISODES": "1"},
                clear=False,
            ), self.assertRaisesRegex(ValueError, "does not match"):
                MODULE.write_discrete_per_episode_metrics(
                    args, "hamt-r2r", "val_unseen",
                    {"instr_id": ["7_0", "8_0"], "spl": [1.0, 1.0]},
                )

    def test_full_run_sidecar_identity_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self._fixture(Path(directory))
            with mock.patch.dict(os.environ, {}, clear=True):
                output = MODULE.write_discrete_per_episode_metrics(
                    args, "hamt-r2r", "val_unseen",
                    {"instr_id": ["7_0", "8_0"], "spl": [1.0, 1.0]},
                )
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["order_sha256"], "a" * 64)
            self.assertNotIn("runtime_episode_limit", document)
            self.assertNotIn("parent_order_sha256", document)

    def test_runtime_prefix_limit_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self._fixture(Path(directory))
            for value, message in (("0", "positive"), ("x", "positive"),
                                   ("3", "exceeds")):
                with self.subTest(value=value), mock.patch.dict(
                    os.environ,
                    {"NAVTTA_SMOKE_EPISODES": value},
                    clear=False,
                ), self.assertRaisesRegex(ValueError, message):
                    MODULE.write_discrete_per_episode_metrics(
                        args, "hamt-r2r", "val_unseen",
                        {"instr_id": ["7_0"], "spl": [1.0]},
                    )

    def test_source_without_diagnostics_remains_unchanged(self):
        args = SimpleNamespace(
            tta_diagnostics="", episode_order_manifest="/does/not/exist"
        )
        self.assertIsNone(MODULE.write_discrete_per_episode_metrics(
            args, "hamt-r2r", "val_seen", {}
        ))

    def test_all_six_discrete_entrypoints_emit_native_evaluator_evidence(self):
        entrypoints = {
            "duet-r2r": "vln/baselines/duet/map_nav_src/r2r/main_nav.py",
            "duet-reverie": (
                "vln/baselines/duet/map_nav_src/reverie/main_nav_obj.py"
            ),
            "hamt-r2r": "vln/baselines/hamt/finetune_src/r2r/main.py",
            "hamt-reverie": (
                "vln/baselines/hamt/finetune_src/reverie/main_navref.py"
            ),
            "goat-r2r": "vln/baselines/goat/map_nav_src/r2r/main_nav.py",
            "goat-reverie": (
                "vln/baselines/goat/map_nav_src/reverie/main_nav_obj.py"
            ),
        }
        for setting, relative in entrypoints.items():
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(
                "from navtta_vln.per_episode_metrics import ", source, relative
            )
            self.assertIn(
                "write_discrete_per_episode_metrics(\n"
                "                    args, {!r}, env_name, per_episode".format(
                    setting
                ),
                source,
                relative,
            )

        launcher = (REPO_ROOT / "vln/scripts/run_source_eval.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('export NAVTTA_RUN_TAG="${RUN_TAG}"', launcher)


if __name__ == "__main__":
    unittest.main()
