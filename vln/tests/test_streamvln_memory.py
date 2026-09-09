"""Regressions for cached generation crossing an environment-step boundary."""

import ast
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vln"))
sys.path.insert(0, str(ROOT / "vln/scripts"))
from tta_config_cli import translate

_spec = importlib.util.spec_from_file_location(
    "streamvln_memory", ROOT / "vln/navtta_vln/streamvln_memory.py"
)
_memory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_memory)
history_frame_indices = _memory.history_frame_indices


class StreamVLNMemoryTest(unittest.TestCase):
    def test_queued_actions_across_reset_keep_memory_and_prompt_together(self):
        segment_start = 0
        new_context = True
        remaining_actions = 0
        generation = {}
        for step in range(201):
            if step % 32 == 0:
                segment_start = step
                new_context = True
            if remaining_actions == 0:
                generation[step] = history_frame_indices(new_context, segment_start, 8, 4)
                remaining_actions = 6 if step == 188 else 4
                new_context = False
            remaining_actions -= 1
        self.assertNotIn(192, generation)
        self.assertEqual(generation[194], tuple(range(0, 192, 24)))
        self.assertEqual(len(generation[194]), 8)
        self.assertEqual(generation[0], ())
        self.assertEqual(generation[32], tuple(range(0, 32, 4)))
        self.assertEqual(generation[198], ())

    def test_aligned_boundaries_preserve_original_history_samples(self):
        for start in range(32, 501, 32):
            self.assertEqual(
                history_frame_indices(True, start, 8, 4),
                tuple(range(0, start, start // 8)),
            )

    def test_no_empty_or_overlong_history_for_supported_context(self):
        self.assertEqual(len(history_frame_indices(True, 35, 8, 4)), 8)
        with self.assertRaises(ValueError):
            history_frame_indices(True, 4, 8, 4)
        self.assertEqual(history_frame_indices(True, 32, None, 4), tuple(range(0, 32, 4)))


def parser_without_model_imports():
    # Execute the actual declaration function, avoiding Habitat/Torch imports.
    # This is a parser contract check, not a copy of its option definitions.
    path = ROOT / "vln/navtta_vln/discrete_tta.py"
    tree = ast.parse(path.read_text())
    declaration = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "add_discrete_tta_args")
    constant = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "TTA_METHODS" for t in n.targets))
    namespace = {"TTA_METHODS": ast.literal_eval(constant.value)}
    exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(path), "exec"), namespace)
    parser = argparse.ArgumentParser()
    namespace["add_discrete_tta_args"](parser, feedtta_scope_profiles=("configured_prefixes",))
    parser.add_argument("--tta_streamvln_readout_protocol", choices=("legacy_v4", "native_residual"))
    return parser


class StreamVLNParserContractTest(unittest.TestCase):
    def test_every_v6_search_config_parses_before_loading_the_model(self):
        spec = json.loads((ROOT / "vln/experiments/streamvln_val_unseen_search_v6.json").read_text())
        parser = parser_without_model_imports()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for method, candidates in spec["methods"].items():
                for candidate in candidates:
                    path.write_text(json.dumps({"method": method, "parameters": candidate["parameters"]}))
                    _, tokens = translate("streamvln-r2r-ce", str(path), str(Path(directory) / "diag.json"))
                    args = parser.parse_args(tokens)
                    self.assertEqual(args.tta_method, method)
                    if method == "feedtta":
                        self.assertEqual(args.tta_feedtta_scope_profile, "configured_prefixes")
                        self.assertEqual(args.tta_feedtta_alpha, -0.2)


if __name__ == "__main__":
    unittest.main()
