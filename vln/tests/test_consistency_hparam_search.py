"""Unit tests for the cross-split consistency hyperparameter search runner.

Covers the pure logic: grid expansion (incl. IDEA opt_steps + paired knobs),
console metric parsing (discrete + continuous), and the floor/worst-split
selection rule.  The launcher itself is not invoked here.
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln" / "scripts"))

from run_consistency_hparam_search import (  # noqa: E402
    expand_candidates,
    parse_console_metrics,
    run_selection,
    select_config,
)


class ExpandCandidatesTest(unittest.TestCase):
    def test_cartesian_product_of_scalar_axes(self):
        spec = {"base": {"norm_scope": "last_k_ln"},
                "grid": {"lr": [1e-5, 1e-4], "update_interval": [1, 4]}}
        cands = expand_candidates("eam", spec, for_search=True)
        self.assertEqual(len(cands), 4)
        for c in cands:
            self.assertEqual(c["norm_scope"], "last_k_ln")
            self.assertIn(c["lr"], (1e-5, 1e-4))
            self.assertIn(c["update_interval"], (1, 4))

    def test_paired_knob_dicts_expand_into_multiple_params(self):
        spec = {"base": {}, "grid": {"lr": [5e-6],
                "pa": [{"p": 0.05, "alpha": -0.2}, {"p": 0.05, "alpha": 0.1}]}}
        cands = expand_candidates("feedtta", spec, for_search=True)
        self.assertEqual(len(cands), 2)
        alphas = sorted(c["alpha"] for c in cands)
        self.assertEqual(alphas, [-0.2, 0.1])
        self.assertTrue(all(c["p"] == 0.05 for c in cands))

    def test_idea_search_vs_final_opt_steps(self):
        spec = {"base": {"prompt_length": 4}, "grid": {"lr": [3e-3]},
                "search_opt_steps": 10, "final_opt_steps": 50}
        search = expand_candidates("idea", spec, for_search=True)
        final = expand_candidates("idea", spec, for_search=False)
        self.assertEqual(search[0]["opt_steps"], 10)
        self.assertEqual(final[0]["opt_steps"], 50)


class ParseConsoleMetricsTest(unittest.TestCase):
    def test_discrete_val_seen_line(self):
        text = ("noise\n"
                "Env name: val_seen, steps: 5.00, lengths: 6.27, nav_error: 1.03, "
                "sr: 79.50, oracle_sr: 85.00, spl: 74.10, nDTW: 93.46\n")
        values = parse_console_metrics(text, "val_seen")
        self.assertAlmostEqual(values["SR"], 79.5)
        self.assertAlmostEqual(values["SPL"], 74.1)
        self.assertAlmostEqual(values["ORACLE_SR"], 85.0)

    def test_discrete_split_marker_is_respected(self):
        text = "Env name: val_unseen, sr: 66.20, spl: 61.50\n"
        self.assertEqual(parse_console_metrics(text, "val_seen"), {})
        values = parse_console_metrics(text, "val_unseen")
        self.assertAlmostEqual(values["SR"], 66.2)

    def test_continuous_average_episode_lines_scaled(self):
        text = ("Average episode success: 0.5824\n"
                "Average episode spl: 0.4826\n"
                "Average episode oracle_success: 0.6645\n"
                "Average episode nav_error: 4.53\n")
        values = parse_console_metrics(text, "val_seen")
        self.assertAlmostEqual(values["SR"], 58.24, places=2)
        self.assertAlmostEqual(values["SPL"], 48.26, places=2)
        self.assertAlmostEqual(values["OSR"], 66.45, places=2)
        self.assertAlmostEqual(values["NAV_ERROR"], 4.53, places=2)


class SelectConfigTest(unittest.TestCase):
    def setUp(self):
        self.source = {
            "val_seen": {"SR": 78.0, "SPL": 72.0},
            "val_unseen": {"SR": 71.0, "SPL": 60.0},
        }
        self.selection = {
            "floor_metric": "SR", "floor_epsilon": 0.3, "gain_metric": "SR",
        }

    def _cand(self, params, sr_seen, sr_unseen, spl_seen=72.0, spl_unseen=60.0,
              drift=0.0, updates=0):
        return {
            "parameters": params,
            "metrics_seen": {"SR": sr_seen, "SPL": spl_seen},
            "metrics_unseen": {"SR": sr_unseen, "SPL": spl_unseen},
            "diagnostics": {"relative_param_drift": drift, "updates": updates},
        }

    def test_rejects_val_unseen_regression_even_if_val_seen_wins(self):
        cands = [
            self._cand({"lr": 1e-3}, 80.0, 68.0),   # big seen gain, unseen below floor
            self._cand({"lr": 1e-5}, 78.5, 71.5),   # modest, consistent
        ]
        winner, ranked = select_config(cands, self.source, self.selection)
        self.assertEqual(winner["parameters"], {"lr": 1e-5})
        # The regressing candidate must be marked as failing the floor.
        regressor = next(r for r in ranked if r["parameters"] == {"lr": 1e-3})
        self.assertFalse(regressor["passes_floor"])

    def test_maximizes_worst_split_gain_among_survivors(self):
        cands = [
            self._cand({"lr": "a"}, 79.0, 71.2),   # worst-split gain = +0.2
            self._cand({"lr": "b"}, 78.4, 72.0),   # worst-split gain = +0.4
        ]
        winner, _ = select_config(cands, self.source, self.selection)
        self.assertEqual(winner["parameters"], {"lr": "b"})

    def test_returns_none_when_no_candidate_clears_floor(self):
        cands = [
            self._cand({"lr": "x"}, 70.0, 60.0),   # both far below source
        ]
        winner, ranked = select_config(cands, self.source, self.selection)
        self.assertIsNone(winner)
        self.assertFalse(ranked[0]["passes_floor"])

    def test_tie_broken_by_lower_drift_then_fewer_updates(self):
        cands = [
            self._cand({"lr": "hi"}, 78.5, 71.5, drift=0.5, updates=100),
            self._cand({"lr": "lo"}, 78.5, 71.5, drift=0.1, updates=100),
        ]
        winner, _ = select_config(cands, self.source, self.selection)
        self.assertEqual(winner["parameters"], {"lr": "lo"})


class RunSelectionEndToEndTest(unittest.TestCase):
    def _write_console(self, path, split, sr, spl):
        path.mkdir(parents=True, exist_ok=True)
        (path / "console.log").write_text(
            "Env name: {}, steps: 5.00, sr: {:.2f}, spl: {:.2f}\n".format(
                split, sr, spl
            ),
            encoding="utf-8",
        )

    def test_writes_selected_config_with_consistency_winner(self):
        spec = {
            "schema": "navtta.vln_tta_consistency_search.v1",
            "benchmark": "r2r",
            "splits": ["val_seen", "val_unseen"],
            "order_seed": 0,
            "settings": ["duet-r2r"],
            "selection": {"floor_metric": "SR", "floor_epsilon": 0.3,
                          "gain_metric": "SR"},
            "methods": {"tent": {"base": {"norm_scope": "last_k_ln"},
                                 "grid": {"lr": [1e-3, 1e-5]},
                                 "paper_default": {"lr": 1e-4}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            source_root = root / "source"
            self._write_console(source_root / "duet-r2r" / "val_seen", "val_seen", 78.0, 72.0)
            self._write_console(source_root / "duet-r2r" / "val_unseen", "val_unseen", 71.0, 60.0)
            out_dir = root / "out"
            # Candidate lr=1e-3 collapses on unseen; lr=1e-5 is consistent.
            from run_consistency_hparam_search import candidate_tag, split_result_root
            tags = {
                (1e-3): (80.0, 68.0),   # unseen below floor
                (1e-5): (78.4, 71.4),   # consistent
            }
            for lr, (sr_u_seen, sr_unseen) in tags.items():
                params = {"norm_scope": "last_k_ln", "lr": lr}
                tag = candidate_tag("tent", params)
                self._write_console(
                    split_result_root(out_dir, "duet-r2r", "tent", "c1", tag, "val_seen"),
                    "val_seen", sr_u_seen, 73.0)
                self._write_console(
                    split_result_root(out_dir, "duet-r2r", "tent", "c1", tag, "val_unseen"),
                    "val_unseen", sr_unseen, 61.0)

            run_selection(str(spec_path), str(out_dir), str(source_root), "c1",
                          set(), set(), final_stage=False)

            selected = json.loads(
                (out_dir / "duet-r2r" / "tent" / "selected_config.json").read_text()
            )
            self.assertFalse(selected["used_fallback"])
            self.assertEqual(selected["winner"]["parameters"]["lr"], 1e-5)

    def test_writes_fallback_when_no_candidate_clears_floor(self):
        spec = {
            "schema": "navtta.vln_tta_consistency_search.v1",
            "benchmark": "r2r",
            "splits": ["val_seen", "val_unseen"],
            "order_seed": 0,
            "settings": ["duet-r2r"],
            "selection": {"floor_metric": "SR", "floor_epsilon": 0.3,
                          "gain_metric": "SR"},
            "methods": {"tent": {"base": {"norm_scope": "last_k_ln"},
                                 "grid": {"lr": [1e-3]},
                                 "paper_default": {"lr": 1e-4}}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            source_root = root / "source"
            self._write_console(source_root / "duet-r2r" / "val_seen", "val_seen", 78.0, 72.0)
            self._write_console(source_root / "duet-r2r" / "val_unseen", "val_unseen", 71.0, 60.0)
            out_dir = root / "out"
            from run_consistency_hparam_search import candidate_tag, split_result_root
            params = {"norm_scope": "last_k_ln", "lr": 1e-3}
            tag = candidate_tag("tent", params)
            self._write_console(
                split_result_root(out_dir, "duet-r2r", "tent", "c1", tag, "val_seen"),
                "val_seen", 70.0, 65.0)
            self._write_console(
                split_result_root(out_dir, "duet-r2r", "tent", "c1", tag, "val_unseen"),
                "val_unseen", 60.0, 50.0)

            run_selection(str(spec_path), str(out_dir), str(source_root), "c1",
                          set(), set(), final_stage=False)
            selected = json.loads(
                (out_dir / "duet-r2r" / "tent" / "selected_config.json").read_text()
            )
            self.assertTrue(selected["used_fallback"])
            self.assertEqual(selected["fallback_parameters"], {"lr": 1e-4})


if __name__ == "__main__":
    unittest.main()
