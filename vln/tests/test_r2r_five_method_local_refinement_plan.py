import itertools
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = REPO_ROOT / "vln/experiments/r2r_five_method_local_refinement_v1.json"
METHODS = ("tent", "fstta", "feedtta", "atena")
SETTINGS = ("duet-r2r", "hamt-r2r", "goat-r2r")
ENABLED_SETTINGS = {
    "tent": SETTINGS,
    "fstta": SETTINGS,
    "feedtta": SETTINGS,
    "atena": ("goat-r2r",),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def expand_setting(setting_spec):
    points = []
    for grid in setting_spec.get("grids", []):
        fixed = dict(grid.get("fixed", {}))
        keys = [key for key in grid if key != "fixed"]
        for values in itertools.product(*(grid[key] for key in keys)):
            point = dict(zip(keys, values))
            point.update(fixed)
            points.append(point)
    for raw in setting_spec.get("points", []):
        points.append({key: value for key, value in raw.items() if key != "role"})
    return points


class FourMethodLowLRRefinementPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_plan_is_launchable_with_reviewed_barriers(self):
        self.assertEqual(
            self.plan["schema"],
            "navtta.vln_r2r_five_method_local_refinement_plan.v1",
        )
        execution = self.plan["execution"]
        self.assertTrue(execution["launchable"])
        self.assertTrue(execution["strict_model_barrier_required"])
        self.assertTrue(execution["strict_method_barrier_required"])
        self.assertEqual(
            execution["barrier_implementation_status"],
            "implemented_validated",
        )
        self.assertEqual(
            execution["enabled_methods_by_setting"],
            {
                "duet-r2r": [
                    "source", "tent", "fstta", "feedtta_control", "feedtta"
                ],
                "hamt-r2r": [
                    "source", "tent", "fstta", "feedtta_control", "feedtta"
                ],
                "goat-r2r": [
                    "source", "tent", "fstta", "feedtta_control", "feedtta",
                    "atena",
                ],
            },
        )

    def test_candidate_counts_and_uniqueness(self):
        by_setting = dict.fromkeys(SETTINGS, 0)
        by_method = {}
        self.assertEqual(set(self.plan["methods"]), set(METHODS))
        for method in METHODS:
            method_spec = self.plan["methods"][method]
            expected_settings = ENABLED_SETTINGS[method]
            self.assertEqual(set(method_spec["settings"]), set(expected_settings))
            total = 0
            for setting in expected_settings:
                setting_spec = method_spec["settings"][setting]
                points = expand_setting(setting_spec)
                self.assertEqual(len(points), len({canonical(p) for p in points}))
                self.assertEqual(len(points), setting_spec["candidate_count"])
                total += len(points)
                by_setting[setting] += len(points)
            self.assertEqual(total, method_spec["candidate_count"])
            by_method[method] = total

        budget = self.plan["budget"]
        self.assertEqual(by_method, budget["search_candidates"])
        self.assertEqual(by_setting, budget["by_setting_before_controls"])
        self.assertEqual(sum(by_method.values()), 466)
        self.assertEqual(budget["search_total"], 466)
        self.assertEqual(budget["total_before_confirmation"], 472)

    def test_low_lr_axes_and_method_caps(self):
        tent_lrs = [
            1e-9, 3e-9, 1e-8, 3e-8, 1e-7,
            3e-7, 1e-6, 3e-6, 1e-5, 1.5625e-5,
        ]
        for setting in SETTINGS:
            points = expand_setting(self.plan["methods"]["tent"]["settings"][setting])
            self.assertEqual(sorted(point["lr"] for point in points), tent_lrs)
            self.assertEqual({point["update_interval"] for point in points}, {1})

        fast = [3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
        slow = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
        expected_fstta = {"duet-r2r": 72, "hamt-r2r": 71, "goat-r2r": 71}
        for setting, count in expected_fstta.items():
            setting_spec = self.plan["methods"]["fstta"]["settings"][setting]
            self.assertEqual(len(setting_spec["grids"]), 2)
            for grid in setting_spec["grids"]:
                self.assertEqual(grid["lr_fast"], fast)
                self.assertEqual(grid["lr_slow"], slow)
            self.assertEqual(setting_spec["candidate_count"], count)
            self.assertLessEqual(count, 81)

        expected_feed = {"duet-r2r": 55, "hamt-r2r": 56, "goat-r2r": 56}
        for setting, count in expected_feed.items():
            setting_spec = self.plan["methods"]["feedtta"]["settings"][setting]
            self.assertEqual(len(setting_spec["grids"]), 3)
            self.assertEqual(setting_spec["candidate_count"], count)
            self.assertLessEqual(count, 250)

        self.assertEqual(
            set(self.plan["methods"]["atena"]["settings"]), {"goat-r2r"}
        )
        self.assertEqual(
            self.plan["methods"]["atena"]["settings"]["goat-r2r"][
                "candidate_count"
            ],
            55,
        )

    def test_formal_concurrency_caps_and_projection_lines(self):
        concurrency = self.plan["execution"]["concurrency_calibration"]
        expected = {
            "tent": {"duet-r2r": 10, "hamt-r2r": 10, "goat-r2r": 10},
            "fstta": {"duet-r2r": 14, "hamt-r2r": 11, "goat-r2r": 14},
            "feedtta": {"duet-r2r": 6, "hamt-r2r": 5, "goat-r2r": 6},
            "atena": {"goat-r2r": 5},
        }
        for method, settings in expected.items():
            for setting, cap in settings.items():
                policy = concurrency[method][setting]
                self.assertEqual(policy["production_cap"], cap)
                self.assertTrue(policy["cap_basis"])

        self.assertEqual(
            concurrency["fstta"]["goat-r2r"]["projected_peak_mib"],
            28102,
        )
        self.assertEqual(
            concurrency["feedtta"]["goat-r2r"]["projected_peak_mib"],
            26400,
        )
        planning_line = self.plan["execution"]["gpu_safety"][
            "production_planned_steady_used_mib_max"
        ]
        for method in ("fstta", "feedtta"):
            self.assertLess(
                concurrency[method]["goat-r2r"]["projected_peak_mib"],
                planning_line,
            )

    def test_every_parent_anchor_occurs_exactly_once(self):
        anchors = self.plan["parent_anchors"]
        self.assertEqual(set(anchors), set(METHODS))
        for method in METHODS:
            self.assertEqual(set(anchors[method]), set(ENABLED_SETTINGS[method]))
            for setting in ENABLED_SETTINGS[method]:
                anchor = anchors[method][setting]
                self.assertTrue(anchor["parent_run_tag"].startswith(
                    "vln-r2r-modelwise-cartesian-v2-seed0-"
                ))
                points = expand_setting(
                    self.plan["methods"][method]["settings"][setting]
                )
                self.assertEqual(
                    sum(
                        canonical(point) == canonical(anchor["candidate"])
                        for point in points
                    ),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
