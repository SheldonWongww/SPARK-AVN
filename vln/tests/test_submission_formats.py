import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

from vln.scripts import canonicalize_discrete_submission


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICALIZER = (
    REPO_ROOT / "vln" / "scripts" / "canonicalize_discrete_submission.py"
)
DISCRETE_VALIDATOR = (
    REPO_ROOT / "vln" / "scripts" / "validate_discrete_submission.py"
)
CE_VALIDATOR = REPO_ROOT / "vln" / "scripts" / "validate_r2r_ce_submission.py"
SOURCE_RUNNER = REPO_ROOT / "vln" / "scripts" / "run_source_eval.sh"


class DiscreteSubmissionCanonicalizationTest(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "scan": "scan-a",
            "start_viewpoint": "start",
            "start_heading": 1.25,
        }
        self.candidates = {
            "scan-a_start": {"middle": [16, 0.0, 0.0, 1.0]},
            "scan-a_middle": {"end": [29, 0.0, 0.0, 1.0]},
        }

    def test_graph_segments_become_official_triplets(self):
        trajectory = [["start"], ["middle", "end"], ["end"]]

        converted = canonicalize_discrete_submission.canonicalize_trajectory(
            trajectory, self.metadata, self.candidates, "1_0"
        )

        self.assertEqual(
            [step[0] for step in converted],
            ["start", "middle", "end", "end"],
        )
        self.assertEqual(converted[0], ["start", 1.25, 0.0])
        self.assertAlmostEqual(converted[1][1], math.radians(120.0))
        self.assertAlmostEqual(converted[1][2], 0.0)
        self.assertAlmostEqual(converted[2][1], math.radians(150.0))
        self.assertAlmostEqual(converted[2][2], math.radians(30.0))
        self.assertEqual(converted[3][1:], converted[2][1:])

    def test_existing_triplets_are_preserved(self):
        trajectory = [
            ["start", 1.25, 0.0],
            ["middle", 2.0, -0.5],
        ]

        converted = canonicalize_discrete_submission.canonicalize_trajectory(
            trajectory, self.metadata, self.candidates, "1_0"
        )

        self.assertEqual(converted, trajectory)

    def test_missing_navigation_edge_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no navigable edge"):
            canonicalize_discrete_submission.canonicalize_trajectory(
                [["start"], ["missing"]],
                self.metadata,
                self.candidates,
                "1_0",
            )

    def test_in_place_cli_preserves_reverie_object_and_validates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "reverie.json"
            candidates = root / "candidates.json"
            submission = root / "submission.json"
            manifest = root / "test.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "id": "route",
                            "scan": "scan-a",
                            "path": ["start"],
                            "heading": 1.25,
                            "instructions": ["find it"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            candidates.write_text(
                json.dumps(self.candidates), encoding="utf-8"
            )
            submission.write_text(
                json.dumps(
                    [
                        {
                            "instr_id": "route_0",
                            "trajectory": [["start"], ["middle"]],
                            "predObjId": 7,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    str(CANONICALIZER),
                    "--task",
                    "reverie",
                    "--submission",
                    str(submission),
                    "--output",
                    str(submission),
                    "--dataset",
                    str(dataset),
                    "--scanvp-candidates",
                    str(candidates),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            converted = json.loads(submission.read_text(encoding="utf-8"))
            self.assertEqual(converted[0]["predObjId"], 7)
            self.assertEqual(len(converted[0]["trajectory"][0]), 3)

            dataset_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "navtta.episode_order.v1",
                        "split": "test",
                        "benchmark": "synthetic-reverie",
                        "dataset": {"sha256": dataset_digest},
                        "episodes": [
                            {"episode_id": "route_0", "scene_id": "scan-a"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            validation = subprocess.run(
                [
                    "python3",
                    str(DISCRETE_VALIDATOR),
                    "--task",
                    "reverie",
                    "--submission",
                    str(submission),
                    "--manifest",
                    str(manifest),
                    "--expected-benchmark",
                    "synthetic-reverie",
                    "--dataset",
                    str(dataset),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(validation.returncode, 0, validation.stderr)


class ContinuousSubmissionValidationTest(unittest.TestCase):
    def _run_validator(self, states):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        dataset = root / "dataset.json"
        manifest = root / "manifest.json"
        submission = root / "submission.json"
        dataset.write_text(
            json.dumps(
                {
                    "episodes": [
                        {
                            "episode_id": "1",
                            "start_position": [0.0, 0.0, 0.0],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
        manifest.write_text(
            json.dumps(
                {
                    "schema": "navtta.episode_order.v1",
                    "split": "test",
                    "benchmark": "synthetic-r2r-ce",
                    "dataset": {"sha256": digest},
                    "episodes": [{"episode_id": "1", "scene_id": "scan-a"}],
                }
            ),
            encoding="utf-8",
        )
        submission.write_text(json.dumps({"1": states}), encoding="utf-8")
        return subprocess.run(
            [
                "python3",
                str(CE_VALIDATOR),
                "--submission",
                str(submission),
                "--manifest",
                str(manifest),
                "--expected-benchmark",
                "synthetic-r2r-ce",
                "--dataset",
                str(dataset),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_in_place_turn_is_allowed(self):
        result = self._run_validator(
            [
                {"position": [0.0, 0.0, 0.0], "heading": 0.0, "stop": False},
                {"position": [0.0, 0.0, 0.0], "heading": 0.5, "stop": False},
                {"position": [0.25, 0.0, 0.0], "heading": 0.5, "stop": True},
            ]
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_simulator_recorded_slope_or_slide_displacement_is_allowed(self):
        result = self._run_validator(
            [
                {"position": [0.0, 0.0, 0.0], "heading": 0.0, "stop": False},
                {
                    "position": [0.04, -0.58, 0.21],
                    "heading": 0.0,
                    "stop": True,
                },
            ]
        )

        self.assertEqual(result.returncode, 0, result.stderr)


class SourceRunnerSubmissionWiringTest(unittest.TestCase):
    def test_graph_agents_are_canonicalized_before_validation(self):
        source = SOURCE_RUNNER.read_text(encoding="utf-8")

        self.assertIn("canonicalize_discrete_output()", source)
        self.assertEqual(source.count("canonicalize_discrete_output r2r"), 2)
        self.assertEqual(source.count("canonicalize_discrete_output reverie"), 2)
        self.assertEqual(source.count("submission_viewpoint_candidates="), 3)
        self.assertIn("--scanvp-candidates", source)


if __name__ == "__main__":
    unittest.main()
