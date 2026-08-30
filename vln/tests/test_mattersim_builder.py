from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER = REPO_ROOT / "vln" / "scripts" / "build_mattersim.sh"
RUNTIME_CHECK = REPO_ROOT / "vln" / "scripts" / "verify_runtime_imports.sh"
SOURCE_RUNNER = REPO_ROOT / "vln" / "scripts" / "run_source_eval.sh"


class MatterSimBuilderTest(unittest.TestCase):
    def test_help_and_pinned_build_contract(self):
        completed = subprocess.run(
            ["bash", str(BUILDER), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        source = BUILDER.read_text(encoding="utf-8")
        self.assertIn("589d091b111333f9e9f9d6cfd021b2eb68435925", source)
        self.assertIn("86e2ad4f77442c3350f9a2476650da6bee253c52", source)
        self.assertIn(
            "vln/data/simulators/Matterport3DSimulator", source
        )
        self.assertIn("CV_LOAD_IMAGE_ANYDEPTH", source)
        self.assertIn("cv::IMREAD_ANYDEPTH", source)
        self.assertIn('export PATH="${ENV_ROOT}/duet/bin:${PATH}"', source)
        self.assertIn('STAGED_BUILD="${SOURCE_ROOT}/build"', source)
        self.assertIn('BUILD_STAMP=.navtta-build-revisions', source)
        self.assertIn('"${SOURCE_ROOT}/LICENSE"', source)
        self.assertIn('"${SOURCE_ROOT}/pybind11/."', source)
        for environment in ("duet", "hamt", "goat"):
            self.assertIn(environment, source)

    def test_runtime_check_fails_on_mattersim_before_streamvln(self):
        source = RUNTIME_CHECK.read_text(encoding="utf-8")
        guard = source.index('build_mattersim.sh" --check')
        stream = source.index('prefix="${ENV_ROOT}/streamvln"')
        self.assertLess(guard, stream)

    def test_discrete_source_runner_uses_guarded_build_path(self):
        source = SOURCE_RUNNER.read_text(encoding="utf-8")
        self.assertIn('MATTERSIM_BUILD="${MATTERSIM_ROOT}/build"', source)
        self.assertIn('build_mattersim.sh" --check', source)
        self.assertEqual(source.count('export PYTHONPATH="${MATTERSIM_BUILD}:'), 5)


if __name__ == "__main__":
    unittest.main()
