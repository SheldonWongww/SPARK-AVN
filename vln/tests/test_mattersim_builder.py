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
        self.assertIn("<jsoncpp/json/json.h>", source)
        self.assertIn("<json/json.h>", source)
        self.assertIn('NATIVE_ROOT="${ENV_ROOT}/mattersim-native"', source)
        self.assertIn('NATIVE_CMAKE="${NATIVE_BIN}/cmake"', source)
        self.assertIn('NATIVE_CC="${NATIVE_BIN}/x86_64-conda-linux-gnu-cc"', source)
        self.assertIn('NATIVE_CXX="${NATIVE_BIN}/x86_64-conda-linux-gnu-c++"', source)
        self.assertIn('OPENCV_DIR="${NATIVE_LIB}/cmake/opencv4"', source)
        self.assertIn('JSONCPP_PC="${NATIVE_LIB}/pkgconfig/jsoncpp.pc"', source)
        self.assertIn('missing native libxcrypt header', source)
        self.assertIn('export CMAKE_PREFIX_PATH="${NATIVE_ROOT}"', source)
        self.assertIn('export PATH="${ENV_ROOT}/duet/bin:${NATIVE_BIN}:${PATH}"', source)
        self.assertIn('-DPKG_CONFIG_EXECUTABLE="${NATIVE_PKG_CONFIG}"', source)
        self.assertIn('-DCMAKE_BUILD_RPATH="${NATIVE_LIB}"', source)
        self.assertIn('-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON', source)
        self.assertIn('BUILDER_CONTRACT=2', source)
        self.assertIn('builder_contract=%s', source)
        self.assertIn('native_prefix=%s', source)
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
        self.assertIn(
            'MATTERSIM_NATIVE_LIB="${ENV_ROOT}/mattersim-native/lib"', source
        )
        self.assertIn(
            'LD_LIBRARY_PATH="${SIM_BUILD}:${MATTERSIM_NATIVE_LIB}:', source
        )

    def test_discrete_source_runner_uses_guarded_build_path(self):
        source = SOURCE_RUNNER.read_text(encoding="utf-8")
        self.assertIn('MATTERSIM_BUILD="${MATTERSIM_ROOT}/build"', source)
        self.assertIn(
            'MATTERSIM_NATIVE_LIB="${ENV_ROOT}/mattersim-native/lib"', source
        )
        self.assertIn('build_mattersim.sh" --check', source)
        self.assertEqual(source.count('export PYTHONPATH="${MATTERSIM_BUILD}:'), 5)
        self.assertEqual(
            source.count(
                'export LD_LIBRARY_PATH="${MATTERSIM_BUILD}:${MATTERSIM_NATIVE_LIB}:'
            ),
            5,
        )


if __name__ == "__main__":
    unittest.main()
