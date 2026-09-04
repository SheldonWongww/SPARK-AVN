import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "vln" / "scripts" / "run_source_eval.sh"
BASE_TRAINERS = (
    REPO_ROOT
    / "vln"
    / "baselines"
    / "etpnav"
    / "vlnce_baselines"
    / "common"
    / "base_il_trainer.py",
    REPO_ROOT
    / "vln"
    / "baselines"
    / "bevbert"
    / "bevbert_ce"
    / "vlnce_baselines"
    / "common"
    / "base_il_trainer.py",
)


class SourceRunnerGuardTest(unittest.TestCase):
    @staticmethod
    def _order_config_gate(config, seed=None, cli_present=True):
        source = RUNNER.read_text(encoding="utf-8")
        marker = 'import json\nimport sys\n\nwith open(sys.argv[1], "r", encoding="utf-8")'
        start = source.index(marker)
        end = source.index("\nPY\n", start)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            return subprocess.run(
                [
                    "python3", "-", str(path),
                    "1" if cli_present else "0",
                    str(seed) if cli_present else "",
                ],
                input=source[start:end],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

    def test_runner_has_no_formal_manifest_or_clean_tree_gate(self):
        source = RUNNER.read_text(encoding="utf-8")

        for marker in (
            "check_formal_git_state",
            "create_run_manifest.py",
            "finalize_run_manifest.py",
            "validate_run_manifest.py",
            "RUN_MANIFEST_PATH",
            "FORMAL_RUN",
        ):
            with self.subTest(marker=marker):
                self.assertNotIn(marker, source)

    def test_server_roots_are_explicit_and_not_remapped(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("REPO_ROOT=/data1/wxy/code/NavTTA", source)
        self.assertIn("VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln", source)
        self.assertIn('ENV_ROOT="${VLN_ROOT}/envs"', source)
        self.assertIn('CACHE_ROOT="${VLN_ROOT}/cache"', source)
        self.assertIn('TMP_ROOT="${VLN_ROOT}/tmp"', source)
        self.assertNotIn("runtime_paths.sh", source)
        self.assertNotIn("/root/autodl-tmp", source)

    def test_pre_env_guards_use_the_setting_environment_python(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'BOOTSTRAP_PYTHON="${ENV_ROOT}/'
            '${BOOTSTRAP_ENV_NAME}/bin/python"',
            source,
        )
        for environment in ("duet", "hamt", "goat", "vlnce017", "streamvln"):
            self.assertIn("BOOTSTRAP_ENV_NAME={}".format(environment), source)
        self.assertIn('missing bootstrap Python: ${BOOTSTRAP_PYTHON}', source)
        self.assertNotRegex(source, r"(?m)^\s*python3\b")

    def test_every_runtime_prefers_and_verifies_current_repository_core(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn('CORE_ROOT="${REPO_ROOT}/core"', source)
        self.assertIn('export PYTHONPATH="${CORE_ROOT}"', source)
        self.assertIn(
            'export NAVTTA_EXPECTED_CORE_ROOT="${CORE_ROOT}"', source
        )
        self.assertIn("actual = Path(navtta_core.__file__).resolve()", source)
        self.assertIn("actual.relative_to(expected)", source)
        for baseline in ("duet", "hamt", "goat"):
            self.assertIn(
                'export PYTHONPATH="${CORE_ROOT}:${MATTERSIM_BUILD}:'
                '${REPO_ROOT}/vln:${REPO_ROOT}/vln/baselines/' + baseline,
                source,
            )

    def test_feedtta_llm_uses_only_a_verified_headless_mattersim_override(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("NAVTTA_REVERIE_RENDER_MATTERSIM_BUILD", source)
        self.assertIn('TTA_FEEDBACK_PROVIDER}" == "qwen2_vl_2b_v1', source)
        self.assertIn("EGL_RENDERING", source)
        self.assertIn("OSMESA_RENDERING", source)
        self.assertIn("sum(values.values()) != 1", source)
        self.assertIn('MATTERSIM_BUILD="${RENDER_BUILD}"', source)

    def test_base_test_inference_reads_annotations_without_ground_truth(self):
        for path in BASE_TRAINERS:
            with self.subTest(path=path):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                trainer = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef)
                    and node.name == "BaseVLNCETrainer"
                )
                inference = next(
                    node
                    for node in trainer.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "inference"
                )
                called_attributes = {
                    node.func.attr
                    for node in ast.walk(inference)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                }
                self.assertIn("collect_infer_traj", called_attributes)
                self.assertNotIn("collect_val_traj", called_attributes)

    def test_ce_prefix_keeps_manifest_episode_count_canonical(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("CE_EPISODE_COUNT=-1", source)
        self.assertNotIn('CE_EPISODE_COUNT="${EPISODE_LIMIT}"', source)
        self.assertNotIn('CE_EPISODE_COUNT="${SMOKE_EPISODES}"', source)
        self.assertIn('export NAVTTA_SMOKE_EPISODES="${EPISODE_LIMIT}"', source)

    def test_full_tta_runs_retain_prediction_evidence(self):
        source = RUNNER.read_text(encoding="utf-8")

        start = source.index("append_submit_flag() {")
        end = source.index("\n}\n", start)
        function = source[start:end]
        self.assertIn('-n "${TTA_CONFIG}"', function)
        self.assertIn("COMMAND+=(--submit)", function)

    def test_tta_cannot_silently_drop_config_through_split_all(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'if [[ "${SPLIT}" == "all" && -n "${TTA_CONFIG}" ]]', source
        )
        self.assertIn(
            "TTA jobs require one explicit split; --tta-config cannot use split all",
            source,
        )

    def test_streamvln_checks_the_scene_that_failed_before_model_loading(self):
        source = RUNNER.read_text(encoding="utf-8")

        checker = (
            '"${REPO_ROOT}/vln/scripts/verify_streamvln_scene_assets.py"'
        )
        self.assertIn(checker, source)
        self.assertIn("1pXnuDYAj8r", source)
        guard = (
            'if [[ "${DRY_RUN}" -eq 0 && "${SPLIT}" == "val_seen" ]]; then'
        )
        self.assertLess(source.rindex(guard, 0, source.index(checker)), source.index(checker))
        self.assertLess(source.index(checker), source.index('MODEL_DIR="${CHECKPOINT_ROOT}/streamvln/'))

    def test_tuning_result_root_override_is_narrow_and_path_bound(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("--result-root DIR", source)
        self.assertIn(
            "--result-root is reserved for TTA search jobs", source
        )
        self.assertIn(
            "--result-root must stay inside vln/results/tuning", source
        )
        self.assertIn(
            "--result-root must end with RUN_TAG/SPLIT", source
        )
        self.assertIn(
            "--result-root is restricted to validation or test tuning jobs", source
        )
        self.assertIn("val_seen|val_unseen|test", source)

    def test_adapter_parity_namespace_requires_its_dedicated_runner_flag(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("--adapter-parity-audit", source)
        self.assertIn(
            'adapter-parity config requires --adapter-parity-audit', source
        )
        self.assertIn(
            '--adapter-parity-audit requires the adapter-parity config schema',
            source,
        )
        self.assertIn(
            'vln/results/audits/adapter_parity/runs/${RUN_TAG}', source
        )
        self.assertIn(
            'adapter-parity audit requires --episode-limit 256', source
        )
        self.assertNotIn("create_episode_order_prefix.py", source)

    def test_idea_source_statistics_binding_is_still_checked(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("IDEA source-statistics SHA256 mismatch", source)
        self.assertIn("IDEA source_stats_path must be absolute", source)

    def test_order_seed_is_narrow_and_controls_runtime_seed(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn('--order-seed 0|1|2|3', source)
        self.assertIn('0|1|2|3) ORDER_SEED="$2"', source)
        self.assertIn('"stage": "orders"', source)
        self.assertIn('(("episodes", -1), ("order_seed", seed))', source)
        self.assertIn('type(value) is not int', source)
        self.assertIn('order_seed_%s/%s', source)
        self.assertIn('--seed "${MODEL_SEED}"', source)
        self.assertIn('TASK_CONFIG.SEED "${MODEL_SEED}"', source)
        for arguments, message in (
            (("duet-r2r", "val_seen", "--order-seed", "4"),
             "exactly 0, 1, 2, or 3"),
            (("duet-r2r", "val_seen", "--order-seed", "1"),
             "requires a TTA robustness job config"),
            (("duet-r2r", "val_unseen", "--order-seed", "3"),
             "requires a TTA robustness job config"),
            (("streamvln-r2r-ce", "val_seen", "--tta-config", "missing.json",
              "--order-seed", "1"), "does not support StreamVLN"),
            (("duet-r2r", "all", "--tta-config", "missing.json",
              "--order-seed", "1"), "complete val_seen or val_unseen"),
            (("duet-r2r", "test", "--tta-config", "missing.json",
              "--order-seed", "1"), "complete val_seen or val_unseen"),
            (("duet-r2r", "val_seen", "--tta-config", "missing.json",
              "--smoke-episodes", "1", "--order-seed", "1"),
             "cannot be combined with --smoke-episodes"),
            (("duet-r2r", "val_seen", "--tta-config", "missing.json",
              "--episode-limit", "2", "--order-seed", "1"),
             "cannot be combined with --episode-limit"),
        ):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [str(RUNNER), *arguments],
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_order_config_gate_rejects_boolean_and_float_integer_lookalikes(self):
        valid = {
            "schema": "navtta.vln_tta_job.v1",
            "method": "tent",
            "search_method": "tent",
            "stage": "orders",
            "episodes": -1,
            "order_seed": 1,
        }
        self.assertEqual(self._order_config_gate(valid, 1).returncode, 0)
        cases = (
            ("boolean order seed 1", "order_seed", True, 1),
            ("boolean order seed 0", "order_seed", False, 0),
            ("float order seed", "order_seed", 1.0, 1),
            ("false episodes", "episodes", False, 1),
            ("true episodes", "episodes", True, 1),
            ("float episodes", "episodes", -1.0, 1),
        )
        for label, key, value, seed in cases:
            config = dict(valid)
            config["order_seed"] = seed
            config[key] = value
            with self.subTest(label=label):
                result = self._order_config_gate(config, seed)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("expected exact integer", result.stderr)

    def test_order_config_gate_requires_bidirectional_cli_coupling(self):
        orders = {
            "schema": "navtta.vln_tta_job.v1",
            "method": "tent",
            "search_method": "tent",
            "stage": "orders",
            "episodes": -1,
            "order_seed": 1,
        }
        result = self._order_config_gate(orders, cli_present=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--order-seed is absent", result.stderr)

        result = self._order_config_gate(orders, seed=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected exact integer", result.stderr)

        ordinary = {
            "schema": "navtta.vln_tta_job.v1",
            "method": "tent",
            "search_method": "tent",
            "stage": "final",
            "episodes": -1,
            "parameters": {},
        }
        self.assertEqual(
            self._order_config_gate(ordinary, cli_present=False).returncode, 0
        )
        for smuggled_seed in (None, 1):
            smuggled = dict(ordinary, order_seed=smuggled_seed)
            with self.subTest(smuggled_seed=smuggled_seed):
                result = self._order_config_gate(smuggled, cli_present=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--order-seed is absent", result.stderr)

        result = self._order_config_gate(ordinary, seed=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stage mismatch", result.stderr)

        audit = {
            "schema": "navtta.vln_tta_adapter_parity_job.v1",
            "namespace": "adapter_parity_audit",
            "method": "tent",
            "episodes": 256,
            "order_seed": 0,
            "parameters": {},
        }
        self.assertEqual(
            self._order_config_gate(audit, cli_present=False).returncode, 0
        )
        for invalid_seed in (1, 2, True, False, 0.0, None, "0"):
            invalid_audit = dict(audit, order_seed=invalid_seed)
            with self.subTest(adapter_audit_order_seed=invalid_seed):
                result = self._order_config_gate(
                    invalid_audit, cli_present=False
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("exact integer 0", result.stderr)


if __name__ == "__main__":
    unittest.main()
