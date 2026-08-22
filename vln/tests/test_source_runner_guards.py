import ast
import json
from pathlib import Path
import shlex
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
    def _guard_source():
        source = RUNNER.read_text(encoding="utf-8")
        start = source.index("check_formal_git_state() {")
        end = source.index("\n}\n\nFORMAL_RUN=", start) + len("\n}")
        return source[start:end]

    def _run_guard(self, repo_root, expected_commit):
        script = "\n".join(
            (
                "set -u",
                "REPO_ROOT={}".format(shlex.quote(str(repo_root))),
                "FORMAL_EXECUTION_PATHS=(core tools vln/baselines vln/scripts vln/manifests)",
                self._guard_source(),
                "check_formal_git_state {}".format(
                    shlex.quote(expected_commit)
                ),
            )
        )
        return subprocess.run(
            ["bash", "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

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

    def test_formal_git_state_is_checked_before_and_after_execution(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("check_formal_git_state()", source)
        self.assertEqual(
            source.count('check_formal_git_state "${RUN_GIT_COMMIT}"'), 3
        )
        self.assertIn("rev-parse --verify HEAD 2>&1", source)
        self.assertIn("status --porcelain --untracked-files=no 2>&1", source)
        self.assertIn("ls-files --others --exclude-standard", source)

    def test_formal_git_guard_rejects_mutation_and_command_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo = Path(temporary_directory) / "repo"
            (repo / "core").mkdir(parents=True)
            tracked = repo / "core" / "tracked.py"
            tracked.write_text("original\n", encoding="utf-8")
            for command in (
                ("git", "init", "-q"),
                ("git", "config", "user.name", "NavTTA Test"),
                ("git", "config", "user.email", "navtta-test@example.invalid"),
                ("git", "add", "core/tracked.py"),
                ("git", "commit", "-qm", "initial"),
            ):
                subprocess.run(command, cwd=repo, check=True)
            commit = subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=repo, text=True
            ).strip()

            self.assertEqual(self._run_guard(repo, commit).returncode, 0)

            tracked.write_text("changed\n", encoding="utf-8")
            result = self._run_guard(repo, commit)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("tracked worktree changes", result.stderr)
            tracked.write_text("original\n", encoding="utf-8")

            untracked = repo / "core" / "new.py"
            untracked.write_text("new\n", encoding="utf-8")
            result = self._run_guard(repo, commit)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("untracked execution files", result.stderr)
            untracked.unlink()

            history_marker = repo / "history.txt"
            history_marker.write_text("new commit\n", encoding="utf-8")
            subprocess.run(
                ("git", "add", "history.txt"), cwd=repo, check=True
            )
            subprocess.run(
                ("git", "commit", "-qm", "advance HEAD"),
                cwd=repo,
                check=True,
            )
            result = self._run_guard(repo, commit)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Git HEAD changed", result.stderr)

            result = self._run_guard(repo / "missing", commit)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("git rev-parse failed", result.stderr)

    def test_native_module_and_environment_manifest_are_pinned(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            "MatterSim.cpython-38-x86_64-linux-gnu.so", source
        )
        self.assertIn(
            'RUN_AUX_CHECKPOINTS+=("mattersim_python=${MATTERSIM_MODULE}")',
            source,
        )
        self.assertIn(
            '--environment-manifest "${REPO_ROOT}/vln/manifests/environments/eval_environments.json"',
            source,
        )
        self.assertIn("--require-immutable-identity", source)

    def test_pre_env_guards_use_the_setting_environment_python(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn(
            'BOOTSTRAP_PYTHON="/root/autodl-tmp/conda/envs/'
            '${BOOTSTRAP_ENV_NAME}/bin/python"',
            source,
        )
        for environment in ("duet", "hamt", "goat", "vlnce017", "streamvln"):
            self.assertIn("BOOTSTRAP_ENV_NAME={}".format(environment), source)
        self.assertIn('missing bootstrap Python: ${BOOTSTRAP_PYTHON}', source)
        self.assertNotRegex(source, r"(?m)^\s*python3\b")

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
            "--result-root is restricted to val_seen or val_unseen", source
        )

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
        self.assertIn("create_episode_order_prefix.py", source)
        self.assertIn('"audit_job_config=${TTA_CONFIG}"', source)
        self.assertIn(
            '"canonical_episode_order_parent=${RUN_ORDER_DIR}/${SPLIT}.json"',
            source,
        )

    def test_order_seed_is_narrow_and_controls_runtime_and_manifest_seed(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn('--order-seed 0|1|2|3', source)
        self.assertIn('0|1|2|3) ORDER_SEED="$2"', source)
        self.assertIn('"stage": "orders"', source)
        self.assertIn('(("episodes", -1), ("order_seed", seed))', source)
        self.assertIn('type(value) is not int', source)
        self.assertIn('order_seed_%s/%s', source)
        self.assertIn('--seed "${MODEL_SEED}"', source)
        self.assertIn('TASK_CONFIG.SEED "${MODEL_SEED}"', source)
        self.assertEqual(
            source.count('--source-setting "${RUN_SOURCE_SETTING}" --seed "${MODEL_SEED}"'),
            2,
        )
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
