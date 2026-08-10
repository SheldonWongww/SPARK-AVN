import ast
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

    def test_formal_git_state_is_checked_before_and_after_execution(self):
        source = RUNNER.read_text(encoding="utf-8")

        self.assertIn("check_formal_git_state()", source)
        self.assertEqual(
            source.count('check_formal_git_state "${RUN_GIT_COMMIT}"'), 2
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


if __name__ == "__main__":
    unittest.main()
