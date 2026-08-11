from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/migrate_tta_result_layout.py"
SPEC = importlib.util.spec_from_file_location("migrate_tta_layout", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class MigrationFixture:
    batch_id = "batch-unit"
    stage = "controls"
    setting = "duet-r2r"
    run_tag = "batch-unit-controls-0000-duet-r2r-source"

    def __init__(self, root, methods=("tent", "fstta")):
        self.root = Path(root)
        self.search_root = self.root / "vln/results/logs/hparam_search"
        self.tuning_root = self.root / "vln/results/tuning"
        self.runs_root = self.root / "vln/results/runs"
        self.legacy_root = (
            self.tuning_root / self.run_tag / self.setting / "val_seen"
        )
        self.legacy_root.mkdir(parents=True)
        (self.legacy_root / "tta_diagnostics.json").write_text(
            '{"ok":true}\n', encoding="utf-8"
        )
        self.jobs = []
        self.attempt_bytes = {}
        for method in methods:
            self._job(method)
        self._formal_manifest()

    def _job(self, method):
        job_dir = (
            self.search_root / method / self.batch_id / "stages" / self.stage
            / "jobs" / "0000-duet-r2r-source"
        )
        job_dir.mkdir(parents=True)
        config_path = job_dir / "parameters.json"
        command = [
            str(self.root / "vln/scripts/run_source_eval.sh"),
            self.setting,
            "val_seen",
            "0",
            "--run-tag",
            self.run_tag,
            "--tta-config",
            str(config_path),
        ]
        job = {
            "attempt": 1,
            "base_run_tag": self.run_tag,
            "batch_id": self.batch_id,
            "command": command,
            "config_method": "source",
            "config_path": str(config_path),
            "episodes": -1,
            "job_dir": str(job_dir),
            "ordinal": 0,
            "result_root": str(self.legacy_root),
            "run_tag": self.run_tag,
            "search_method": method,
            "setting": self.setting,
            "stage": self.stage,
        }
        write_json(config_path, {
            "schema": "navtta.vln_tta_job.v1",
            "method": "source",
            "search_method": method,
            "stage": self.stage,
            "episodes": -1,
            "parameters": {"action_selection": "argmax"},
        })
        write_json(job_dir / "job.json", job)
        metrics = dict(job)
        metrics.update({
            "diagnostics_path": str(
                self.legacy_root / "tta_diagnostics.json"
            ),
            "metrics": {"SR": 1.0, "SPL": 1.0},
        })
        write_json(job_dir / "metrics.json", metrics)

        grid_path = job_dir.parents[1] / "grid.csv"
        fields = [
            "ordinal", "run_tag", "setting", "search_method", "stage",
            "config_path", "job_dir", "result_root", "command",
        ]
        with grid_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({
                "ordinal": 0,
                "run_tag": self.run_tag,
                "setting": self.setting,
                "search_method": method,
                "stage": self.stage,
                "config_path": str(config_path),
                "job_dir": str(job_dir),
                "result_root": str(self.legacy_root),
                "command": json.dumps(command),
            })

        attempt = job_dir / "attempts/attempt-00/job.json"
        attempt.parent.mkdir(parents=True)
        attempt.write_text(
            '{"historical_result_root":"%s"}\n' % self.legacy_root,
            encoding="utf-8",
        )
        self.attempt_bytes[attempt] = attempt.read_bytes()
        self.jobs.append(job_dir)

    def _formal_manifest(self):
        run_id = "{}-{}-val_seen-native".format(self.run_tag, self.setting)
        self.formal_path = self.runs_root / run_id / "manifest.json"
        manifest = {
            "run_id": run_id,
            "run_tag": self.run_tag,
            "task": "vln",
            "benchmark": "r2r",
            "model": "duet",
            "method": "source",
            "source_setting": "duet-r2r:val_seen:native:source",
            "seed": 0,
            "git_commit": "a" * 40,
            "config": str(self.jobs[0] / "parameters.json"),
            "config_overrides": [
                "--diagnostics",
                str(self.legacy_root / "tta_diagnostics.json"),
            ],
            "checkpoint": None,
            "auxiliary_checkpoints": [],
            "dataset": None,
            "pinned_manifests": {},
            "hardware": {"hostname": "fixture"},
            "started_at": "2026-08-11T00:00:00+00:00",
            "completed_at": "2026-08-11T00:01:00+00:00",
            "status": "completed",
            "exit_code": 0,
            "result_artifacts": [{
                "name": "tta_diagnostics.json",
                "path": str(self.legacy_root / "tta_diagnostics.json"),
                "size": 12,
                "sha256": "b" * 64,
            }],
        }
        manifest[MODULE.IMMUTABLE_IDENTITY_SHA256_FIELD] = (
            MODULE.immutable_identity_sha256(manifest)
        )
        write_json(self.formal_path, manifest)


class MigrateTTAResultLayoutTest(unittest.TestCase):
    def test_default_cli_is_a_read_only_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = MigrationFixture(directory)
            job_before = (fixture.jobs[0] / "job.json").read_bytes()
            output = io.StringIO()
            with redirect_stdout(output):
                code = MODULE.main([
                    fixture.batch_id, "--repo-root", directory,
                ])
            self.assertEqual(code, 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["status"], "dry_run")
            self.assertEqual(plan["groups"][0]["namespace"], "_shared")
            self.assertEqual(plan["groups"][0]["result_action"], "move")
            self.assertTrue(fixture.legacy_root.is_dir())
            self.assertEqual(
                (fixture.jobs[0] / "job.json").read_bytes(), job_before
            )
            self.assertFalse(
                (fixture.tuning_root / "_migrations"
                 / (fixture.batch_id + ".json")).exists()
            )

    def test_apply_groups_shared_root_and_updates_only_current_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = MigrationFixture(directory)
            destination = (
                fixture.tuning_root / "_shared" / fixture.batch_id
                / fixture.stage / fixture.setting / fixture.run_tag / "val_seen"
            )
            migration = MODULE.Migration(directory, fixture.batch_id).discover()
            result = migration.run(apply=True)

            self.assertFalse(fixture.legacy_root.exists())
            self.assertEqual(
                (destination / "tta_diagnostics.json").read_text(
                    encoding="utf-8"
                ),
                '{"ok":true}\n',
            )
            self.assertFalse(result["idempotent_noop"])
            self.assertEqual(result["groups"][0]["namespace"], "_shared")
            self.assertEqual(len(result["groups"][0]["current_jobs"]), 2)

            for job_dir in fixture.jobs:
                job = json.loads(
                    (job_dir / "job.json").read_text(encoding="utf-8")
                )
                self.assertEqual(job["result_root"], str(destination))
                self.assertEqual(job["result_layout"], MODULE.RESULT_LAYOUT)
                self.assertEqual(job["result_namespace"], "_shared")
                self.assertEqual(job["command"].count("--result-root"), 1)
                option = job["command"].index("--result-root")
                self.assertEqual(job["command"][option + 1], str(destination))

                config = json.loads(
                    (job_dir / "parameters.json").read_text(encoding="utf-8")
                )
                self.assertEqual(config["result_layout"], MODULE.RESULT_LAYOUT)
                self.assertEqual(config["result_namespace"], "_shared")
                self.assertEqual(config["batch_id"], fixture.batch_id)
                self.assertEqual(config["setting"], fixture.setting)
                self.assertEqual(config["run_tag"], fixture.run_tag)
                metrics = json.loads(
                    (job_dir / "metrics.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metrics["result_root"], str(destination))
                self.assertEqual(metrics["result_namespace"], "_shared")
                self.assertEqual(
                    metrics["diagnostics_path"],
                    str(destination / "tta_diagnostics.json"),
                )
                self.assertEqual(metrics["command"].count("--result-root"), 1)

                grid_path = job_dir.parents[1] / "grid.csv"
                with grid_path.open("r", encoding="utf-8", newline="") as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(row["result_root"], str(destination))
                grid_command = json.loads(row["command"])
                self.assertEqual(grid_command.count("--result-root"), 1)

            formal = json.loads(
                fixture.formal_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                formal["result_artifacts"][0]["path"],
                str(destination / "tta_diagnostics.json"),
            )
            self.assertEqual(
                formal[MODULE.IMMUTABLE_IDENTITY_SHA256_FIELD],
                MODULE.immutable_identity_sha256(formal),
            )
            for path, before in fixture.attempt_bytes.items():
                self.assertEqual(path.read_bytes(), before)

            ledger = (
                fixture.tuning_root / "_migrations"
                / (fixture.batch_id + ".json")
            )
            self.assertTrue(ledger.is_file())
            ledger_before = ledger.read_bytes()
            current_before = {
                path: path.read_bytes()
                for job_dir in fixture.jobs
                for path in (
                    job_dir / "job.json",
                    job_dir / "parameters.json",
                    job_dir / "metrics.json",
                    job_dir.parents[1] / "grid.csv",
                )
            }
            second = MODULE.Migration(
                directory, fixture.batch_id
            ).discover().run(apply=True)
            self.assertTrue(second["idempotent_noop"])
            self.assertEqual(ledger.read_bytes(), ledger_before)
            for path, before in current_before.items():
                self.assertEqual(path.read_bytes(), before)
            for path, before in fixture.attempt_bytes.items():
                self.assertEqual(path.read_bytes(), before)

    def test_apply_refuses_to_merge_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = MigrationFixture(directory, methods=("tent",))
            destination = (
                fixture.tuning_root / "tent" / fixture.batch_id
                / fixture.stage / fixture.setting / fixture.run_tag / "val_seen"
            )
            destination.mkdir(parents=True)
            (destination / "unrelated.json").write_text("{}\n", encoding="utf-8")
            job_before = (fixture.jobs[0] / "job.json").read_bytes()
            with self.assertRaisesRegex(
                    MODULE.MigrationError, "refusing to merge"):
                MODULE.Migration(
                    directory, fixture.batch_id
                ).discover().run(apply=True)
            self.assertTrue(fixture.legacy_root.is_dir())
            self.assertTrue(destination.is_dir())
            self.assertEqual(
                (fixture.jobs[0] / "job.json").read_bytes(), job_before
            )

    def test_unique_root_uses_owning_method_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = MigrationFixture(directory, methods=("tent",))
            result = MODULE.Migration(
                directory, fixture.batch_id
            ).discover().run(apply=True)
            destination = (
                fixture.tuning_root / "tent" / fixture.batch_id
                / fixture.stage / fixture.setting / fixture.run_tag / "val_seen"
            )
            self.assertEqual(result["groups"][0]["namespace"], "tent")
            self.assertTrue(destination.is_dir())
            job = json.loads(
                (fixture.jobs[0] / "job.json").read_text(encoding="utf-8")
            )
            config = json.loads(
                (fixture.jobs[0] / "parameters.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(job["result_namespace"], "tent")
            self.assertEqual(config["result_namespace"], "tent")

    def test_active_worker_blocks_even_a_stale_migration_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = MigrationFixture(directory, methods=("tent",))
            write_json(fixture.jobs[0] / "worker_state.json", {
                "status": "running",
                "worker_pid": 1234,
            })
            with self.assertRaisesRegex(
                    MODULE.MigrationError, "worker is still active"):
                MODULE.Migration(
                    directory, fixture.batch_id
                ).discover().run(apply=False)
            self.assertTrue(fixture.legacy_root.is_dir())


if __name__ == "__main__":
    unittest.main()
