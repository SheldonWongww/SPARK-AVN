import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/export_tta_hparam_search.py"
MODULE_SPEC = importlib.util.spec_from_file_location("tta_export", SCRIPT)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
SEARCH = MODULE.search


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class CampaignFixture:
    def __init__(self, root, batch_id="compact-unit"):
        self.root = Path(root)
        self.batch_id = batch_id
        self.search_root = self.root / "search"
        self.runs_root = self.root / "runs"
        self.results_root = self.root / "results"
        self.spec = SEARCH.load_spec()
        self.spec_sha = MODULE.sha256(SEARCH.SPEC_PATH)
        self.git_commit = "0123456789abcdef0123456789abcdef01234567"
        self.settings = tuple(self.spec["settings"])
        self.winners = {}
        self.assets = self.root / "assets.json"
        self.environment = self.root / "environment.json"
        self.order_root = self.root / "episode_order"
        canonical_root = self.root / "canonical_assets"
        canonical_root.mkdir()
        required_asset_ids = set(MODULE.SETTING_CHECKPOINT_ASSET.values())
        for mapping in MODULE.SETTING_AUXILIARY_ASSETS.values():
            required_asset_ids.update(
                value for value in mapping.values() if not value.startswith("@")
            )
        self.asset_files = {}
        asset_records = []
        for asset_id in sorted(required_asset_ids):
            path = canonical_root / (asset_id + ".dat")
            path.write_bytes((asset_id + "-payload").encode("utf-8"))
            self.asset_files[asset_id] = path
            asset_records.append({
                "id": asset_id,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": MODULE.sha256(path),
            })
        write_json(self.assets, {
            "schema_version": 1,
            "assets": asset_records,
        })
        self.mattersim = canonical_root / "MatterSim.so"
        self.mattersim.write_bytes(b"fixture-mattersim")
        write_json(self.environment, {
            "schema_version": 1,
            "environments": {},
            "native_dependencies": {
                "matterport3d_simulator_build": {
                    "python_module": str(self.mattersim),
                    "sha256": MODULE.sha256(self.mattersim),
                },
            },
        })
        order_assets = {}
        for order_directory in sorted(set(MODULE.SETTING_ORDER_DIRECTORY.values())):
            directory = self.order_root / order_directory
            directory.mkdir(parents=True)
            dataset = directory / "dataset.json"
            dataset.write_text('{"episodes":[]}\n', encoding="utf-8")
            benchmark = "fixture_{}".format(order_directory)
            order = directory / "val_seen.json"
            related_settings = [
                setting for setting, value in MODULE.SETTING_ORDER_DIRECTORY.items()
                if value == order_directory
            ]
            episode_count = int(self.spec["setting_episode_counts"][
                related_settings[0]
            ])
            episode_ids = [
                "{}-{:04d}".format(order_directory, index)
                for index in range(episode_count)
            ]
            write_json(order, {
                "schema": "navtta.episode_order.v1",
                "benchmark": benchmark,
                "split": "val_seen",
                "episode_count": episode_count,
                "episodes": [{
                    "episode_id": identifier,
                    "scene_id": "fixture-scene-{:02d}".format(index % 11),
                } for index, identifier in enumerate(episode_ids)],
                "dataset": {
                    "path": str(dataset),
                    "sha256": MODULE.sha256(dataset),
                },
                "order_sha256": "d" * 64,
            })
            order_assets[order_directory] = {
                "dataset": dataset,
                "order": order,
                "benchmark": benchmark,
            }
        self.setting_assets = {}
        for setting in self.settings:
            order_identity = order_assets[
                MODULE.SETTING_ORDER_DIRECTORY[setting]
            ]
            auxiliary = {}
            for name, asset_id in MODULE.SETTING_AUXILIARY_ASSETS[setting].items():
                auxiliary[name] = (
                    self.mattersim if asset_id.startswith("@")
                    else self.asset_files[asset_id]
                )
            self.setting_assets[setting] = {
                "checkpoint": self.asset_files[
                    MODULE.SETTING_CHECKPOINT_ASSET[setting]
                ],
                "auxiliary": auxiliary,
                **order_identity,
            }

    def _metrics(self, setting, primary, sr):
        if setting.endswith("reverie"):
            return {
                "SR": sr, "SPL": primary - 2.0,
                "RGS": primary + 1.0, "RGSPL": primary,
            }
        if setting.endswith("r2r-ce"):
            parsed_sr = 100.0 * float("{:.10f}".format(sr / 100.0))
            parsed_spl = 100.0 * float("{:.10f}".format(primary / 100.0))
            parsed_osr = 100.0 * float("{:.10f}".format((sr + 4.0) / 100.0))
            return {
                "SR": parsed_sr, "SUCCESS": parsed_sr,
                "SPL": parsed_spl, "OSR": parsed_osr,
                "ORACLE_SUCCESS": parsed_osr,
            }
        return {"SR": sr, "SPL": primary, "OSR": sr + 4.0}

    @staticmethod
    def _metadata(path, hash_field="sha256"):
        return {
            "path": str(path),
            "size": path.stat().st_size,
            hash_field: MODULE.sha256(path),
        }

    def _formal_manifest(self, job, artifact, successful=True):
        run_tag = job["run_tag"]
        method = job["config_method"]
        setting = job["setting"]
        identity = self.setting_assets[setting]
        artifacts = list(artifact) if isinstance(artifact, (list, tuple)) else [artifact]
        data_version = "fixture-v1"
        run_id = "{}-{}-val_seen-{}".format(run_tag, setting, data_version)
        path = self.runs_root / run_id / "manifest.json"
        manifest = {
            "run_id": run_id,
            "run_tag": run_tag,
            "task": "vln",
            "model": SEARCH.SETTING_MODEL[setting],
            "method": method,
            "benchmark": identity["benchmark"],
            "source_setting": "{}:val_seen:{}:{}".format(
                setting, data_version, method
            ),
            "seed": 0,
            "status": "completed" if successful else "failed",
            "exit_code": 0 if successful else 1,
            "git_commit": self.git_commit,
            "config": job["config_path"],
            "config_overrides": ["python", "--split", "val_seen"],
            "checkpoint": self._metadata(identity["checkpoint"]),
            "auxiliary_checkpoints": [{
                "name": name,
                **self._metadata(auxiliary_path),
            } for name, auxiliary_path in sorted(identity["auxiliary"].items())],
            "dataset": {
                **self._metadata(identity["dataset"], "index_sha256"),
                "version": identity["benchmark"],
                "stream_order_sha256": "d" * 64,
                "stream_content_sha256": MODULE.sha256(identity["dataset"]),
            },
            "pinned_manifests": {
                "assets": self._metadata(self.assets),
                "environment": self._metadata(self.environment),
                "episode_order": self._metadata(identity["order"]),
            },
            "hardware": {
                "hostname": "fixture-host",
                "platform": "fixture-linux",
                "python": "3.8.20",
                "cuda_visible_devices": "0",
                "torch": "2.1.2+cu121",
                "torch_cuda": "12.1",
                "cuda_available": True,
                "cudnn": 8902,
                "gpu_name": "Fixture GPU",
                "gpu_capability": [8, 9],
            },
            "started_at": "2026-08-11T00:00:00+00:00",
            "completed_at": "2026-08-11T00:01:00+00:00",
            "result_artifacts": [{
                "name": item.relative_to(Path(job["result_root"])).as_posix(),
                **self._metadata(item),
            } for item in artifacts],
        }
        manifest["immutable_identity_sha256"] = (
            MODULE.immutable_identity_sha256(manifest)
        )
        write_json(path, manifest)

    def _job(self, method, stage, setting, ordinal, candidate_index=0):
        run_tag = "{}-{}-{}-{}-{:04d}".format(
            self.batch_id, method, stage, setting, ordinal
        )
        job_dir = (
            self.search_root / method / self.batch_id / "stages" / stage
            / "jobs" / "{:04d}-{}".format(ordinal, setting)
        )
        result_root = self.results_root / run_tag / setting / "val_seen"
        result_root.mkdir(parents=True)
        config_method = "source" if stage in ("controls", "final_controls") else method
        parameters = (
            {"action_selection": "sample" if method == "feedtta" else "argmax",
             "action_seed": 0}
            if config_method == "source" else
            dict(SEARCH.anchor_for(method, setting, self.spec))
        )
        if config_method != "source":
            parameters["fixture_candidate"] = candidate_index
        full_count = int(self.spec["setting_episode_counts"][setting])
        episodes = (
            2 if stage == "smoke" else
            -1 if stage in ("final_controls", "final") else
            int(self.spec["screening_episodes"])
        )
        expected_episodes = full_count if episodes == -1 else episodes
        source_primary = 70.0
        primary = source_primary if config_method == "source" else (
            71.0 + candidate_index if stage == "final" else 71.0
        )
        sr = 80.0 if config_method == "source" else 80.0 + candidate_index / 10.0
        metric_values = self._metrics(setting, primary, sr)
        diagnostics = None
        adapter = None
        continuous_source = (
            config_method == "source" and setting.endswith("r2r-ce")
        )
        sampled_discrete_source = (
            config_method == "source"
            and not setting.endswith("r2r-ce")
            and parameters["action_selection"] == "sample"
        )
        if config_method != "source" or continuous_source or sampled_discrete_source:
            diagnostics = result_root / "tta_diagnostics.json"
            if sampled_discrete_source:
                adapter = {
                    "action_steps": expected_episodes * 4,
                    "episodes": expected_episodes,
                    "updates": 0,
                    "relative_param_drift": 0.0,
                    "control": "matched_source_sampling_no_update",
                }
            elif not continuous_source:
                adapter = {
                    "episodes": expected_episodes,
                    "updates": 10 + candidate_index,
                    "relative_param_drift": 0.001 * (candidate_index + 1),
                }
                if method in ("feedtta", "atena"):
                    adapter.update({
                        "feedback_observed_episodes": expected_episodes,
                        "feedback_observation_rate": 1.0,
                    })
                if method == "atena":
                    adapter.update({"queries": expected_episodes // 2,
                                    "query_rate": 0.5})
            write_json(diagnostics, {
                "schema": (
                    "navtta.vln_ce_tta.v1"
                    if setting.endswith("r2r-ce")
                    else "navtta.vln_discrete_tta.v1"
                ),
                "method": config_method,
                "stream": "val_seen",
                "episode_count": expected_episodes,
                "adapter": adapter,
            })
            # These artifacts must never be followed into the compact export.
            (result_root / "predictions.json").write_text("[]\n", encoding="utf-8")
            (result_root / "model.pth").write_bytes(b"not a real checkpoint")
            tensorboard = result_root / "tensorboard"
            tensorboard.mkdir()
            (tensorboard / "events.out.tfevents.unit").write_bytes(b"event")
            artifact = diagnostics
        else:
            artifact = result_root / "source_metrics.json"
            artifact.write_text('{"source":true}\n', encoding="utf-8")
        formal_artifacts = [artifact]
        if setting.endswith("r2r-ce") and episodes == -1:
            per_episode_path = (
                result_root / MODULE.CONTINUOUS_PER_EPISODE_ARTIFACT[setting]
            )
            order = json.loads(
                self.setting_assets[setting]["order"].read_text(encoding="utf-8")
            )
            per_episode = {}
            for index, record in enumerate(order["episodes"]):
                success = float(index % 2 == 0)
                per_episode[record["episode_id"]] = {
                    "steps_taken": float(10 + index % 50),
                    "distance_to_goal": float(index % 7) / 2.0,
                    "success": success,
                    "oracle_success": 1.0,
                    "path_length": float(5 + index % 20),
                    "collisions": float(index % 3) / 10.0,
                    "spl": success * 0.5,
                    "ndtw": 0.75,
                    "sdtw": success * 0.75,
                    "ghost_cnt": index % 30,
                }
            write_json(per_episode_path, per_episode)
            formal_artifacts.append(per_episode_path)
        job = {
            "batch_id": self.batch_id,
            "ordinal": ordinal,
            "base_run_tag": run_tag,
            "run_tag": run_tag,
            "attempt": 0,
            "setting": setting,
            "model": SEARCH.SETTING_MODEL[setting],
            "family": "continuous" if setting.endswith("r2r-ce") else "discrete",
            "benchmark": (
                "reverie" if setting.endswith("reverie")
                else "r2r-ce" if setting.endswith("r2r-ce") else "r2r"
            ),
            "search_method": method,
            "config_method": config_method,
            "stage": stage,
            "episodes": episodes,
            "order_seed": None,
            "parameters": parameters,
            "parent_run_tags": [],
            "config_path": str(job_dir / "parameters.json"),
            "job_dir": str(job_dir),
            "result_root": str(result_root),
            "command": [
                str(SEARCH.RUNNER), setting, "val_seen", "0",
                "--run-tag", run_tag,
                "--tta-config", str(job_dir / "parameters.json"),
            ] + (["--episode-limit", str(episodes)] if episodes > 0 else []),
        }
        result = dict(job)
        result.update({
            "metrics": metric_values,
            "expected_episodes": expected_episodes,
            "diagnostics_path": str(diagnostics) if diagnostics else None,
            "diagnostics_sha256": (
                MODULE.sha256(diagnostics) if diagnostics else None
            ),
            "adapter_diagnostics": adapter,
            "requires_posthoc_late_collapse_check": True,
        })
        write_json(job_dir / "job.json", job)
        write_json(job_dir / "parameters.json", {
            "schema": "navtta.vln_tta_job.v1",
            "method": config_method,
            "search_method": method,
            "stage": stage,
            "episodes": episodes,
            "parameters": parameters,
        })
        write_json(job_dir / "metrics.json", result)
        (job_dir / "exitcode").write_text("0\n", encoding="utf-8")
        write_json(job_dir / "worker_state.json", {
            "status": "finished", "exit_code": 0,
        })
        if setting.endswith("r2r-ce"):
            console_metrics = (
                "Average episode success: {:.10f}\n"
                "Average episode spl: {:.10f}\n"
                "Average episode oracle_success: {:.10f}\n"
            ).format(
                metric_values["SR"] / 100.0,
                metric_values["SPL"] / 100.0,
                metric_values["OSR"] / 100.0,
            )
        else:
            console_metrics = "Env name: val_seen, " + ", ".join(
                "{}: {:.10f}".format(key.lower(), value)
                for key, value in metric_values.items()
            ) + "\n"
        (job_dir / "console.log").write_text(
            "workdir: /server/model\n"
            "episode progress that is not retained\n" + console_metrics,
            encoding="utf-8",
        )
        if episodes == -1:
            self._formal_manifest(job, formal_artifacts)
        return job, result

    def _stage(self, method, stage):
        stage_dir = self.search_root / method / self.batch_id / "stages" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        results = []
        ordinal = 0
        finalists = int(self.spec["protocol"][
            "full_val_seen_finalists_per_setting"
        ]) if stage == "final" else 1
        for setting in self.settings:
            for candidate in range(finalists):
                job, result = self._job(
                    method, stage, setting, ordinal, candidate
                )
                jobs.append(job)
                results.append(result)
                ordinal += 1
        episodes = jobs[0]["episodes"]
        write_json(stage_dir / "stage_manifest.json", {
            "schema": "navtta.vln_tta_search_stage.v1",
            "batch_id": self.batch_id,
            "method": method,
            "stage": stage,
            "episodes": episodes,
            "job_count": len(jobs),
            "settings": list(self.settings),
            "git_commit": self.git_commit,
            "spec_sha256": self.spec_sha,
        })
        write_json(stage_dir / "SUMMARY.json", {
            "schema": "navtta.vln_tta_stage_summary.v1",
            "planned": len(jobs),
            "validated": len(jobs),
            "errors": [],
            "terminal": True,
            "promotion_ready": stage in ("stage1", "stage2", "stage3"),
            "complete": True,
        })
        fields = ["run_tag", "setting", "search_method", "config_method",
                  "stage", "order_seed", "SR", "SUCCESS", "OSR",
                  "ORACLE_SUCCESS", "SPL", "RGS", "RGSPL"]
        with (stage_dir / "metrics.csv").open(
                "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for result in results:
                row = {key: result.get(key) for key in fields}
                row.update(result["metrics"])
                writer.writerow(row)
        if stage in ("stage1", "stage2", "stage3"):
            write_json(stage_dir / "promotion_preview.json", {
                "schema": "navtta.vln_tta_promotion_preview.v1",
                "from_stage": stage,
                "to_stage": "final",
                "settings": [{
                    "setting": setting,
                    "method": method,
                    "ranked": [{
                        "run_tag": next(
                            result["run_tag"] for result in results
                            if result["setting"] == setting
                        ),
                        "eligible": True,
                        "reasons": [],
                        "score": [71.0, 80.0],
                    }],
                } for setting in self.settings],
            })
        (stage_dir / "scheduler.log").write_text(
            "scheduler_start jobs={}\nscheduler_finish failed=0 pending=0\n".format(
                len(jobs)
            ), encoding="utf-8"
        )
        with (stage_dir / "resource.csv").open(
                "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["unix_time", "gpu_memory_mib", "gpu_util_percent",
                             "cgroup_memory_gib", "running_jobs",
                             "running_run_tags"])
            writer.writerow([1, 1000, 50, 10.0, 1, jobs[0]["run_tag"]])
        return results

    def build(self, official=True):
        for method in MODULE.METHODS:
            stage_results = {}
            for stage in SEARCH.workflow_stages(method, include_orders=False):
                stage_results[stage] = self._stage(method, stage)
            final_controls = {
                item["setting"]: item for item in stage_results["final_controls"]
            }
            final = stage_results["final"]
            method_root = self.search_root / method / self.batch_id
            selection = {
                "schema": "navtta.vln_tta_final_selection.v1",
                "method": method,
                "split": "val_seen",
                "finalist_stage": "final",
                "matched_source_stage": "final_controls",
                "git_commit": self.git_commit,
                "spec_sha256": self.spec_sha,
                "settings": {},
            }
            frozen = {
                "schema": "navtta.vln_tta_frozen_hparams.v1",
                "method": method,
                "split": "val_seen",
                "generated_from": "FINAL_SELECTION.json",
                "git_commit": self.git_commit,
                "spec_sha256": self.spec_sha,
                "settings": {},
            }
            for setting in self.settings:
                candidates = [item for item in final if item["setting"] == setting]
                source = final_controls[setting]
                winners, ranking = SEARCH.rank_and_promote(
                    candidates, source, method, setting, 1, self.spec,
                    force_anchor=False,
                )
                winner = winners[0]
                selection["settings"][setting] = {
                    "winner_run_tag": winner["run_tag"],
                    "source_run_tag": source["run_tag"],
                    "frozen_parameters": winner["parameters"],
                    "winner_metrics": winner["metrics"],
                    "source_metrics": source["metrics"],
                    "selection": ranking,
                }
                frozen["settings"][setting] = winner["parameters"]
                self.winners[(method, setting)] = winner["run_tag"]
            if official:
                write_json(method_root / "FINAL_SELECTION.json", selection)
                write_json(method_root / "FROZEN_HPARAMETERS.json", frozen)
        return self

    def method_roots(self):
        return MODULE.standard_method_roots(self.search_root, self.batch_id)

    def exporter(self, output, method_roots=None, **kwargs):
        return MODULE.BatchExporter(
            self.batch_id, method_roots or self.method_roots(), self.runs_root,
            output,
            export_root=self.root,
            asset_manifest_path=self.assets,
            environment_manifest_path=self.environment,
            order_root=self.order_root,
            tuning_root=self.results_root,
            **kwargs
        )


class CompactExportTest(unittest.TestCase):
    def test_v2_result_layout_and_explicit_runner_root_are_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            exporter = object.__new__(MODULE.BatchExporter)
            exporter.tuning_root = Path(directory).absolute()
            exporter.settings = tuple(SEARCH.load_spec()["settings"])
            run_tag = "batch-eam-stage1-0000-duet-r2r-deadbeef00"
            result_root = (
                exporter.tuning_root / "eam" / "batch" / "stage1"
                / "duet-r2r" / run_tag / "val_seen"
            )
            job = {
                "batch_id": "batch",
                "search_method": "eam",
                "config_method": "eam",
                "stage": "stage1",
                "setting": "duet-r2r",
                "run_tag": run_tag,
                "config_path": "/tmp/parameters.json",
                "result_layout": SEARCH.RESULT_LAYOUT,
                "result_namespace": "eam",
                "result_root": str(result_root),
                "episodes": 256,
                "order_seed": None,
            }
            job["command"] = [
                str(SEARCH.RUNNER), "duet-r2r", "val_seen", "0",
                "--run-tag", run_tag,
                "--tta-config", job["config_path"],
                "--result-root", str(result_root),
                "--episode-limit", "256",
            ]
            self.assertEqual(
                exporter._expected_tuning_result_root(job, run_tag),
                result_root,
            )
            exporter._validated_job_command(job, "v2 fixture")

    def _single_evidence(self, directory, method="tent", setting="duet-r2r"):
        fixture = CampaignFixture(Path(directory))
        job, result = fixture._job(method, "final", setting, 0, 4)
        exporter = fixture.exporter(Path(directory) / "export")
        job_dir = Path(job["job_dir"])
        config = json.loads(
            (job_dir / "parameters.json").read_text(encoding="utf-8")
        )
        csv_row = {
            "run_tag": job["run_tag"],
            "setting": job["setting"],
            "search_method": job["search_method"],
            "config_method": job["config_method"],
            "stage": job["stage"],
            "order_seed": "",
        }
        csv_row.update({key: str(value) for key, value in result["metrics"].items()})
        manifest_path, manifest = exporter.formal_by_run_tag[job["run_tag"]][0]
        stage_manifest = {"git_commit": fixture.git_commit}
        return {
            "fixture": fixture,
            "exporter": exporter,
            "job": job,
            "job_dir": job_dir,
            "config": config,
            "result": result,
            "csv_row": csv_row,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "stage_manifest": stage_manifest,
        }

    @staticmethod
    def _authenticated_stage_manifest(evidence, commit):
        fixture = evidence["fixture"]
        job = evidence["job"]
        return {
            "schema": "navtta.vln_tta_search_stage.v1",
            "batch_id": fixture.batch_id,
            "method": job["search_method"],
            "stage": job["stage"],
            "episodes": job["episodes"],
            "job_count": 1,
            "settings": list(fixture.settings),
            "git_commit": commit,
            "spec_sha256": fixture.spec_sha,
        }

    def _collision_evidence(self, directory, setting="duet-r2r"):
        fixture = CampaignFixture(Path(directory))
        owner, _ = fixture._job(
            "tent", "final_controls", setting, 0, 0
        )
        owner_stage = (
            fixture.search_root / "tent" / fixture.batch_id
            / "stages/final_controls"
        )
        owner_manifest = {
            "schema": "navtta.vln_tta_search_stage.v1",
            "batch_id": fixture.batch_id,
            "method": "tent",
            "stage": "final_controls",
            "episodes": -1,
            "job_count": 1,
            "settings": list(fixture.settings),
            "git_commit": fixture.git_commit,
            "spec_sha256": fixture.spec_sha,
        }
        write_json(owner_stage / "stage_manifest.json", owner_manifest)

        current = copy.deepcopy(owner)
        current["search_method"] = "fstta"
        current_dir = (
            fixture.search_root / "fstta" / fixture.batch_id
            / "stages/final_controls/jobs/0000-{}".format(setting)
        )
        current["job_dir"] = str(current_dir)
        current["config_path"] = str(current_dir / "parameters.json")
        current["command"] = list(owner["command"])
        current["command"][
            current["command"].index("--tta-config") + 1
        ] = current["config_path"]
        write_json(current_dir / "job.json", current)
        write_json(current_dir / "parameters.json", {
            "schema": "navtta.vln_tta_job.v1",
            "method": "source",
            "search_method": "fstta",
            "stage": "final_controls",
            "episodes": -1,
            "parameters": current["parameters"],
        })
        current_manifest = dict(owner_manifest, method="fstta")
        return {
            "fixture": fixture,
            "exporter": fixture.exporter(Path(directory) / "export"),
            "job": current,
            "job_dir": current_dir,
            "stage_manifest": current_manifest,
            "owner": owner,
        }

    @staticmethod
    def _collision_attempt(evidence, declared=True, present=False):
        job = copy.deepcopy(evidence["job"])
        job["attempt"] = 1
        job["run_tag"] = job["base_run_tag"] + "-retry1"
        old_result_root = Path(job["result_root"])
        old_parts = list(old_result_root.parts)
        old_parts[old_parts.index(job["base_run_tag"])] = job["run_tag"]
        job["result_root"] = str(Path(*old_parts))
        run_tag_index = job["command"].index("--run-tag")
        job["command"][run_tag_index + 1] = job["run_tag"]
        attempt = evidence["job_dir"] / "attempts" / "attempt-00"
        attempt.mkdir(parents=True)
        write_json(attempt / "job.json", evidence["job"])
        archived = {}
        archived_result = attempt / "result_root"
        archived_formal = attempt / "formal_run_manifest"
        if declared:
            archived["result_root"] = str(archived_result)
            archived["formal_run_manifest"] = str(archived_formal)
        if present:
            archived_result.mkdir()
            (archived_result / "owner-result.json").write_text(
                '{"owner":"other-method"}\n', encoding="utf-8"
            )
        write_json(attempt / "archived_evidence.json", archived)
        (attempt / "console.log").write_text(
            "error: output is not empty; use a new run directory: {}\n".format(
                evidence["job"]["result_root"]
            ),
            encoding="utf-8",
        )
        (attempt / "exitcode").write_text("1\n", encoding="utf-8")
        write_json(attempt / "worker_state.json", {
            "status": "finished", "exit_code": 1,
        })
        return job, attempt

    def _successful_collision_owner(self, evidence, method, attempt_index):
        fixture = evidence["fixture"]
        job = copy.deepcopy(evidence["owner"])
        setting = job["setting"]
        job_dir = (
            fixture.search_root / method / fixture.batch_id
            / "stages/final_controls/jobs/0000-{}".format(setting)
        )
        job["search_method"] = method
        job["job_dir"] = str(job_dir)
        job["config_path"] = str(job_dir / "parameters.json")
        job["attempt"] = attempt_index
        job["run_tag"] = MODULE.BatchExporter._canonical_attempt_run_tag(
            job["base_run_tag"], attempt_index
        )
        job["result_root"] = str(
            fixture.results_root / job["run_tag"] / setting / "val_seen"
        )
        job["command"] = list(job["command"])
        job["command"][job["command"].index("--run-tag") + 1] = (
            job["run_tag"]
        )
        job["command"][job["command"].index("--tta-config") + 1] = (
            job["config_path"]
        )

        result_root = Path(job["result_root"])
        result_root.mkdir(parents=True)
        artifact = result_root / "source_metrics.json"
        artifact.write_text('{"source":true}\n', encoding="utf-8")
        metrics = fixture._metrics(setting, 70.0, 80.0)
        result = dict(job)
        result.update({
            "metrics": metrics,
            "expected_episodes": int(
                fixture.spec["setting_episode_counts"][setting]
            ),
            "diagnostics_path": None,
            "diagnostics_sha256": None,
            "adapter_diagnostics": None,
            "requires_posthoc_late_collapse_check": True,
        })
        write_json(job_dir / "job.json", job)
        write_json(job_dir / "parameters.json", {
            "schema": "navtta.vln_tta_job.v1",
            "method": "source",
            "search_method": method,
            "stage": "final_controls",
            "episodes": -1,
            "parameters": job["parameters"],
        })
        write_json(job_dir / "metrics.json", result)
        (job_dir / "exitcode").write_text("0\n", encoding="utf-8")
        write_json(job_dir / "worker_state.json", {
            "status": "finished", "exit_code": 0,
        })
        (job_dir / "console.log").write_text(
            "Env name: val_seen, " + ", ".join(
                "{}: {:.10f}".format(name.lower(), value)
                for name, value in metrics.items()
            ) + "\n",
            encoding="utf-8",
        )
        write_json(
            job_dir.parent.parent / "stage_manifest.json",
            dict(evidence["stage_manifest"], method=method),
        )
        fixture._formal_manifest(job, artifact)
        return job

    @staticmethod
    def _install_collision_ladder(job):
        job_dir = Path(job["job_dir"])
        for attempt_index in range(job["attempt"]):
            attempt_job = copy.deepcopy(job)
            attempt_job["attempt"] = attempt_index
            attempt_job["run_tag"] = (
                MODULE.BatchExporter._canonical_attempt_run_tag(
                    job["base_run_tag"], attempt_index
                )
            )
            attempt_job["result_root"] = str(
                Path(job["result_root"]).parents[2]
                / attempt_job["run_tag"] / job["setting"] / "val_seen"
            )
            attempt_job["command"][
                attempt_job["command"].index("--run-tag") + 1
            ] = attempt_job["run_tag"]
            attempt = (
                job_dir / "attempts"
                / "attempt-{:02d}".format(attempt_index)
            )
            attempt.mkdir(parents=True)
            write_json(attempt / "job.json", attempt_job)
            write_json(attempt / "archived_evidence.json", {
                "result_root": str(attempt / "result_root"),
                "formal_run_manifest": str(
                    attempt / "formal_run_manifest"
                ),
            })
            (attempt / "console.log").write_text(
                "error: output is not empty; use a new run directory: {}\n"
                .format(attempt_job["result_root"]),
                encoding="utf-8",
            )
            (attempt / "exitcode").write_text("1\n", encoding="utf-8")
            write_json(attempt / "worker_state.json", {
                "status": "finished", "exit_code": 1,
            })

    def _archived_formal_attempt(self, directory):
        fixture = CampaignFixture(Path(directory))
        job, _ = fixture._job("tent", "final", "duet-r2r", 0)
        exporter = fixture.exporter(Path(directory) / "export")
        job_dir = Path(job["job_dir"])
        attempt = job_dir / "attempts" / "attempt-00"
        attempt.mkdir(parents=True)
        write_json(attempt / "job.json", job)
        for name in ("console.log", "exitcode", "metrics.json", "worker_state.json"):
            (job_dir / name).rename(attempt / name)
        archived_result = attempt / "result_root"
        Path(job["result_root"]).rename(archived_result)
        formal_path, _ = exporter.formal_by_run_tag[job["run_tag"]][0]
        archived_formal = attempt / "formal_run_manifest"
        formal_path.parent.rename(archived_formal)
        write_json(attempt / "archived_evidence.json", {
            "result_root": str(archived_result),
            "formal_run_manifest": str(archived_formal),
        })

        current = copy.deepcopy(job)
        current["attempt"] = 1
        current["run_tag"] = current["base_run_tag"] + "-retry1"
        old_root = Path(current["result_root"])
        parts = list(old_root.parts)
        parts[parts.index(current["base_run_tag"])] = current["run_tag"]
        current["result_root"] = str(Path(*parts))
        run_tag_index = current["command"].index("--run-tag")
        current["command"][run_tag_index + 1] = current["run_tag"]
        stage_manifest = {
            "schema": "navtta.vln_tta_search_stage.v1",
            "batch_id": fixture.batch_id,
            "method": "tent",
            "stage": "final",
            "episodes": -1,
            "job_count": 1,
            "settings": list(fixture.settings),
            "git_commit": fixture.git_commit,
            "spec_sha256": fixture.spec_sha,
        }
        exporter.working_dir = Path(directory)
        return exporter, current, job_dir, attempt, stage_manifest

    def test_carriage_return_progress_is_split_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "ce-console.log"
            destination = Path(directory) / "compact.log"
            with source.open("wb") as stream:
                for index in range(2000):
                    stream.write(
                        ("val_seen: {:04d}/2000 42%".format(index)).encode("utf-8")
                        + b"\r"
                    )
                stream.write(b"Average episode success: 0.75\n")
                stream.write(b"X" * (1024 * 1024) + b"\n")
                for _ in range(1000):
                    stream.write(b"RuntimeError: bounded signal retention\n")
            MODULE.compact_console(
                source, destination, "fixture/ce-console.log"
            )
            compact = destination.read_text(encoding="utf-8")
            self.assertIn("# source_lines: 3002", compact)
            self.assertIn("# source_signal_lines: 1001", compact)
            self.assertIn("Average episode success: 0.75", compact)
            self.assertLess(len(compact.splitlines()), 310)
            self.assertLess(
                max(map(len, compact.splitlines())),
                MODULE.MAX_CONSOLE_LINE_BYTES + 100,
            )

    def test_derived_order_parent_and_stage_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            setting = "duet-r2r"
            identity = fixture.setting_assets[setting]
            parent = json.loads(identity["order"].read_text(encoding="utf-8"))
            parent.update({
                "split_ordinal": 0,
                "order_policy": "scene_id_then_natural_episode_id_v1",
                "source_id_field": "episode_id",
            })
            write_json(identity["order"], parent)
            records = MODULE.seeded_episode_records(
                parent["episodes"],
                order_seed=1,
                parent_order_sha256=parent["order_sha256"],
            )
            order_sha = hashlib.sha256(json.dumps(
                records, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")).hexdigest()
            derived = copy.deepcopy(parent)
            derived.update({
                "split_ordinal": 0,
                "order_policy": MODULE.SEEDED_ORDER_POLICY,
                "source_id_field": "episode_id",
                "order_seed": 1,
                "derivation": {
                    "algorithm": MODULE.SEEDED_ORDER_ALGORITHM,
                    "domain_separator": MODULE.SEEDED_ORDER_DOMAIN_SEPARATOR,
                    "order_seed": 1,
                    "parent_manifest_path": str(identity["order"]),
                    "parent_manifest_sha256": MODULE.sha256(identity["order"]),
                    "parent_order_sha256": parent["order_sha256"],
                },
                "episodes": records,
                "order_sha256": order_sha,
            })
            derived_path = (
                fixture.order_root / "order_seed_1"
                / MODULE.SETTING_ORDER_DIRECTORY[setting] / "val_seen.json"
            )
            write_json(derived_path, derived)
            exporter = fixture.exporter(Path(directory) / "export")
            selected = exporter._order_identity({
                "setting": setting, "stage": "orders", "order_seed": 1,
            })
            self.assertEqual(selected["path"], derived_path.absolute())
            self.assertEqual(selected["document"]["order_sha256"], order_sha)
            with self.assertRaisesRegex(MODULE.ExportError, "only the orders stage"):
                exporter._order_identity({
                    "setting": setting, "stage": "final", "order_seed": 1,
                })
            with self.assertRaisesRegex(MODULE.ExportError, "invalid order_seed"):
                exporter._order_identity({
                    "setting": setting, "stage": "orders", "order_seed": 1.0,
                })

            derived["derivation"]["parent_manifest_sha256"] = "0" * 64
            write_json(derived_path, derived)
            exporter = fixture.exporter(Path(directory) / "export-2")
            with self.assertRaisesRegex(MODULE.ExportError, "parent_manifest_sha256"):
                exporter._order_identity({
                    "setting": setting, "stage": "orders", "order_seed": 1,
                })

    def test_order_robustness_is_recomputed_from_validated_stage_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            exporter = fixture.exporter(Path(directory) / "export")
            method = "tent"
            frozen = {
                "schema": "navtta.vln_tta_frozen_hparams.v1",
                "method": method,
                "split": "val_seen",
                "generated_from": "FINAL_SELECTION.json",
                "git_commit": fixture.git_commit,
                "spec_sha256": fixture.spec_sha,
                "settings": {},
            }
            results = []
            expected_settings = {}
            for setting in fixture.settings:
                parameters = dict(SEARCH.anchor_for(method, setting, fixture.spec))
                frozen["settings"][setting] = parameters
                runs = []
                values = []
                for seed in (0, 1, 2):
                    metrics = {
                        "SR": 80.0 + seed,
                        "SPL": 70.0 + 2 * seed,
                    }
                    item = {
                        "run_tag": "{}-{}-{}".format(method, setting, seed),
                        "setting": setting,
                        "order_seed": seed,
                        "parameters": dict(parameters),
                        "metrics": metrics,
                    }
                    results.append(item)
                    values.append(item)
                    runs.append({
                        "run_tag": item["run_tag"],
                        "order_seed": seed,
                        "metrics": metrics,
                    })
                expected_settings[setting] = {
                    "frozen_parameters": parameters,
                    "runs": runs,
                    "aggregate": {
                        metric: {
                            "mean": MODULE.statistics.fmean(
                                item["metrics"][metric] for item in values
                            ),
                            "sample_std": MODULE.statistics.stdev(
                                item["metrics"][metric] for item in values
                            ),
                        } for metric in ("SPL", "SR")
                    },
                }
            document = {
                "schema": "navtta.vln_tta_order_robustness.v1",
                "method": method,
                "split": "val_seen",
                "git_commit": fixture.git_commit,
                "spec_sha256": fixture.spec_sha,
                "required_order_seeds": [0, 1, 2],
                "settings": expected_settings,
            }
            orders_stage = {"results": results}
            exporter._validate_order_robustness(
                method, document, orders_stage, frozen,
                fixture.git_commit, fixture.spec_sha,
            )

            forged = copy.deepcopy(document)
            forged["settings"][fixture.settings[0]]["aggregate"]["SPL"][
                "mean"
            ] += 1.0
            with self.assertRaisesRegex(MODULE.ExportError, "disagrees"):
                exporter._validate_order_robustness(
                    method, forged, orders_stage, frozen,
                    fixture.git_commit, fixture.spec_sha,
                )

            forged_stage = copy.deepcopy(orders_stage)
            forged_stage["results"][0]["parameters"] = {"lr": 999.0}
            with self.assertRaisesRegex(MODULE.ExportError, "frozen winner"):
                exporter._validate_order_robustness(
                    method, document, forged_stage, frozen,
                    fixture.git_commit, fixture.spec_sha,
                )

            float_seed_stage = copy.deepcopy(orders_stage)
            float_seed_stage["results"][1]["order_seed"] = 1.0
            with self.assertRaisesRegex(MODULE.ExportError, "non-integer seed"):
                exporter._validate_order_robustness(
                    method, document, float_seed_stage, frozen,
                    fixture.git_commit, fixture.spec_sha,
                )

            forged_run = copy.deepcopy(document)
            forged_run["settings"][fixture.settings[0]]["runs"][0]["metrics"][
                "SR"
            ] = -1.0
            with self.assertRaisesRegex(MODULE.ExportError, "disagrees"):
                exporter._validate_order_robustness(
                    method, forged_run, orders_stage, frozen,
                    fixture.git_commit, fixture.spec_sha,
                )

            selection = {
                "schema": "navtta.vln_tta_final_selection.v1",
                "method": method,
                "split": "val_seen",
                "finalist_stage": "final",
                "matched_source_stage": "final_controls",
                "git_commit": fixture.git_commit,
                "spec_sha256": fixture.spec_sha,
                "settings": {
                    setting: {
                        "winner_run_tag": "winner-{}".format(setting),
                        "source_run_tag": "source-{}".format(setting),
                        "frozen_parameters": frozen["settings"][setting],
                        "winner_metrics": {},
                        "source_metrics": {},
                        "selection": {},
                    } for setting in fixture.settings
                },
            }
            forged_stage_document = copy.deepcopy(document)
            forged_stage_document["settings"][fixture.settings[0]][
                "aggregate"
            ]["SPL"]["mean"] += 1.0
            documents = [
                {
                    "name": "FINAL_SELECTION.json",
                    "source_relative": "FINAL_SELECTION.json",
                    "document": selection,
                },
                {
                    "name": "FROZEN_HPARAMETERS.json",
                    "source_relative": "FROZEN_HPARAMETERS.json",
                    "document": frozen,
                },
                {
                    "name": "ORDER_ROBUSTNESS.json",
                    "source_relative": "ORDER_ROBUSTNESS.json",
                    "document": document,
                },
                {
                    "name": "ORDER_ROBUSTNESS.json",
                    "source_relative": "stages/orders/ORDER_ROBUSTNESS.json",
                    "document": forged_stage_document,
                },
            ]
            with self.assertRaisesRegex(MODULE.ExportError, "disagrees"):
                exporter._validate_official_documents(
                    method, Path(directory), documents,
                    fixture.git_commit, fixture.spec_sha,
                    {"orders": orders_stage},
                )

    def test_ordinary_job_config_rejects_explicit_null_order_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._single_evidence(directory)
            evidence["config"]["order_seed"] = None
            with self.assertRaisesRegex(
                    MODULE.ExportError, "must omit order_seed entirely"):
                evidence["exporter"]._validate_job_config(
                    evidence["job"], evidence["job_dir"], evidence["config"]
                )

    def test_legacy_null_order_seed_requires_authenticated_pinned_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._single_evidence(directory)
            evidence["config"]["order_seed"] = None
            pinned = self._authenticated_stage_manifest(
                evidence, MODULE.LEGACY_NULL_ORDER_SEED_COMMIT
            )
            evidence["exporter"]._validate_job_config(
                evidence["job"], evidence["job_dir"], evidence["config"],
                stage_manifest=pinned,
            )

            current = copy.deepcopy(pinned)
            current["git_commit"] = evidence["fixture"].git_commit
            with self.assertRaisesRegex(
                    MODULE.ExportError, "must omit order_seed entirely"):
                evidence["exporter"]._validate_job_config(
                    evidence["job"], evidence["job_dir"], evidence["config"],
                    stage_manifest=current,
                )

            forged = copy.deepcopy(pinned)
            forged["method"] = "eam"
            with self.assertRaisesRegex(
                    MODULE.ExportError, "must omit order_seed entirely"):
                evidence["exporter"]._validate_job_config(
                    evidence["job"], evidence["job_dir"], evidence["config"],
                    stage_manifest=forged,
                )

            evidence["config"]["order_seed"] = 0
            with self.assertRaisesRegex(
                    MODULE.ExportError, "must omit order_seed entirely"):
                evidence["exporter"]._validate_job_config(
                    evidence["job"], evidence["job_dir"], evidence["config"],
                    stage_manifest=pinned,
                )

    def test_pinned_stage_with_legacy_null_configs_exports_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            fixture._stage("tent", "stage1")
            stage = (
                fixture.search_root / "tent" / fixture.batch_id
                / "stages/stage1"
            )
            manifest_path = stage / "stage_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["git_commit"] = MODULE.LEGACY_NULL_ORDER_SEED_COMMIT
            write_json(manifest_path, manifest)
            config_paths = sorted((stage / "jobs").glob("*/parameters.json"))
            before = {}
            for path in config_paths:
                document = json.loads(path.read_text(encoding="utf-8"))
                document["order_seed"] = None
                write_json(path, document)
                before[path] = path.read_bytes()

            exporter = fixture.exporter(Path(directory) / "export")
            exporter.working_dir = Path(directory) / "working"
            exporter.working_dir.mkdir()
            result = exporter._validate_stage("tent", "stage1", stage)
            self.assertEqual(len(result["jobs"]), len(fixture.settings))
            self.assertEqual(
                {path: path.read_bytes() for path in config_paths}, before
            )

    def test_legacy_null_rejects_mutable_nonpinned_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._single_evidence(directory)
            modified_spec = copy.deepcopy(evidence["fixture"].spec)
            modified_spec["experiment_id"] = "mutable-forged-spec"
            modified_spec_path = Path(directory) / "modified-spec.json"
            write_json(modified_spec_path, modified_spec)
            exporter = evidence["fixture"].exporter(
                Path(directory) / "modified-export",
                spec_path=modified_spec_path,
            )
            evidence["config"]["order_seed"] = None
            forged = self._authenticated_stage_manifest(
                evidence, MODULE.LEGACY_NULL_ORDER_SEED_COMMIT
            )
            forged["spec_sha256"] = MODULE.sha256(modified_spec_path)
            with self.assertRaisesRegex(
                    MODULE.ExportError, "must omit order_seed entirely"):
                exporter._validate_job_config(
                    evidence["job"], evidence["job_dir"], evidence["config"],
                    stage_manifest=forged,
                )

    def test_restored_collision_attempt_exports_derived_recovery_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            current, _ = self._collision_attempt(
                evidence, declared=True, present=False
            )
            destination = Path(directory) / "compact-job"
            evidence["exporter"].working_dir = Path(directory)
            evidence["exporter"]._copy_attempts(
                current, evidence["job_dir"], destination, "fixture/job",
                evidence["stage_manifest"],
            )
            ledger = json.loads((
                destination / "attempts/attempt-00/evidence.json"
            ).read_text(encoding="utf-8"))
            self.assertTrue(ledger["output_collision"])
            self.assertEqual(
                ledger["recovery_status"], "owner_evidence_restored"
            )
            self.assertTrue(ledger["archived"]["result_root"]["declared"])
            self.assertFalse(ledger["archived"]["result_root"]["present"])
            self.assertTrue(ledger["collision_owner"]["authenticated"])
            self.assertEqual(
                ledger["collision_owner"]["search_method"], "tent"
            )
            self.assertEqual(
                ledger["campaign_identity"]["git_commit"],
                evidence["fixture"].git_commit,
            )
            self.assertRegex(
                ledger["source_evidence"]["attempt_job_sha256"],
                r"^[0-9a-f]{64}$",
            )

    def test_collision_attempt_held_owner_exports_authenticated_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            current, _ = self._collision_attempt(
                evidence, declared=False, present=False
            )
            destination = Path(directory) / "compact-job"
            evidence["exporter"].working_dir = Path(directory)
            evidence["exporter"]._copy_attempts(
                current, evidence["job_dir"], destination, "fixture/job",
                evidence["stage_manifest"],
            )
            ledger = json.loads((
                destination / "attempts/attempt-00/evidence.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(
                ledger["recovery_status"],
                "owner_evidence_held_before_retry",
            )
            self.assertTrue(ledger["collision_owner"]["authenticated"])

    def test_shared_source_retry_ladders_authenticate_retry_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            jobs = {
                "fstta": self._successful_collision_owner(
                    evidence, "fstta", 1
                ),
                "eam": self._successful_collision_owner(
                    evidence, "eam", 2
                ),
                "atena": self._successful_collision_owner(
                    evidence, "atena", 3
                ),
            }
            for job in jobs.values():
                self._install_collision_ladder(job)

            exporter = evidence["fixture"].exporter(
                Path(directory) / "export"
            )
            exporter.working_dir = Path(directory) / "working"
            exporter.working_dir.mkdir()
            expected_owners = {
                "fstta": ["tent"],
                "eam": ["tent", "fstta"],
                "atena": ["tent", "fstta", "eam"],
            }
            for method, job in jobs.items():
                with self.subTest(method=method):
                    destination = exporter.working_dir / method
                    exporter._copy_attempts(
                        job, Path(job["job_dir"]), destination,
                        "fixture/{}".format(method),
                        dict(evidence["stage_manifest"], method=method),
                    )
                    ledgers = [
                        json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted(
                            (destination / "attempts").glob(
                                "attempt-*/evidence.json"
                            )
                        )
                    ]
                    self.assertEqual(
                        [
                            item["collision_owner"]["search_method"]
                            for item in ledgers
                        ],
                        expected_owners[method],
                    )
                    self.assertTrue(all(
                        item["recovery_status"]
                        == "owner_evidence_restored"
                        for item in ledgers
                    ))

    def test_retry_collision_owner_keeps_canonical_retry_identity(self):
        cases = (
            ("attempt_type", "attempt number is not canonical"),
            ("attempt_tag", "run_tag is not canonical"),
            ("result_root", "result_root mismatch"),
            ("command", "command is not canonical"),
        )
        for mutation, pattern in cases:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    evidence = self._collision_evidence(directory)
                    owner = self._successful_collision_owner(
                        evidence, "fstta", 1
                    )
                    collision = copy.deepcopy(owner)
                    collision["search_method"] = "eam"
                    collision_dir = (
                        evidence["fixture"].search_root / "eam"
                        / evidence["fixture"].batch_id
                        / "stages/final_controls/jobs/0000-duet-r2r"
                    )
                    collision["job_dir"] = str(collision_dir)
                    collision["config_path"] = str(
                        collision_dir / "parameters.json"
                    )
                    collision["command"][
                        collision["command"].index("--tta-config") + 1
                    ] = collision["config_path"]

                    owner_path = Path(owner["job_dir"]) / "job.json"
                    mutated_owner = json.loads(
                        owner_path.read_text(encoding="utf-8")
                    )
                    if mutation == "attempt_type":
                        mutated_owner["attempt"] = True
                    elif mutation == "attempt_tag":
                        mutated_owner["attempt"] = 2
                    elif mutation == "result_root":
                        forged_root = str(
                            evidence["fixture"].results_root / owner["run_tag"]
                            / owner["setting"] / "val_unseen"
                        )
                        mutated_owner["result_root"] = forged_root
                        collision["result_root"] = forged_root
                    else:
                        mutated_owner["command"].append("--dry-run")
                    write_json(owner_path, mutated_owner)

                    exporter = evidence["fixture"].exporter(
                        Path(directory) / "export"
                    )
                    with self.assertRaisesRegex(MODULE.ExportError, pattern):
                        exporter._authenticated_collision_owner(
                            collision, collision_dir,
                            dict(evidence["stage_manifest"], method="eam"),
                        )

    def test_collision_attempt_rejects_retained_owner_result(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            current, _ = self._collision_attempt(
                evidence, declared=True, present=True
            )
            with self.assertRaisesRegex(
                    MODULE.ExportError, "retains archived owner evidence"):
                evidence["exporter"]._copy_attempts(
                    current, evidence["job_dir"],
                    Path(directory) / "compact-job", "fixture/job",
                    evidence["stage_manifest"],
                )

    def test_collision_attempt_requires_restored_authenticated_owner(self):
        cases = ("missing_result", "missing_job")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                evidence = self._collision_evidence(directory)
                current, _ = self._collision_attempt(
                    evidence, declared=True, present=False
                )
                if case == "missing_result":
                    shutil.rmtree(Path(evidence["owner"]["result_root"]))
                    pattern = "was not restored"
                else:
                    Path(evidence["owner"]["job_dir"], "job.json").unlink()
                    pattern = "not uniquely authenticated"
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    evidence["exporter"]._copy_attempts(
                        current, evidence["job_dir"],
                        Path(directory) / "compact-job", "fixture/job",
                        evidence["stage_manifest"],
                    )

    def test_collision_attempt_requires_exact_terminal_exit_one(self):
        cases = (
            ("missing_state", None, None, "worker state"),
            ("state_zero", "1", 0, "consistent terminal pair"),
            ("exit_two", "2", 1, "consistent terminal pair"),
            ("exit_whitespace", " 1 ", 1, "canonical integer line"),
        )
        for case, exit_text, state_code, pattern in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                evidence = self._collision_evidence(directory)
                current, attempt = self._collision_attempt(
                    evidence, declared=True, present=False
                )
                if case == "missing_state":
                    (attempt / "worker_state.json").unlink()
                else:
                    (attempt / "exitcode").write_text(
                        exit_text + "\n", encoding="utf-8"
                    )
                    write_json(attempt / "worker_state.json", {
                        "status": "finished", "exit_code": state_code,
                    })
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    evidence["exporter"]._copy_attempts(
                        current, evidence["job_dir"],
                        Path(directory) / "compact-job", "fixture/job",
                        evidence["stage_manifest"],
                    )

    def test_collision_attempt_requires_canonical_path_and_command(self):
        for case in (
                "attempt", "result_root", "current_result_root", "command"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                evidence = self._collision_evidence(directory)
                current, attempt = self._collision_attempt(
                    evidence, declared=True, present=False
                )
                archived_job_path = attempt / "job.json"
                archived_job = json.loads(
                    archived_job_path.read_text(encoding="utf-8")
                )
                if case == "attempt":
                    archived_job["attempt"] = False
                    pattern = "attempt number disagrees"
                elif case == "result_root":
                    archived_job["result_root"] = "/tmp/forged-owner-location"
                    (attempt / "console.log").write_text(
                        "error: output is not empty; use a new run directory: "
                        "/tmp/forged-owner-location\n",
                        encoding="utf-8",
                    )
                    pattern = "result_root is not canonical"
                elif case == "current_result_root":
                    current["result_root"] = str(
                        Path(directory) / current["run_tag"]
                    )
                    pattern = "current job result_root is not canonical"
                else:
                    index = archived_job["command"].index("--run-tag")
                    archived_job["command"][index + 1] = "forged-run-tag"
                    pattern = "command/run_tag is not canonical"
                write_json(archived_job_path, archived_job)
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    evidence["exporter"]._copy_attempts(
                        current, evidence["job_dir"],
                        Path(directory) / "compact-job", "fixture/job",
                        evidence["stage_manifest"],
                    )

    def test_attempt_archive_rejects_unvalidated_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            current, _ = self._collision_attempt(
                evidence, declared=True, present=False
            )
            (evidence["job_dir"] / "attempts/README.txt").write_text(
                "unexpected\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                    MODULE.ExportError, "contains a non-directory"):
                evidence["exporter"]._copy_attempts(
                    current, evidence["job_dir"],
                    Path(directory) / "compact-job", "fixture/job",
                    evidence["stage_manifest"],
                )

    def test_missing_noncollision_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            current, attempt = self._collision_attempt(
                evidence, declared=True, present=False
            )
            (attempt / "console.log").write_text(
                "RuntimeError: unrelated failure\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                    MODULE.ExportError, "missing without a verified"):
                evidence["exporter"]._copy_attempts(
                    current, evidence["job_dir"],
                    Path(directory) / "compact-job", "fixture/job",
                    evidence["stage_manifest"],
                )

    def test_retry_result_root_rejects_symlink_component_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            current, _ = self._collision_attempt(
                evidence, declared=False, present=False
            )
            outside = Path(directory) / "outside-result"
            outside.mkdir()
            (evidence["fixture"].results_root / current["run_tag"]).symlink_to(
                outside, target_is_directory=True
            )
            with self.assertRaisesRegex(MODULE.ExportError, "symlink"):
                evidence["exporter"]._copy_attempts(
                    current, evidence["job_dir"],
                    Path(directory) / "compact-job", "fixture/job",
                    evidence["stage_manifest"],
                )

    def test_noncollision_attempt_requires_worker_exit_consistency(self):
        cases = (
            ("state_mismatch", "0\n", {"status": "finished", "exit_code": 1},
             "consistent terminal pair"),
            ("noncanonical_exit", " 1 \n",
             {"status": "finished", "exit_code": 1}, "canonical integer line"),
            ("negative_zero", "-0\n",
             {"status": "finished", "exit_code": 0}, "canonical integer line"),
            ("missing_state", "1\n", None, "worker state"),
        )
        for name, exit_text, state, pattern in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                evidence = self._collision_evidence(directory)
                current, attempt = self._collision_attempt(
                    evidence, declared=False, present=False
                )
                (attempt / "console.log").write_text(
                    "RuntimeError: unrelated failure\n", encoding="utf-8"
                )
                (attempt / "exitcode").write_text(exit_text, encoding="utf-8")
                if state is None:
                    (attempt / "worker_state.json").unlink()
                else:
                    write_json(attempt / "worker_state.json", state)
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    evidence["exporter"]._copy_attempts(
                        current, evidence["job_dir"],
                        Path(directory) / "compact-job", "fixture/job",
                        evidence["stage_manifest"],
                    )

    def test_orphaned_running_attempt_is_a_valid_retry_rung(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            current, attempt = self._collision_attempt(
                evidence, declared=False, present=False
            )
            (attempt / "console.log").write_text(
                "worker disappeared before exit\n", encoding="utf-8"
            )
            (attempt / "exitcode").unlink()
            write_json(attempt / "worker_state.json", {
                "status": "running", "worker_pid": 123, "runner_pid": 456,
            })
            destination = Path(directory) / "compact-job"
            evidence["exporter"].working_dir = Path(directory)
            evidence["exporter"]._copy_attempts(
                current, evidence["job_dir"], destination, "fixture/job",
                evidence["stage_manifest"],
            )
            ledger = json.loads((
                destination / "attempts/attempt-00/evidence.json"
            ).read_text(encoding="utf-8"))
            self.assertIsNone(ledger["source_evidence"]["exitcode_sha256"])
            self.assertEqual(ledger["recovery_status"], "no_archived_output")

    def test_archived_formal_manifest_is_fully_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            exporter, current, job_dir, _, stage_manifest = (
                self._archived_formal_attempt(directory)
            )
            destination = Path(directory) / "compact-job"
            exporter._copy_attempts(
                current, job_dir, destination, "fixture/job", stage_manifest
            )
            ledger = json.loads((
                destination / "attempts/attempt-00/evidence.json"
            ).read_text(encoding="utf-8"))
            self.assertRegex(
                ledger["source_evidence"]["formal_manifest_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertTrue((
                destination / "attempts/attempt-00/formal_run_manifest.json"
            ).is_file())

        for mutation, pattern in (
                ("identity", "immutable identity SHA256 mismatch"),
                ("extra", "exactly manifest.json")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                exporter, current, job_dir, attempt, stage_manifest = (
                    self._archived_formal_attempt(directory)
                )
                formal_root = attempt / "formal_run_manifest"
                if mutation == "identity":
                    manifest_path = formal_root / "manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["immutable_identity_sha256"] = "0" * 64
                    write_json(manifest_path, manifest)
                else:
                    (formal_root / "unexpected.txt").write_text(
                        "unexpected\n", encoding="utf-8"
                    )
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    exporter._copy_attempts(
                        current, job_dir, Path(directory) / "compact-job",
                        "fixture/job", stage_manifest,
                    )

    def test_attempt_directory_rejects_internal_unexpected_entries(self):
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                evidence = self._collision_evidence(directory)
                current, attempt = self._collision_attempt(
                    evidence, declared=False, present=False
                )
                unexpected = attempt / "unexpected"
                if kind == "file":
                    unexpected.write_text("unexpected\n", encoding="utf-8")
                    pattern = "unexpected entry"
                elif kind == "directory":
                    unexpected.mkdir()
                    pattern = "unexpected entry"
                else:
                    unexpected.symlink_to(attempt / "job.json")
                    pattern = "symlink"
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    evidence["exporter"]._copy_attempts(
                        current, evidence["job_dir"],
                        Path(directory) / "compact-job", "fixture/job",
                        evidence["stage_manifest"],
                    )

        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            current, attempt = self._collision_attempt(
                evidence, declared=True, present=True
            )
            nested = attempt / "result_root" / "nested-link"
            nested.symlink_to(attempt / "job.json")
            with self.assertRaisesRegex(MODULE.ExportError, "uses a symlink"):
                evidence["exporter"]._copy_attempts(
                    current, evidence["job_dir"],
                    Path(directory) / "compact-job", "fixture/job",
                    evidence["stage_manifest"],
                )

    def test_retry_commands_require_exact_scheduler_executable_and_shape(self):
        for mutation in ("runner", "split", "gpu", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                evidence = self._collision_evidence(directory)
                current, _ = self._collision_attempt(
                    evidence, declared=False, present=False
                )
                if mutation == "runner":
                    current["command"][0] = "/tmp/forged-runner"
                elif mutation == "split":
                    current["command"][2] = "val_unseen"
                elif mutation == "gpu":
                    current["command"][3] = "+0"
                else:
                    current["command"].append("--dry-run")
                with self.assertRaisesRegex(MODULE.ExportError, "command.*canonical"):
                    evidence["exporter"]._copy_attempts(
                        current, evidence["job_dir"],
                        Path(directory) / "compact-job", "fixture/job",
                        evidence["stage_manifest"],
                    )

    def test_batch_id_is_bound_across_job_metrics_and_retry_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._single_evidence(directory)
            evidence["job"]["batch_id"] = "other-batch"
            with self.assertRaisesRegex(MODULE.ExportError, "batch_id"):
                evidence["exporter"]._validate_job_config(
                    evidence["job"], evidence["job_dir"], evidence["config"]
                )

        with tempfile.TemporaryDirectory() as directory:
            evidence = self._single_evidence(directory)
            evidence["result"]["batch_id"] = "other-batch"
            with self.assertRaisesRegex(MODULE.ExportError, "batch_id"):
                evidence["exporter"]._validate_metrics_identity(
                    evidence["job"], evidence["result"], evidence["csv_row"],
                    evidence["result"]["expected_episodes"],
                )

        with tempfile.TemporaryDirectory() as directory:
            evidence = self._collision_evidence(directory)
            current, _ = self._collision_attempt(
                evidence, declared=False, present=False
            )
            owner_path = Path(evidence["owner"]["job_dir"]) / "job.json"
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            owner["batch_id"] = "other-batch"
            write_json(owner_path, owner)
            with self.assertRaisesRegex(MODULE.ExportError, "batch_id"):
                evidence["exporter"]._copy_attempts(
                    current, evidence["job_dir"],
                    Path(directory) / "compact-job", "fixture/job",
                    evidence["stage_manifest"],
                )

    def test_job_config_rejects_noninteger_rng_seed_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._single_evidence(
                directory, method="feedtta", setting="duet-r2r"
            )
            evidence["job"]["parameters"]["action_seed"] = False
            evidence["config"]["parameters"]["action_seed"] = False
            with self.assertRaisesRegex(
                    MODULE.ExportError, "must be an exact integer"):
                evidence["exporter"]._validate_job_config(
                    evidence["job"], evidence["job_dir"], evidence["config"]
                )

    def test_job_parameter_metrics_and_csv_identity_tampering_is_rejected(self):
        cases = (
            ("parameters", "job/parameters identity", lambda evidence: (
                evidence["config"]["parameters"].__setitem__("lr", 99.0),
                evidence["exporter"]._validate_job_config(
                    evidence["job"], evidence["job_dir"], evidence["config"]
                ),
            )),
            ("metrics identity", "job/metrics identity", lambda evidence: (
                evidence["result"].__setitem__("setting", "tampered"),
                evidence["exporter"]._validate_metrics_identity(
                    evidence["job"], evidence["result"], evidence["csv_row"],
                    evidence["result"]["expected_episodes"],
                ),
            )),
            ("metrics csv", "metrics.csv value mismatch", lambda evidence: (
                evidence["csv_row"].__setitem__("SPL", "999"),
                evidence["exporter"]._validate_metrics_identity(
                    evidence["job"], evidence["result"], evidence["csv_row"],
                    evidence["result"]["expected_episodes"],
                ),
            )),
            ("expected episodes", "expected_episodes mismatch", lambda evidence: (
                evidence["result"].__setitem__("expected_episodes", 1),
                evidence["exporter"]._validate_metrics_identity(
                    evidence["job"], evidence["result"], evidence["csv_row"],
                    int(evidence["fixture"].spec["setting_episode_counts"][
                        evidence["job"]["setting"]
                    ]),
                ),
            )),
        )
        for name, pattern, action in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                evidence = self._single_evidence(directory)
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    action(evidence)

    def test_diagnostics_path_hash_schema_and_count_tampering_is_rejected(self):
        def path_attack(evidence):
            result = evidence["result"]
            outside = Path(evidence["fixture"].root) / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            result["diagnostics_path"] = str(outside)

        def hash_attack(evidence):
            evidence["result"]["diagnostics_sha256"] = "0" * 64

        def document_attack(evidence, key, value):
            path = Path(evidence["result"]["diagnostics_path"])
            document = json.loads(path.read_text(encoding="utf-8"))
            document[key] = value
            write_json(path, document)
            evidence["result"]["diagnostics_sha256"] = MODULE.sha256(path)

        def adapter_count_attack(evidence):
            path = Path(evidence["result"]["diagnostics_path"])
            document = json.loads(path.read_text(encoding="utf-8"))
            document["adapter"]["episodes"] = 1
            write_json(path, document)
            evidence["result"]["diagnostics_sha256"] = MODULE.sha256(path)

        cases = (
            ("path", "diagnostics must be", path_attack),
            ("hash", "diagnostics SHA256 mismatch", hash_attack),
            ("schema", "schema mismatch", lambda evidence: document_attack(
                evidence, "schema", "evil.schema"
            )),
            ("count", "episode_count mismatch", lambda evidence: document_attack(
                evidence, "episode_count", 1
            )),
            ("adapter count", "adapter episode count mismatch", adapter_count_attack),
        )
        for name, pattern, attack in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                evidence = self._single_evidence(directory)
                attack(evidence)
                expected = int(evidence["fixture"].spec[
                    "setting_episode_counts"
                ][evidence["job"]["setting"]])
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    evidence["exporter"]._validate_diagnostics(
                        evidence["job"], evidence["result"], expected
                    )

    def test_all_source_diagnostics_variants_are_validated(self):
        cases = (
            ("continuous", "tent", "etpnav-r2r-ce", True, None),
            ("discrete sampled", "feedtta", "duet-r2r", True, "sampled"),
            ("discrete argmax", "tent", "duet-r2r", False, None),
        )
        for name, method, setting, has_diagnostics, adapter_kind in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = CampaignFixture(Path(directory))
                job, result = fixture._job(
                    method, "final_controls", setting, 0
                )
                exporter = fixture.exporter(Path(directory) / "export")
                expected = int(fixture.spec["setting_episode_counts"][setting])
                diagnostics = exporter._validate_diagnostics(
                    job, result, expected
                )
                self.assertEqual(diagnostics is not None, has_diagnostics)
                if adapter_kind == "sampled":
                    self.assertEqual(result["adapter_diagnostics"]["updates"], 0)
                    self.assertEqual(
                        result["adapter_diagnostics"]["relative_param_drift"],
                        0.0,
                    )
                    self.assertEqual(
                        result["adapter_diagnostics"]["control"],
                        "matched_source_sampling_no_update",
                    )
                else:
                    self.assertIsNone(result["adapter_diagnostics"])

                manifest_path, manifest = exporter.formal_by_run_tag[
                    job["run_tag"]
                ][0]
                exporter._validate_formal_manifest(
                    manifest_path, manifest, job,
                    {"git_commit": fixture.git_commit}, diagnostics,
                )
                artifact_names = {
                    item["name"] for item in manifest["result_artifacts"]
                }
                if has_diagnostics:
                    self.assertIn("tta_diagnostics.json", artifact_names)
                else:
                    self.assertNotIn("tta_diagnostics.json", artifact_names)

    def test_source_diagnostics_require_expected_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            job, result = fixture._job(
                "tent", "final_controls", "etpnav-r2r-ce", 0
            )
            diagnostics = Path(result["diagnostics_path"])
            diagnostics.unlink()
            result["diagnostics_path"] = None
            result["diagnostics_sha256"] = None
            exporter = fixture.exporter(Path(directory) / "export")
            expected = int(fixture.spec["setting_episode_counts"][job["setting"]])
            with self.assertRaisesRegex(
                    MODULE.ExportError, "omits required TTA diagnostics"):
                exporter._validate_diagnostics(job, result, expected)

        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            job, result = fixture._job(
                "feedtta", "final_controls", "duet-r2r", 0
            )
            diagnostics = Path(result["diagnostics_path"])
            document = json.loads(diagnostics.read_text(encoding="utf-8"))
            document["adapter"]["updates"] = 1
            write_json(diagnostics, document)
            result["diagnostics_sha256"] = MODULE.sha256(diagnostics)
            result["adapter_diagnostics"] = document["adapter"]
            exporter = fixture.exporter(Path(directory) / "export")
            expected = int(fixture.spec["setting_episode_counts"][job["setting"]])
            with self.assertRaisesRegex(
                    MODULE.ExportError, "sampled Source adapter performed updates"):
                exporter._validate_diagnostics(job, result, expected)

        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            job, result = fixture._job(
                "tent", "final_controls", "duet-r2r", 0
            )
            result["adapter_diagnostics"] = {"updates": 0}
            exporter = fixture.exporter(Path(directory) / "export")
            expected = int(fixture.spec["setting_episode_counts"][job["setting"]])
            with self.assertRaisesRegex(
                    MODULE.ExportError, "adapter diagnostics without evidence"):
                exporter._validate_diagnostics(job, result, expected)

    def test_source_diagnostics_must_be_a_formal_result_artifact(self):
        for method, setting in (
                ("tent", "etpnav-r2r-ce"),
                ("feedtta", "duet-r2r")):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                fixture = CampaignFixture(Path(directory))
                job, result = fixture._job(
                    method, "final_controls", setting, 0
                )
                exporter = fixture.exporter(Path(directory) / "export")
                manifest_path, manifest = exporter.formal_by_run_tag[
                    job["run_tag"]
                ][0]
                alternate = Path(job["result_root"]) / "source_metrics.json"
                alternate.write_text('{"source":true}\n', encoding="utf-8")
                manifest["result_artifacts"] = [{
                    "name": alternate.name,
                    **CampaignFixture._metadata(alternate),
                }]
                diagnostics = Path(result["diagnostics_path"])
                with self.assertRaisesRegex(
                        MODULE.ExportError,
                        "formal artifacts omit tta_diagnostics.json"):
                    exporter._validate_formal_manifest(
                        manifest_path, manifest, job,
                        {"git_commit": fixture.git_commit}, diagnostics,
                    )

    def test_symlinked_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._single_evidence(directory)
            parameters = evidence["job_dir"] / "parameters.json"
            replacement = evidence["job_dir"] / "parameters-real.json"
            parameters.rename(replacement)
            os.symlink(replacement, parameters)
            with self.assertRaisesRegex(MODULE.ExportError, "symlink"):
                MODULE.require_regular_file(parameters, "parameters")

    def test_duplicate_stage_ordinals_tags_and_configs_are_rejected(self):
        attacks = ("ordinal", "run_tag", "config")
        for attack in attacks:
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as directory:
                fixture = CampaignFixture(Path(directory))
                fixture._stage("tent", "final")
                stage = (
                    fixture.search_root / "tent" / fixture.batch_id / "stages"
                    / "final"
                )
                job_paths = sorted((stage / "jobs").glob("*/job.json"))[:2]
                first = json.loads(job_paths[0].read_text(encoding="utf-8"))
                second = json.loads(job_paths[1].read_text(encoding="utf-8"))
                if attack == "ordinal":
                    second["ordinal"] = first["ordinal"]
                    write_json(job_paths[1], second)
                    pattern = "ordinals"
                elif attack == "run_tag":
                    second["run_tag"] = first["run_tag"]
                    write_json(job_paths[1], second)
                    metrics = json.loads(
                        (job_paths[1].parent / "metrics.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    metrics["run_tag"] = first["run_tag"]
                    write_json(job_paths[1].parent / "metrics.json", metrics)
                    rows = []
                    with (stage / "metrics.csv").open(
                            "r", encoding="utf-8", newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    rows[1]["run_tag"] = rows[0]["run_tag"]
                    with (stage / "metrics.csv").open(
                            "w", encoding="utf-8", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    pattern = "duplicate/empty metrics.csv run tags"
                else:
                    second["parameters"] = first["parameters"]
                    write_json(job_paths[1], second)
                    config = json.loads(
                        (job_paths[1].parent / "parameters.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    config["parameters"] = first["parameters"]
                    write_json(job_paths[1].parent / "parameters.json", config)
                    metrics = json.loads(
                        (job_paths[1].parent / "metrics.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    metrics["parameters"] = first["parameters"]
                    write_json(job_paths[1].parent / "metrics.json", metrics)
                    pattern = "duplicate configurations"
                exporter = fixture.exporter(
                    Path(directory) / "export",
                    allow_legacy_prefix_controls=True,
                )
                exporter.working_dir = Path(directory) / "working"
                exporter.working_dir.mkdir()
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    exporter._validate_stage("tent", "final", stage)

    def test_self_consistent_summary_metrics_tamper_is_rejected_by_reparse(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            fixture._stage("tent", "final")
            stage = (
                fixture.search_root / "tent" / fixture.batch_id / "stages"
                / "final"
            )
            metrics_path = sorted((stage / "jobs").glob("*/metrics.json"))[0]
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["metrics"]["SPL"] = 999.0
            write_json(metrics_path, metrics)
            with (stage / "metrics.csv").open(
                    "r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            target_tag = metrics["run_tag"]
            for row in rows:
                if row["run_tag"] == target_tag:
                    row["SPL"] = "999.0"
            with (stage / "metrics.csv").open(
                    "w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            exporter = fixture.exporter(
                Path(directory) / "export",
                allow_legacy_prefix_controls=True,
            )
            exporter.working_dir = Path(directory) / "working"
            exporter.working_dir.mkdir()
            with self.assertRaisesRegex(
                    MODULE.ExportError, "parse_metrics disagrees.*metrics"):
                exporter._validate_stage("tent", "final", stage)

    def test_formal_manifests_are_required_only_for_full_stream_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            prefix_job, _ = fixture._job(
                "tent", "stage1", "duet-r2r", 0, 0
            )
            exporter = fixture.exporter(Path(directory) / "export")
            self.assertIsNone(exporter._formal_manifest_for(
                prefix_job, successful=True, full_stream=False
            ))

            full_job, _ = fixture._job(
                "tent", "final", "duet-r2r", 1, 0
            )
            # The exporter index was created before the full job manifest.
            with self.assertRaisesRegex(MODULE.ExportError, "missing formal"):
                exporter._formal_manifest_for(
                    full_job, successful=True, full_stream=True
                )

            fixture._formal_manifest(
                prefix_job,
                Path(prefix_job["result_root"]) / "tta_diagnostics.json",
            )
            exporter_with_prefix_manifest = fixture.exporter(
                Path(directory) / "export-2"
            )
            with self.assertRaisesRegex(
                    MODULE.ExportError, "prefix job unexpectedly"):
                exporter_with_prefix_manifest._formal_manifest_for(
                    prefix_job, successful=True, full_stream=False
                )

    def test_formal_manifest_identity_and_referenced_evidence_attacks_rejected(self):
        attacks = (
            ("schema", "misses fields"),
            ("commit", "git_commit mismatch"),
            ("setting_split", "source_setting mismatch"),
            ("seed", "seed mismatch"),
            ("seed_bool", "seed must be an exact integer"),
            ("config", "config reference mismatch"),
            ("immutable", "immutable identity SHA256 mismatch"),
            ("checkpoint", "checkpoint SHA256 mismatch"),
            ("checkpoint_rehashed", "checkpoint is not the canonical"),
            ("auxiliary_rehashed", "auxiliary assets are not canonical"),
            ("dataset", "dataset SHA256 mismatch"),
            ("dataset_rehashed", "dataset/order paths disagree"),
            ("order", "pinned episode_order is not canonical"),
            ("hardware", "hardware metadata is incomplete"),
            ("pinned", "pinned manifests are incomplete"),
            ("pinned_schema", "asset manifest schema is invalid"),
            ("artifact_escape", "artifact name/path mismatch"),
            ("artifact_dotdot", "artifact name is not canonical"),
            ("artifact_hash", "result artifact.*SHA256 mismatch"),
        )
        identity_attacks = {
            "commit", "setting_split", "seed", "seed_bool", "config", "order",
            "hardware", "pinned",
            "pinned_schema",
            "checkpoint_rehashed", "auxiliary_rehashed", "dataset_rehashed",
        }
        for attack, pattern in attacks:
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as directory:
                evidence = self._single_evidence(directory)
                manifest = copy.deepcopy(evidence["manifest"])
                if attack == "schema":
                    manifest.pop("hardware")
                elif attack == "commit":
                    manifest["git_commit"] = "f" * 40
                elif attack == "setting_split":
                    manifest["source_setting"] = (
                        "duet-r2r:val_unseen:fixture-v1:tent"
                    )
                elif attack == "seed":
                    manifest["seed"] = 7
                elif attack == "seed_bool":
                    manifest["seed"] = False
                elif attack == "config":
                    manifest["config"] = "/tampered/config.json"
                elif attack == "immutable":
                    manifest["config_overrides"].append("tampered")
                elif attack == "checkpoint":
                    target = Path(manifest["checkpoint"]["path"])
                    payload = target.read_bytes()
                    target.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                elif attack == "checkpoint_rehashed":
                    alternate = Path(directory) / "alternate-checkpoint.dat"
                    alternate.write_bytes(b"alternate-checkpoint")
                    manifest["checkpoint"] = CampaignFixture._metadata(alternate)
                elif attack == "auxiliary_rehashed":
                    alternate = Path(directory) / "alternate-auxiliary.dat"
                    alternate.write_bytes(b"alternate-auxiliary")
                    manifest["auxiliary_checkpoints"][0].update(
                        CampaignFixture._metadata(alternate)
                    )
                elif attack == "dataset":
                    target = Path(manifest["dataset"]["path"])
                    payload = target.read_bytes()
                    target.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                elif attack == "dataset_rehashed":
                    alternate = Path(directory) / "alternate-dataset.json"
                    alternate.write_text('{"episodes":[]}\n', encoding="utf-8")
                    manifest["dataset"].update(
                        CampaignFixture._metadata(alternate, "index_sha256")
                    )
                    manifest["dataset"]["stream_content_sha256"] = (
                        MODULE.sha256(alternate)
                    )
                elif attack == "order":
                    order_path = Path(
                        manifest["pinned_manifests"]["episode_order"]["path"]
                    )
                    order = json.loads(order_path.read_text(encoding="utf-8"))
                    order["split"] = "val_unseen"
                    write_json(order_path, order)
                    metadata = manifest["pinned_manifests"]["episode_order"]
                    metadata["size"] = order_path.stat().st_size
                    metadata["sha256"] = MODULE.sha256(order_path)
                elif attack == "hardware":
                    manifest["hardware"].pop("gpu_name")
                elif attack == "pinned":
                    manifest["pinned_manifests"].pop("environment")
                elif attack == "pinned_schema":
                    assets_path = Path(
                        manifest["pinned_manifests"]["assets"]["path"]
                    )
                    write_json(assets_path, {"schema_version": 999})
                    metadata = manifest["pinned_manifests"]["assets"]
                    metadata["size"] = assets_path.stat().st_size
                    metadata["sha256"] = MODULE.sha256(assets_path)
                elif attack == "artifact_escape":
                    outside = Path(directory) / "outside-result.json"
                    outside.write_text("{}\n", encoding="utf-8")
                    manifest["result_artifacts"][0] = {
                        "name": "outside-result.json",
                        **CampaignFixture._metadata(outside),
                    }
                elif attack == "artifact_dotdot":
                    outside = evidence["job_dir"] / "job.json"
                    manifest["result_artifacts"][0] = {
                        "name": "../job.json",
                        **CampaignFixture._metadata(outside),
                    }
                elif attack == "artifact_hash":
                    target = Path(manifest["result_artifacts"][0]["path"])
                    payload = target.read_bytes()
                    target.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                if attack in identity_attacks:
                    manifest["immutable_identity_sha256"] = (
                        MODULE.immutable_identity_sha256(manifest)
                    )
                diagnostics = Path(evidence["result"]["diagnostics_path"])
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    evidence["exporter"]._validate_formal_manifest(
                        evidence["manifest_path"], manifest, evidence["job"],
                        evidence["stage_manifest"], diagnostics,
                    )

    def test_cross_method_full_stream_asset_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._single_evidence(directory)
            diagnostics = Path(evidence["result"]["diagnostics_path"])
            exporter = evidence["exporter"]
            exporter._validate_formal_manifest(
                evidence["manifest_path"], evidence["manifest"],
                evidence["job"], evidence["stage_manifest"], diagnostics,
            )
            exporter.formal_identity_by_setting[
                evidence["job"]["setting"]
            ]["checkpoint_path"] = "/different/method/checkpoint"
            with self.assertRaisesRegex(
                    MODULE.ExportError, "differs across methods"):
                exporter._validate_formal_manifest(
                    evidence["manifest_path"], evidence["manifest"],
                    evidence["job"], evidence["stage_manifest"], diagnostics,
                )

    def test_continuous_per_episode_stats_tampering_is_rejected(self):
        def stats_artifact(evidence):
            expected_name = MODULE.CONTINUOUS_PER_EPISODE_ARTIFACT[
                evidence["job"]["setting"]
            ]
            return next(
                item for item in evidence["manifest"]["result_artifacts"]
                if item["name"] == expected_name
            )

        def refresh(item, path):
            item.update(CampaignFixture._metadata(path))

        def omit(evidence):
            item = stats_artifact(evidence)
            evidence["manifest"]["result_artifacts"].remove(item)

        def wrong_filename(evidence):
            item = stats_artifact(evidence)
            path = Path(item["path"])
            replacement = path.with_name(
                "stats_ep_ckpt_999_val_seen_r0_w1.json"
            )
            path.rename(replacement)
            item["name"] = replacement.relative_to(
                Path(evidence["job"]["result_root"])
            ).as_posix()
            refresh(item, replacement)

        def hash_attack(evidence):
            item = stats_artifact(evidence)
            path = Path(item["path"])
            payload = path.read_bytes()
            path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

        def mutate_document(evidence, mutation, preserve_order=False):
            item = stats_artifact(evidence)
            path = Path(item["path"])
            document = json.loads(path.read_text(encoding="utf-8"))
            document = mutation(document)
            path.write_text(
                json.dumps(
                    document, indent=2, sort_keys=not preserve_order
                ) + "\n",
                encoding="utf-8",
            )
            refresh(item, path)

        def remove_id(document):
            document.pop(next(iter(document)))
            return document

        def reverse_order(document):
            return dict(reversed(list(document.items())))

        def extra_metric(document):
            document[next(iter(document))]["trajectory"] = []
            return document

        def nonnumeric(document):
            document[next(iter(document))]["success"] = "1.0"
            return document

        def nonfinite(document):
            document[next(iter(document))]["spl"] = float("nan")
            return document

        cases = (
            ("omitted", "must contain exactly", omit),
            ("filename", "must contain exactly", wrong_filename),
            ("hash", "SHA256 mismatch", hash_attack),
            ("ID count", "ID count/order mismatch", lambda value: (
                mutate_document(value, remove_id)
            )),
            ("ID order", "ID count/order mismatch", lambda value: (
                mutate_document(value, reverse_order, preserve_order=True)
            )),
            ("metric schema", "metric schema mismatch", lambda value: (
                mutate_document(value, extra_metric)
            )),
            ("nonnumeric", "is not numeric", lambda value: (
                mutate_document(value, nonnumeric)
            )),
            ("nonfinite", "is not finite", lambda value: (
                mutate_document(value, nonfinite)
            )),
        )
        for name, pattern, attack in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                evidence = self._single_evidence(
                    directory, method="tent", setting="etpnav-r2r-ce"
                )
                attack(evidence)
                diagnostics = Path(evidence["result"]["diagnostics_path"])
                with self.assertRaisesRegex(MODULE.ExportError, pattern):
                    evidence["exporter"]._validate_formal_manifest(
                        evidence["manifest_path"], evidence["manifest"],
                        evidence["job"], evidence["stage_manifest"], diagnostics,
                    )

    def test_prefix_continuous_stats_are_optional_and_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory))
            fixture._stage("tent", "stage1")
            stage = (
                fixture.search_root / "tent" / fixture.batch_id / "stages"
                / "stage1"
            )
            ce_job_path = next(
                path for path in sorted((stage / "jobs").glob("*/job.json"))
                if json.loads(path.read_text(encoding="utf-8"))[
                    "setting"
                ] == "etpnav-r2r-ce"
            )
            ce_job = json.loads(ce_job_path.read_text(encoding="utf-8"))
            raw_stats = (
                Path(ce_job["result_root"])
                / MODULE.CONTINUOUS_PER_EPISODE_ARTIFACT["etpnav-r2r-ce"]
            )
            write_json(raw_stats, {"not": {"trajectory": [1, 2, 3]}})
            exporter = fixture.exporter(Path(directory) / "export")
            exporter.working_dir = Path(directory) / "working"
            exporter.working_dir.mkdir()
            exporter._validate_stage("tent", "stage1", stage)
            self.assertFalse(list(
                exporter.working_dir.rglob("per_episode_metrics.json")
            ))
            self.assertFalse(any(
                row["source"] == str(raw_stats) for row in exporter.source_inventory
            ))

    def test_batch_id_and_export_root_path_attacks_are_rejected(self):
        for value in ("", ".", "..", "../escape", "a/b", "/absolute"):
            with self.subTest(batch_id=value), self.assertRaisesRegex(
                    MODULE.ExportError, "safe path component"):
                MODULE.validate_batch_id(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "exports"
            outside = Path(directory) / "outside"
            exporter = MODULE.BatchExporter(
                "safe-batch", {method: Path(directory) / method
                               for method in MODULE.METHODS},
                Path(directory) / "runs", outside,
                export_root=root,
            )
            with self.assertRaisesRegex(MODULE.ExportError, "child of export root"):
                exporter._validated_output_paths()

            root.mkdir()
            victim = Path(directory) / "victim"
            victim.mkdir()
            link = root / "safe-batch"
            os.symlink(victim, link)
            exporter.output_dir = link
            with self.assertRaisesRegex(MODULE.ExportError, "symlink"):
                exporter._validated_output_paths()

    def test_current_protocol_requires_final_selection_and_frozen_files(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._single_evidence(directory)
            with self.assertRaisesRegex(
                    MODULE.ExportError, "requires root FINAL_SELECTION"):
                evidence["exporter"]._validate_official_documents(
                    "tent", Path(directory), [],
                    evidence["fixture"].git_commit,
                    evidence["fixture"].spec_sha,
                    {},
                )

    def test_complete_campaign_export_is_deterministic_and_binary_free(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory)).build()
            output_a = Path(directory) / "export-a"
            output_b = Path(directory) / "export-b"
            report_a = fixture.exporter(output_a).export()
            report_b = fixture.exporter(output_b).export()

            self.assertEqual(report_a, report_b)
            self.assertEqual(report_a["validation"]["method_setting_cells"], 40)
            expected_per_episode = (
                len(MODULE.METHODS)
                * len(MODULE.CONTINUOUS_PER_EPISODE_ARTIFACT)
                * (1 + int(fixture.spec["protocol"][
                    "full_val_seen_finalists_per_setting"
                ]))
            )
            self.assertEqual(
                report_a["validation"][
                    "continuous_per_episode_metric_exports"
                ],
                expected_per_episode,
            )
            self.assertEqual(
                len(report_a["per_episode_metrics"]), expected_per_episode
            )
            with (output_a / "selected_results.csv").open(
                    "r", encoding="utf-8", newline="") as stream:
                selected = list(csv.DictReader(stream))
            self.assertEqual(len(selected), 40)
            self.assertTrue(all(
                row["matched_source_stage"] == "final_controls"
                and row["matched_source_full_stream"] == "True"
                and len(row["checkpoint_sha256"]) == 64
                and len(row["dataset_index_sha256"]) == 64
                and len(row["episode_order_manifest_sha256"]) == 64
                for row in selected
            ))
            for row in selected:
                if row["setting"] in MODULE.CONTINUOUS_PER_EPISODE_ARTIFACT:
                    self.assertEqual(len(row["per_episode_metrics_sha256"]), 64)
                    self.assertEqual(
                        len(row[
                            "matched_source_per_episode_metrics_sha256"
                        ]),
                        64,
                    )
                else:
                    self.assertEqual(row["per_episode_metrics_path"], "")
                    self.assertEqual(
                        row["matched_source_per_episode_metrics_path"], ""
                    )
            per_episode_files = sorted(
                output_a.rglob("per_episode_metrics.json")
            )
            self.assertEqual(len(per_episode_files), expected_per_episode)
            metadata_by_path = {
                item["exported_path"]: item
                for item in report_a["per_episode_metrics"]
            }
            for path in per_episode_files:
                relative = path.relative_to(output_a).as_posix()
                document = json.loads(path.read_text(encoding="utf-8"))
                metadata = metadata_by_path[relative]
                order_document = json.loads(
                    fixture.setting_assets[document["setting"]][
                        "order"
                    ].read_text(encoding="utf-8")
                )
                expected_ids = [
                    item["episode_id"] for item in order_document["episodes"]
                ]
                self.assertEqual(
                    document["schema"], "navtta.vln_per_episode_metrics.v1"
                )
                self.assertEqual(document["split"], "val_seen")
                self.assertEqual(document["episode_count"], len(expected_ids))
                self.assertEqual(
                    [item["ordinal"] for item in document["episodes"]],
                    list(range(len(expected_ids))),
                )
                self.assertEqual(
                    [item["episode_id"] for item in document["episodes"]],
                    expected_ids,
                )
                self.assertTrue(all(
                    set(item["metrics"])
                    == set(MODULE.CONTINUOUS_PER_EPISODE_METRICS)
                    for item in document["episodes"]
                ))
                self.assertEqual(
                    document["source_artifact"]["sha256"],
                    metadata["source_sha256"],
                )
                self.assertEqual(MODULE.sha256(path), metadata["exported_sha256"])
            with (output_a / "source_inventory.csv").open(
                    "r", encoding="utf-8", newline="") as stream:
                inventory = list(csv.DictReader(stream))
            per_episode_inventory = [
                row for row in inventory
                if row["exported"].endswith("/per_episode_metrics.json")
            ]
            self.assertEqual(len(per_episode_inventory), expected_per_episode)
            self.assertTrue(all(
                "/stats_ep_ckpt_" in row["source"]
                and len(row["source_sha256"]) == 64
                and len(row["exported_sha256"]) == 64
                for row in per_episode_inventory
            ))
            self.assertEqual(
                (output_a / "SHA256SUMS").read_text(encoding="utf-8"),
                (output_b / "SHA256SUMS").read_text(encoding="utf-8"),
            )
            for path in output_a.rglob("*"):
                if not path.is_file():
                    continue
                lowered = {part.lower() for part in path.relative_to(output_a).parts}
                self.assertFalse(lowered.intersection(MODULE.PROHIBITED_PARTS))
                self.assertNotIn(path.suffix.lower(), MODULE.PROHIBITED_SUFFIXES)
                self.assertFalse(path.name.startswith("events.out.tfevents"))
                self.assertFalse(path.name.startswith("stats_ep_ckpt_"))

    def test_missing_method_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory)).build()
            roots = fixture.method_roots()
            roots["atena"] = Path(directory) / "does-not-exist"
            with self.assertRaisesRegex(MODULE.ExportError, "missing method batch"):
                fixture.exporter(
                    Path(directory) / "export", method_roots=roots
                ).export()

    def test_incomplete_strict_stage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory)).build()
            path = (
                fixture.search_root / "tent" / fixture.batch_id / "stages"
                / "final_controls" / "SUMMARY.json"
            )
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["complete"] = False
            summary["validated"] -= 1
            summary["errors"] = [{"run_tag": "failed-control", "exit_code": 1}]
            write_json(path, summary)
            with self.assertRaisesRegex(MODULE.ExportError, "strict stage"):
                fixture.exporter(Path(directory) / "export").export()

    def test_pre_final_controls_batch_is_exported_with_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory)).build(official=False)
            for method in MODULE.METHODS:
                shutil.rmtree(
                    fixture.search_root / method / fixture.batch_id / "stages"
                    / "final_controls"
                )
            output = Path(directory) / "export"
            report = fixture.exporter(
                output,
                allow_legacy_prefix_controls=True,
            ).export()
            self.assertFalse(report["validation"][
                "formal_matched_comparison_ready"
            ])
            self.assertTrue(report["validation"]["qualifications"])
            with (output / "selected_results.csv").open(
                    "r", encoding="utf-8", newline="") as stream:
                selected = list(csv.DictReader(stream))
            self.assertEqual(len(selected), 40)
            self.assertTrue(all(
                row["matched_source_stage"] == "controls"
                and row["matched_source_full_stream"] == "False"
                for row in selected
            ))

    def test_pre_final_controls_batch_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory)).build(official=False)
            for method in MODULE.METHODS:
                shutil.rmtree(
                    fixture.search_root / method / fixture.batch_id / "stages"
                    / "final_controls"
                )
            with self.assertRaisesRegex(MODULE.ExportError, "missing stages"):
                fixture.exporter(Path(directory) / "export").export()

    def test_official_selection_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = CampaignFixture(Path(directory)).build()
            path = (
                fixture.search_root / "tent" / fixture.batch_id
                / "FINAL_SELECTION.json"
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            document["settings"][fixture.settings[0]]["winner_run_tag"] = "wrong"
            write_json(path, document)
            with self.assertRaisesRegex(
                    MODULE.ExportError, "official and derived.*disagree"):
                fixture.exporter(Path(directory) / "export").export()


if __name__ == "__main__":
    unittest.main()
