import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import importlib.util


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "vln/scripts/run_targeted_gap_campaign.py"
SPEC_PATH = REPO_ROOT / "vln/experiments/vln_targeted_gap_campaign_v1.json"

MODULE_SPEC = importlib.util.spec_from_file_location(
    "targeted_gap_render_preflight_runner", str(RUNNER_PATH)
)
RUNNER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(RUNNER)


class TargetedGapRenderPreflightBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec_path, cls.spec = RUNNER.load_spec(SPEC_PATH)

    @staticmethod
    def _file_record(path, role=None, size_key="size"):
        path = Path(path).resolve()
        record = {
            "path": str(path),
            size_key: path.stat().st_size,
            "sha256": RUNNER.sha256(path),
        }
        if role is not None:
            record["role"] = role
        return record

    def _synthetic_precheck(self, root):
        root = Path(root).resolve()
        provider = self.spec["hidden_test_transfer"]["provider"]

        build = root / "render-build"
        build.mkdir(parents=True)
        cmake = build / "CMakeCache.txt"
        cmake.write_text(
            "EGL_RENDERING:BOOL=ON\nOSMESA_RENDERING:BOOL=OFF\n",
            encoding="utf-8",
        )
        module = build / "MatterSim.cpython-38-x86_64-linux-gnu.so"
        module.write_bytes(b"synthetic-render-extension")

        scans = root / "scans.txt"
        connectivity = root / "scan-a_connectivity.json"
        raw_rgb = root / "vp-1_skybox_small.jpg"
        scans.write_text("scan-a\n", encoding="utf-8")
        connectivity.write_text("[]\n", encoding="utf-8")
        raw_rgb.write_bytes(b"synthetic-raw-rgb")

        panorama = root / "endpoint_panorama.png"
        panorama.write_bytes(b"synthetic-36-view-panorama")

        expected_model = {
            "model_id": provider["model_id"],
            "model_dtype": "float16",
            "revision": provider["revision"],
            "weights_sha256": provider["weights_sha256"],
            "bundle_sha256": provider["model_bundle_sha256"],
            "prompt_bundle_sha256": provider["prompt_bundle_sha256"],
            "stage1_prompt_sha256": provider["pipeline"][0]["prompt_sha256"],
            "stage2_prompt_sha256": provider["pipeline"][1]["prompt_sha256"],
        }
        model_verify = root / "model_verify.json"
        RUNNER.atomic_json(model_verify, {
            **expected_model,
            "runtime_device_type": "cuda",
            "runtime_dtype": "float16",
            "cuda_available": True,
        })

        expected_health = {
            "ready": True,
            "provider_id": "qwen2_vl_2b_v1",
            "model_id": provider["model_id"],
            "revision": provider["revision"],
            "weights_sha256": provider["weights_sha256"],
            "bundle_sha256": provider["model_bundle_sha256"],
            "prompt_bundle_sha256": provider["prompt_bundle_sha256"],
            "stage1_prompt_sha256": provider["pipeline"][0]["prompt_sha256"],
            "stage2_prompt_sha256": provider["pipeline"][1]["prompt_sha256"],
            "runtime_device_type": "cuda",
            "runtime_dtype": "float16",
            "cuda_available": True,
            "model_loaded": True,
            "model_eval_mode": True,
            "model_parameter_device_types": ["cuda"],
            "model_parameter_dtypes": ["float16"],
            "auth_token_sha256": "a" * 64,
        }
        provider_smoke = root / "provider_smoke.json"
        RUNNER.atomic_json(provider_smoke, {
            "service": expected_health,
            "result": {"available": True},
        })

        transcript = root / "provider_smoke_transcript.ndjson"
        transcript.write_text(json.dumps({
            "status": "label",
            "scan_id": "scan-a",
            "endpoint_viewpoint_id": "vp-1",
            "panorama_png_sha256": RUNNER.sha256(panorama),
        }, sort_keys=True) + "\n", encoding="utf-8")

        cmake_record = self._file_record(
            cmake, "mattersim_cmake_cache", "size_bytes"
        )
        module_record = self._file_record(
            module, "mattersim_extension", "size_bytes"
        )
        core_record = self._file_record(
            REPO_ROOT / "core/navtta_core/__init__.py",
            "navtta_core_module",
            "size_bytes",
        )
        verifier_record = self._file_record(
            REPO_ROOT / "vln/scripts/verify_reverie_render_preflight.py",
            "render_preflight_verifier",
            "size_bytes",
        )
        required_inputs = [
            self._file_record(
                scans, "connectivity_scan_list", "size_bytes"
            ),
            self._file_record(
                connectivity, "connectivity_graph", "size_bytes"
            ),
            self._file_record(raw_rgb, "raw_rgb_cubemap", "size_bytes"),
        ]
        render_evidence = root / "render_evidence.json"
        RUNNER.atomic_json(render_evidence, {
            "schema": RUNNER.RENDER_PREFLIGHT_SCHEMA,
            "status": "passed",
            "repository": {
                "root": str(REPO_ROOT),
                "git_commit": RUNNER.git_commit(),
                "tracked_tree_clean": True,
            },
            "verifier": verifier_record,
            "scan_id": "scan-a",
            "viewpoint_id": "vp-1",
            "headless_build": {
                "build_dir": str(build),
                "backend": "egl",
                "cmake_cache": cmake_record,
                "module_candidate": module_record,
            },
            "imports": {
                "navtta_core": core_record,
                "mattersim": dict(
                    module_record, role="loaded_mattersim_extension"
                ),
            },
            "required_input_files": required_inputs,
            "raw_rgb": {"file": required_inputs[-1]},
            "render": {
                "metadata": {
                    "view_count": 36,
                    "panorama_png_sha256": RUNNER.sha256(panorama),
                },
                "output_path": str(panorama),
                "output_size_bytes": panorama.stat().st_size,
                "output_sha256": RUNNER.sha256(panorama),
            },
        })

        precheck = root / "PRECHECK.json"
        artifacts = {
            "model_verify": self._file_record(model_verify),
            "render_evidence": self._file_record(render_evidence),
            "rendered_panorama": self._file_record(panorama),
            "provider_smoke": self._file_record(provider_smoke),
            "provider_transcript": self._file_record(transcript),
        }
        RUNNER.atomic_json(precheck, {
            "schema": RUNNER.LLM_PREFLIGHT_SCHEMA,
            "status": "passed",
            "created_at": "2026-08-31T00:00:00+00:00",
            "git_commit": RUNNER.git_commit(),
            "render_mattersim_build": str(build),
            "service_url": provider["runtime"]["service_url"],
            "scan_id": "scan-a",
            "viewpoint_id": "vp-1",
            "model_identity": expected_model,
            "artifacts": artifacts,
        })
        binding = RUNNER._validate_provider_preflight_document(
            self.spec, precheck, build
        )
        return {
            "binding": binding,
            "build": build,
            "model_verify": model_verify,
            "precheck": precheck,
        }

    def test_missing_precheck_environment_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {
                RUNNER.LLM_PREFLIGHT_ENV: "",
                RUNNER.RENDER_BUILD_ENV: str(Path(directory).resolve()),
            },
        ):
            with self.assertRaisesRegex(
                RUNNER.CampaignError, RUNNER.LLM_PREFLIGHT_ENV
            ):
                RUNNER.validate_provider_preflight(
                    self.spec, {"health": {}}
                )
        with mock.patch.dict(
            os.environ,
            {
                RUNNER.LLM_PREFLIGHT_ENV: "/outside/PRECHECK.json",
                RUNNER.RENDER_BUILD_ENV: "",
            },
        ):
            with self.assertRaisesRegex(
                RUNNER.CampaignError, RUNNER.RENDER_BUILD_ENV
            ):
                RUNNER.validate_provider_preflight(
                    self.spec, {"health": {}}
                )

    def test_precheck_and_nested_artifact_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._synthetic_precheck(directory)
            precheck = json.loads(
                evidence["precheck"].read_text(encoding="utf-8")
            )
            precheck["status"] = "failed"
            RUNNER.atomic_json(evidence["precheck"], precheck)
            with self.assertRaisesRegex(
                RUNNER.CampaignError, "PRECHECK status mismatch"
            ):
                RUNNER.validate_provider_preflight_binding(
                    self.spec, evidence["binding"]
                )

        with tempfile.TemporaryDirectory() as directory:
            evidence = self._synthetic_precheck(directory)
            evidence["model_verify"].write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RUNNER.CampaignError,
                "PRECHECK model_verify evidence changed",
            ):
                RUNNER.validate_provider_preflight_binding(
                    self.spec, evidence["binding"]
                )

    def test_llm_manifest_binds_precheck_as_auxiliary_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = self._synthetic_precheck(root / "preflight")
            config = root / "config.json"
            checkpoint = root / "checkpoint.bin"
            dataset = root / "dataset.json"
            order = root / "order.json"
            for path, payload in (
                (config, b"{}\n"),
                (checkpoint, b"checkpoint"),
                (dataset, b"dataset"),
                (order, b"{}\n"),
            ):
                path.write_bytes(payload)
            manifest_path = root / "formal" / "manifest.json"
            metadata = {
                "formal_manifest": str(manifest_path),
                "spec_path": str(self.spec_path),
                "spec_sha256": RUNNER.sha256(self.spec_path),
                "order_manifest_path": str(order),
                "order_manifest_sha256": RUNNER.sha256(order),
                "order_sha256": "a" * 64,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": RUNNER.sha256(checkpoint),
                "dataset_path": str(dataset),
                "dataset_sha256": RUNNER.sha256(dataset),
                "config_path": str(config),
                "config_sha256": RUNNER.sha256(config),
                "runtime_parameters": {},
                "feedback_provider": "qwen2_vl_2b_v1",
                "feedback_provider_preflight": evidence["binding"],
                "expected_benchmark": "reverie_discrete_goat",
                "model": "goat",
                "reported_method_label": "FeedTTA-LLM",
                "run_tag": "synthetic-run",
                "setting": "goat-reverie",
                "split": "test",
                "data_version": "discrete-native",
                "git_commit": RUNNER.git_commit(),
                "command": ["synthetic-runner"],
                "gpu": 3,
                "batch_id": "synthetic-batch",
                "stage": "reverie-test",
                "cell_id": "goat-reverie-feedtta",
                "candidate_id": "lr_1em5",
                "queue_id": 3,
                "gpu_slot": 3,
                "job_identity_sha256": "b" * 64,
                "supervision": "pseudo_label",
                "attempt": 1,
                "retry_count": 0,
                "retry_history_sha256": "c" * 64,
            }
            hardware = {
                "hostname": "host",
                "platform": "linux",
                "python": "3",
                "cuda_visible_devices": "3",
                "gpu_name": "GPU",
                "gpu_uuid": "GPU-1",
                "gpu_memory_mib": 100,
            }
            with mock.patch.object(
                RUNNER, "_hardware", return_value=hardware
            ):
                RUNNER._create_run_manifest(metadata)
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            auxiliary = {
                item["name"]: item
                for item in manifest["auxiliary_checkpoints"]
            }
            bound = auxiliary["feedback_provider_preflight"]
            self.assertEqual(
                Path(bound["path"]).resolve(), evidence["precheck"].resolve()
            )
            self.assertEqual(bound["size"], evidence["binding"]["size"])
            self.assertEqual(bound["sha256"], evidence["binding"]["sha256"])
            self.assertEqual(
                manifest["campaign"]["feedback_provider_preflight"],
                evidence["binding"],
            )

    def test_stage_plan_and_job_identity_bind_only_the_llm_precheck(self):
        search = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))
        first = {}
        for job in search:
            first.setdefault(job["cell_id"], job)
        frozen = {"winners": [
            {
                "cell_id": cell["cell_id"],
                "candidate_id": first[cell["cell_id"]]["candidate_id"],
                "candidate_role": first[cell["cell_id"]]["candidate_role"],
                "parameters": first[cell["cell_id"]]["parameters"],
                "frozen_config_sha256": "a" * 64,
            }
            for cell in self.spec["cells"]
        ]}
        with tempfile.TemporaryDirectory() as directory:
            evidence = self._synthetic_precheck(Path(directory) / "preflight")
            jobs = RUNNER.bind_provider_preflight(
                RUNNER.expand_frozen_jobs(
                    self.spec, (0, 1, 2, 3), frozen, "reverie-test", "batch"
                ),
                evidence["binding"],
            )
            bound = [
                job for job in jobs if "feedback_provider_preflight" in job
            ]
            self.assertEqual(len(bound), 1)
            self.assertEqual(bound[0]["feedback_provider"], "qwen2_vl_2b_v1")
            binding = {
                "batch_id": "batch", "git_commit": RUNNER.git_commit(),
                "spec_sha256": RUNNER.sha256(self.spec_path),
                "runner_sha256": RUNNER.sha256(RUNNER_PATH),
            }
            payload = RUNNER._stage_plan_payload(
                binding, "reverie-test", jobs,
                provider_preflight=evidence["binding"],
            )
            self.assertEqual(
                payload["provider_preflight"], evidence["binding"]
            )
            without = dict(bound[0])
            without.pop("feedback_provider_preflight")
            self.assertNotEqual(
                RUNNER._job_identity(self.spec_path, "batch", bound[0]),
                RUNNER._job_identity(self.spec_path, "batch", without),
            )

    def test_non_llm_worker_does_not_inherit_render_or_precheck_env(self):
        class Process(object):
            pid = 424242

            @staticmethod
            def wait():
                return 0

        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt-001"
            attempt.mkdir()
            metadata = {
                "command": ["synthetic-runner"],
                "run_tag": "synthetic-run",
                "feedback_provider": "none",
            }
            observed = {}

            def launch(*args, **kwargs):
                observed.update(kwargs)
                return Process()

            with mock.patch.dict(os.environ, {
                RUNNER.LLM_PREFLIGHT_ENV: "/private/precheck.json",
                RUNNER.RENDER_BUILD_ENV: "/private/render-build",
            }), mock.patch.object(
                RUNNER, "_assert_runtime_binding"
            ), mock.patch.object(
                RUNNER, "materialize_attempt",
                return_value=(attempt, metadata),
            ), mock.patch.object(
                RUNNER, "_create_run_manifest",
                return_value=Path(directory) / "manifest.json",
            ), mock.patch.object(
                RUNNER, "_scheduler_log"
            ), mock.patch.object(
                RUNNER.subprocess, "Popen", side_effect=launch
            ), mock.patch.object(
                RUNNER, "_process_identity",
                return_value={"schema": RUNNER.PROCESS_SCHEMA},
            ), mock.patch.object(
                RUNNER, "validate_attempt"
            ):
                self.assertTrue(RUNNER._run_attempt(
                    self.spec_path, self.spec, "batch", Path(directory),
                    {"gpu": 0, "queue_id": 0, "stage": "search"}, 1,
                ))
            child_environment = observed["env"]
            self.assertNotIn(RUNNER.LLM_PREFLIGHT_ENV, child_environment)
            self.assertNotIn(RUNNER.RENDER_BUILD_ENV, child_environment)


if __name__ == "__main__":
    unittest.main()
