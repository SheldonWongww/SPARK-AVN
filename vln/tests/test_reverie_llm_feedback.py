import hashlib
import importlib.util
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln" / "scripts"))

PROVIDER_PATH = REPO_ROOT / "vln" / "navtta_vln" / "reverie_llm_feedback.py"
PROVIDER_SPEC = importlib.util.spec_from_file_location(
    "_test_reverie_llm_feedback", str(PROVIDER_PATH)
)
PROVIDER = importlib.util.module_from_spec(PROVIDER_SPEC)
PROVIDER_SPEC.loader.exec_module(PROVIDER)

MODEL_ID = PROVIDER.MODEL_ID
PROMPT_BUNDLE_SHA256 = PROVIDER.PROMPT_BUNDLE_SHA256
PROVIDER_ID = PROVIDER.PROVIDER_ID
PROVIDER_VERSION = PROVIDER.PROVIDER_VERSION
REQUEST_SCHEMA = PROVIDER.REQUEST_SCHEMA
RESPONSE_SCHEMA = PROVIDER.RESPONSE_SCHEMA
STAGE1_PROMPT_SHA256 = PROVIDER.STAGE1_PROMPT_SHA256
STAGE2_PROMPT_SHA256 = PROVIDER.STAGE2_PROMPT_SHA256
Qwen2VLFeedbackProvider = PROVIDER.Qwen2VLFeedbackProvider
_canonical_json_bytes = PROVIDER._canonical_json_bytes
_sha256_bytes = PROVIDER._sha256_bytes
compute_weights_sha256 = PROVIDER.compute_weights_sha256
deploy_time_episode_inputs = PROVIDER.deploy_time_episode_inputs
discover_local_revision = PROVIDER.discover_local_revision
validate_loopback_url = PROVIDER.validate_loopback_url
provider_contract = PROVIDER.provider_contract
strict_json_loads = PROVIDER.strict_json_loads
read_bearer_token = PROVIDER.read_bearer_token
from tta_config_cli import translate  # noqa: E402

SERVICE_PATH = REPO_ROOT / "vln" / "scripts" / "serve_reverie_llm_feedback.py"
SERVICE_SPEC = importlib.util.spec_from_file_location(
    "_test_serve_reverie_llm_feedback", str(SERVICE_PATH)
)
SERVICE = importlib.util.module_from_spec(SERVICE_SPEC)
SERVICE_SPEC.loader.exec_module(SERVICE)


REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
WEIGHTS_SHA256 = "4faeb74ee719f9c35d7fa254d7cc2d7131e4b4ada7d0340615c66b22d5e16fc5"
BUNDLE_SHA256 = PROVIDER.PINNED_BUNDLE_SHA256
TEST_TOKEN = "0123456789abcdef0123456789abcdef"
TEST_TOKEN_SHA256 = hashlib.sha256(TEST_TOKEN.encode("ascii")).hexdigest()


class _FakeRenderer(object):
    def __init__(self):
        self.calls = []

    def render(self, scan_id, viewpoint_id):
        from io import BytesIO
        from PIL import Image

        self.calls.append((scan_id, viewpoint_id))
        buffer = BytesIO()
        Image.new("RGB", (1920, 360), color=(1, 2, 3)).save(
            buffer, format="PNG", optimize=False, compress_level=6
        )
        payload = buffer.getvalue()
        return payload, {
            "view_count": 36,
            "layout": "12_columns_x_3_elevation_rows",
            "image_format": "PNG_RGB",
            "camera_width": 640,
            "camera_height": 480,
            "tile_width": 160,
            "tile_height": 120,
            "vfov_degrees": 60.0,
            "panorama_png_sha256": hashlib.sha256(payload).hexdigest(),
        }


class _FakeTransport(object):
    def __init__(self, binary_output="Yes"):
        self.calls = []
        self.binary_output = binary_output

    def __call__(self, method, path, document=None):
        self.calls.append((method, path, document))
        if method == "GET":
            response = provider_contract()
            response.update({
                "revision": REVISION,
                "weights_sha256": WEIGHTS_SHA256,
                "bundle_sha256": BUNDLE_SHA256,
                "auth_token_sha256": TEST_TOKEN_SHA256,
                "runtime_packages": PROVIDER.REQUIRED_SERVICE_PACKAGES,
                "service_url": "http://127.0.0.1:8765",
                "ready": True,
            })
            return response, _canonical_json_bytes(response)
        output = (
            "red chair"
            if document["stage"] == "goal_phrase" else self.binary_output
        )
        response = {
            "schema": RESPONSE_SCHEMA,
            "provider_id": PROVIDER_ID,
            "provider_version": PROVIDER_VERSION,
            "model_id": MODEL_ID,
            "revision": REVISION,
            "weights_sha256": WEIGHTS_SHA256,
            "bundle_sha256": BUNDLE_SHA256,
            "auth_token_sha256": TEST_TOKEN_SHA256,
            "request_id": document["request_id"],
            "request_sha256": _sha256_bytes(
                _canonical_json_bytes(document)
            ),
            "stage": document["stage"],
            "prompt_sha256": document["prompt_sha256"],
            "output": output,
            "prompt_tokens": 12,
            "completion_tokens": 2,
        }
        return response, _canonical_json_bytes(response)


def _episode():
    return {
        "episode_id": "episode_0",
        "instruction": "Go to the red chair.",
        "scan_id": "scan",
        "endpoint_viewpoint_id": "reranked_endpoint",
        "submitted_trajectory_viewpoint_ids": [
            "start", "simulator_stop", "reranked_endpoint"
        ],
    }


class ReverieLLMFeedbackProviderTest(unittest.TestCase):
    def _provider(self, directory, transport, renderer):
        token_file = Path(directory) / "token"
        token_file.write_text(TEST_TOKEN + "\n", encoding="ascii")
        token_file.chmod(0o600)
        return Qwen2VLFeedbackProvider(
            service_url="http://127.0.0.1:8765",
            revision=REVISION,
            weights_sha256=WEIGHTS_SHA256,
            bundle_sha256=BUNDLE_SHA256,
            prompt_bundle_sha256=PROMPT_BUNDLE_SHA256,
            token_file=token_file,
            cache_dir=directory,
            transcript_path=Path(directory) / "feedback.ndjson",
            connectivity_dir=directory,
            scan_data_dir=directory,
            renderer=renderer,
            http_transport=transport,
            verify_health=True,
        )

    def test_only_loopback_http_service_is_allowed(self):
        self.assertEqual(
            validate_loopback_url("http://127.0.0.1:8765"),
            "http://127.0.0.1:8765",
        )
        self.assertEqual(
            validate_loopback_url("http://[::1]:8765"),
            "http://[::1]:8765",
        )
        for invalid in (
            "https://127.0.0.1:8765",
            "http://example.com:8765",
            "http://127.0.0.1:8765/path",
            "http://user@127.0.0.1:8765",
        ):
            with self.subTest(url=invalid), self.assertRaises(ValueError):
                validate_loopback_url(invalid)

    def test_duplicate_json_fields_are_rejected(self):
        with self.assertRaisesRegex(
            PROVIDER.FeedbackProviderError, "duplicate JSON field"
        ):
            strict_json_loads('{"output":"Yes","output":"No"}')

    def test_bearer_token_file_must_be_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text(TEST_TOKEN + "\n", encoding="ascii")
            path.chmod(0o644)
            with self.assertRaisesRegex(
                PROVIDER.FeedbackProviderError, "group/world"
            ):
                read_bearer_token(path)
            path.chmod(0o600)
            token, digest = read_bearer_token(path)
            self.assertEqual(token, TEST_TOKEN)
            self.assertEqual(digest, TEST_TOKEN_SHA256)

    def test_deploy_time_resolution_uses_reranked_endpoint_without_truth(self):
        class Environment(object):
            batch = [{
                "instr_id": "episode_0",
                "scan": "scan",
                "instruction": "Go to the red chair.",
            }]

            @property
            def gt_trajs(self):
                raise AssertionError("hidden-test branch accessed gt_trajs")

            @property
            def distance(self):
                raise AssertionError("hidden-test branch accessed distance")

        episode = deploy_time_episode_inputs(
            Environment(),
            [{
                "instr_id": "episode_0",
                "path": [["start"], ["simulator_stop", "reranked_endpoint"]],
                "pred_objid": "must-not-be-consumed",
            }],
            "nested_graph_path",
        )
        self.assertEqual(episode, _episode())

    def test_two_stage_result_is_cached_and_strictly_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = _FakeTransport()
            renderer = _FakeRenderer()
            provider = self._provider(directory, transport, renderer)
            first = provider.query(_episode())
            second = provider.query(_episode())
            self.assertTrue(first["available"])
            self.assertTrue(first["success"])
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["cache_key"], second["cache_key"])
            # One health request and exactly two model requests; cache replay
            # must never contact the model service.
            self.assertEqual(len(transport.calls), 3)
            self.assertEqual(provider.http_requests, 2)
            cache_files = list(Path(directory).glob("*/*.json"))
            self.assertEqual(len(cache_files), 1)
            cache = json.loads(cache_files[0].read_text(encoding="utf-8"))
            self.assertEqual(cache["parsed_label"], "Yes")
            self.assertNotIn("instruction", cache)
            self.assertNotIn("panorama_png_base64", cache)
            transcript = (
                Path(directory) / "feedback.ndjson"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(transcript), 2)
            first_event, second_event = map(json.loads, transcript)
            self.assertEqual(first_event["sequence"], 0)
            self.assertEqual(second_event["sequence"], 1)
            self.assertEqual(
                second_event["previous_event_sha256"],
                first_event["event_sha256"],
            )
            self.assertEqual(first_event["provider_id"], "qwen2_vl_2b_v1")
            self.assertEqual(first_event["model_id"], MODEL_ID)
            self.assertEqual(
                first_event["revision"], PROVIDER.PINNED_MODEL_REVISION
            )
            self.assertEqual(
                first_event["weights_sha256"], PROVIDER.PINNED_WEIGHTS_SHA256
            )
            self.assertEqual(
                first_event["bundle_sha256"], PROVIDER.PINNED_BUNDLE_SHA256
            )
            self.assertEqual(
                first_event["prompt_sha256_values"],
                [STAGE1_PROMPT_SHA256, STAGE2_PROMPT_SHA256],
            )
            self.assertEqual(len(first_event["latency_seconds_values"]), 2)
            self.assertIn(first_event["parsed_label"], ("Yes", "No"))
            self.assertNotIn("panorama_png_base64", transcript[0])

    def test_poisoned_cached_label_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = _FakeTransport()
            provider = self._provider(directory, transport, _FakeRenderer())
            first = provider.query(_episode())
            cache_path = next(Path(directory).glob("[0-9a-f][0-9a-f]/*.json"))
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["parsed_label"] = "No"
            cache_path.write_text(json.dumps(cached), encoding="utf-8")
            second = provider.query(_episode())
            self.assertTrue(first["available"])
            self.assertFalse(second["available"])
            self.assertIn("cache label is invalid", second["failure"])
            self.assertEqual(provider.http_requests, 2)

    def test_malformed_binary_response_fails_closed_without_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(
                directory, _FakeTransport(binary_output="Yes\nNo"),
                _FakeRenderer(),
            )
            result = provider.query(_episode())
            self.assertFalse(result["available"])
            self.assertIn("exactly Yes or No", result["failure"])
            self.assertEqual(
                list(Path(directory).glob("[0-9a-f][0-9a-f]/*.json")), []
            )
            self.assertEqual(
                len(list((Path(directory) / "failures").glob("*.json"))), 1
            )
            self.assertEqual(provider.labels, 0)
            self.assertEqual(provider.failures, 1)

    def test_weight_bundle_digest_is_filename_and_content_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.safetensors").write_bytes(b"alpha")
            (root / "b.safetensors").write_bytes(b"beta")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"one": "b.safetensors", "two": "a.safetensors"}
            }), encoding="utf-8")
            lines = "".join(
                "{}:{}\n".format(name, hashlib.sha256(data).hexdigest())
                for name, data in (
                    ("a.safetensors", b"alpha"),
                    ("b.safetensors", b"beta"),
                )
            )
            self.assertEqual(
                compute_weights_sha256(root),
                hashlib.sha256(lines.encode("utf-8")).hexdigest(),
            )

    def test_tracked_model_manifest_uses_canonical_bundle_digest(self):
        manifest = json.loads((
            REPO_ROOT / "vln" / "manifests" / "models"
            / "qwen2_vl_2b_instruct.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_id"], MODEL_ID)
        self.assertEqual(manifest["revision"], REVISION)
        lines = "".join(
            "{}:{}\n".format(item["path"], item["sha256"])
            for item in sorted(
                manifest["files"], key=lambda item: item["path"]
            )
        )
        self.assertEqual(
            hashlib.sha256(lines.encode("utf-8")).hexdigest(),
            manifest["bundle_sha256"],
        )
        self.assertEqual(len(manifest["files"]), 14)
        self.assertEqual(manifest["bundle_sha256"], BUNDLE_SHA256)
        self.assertEqual(manifest["weights_sha256"], WEIGHTS_SHA256)

    def test_revision_uses_metadata_commit_not_git_blob_etag(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = (
                Path(directory) / ".cache" / "huggingface" / "download"
                / "config.json.metadata"
            )
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                REVISION + "\n" + ("a" * 40) + "\n1720000000.0\n",
                encoding="utf-8",
            )
            self.assertEqual(discover_local_revision(directory), REVISION)

    def test_config_translation_pins_hidden_test_provider(self):
        parameters = {
            "lr": 5e-6,
            "feedback_provider": PROVIDER_ID,
            "llm_feedback_url": "http://127.0.0.1:8765",
            "llm_feedback_model_id": MODEL_ID,
            "llm_feedback_revision": REVISION,
            "llm_feedback_weights_sha256": WEIGHTS_SHA256,
            "llm_feedback_bundle_sha256": BUNDLE_SHA256,
            "llm_feedback_prompt_sha256": PROMPT_BUNDLE_SHA256,
            "llm_feedback_token_file": "/tmp/navtta-feedback-token",
            "llm_feedback_cache_dir": "/tmp/navtta-feedback-cache",
            "llm_feedback_transcript_path": "/tmp/navtta-feedback.ndjson",
            "llm_feedback_timeout_seconds": 90,
            "llm_feedback_abort_on_failure": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            parameters["llm_feedback_transcript_path"] = str(
                Path(directory) / "llm_feedback_transcript.ndjson"
            )
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({
                "schema": "navtta.vln_tta_job.v1",
                "stage": "test",
                "method": "feedtta",
                "parameters": parameters,
            }), encoding="utf-8")
            method, tokens = translate(
                "goat-reverie", str(config), str(Path(directory) / "diag.json")
            )
        self.assertEqual(method, "feedtta")
        self.assertEqual(
            tokens[tokens.index("--tta_feedback_provider") + 1], PROVIDER_ID
        )
        self.assertEqual(
            tokens[tokens.index("--tta_llm_feedback_revision") + 1], REVISION
        )
        self.assertEqual(
            tokens[tokens.index("--tta_llm_feedback_bundle_sha256") + 1],
            BUNDLE_SHA256,
        )
        self.assertEqual(
            tokens[tokens.index("--tta_llm_feedback_transcript_path") + 1],
            parameters["llm_feedback_transcript_path"],
        )
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({
                "method": "feedtta", "parameters": parameters,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "goat-reverie"):
                translate(
                    "hamt-reverie", str(config),
                    str(Path(directory) / "diag.json"),
                )
        for key, value, message in (
            ("llm_feedback_revision", "a" * 40, "revision is not code-pinned"),
            (
                "llm_feedback_weights_sha256", "b" * 64,
                "weights digest is not code-pinned",
            ),
        ):
            altered = dict(parameters)
            altered[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                with tempfile.TemporaryDirectory() as directory:
                    config = Path(directory) / "config.json"
                    config.write_text(json.dumps({
                        "method": "feedtta", "parameters": altered,
                    }), encoding="utf-8")
                    translate(
                        "goat-reverie", str(config),
                        str(Path(directory) / "diag.json"),
                    )
        altered = dict(parameters)
        altered["llm_feedback_abort_on_failure"] = False
        with tempfile.TemporaryDirectory() as directory:
            altered["llm_feedback_transcript_path"] = str(
                Path(directory) / "llm_feedback_transcript.ndjson"
            )
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({
                "method": "feedtta", "parameters": altered,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "abort_on_failure=true"):
                translate(
                    "goat-reverie", str(config),
                    str(Path(directory) / "diag.json"),
                )

    def test_metadata_queries_do_not_apply_runtime_transcript_adjacency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameters = {
                "lr": 5e-6,
                "feedback_provider": PROVIDER_ID,
                "llm_feedback_url": "http://127.0.0.1:8765",
                "llm_feedback_model_id": MODEL_ID,
                "llm_feedback_revision": REVISION,
                "llm_feedback_weights_sha256": WEIGHTS_SHA256,
                "llm_feedback_bundle_sha256": BUNDLE_SHA256,
                "llm_feedback_prompt_sha256": PROMPT_BUNDLE_SHA256,
                "llm_feedback_token_file": str(root / "token"),
                "llm_feedback_cache_dir": str(root / "cache"),
                "llm_feedback_transcript_path": str(root / "result" / "llm_feedback_transcript.ndjson"),
                "llm_feedback_abort_on_failure": True,
            }
            config = root / "config.json"
            config.write_text(json.dumps({
                "schema": "navtta.vln_tta_job.v1",
                "namespace": "tuning",
                "stage": "targeted_gap_reverie_test",
                "method": "feedtta",
                "parameters": parameters,
            }), encoding="utf-8")
            cli = REPO_ROOT / "vln" / "scripts" / "tta_config_cli.py"
            common = [
                sys.executable, str(cli), "--setting", "goat-reverie",
                "--config", str(config), "--diagnostics",
                "/tmp/navtta-unused.json",
            ]
            method = subprocess.run(
                common + ["--print-method"], check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            namespace = subprocess.run(
                common + ["--print-namespace"], check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(method.returncode, 0, method.stderr)
            self.assertEqual(method.stdout.strip(), "feedtta")
            self.assertEqual(namespace.returncode, 0, namespace.stderr)
            self.assertEqual(namespace.stdout.strip(), "tuning")

            runtime = subprocess.run(
                common, check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(runtime.returncode, 0)
            self.assertIn("current job evidence file", runtime.stderr)

    def test_service_rejects_unknown_fields_and_nonbinary_output(self):
        engine = object.__new__(SERVICE.Qwen2VLInference)
        engine.identity = {
            "revision": REVISION,
            "weights_sha256": WEIGHTS_SHA256,
            "bundle_sha256": BUNDLE_SHA256,
            "auth_token_sha256": TEST_TOKEN_SHA256,
        }
        engine._generate = lambda *args, **kwargs: ("red chair", 10, 2)
        stage1 = {
            "schema": REQUEST_SCHEMA,
            "provider_id": PROVIDER_ID,
            "request_id": "1" * 64,
            "stage": "goal_phrase",
            "prompt_sha256": STAGE1_PROMPT_SHA256,
            "instruction": "Go to the red chair.",
        }
        response = engine.infer(stage1)
        self.assertEqual(response["output"], "red chair")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            engine.infer(dict(stage1, distance_to_goal=0.0))

        engine._generate = lambda *args, **kwargs: ("Yes.", 20, 2)
        # A tiny valid PNG is sufficient because model execution is stubbed.
        from PIL import Image
        from io import BytesIO
        import base64

        buffer = BytesIO()
        Image.new("RGB", (1920, 360)).save(buffer, format="PNG")
        stage2 = {
            "schema": REQUEST_SCHEMA,
            "provider_id": PROVIDER_ID,
            "request_id": "2" * 64,
            "stage": "binary_success",
            "prompt_sha256": STAGE2_PROMPT_SHA256,
            "goal_phrase": "red chair",
            "panorama_png_base64": base64.b64encode(
                buffer.getvalue()
            ).decode("ascii"),
            "panorama_png_sha256": hashlib.sha256(
                buffer.getvalue()
            ).hexdigest(),
        }
        with self.assertRaisesRegex(ValueError, "exactly Yes or No"):
            engine.infer(stage2)

    def test_service_can_print_contract_without_model_directory(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(SERVICE.main(["--print-contract"]), 0)
        contract = json.loads(output.getvalue())
        self.assertEqual(contract["provider_id"], PROVIDER_ID)
        self.assertEqual(
            contract["prompt_bundle_sha256"], PROMPT_BUNDLE_SHA256
        )

    def test_provider_runtime_rejects_cpu_and_non_fp16_before_loading(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA device"):
            SERVICE.verify_cuda_float16_runtime(
                "cpu", "float16", torch_module=object()
            )
        with self.assertRaisesRegex(RuntimeError, "runtime dtype float16"):
            SERVICE.verify_cuda_float16_runtime(
                "cuda:0", "float32", torch_module=object()
            )

    def test_provider_runtime_records_a_concrete_cuda_fp16_probe(self):
        float16_dtype = object()

        class Device(object):
            type = "cuda"
            index = 1

        class Tensor(object):
            device = Device()
            dtype = float16_dtype

        class Cuda(object):
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def device_count():
                return 2

            @staticmethod
            def get_device_name(index):
                return "fake-gpu-{}".format(index)

            @staticmethod
            def get_device_capability(index):
                return (8, 0)

        class Torch(object):
            cuda = Cuda()
            float16 = float16_dtype

            @staticmethod
            def device(value):
                self = Device()
                self.index = int(value.rsplit(":", 1)[1])
                return self

            @staticmethod
            def empty(shape, device, dtype):
                self = Tensor()
                self.device = Torch.device(device)
                self.dtype = dtype
                return self

        evidence = SERVICE.verify_cuda_float16_runtime(
            "cuda:1", "float16", torch_module=Torch
        )
        self.assertEqual(evidence["runtime_device"], "cuda:1")
        self.assertEqual(evidence["runtime_dtype"], "float16")
        self.assertEqual(evidence["cuda_capability"], [8, 0])

    def test_live_service_health_requires_loaded_cuda_fp16_model(self):
        healthy = {
            "ready": True,
            "runtime_device_type": "cuda",
            "runtime_device": "cuda:0",
            "runtime_dtype": "float16",
            "cuda_available": True,
            "model_loaded": True,
            "model_eval_mode": True,
            "model_parameter_device_types": ["cuda"],
            "model_parameter_dtypes": ["float16"],
        }
        self.assertIs(
            SERVICE.validate_service_runtime_health(healthy), healthy
        )
        for key, invalid in (
            ("runtime_device_type", "cpu"),
            ("runtime_dtype", "float32"),
            ("model_loaded", False),
            ("model_parameter_dtypes", ["float16", "float32"]),
        ):
            document = dict(healthy)
            document[key] = invalid
            with self.subTest(key=key), self.assertRaisesRegex(
                RuntimeError, "health"
            ):
                SERVICE.validate_service_runtime_health(document)


if __name__ == "__main__":
    unittest.main()
