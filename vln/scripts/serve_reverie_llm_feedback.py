#!/usr/bin/env python3
"""Verify or serve the pinned local Qwen2-VL REVERIE feedback provider.

The server binds only to a loopback address.  It implements the two request
stages defined in :mod:`navtta_vln.reverie_llm_feedback`; callers cannot submit
ground truth, evaluator outputs, distances, rewards, or oracle actions because
the request schemas reject every unknown field.
"""

from __future__ import absolute_import

import argparse
import base64
import hmac
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
import threading


VLN_ROOT = Path(__file__).resolve().parents[1]
PROVIDER_PATH = VLN_ROOT / "navtta_vln" / "reverie_llm_feedback.py"
PROVIDER_SPEC = importlib.util.spec_from_file_location(
    "_navtta_reverie_llm_feedback_service", str(PROVIDER_PATH)
)
PROVIDER = importlib.util.module_from_spec(PROVIDER_SPEC)
PROVIDER_SPEC.loader.exec_module(PROVIDER)
MODEL_ID = PROVIDER.MODEL_ID
MODEL_DTYPE = PROVIDER.MODEL_DTYPE
PROVIDER_ID = PROVIDER.PROVIDER_ID
PROVIDER_VERSION = PROVIDER.PROVIDER_VERSION
PINNED_MODEL_REVISION = PROVIDER.PINNED_MODEL_REVISION
PINNED_WEIGHTS_SHA256 = PROVIDER.PINNED_WEIGHTS_SHA256
REQUEST_SCHEMA = PROVIDER.REQUEST_SCHEMA
RESPONSE_SCHEMA = PROVIDER.RESPONSE_SCHEMA
STAGE1_PROMPT_SHA256 = PROVIDER.STAGE1_PROMPT_SHA256
STAGE1_PROMPT_TEMPLATE = PROVIDER.STAGE1_PROMPT_TEMPLATE
STAGE2_PROMPT_SHA256 = PROVIDER.STAGE2_PROMPT_SHA256
STAGE2_PROMPT_TEMPLATE = PROVIDER.STAGE2_PROMPT_TEMPLATE
_canonical_json_bytes = PROVIDER._canonical_json_bytes
_sha256_bytes = PROVIDER._sha256_bytes
discover_local_revision = PROVIDER.discover_local_revision
provider_contract = PROVIDER.provider_contract
strict_json_loads = PROVIDER.strict_json_loads
validate_loopback_url = PROVIDER.validate_loopback_url
verify_model_bundle = PROVIDER.verify_model_bundle
verify_full_bundle = PROVIDER.verify_full_bundle
read_bearer_token = PROVIDER.read_bearer_token
Qwen2VLFeedbackProvider = PROVIDER.Qwen2VLFeedbackProvider


REQUIRED_RUNTIME_PACKAGES = PROVIDER.REQUIRED_SERVICE_PACKAGES


MAX_REQUEST_BYTES = 32 * 1024 * 1024


def verify_runtime_packages():
    try:
        from importlib import metadata
    except ImportError:  # Python 3.7 fallback; service normally uses 3.9.
        import importlib_metadata as metadata
    actual = {}
    for package, expected in sorted(REQUIRED_RUNTIME_PACKAGES.items()):
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError("missing runtime package {}".format(package)) from error
        normalized = version.split("+", 1)[0]
        if normalized != expected:
            raise RuntimeError(
                "runtime package {} mismatch: expected {}, actual {}".format(
                    package, expected, version
                )
            )
        actual[package] = version
    return actual


def inspect_bundle(model_dir):
    model_dir = Path(model_dir).expanduser().resolve()
    bundle_sha256 = verify_full_bundle(model_dir)
    output = provider_contract()
    output.update({
        "model_dir": str(model_dir),
        "revision": discover_local_revision(model_dir),
        "weights_sha256": PROVIDER.PINNED_WEIGHTS_SHA256,
        "bundle_sha256": bundle_sha256,
    })
    return output


def verify_cuda_float16_runtime(device, dtype, torch_module=None):
    """Fail closed unless the provider runtime is CUDA with FP16 tensors."""
    if dtype != MODEL_DTYPE:
        raise RuntimeError(
            "qwen2_vl_2b_v1 requires runtime dtype {}".format(MODEL_DTYPE)
        )
    if torch_module is None:
        try:
            import torch as torch_module
        except Exception as error:
            raise RuntimeError("provider verification requires PyTorch") from error
    try:
        resolved = torch_module.device(device)
    except Exception as error:
        raise RuntimeError("invalid provider CUDA device: {}".format(device)) from error
    if resolved.type != "cuda":
        raise RuntimeError("qwen2_vl_2b_v1 provider requires a CUDA device")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for qwen2_vl_2b_v1 provider")
    index = (
        torch_module.cuda.current_device()
        if resolved.index is None else int(resolved.index)
    )
    count = int(torch_module.cuda.device_count())
    if index < 0 or index >= count:
        raise RuntimeError(
            "provider CUDA device index {} is outside {} visible devices"
            .format(index, count)
        )
    canonical_device = "cuda:{}".format(index)
    try:
        probe = torch_module.empty(
            (1,), device=canonical_device, dtype=torch_module.float16
        )
        if probe.device.type != "cuda" or probe.dtype != torch_module.float16:
            raise RuntimeError("CUDA float16 allocation returned the wrong tensor")
        name = str(torch_module.cuda.get_device_name(index))
        capability = list(torch_module.cuda.get_device_capability(index))
    except Exception as error:
        raise RuntimeError("CUDA float16 allocation probe failed") from error
    return {
        "runtime_device_type": "cuda",
        "runtime_device": canonical_device,
        "runtime_dtype": MODEL_DTYPE,
        "cuda_available": True,
        "cuda_device_index": index,
        "cuda_device_count": count,
        "cuda_device_name": name,
        "cuda_capability": capability,
    }


def validate_service_runtime_health(document):
    """Require live health evidence from a loaded CUDA/FP16 model."""
    if not isinstance(document, dict):
        raise RuntimeError("feedback service health is not an object")
    expected = {
        "ready": True,
        "runtime_device_type": "cuda",
        "runtime_dtype": MODEL_DTYPE,
        "cuda_available": True,
        "model_loaded": True,
        "model_eval_mode": True,
        "model_parameter_device_types": ["cuda"],
        "model_parameter_dtypes": [MODEL_DTYPE],
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RuntimeError(
                "feedback service health {} mismatch: expected {!r}, got {!r}"
                .format(key, value, document.get(key))
            )
    device = document.get("runtime_device")
    if not isinstance(device, str) or not device.startswith("cuda:"):
        raise RuntimeError("feedback service did not report a concrete CUDA device")
    return document


class Qwen2VLInference(object):
    """Single-process deterministic Qwen2-VL inference engine."""

    def __init__(self, model_dir, identity, device="cuda:0", dtype="float16"):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except Exception as error:
            raise RuntimeError(
                "Qwen2-VL serving requires torch and a transformers release "
                "that provides Qwen2VLForConditionalGeneration"
            ) from error
        self.torch = torch
        self.runtime_contract = verify_cuda_float16_runtime(
            device, dtype, torch_module=torch
        )
        self.device = torch.device(self.runtime_contract["runtime_device"])
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda, "matmul"
        ):
            torch.backends.cuda.matmul.allow_tf32 = False
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = None if dtype == "auto" else dtype_map[dtype]
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_dir), **load_kwargs
        ).to(self.device)
        self.model.eval()
        floating_parameters = [
            parameter for parameter in self.model.parameters()
            if parameter.is_floating_point()
        ]
        if not floating_parameters:
            raise RuntimeError("Qwen2-VL model exposes no floating parameters")
        parameter_devices = sorted({
            parameter.device.type for parameter in floating_parameters
        })
        parameter_dtypes = sorted({
            str(parameter.dtype).split(".")[-1]
            for parameter in floating_parameters
        })
        if parameter_devices != ["cuda"] or parameter_dtypes != [MODEL_DTYPE]:
            raise RuntimeError(
                "Qwen2-VL model must be entirely CUDA/{}; devices={}, dtypes={}"
                .format(MODEL_DTYPE, parameter_devices, parameter_dtypes)
            )
        self.runtime_contract.update({
            "model_loaded": True,
            "model_eval_mode": not bool(self.model.training),
            "model_parameter_device_types": parameter_devices,
            "model_parameter_dtypes": parameter_dtypes,
        })
        self.identity = identity
        self._lock = threading.Lock()

    def _move_inputs(self, inputs):
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _generate(self, prompt, image=None, max_new_tokens=32):
        if image is None:
            messages = [{"role": "user", "content": prompt}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], padding=True, return_tensors="pt"
            )
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
        inputs = self._move_inputs(inputs)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        with self._lock, self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
            )
        generated_only = generated[:, inputs["input_ids"].shape[-1]:]
        output = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        completion_tokens = int(generated_only.shape[-1])
        return output.strip(), prompt_tokens, completion_tokens

    def infer(self, request_document):
        common = {
            "schema", "provider_id", "request_id", "stage", "prompt_sha256"
        }
        if not isinstance(request_document, dict):
            raise ValueError("request must be a JSON object")
        stage = request_document.get("stage")
        if stage == "goal_phrase":
            expected = common | {"instruction"}
            prompt_sha256 = STAGE1_PROMPT_SHA256
        elif stage == "binary_success":
            expected = common | {
                "goal_phrase", "panorama_png_base64", "panorama_png_sha256"
            }
            prompt_sha256 = STAGE2_PROMPT_SHA256
        else:
            raise ValueError("unknown inference stage")
        if set(request_document) != expected:
            raise ValueError("request fields do not match the stage allowlist")
        if request_document["schema"] != REQUEST_SCHEMA:
            raise ValueError("request schema mismatch")
        if request_document["provider_id"] != PROVIDER_ID:
            raise ValueError("provider id mismatch")
        request_id = request_document["request_id"]
        if (
            not isinstance(request_id, str)
            or len(request_id) != 64
            or any(char not in "0123456789abcdef" for char in request_id)
        ):
            raise ValueError("request id must be a lowercase SHA256")
        if request_document["prompt_sha256"] != prompt_sha256:
            raise ValueError("prompt digest mismatch")
        if stage == "goal_phrase":
            instruction = request_document["instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be nonempty text")
            prompt = STAGE1_PROMPT_TEMPLATE.format(
                instruction=instruction.strip()
            )
            output, prompt_tokens, completion_tokens = self._generate(
                prompt, max_new_tokens=32
            )
            if (
                not output
                or len(output) > 160
                or "\n" in output
                or "\r" in output
                or any(ord(char) < 32 for char in output)
            ):
                raise ValueError("model returned an invalid goal phrase")
        else:
            goal_phrase = request_document["goal_phrase"]
            if not isinstance(goal_phrase, str) or not goal_phrase.strip():
                raise ValueError("goal phrase must be nonempty text")
            encoded = request_document["panorama_png_base64"]
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("panorama must be nonempty base64")
            try:
                image_bytes = base64.b64decode(encoded, validate=True)
                from PIL import Image

                image = Image.open(BytesIO(image_bytes))
                image.load()
            except Exception as error:
                raise ValueError("panorama is not a valid RGB image") from error
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (1920, 360)
            ):
                raise ValueError("panorama dimensions, mode, or format are invalid")
            if request_document["panorama_png_sha256"] != _sha256_bytes(
                image_bytes
            ):
                raise ValueError("panorama digest mismatch")
            prompt = STAGE2_PROMPT_TEMPLATE.format(
                goal_phrase=goal_phrase.strip()
            )
            output, prompt_tokens, completion_tokens = self._generate(
                prompt, image=image, max_new_tokens=4
            )
            if output not in ("Yes", "No"):
                raise ValueError("model output must be exactly Yes or No")
        request_sha256 = _sha256_bytes(_canonical_json_bytes(request_document))
        return {
            "schema": RESPONSE_SCHEMA,
            "provider_id": PROVIDER_ID,
            "provider_version": PROVIDER_VERSION,
            "model_id": MODEL_ID,
            "revision": self.identity["revision"],
            "weights_sha256": self.identity["weights_sha256"],
            "bundle_sha256": self.identity["bundle_sha256"],
            "auth_token_sha256": self.identity["auth_token_sha256"],
            "request_id": request_id,
            "request_sha256": request_sha256,
            "stage": stage,
            "prompt_sha256": prompt_sha256,
            "output": output,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def runtime_health(self):
        output = dict(self.runtime_contract)
        output["cuda_device"] = str(self.device)
        output["cuda_peak_memory_allocated_bytes"] = int(
            self.torch.cuda.max_memory_allocated(self.device)
        )
        output["cuda_peak_memory_reserved_bytes"] = int(
            self.torch.cuda.max_memory_reserved(self.device)
        )
        return output


def make_handler(engine, identity, bearer_token):
    from http.server import BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        server_version = "NavTTAQwen2VL/1"

        def log_message(self, fmt, *args):
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def _send(self, status, document):
            payload = _canonical_json_bytes(document)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self):
            supplied = self.headers.get("Authorization", "")
            expected = "Bearer " + bearer_token
            return hmac.compare_digest(supplied, expected)

        def do_GET(self):  # noqa: N802
            if self.path != "/v1/health":
                self._send(404, {"error": "not_found"})
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            payload = dict(identity)
            payload.update(engine.runtime_health())
            payload["ready"] = True
            self._send(200, payload)

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/infer":
                self._send(404, {"error": "not_found"})
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 2 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("truncated request")
                request_document = strict_json_loads(raw.decode("utf-8"))
                response = engine.infer(request_document)
            except Exception as error:
                self._send(422, {
                    "error": "invalid_or_unavailable_feedback",
                    "detail": "{}: {}".format(type(error).__name__, str(error)),
                })
                return
            self._send(200, response)

    return Handler


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--weights-sha256", default=PINNED_WEIGHTS_SHA256)
    parser.add_argument("--token-file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"),
        default=MODEL_DTYPE,
    )
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--smoke-service-url")
    parser.add_argument("--smoke-scan")
    parser.add_argument("--smoke-viewpoint")
    parser.add_argument("--smoke-instruction")
    parser.add_argument("--connectivity-dir")
    parser.add_argument("--scan-data-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--transcript-path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.print_contract:
        print(json.dumps(provider_contract(), indent=2, sort_keys=True))
        return 0
    smoke_values = (
        args.smoke_service_url, args.smoke_scan, args.smoke_viewpoint,
        args.smoke_instruction, args.connectivity_dir, args.scan_data_dir,
        args.cache_dir, args.transcript_path,
    )
    if any(value is not None for value in smoke_values):
        if not all(value is not None for value in smoke_values) or not args.token_file:
            raise SystemExit(
                "smoke requires --smoke-service-url, --smoke-scan, "
                "--smoke-viewpoint, --smoke-instruction, --connectivity-dir, "
                "--scan-data-dir, --cache-dir, --transcript-path, and --token-file"
            )
        provider = Qwen2VLFeedbackProvider(
            service_url=args.smoke_service_url,
            revision=PINNED_MODEL_REVISION,
            weights_sha256=PINNED_WEIGHTS_SHA256,
            bundle_sha256=PROVIDER.PINNED_BUNDLE_SHA256,
            prompt_bundle_sha256=PROVIDER.PROMPT_BUNDLE_SHA256,
            token_file=args.token_file,
            cache_dir=args.cache_dir,
            transcript_path=args.transcript_path,
            connectivity_dir=args.connectivity_dir,
            scan_data_dir=args.scan_data_dir,
            timeout_seconds=120.0,
        )
        result = provider.query({
            "episode_id": "provider-smoke",
            "instruction": args.smoke_instruction,
            "scan_id": args.smoke_scan,
            "endpoint_viewpoint_id": args.smoke_viewpoint,
            "submitted_trajectory_viewpoint_ids": [args.smoke_viewpoint],
        })
        output = {
            "result": result,
            "provider": provider.diagnostics(),
            "service": provider.service_health(),
        }
        validate_service_runtime_health(output["service"])
        print(json.dumps(output, indent=2, sort_keys=True))
        if not result.get("available"):
            raise SystemExit("provider smoke failed closed")
        return 0
    if not args.model_dir:
        raise SystemExit("--model-dir is required for inspect, verify, or serve")
    if args.dtype != MODEL_DTYPE:
        raise SystemExit(
            "qwen2_vl_2b_v1 requires --dtype {}".format(MODEL_DTYPE)
        )
    if args.inspect:
        print(json.dumps(inspect_bundle(args.model_dir), indent=2, sort_keys=True))
        return 0
    accelerator = verify_cuda_float16_runtime(args.device, args.dtype)
    identity = verify_model_bundle(
        args.model_dir, args.revision, args.weights_sha256
    )
    identity["runtime_packages"] = verify_runtime_packages()
    identity.update(accelerator)
    if args.verify_only:
        identity["model_loaded"] = False
        print(json.dumps(identity, indent=2, sort_keys=True))
        return 0
    if not args.token_file:
        raise SystemExit("--token-file is required when serving")
    bearer_token, token_sha256 = read_bearer_token(args.token_file)
    identity["auth_token_sha256"] = token_sha256
    validate_loopback_url("http://{}:{}".format(args.host, args.port))
    engine = Qwen2VLInference(
        args.model_dir,
        identity,
        device=args.device,
        dtype=args.dtype,
    )
    health = dict(identity)
    health.pop("model_dir", None)
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(engine, health, bearer_token)
    )
    print(json.dumps({
        "event": "ready",
        "url": "http://{}:{}".format(args.host, args.port),
        "identity": health,
    }, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
