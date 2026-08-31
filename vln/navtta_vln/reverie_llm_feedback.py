"""Auditable pseudo-feedback provider for the REVERIE hidden test split.

This module deliberately lives in ``vln``: rendering a Matterport panorama and
mapping a submitted REVERIE trajectory to its endpoint are task-specific.  The
provider never receives evaluator state.  It uses only the current instruction,
the submitted trajectory identifiers, and RGB rendered at the final submitted
endpoint.

The protocol is intentionally narrow and deterministic:

1. Qwen2-VL converts the instruction to one short target-goal phrase.
2. Qwen2-VL sees that phrase plus a contact sheet containing all 36 endpoint
   views and returns exactly ``Yes`` or ``No``.

The result is pseudo-feedback, not evaluator success.  Invalid responses and
transport/rendering failures are fail-closed: the caller receives no label and
must discard the episode gradient without updating the navigation policy.
"""

from __future__ import absolute_import

import base64
from contextlib import contextmanager
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


PROVIDER_ID = "qwen2_vl_2b_v1"
PROVIDER_VERSION = "navtta.reverie_feedtta_llm.qwen2_vl_2b.v1"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MODEL_DTYPE = "float16"
PINNED_MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
PINNED_WEIGHTS_SHA256 = (
    "4faeb74ee719f9c35d7fa254d7cc2d7131e4b4ada7d0340615c66b22d5e16fc5"
)
PINNED_BUNDLE_SHA256 = (
    "cf7dd27d27987b7ae71458529e6a72e3dcc3db6e91fb7298577e03f88e238a7e"
)
PINNED_BUNDLE_FILES = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "README.md": "74fbe869df708593087ae2ee66a6605d33fc2025937644721dc1360963d037f4",
    "chat_template.json": "ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96",
    "config.json": "422adefa19e62dd175961cec85bc0400344fe5bf9b22bd1182e05aaae78556e0",
    "generation_config.json": "d2864bf1edea5863d331edfff48106b586a366f5a2c41aa77731fadc53aa25d2",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model-00001-of-00002.safetensors": "994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9",
    "model-00002-of-00002.safetensors": "92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833",
    "model.safetensors.index.json": "260ab9fa1418d6d6ab79daa1d9da2c47264f3b72edb4630fc799077ac67d27c6",
    "preprocessor_config.json": "b5eaad0c2815f07631535dcc58f3c462b0d73693638ad21d19f3c50820eae1cc",
    "tokenizer_config.json": "ff5c4fd898fe8c39591eb70e5d39d2782802d4204d6ae9ba1223252f354842a0",
    "tokenizer.json": "cb63a0a23eef3d5b01063a9880a1925a65aaf4d1591d519910ee3527852950a0",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
REQUEST_SCHEMA = "navtta.reverie_llm_feedback_request.v1"
RESPONSE_SCHEMA = "navtta.reverie_llm_feedback_response.v1"
CACHE_SCHEMA = "navtta.reverie_llm_feedback_cache.v1"
PARSER_ID = "strict_yes_no_v1"
REQUIRED_SERVICE_PACKAGES = {
    "accelerate": "0.28.0",
    "pillow": "11.2.1",
    "safetensors": "0.5.3",
    "tokenizers": "0.20.3",
    "torch": "2.1.2",
    "transformers": "4.45.1",
}
BINARY_FEEDBACK_ENDPOINT = (
    "reverie_submitted_endpoint_panorama_llm_pseudo_success_every_episode"
)

STAGE1_PROMPT_TEMPLATE = """You are extracting a visual navigation goal.
Read the instruction below and return only one short noun phrase naming the
place, object, or scene that should be visible at the destination. Do not
explain, judge success, or add quotation marks.

Instruction: {instruction}
Target goal phrase:"""

STAGE2_PROMPT_TEMPLATE = """You are judging a navigation endpoint from RGB
evidence only. The image is a 12-by-3 contact sheet of all 36 views captured at
one submitted endpoint: the top row looks downward, the middle row is level,
and the bottom row looks upward; columns rotate clockwise in 30 degree steps.

Target goal: {goal_phrase}

Is the target goal visibly present at this endpoint? Reply with exactly Yes or
No and no other text."""


class FeedbackProviderError(RuntimeError):
    """A provider request cannot legally produce pseudo-feedback."""


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(document):
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_duplicate_json_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise FeedbackProviderError(
                "duplicate JSON field in feedback message: {}".format(key)
            )
        output[key] = value
    return output


def strict_json_loads(payload):
    """Decode JSON while rejecting duplicate keys at every object level."""
    return json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)


def _prompt_sha256(template):
    return _sha256_bytes(template.encode("utf-8"))


STAGE1_PROMPT_SHA256 = _prompt_sha256(STAGE1_PROMPT_TEMPLATE)
STAGE2_PROMPT_SHA256 = _prompt_sha256(STAGE2_PROMPT_TEMPLATE)
PROMPT_BUNDLE_SHA256 = _sha256_bytes(_canonical_json_bytes({
    "provider_id": PROVIDER_ID,
    "provider_version": PROVIDER_VERSION,
    "stage1_prompt_sha256": STAGE1_PROMPT_SHA256,
    "stage2_prompt_sha256": STAGE2_PROMPT_SHA256,
    "parser_id": PARSER_ID,
}))


def provider_contract():
    """Return the code-pinned part of the provider identity."""
    return {
        "provider_id": PROVIDER_ID,
        "provider_version": PROVIDER_VERSION,
        "model_id": MODEL_ID,
        "model_dtype": MODEL_DTYPE,
        "pinned_revision": PINNED_MODEL_REVISION,
        "pinned_weights_sha256": PINNED_WEIGHTS_SHA256,
        "pinned_bundle_sha256": PINNED_BUNDLE_SHA256,
        "request_schema": REQUEST_SCHEMA,
        "response_schema": RESPONSE_SCHEMA,
        "cache_schema": CACHE_SCHEMA,
        "parser_id": PARSER_ID,
        "required_runtime_packages": REQUIRED_SERVICE_PACKAGES,
        "stage1_prompt_sha256": STAGE1_PROMPT_SHA256,
        "stage2_prompt_sha256": STAGE2_PROMPT_SHA256,
        "prompt_bundle_sha256": PROMPT_BUNDLE_SHA256,
        "decoding": {
            "do_sample": False,
            "temperature": 0.0,
            "stage1_max_new_tokens": 32,
            "stage2_max_new_tokens": 4,
        },
        "panorama": {
            "view_count": 36,
            "camera_width": 640,
            "camera_height": 480,
            "vfov_degrees": 60.0,
            "tile_width": 160,
            "tile_height": 120,
            "layout": "12_columns_x_3_elevation_rows",
            "image_format": "PNG_RGB",
        },
    }


def _validate_hex_digest(value, name, lengths=(64,)):
    value = str(value or "").lower()
    if len(value) not in lengths or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("{} must be a lowercase hexadecimal digest".format(name))
    return value


def compute_weights_sha256(model_dir):
    """Hash the exact safetensor shards named by the model index.

    The aggregate is stable across download locations.  It is SHA256 over
    lexicographically sorted UTF-8 ``<filename>:<file_sha256>\n`` lines, as
    recorded in the tracked model provenance manifest.
    """
    model_dir = Path(model_dir).expanduser().resolve()
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FeedbackProviderError(
            "missing model.safetensors.index.json in {}".format(model_dir)
        )
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise FeedbackProviderError("invalid safetensors weight_map")
    filenames = sorted(set(weight_map.values()))
    if not filenames or not all(isinstance(name, str) and name for name in filenames):
        raise FeedbackProviderError("invalid safetensors shard names")
    aggregate_lines = []
    for filename in filenames:
        path = (model_dir / filename).resolve()
        try:
            path.relative_to(model_dir)
        except ValueError as error:
            raise FeedbackProviderError("weight shard escapes model directory") from error
        if not path.is_file():
            raise FeedbackProviderError("missing weight shard: {}".format(path))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        aggregate_lines.append("{}:{}\n".format(filename, digest.hexdigest()))
    return _sha256_bytes("".join(aggregate_lines).encode("utf-8"))


def verify_full_bundle(model_dir):
    """Hash and verify every one of the 14 files in the pinned snapshot."""
    model_dir = Path(model_dir).expanduser().resolve()
    incomplete = list((model_dir / ".cache").rglob("*.incomplete"))
    if incomplete:
        raise FeedbackProviderError(
            "model download still has incomplete files: {}".format(
                ", ".join(str(path) for path in incomplete[:3])
            )
        )
    actual_files = {}
    for filename, expected in sorted(PINNED_BUNDLE_FILES.items()):
        path = model_dir / filename
        if not path.is_file():
            raise FeedbackProviderError("missing model file: {}".format(path))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise FeedbackProviderError(
                "model file digest mismatch for {}: expected {}, actual {}"
                .format(filename, expected, actual)
            )
        actual_files[filename] = actual
    aggregate = _sha256_bytes("".join(
        "{}:{}\n".format(filename, actual_files[filename])
        for filename in sorted(actual_files)
    ).encode("utf-8"))
    if aggregate != PINNED_BUNDLE_SHA256:
        raise FeedbackProviderError("full model bundle digest mismatch")
    return aggregate


def read_bearer_token(token_file):
    """Read a private local bearer token without exposing it in diagnostics."""
    path = Path(token_file).expanduser().resolve()
    if not path.is_file():
        raise FeedbackProviderError("missing feedback bearer token file")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise FeedbackProviderError(
            "feedback bearer token file must not be group/world accessible"
        )
    raw = path.read_bytes()
    token = raw.strip()
    if len(token) < 32 or len(token) > 4096 or any(byte < 33 or byte > 126 for byte in token):
        raise FeedbackProviderError("feedback bearer token is invalid")
    return token.decode("ascii"), _sha256_bytes(token)


def discover_local_revision(model_dir):
    """Read the unique Hugging Face commit recorded by ``--local-dir`` cache."""
    metadata_root = Path(model_dir).expanduser().resolve() / ".cache" / "huggingface"
    revisions = set()
    if metadata_root.is_dir():
        for path in metadata_root.rglob("*.metadata"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = text.splitlines()
            if not lines:
                continue
            # huggingface_hub local-dir metadata is commit, etag, timestamp.
            # The etag can itself be a 40-hex Git blob, so only the first line
            # is a repository revision.
            candidate = lines[0].strip().lower()
            if re.fullmatch(r"[0-9a-f]{40}", candidate):
                revisions.add(candidate)
    if len(revisions) != 1:
        raise FeedbackProviderError(
            "expected one pinned Hugging Face revision in local-dir metadata; "
            "found {}".format(sorted(revisions))
        )
    return next(iter(revisions))


def verify_model_bundle(
    model_dir, revision, weights_sha256, bundle_sha256=PINNED_BUNDLE_SHA256
):
    """Verify the local model identity without loading it into GPU memory."""
    model_dir = Path(model_dir).expanduser().resolve()
    required = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FeedbackProviderError(
            "incomplete Qwen2-VL model directory; missing {}".format(
                ", ".join(missing)
            )
        )
    revision = _validate_hex_digest(revision, "model revision", lengths=(40,))
    if revision != PINNED_MODEL_REVISION:
        raise FeedbackProviderError(
            "model revision is not the qwen2_vl_2b_v1 pinned revision"
        )
    local_revision = discover_local_revision(model_dir)
    if revision != local_revision:
        raise FeedbackProviderError(
            "model revision mismatch: expected {}, local metadata {}".format(
                revision, local_revision
            )
        )
    expected = _validate_hex_digest(weights_sha256, "weights_sha256")
    if expected != PINNED_WEIGHTS_SHA256:
        raise FeedbackProviderError(
            "weights_sha256 is not the qwen2_vl_2b_v1 pinned digest"
        )
    expected_bundle = _validate_hex_digest(bundle_sha256, "bundle_sha256")
    if expected_bundle != PINNED_BUNDLE_SHA256:
        raise FeedbackProviderError("model bundle digest is not pinned")
    actual_bundle = verify_full_bundle(model_dir)
    # Full-bundle verification checked both shard digests individually, so the
    # pinned weight-only aggregate is now established without rereading 4.4 GB.
    actual = PINNED_WEIGHTS_SHA256
    output = provider_contract()
    output.update({
        "model_dir": str(model_dir),
        "revision": revision,
        "weights_sha256": actual,
        "bundle_sha256": actual_bundle,
    })
    return output


def validate_loopback_url(url):
    """Reject every non-HTTP or non-loopback feedback service URL."""
    parsed = urllib_parse.urlparse(str(url))
    if parsed.scheme != "http" or parsed.username or parsed.password:
        raise ValueError("feedback URL must be unauthenticated http on loopback")
    if parsed.path.rstrip("/") not in ("", "/v1") or parsed.query or parsed.fragment:
        raise ValueError("feedback URL must contain only a loopback origin")
    host = parsed.hostname
    if host == "localhost":
        pass
    else:
        try:
            if host is None or not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError as error:
            raise ValueError("feedback URL host must be loopback") from error
    if parsed.port is None:
        raise ValueError("feedback URL must include an explicit port")
    return "{}://{}:{}".format(parsed.scheme, "[{}]".format(host) if ":" in host else host, parsed.port)


def _trajectory_viewpoints(item, path_format):
    path = item.get("path")
    if not isinstance(path, (list, tuple)) or not path:
        raise FeedbackProviderError("submitted trajectory has no path")
    if path_format == "viewpoint_tuples":
        viewpoints = []
        for step in path:
            if not isinstance(step, (list, tuple)) or not step:
                raise FeedbackProviderError("invalid viewpoint-tuple trajectory")
            viewpoints.append(step[0])
    elif path_format == "nested_graph_path":
        viewpoints = []

        def visit(value):
            if isinstance(value, str):
                viewpoints.append(value)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    visit(child)
            else:
                raise FeedbackProviderError("invalid nested graph trajectory")

        visit(path)
    else:
        raise FeedbackProviderError(
            "unknown REVERIE trajectory format: {}".format(path_format)
        )
    if not viewpoints or not all(isinstance(value, str) and value for value in viewpoints):
        raise FeedbackProviderError("submitted trajectory has no valid viewpoint")
    return viewpoints


def deploy_time_episode_inputs(environment, trajectories, path_format):
    """Resolve only deploy-time fields for one submitted REVERIE episode.

    Deliberately use the active ``batch`` record rather than any evaluator map.
    The caller supplies a finalized trajectory; object predictions are ignored.
    """
    if not isinstance(trajectories, (list, tuple)) or len(trajectories) != 1:
        raise FeedbackProviderError("FeedTTA-LLM requires batch size one")
    item = trajectories[0]
    if not isinstance(item, dict):
        raise FeedbackProviderError("submitted trajectory must be an object")
    episode_id = item.get("instr_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise FeedbackProviderError("submitted trajectory has no instruction id")
    batch = getattr(environment, "batch", None)
    if not isinstance(batch, (list, tuple)):
        raise FeedbackProviderError("environment does not expose its active batch")
    matches = [
        record for record in batch
        if isinstance(record, dict) and record.get("instr_id") == episode_id
    ]
    if len(matches) != 1:
        raise FeedbackProviderError(
            "active batch must contain exactly one matching instruction"
        )
    record = matches[0]
    instruction = record.get("instruction")
    scan_id = record.get("scan")
    if not isinstance(instruction, str) or not instruction.strip():
        raise FeedbackProviderError("active episode has no instruction text")
    if not isinstance(scan_id, str) or not scan_id:
        raise FeedbackProviderError("active episode has no scan id")
    viewpoints = _trajectory_viewpoints(item, path_format)
    return {
        "episode_id": episode_id,
        "instruction": instruction.strip(),
        "scan_id": scan_id,
        "endpoint_viewpoint_id": viewpoints[-1],
        "submitted_trajectory_viewpoint_ids": viewpoints,
    }


class MatterSimPanoramaRenderer(object):
    """Render one deterministic 36-view RGB endpoint contact sheet."""

    def __init__(
        self,
        connectivity_dir,
        scan_data_dir,
        camera_width=640,
        camera_height=480,
        tile_width=160,
        tile_height=120,
        vfov_degrees=60.0,
    ):
        self.connectivity_dir = str(Path(connectivity_dir).expanduser().resolve())
        self.scan_data_dir = str(Path(scan_data_dir).expanduser().resolve())
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.tile_width = int(tile_width)
        self.tile_height = int(tile_height)
        self.vfov_degrees = float(vfov_degrees)
        if min(
            self.camera_width, self.camera_height,
            self.tile_width, self.tile_height,
        ) <= 0:
            raise ValueError("panorama dimensions must be positive")
        if not Path(self.connectivity_dir).is_dir():
            raise FeedbackProviderError("missing connectivity directory")
        if not Path(self.scan_data_dir).is_dir():
            raise FeedbackProviderError("missing Matterport scan directory")
        self._simulator = None

    def _build_simulator(self):
        try:
            import MatterSim
        except Exception as error:
            raise FeedbackProviderError("MatterSim import failed") from error
        simulator = MatterSim.Simulator()
        simulator.setNavGraphPath(self.connectivity_dir)
        simulator.setDatasetPath(self.scan_data_dir)
        simulator.setRenderingEnabled(True)
        simulator.setDepthEnabled(False)
        simulator.setPreloadingEnabled(False)
        simulator.setCameraResolution(self.camera_width, self.camera_height)
        simulator.setCameraVFOV(math.radians(self.vfov_degrees))
        simulator.setDiscretizedViewingAngles(True)
        simulator.setBatchSize(1)
        simulator.initialize()
        return simulator

    def render(self, scan_id, viewpoint_id):
        try:
            import numpy as np
            from PIL import Image
        except Exception as error:
            raise FeedbackProviderError(
                "endpoint rendering requires NumPy and Pillow"
            ) from error
        if self._simulator is None:
            self._simulator = self._build_simulator()
        views = []
        for index in range(36):
            if index == 0:
                self._simulator.newEpisode(
                    [scan_id], [viewpoint_id], [0], [math.radians(-30)]
                )
            elif index % 12 == 0:
                self._simulator.makeAction([0], [1.0], [1.0])
            else:
                self._simulator.makeAction([0], [1.0], [0])
            state = self._simulator.getState()[0]
            if state.viewIndex != index:
                raise FeedbackProviderError("MatterSim returned a noncanonical view")
            # MatterSim exposes BGR; Qwen/Pillow expects RGB.
            rgb = np.array(state.rgb, copy=True)[:, :, ::-1]
            image = Image.fromarray(rgb)
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            views.append(image.resize((self.tile_width, self.tile_height), resampling))
        sheet = Image.new(
            "RGB", (12 * self.tile_width, 3 * self.tile_height), (0, 0, 0)
        )
        for index, view in enumerate(views):
            sheet.paste(
                view,
                ((index % 12) * self.tile_width, (index // 12) * self.tile_height),
            )
        from io import BytesIO

        output = BytesIO()
        sheet.save(output, format="PNG", optimize=False, compress_level=6)
        payload = output.getvalue()
        return payload, {
            "view_count": 36,
            "layout": "12_columns_x_3_elevation_rows",
            "image_format": "PNG_RGB",
            "camera_width": self.camera_width,
            "camera_height": self.camera_height,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "vfov_degrees": self.vfov_degrees,
            "panorama_png_sha256": _sha256_bytes(payload),
        }


class Qwen2VLFeedbackProvider(object):
    """Strict loopback HTTP client plus content-addressed result cache."""

    def __init__(
        self,
        service_url,
        revision,
        weights_sha256,
        bundle_sha256,
        prompt_bundle_sha256,
        token_file,
        cache_dir,
        transcript_path,
        connectivity_dir,
        scan_data_dir,
        timeout_seconds=120.0,
        renderer=None,
        http_transport=None,
        verify_health=True,
        max_labels=6292,
        max_requests=12584,
        abort_on_failure=True,
    ):
        self.service_url = validate_loopback_url(service_url)
        self.revision = _validate_hex_digest(
            revision, "model revision", lengths=(40,)
        )
        self.weights_sha256 = _validate_hex_digest(
            weights_sha256, "weights_sha256"
        )
        self.bundle_sha256 = _validate_hex_digest(
            bundle_sha256, "bundle_sha256"
        )
        if self.revision != PINNED_MODEL_REVISION:
            raise FeedbackProviderError("feedback model revision is not pinned")
        if self.weights_sha256 != PINNED_WEIGHTS_SHA256:
            raise FeedbackProviderError("feedback model weights are not pinned")
        if self.bundle_sha256 != PINNED_BUNDLE_SHA256:
            raise FeedbackProviderError("feedback model bundle is not pinned")
        self.bearer_token, self.auth_token_sha256 = read_bearer_token(token_file)
        self.prompt_bundle_sha256 = _validate_hex_digest(
            prompt_bundle_sha256, "prompt_bundle_sha256"
        )
        if self.prompt_bundle_sha256 != PROMPT_BUNDLE_SHA256:
            raise FeedbackProviderError(
                "prompt bundle mismatch: expected code-pinned {}".format(
                    PROMPT_BUNDLE_SHA256
                )
            )
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = Path(transcript_path).expanduser().resolve()
        if not self.transcript_path.is_absolute():
            raise ValueError("feedback transcript path must be absolute")
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        if self.transcript_path.exists() and self.transcript_path.stat().st_size:
            raise FeedbackProviderError(
                "feedback transcript must be new or empty for this run"
            )
        self.abort_on_failure = bool(abort_on_failure)
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("feedback timeout must be finite and positive")
        self.max_labels = int(max_labels)
        self.max_requests = int(max_requests)
        if self.max_labels <= 0 or self.max_requests != 2 * self.max_labels:
            raise ValueError("provider budget must allow two requests per label")
        self.renderer = renderer or MatterSimPanoramaRenderer(
            connectivity_dir, scan_data_dir
        )
        self._http_transport = http_transport or self._urllib_transport
        self.query_episodes = 0
        self.http_requests = 0
        self.labels = 0
        self.positive_labels = 0
        self.negative_labels = 0
        self.failures = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_record_sha256 = None
        self.last_cache_key = None
        self.last_failure = None
        self.transcript_events = 0
        self.transcript_sha256 = "0" * 64
        self.service_runtime_packages = None
        self.last_health = None
        if verify_health:
            self._verify_health()

    def reset(self):
        """Reset stream counters while retaining the immutable disk cache."""
        self.query_episodes = 0
        self.http_requests = 0
        self.labels = 0
        self.positive_labels = 0
        self.negative_labels = 0
        self.failures = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_record_sha256 = None
        self.last_cache_key = None
        self.last_failure = None

    @property
    def identity(self):
        result = provider_contract()
        result.update({
            "revision": self.revision,
            "weights_sha256": self.weights_sha256,
            "bundle_sha256": self.bundle_sha256,
            "auth_token_sha256": self.auth_token_sha256,
            "service_url": self.service_url,
        })
        if self.service_runtime_packages is not None:
            result["runtime_packages"] = self.service_runtime_packages
        return result

    def _urllib_transport(self, method, path, document=None):
        data = None if document is None else _canonical_json_bytes(document)
        request = urllib_request.Request(
            self.service_url + path,
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.bearer_token,
            },
        )
        try:
            # Ignore HTTP(S)_PROXY: RGB evidence must never leave loopback.
            opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
            with opener.open(request, timeout=self.timeout_seconds) as response:
                payload = response.read(1024 * 1024 + 1)
        except (OSError, urllib_error.URLError, urllib_error.HTTPError) as error:
            raise FeedbackProviderError("feedback service request failed") from error
        if len(payload) > 1024 * 1024:
            raise FeedbackProviderError("feedback service response is too large")
        try:
            document = strict_json_loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeedbackProviderError("feedback service returned malformed JSON") from error
        return document, payload

    def _verify_health(self):
        document, _ = self._http_transport("GET", "/v1/health")
        expected = self.identity
        for key in (
            "provider_id", "provider_version", "model_id", "revision",
            "weights_sha256", "request_schema", "response_schema",
            "parser_id", "stage1_prompt_sha256", "stage2_prompt_sha256",
            "prompt_bundle_sha256", "pinned_revision",
            "pinned_weights_sha256", "pinned_bundle_sha256", "model_dtype",
            "decoding", "panorama", "required_runtime_packages",
        ):
            if document.get(key) != expected.get(key):
                raise FeedbackProviderError(
                    "feedback service identity mismatch for {}".format(key)
                )
        if document.get("ready") is not True:
            raise FeedbackProviderError("feedback service is not ready")
        if document.get("bundle_sha256") != self.bundle_sha256:
            raise FeedbackProviderError("feedback service bundle digest mismatch")
        if document.get("auth_token_sha256") != self.auth_token_sha256:
            raise FeedbackProviderError("feedback service authentication mismatch")
        runtime_packages = document.get("runtime_packages")
        if not isinstance(runtime_packages, dict) or not runtime_packages:
            raise FeedbackProviderError("feedback service runtime is unpinned")
        normalized = {
            key: str(value).split("+", 1)[0]
            for key, value in runtime_packages.items()
        }
        if normalized != REQUIRED_SERVICE_PACKAGES:
            raise FeedbackProviderError(
                "feedback service runtime package versions mismatch"
            )
        self.service_runtime_packages = runtime_packages
        self.last_health = document

    def service_health(self):
        self._verify_health()
        return self.last_health

    def _cache_path(self, cache_key):
        return self.cache_dir / cache_key[:2] / (cache_key + ".json")

    @staticmethod
    def _validate_panorama(png, panorama):
        try:
            from io import BytesIO
            from PIL import Image

            image = Image.open(BytesIO(png))
            image.load()
        except Exception as error:
            raise FeedbackProviderError("renderer did not return a valid PNG") from error
        expected = {
            "view_count": 36,
            "layout": "12_columns_x_3_elevation_rows",
            "image_format": "PNG_RGB",
            "camera_width": 640,
            "camera_height": 480,
            "tile_width": 160,
            "tile_height": 120,
            "vfov_degrees": 60.0,
            "panorama_png_sha256": _sha256_bytes(png),
        }
        if image.format != "PNG" or image.mode != "RGB" or image.size != (1920, 360):
            raise FeedbackProviderError("endpoint panorama encoding is noncanonical")
        if panorama != expected:
            raise FeedbackProviderError("endpoint panorama metadata is noncanonical")

    def _validate_cache_record(
        self, record, cache_key, input_sha256, episode, png, panorama,
        trajectory_sha256,
    ):
        if not isinstance(record, dict):
            raise FeedbackProviderError("feedback cache record is not an object")
        required = {
            "schema", "provider", "cache_key", "input_sha256", "episode_id",
            "scan_id", "endpoint_viewpoint_id", "trajectory_sha256",
            "panorama", "stage1", "stage2", "parsed_label", "created_at_unix",
            "token_count", "cost",
        }
        if set(record) != required:
            raise FeedbackProviderError("feedback cache record fields are invalid")
        if record["schema"] != CACHE_SCHEMA:
            raise FeedbackProviderError("feedback cache schema mismatch")
        if record["provider"] != self.identity:
            raise FeedbackProviderError("feedback cache provider identity mismatch")
        if record["cache_key"] != cache_key or record["input_sha256"] != input_sha256:
            raise FeedbackProviderError("feedback cache input identity mismatch")
        if (
            record["episode_id"] != episode["episode_id"]
            or record["scan_id"] != episode["scan_id"]
            or record["endpoint_viewpoint_id"] != episode["endpoint_viewpoint_id"]
            or record["trajectory_sha256"] != trajectory_sha256
            or record["panorama"] != panorama
        ):
            raise FeedbackProviderError("feedback cache episode binding mismatch")
        if (
            record["parsed_label"] not in ("Yes", "No")
            or record["parsed_label"] != record["stage2"].get("output")
        ):
            raise FeedbackProviderError("feedback cache label is invalid")
        for stage_name in ("stage1", "stage2"):
            stage = record[stage_name]
            if not isinstance(stage, dict) or set(stage) != {
                "request_sha256", "response_sha256", "output",
                "latency_seconds", "prompt_tokens", "completion_tokens",
            }:
                raise FeedbackProviderError("feedback cache stage record is invalid")
            for digest_key in ("request_sha256", "response_sha256"):
                _validate_hex_digest(stage[digest_key], digest_key)
        goal_phrase = self._parse_goal_phrase(record["stage1"]["output"])
        stage1_request = {
            "schema": REQUEST_SCHEMA,
            "provider_id": PROVIDER_ID,
            "request_id": _sha256_bytes((cache_key + "\0stage1").encode("ascii")),
            "stage": "goal_phrase",
            "prompt_sha256": STAGE1_PROMPT_SHA256,
            "instruction": episode["instruction"],
        }
        stage2_request = {
            "schema": REQUEST_SCHEMA,
            "provider_id": PROVIDER_ID,
            "request_id": _sha256_bytes((cache_key + "\0stage2").encode("ascii")),
            "stage": "binary_success",
            "prompt_sha256": STAGE2_PROMPT_SHA256,
            "goal_phrase": goal_phrase,
            "panorama_png_base64": base64.b64encode(png).decode("ascii"),
            "panorama_png_sha256": panorama["panorama_png_sha256"],
        }
        expected_requests = (stage1_request, stage2_request)
        for stage_name, request_document in zip(
            ("stage1", "stage2"), expected_requests
        ):
            expected_digest = _sha256_bytes(
                _canonical_json_bytes(request_document)
            )
            if record[stage_name]["request_sha256"] != expected_digest:
                raise FeedbackProviderError(
                    "feedback cache request binding mismatch"
                )
        expected_token_count = {
            "prompt": (
                record["stage1"]["prompt_tokens"]
                + record["stage2"]["prompt_tokens"]
            ),
            "completion": (
                record["stage1"]["completion_tokens"]
                + record["stage2"]["completion_tokens"]
            ),
        }
        if record["token_count"] != expected_token_count or record["cost"] != 0.0:
            raise FeedbackProviderError("feedback cache accounting is invalid")
        return record

    def _append_transcript(self, event):
        """Append one ordered, hash-chained audit event without raw images."""
        try:
            import fcntl
        except ImportError as error:
            raise FeedbackProviderError(
                "feedback transcript requires POSIX file locking"
            ) from error
        body = dict(event)
        body.update({
            "schema": "navtta.reverie_llm_feedback_transcript.v1",
            "sequence": self.transcript_events,
            "previous_event_sha256": self.transcript_sha256,
        })
        event_sha256 = _sha256_bytes(_canonical_json_bytes(body))
        body["event_sha256"] = event_sha256
        encoded = _canonical_json_bytes(body) + b"\n"
        if len(encoded) > 64 * 1024:
            raise FeedbackProviderError("feedback transcript event is too large")
        with self.transcript_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        self.transcript_events += 1
        self.transcript_sha256 = event_sha256

    def _read_cache(
        self, path, cache_key, input_sha256, episode, png, panorama,
        trajectory_sha256,
    ):
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = strict_json_loads(handle.read())
        except (OSError, json.JSONDecodeError) as error:
            raise FeedbackProviderError("feedback cache record is unreadable") from error
        return self._validate_cache_record(
            record, cache_key, input_sha256, episode, png, panorama,
            trajectory_sha256,
        )

    @staticmethod
    def _atomic_write(path, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    document, handle, indent=2, sort_keys=True,
                    ensure_ascii=False, allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, str(path))
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    @contextmanager
    def _cache_lock(path):
        """Serialize cache creation so the first valid response is immutable."""
        try:
            import fcntl
        except ImportError as error:
            raise FeedbackProviderError(
                "atomic feedback caching requires POSIX file locking"
            ) from error
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _request(self, stage, request_id, payload):
        if self.http_requests >= self.max_requests:
            raise FeedbackProviderError("feedback provider request budget exhausted")
        prompt_digest = (
            STAGE1_PROMPT_SHA256 if stage == "goal_phrase" else STAGE2_PROMPT_SHA256
        )
        request_document = {
            "schema": REQUEST_SCHEMA,
            "provider_id": PROVIDER_ID,
            "request_id": request_id,
            "stage": stage,
            "prompt_sha256": prompt_digest,
        }
        request_document.update(payload)
        request_sha256 = _sha256_bytes(_canonical_json_bytes(request_document))
        started = time.monotonic()
        self.http_requests += 1
        response, raw_response = self._http_transport(
            "POST", "/v1/infer", request_document
        )
        latency = time.monotonic() - started
        expected_fields = {
            "schema", "provider_id", "provider_version", "model_id",
            "revision", "weights_sha256", "bundle_sha256",
            "auth_token_sha256", "request_id", "request_sha256", "stage",
            "prompt_sha256", "output", "prompt_tokens",
            "completion_tokens",
        }
        if not isinstance(response, dict) or set(response) != expected_fields:
            raise FeedbackProviderError("feedback response fields are invalid")
        expected_identity = self.identity
        for key in (
            "provider_id", "provider_version", "model_id", "revision",
            "weights_sha256",
            "bundle_sha256", "auth_token_sha256",
        ):
            if response.get(key) != expected_identity[key]:
                raise FeedbackProviderError("feedback response identity mismatch")
        if (
            response["schema"] != RESPONSE_SCHEMA
            or response["request_id"] != request_id
            or response["request_sha256"] != request_sha256
            or response["stage"] != stage
            or response["prompt_sha256"] != prompt_digest
        ):
            raise FeedbackProviderError("feedback response request binding mismatch")
        if type(response["prompt_tokens"]) is not int or response["prompt_tokens"] < 0:
            raise FeedbackProviderError("invalid prompt token count")
        if type(response["completion_tokens"]) is not int or response["completion_tokens"] < 1:
            raise FeedbackProviderError("invalid completion token count")
        self.prompt_tokens += response["prompt_tokens"]
        self.completion_tokens += response["completion_tokens"]
        output = response["output"]
        if not isinstance(output, str):
            raise FeedbackProviderError("feedback output is not text")
        return output, {
            "request_sha256": request_sha256,
            "response_sha256": _sha256_bytes(raw_response),
            "output": output,
            "latency_seconds": latency,
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
        }

    @staticmethod
    def _parse_goal_phrase(value):
        if (
            not isinstance(value, str)
            or value != value.strip()
            or not value
            or len(value) > 160
            or "\n" in value
            or "\r" in value
            or any(ord(char) < 32 for char in value)
        ):
            raise FeedbackProviderError("stage-one goal phrase is malformed")
        return value

    @staticmethod
    def _parse_binary(value):
        if value not in ("Yes", "No"):
            raise FeedbackProviderError(
                "stage-two output must be exactly Yes or No"
            )
        return value

    def query(self, episode):
        """Return a success dict plus audit record, or a fail-closed result."""
        self.query_episodes += 1
        self.last_failure = None
        if self.query_episodes > self.max_labels:
            error = FeedbackProviderError("feedback provider label budget exhausted")
            return self._failed(error, episode=episode)
        try:
            png, panorama = self.renderer.render(
                episode["scan_id"], episode["endpoint_viewpoint_id"]
            )
            self._validate_panorama(png, panorama)
            trajectory_sha256 = _sha256_bytes(_canonical_json_bytes(
                episode["submitted_trajectory_viewpoint_ids"]
            ))
            input_document = {
                "provider": self.identity,
                "episode_id": episode["episode_id"],
                "instruction": episode["instruction"],
                "scan_id": episode["scan_id"],
                "endpoint_viewpoint_id": episode["endpoint_viewpoint_id"],
                "trajectory_sha256": trajectory_sha256,
                "panorama_png_sha256": panorama["panorama_png_sha256"],
            }
            input_sha256 = _sha256_bytes(_canonical_json_bytes(input_document))
            cache_key = input_sha256
            cache_path = self._cache_path(cache_key)
            with self._cache_lock(cache_path):
                cached = self._read_cache(
                    cache_path, cache_key, input_sha256, episode, png,
                    panorama, trajectory_sha256,
                )
                if cached is not None:
                    self.cache_hits += 1
                    return self._succeeded(cached, cache_hit=True)
                self.cache_misses += 1
                stage1_id = _sha256_bytes(
                    (cache_key + "\0stage1").encode("ascii")
                )
                stage1_output, stage1 = self._request(
                    "goal_phrase", stage1_id,
                    {"instruction": episode["instruction"]},
                )
                goal_phrase = self._parse_goal_phrase(stage1_output)
                stage2_id = _sha256_bytes(
                    (cache_key + "\0stage2").encode("ascii")
                )
                stage2_output, stage2 = self._request(
                    "binary_success", stage2_id,
                    {
                        "goal_phrase": goal_phrase,
                        "panorama_png_base64": base64.b64encode(png).decode("ascii"),
                        "panorama_png_sha256": panorama["panorama_png_sha256"],
                    },
                )
                label = self._parse_binary(stage2_output)
                record = {
                    "schema": CACHE_SCHEMA,
                    "provider": self.identity,
                    "cache_key": cache_key,
                    "input_sha256": input_sha256,
                    "episode_id": episode["episode_id"],
                    "scan_id": episode["scan_id"],
                    "endpoint_viewpoint_id": episode["endpoint_viewpoint_id"],
                    "trajectory_sha256": trajectory_sha256,
                    "panorama": panorama,
                    "stage1": stage1,
                    "stage2": stage2,
                    "parsed_label": label,
                    "token_count": {
                        "prompt": stage1["prompt_tokens"] + stage2["prompt_tokens"],
                        "completion": (
                            stage1["completion_tokens"]
                            + stage2["completion_tokens"]
                        ),
                    },
                    "cost": 0.0,
                    "created_at_unix": time.time(),
                }
                # Only a fully parsed, provider-bound response reaches the cache.
                self._atomic_write(cache_path, record)
                return self._succeeded(record, cache_hit=False)
        except Exception as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return self._failed(error, episode=episode)

    def _succeeded(self, record, cache_hit):
        label = record["parsed_label"]
        self.last_cache_key = record["cache_key"]
        self.last_record_sha256 = _sha256_bytes(_canonical_json_bytes(record))
        result = {
            "available": True,
            "success": label == "Yes",
            "cache_hit": bool(cache_hit),
            "cache_key": record["cache_key"],
            "record_sha256": self.last_record_sha256,
            "request_sha256_values": [
                record["stage1"]["request_sha256"],
                record["stage2"]["request_sha256"],
            ],
            "response_sha256_values": [
                record["stage1"]["response_sha256"],
                record["stage2"]["response_sha256"],
            ],
            "parsed_label": label,
        }
        self._append_transcript({
            "status": "label",
            "provider_id": PROVIDER_ID,
            "model_id": self.identity["model_id"],
            "revision": self.revision,
            "weights_sha256": self.weights_sha256,
            "bundle_sha256": self.bundle_sha256,
            "prompt_sha256_values": [
                STAGE1_PROMPT_SHA256, STAGE2_PROMPT_SHA256,
            ],
            "episode_id": record["episode_id"],
            "scan_id": record["scan_id"],
            "endpoint_viewpoint_id": record["endpoint_viewpoint_id"],
            "trajectory_sha256": record["trajectory_sha256"],
            "panorama_png_sha256": record["panorama"]["panorama_png_sha256"],
            "cache_key": record["cache_key"],
            "cache_hit": bool(cache_hit),
            "record_sha256": self.last_record_sha256,
            "request_sha256_values": result["request_sha256_values"],
            "response_sha256_values": result["response_sha256_values"],
            "parsed_label": label,
            "latency_seconds_values": [
                record["stage1"]["latency_seconds"],
                record["stage2"]["latency_seconds"],
            ],
            "token_count": record["token_count"],
            "cost": record["cost"],
        })
        self.labels += 1
        self.positive_labels += int(label == "Yes")
        self.negative_labels += int(label == "No")
        return result

    def _failed(self, error, episode=None):
        self.failures += 1
        self.last_failure = "{}: {}".format(type(error).__name__, str(error))
        failure_document = {
            "schema": "navtta.reverie_llm_feedback_failure.v1",
            "provider": self.identity,
            "query_index": self.query_episodes,
            "episode_id": (
                episode.get("episode_id") if isinstance(episode, dict) else None
            ),
            "scan_id": (
                episode.get("scan_id") if isinstance(episode, dict) else None
            ),
            "endpoint_viewpoint_id": (
                episode.get("endpoint_viewpoint_id")
                if isinstance(episode, dict) else None
            ),
            "failure": self.last_failure,
            "http_requests_so_far": self.http_requests,
            "created_at_unix": time.time(),
        }
        failure_key = _sha256_bytes(_canonical_json_bytes(failure_document))
        failure_path = (
            self.cache_dir / "failures"
            / ("{:06d}-{}.json".format(self.query_episodes, failure_key))
        )
        try:
            self._atomic_write(failure_path, failure_document)
        except OSError:
            # Failure logging must not turn a skipped update into a truth
            # fallback or a fabricated negative label.
            pass
        result = {
            "available": False,
            "failure": self.last_failure,
            "cache_hit": False,
            "failure_record_sha256": failure_key,
        }
        try:
            self._append_transcript({
                "status": "failed_closed",
                "episode_id": failure_document["episode_id"],
                "scan_id": failure_document["scan_id"],
                "endpoint_viewpoint_id": failure_document["endpoint_viewpoint_id"],
                "failure_record_sha256": failure_key,
                "failure": self.last_failure,
            })
        except Exception as transcript_error:
            result["transcript_failure"] = "{}: {}".format(
                type(transcript_error).__name__, str(transcript_error)
            )
        return result

    def fail_closed(self, error, episode=None):
        """Expose the same no-label policy for pre-query input failures."""
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise error
        self.query_episodes += 1
        return self._failed(error, episode=episode)

    def diagnostics(self):
        result = self.identity
        result.update({
            "supervision": "external_mllm_pseudo_feedback",
            "reported_method_label": "FeedTTA-LLM",
            "binary_feedback_endpoint": BINARY_FEEDBACK_ENDPOINT,
            "query_episodes": self.query_episodes,
            "http_requests": self.http_requests,
            "pseudo_labels": self.labels,
            "positive_labels": self.positive_labels,
            "negative_labels": self.negative_labels,
            "failed_closed_episodes": self.failures,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": 0.0,
            "cache_dir": str(self.cache_dir),
            "transcript_path": str(self.transcript_path),
            "transcript_events": self.transcript_events,
            "transcript_sha256": self.transcript_sha256,
            "last_cache_key": self.last_cache_key,
            "last_record_sha256": self.last_record_sha256,
            "last_failure": self.last_failure,
            "max_labels": self.max_labels,
            "max_http_requests": self.max_requests,
            "requests_per_uncached_episode": 2,
            "failure_policy": "no_parameter_update_no_truth_fallback",
            "abort_on_failure": self.abort_on_failure,
            "feedback_timing": "post_episode_affects_future_episodes_only",
        })
        return result
