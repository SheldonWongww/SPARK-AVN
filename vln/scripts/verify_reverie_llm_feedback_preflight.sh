#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: vln/scripts/verify_reverie_llm_feedback_preflight.sh \
         TAG GPU SCAN VIEWPOINT TOKEN_FILE RENDER_MATTERSIM_BUILD [SERVICE_URL]

Run the formal FeedTTA-LLM provider preflight against an already-running
loopback service.  The script verifies the pinned model on CUDA/float16,
performs one real 36-view headless MatterSim render from licensed raw RGB,
then sends one two-stage provider smoke request.  Evidence is written outside
Git under /data1/wxy/exp_data/NavTTA/vln/tmp/reverie-llm-preflight/TAG.

The ordinary NavTTA MatterSim build has EGL_RENDERING=OFF and
OSMESA_RENDERING=OFF and is intentionally rejected.  Supply a separate,
pinned Python-3.8-compatible EGL or OSMesa build; this script does not build
it and does not download Matterport assets.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

[[ "$#" -ge 6 && "$#" -le 7 ]] || { usage >&2; exit 2; }
TAG="$1"
GPU="$2"
SCAN="$3"
VIEWPOINT="$4"
TOKEN_FILE="$5"
RENDER_BUILD="$6"
SERVICE_URL="${7:-http://127.0.0.1:8765}"

[[ "${TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid evidence tag"
[[ "${GPU}" =~ ^[0-9]+$ ]] || die "GPU must be a nonnegative integer"

REPO_ROOT=/data1/wxy/code/NavTTA
VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln
CORE_ROOT="${REPO_ROOT}/core"
STREAM_PYTHON="${VLN_ROOT}/envs/streamvln/bin/python"
GOAT_PYTHON="${VLN_ROOT}/envs/goat/bin/python"
MODEL_DIR="${VLN_ROOT}/models/Qwen2-VL-2B-Instruct"
CONNECTIVITY_DIR="${NAVTTA_REVERIE_CONNECTIVITY_DIR:-${REPO_ROOT}/vln/data/goat/R2R/connectivity}"
SCAN_DATA_DIR="${NAVTTA_REVERIE_SCAN_DATA_DIR:-${REPO_ROOT}/vln/data/goat/Matterport3D/v1_unzip_scans}"
NATIVE_LIB="${VLN_ROOT}/envs/mattersim-native/lib"
OUTPUT_ROOT="${VLN_ROOT}/tmp/reverie-llm-preflight/${TAG}"

[[ -x "${STREAM_PYTHON}" ]] || die "missing StreamVLN Python: ${STREAM_PYTHON}"
[[ -x "${GOAT_PYTHON}" ]] || die "missing GOAT Python: ${GOAT_PYTHON}"
[[ -f "${TOKEN_FILE}" ]] || die "missing token file: ${TOKEN_FILE}"
[[ "$(stat -c '%a' "${TOKEN_FILE}")" == "600" ]] || \
    die "token file permissions must be 0600"
[[ -d "${RENDER_BUILD}" ]] || die "missing render MatterSim build: ${RENDER_BUILD}"
[[ ! -e "${OUTPUT_ROOT}" ]] || die "evidence directory already exists: ${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export PYTHONNOUSERSITE=1
export NAVTTA_EXPECTED_CORE_ROOT="${CORE_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

assert_current_core() {
    local python="$1"
    PYTHONPATH="${CORE_ROOT}" "${python}" - <<'PY'
import os
from pathlib import Path
import navtta_core

expected = Path(os.environ["NAVTTA_EXPECTED_CORE_ROOT"]).resolve()
actual = Path(navtta_core.__file__).resolve()
try:
    actual.relative_to(expected)
except ValueError:
    raise SystemExit("unexpected navtta_core import: {}".format(actual))
PY
}

assert_current_core "${STREAM_PYTHON}"
assert_current_core "${GOAT_PYTHON}"

PYTHONPATH="${CORE_ROOT}:${REPO_ROOT}/vln" \
    "${STREAM_PYTHON}" \
    "${REPO_ROOT}/vln/scripts/serve_reverie_llm_feedback.py" \
    --model-dir "${MODEL_DIR}" --device cuda:0 --dtype float16 --verify-only \
    >"${OUTPUT_ROOT}/model_verify.json"

PYTHONPATH="${CORE_ROOT}:${REPO_ROOT}/vln:${RENDER_BUILD}" \
LD_LIBRARY_PATH="${RENDER_BUILD}:${NATIVE_LIB}:${VLN_ROOT}/envs/goat/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${GOAT_PYTHON}" \
    "${REPO_ROOT}/vln/scripts/verify_reverie_render_preflight.py" \
    --mattersim-build "${RENDER_BUILD}" \
    --connectivity-dir "${CONNECTIVITY_DIR}" \
    --scan-data-dir "${SCAN_DATA_DIR}" \
    --scan "${SCAN}" --viewpoint "${VIEWPOINT}" \
    --output-dir "${OUTPUT_ROOT}/render" --require-clean-git \
    >"${OUTPUT_ROOT}/render_pointer.json"

PYTHONPATH="${CORE_ROOT}:${REPO_ROOT}/vln:${RENDER_BUILD}" \
LD_LIBRARY_PATH="${RENDER_BUILD}:${NATIVE_LIB}:${VLN_ROOT}/envs/goat/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${GOAT_PYTHON}" \
    "${REPO_ROOT}/vln/scripts/serve_reverie_llm_feedback.py" \
    --smoke-service-url "${SERVICE_URL}" \
    --smoke-scan "${SCAN}" --smoke-viewpoint "${VIEWPOINT}" \
    --smoke-instruction "Go to the destination shown at the endpoint." \
    --connectivity-dir "${CONNECTIVITY_DIR}" \
    --scan-data-dir "${SCAN_DATA_DIR}" \
    --cache-dir "${OUTPUT_ROOT}/provider-cache" \
    --transcript-path "${OUTPUT_ROOT}/provider_smoke_transcript.ndjson" \
    --token-file "${TOKEN_FILE}" \
    >"${OUTPUT_ROOT}/provider_smoke.json"

"${STREAM_PYTHON}" - \
    "${OUTPUT_ROOT}" "${REPO_ROOT}" "${RENDER_BUILD}" \
    "${SERVICE_URL}" "${SCAN}" "${VIEWPOINT}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
render_build = Path(sys.argv[3]).resolve()
service_url, scan, viewpoint = sys.argv[4:7]

def load(name):
    path = root / name
    with path.open("r", encoding="utf-8") as stream:
        return path, json.load(stream)

def evidence(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path), "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }

model_path, model = load("model_verify.json")
render_path, render = load("render/render_evidence.json")
smoke_path, smoke = load("provider_smoke.json")
transcript_path = root / "provider_smoke_transcript.ndjson"
if render.get("status") != "passed":
    raise SystemExit("render preflight did not pass")
if smoke.get("result", {}).get("available") is not True:
    raise SystemExit("provider smoke did not return an available pseudo-label")
service = smoke.get("service", {})
if service.get("ready") is not True or not str(service.get("cuda_device", "")).startswith("cuda"):
    raise SystemExit("provider smoke was not served by CUDA")
if model.get("model_dtype") != "float16" or model.get("bundle_sha256") != service.get("bundle_sha256"):
    raise SystemExit("model verify/provider service identity mismatch")
if Path(render.get("headless_build", {}).get("build_dir", "")).resolve() != render_build:
    raise SystemExit("render evidence build path mismatch")
commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
if render.get("repository", {}).get("git_commit") != commit:
    raise SystemExit("render evidence Git commit mismatch")
document = {
    "schema": "navtta.reverie_llm_feedback_preflight.v1",
    "status": "passed",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": commit,
    "render_mattersim_build": str(render_build),
    "service_url": service_url,
    "scan_id": scan,
    "viewpoint_id": viewpoint,
    "model_identity": {
        key: model.get(key) for key in (
            "model_id", "model_dtype", "revision", "weights_sha256",
            "bundle_sha256", "prompt_bundle_sha256",
        )
    },
    "artifacts": {
        "model_verify": evidence(model_path),
        "render_evidence": evidence(render_path),
        "rendered_panorama": evidence(root / "render/endpoint_panorama.png"),
        "provider_smoke": evidence(smoke_path),
        "provider_transcript": evidence(transcript_path),
    },
}
temporary = root / ("PRECHECK.json.tmp.{}".format(os.getpid()))
with temporary.open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(document, stream, indent=2, sort_keys=True, ensure_ascii=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(str(temporary), str(root / "PRECHECK.json"))
PY

(
    cd "${OUTPUT_ROOT}"
    sha256sum \
        model_verify.json \
        render_pointer.json \
        render/render_evidence.json \
        render/endpoint_panorama.png \
        provider_smoke.json \
        provider_smoke_transcript.ndjson \
        PRECHECK.json \
        >SHA256SUMS
)

printf 'REVERIE FeedTTA-LLM preflight passed\n'
printf 'Evidence: %s\n' "${OUTPUT_ROOT}"
printf 'For the formal test stage export NAVTTA_REVERIE_RENDER_MATTERSIM_BUILD=%s\n' "${RENDER_BUILD}"
printf 'For the formal test stage export NAVTTA_REVERIE_LLM_PREFLIGHT=%s\n' "${OUTPUT_ROOT}/PRECHECK.json"
