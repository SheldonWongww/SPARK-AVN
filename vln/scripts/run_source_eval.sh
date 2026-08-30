#!/usr/bin/env bash
set -euo pipefail

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export MKL_NUM_THREADS=1
fi

usage() {
    cat <<'EOF'
Usage: vln/scripts/run_source_eval.sh SETTING SPLIT [GPU] [--run-tag TAG]
                                      [--ce-data-version VERSION]
                                      [--smoke-episodes N]
                                      [--tta-config FILE]
                                      [--result-root DIR]
                                      [--order-seed 0|1|2|3]
                                      [--source-order-manifest DIR]
                                      [--adapter-parity-audit]
                                      [--episode-limit N] [--dry-run]

SETTING:
  duet-r2r duet-reverie hamt-r2r hamt-reverie goat-r2r goat-reverie
  etpnav-r2r-ce bevbert-r2r-ce streamvln-r2r-ce

SPLIT is val_seen, val_unseen, test, train, or all.  `train` is accepted only
for an IDEA offline source-statistics collection config and an exact 128-item
--source-order-manifest.  "all" always runs in the
canonical val_seen -> val_unseen -> test order, using a fresh process and
output directory for each split.  Test produces submission trajectories; it
does not produce a local test score.

VERSION controls ETPNav/BEVBert only: v1.3-unified (default, comparable to
StreamVLN start states) or v1.2-native (paper/upstream reproduction).

--smoke-episodes N is a short GPU lifecycle check.  It is restricted to
val_seen, uses the canonical prefix, and writes under vln/results/smoke/.

--tta-config FILE enables a TTA/control job described by one JSON file.
--episode-limit N is a development prefix used by the hyperparameter
scheduler.  The isolated adapter-parity audit requires exactly 256 episodes.
--adapter-parity-audit is required by the isolated zero-write parity schema
and is rejected for every ordinary tuning/source configuration.

--order-seed is reserved for complete, frozen-hyperparameter val_seen or
val_unseen shuffled-order jobs on the eight staged-search settings.  Seed 0
keeps the canonical scene-blocked manifest; seeds 1, 2, and 3 use tracked
global SHA256-ranked permutations.  It cannot be combined with smoke/prefix
jobs, Source, StreamVLN, split test/all, or native CE v1.2.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

SETTING="${1:-}"
SPLIT="${2:-}"
[[ -n "${SETTING}" && -n "${SPLIT}" ]] || { usage; exit 2; }
shift 2

GPU=0
GPU_SET=0
DRY_RUN=0
RUN_TAG="${NAVTTA_RUN_TAG:-}"
RUN_TAG_SET=0
CE_DATA_VERSION="${NAVTTA_CE_DATA_VERSION:-v1.3-unified}"
CE_DATA_VERSION_SET=0
SMOKE_EPISODES=""
TTA_CONFIG=""
RESULT_ROOT_OVERRIDE=""
RESULT_ROOT_SET=0
EPISODE_LIMIT=""
ORDER_SEED=""
ORDER_SEED_SET=0
ADAPTER_PARITY_AUDIT=0
SOURCE_ORDER_MANIFEST=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run)
            [[ "${DRY_RUN}" -eq 0 ]] || die "duplicate option: --dry-run"
            DRY_RUN=1
            shift
            ;;
        --run-tag)
            [[ "$#" -ge 2 ]] || die "--run-tag requires a value"
            [[ "${RUN_TAG_SET}" -eq 0 ]] || die "run tag specified more than once"
            RUN_TAG="$2"
            RUN_TAG_SET=1
            shift 2
            ;;
        --ce-data-version)
            [[ "$#" -ge 2 ]] || die "--ce-data-version requires a value"
            [[ "${CE_DATA_VERSION_SET}" -eq 0 ]] || die "CE data version specified more than once"
            CE_DATA_VERSION="$2"
            CE_DATA_VERSION_SET=1
            shift 2
            ;;
        --smoke-episodes)
            [[ "$#" -ge 2 ]] || die "--smoke-episodes requires a value"
            [[ -z "${SMOKE_EPISODES}" ]] || die "smoke episode count specified more than once"
            [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "--smoke-episodes must be a positive integer"
            SMOKE_EPISODES="$2"
            shift 2
            ;;
        --tta-config)
            [[ "$#" -ge 2 ]] || die "--tta-config requires a value"
            [[ -z "${TTA_CONFIG}" ]] || die "TTA config specified more than once"
            TTA_CONFIG="$2"
            shift 2
            ;;
        --result-root)
            [[ "$#" -ge 2 ]] || die "--result-root requires a value"
            [[ "${RESULT_ROOT_SET}" -eq 0 ]] || \
                die "result root specified more than once"
            RESULT_ROOT_OVERRIDE="$2"
            RESULT_ROOT_SET=1
            shift 2
            ;;
        --episode-limit)
            [[ "$#" -ge 2 ]] || die "--episode-limit requires a value"
            [[ -z "${EPISODE_LIMIT}" ]] || die "episode limit specified more than once"
            [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "--episode-limit must be a positive integer"
            EPISODE_LIMIT="$2"
            shift 2
            ;;
        --order-seed)
            [[ "$#" -ge 2 ]] || die "--order-seed requires a value"
            [[ "${ORDER_SEED_SET}" -eq 0 ]] || die "order seed specified more than once"
            case "$2" in
                0|1|2|3) ORDER_SEED="$2" ;;
                *) die "--order-seed must be exactly 0, 1, 2, or 3" ;;
            esac
            ORDER_SEED_SET=1
            shift 2
            ;;
        --adapter-parity-audit)
            [[ "${ADAPTER_PARITY_AUDIT}" -eq 0 ]] || \
                die "duplicate option: --adapter-parity-audit"
            ADAPTER_PARITY_AUDIT=1
            shift
            ;;
        --source-order-manifest)
            [[ "$#" -ge 2 ]] || die "--source-order-manifest requires a value"
            [[ -z "${SOURCE_ORDER_MANIFEST}" ]] || \
                die "source order manifest specified more than once"
            SOURCE_ORDER_MANIFEST="$2"
            shift 2
            ;;
        ''|*[!0-9]*)
            die "unknown option or invalid GPU index: $1"
            ;;
        *)
            [[ "${GPU_SET}" -eq 0 ]] || die "GPU index specified more than once"
            GPU="$1"
            GPU_SET=1
            shift
            ;;
    esac
done
if [[ -z "${RUN_TAG}" ]]; then
    if [[ -n "${SMOKE_EPISODES}" ]]; then
        RUN_TAG="gpu-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
    elif [[ -n "${TTA_CONFIG}" ]]; then
        RUN_TAG="tta-$(date -u +%Y%m%dT%H%M%SZ)"
    else
        RUN_TAG="source-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
fi
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid run tag: ${RUN_TAG}"
case "${CE_DATA_VERSION}" in
    v1.3-unified|v1.2-native) ;;
    *) die "invalid CE data version: ${CE_DATA_VERSION}" ;;
esac
case "${SETTING}" in
    etpnav-r2r-ce|bevbert-r2r-ce) ;;
    streamvln-r2r-ce)
        [[ "${CE_DATA_VERSION_SET}" -eq 0 && "${CE_DATA_VERSION}" == "v1.3-unified" ]] || \
            die "--ce-data-version applies only to ETPNav/BEVBert; StreamVLN is fixed to v1.3"
        ;;
    *)
        [[ "${CE_DATA_VERSION_SET}" -eq 0 && "${CE_DATA_VERSION}" == "v1.3-unified" ]] || \
            die "--ce-data-version applies only to ETPNav/BEVBert"
        ;;
esac
if [[ -n "${SMOKE_EPISODES}" && "${SPLIT}" != "val_seen" ]]; then
    die "--smoke-episodes is restricted to split val_seen"
fi
if [[ -n "${SMOKE_EPISODES}" && -n "${EPISODE_LIMIT}" ]]; then
    die "--smoke-episodes and --episode-limit are mutually exclusive"
fi
if [[ -n "${EPISODE_LIMIT}" && -z "${TTA_CONFIG}" ]]; then
    die "--episode-limit is reserved for TTA/control development jobs"
fi
if [[ -n "${EPISODE_LIMIT}" && "${SPLIT}" != "val_seen" ]]; then
    die "prefix hyperparameter jobs are restricted to val_seen"
fi
if [[ "${ORDER_SEED_SET}" -eq 1 ]]; then
    case "${SPLIT}" in
        val_seen|val_unseen) ;;
        *) die "--order-seed is restricted to complete val_seen or val_unseen jobs" ;;
    esac
    [[ -z "${SMOKE_EPISODES}" ]] || \
        die "--order-seed cannot be combined with --smoke-episodes"
    [[ -z "${EPISODE_LIMIT}" ]] || \
        die "--order-seed cannot be combined with --episode-limit"
    [[ -n "${TTA_CONFIG}" ]] || \
        die "--order-seed requires a TTA robustness job config"
    [[ "${CE_DATA_VERSION}" == "v1.3-unified" ]] || \
        die "--order-seed requires the unified CE v1.3 protocol"
    case "${SETTING}" in
        duet-r2r|duet-reverie|hamt-r2r|hamt-reverie|goat-r2r|goat-reverie|etpnav-r2r-ce|bevbert-r2r-ce) ;;
        streamvln-r2r-ce) die "--order-seed does not support StreamVLN" ;;
        *) die "--order-seed supports only the eight staged-search settings" ;;
    esac
fi
MODEL_SEED=0
if [[ "${ORDER_SEED_SET}" -eq 1 ]]; then
    MODEL_SEED="${ORDER_SEED}"
fi

REPO_ROOT=/data1/wxy/code/NavTTA
VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln
ENV_ROOT="${VLN_ROOT}/envs"
CACHE_ROOT="${VLN_ROOT}/cache"
TMP_ROOT="${VLN_ROOT}/tmp"
HOME_ROOT="${VLN_ROOT}/home"
XDG_CACHE_ROOT="${CACHE_ROOT}/xdg"

# Config translation and protocol guards run before select_env() amends PATH.
# Resolve them through the setting's pinned environment instead of assuming a
# login-shell `python3` exists (minimal AutoDL images do not provide one).
case "${SETTING}" in
    duet-r2r|duet-reverie) BOOTSTRAP_ENV_NAME=duet ;;
    hamt-r2r|hamt-reverie) BOOTSTRAP_ENV_NAME=hamt ;;
    goat-r2r|goat-reverie) BOOTSTRAP_ENV_NAME=goat ;;
    etpnav-r2r-ce|bevbert-r2r-ce) BOOTSTRAP_ENV_NAME=vlnce017 ;;
    streamvln-r2r-ce) BOOTSTRAP_ENV_NAME=streamvln ;;
    *) die "invalid setting: ${SETTING}" ;;
esac
BOOTSTRAP_PYTHON="${ENV_ROOT}/${BOOTSTRAP_ENV_NAME}/bin/python"
[[ -x "${BOOTSTRAP_PYTHON}" ]] || \
    die "missing bootstrap Python: ${BOOTSTRAP_PYTHON}"

TTA_METHOD=source
TTA_NAMESPACE=tuning
TTA_TRANSLATOR="${REPO_ROOT}/vln/scripts/tta_config_cli.py"
IDEA_SOURCE_STATS_PATH=""
IDEA_SOURCE_STATS_SHA256=""
IDEA_SOURCE_STATS_OUTPUT=""
IDEA_SOURCE_CHECKPOINT_SHA256=""
IDEA_SOURCE_DATASET=""
IDEA_SOURCE_DATASET_VERSION=""
IDEA_SOURCE_SPLIT=""
IDEA_SOURCE_TRAJECTORIES=""
IDEA_SOURCE_COLLECTION_POLICY=""
IDEA_SOURCE_COLLECTION=0
if [[ -n "${TTA_CONFIG}" ]]; then
    [[ "${TTA_CONFIG}" = /* ]] || TTA_CONFIG="${REPO_ROOT}/${TTA_CONFIG}"
    [[ -f "${TTA_CONFIG}" ]] || die "missing TTA config: ${TTA_CONFIG}"
    [[ -f "${TTA_TRANSLATOR}" ]] || die "missing TTA config translator"
    if ! TTA_METHOD="$(
        "${BOOTSTRAP_PYTHON}" "${TTA_TRANSLATOR}" --setting "${SETTING}" \
            --config "${TTA_CONFIG}" --diagnostics /tmp/navtta-unused.json \
            --print-method
    )"; then
        die "invalid TTA config: ${TTA_CONFIG}"
    fi
    if ! TTA_NAMESPACE="$(
        "${BOOTSTRAP_PYTHON}" "${TTA_TRANSLATOR}" --setting "${SETTING}" \
            --config "${TTA_CONFIG}" --diagnostics /tmp/navtta-unused.json \
            --print-namespace
    )"; then
        die "cannot resolve TTA result namespace: ${TTA_CONFIG}"
    fi
    case "${TTA_NAMESPACE}" in
        tuning|adapter_parity_audit|idea_source_statistics) ;;
        *) die "unsupported TTA result namespace: ${TTA_NAMESPACE}" ;;
    esac
    if [[ "${TTA_METHOD}" == "idea" ]]; then
        if ! IDEA_SOURCE_STATS_BINDING_TEXT="$(
            "${BOOTSTRAP_PYTHON}" - "${TTA_CONFIG}" <<'PY'
import hashlib
import json
import os
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    parameters = json.load(stream).get("parameters", {})
collecting = parameters.get("collect_source_stats", False)
if type(collecting) is not bool:
    raise SystemExit("IDEA collect_source_stats must be boolean")
if collecting:
    output = parameters.get("source_stats_output")
    checkpoint = parameters.get("source_checkpoint_sha256")
    if not isinstance(output, str) or not os.path.isabs(output):
        raise SystemExit("IDEA source_stats_output must be absolute")
    if not isinstance(checkpoint, str) or len(checkpoint) != 64:
        raise SystemExit("IDEA source_checkpoint_sha256 is invalid")
    try:
        int(checkpoint, 16)
    except ValueError:
        raise SystemExit("IDEA source_checkpoint_sha256 is invalid")
    print("collection")
    print(output)
    print(checkpoint.lower())
    print(parameters["source_dataset"])
    print(parameters["source_dataset_version"])
    print(parameters["source_split"])
    print(parameters.get("source_trajectories", 128))
    print(parameters["collection_policy"])
    raise SystemExit(0)
path = parameters.get("source_stats_path")
expected = parameters.get("source_stats_sha256")
if not isinstance(path, str) or not os.path.isabs(path):
    raise SystemExit("IDEA source_stats_path must be absolute")
if not isinstance(expected, str) or len(expected) != 64:
    raise SystemExit("IDEA source_stats_sha256 is invalid")
try:
    int(expected, 16)
except ValueError:
    raise SystemExit("IDEA source_stats_sha256 is invalid")
if not os.path.isfile(path):
    raise SystemExit("IDEA source-statistics artifact is missing: {}".format(path))
digest = hashlib.sha256()
with open(path, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected.lower():
    raise SystemExit(
        "IDEA source-statistics SHA256 mismatch: expected {}, got {}"
        .format(expected, actual)
    )
print("evaluation")
print(path)
print(actual)
PY
        )"; then
            die "invalid IDEA source-statistics binding"
        fi
        mapfile -t IDEA_SOURCE_STATS_BINDING <<< "${IDEA_SOURCE_STATS_BINDING_TEXT}"
        if [[ "${IDEA_SOURCE_STATS_BINDING[0]}" == "collection" ]]; then
            [[ "${#IDEA_SOURCE_STATS_BINDING[@]}" -eq 8 ]] || \
                die "invalid IDEA source-collection binding"
            IDEA_SOURCE_COLLECTION=1
            IDEA_SOURCE_STATS_OUTPUT="${IDEA_SOURCE_STATS_BINDING[1]}"
            IDEA_SOURCE_CHECKPOINT_SHA256="${IDEA_SOURCE_STATS_BINDING[2]}"
            IDEA_SOURCE_DATASET="${IDEA_SOURCE_STATS_BINDING[3]}"
            IDEA_SOURCE_DATASET_VERSION="${IDEA_SOURCE_STATS_BINDING[4]}"
            IDEA_SOURCE_SPLIT="${IDEA_SOURCE_STATS_BINDING[5]}"
            IDEA_SOURCE_TRAJECTORIES="${IDEA_SOURCE_STATS_BINDING[6]}"
            IDEA_SOURCE_COLLECTION_POLICY="${IDEA_SOURCE_STATS_BINDING[7]}"
        else
            [[ "${#IDEA_SOURCE_STATS_BINDING[@]}" -eq 3 ]] || \
                die "invalid IDEA source-statistics binding"
            IDEA_SOURCE_STATS_PATH="${IDEA_SOURCE_STATS_BINDING[1]}"
            IDEA_SOURCE_STATS_SHA256="${IDEA_SOURCE_STATS_BINDING[2]}"
        fi
    fi
fi
if [[ "${IDEA_SOURCE_COLLECTION}" -eq 1 ]]; then
    [[ "${TTA_NAMESPACE}" == "idea_source_statistics" ]] || \
        die "IDEA source collection requires namespace=idea_source_statistics"
    [[ "${SPLIT}" == "train" ]] || \
        die "IDEA source collection requires split train"
    [[ "${IDEA_SOURCE_SPLIT}" == "train" ]] || \
        die "IDEA source collection config requires source_split=train"
    [[ "${IDEA_SOURCE_TRAJECTORIES}" == "128" ]] || \
        die "IDEA source collection requires exactly 128 trajectories"
    [[ "${IDEA_SOURCE_COLLECTION_POLICY}" == \
       "frozen_source_argmax_rollout" ]] || \
        die "IDEA source collection policy must be frozen_source_argmax_rollout"
    [[ "${ORDER_SEED_SET}" -eq 0 ]] || \
        die "IDEA source collection cannot use --order-seed"
    [[ -z "${SMOKE_EPISODES}" && -z "${EPISODE_LIMIT}" ]] || \
        die "IDEA source collection cannot use smoke/prefix episode limits"
    [[ -n "${SOURCE_ORDER_MANIFEST}" ]] || \
        die "IDEA source collection requires --source-order-manifest"
    [[ "${SOURCE_ORDER_MANIFEST}" = /* ]] || \
        SOURCE_ORDER_MANIFEST="${REPO_ROOT}/${SOURCE_ORDER_MANIFEST}"
    [[ -d "${SOURCE_ORDER_MANIFEST}" && \
       -f "${SOURCE_ORDER_MANIFEST}/train.json" ]] || \
        die "missing IDEA source-order manifest: ${SOURCE_ORDER_MANIFEST}/train.json"
elif [[ "${SPLIT}" == "train" || -n "${SOURCE_ORDER_MANIFEST}" || \
        "${TTA_NAMESPACE}" == "idea_source_statistics" ]]; then
    die "train/source-order collection options require an IDEA source-collection config"
fi
if [[ "${TTA_NAMESPACE}" == "adapter_parity_audit" && \
      "${ADAPTER_PARITY_AUDIT}" -ne 1 ]]; then
    die "adapter-parity config requires --adapter-parity-audit"
fi
if [[ "${TTA_NAMESPACE}" != "adapter_parity_audit" && \
      "${ADAPTER_PARITY_AUDIT}" -ne 0 ]]; then
    die "--adapter-parity-audit requires the adapter-parity config schema"
fi
if [[ "${ORDER_SEED_SET}" -eq 1 ]]; then
    [[ "${TTA_METHOD}" != "source" ]] || \
        die "--order-seed cannot be used for Source-only jobs"
    [[ "${TTA_NAMESPACE}" == "tuning" ]] || \
        die "--order-seed cannot be used for adapter-parity jobs"
fi
if [[ -n "${TTA_CONFIG}" ]]; then
    if ! "${BOOTSTRAP_PYTHON}" - \
        "${TTA_CONFIG}" "${ORDER_SEED_SET}" "${ORDER_SEED}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    document = json.load(stream)
cli_present = sys.argv[2] == "1"
seed = int(sys.argv[3]) if cli_present else None
is_adapter_audit = (
    document.get("schema") == "navtta.vln_tta_adapter_parity_job.v1"
    and document.get("namespace") == "adapter_parity_audit"
)
if is_adapter_audit:
    audit_order_seed = document.get("order_seed")
    if type(audit_order_seed) is not int or audit_order_seed != 0:
        raise SystemExit(
            "adapter-parity config order_seed must be the exact integer 0"
        )
declares_order_metadata = (
    document.get("stage") == "orders"
    or (not is_adapter_audit and "order_seed" in document)
)
if declares_order_metadata and not cli_present:
    raise SystemExit(
        "ordinary TTA config declares orders metadata but --order-seed is absent"
    )
if cli_present:
    checks = {
        "schema": "navtta.vln_tta_job.v1",
        "stage": "orders",
    }
    for key, expected in checks.items():
        if document.get(key) != expected:
            raise SystemExit(
                "order robustness config {} mismatch: expected {!r}, got {!r}"
                .format(key, expected, document.get(key))
            )
    for key, expected in (("episodes", -1), ("order_seed", seed)):
        value = document.get(key)
        if type(value) is not int or value != expected:
            raise SystemExit(
                "order robustness config {} mismatch: expected exact integer {!r}, "
                "got {!r}".format(key, expected, value)
            )
    method = document.get("method")
    if method == "source" or document.get("search_method") != method:
        raise SystemExit(
            "order robustness config must name one non-Source search method"
        )
PY
    then
        die "TTA config and --order-seed protocol do not match"
    fi
fi
if [[ "${ADAPTER_PARITY_AUDIT}" -eq 1 ]]; then
    [[ "${SPLIT}" == "val_seen" ]] || \
        die "adapter-parity audit is restricted to val_seen"
    [[ -z "${SMOKE_EPISODES}" ]] || \
        die "adapter-parity audit cannot use --smoke-episodes"
    [[ "${EPISODE_LIMIT}" == "256" ]] || \
        die "adapter-parity audit requires --episode-limit 256"
fi
if [[ "${RESULT_ROOT_SET}" -eq 1 ]]; then
    [[ -n "${TTA_CONFIG}" ]] || \
        die "--result-root is reserved for TTA search jobs"
    [[ "${TTA_NAMESPACE}" == "tuning" || \
       "${TTA_NAMESPACE}" == "idea_source_statistics" ]] || \
        die "--result-root cannot override this result namespace"
    if [[ "${TTA_NAMESPACE}" == "idea_source_statistics" ]]; then
        [[ "${SPLIT}" == "train" ]] || \
            die "IDEA source-statistics result roots require split train"
    else
        case "${SPLIT}" in
            val_seen|val_unseen) ;;
            *) die "--result-root is restricted to val_seen or val_unseen tuning jobs" ;;
        esac
    fi
    [[ "${RESULT_ROOT_OVERRIDE}" = /* ]] || \
        die "--result-root must be absolute"
fi

SOURCE_TAG_LOCK_FD=""
SOURCE_TAG_LOCK_ROOT="${TMP_ROOT}/navtta-source-tag-locks"
SOURCE_TAG_LOCK_FILE="${SOURCE_TAG_LOCK_ROOT}/${RUN_TAG}.lock"

claim_or_verify_source_tag_lock() {
    local inherited_fd
    local inherited_path
    command -v flock >/dev/null 2>&1 || die "flock is required for evaluation runs"
    mkdir -p "${SOURCE_TAG_LOCK_ROOT}"
    if [[ "${NAVTTA_SOURCE_TAG_LOCKED:-}" == "${RUN_TAG}" ]]; then
        inherited_fd="${NAVTTA_SOURCE_TAG_LOCK_FD:-}"
        [[ "${inherited_fd}" =~ ^[1-9][0-9]*$ ]] || \
            die "invalid inherited source tag lock descriptor"
        if ! inherited_path="$(
            readlink -f "/proc/$$/fd/${inherited_fd}" 2>/dev/null
        )"; then
            die "inherited source tag lock descriptor is unavailable"
        fi
        [[ "${inherited_path}" == "${SOURCE_TAG_LOCK_FILE}" ]] || \
            die "inherited source tag lock points to the wrong tag"
        flock -n "${inherited_fd}" || die "inherited source tag lock is not held"
        SOURCE_TAG_LOCK_FD="${inherited_fd}"
    else
        exec {SOURCE_TAG_LOCK_FD}>"${SOURCE_TAG_LOCK_FILE}"
        if ! flock -n "${SOURCE_TAG_LOCK_FD}"; then
            die "source run tag is already active: ${RUN_TAG}"
        fi
        export NAVTTA_SOURCE_TAG_LOCKED="${RUN_TAG}"
        export NAVTTA_SOURCE_TAG_LOCK_FD="${SOURCE_TAG_LOCK_FD}"
    fi
}

if [[ "${DRY_RUN}" -eq 0 && -z "${SMOKE_EPISODES}" && \
      ( -z "${EPISODE_LIMIT}" || "${ADAPTER_PARITY_AUDIT}" -eq 1 ) ]]; then
    claim_or_verify_source_tag_lock
fi

case "${SPLIT}" in
    val_seen|val_unseen|test|train) ;;
    all)
        for ordered_split in val_seen val_unseen test; do
            child_args=(
                "${SETTING}" "${ordered_split}" "${GPU}"
                --run-tag "${RUN_TAG}"
            )
            if [[ "${SETTING}" == "etpnav-r2r-ce" || "${SETTING}" == "bevbert-r2r-ce" ]]; then
                child_args+=(--ce-data-version "${CE_DATA_VERSION}")
            fi
            [[ "${DRY_RUN}" -eq 0 ]] || child_args+=(--dry-run)
            "${BASH_SOURCE[0]}" "${child_args[@]}"
        done
        exit 0
        ;;
    *) die "invalid split: ${SPLIT}" ;;
esac

DATA_ROOT="${REPO_ROOT}/vln/data"
CHECKPOINT_ROOT="${REPO_ROOT}/vln/checkpoints"
BERT_BASE_UNCASED_ROOT="${CHECKPOINT_ROOT}/duet/.hf_cache/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
if [[ -n "${SMOKE_EPISODES}" ]]; then
    RESULT_ROOT="${REPO_ROOT}/vln/results/smoke/${RUN_TAG}/${SETTING}/${SPLIT}"
elif [[ -n "${TTA_CONFIG}" && "${TTA_NAMESPACE}" == "adapter_parity_audit" ]]; then
    RESULT_ROOT="${REPO_ROOT}/vln/results/audits/adapter_parity/runs/${RUN_TAG}/${SETTING}/${SPLIT}"
elif [[ "${RESULT_ROOT_SET}" -eq 1 ]]; then
    RESULT_ROOT="$(realpath -m -- "${RESULT_ROOT_OVERRIDE}")"
    if [[ "${TTA_NAMESPACE}" == "idea_source_statistics" ]]; then
        case "${RESULT_ROOT}" in
            "${REPO_ROOT}/vln/results/idea_source_statistics/"*) ;;
            *) die "IDEA collection result root must stay inside vln/results/idea_source_statistics" ;;
        esac
    else
        case "${RESULT_ROOT}" in
            "${REPO_ROOT}/vln/results/tuning/"*) ;;
            *) die "--result-root must stay inside vln/results/tuning" ;;
        esac
    fi
    case "${RESULT_ROOT}" in
        */"${RUN_TAG}"/"${SPLIT}") ;;
        *) die "--result-root must end with RUN_TAG/SPLIT" ;;
    esac
elif [[ -n "${TTA_CONFIG}" ]]; then
    if [[ "${TTA_NAMESPACE}" == "idea_source_statistics" ]]; then
        RESULT_ROOT="${REPO_ROOT}/vln/results/idea_source_statistics/${RUN_TAG}/${SETTING}/${SPLIT}"
    else
        RESULT_ROOT="${REPO_ROOT}/vln/results/tuning/${RUN_TAG}/${SETTING}/${SPLIT}"
    fi
else
    RESULT_ROOT="${REPO_ROOT}/vln/results/source/${RUN_TAG}/${SETTING}/${SPLIT}"
fi
if [[ "${IDEA_SOURCE_COLLECTION}" -eq 1 ]]; then
    CANONICAL_IDEA_SOURCE_STATS_OUTPUT="$(realpath -m -- "${IDEA_SOURCE_STATS_OUTPUT}")"
    [[ "${IDEA_SOURCE_STATS_OUTPUT}" == "${CANONICAL_IDEA_SOURCE_STATS_OUTPUT}" ]] || \
        die "IDEA source_stats_output must be a canonical absolute path"
    case "${IDEA_SOURCE_STATS_OUTPUT}" in
        "${RESULT_ROOT}/"*) ;;
        *) die "IDEA source_stats_output must stay inside the collection result root" ;;
    esac
    [[ ! -e "${IDEA_SOURCE_STATS_OUTPUT}" ]] || \
        die "refusing to overwrite IDEA source-statistics artifact"
fi
MATTERSIM_ROOT="${DATA_ROOT}/simulators/Matterport3DSimulator"
MATTERSIM_BUILD="${MATTERSIM_ROOT}/build"
MATTERSIM_NATIVE_LIB="${ENV_ROOT}/mattersim-native/lib"
case "${SETTING}" in
    duet-r2r|hamt-r2r) ORDER_FAMILY=r2r_duet_hamt ;;
    duet-reverie|hamt-reverie) ORDER_FAMILY=reverie_duet_hamt ;;
    goat-r2r) ORDER_FAMILY=r2r_goat ;;
    goat-reverie) ORDER_FAMILY=reverie_goat ;;
    etpnav-r2r-ce|bevbert-r2r-ce) ORDER_FAMILY=r2r_ce_v1_3_unified ;;
    *) ORDER_FAMILY="" ;;
esac

if [[ "${DRY_RUN}" -eq 0 ]]; then
    case "${SETTING}" in
        duet-r2r|duet-reverie|hamt-r2r|hamt-reverie|goat-r2r|goat-reverie)
            "${REPO_ROOT}/vln/scripts/build_mattersim.sh" --check
            ;;
    esac
fi

manifest_for_family() {
    local family="$1"
    if [[ "${ORDER_SEED_SET}" -eq 1 && "${ORDER_SEED}" != "0" ]]; then
        printf '%s/vln/manifests/episode_order/order_seed_%s/%s' \
            "${REPO_ROOT}" "${ORDER_SEED}" "${family}"
    else
        printf '%s/vln/manifests/episode_order/%s' "${REPO_ROOT}" "${family}"
    fi
}

R2R_DUET_HAMT_MANIFEST="$(manifest_for_family r2r_duet_hamt)"
REVERIE_DUET_HAMT_MANIFEST="$(manifest_for_family reverie_duet_hamt)"
R2R_GOAT_MANIFEST="$(manifest_for_family r2r_goat)"
REVERIE_GOAT_MANIFEST="$(manifest_for_family reverie_goat)"
R2R_CE_UNIFIED_MANIFEST="$(manifest_for_family r2r_ce_v1_3_unified)"
if [[ "${IDEA_SOURCE_COLLECTION}" -eq 1 ]]; then
    R2R_DUET_HAMT_MANIFEST="${SOURCE_ORDER_MANIFEST}"
    REVERIE_DUET_HAMT_MANIFEST="${SOURCE_ORDER_MANIFEST}"
    R2R_GOAT_MANIFEST="${SOURCE_ORDER_MANIFEST}"
    REVERIE_GOAT_MANIFEST="${SOURCE_ORDER_MANIFEST}"
    R2R_CE_UNIFIED_MANIFEST="${SOURCE_ORDER_MANIFEST}"
    "${BOOTSTRAP_PYTHON}" - \
        "${SOURCE_ORDER_MANIFEST}/train.json" "${SETTING}" \
        "${IDEA_SOURCE_DATASET}" "${IDEA_SOURCE_DATASET_VERSION}" <<'PY'
import hashlib
import json
import re
import sys

manifest_path, setting, configured_dataset, configured_version = sys.argv[1:]
with open(manifest_path, "r", encoding="utf-8") as stream:
    manifest = json.load(stream)
expected_benchmarks = {
    "duet-r2r": "r2r_discrete_duet_hamt",
    "hamt-r2r": "r2r_discrete_duet_hamt",
    "goat-r2r": "r2r_discrete_goat",
    "duet-reverie": "reverie_discrete_duet_hamt",
    "hamt-reverie": "reverie_discrete_duet_hamt",
    "goat-reverie": "reverie_discrete_goat",
    "etpnav-r2r-ce": "r2r_ce_v1_3_unified_etpnav_bevbert",
    "bevbert-r2r-ce": "r2r_ce_v1_3_unified_etpnav_bevbert",
}
if manifest.get("schema") != "navtta.episode_order.v1":
    raise SystemExit("invalid IDEA source-order schema")
if manifest.get("split") != "train" or manifest.get("split_ordinal") != -1:
    raise SystemExit("IDEA source-order manifest must describe train")
if manifest.get("benchmark") != expected_benchmarks[setting]:
    raise SystemExit("IDEA source-order benchmark does not match the setting")
episodes = manifest.get("episodes")
if type(manifest.get("episode_count")) is not int or manifest["episode_count"] != 128:
    raise SystemExit("IDEA source-order manifest must contain exactly 128 episodes")
if not isinstance(episodes, list) or len(episodes) != 128:
    raise SystemExit("IDEA source-order episode list is incomplete")
keys = [(str(item.get("scene_id")), str(item.get("episode_id"))) for item in episodes]
if len(set(keys)) != 128:
    raise SystemExit("IDEA source-order manifest contains duplicate episodes")
encoded = json.dumps(
    [{"episode_id": episode, "scene_id": scene} for scene, episode in keys],
    sort_keys=True, separators=(",", ":"), ensure_ascii=False,
).encode("utf-8")
if hashlib.sha256(encoded).hexdigest() != manifest.get("order_sha256"):
    raise SystemExit("IDEA source-order digest is invalid")
dataset = manifest.get("dataset", {})
if dataset.get("path") != configured_dataset:
    raise SystemExit("IDEA collection config and source-order dataset differ")
if configured_version != "sha256:" + str(dataset.get("sha256", "")):
    raise SystemExit("IDEA source_dataset_version must pin the dataset SHA256")
selection = manifest.get("selection", {})
if (
    selection.get("algorithm") != "sha256_rank_without_replacement_v1"
    or selection.get("setting") != setting
    or selection.get("sample_count") != 128
):
    raise SystemExit("IDEA source-order selection provenance is invalid")
if not re.fullmatch(r"[0-9a-f]{64}", str(dataset.get("sha256", ""))):
    raise SystemExit("IDEA source-order dataset SHA256 is invalid")
PY
fi
if [[ "${ORDER_SEED_SET}" -eq 1 && "${ORDER_SEED}" != "0" ]]; then
    [[ -n "${ORDER_FAMILY}" ]] || die "cannot resolve order-manifest family"
    "${BOOTSTRAP_PYTHON}" \
        "${REPO_ROOT}/vln/scripts/build_order_seed_manifests.py" \
        --check --family "${ORDER_FAMILY}" --order-seed "${ORDER_SEED}" \
        --split "${SPLIT}" || \
        die "tracked order-seed manifest failed deterministic verification"
fi
if [[ "${CE_DATA_VERSION}" == "v1.3-unified" ]]; then
    CE_MANIFEST="${R2R_CE_UNIFIED_MANIFEST}"
    CE_DATA_TEMPLATE="${DATA_ROOT}/etpnav/datasets/R2R_VLNCE_v1-3_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz"
    CE_TEST_DATA="${DATA_ROOT}/etpnav/datasets/R2R_VLNCE_v1-3_preprocessed_BERTidx/test/test_bertidx.json.gz"
    CE_BENCHMARK=r2r_ce_v1_3_unified_etpnav_bevbert
else
    CE_MANIFEST="${REPO_ROOT}/vln/manifests/episode_order/r2r_ce_v1_2"
    CE_DATA_TEMPLATE="${DATA_ROOT}/etpnav/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz"
    CE_TEST_DATA="${DATA_ROOT}/etpnav/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/test/test_bertidx.json.gz"
    CE_BENCHMARK=r2r_ce_v1_2_etpnav_bevbert
fi
STREAM_MANIFEST="${REPO_ROOT}/vln/manifests/episode_order/r2r_vlnce_v1_3"
CLIP_CACHE="${CACHE_ROOT}/clip"
DISCRETE_SUBMISSION_CANONICALIZER="${REPO_ROOT}/vln/scripts/canonicalize_discrete_submission.py"
DISCRETE_SCANVP_CANDIDATES="${DATA_ROOT}/goat/R2R/annotations/scanvp_candview_relangles.json"

mkdir -p "${HOME_ROOT}" "${XDG_CACHE_ROOT}"
export HOME="${HOME_ROOT}"
export XDG_CACHE_HOME="${XDG_CACHE_ROOT}"
export HF_HOME="${CHECKPOINT_ROOT}/duet/.hf_cache"
export HF_HUB_CACHE="${CHECKPOINT_ROOT}/duet/.hf_cache/hub"
export TRANSFORMERS_CACHE="${CHECKPOINT_ROOT}/duet/.hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="${CACHE_ROOT}/nltk_data"
export NAVTTA_CLIP_CACHE="${CLIP_CACHE}"
export NAVTTA_BERT_BASE_UNCASED="${BERT_BASE_UNCASED_ROOT}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES="${GPU}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
unset PYTHONPATH
unset NAVTTA_SMOKE_EPISODES
if [[ -n "${SMOKE_EPISODES}" ]]; then
    export NAVTTA_SMOKE_EPISODES="${SMOKE_EPISODES}"
elif [[ -n "${EPISODE_LIMIT}" ]]; then
    export NAVTTA_SMOKE_EPISODES="${EPISODE_LIMIT}"
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ -d "${RESULT_ROOT}" ]] && find "${RESULT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
        die "output is not empty; use a new run directory: ${RESULT_ROOT}"
    fi
    mkdir -p "${RESULT_ROOT}"
    CONSOLE_LOG="${RESULT_ROOT}/console.log"
    # Mirror diagnostics without placing the evaluated command behind a
    # pipeline, so its original exit status still reaches `set -e`.
    exec > >(tee -a "${CONSOLE_LOG}") 2>&1
    printf 'console log: %s\n' "${CONSOLE_LOG}"
fi

RUN_ORDER_DIR=""
RUN_PRIMARY_CHECKPOINT=""
set_run_context() {
    RUN_ORDER_DIR="$1"
    RUN_PRIMARY_CHECKPOINT="$2"
}

run_in() {
    local workdir="$1"
    shift
    local -a evaluated_command=("$@")
    local -a tta_args=()
    if [[ -n "${TTA_CONFIG}" ]]; then
        mapfile -d '' -t tta_args < <(
            "${PYTHON}" "${TTA_TRANSLATOR}" --setting "${SETTING}" \
                --config "${TTA_CONFIG}" \
                --diagnostics "${RESULT_ROOT}/tta_diagnostics.json" --nul
        )
        [[ "${#tta_args[@]}" -gt 0 ]] || die "TTA config produced no model arguments"
        evaluated_command+=("${tta_args[@]}")
    fi
    printf 'workdir: %s\ncommand:' "${workdir}"
    printf ' %q' "${evaluated_command[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        (cd "${workdir}" && "${evaluated_command[@]}")
    fi
}

select_env() {
    local name="$1"
    ENV_PREFIX="${ENV_ROOT}/${name}"
    PYTHON="${ENV_PREFIX}/bin/python"
    [[ -x "${PYTHON}" ]] || die "missing environment Python: ${PYTHON}"
    export CONDA_PREFIX="${ENV_PREFIX}"
    export PATH="${ENV_PREFIX}/bin:${PATH}"
    export LD_LIBRARY_PATH="${ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

assert_gpu() {
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
    fi
}

append_submit_flag() {
    if [[ "${SPLIT}" == "test" || -n "${SMOKE_EPISODES}" || \
          "${TTA_NAMESPACE}" == "adapter_parity_audit" ]]; then
        COMMAND+=(--submit)
    fi
}

validate_discrete_output() {
    local task="$1"
    local submission="$2"
    local manifest="$3"
    local benchmark="$4"
    local dataset="$5"
    if [[ "${SPLIT}" == "test" && "${DRY_RUN}" -eq 0 ]]; then
        "${REPO_ROOT}/vln/scripts/validate_discrete_submission.py" \
            --task "${task}" --submission "${submission}" \
            --manifest "${manifest}/test.json" \
            --expected-benchmark "${benchmark}" --dataset "${dataset}"
    fi
}

canonicalize_discrete_output() {
    local task="$1"
    local submission="$2"
    local dataset="$3"
    if [[ "${SPLIT}" == "test" && "${DRY_RUN}" -eq 0 ]]; then
        [[ -f "${DISCRETE_SUBMISSION_CANONICALIZER}" ]] || \
            die "missing discrete submission canonicalizer"
        [[ -f "${DISCRETE_SCANVP_CANDIDATES}" ]] || \
            die "missing discrete viewpoint candidate map"
        "${PYTHON}" "${DISCRETE_SUBMISSION_CANONICALIZER}" \
            --task "${task}" --submission "${submission}" \
            --output "${submission}" --dataset "${dataset}" \
            --scanvp-candidates "${DISCRETE_SCANVP_CANDIDATES}"
    fi
}

case "${SETTING}" in
    duet-r2r)
        select_env duet
        export LD_LIBRARY_PATH="${MATTERSIM_BUILD}:${MATTERSIM_NATIVE_LIB}:${LD_LIBRARY_PATH}"
        assert_gpu
        export PYTHONPATH="${MATTERSIM_BUILD}:${REPO_ROOT}/vln/baselines/duet/map_nav_src"
        COMMAND=(
            "${PYTHON}" r2r/main_nav.py
            --root_dir ../datasets --dataset r2r --output_dir "${RESULT_ROOT}"
            --world_size 1 --seed "${MODEL_SEED}" --tokenizer bert --enc_full_graph
            --graph_sprels --fusion dynamic --expert_policy spl --train_alg dagger
            --num_l_layers 9 --num_x_layers 4 --num_pano_layers 2
            --max_action_len 15 --max_instr_len 200 --batch_size 1
            --features vitbase --image_feat_size 768 --angle_feat_size 4
            --ml_weight 0.2 --feat_dropout 0.4 --dropout 0.5 --gamma 0
            --resume_file ../datasets/R2R/trained_models/best_val_unseen
            --strict_checkpoint_keys
            --test --eval_splits "${SPLIT}"
            --episode_order_manifest "${R2R_DUET_HAMT_MANIFEST}"
        )
        append_submit_flag
        set_run_context "${R2R_DUET_HAMT_MANIFEST}" \
            "${CHECKPOINT_ROOT}/duet/R2R/best_val_unseen"
        run_in "${REPO_ROOT}/vln/baselines/duet/map_nav_src" "${COMMAND[@]}"
        canonicalize_discrete_output r2r \
            "${RESULT_ROOT}/preds/submit_test.json" \
            "${DATA_ROOT}/duet/R2R/annotations/R2R_test_enc.json"
        validate_discrete_output r2r \
            "${RESULT_ROOT}/preds/submit_test.json" \
            "${R2R_DUET_HAMT_MANIFEST}" \
            r2r_discrete_duet_hamt \
            "${DATA_ROOT}/duet/R2R/annotations/R2R_test_enc.json"
        ;;
    duet-reverie)
        select_env duet
        export LD_LIBRARY_PATH="${MATTERSIM_BUILD}:${MATTERSIM_NATIVE_LIB}:${LD_LIBRARY_PATH}"
        assert_gpu
        export PYTHONPATH="${MATTERSIM_BUILD}:${REPO_ROOT}/vln/baselines/duet/map_nav_src"
        COMMAND=(
            "${PYTHON}" reverie/main_nav_obj.py
            --root_dir ../datasets --dataset reverie --output_dir "${RESULT_ROOT}"
            --world_size 1 --seed "${MODEL_SEED}" --tokenizer bert --enc_full_graph
            --graph_sprels --fusion dynamic --multi_endpoints --dagger_sample sample
            --train_alg dagger --num_l_layers 9 --num_x_layers 4
            --num_pano_layers 2 --max_action_len 15 --max_instr_len 200
            --max_objects 20 --batch_size 1 --features vitbase
            --obj_features vitbase --image_feat_size 768 --obj_feat_size 768
            --angle_feat_size 4 --ml_weight 0.2 --feat_dropout 0.4
            --dropout 0.5 --gamma 0
            --resume_file ../datasets/REVERIE/trained_models/best_val_unseen
            --strict_checkpoint_keys
            --test --eval_splits "${SPLIT}"
            --episode_order_manifest "${REVERIE_DUET_HAMT_MANIFEST}"
        )
        append_submit_flag
        set_run_context "${REVERIE_DUET_HAMT_MANIFEST}" \
            "${CHECKPOINT_ROOT}/duet/REVERIE/best_val_unseen"
        run_in "${REPO_ROOT}/vln/baselines/duet/map_nav_src" "${COMMAND[@]}"
        canonicalize_discrete_output reverie \
            "${RESULT_ROOT}/preds/submit_test_dynamic.json" \
            "${DATA_ROOT}/duet/REVERIE/annotations/REVERIE_test_enc.json"
        validate_discrete_output reverie \
            "${RESULT_ROOT}/preds/submit_test_dynamic.json" \
            "${REVERIE_DUET_HAMT_MANIFEST}" \
            reverie_discrete_duet_hamt \
            "${DATA_ROOT}/duet/REVERIE/annotations/REVERIE_test_enc.json"
        ;;
    hamt-r2r)
        select_env hamt
        export LD_LIBRARY_PATH="${MATTERSIM_BUILD}:${MATTERSIM_NATIVE_LIB}:${LD_LIBRARY_PATH}"
        export HF_HOME="${CACHE_ROOT}/transformers/hamt"
        export HF_HUB_CACHE="${CACHE_ROOT}/transformers/hamt"
        export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers/hamt"
        assert_gpu
        export PYTHONPATH="${MATTERSIM_BUILD}:${REPO_ROOT}/vln/baselines/hamt/finetune_src"
        COMMAND=(
            "${PYTHON}" r2r/main.py
            --root_dir ../datasets --dataset r2r --output_dir "${RESULT_ROOT}"
            --world_size 1 --seed "${MODEL_SEED}" --tokenizer bert --ob_type pano
            --num_l_layers 9 --num_x_layers 4 --hist_enc_pano
            --hist_pano_num_layers 2 --fix_lang_embedding --fix_hist_embedding
            --features vitbase_r2rfte2e --feedback sample --max_action_len 15
            --max_action_steps 50 --max_instr_len 60 --image_feat_size 768
            --angle_feat_size 4
            --batch_size 1 --ml_weight 0.2 --feat_dropout 0.4 --dropout 0.5
            --resume_file ../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen
            --strict_checkpoint_keys
            --test --eval_splits "${SPLIT}"
            --episode_order_manifest "${R2R_DUET_HAMT_MANIFEST}"
        )
        append_submit_flag
        set_run_context "${R2R_DUET_HAMT_MANIFEST}" \
            "${CHECKPOINT_ROOT}/hamt/R2R/vitbase-finetune-e2e/best_val_unseen"
        run_in "${REPO_ROOT}/vln/baselines/hamt/finetune_src" "${COMMAND[@]}"
        validate_discrete_output r2r \
            "${RESULT_ROOT}/preds/submit_test.json" \
            "${R2R_DUET_HAMT_MANIFEST}" \
            r2r_discrete_duet_hamt \
            "${DATA_ROOT}/hamt/R2R/annotations/R2R_test_enc.json"
        ;;
    hamt-reverie)
        select_env hamt
        export LD_LIBRARY_PATH="${MATTERSIM_BUILD}:${MATTERSIM_NATIVE_LIB}:${LD_LIBRARY_PATH}"
        export HF_HOME="${CACHE_ROOT}/transformers/hamt"
        export HF_HUB_CACHE="${CACHE_ROOT}/transformers/hamt"
        export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers/hamt"
        assert_gpu
        export PYTHONPATH="${MATTERSIM_BUILD}:${REPO_ROOT}/vln/baselines/hamt/finetune_src"
        COMMAND=(
            "${PYTHON}" reverie/main_navref.py
            --root_dir ../datasets --dataset reverie --output_dir "${RESULT_ROOT}"
            --world_size 1 --seed "${MODEL_SEED}" --tokenizer bert --multi_endpoints
            --ob_type pano --num_l_layers 9 --num_x_layers 4 --hist_enc_pano
            --hist_pano_num_layers 2 --no_lang_ca --features vitbase_r2rfte2e
            --feedback sample --max_action_len 15 --max_instr_len 60
            --image_feat_size 768 --obj_feat_size 768 --angle_feat_size 4
            --batch_size 1 --ml_weight 0.2 --feat_dropout 0.4 --dropout 0.5
            --resume_file ../datasets/REVERIE/trained_models/best_val_unseen
            --strict_checkpoint_keys
            --test --eval_splits "${SPLIT}"
            --episode_order_manifest "${REVERIE_DUET_HAMT_MANIFEST}"
        )
        append_submit_flag
        set_run_context "${REVERIE_DUET_HAMT_MANIFEST}" \
            "${CHECKPOINT_ROOT}/hamt/REVERIE/best_val_unseen"
        run_in "${REPO_ROOT}/vln/baselines/hamt/finetune_src" "${COMMAND[@]}"
        validate_discrete_output reverie \
            "${RESULT_ROOT}/preds/submit_test.json" \
            "${REVERIE_DUET_HAMT_MANIFEST}" \
            reverie_discrete_duet_hamt \
            "${DATA_ROOT}/hamt/REVERIE/annotations/REVERIE_test_enc.json"
        ;;
    goat-r2r|goat-reverie)
        select_env goat
        export LD_LIBRARY_PATH="${MATTERSIM_BUILD}:${MATTERSIM_NATIVE_LIB}:${LD_LIBRARY_PATH}"
        assert_gpu
        export PYTHONPATH="${MATTERSIM_BUILD}:${REPO_ROOT}/vln/baselines/goat/map_nav_src"
        if [[ "${SETTING}" == "goat-r2r" ]]; then
            ENTRY=r2r/main_nav.py
            DATASET=r2r
            TASK_ROOT=R2R
            NAME=goat_r2r_source
            MANIFEST="${R2R_GOAT_MANIFEST}"
            RESUME=../datasets/R2R/navigator/goat_r2r/ckpts/best_val_unseen.pt
            BACKDOOR=../datasets/R2R/navigator/goat_r2r/logs/backdoor/backdoor_update_features.tsv
            FRONTDOOR=../datasets/R2R/navigator/goat_r2r/logs/frontdoor/frontdoor_update_features.tsv
            TASK_ARGS=(--expert_policy spl --max_instr_len 200)
        else
            ENTRY=reverie/main_nav_obj.py
            DATASET=reverie
            TASK_ROOT=REVERIE
            NAME=goat_reverie_source
            MANIFEST="${REVERIE_GOAT_MANIFEST}"
            RESUME=../datasets/REVERIE/navigator/goat_reverie/ckpts/best_val_unseen.pt
            BACKDOOR=../datasets/REVERIE/navigator/goat_reverie/logs/backdoor/backdoor_update_features.tsv
            FRONTDOOR=../datasets/REVERIE/navigator/goat_reverie/logs/frontdoor/frontdoor_update_features.tsv
            TASK_ARGS=(--multi_endpoints --dagger_sample sample --max_instr_len 80 --max_objects 20 --obj_features vitbase --obj_feat_size 768 --z_instr_update)
        fi
        COMMAND=(
            "${PYTHON}" "${ENTRY}"
            --root_dir ../datasets --dataset "${DATASET}"
            --output_dir "${RESULT_ROOT}" --world_size 1 --seed "${MODEL_SEED}"
            --tokenizer roberta --mode valid --name "${NAME}"
            --enc_full_graph --graph_sprels --fusion dynamic --train_alg dagger
            --num_l_layers 6 --num_x_layers 3 --num_pano_layers 2
            --max_action_len 15 --batch_size 1 --features clip768
            --image_feat_size 768 --angle_feat_size 4 --ml_weight 0.2
            --feat_dropout 0 --dropout 0 --do_back_txt --do_back_img
            --do_back_txt_type type_2 --do_back_img_type type_1 --do_add_method door
            --do_front_txt --do_front_img --do_front_his
            --backdoor_dict_file "${BACKDOOR}" --frontdoor_dict_file "${FRONTDOOR}"
            --resume_file "${RESUME}" --strict_checkpoint_keys
            --test --eval_splits "${SPLIT}"
            --episode_order_manifest "${MANIFEST}" "${TASK_ARGS[@]}"
        )
        append_submit_flag
        set_run_context "${MANIFEST}" \
            "${CHECKPOINT_ROOT}/goat/${TASK_ROOT}/best_val_unseen.pt"
        run_in "${REPO_ROOT}/vln/baselines/goat/map_nav_src" "${COMMAND[@]}"
        if [[ "${SETTING}" == "goat-r2r" ]]; then
            canonicalize_discrete_output r2r \
                "${RESULT_ROOT}/test/${NAME}/preds/submit_test.json" \
                "${DATA_ROOT}/goat/R2R/annotations/R2R_test_roberta_enc.json"
            validate_discrete_output r2r \
                "${RESULT_ROOT}/test/${NAME}/preds/submit_test.json" \
                "${MANIFEST}" r2r_discrete_goat \
                "${DATA_ROOT}/goat/R2R/annotations/R2R_test_roberta_enc.json"
        else
            canonicalize_discrete_output reverie \
                "${RESULT_ROOT}/test/${NAME}/preds/submit_test_dynamic.json" \
                "${DATA_ROOT}/goat/REVERIE/annotations/REVERIE_test_roberta_enc.json"
            validate_discrete_output reverie \
                "${RESULT_ROOT}/test/${NAME}/preds/submit_test_dynamic.json" \
                "${MANIFEST}" reverie_discrete_goat \
                "${DATA_ROOT}/goat/REVERIE/annotations/REVERIE_test_roberta_enc.json"
        fi
        ;;
    etpnav-r2r-ce|bevbert-r2r-ce)
        select_env vlnce017
        export HF_HOME="${CACHE_ROOT}/huggingface"
        export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
        export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
        export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
        export PYTORCH_PRETRAINED_BERT_CACHE="${CACHE_ROOT}/pytorch_pretrained_bert"
        assert_gpu
        export PYTHONPATH="${REPO_ROOT}/core"
        if [[ "${SETTING}" == "etpnav-r2r-ce" ]]; then
            WORKDIR="${REPO_ROOT}/vln/baselines/etpnav"
            CHECKPOINT="${CHECKPOINT_ROOT}/etpnav/ckpt.iter12000.pth"
        else
            WORKDIR="${REPO_ROOT}/vln/baselines/bevbert/bevbert_ce"
            CHECKPOINT="${CHECKPOINT_ROOT}/bevbert/ckpt.iter9600.pth"
        fi
        # CE loaders require EPISODE_COUNT to be -1 (or the full stream size).
        # Development prefixes are applied after the episode-order input is
        # loaded, so passing the prefix here would fail during construction.
        CE_EPISODE_COUNT=-1
        COMMON=(
            SIMULATOR_GPU_IDS '[0]' TORCH_GPU_ID 0 TORCH_GPU_IDS '[0]'
            GPU_NUMBERS 1 NUM_ENVIRONMENTS 1
            TASK_CONFIG.SEED "${MODEL_SEED}"
            TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
            TASK_CONFIG.DATASET.DATA_PATH "${CE_DATA_TEMPLATE}"
            MODEL.pretrained_path None
            IL.back_algo control TENSORBOARD_DIR "${RESULT_ROOT}/tensorboard/"
            CHECKPOINT_FOLDER "${RESULT_ROOT}/checkpoints/"
            RESULTS_DIR "${RESULT_ROOT}/metrics/" VIDEO_DIR "${RESULT_ROOT}/videos/"
            LOG_FILE "${RESULT_ROOT}/run.log"
        )
        if [[ "${IDEA_SOURCE_COLLECTION}" -eq 1 ]]; then
            # IDEA statistics need observations and policy rollouts, not
            # validation labels.  The inference path reads episode IDs from
            # DATA_PATH and avoids the upstream v1.2 train_gt dependency.
            COMMAND=(
                "${PYTHON}" run.py --exp_name idea_source_statistics --run-type inference
                --exp-config run_r2r/iter_train.yaml "${COMMON[@]}"
                INFERENCE.SPLIT train INFERENCE.CKPT_PATH "${CHECKPOINT}"
                INFERENCE.PREDICTIONS_FILE "${RESULT_ROOT}/source_rollouts.json"
                INFERENCE.EPISODE_COUNT "${CE_EPISODE_COUNT}"
                INFERENCE.EPISODE_ORDER_MANIFEST "${CE_MANIFEST}"
            )
        elif [[ "${SPLIT}" == "test" ]]; then
            COMMAND=(
                "${PYTHON}" run.py --exp_name source_test --run-type inference
                --exp-config run_r2r/iter_train.yaml "${COMMON[@]}"
                INFERENCE.SPLIT test INFERENCE.CKPT_PATH "${CHECKPOINT}"
                INFERENCE.PREDICTIONS_FILE "${RESULT_ROOT}/predictions.json"
                INFERENCE.EPISODE_COUNT "${CE_EPISODE_COUNT}"
                INFERENCE.EPISODE_ORDER_MANIFEST "${CE_MANIFEST}"
            )
        else
            COMMAND=(
                "${PYTHON}" run.py --exp_name "source_${SPLIT}" --run-type eval
                --exp-config run_r2r/iter_train.yaml "${COMMON[@]}"
                EVAL.SPLIT "${SPLIT}" EVAL.CKPT_PATH_DIR "${CHECKPOINT}"
                EVAL.EPISODE_COUNT "${CE_EPISODE_COUNT}" EVAL.EPISODE_ORDER_MANIFEST "${CE_MANIFEST}"
            )
        fi
        set_run_context "${CE_MANIFEST}" "${CHECKPOINT}"
        run_in "${WORKDIR}" "${COMMAND[@]}"
        if [[ "${SPLIT}" == "test" && "${DRY_RUN}" -eq 0 ]]; then
            "${REPO_ROOT}/vln/scripts/validate_r2r_ce_submission.py" \
                --submission "${RESULT_ROOT}/predictions.json" \
                --manifest "${CE_MANIFEST}/test.json" \
                --expected-benchmark "${CE_BENCHMARK}" \
                --dataset "${CE_TEST_DATA}"
        fi
        ;;
    streamvln-r2r-ce)
        select_env streamvln
        export HF_HOME="${CACHE_ROOT}/huggingface"
        export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
        export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
        assert_gpu
        WORKDIR="${REPO_ROOT}/vln/baselines/streamvln"
        MODEL_DIR="${CHECKPOINT_ROOT}/streamvln/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
        VISION_TOWER="${CACHE_ROOT}/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
        PORT="$((21000 + GPU))"
        COMMAND=(
            "${PYTHON}" -m torch.distributed.run --nproc_per_node=1
            --master_port "${PORT}" streamvln/streamvln_eval.py
            --model_path "${MODEL_DIR}" --habitat_config_path config/vln_r2r.yaml
            --vision_tower_path "${VISION_TOWER}"
            --eval_split "${SPLIT}" --output_path "${RESULT_ROOT}"
            --num_future_steps 4 --num_frames 32 --num_history 8
            --seed 0
            --world_size 1 --rank 0 --gpu 0 --device cuda
            --strict_checkpoint_keys
            --episode_order_manifest "${STREAM_MANIFEST}"
        )
        if [[ "${SPLIT}" == "test" ]]; then
            COMMAND+=(--submission_file "${RESULT_ROOT}/predictions.json")
        fi
        if [[ -n "${SMOKE_EPISODES}" ]]; then
            COMMAND+=(--smoke-episodes "${SMOKE_EPISODES}")
        fi
        set_run_context "${STREAM_MANIFEST}" \
            "${MODEL_DIR}/model.safetensors.index.json"
        run_in "${WORKDIR}" "${COMMAND[@]}"
        if [[ "${SPLIT}" == "test" && "${DRY_RUN}" -eq 0 ]]; then
            "${REPO_ROOT}/vln/scripts/validate_r2r_ce_submission.py" \
                --submission "${RESULT_ROOT}/predictions.json" \
                --manifest "${STREAM_MANIFEST}/test.json" \
                --expected-benchmark r2r_vlnce_v1_3_streamvln \
                --dataset "${DATA_ROOT}/datasets/r2r/test/test.json.gz"
        fi
        ;;
    *)
        die "invalid setting: ${SETTING}"
        ;;
esac

if [[ "${IDEA_SOURCE_COLLECTION}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
    [[ -f "${IDEA_SOURCE_STATS_OUTPUT}" ]] || \
        die "IDEA source collection did not produce its statistics artifact"
    [[ -f "${RESULT_ROOT}/tta_diagnostics.json" ]] || \
        die "IDEA source collection did not produce TTA diagnostics"
    IDEA_SOURCE_MODEL="${SETTING%%-*}"
    ACTUAL_SOURCE_CHECKPOINT_SHA256="$(
        sha256sum "${RUN_PRIMARY_CHECKPOINT}" | awk '{print $1}'
    )"
    "${PYTHON}" - \
        "${IDEA_SOURCE_STATS_OUTPUT}" \
        "${RESULT_ROOT}/tta_diagnostics.json" \
        "${SOURCE_ORDER_MANIFEST}/train.json" \
        "${SETTING}" "${IDEA_SOURCE_MODEL}" \
        "${ACTUAL_SOURCE_CHECKPOINT_SHA256}" \
        "${IDEA_SOURCE_DATASET}" "${IDEA_SOURCE_DATASET_VERSION}" \
        "${IDEA_SOURCE_COLLECTION_POLICY}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    artifact_name, diagnostics_name, order_name, setting, model,
    checkpoint_sha256, dataset_name, dataset_version, collection_policy,
) = sys.argv[1:]
artifact_path = Path(artifact_name).resolve()
diagnostics_path = Path(diagnostics_name).resolve()
with artifact_path.open("r", encoding="utf-8") as stream:
    artifact = json.load(stream)
with diagnostics_path.open("r", encoding="utf-8") as stream:
    diagnostics = json.load(stream)
with open(order_name, "r", encoding="utf-8") as stream:
    order = json.load(stream)

digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
if diagnostics.get("schema") == "navtta.vln_discrete_tta.v1":
    collection = diagnostics.get("adapter")
elif diagnostics.get("schema") == "navtta.vln_ce_tta.v1":
    collection = diagnostics.get("idea_source_collection")
else:
    raise SystemExit("IDEA collection diagnostics use an unknown schema")
if not isinstance(collection, dict):
    raise SystemExit("IDEA collection diagnostics are missing")
expected_collection = {
    "mode": "idea_source_statistics_collection",
    "collection_scope": "all_steps_all_valid_tokens",
    "trajectory_count": 128,
    "expected_trajectory_count": 128,
    "complete": True,
    "output_path": str(artifact_path),
    "artifact_sha256": digest,
}
for key, expected in expected_collection.items():
    actual = collection.get(key)
    if key == "output_path":
        actual = str(Path(str(actual)).resolve())
    if actual != expected:
        raise SystemExit(
            "IDEA collection diagnostic {} mismatch: {!r} != {!r}"
            .format(key, actual, expected)
        )
if diagnostics.get("method") != "idea" or diagnostics.get("episode_count") != 128:
    raise SystemExit("IDEA collection diagnostics do not account for 128 episodes")

provenance = artifact.get("provenance")
if not isinstance(provenance, dict):
    raise SystemExit("IDEA artifact provenance is missing")
# The session diagnostics intentionally contain the configured provenance;
# generated trajectory fields are added only while serialising the artifact.
configured = collection.get("provenance")
expected_configured = {
    key: provenance.get(key) for key in (
        "checkpoint_sha256", "dataset", "dataset_version", "split",
        "model", "setting",
        "collection_policy",
    )
}
if configured != expected_configured:
    raise SystemExit("IDEA collection provenance changed during serialization")
expected_provenance = {
    "checkpoint_sha256": checkpoint_sha256,
    "dataset": dataset_name,
    "dataset_version": dataset_version,
    "split": "train",
    "model": model,
    "setting": setting,
    "trajectory_count": 128,
    "collection_scope": "all_steps_all_valid_tokens",
    "collection_policy": collection_policy,
}
for key, expected in expected_provenance.items():
    if provenance.get(key) != expected:
        raise SystemExit(
            "IDEA artifact provenance {} mismatch: {!r} != {!r}"
            .format(key, provenance.get(key), expected)
        )
expected_ids = sorted(str(item["episode_id"]) for item in order["episodes"])
if provenance.get("trajectory_ids") != expected_ids:
    raise SystemExit("IDEA artifact trajectory IDs differ from the train-128 manifest")
ids_bytes = json.dumps(
    expected_ids, ensure_ascii=False, separators=(",", ":")
).encode("utf-8")
if hashlib.sha256(ids_bytes).hexdigest() != provenance.get("trajectory_ids_sha256"):
    raise SystemExit("IDEA artifact trajectory-ID digest is invalid")
if (
    artifact.get("schema") != "navtta.idea.source_statistics"
    or artifact.get("version") != 1
    or artifact.get("moment_estimator")
    != "global_sum_sumsq_count_sample_std"
    or type(artifact.get("feature_dim")) is not int
    or artifact["feature_dim"] <= 0
    or type(artifact.get("num_layers")) is not int
    or artifact["num_layers"] <= 0
    or not isinstance(artifact.get("layers"), list)
    or len(artifact["layers"]) != artifact["num_layers"]
    or any(type(layer.get("count")) is not int or layer["count"] <= 0
           for layer in artifact["layers"])
):
    raise SystemExit("IDEA source-statistics artifact payload is incomplete")
print("IDEA source statistics validated: {}".format(digest))
PY
fi

if [[ -n "${SMOKE_EPISODES}" && "${DRY_RUN}" -eq 0 ]]; then
    "${PYTHON}" "${REPO_ROOT}/vln/scripts/validate_smoke_output.py" \
        --setting "${SETTING}" --result-root "${RESULT_ROOT}" \
        --manifest "${RUN_ORDER_DIR}/val_seen.json" \
        --expected-count "${SMOKE_EPISODES}"
fi
