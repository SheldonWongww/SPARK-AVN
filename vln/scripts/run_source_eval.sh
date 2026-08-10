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
                                      [--episode-limit N] [--dry-run]

SETTING:
  duet-r2r duet-reverie hamt-r2r hamt-reverie goat-r2r goat-reverie
  etpnav-r2r-ce bevbert-r2r-ce streamvln-r2r-ce

SPLIT is val_seen, val_unseen, test, or all.  "all" always runs in the
canonical val_seen -> val_unseen -> test order, using a fresh process and
output directory for each split.  Test produces submission trajectories; it
does not produce a local test score.

VERSION controls ETPNav/BEVBert only: v1.3-unified (default, comparable to
StreamVLN start states) or v1.2-native (paper/upstream reproduction).

--smoke-episodes N is a non-formal GPU lifecycle check.  It is restricted to
val_seen, uses the canonical prefix, and writes under vln/results/smoke/.

--tta-config FILE enables a provenance-recorded TTA/control job described by
one JSON file.  --episode-limit N is a non-formal development prefix used by
the hyperparameter scheduler; complete finalist jobs omit it.
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
EPISODE_LIMIT=""
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
        --episode-limit)
            [[ "$#" -ge 2 ]] || die "--episode-limit requires a value"
            [[ -z "${EPISODE_LIMIT}" ]] || die "episode limit specified more than once"
            [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "--episode-limit must be a positive integer"
            EPISODE_LIMIT="$2"
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
    etpnav-r2r-ce|bevbert-r2r-ce)
        RUN_DATA_VERSION="${CE_DATA_VERSION}"
        ;;
    streamvln-r2r-ce)
        [[ "${CE_DATA_VERSION_SET}" -eq 0 && "${CE_DATA_VERSION}" == "v1.3-unified" ]] || \
            die "--ce-data-version applies only to ETPNav/BEVBert; StreamVLN is fixed to v1.3"
        RUN_DATA_VERSION="v1.3"
        ;;
    *)
        [[ "${CE_DATA_VERSION_SET}" -eq 0 && "${CE_DATA_VERSION}" == "v1.3-unified" ]] || \
            die "--ce-data-version applies only to ETPNav/BEVBert"
        RUN_DATA_VERSION="native"
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${REPO_ROOT}" in
    /root/autodl-tmp/*) ;;
    *) die "refusing to run outside /root/autodl-tmp: ${REPO_ROOT}" ;;
esac

TTA_METHOD=source
TTA_TRANSLATOR="${REPO_ROOT}/vln/scripts/tta_config_cli.py"
if [[ -n "${TTA_CONFIG}" ]]; then
    [[ "${TTA_CONFIG}" = /* ]] || TTA_CONFIG="${REPO_ROOT}/${TTA_CONFIG}"
    [[ -f "${TTA_CONFIG}" ]] || die "missing TTA config: ${TTA_CONFIG}"
    [[ -f "${TTA_TRANSLATOR}" ]] || die "missing TTA config translator"
    if ! TTA_METHOD="$(
        python3 "${TTA_TRANSLATOR}" --setting "${SETTING}" \
            --config "${TTA_CONFIG}" --diagnostics /tmp/navtta-unused.json \
            --print-method
    )"; then
        die "invalid TTA config: ${TTA_CONFIG}"
    fi
fi

SOURCE_TAG_LOCK_FD=""
SOURCE_TAG_LOCK_ROOT="/root/autodl-tmp/tmp/navtta-source-tag-locks"
SOURCE_TAG_LOCK_FILE="${SOURCE_TAG_LOCK_ROOT}/${RUN_TAG}.lock"

claim_or_verify_source_tag_lock() {
    local inherited_fd
    local inherited_path
    command -v flock >/dev/null 2>&1 || die "flock is required for formal runs"
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

if [[ "${DRY_RUN}" -eq 0 && -z "${SMOKE_EPISODES}" && -z "${EPISODE_LIMIT}" ]]; then
    claim_or_verify_source_tag_lock
fi

case "${SPLIT}" in
    val_seen|val_unseen|test) ;;
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
if [[ -n "${SMOKE_EPISODES}" ]]; then
    RESULT_ROOT="${REPO_ROOT}/vln/results/smoke/${RUN_TAG}/${SETTING}/${SPLIT}"
elif [[ -n "${TTA_CONFIG}" ]]; then
    RESULT_ROOT="${REPO_ROOT}/vln/results/tuning/${RUN_TAG}/${SETTING}/${SPLIT}"
else
    RESULT_ROOT="${REPO_ROOT}/vln/results/source/${RUN_TAG}/${SETTING}/${SPLIT}"
fi
MATTERSIM_ROOT="${DATA_ROOT}/simulators/Matterport3DSimulator"
MATTERSIM_MODULE="${MATTERSIM_ROOT}/build/MatterSim.cpython-38-x86_64-linux-gnu.so"
if [[ "${CE_DATA_VERSION}" == "v1.3-unified" ]]; then
    CE_MANIFEST="${REPO_ROOT}/vln/manifests/episode_order/r2r_ce_v1_3_unified"
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
CLIP_CACHE="/root/autodl-tmp/cache/clip"
DISCRETE_SUBMISSION_CANONICALIZER="${REPO_ROOT}/vln/scripts/canonicalize_discrete_submission.py"
DISCRETE_SCANVP_CANDIDATES="${DATA_ROOT}/goat/R2R/annotations/scanvp_candview_relangles.json"

export HOME="/root/autodl-tmp"
export XDG_CACHE_HOME="/root/autodl-tmp/.cache"
export HF_HOME="${CHECKPOINT_ROOT}/duet/.hf_cache"
export HF_HUB_CACHE="${CHECKPOINT_ROOT}/duet/.hf_cache/hub"
export TRANSFORMERS_CACHE="${CHECKPOINT_ROOT}/duet/.hf_cache/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="/root/autodl-tmp/cache/nltk_data"
export NAVTTA_CLIP_CACHE="${CLIP_CACHE}"
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

FORMAL_EXECUTION_PATHS=(
    core tools vln/baselines vln/navtta_vln vln/scripts
    vln/experiments vln/manifests
)

check_formal_git_state() {
    local expected_commit="$1"
    local current_commit tracked_changes untracked_files

    if ! current_commit="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD 2>&1)"; then
        printf 'error: git rev-parse failed: %s\n' "${current_commit}" >&2
        return 1
    fi
    if [[ ! "${current_commit}" =~ ^[0-9a-f]{40}$ ]]; then
        printf 'error: Git HEAD is not a full commit: %s\n' "${current_commit}" >&2
        return 1
    fi
    if [[ -n "${expected_commit}" && "${current_commit}" != "${expected_commit}" ]]; then
        printf 'error: Git HEAD changed during the formal run: expected %s, got %s\n' \
            "${expected_commit}" "${current_commit}" >&2
        return 1
    fi
    if ! tracked_changes="$(
        git -C "${REPO_ROOT}" status --porcelain --untracked-files=no 2>&1
    )"; then
        printf 'error: git status failed: %s\n' "${tracked_changes}" >&2
        return 1
    fi
    if [[ -n "${tracked_changes}" ]]; then
        printf 'error: tracked worktree changes detected during formal run:\n%s\n' \
            "${tracked_changes}" >&2
        return 1
    fi
    if ! untracked_files="$(
        git -C "${REPO_ROOT}" ls-files --others --exclude-standard -- \
            "${FORMAL_EXECUTION_PATHS[@]}" 2>&1
    )"; then
        printf 'error: git ls-files failed: %s\n' "${untracked_files}" >&2
        return 1
    fi
    if [[ -n "${untracked_files}" ]]; then
        printf 'error: untracked execution files detected during formal run:\n%s\n' \
            "${untracked_files}" >&2
        return 1
    fi
}

FORMAL_RUN=0
if [[ "${DRY_RUN}" -eq 0 && -z "${SMOKE_EPISODES}" && -z "${EPISODE_LIMIT}" ]]; then
    FORMAL_RUN=1
    if ! RUN_GIT_COMMIT="$(
        git -C "${REPO_ROOT}" rev-parse --verify HEAD 2>&1
    )"; then
        die "cannot resolve Git HEAD: ${RUN_GIT_COMMIT}"
    fi
    check_formal_git_state "${RUN_GIT_COMMIT}" || \
        die "formal runs require an unchanged, clean execution tree"
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
    if [[ -d "${RESULT_ROOT}" ]] && find "${RESULT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
        die "output is not empty; use a new run directory: ${RESULT_ROOT}"
    fi
    mkdir -p "${RESULT_ROOT}"
    CONSOLE_LOG="${RESULT_ROOT}/console.log"
    # Mirror every subsequent diagnostic without placing the evaluated command
    # behind a pipeline.  Its original exit status therefore reaches `set -e`
    # and the formal-manifest EXIT trap unchanged; `pipefail` remains active for
    # the runner's own validation pipelines.
    exec > >(tee -a "${CONSOLE_LOG}") 2>&1
    printf 'console log: %s\n' "${CONSOLE_LOG}"
fi

RUN_MODEL=""
RUN_ORDER_DIR=""
RUN_PRIMARY_CHECKPOINT=""
RUN_CONFIG_REF=""
RUN_AUX_CHECKPOINTS=()
RUN_MANIFEST_ACTIVE=0
RUN_MANIFEST_PATH=""

set_run_identity() {
    RUN_MODEL="$1"
    RUN_ORDER_DIR="$2"
    RUN_PRIMARY_CHECKPOINT="$3"
    RUN_CONFIG_REF="$4"
    shift 4
    RUN_AUX_CHECKPOINTS=("$@")
    case "${RUN_MODEL}" in
        duet|hamt|goat)
            # The discrete evaluators import this ignored native module at
            # runtime.  Hash the exact binary in every formal run manifest.
            RUN_AUX_CHECKPOINTS+=("mattersim_python=${MATTERSIM_MODULE}")
            ;;
    esac
}

validate_run_identity_paths() {
    [[ -d "${RUN_ORDER_DIR}" ]] || die "missing episode-order directory: ${RUN_ORDER_DIR}"
    [[ -f "${RUN_ORDER_DIR}/${SPLIT}.json" ]] || \
        die "missing episode-order manifest for ${SPLIT}"
    [[ -f "${RUN_PRIMARY_CHECKPOINT}" ]] || \
        die "missing primary checkpoint: ${RUN_PRIMARY_CHECKPOINT}"
    if [[ "${RUN_CONFIG_REF}" != *'#'* ]]; then
        local config_path="${RUN_CONFIG_REF}"
        [[ "${config_path}" = /* ]] || config_path="${REPO_ROOT}/${config_path}"
        [[ -f "${config_path}" ]] || die "missing run configuration: ${config_path}"
    fi
    local item name path
    for item in "${RUN_AUX_CHECKPOINTS[@]}"; do
        [[ "${item}" == *=* ]] || die "invalid auxiliary checkpoint mapping: ${item}"
        name="${item%%=*}"
        path="${item#*=}"
        [[ -n "${name}" && -f "${path}" ]] || \
            die "missing auxiliary checkpoint ${name}: ${path}"
    done
}

finalize_formal_manifest() {
    local status=$?
    local finalize_status=0
    local validate_status=0
    local artifact relative
    local -a artifact_args=()
    trap - EXIT
    set +e
    if [[ "${RUN_MANIFEST_ACTIVE}" -eq 1 ]]; then
        if [[ "${status}" -eq 0 ]] && \
           ! check_formal_git_state "${RUN_GIT_COMMIT}"; then
            printf 'error: refusing to mark a run successful after Git state changed\n' >&2
            status=1
        fi
        if [[ "${status}" -eq 0 ]]; then
            while IFS= read -r -d '' artifact; do
                relative="${artifact#${RESULT_ROOT}/}"
                artifact_args+=(--artifact "${relative}=${artifact}")
            done < <(
                find "${RESULT_ROOT}" -type f \
                    \( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' \) \
                    -print0 | sort -z
            )
            if [[ "${#artifact_args[@]}" -eq 0 ]]; then
                printf 'error: formal run produced no compact result artifacts\n' >&2
                status=1
            fi
        fi
        "${PYTHON}" "${REPO_ROOT}/tools/finalize_run_manifest.py" \
            --manifest "${RUN_MANIFEST_PATH}" --exit-code "${status}" \
            "${artifact_args[@]}"
        finalize_status=$?
        if [[ "${status}" -eq 0 && "${finalize_status}" -eq 0 ]]; then
            "${PYTHON}" "${REPO_ROOT}/tools/validate_run_manifest.py" \
                --manifest "${RUN_MANIFEST_PATH}" --task vln \
                --benchmark "${RUN_BENCHMARK}" --run-tag "${RUN_TAG}" \
                --model "${RUN_MODEL}" --method "${TTA_METHOD}" \
                --source-setting "${RUN_SOURCE_SETTING}" --seed 0 \
                --git-commit "${RUN_GIT_COMMIT}" \
                --checkpoint-sha256 "${RUN_PRIMARY_SHA256}" \
                --stream-order-sha256 "${RUN_ORDER_SHA256}" \
                --stream-content-sha256 "${RUN_DATASET_SHA256}" \
                --require-immutable-identity \
                --require-result-artifacts
            validate_status=$?
        fi
        if [[ "${status}" -eq 0 && "${finalize_status}" -ne 0 ]]; then
            status=${finalize_status}
        elif [[ "${status}" -eq 0 && "${validate_status}" -ne 0 ]]; then
            status=${validate_status}
            "${PYTHON}" "${REPO_ROOT}/tools/finalize_run_manifest.py" \
                --manifest "${RUN_MANIFEST_PATH}" --exit-code "${status}" >/dev/null 2>&1
        fi
    fi
    exit "${status}"
}

prepare_formal_manifest() {
    [[ "${FORMAL_RUN}" -eq 1 ]] || return 0
    [[ -n "${RUN_MODEL}" && -n "${RUN_ORDER_DIR}" && \
       -n "${RUN_PRIMARY_CHECKPOINT}" && -n "${RUN_CONFIG_REF}" ]] || \
        die "formal run identity is incomplete"

    local order_file="${RUN_ORDER_DIR}/${SPLIT}.json"
    [[ -f "${order_file}" ]] || die "missing episode-order manifest: ${order_file}"
    local -a order_metadata
    mapfile -t order_metadata < <(
        "${PYTHON}" - "${order_file}" <<'PY'
import json
import sys
with open(sys.argv[1], "r") as stream:
    document = json.load(stream)
print(document["benchmark"])
print(document["dataset"]["path"])
print(document["dataset"]["sha256"])
print(document["order_sha256"])
PY
    )
    [[ "${#order_metadata[@]}" -eq 4 ]] || die "invalid episode-order metadata"
    RUN_BENCHMARK="${order_metadata[0]}"
    local dataset_path="${order_metadata[1]}"
    RUN_DATASET_SHA256="${order_metadata[2]}"
    RUN_ORDER_SHA256="${order_metadata[3]}"
    [[ "${dataset_path}" = /* ]] || dataset_path="${REPO_ROOT}/${dataset_path}"
    [[ -f "${dataset_path}" ]] || die "missing manifest dataset: ${dataset_path}"
    local actual_dataset_sha256
    actual_dataset_sha256="$(sha256sum "${dataset_path}" | awk '{print $1}')"
    [[ "${actual_dataset_sha256}" == "${RUN_DATASET_SHA256}" ]] || \
        die "episode-order dataset SHA256 mismatch"
    [[ -f "${RUN_PRIMARY_CHECKPOINT}" ]] || \
        die "missing primary checkpoint: ${RUN_PRIMARY_CHECKPOINT}"
    RUN_PRIMARY_SHA256="$(sha256sum "${RUN_PRIMARY_CHECKPOINT}" | awk '{print $1}')"
    RUN_SOURCE_SETTING="${SETTING}:${SPLIT}:${RUN_DATA_VERSION}:${TTA_METHOD}"
    local run_id="${RUN_TAG}-${SETTING}-${SPLIT}-${RUN_DATA_VERSION}"
    local run_dir="${REPO_ROOT}/vln/results/runs/${run_id}"
    mkdir -p "${REPO_ROOT}/vln/results/runs"
    mkdir "${run_dir}" || die "run manifest directory already exists: ${run_dir}"
    RUN_MANIFEST_PATH="${run_dir}/manifest.json"

    local auxiliary_args=()
    local item name path
    for item in "${RUN_AUX_CHECKPOINTS[@]}"; do
        [[ "${item}" == *=* ]] || die "invalid auxiliary checkpoint mapping: ${item}"
        name="${item%%=*}"
        path="${item#*=}"
        [[ -f "${path}" ]] || die "missing auxiliary checkpoint ${name}: ${path}"
        auxiliary_args+=(--aux-checkpoint "${name}=${path}")
    done

    "${PYTHON}" "${REPO_ROOT}/tools/create_run_manifest.py" \
        --output "${RUN_MANIFEST_PATH}" --run-id "${run_id}" \
        --task vln --benchmark "${RUN_BENCHMARK}" --model "${RUN_MODEL}" \
        --method "${TTA_METHOD}" --run-tag "${RUN_TAG}" \
        --source-setting "${RUN_SOURCE_SETTING}" --seed 0 \
        --config "${RUN_CONFIG_REF}" --checkpoint "${RUN_PRIMARY_CHECKPOINT}" \
        "${auxiliary_args[@]}" --dataset "${dataset_path}" \
        --dataset-version "${RUN_BENCHMARK}" \
        --stream-order-sha256 "${RUN_ORDER_SHA256}" \
        --stream-content-sha256 "${RUN_DATASET_SHA256}" \
        --asset-manifest "${REPO_ROOT}/vln/manifests/assets/eval_assets.json" \
        --environment-manifest "${REPO_ROOT}/vln/manifests/environments/eval_environments.json" \
        --episode-order-manifest "${order_file}" --extra "$@"
    RUN_MANIFEST_ACTIVE=1
    trap finalize_formal_manifest EXIT
}

run_in() {
    local workdir="$1"
    shift
    local -a evaluated_command=("$@")
    local -a tta_args=()
    if [[ -n "${TTA_CONFIG}" ]]; then
        mapfile -d '' -t tta_args < <(
            python3 "${TTA_TRANSLATOR}" --setting "${SETTING}" \
                --config "${TTA_CONFIG}" \
                --diagnostics "${RESULT_ROOT}/tta_diagnostics.json" --nul
        )
        [[ "${#tta_args[@]}" -gt 0 ]] || die "TTA config produced no model arguments"
        evaluated_command+=("${tta_args[@]}")
        RUN_CONFIG_REF="${TTA_CONFIG}"
    fi
    validate_run_identity_paths
    printf 'workdir: %s\ncommand:' "${workdir}"
    printf ' %q' "${evaluated_command[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        prepare_formal_manifest "${evaluated_command[@]}"
        (cd "${workdir}" && "${evaluated_command[@]}")
    fi
}

select_env() {
    local name="$1"
    ENV_PREFIX="/root/autodl-tmp/conda/envs/${name}"
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
    if [[ "${SPLIT}" == "test" || -n "${SMOKE_EPISODES}" ]]; then
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
        assert_gpu
        export PYTHONPATH="${MATTERSIM_ROOT}/build:${REPO_ROOT}/vln/baselines/duet/map_nav_src"
        COMMAND=(
            "${PYTHON}" r2r/main_nav.py
            --root_dir ../datasets --dataset r2r --output_dir "${RESULT_ROOT}"
            --world_size 1 --seed 0 --tokenizer bert --enc_full_graph
            --graph_sprels --fusion dynamic --expert_policy spl --train_alg dagger
            --num_l_layers 9 --num_x_layers 4 --num_pano_layers 2
            --max_action_len 15 --max_instr_len 200 --batch_size 1
            --features vitbase --image_feat_size 768 --angle_feat_size 4
            --ml_weight 0.2 --feat_dropout 0.4 --dropout 0.5 --gamma 0
            --resume_file ../datasets/R2R/trained_models/best_val_unseen
            --strict_checkpoint_keys
            --test --eval_splits "${SPLIT}"
            --episode_order_manifest "${REPO_ROOT}/vln/manifests/episode_order/r2r_duet_hamt"
        )
        append_submit_flag
        set_run_identity duet "${REPO_ROOT}/vln/manifests/episode_order/r2r_duet_hamt" \
            "${CHECKPOINT_ROOT}/duet/R2R/best_val_unseen" \
            "vln/scripts/run_source_eval.sh#duet-r2r" \
            "pano_features=${DATA_ROOT}/duet/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5" \
            "submission_viewpoint_candidates=${DISCRETE_SCANVP_CANDIDATES}"
        run_in "${REPO_ROOT}/vln/baselines/duet/map_nav_src" "${COMMAND[@]}"
        canonicalize_discrete_output r2r \
            "${RESULT_ROOT}/preds/submit_test.json" \
            "${DATA_ROOT}/duet/R2R/annotations/R2R_test_enc.json"
        validate_discrete_output r2r \
            "${RESULT_ROOT}/preds/submit_test.json" \
            "${REPO_ROOT}/vln/manifests/episode_order/r2r_duet_hamt" \
            r2r_discrete_duet_hamt \
            "${DATA_ROOT}/duet/R2R/annotations/R2R_test_enc.json"
        ;;
    duet-reverie)
        select_env duet
        assert_gpu
        export PYTHONPATH="${MATTERSIM_ROOT}/build:${REPO_ROOT}/vln/baselines/duet/map_nav_src"
        COMMAND=(
            "${PYTHON}" reverie/main_nav_obj.py
            --root_dir ../datasets --dataset reverie --output_dir "${RESULT_ROOT}"
            --world_size 1 --seed 0 --tokenizer bert --enc_full_graph
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
            --episode_order_manifest "${REPO_ROOT}/vln/manifests/episode_order/reverie_duet_hamt"
        )
        append_submit_flag
        set_run_identity duet "${REPO_ROOT}/vln/manifests/episode_order/reverie_duet_hamt" \
            "${CHECKPOINT_ROOT}/duet/REVERIE/best_val_unseen" \
            "vln/scripts/run_source_eval.sh#duet-reverie" \
            "pano_features=${DATA_ROOT}/duet/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5" \
            "object_features=${DATA_ROOT}/duet/REVERIE/features/obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5" \
            "object_boxes=${DATA_ROOT}/duet/REVERIE/annotations/BBoxes.json" \
            "submission_viewpoint_candidates=${DISCRETE_SCANVP_CANDIDATES}"
        run_in "${REPO_ROOT}/vln/baselines/duet/map_nav_src" "${COMMAND[@]}"
        canonicalize_discrete_output reverie \
            "${RESULT_ROOT}/preds/submit_test_dynamic.json" \
            "${DATA_ROOT}/duet/REVERIE/annotations/REVERIE_test_enc.json"
        validate_discrete_output reverie \
            "${RESULT_ROOT}/preds/submit_test_dynamic.json" \
            "${REPO_ROOT}/vln/manifests/episode_order/reverie_duet_hamt" \
            reverie_discrete_duet_hamt \
            "${DATA_ROOT}/duet/REVERIE/annotations/REVERIE_test_enc.json"
        ;;
    hamt-r2r)
        select_env hamt
        export HF_HOME="/root/autodl-tmp/cache/transformers/hamt"
        export HF_HUB_CACHE="/root/autodl-tmp/cache/transformers/hamt"
        export TRANSFORMERS_CACHE="/root/autodl-tmp/cache/transformers/hamt"
        assert_gpu
        export PYTHONPATH="${MATTERSIM_ROOT}/build:${REPO_ROOT}/vln/baselines/hamt/finetune_src"
        COMMAND=(
            "${PYTHON}" r2r/main.py
            --root_dir ../datasets --dataset r2r --output_dir "${RESULT_ROOT}"
            --world_size 1 --seed 0 --tokenizer bert --ob_type pano
            --num_l_layers 9 --num_x_layers 4 --hist_enc_pano
            --hist_pano_num_layers 2 --fix_lang_embedding --fix_hist_embedding
            --features vitbase_r2rfte2e --feedback sample --max_action_len 15
            --max_action_steps 50 --max_instr_len 60 --image_feat_size 768
            --angle_feat_size 4
            --batch_size 1 --ml_weight 0.2 --feat_dropout 0.4 --dropout 0.5
            --resume_file ../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen
            --strict_checkpoint_keys
            --test --eval_splits "${SPLIT}"
            --episode_order_manifest "${REPO_ROOT}/vln/manifests/episode_order/r2r_duet_hamt"
        )
        append_submit_flag
        set_run_identity hamt "${REPO_ROOT}/vln/manifests/episode_order/r2r_duet_hamt" \
            "${CHECKPOINT_ROOT}/hamt/R2R/vitbase-finetune-e2e/best_val_unseen" \
            "vln/scripts/run_source_eval.sh#hamt-r2r-e2e" \
            "pano_features=${DATA_ROOT}/hamt/R2R/features/pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5"
        run_in "${REPO_ROOT}/vln/baselines/hamt/finetune_src" "${COMMAND[@]}"
        validate_discrete_output r2r \
            "${RESULT_ROOT}/preds/submit_test.json" \
            "${REPO_ROOT}/vln/manifests/episode_order/r2r_duet_hamt" \
            r2r_discrete_duet_hamt \
            "${DATA_ROOT}/hamt/R2R/annotations/R2R_test_enc.json"
        ;;
    hamt-reverie)
        select_env hamt
        export HF_HOME="/root/autodl-tmp/cache/transformers/hamt"
        export HF_HUB_CACHE="/root/autodl-tmp/cache/transformers/hamt"
        export TRANSFORMERS_CACHE="/root/autodl-tmp/cache/transformers/hamt"
        assert_gpu
        export PYTHONPATH="${MATTERSIM_ROOT}/build:${REPO_ROOT}/vln/baselines/hamt/finetune_src"
        COMMAND=(
            "${PYTHON}" reverie/main_navref.py
            --root_dir ../datasets --dataset reverie --output_dir "${RESULT_ROOT}"
            --world_size 1 --seed 0 --tokenizer bert --multi_endpoints
            --ob_type pano --num_l_layers 9 --num_x_layers 4 --hist_enc_pano
            --hist_pano_num_layers 2 --no_lang_ca --features vitbase_r2rfte2e
            --feedback sample --max_action_len 15 --max_instr_len 60
            --image_feat_size 768 --obj_feat_size 768 --angle_feat_size 4
            --batch_size 1 --ml_weight 0.2 --feat_dropout 0.4 --dropout 0.5
            --resume_file ../datasets/REVERIE/trained_models/best_val_unseen
            --strict_checkpoint_keys
            --test --eval_splits "${SPLIT}"
            --episode_order_manifest "${REPO_ROOT}/vln/manifests/episode_order/reverie_duet_hamt"
        )
        append_submit_flag
        set_run_identity hamt "${REPO_ROOT}/vln/manifests/episode_order/reverie_duet_hamt" \
            "${CHECKPOINT_ROOT}/hamt/REVERIE/best_val_unseen" \
            "vln/scripts/run_source_eval.sh#hamt-reverie" \
            "pano_features=${DATA_ROOT}/hamt/R2R/features/pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5" \
            "object_features=${DATA_ROOT}/hamt/REVERIE/features/obj_pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5" \
            "object_boxes=${DATA_ROOT}/hamt/REVERIE/annotations/BBoxes.json"
        run_in "${REPO_ROOT}/vln/baselines/hamt/finetune_src" "${COMMAND[@]}"
        validate_discrete_output reverie \
            "${RESULT_ROOT}/preds/submit_test.json" \
            "${REPO_ROOT}/vln/manifests/episode_order/reverie_duet_hamt" \
            reverie_discrete_duet_hamt \
            "${DATA_ROOT}/hamt/REVERIE/annotations/REVERIE_test_enc.json"
        ;;
    goat-r2r|goat-reverie)
        select_env goat
        assert_gpu
        export PYTHONPATH="${MATTERSIM_ROOT}/build:${REPO_ROOT}/vln/baselines/goat/map_nav_src"
        if [[ "${SETTING}" == "goat-r2r" ]]; then
            ENTRY=r2r/main_nav.py
            DATASET=r2r
            TASK_ROOT=R2R
            NAME=goat_r2r_source
            MANIFEST="${REPO_ROOT}/vln/manifests/episode_order/r2r_goat"
            RESUME=../datasets/R2R/navigator/goat_r2r/ckpts/best_val_unseen.pt
            BACKDOOR=../datasets/R2R/navigator/goat_r2r/logs/backdoor/backdoor_update_features.tsv
            FRONTDOOR=../datasets/R2R/navigator/goat_r2r/logs/frontdoor/frontdoor_update_features.tsv
            TASK_ARGS=(--expert_policy spl --max_instr_len 200)
        else
            ENTRY=reverie/main_nav_obj.py
            DATASET=reverie
            TASK_ROOT=REVERIE
            NAME=goat_reverie_source
            MANIFEST="${REPO_ROOT}/vln/manifests/episode_order/reverie_goat"
            RESUME=../datasets/REVERIE/navigator/goat_reverie/ckpts/best_val_unseen.pt
            BACKDOOR=../datasets/REVERIE/navigator/goat_reverie/logs/backdoor/backdoor_update_features.tsv
            FRONTDOOR=../datasets/REVERIE/navigator/goat_reverie/logs/frontdoor/frontdoor_update_features.tsv
            TASK_ARGS=(--multi_endpoints --dagger_sample sample --max_instr_len 80 --max_objects 20 --obj_features vitbase --obj_feat_size 768 --z_instr_update)
        fi
        COMMAND=(
            "${PYTHON}" "${ENTRY}"
            --root_dir ../datasets --dataset "${DATASET}"
            --output_dir "${RESULT_ROOT}" --world_size 1 --seed 0
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
        set_run_identity goat "${MANIFEST}" "${CHECKPOINT_ROOT}/goat/${TASK_ROOT}/best_val_unseen.pt" \
            "vln/scripts/run_source_eval.sh#${SETTING}" \
            "pano_features=${DATA_ROOT}/goat/R2R/features/CLIP-ViT-B-16-views.hdf5" \
            "backdoor=${BACKDOOR#../datasets/}" \
            "frontdoor=${FRONTDOOR#../datasets/}" \
            "submission_viewpoint_candidates=${DISCRETE_SCANVP_CANDIDATES}"
        RUN_AUX_CHECKPOINTS[1]="backdoor=${DATA_ROOT}/goat/${RUN_AUX_CHECKPOINTS[1]#backdoor=}"
        RUN_AUX_CHECKPOINTS[2]="frontdoor=${DATA_ROOT}/goat/${RUN_AUX_CHECKPOINTS[2]#frontdoor=}"
        if [[ "${SETTING}" == "goat-reverie" ]]; then
            RUN_AUX_CHECKPOINTS+=(
                "object_features=${DATA_ROOT}/goat/REVERIE/features/obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5"
                "object_boxes=${DATA_ROOT}/goat/REVERIE/annotations/BBoxes.json"
            )
        fi
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
        export HF_HOME="/root/autodl-tmp/cache/huggingface"
        export HF_HUB_CACHE="/root/autodl-tmp/cache/huggingface/hub"
        export TRANSFORMERS_CACHE="/root/autodl-tmp/cache/huggingface/transformers"
        export HF_DATASETS_CACHE="/root/autodl-tmp/cache/huggingface/datasets"
        export PYTORCH_PRETRAINED_BERT_CACHE="/root/autodl-tmp/cache/pytorch_pretrained_bert"
        assert_gpu
        export PYTHONPATH="${REPO_ROOT}/core"
        if [[ "${SETTING}" == "etpnav-r2r-ce" ]]; then
            WORKDIR="${REPO_ROOT}/vln/baselines/etpnav"
            CHECKPOINT="${CHECKPOINT_ROOT}/etpnav/ckpt.iter12000.pth"
        else
            WORKDIR="${REPO_ROOT}/vln/baselines/bevbert/bevbert_ce"
            CHECKPOINT="${CHECKPOINT_ROOT}/bevbert/ckpt.iter9600.pth"
        fi
        # Canonical CE loaders require EPISODE_COUNT to be -1 (or the full
        # manifest count). Development prefixes are applied by
        # NAVTTA_SMOKE_EPISODES after the manifest is validated, so passing the
        # prefix here would be rejected before environment construction.
        CE_EPISODE_COUNT=-1
        COMMON=(
            SIMULATOR_GPU_IDS '[0]' TORCH_GPU_ID 0 TORCH_GPU_IDS '[0]'
            GPU_NUMBERS 1 NUM_ENVIRONMENTS 1
            TASK_CONFIG.SEED 0
            TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
            TASK_CONFIG.DATASET.DATA_PATH "${CE_DATA_TEMPLATE}"
            MODEL.pretrained_path None
            IL.back_algo control TENSORBOARD_DIR "${RESULT_ROOT}/tensorboard/"
            CHECKPOINT_FOLDER "${RESULT_ROOT}/checkpoints/"
            RESULTS_DIR "${RESULT_ROOT}/metrics/" VIDEO_DIR "${RESULT_ROOT}/videos/"
            LOG_FILE "${RESULT_ROOT}/run.log"
        )
        if [[ "${SPLIT}" == "test" ]]; then
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
        if [[ "${SETTING}" == "etpnav-r2r-ce" ]]; then
            set_run_identity etpnav "${CE_MANIFEST}" "${CHECKPOINT}" \
                "vln/baselines/etpnav/run_r2r/iter_train.yaml" \
                "waypoint_predictor=${DATA_ROOT}/etpnav/wp_pred/check_cwp_bestdist_hfov90" \
                "depth_encoder=${DATA_ROOT}/etpnav/ddppo-models/gibson-2plus-resnet50.pth" \
                "clip_vit_b32=${CLIP_CACHE}/ViT-B-32.pt"
        else
            set_run_identity bevbert "${CE_MANIFEST}" "${CHECKPOINT}" \
                "vln/baselines/bevbert/bevbert_ce/run_r2r/iter_train.yaml" \
                "waypoint_predictor=${DATA_ROOT}/etpnav/wp_pred/check_cwp_bestdist_hfov90" \
                "depth_encoder=${DATA_ROOT}/etpnav/ddppo-models/gibson-2plus-resnet50.pth" \
                "clip_vit_b16=${CLIP_CACHE}/ViT-B-16.pt"
        fi
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
        export HF_HOME="/root/autodl-tmp/cache/huggingface"
        export HF_HUB_CACHE="/root/autodl-tmp/cache/huggingface/hub"
        export TRANSFORMERS_CACHE="/root/autodl-tmp/cache/huggingface/transformers"
        assert_gpu
        WORKDIR="${REPO_ROOT}/vln/baselines/streamvln"
        MODEL_DIR="${CHECKPOINT_ROOT}/streamvln/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
        VISION_TOWER="/root/autodl-tmp/cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
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
        set_run_identity streamvln "${STREAM_MANIFEST}" \
            "${MODEL_DIR}/model.safetensors.index.json" \
            "vln/baselines/streamvln/config/vln_r2r.yaml" \
            "model_shard_1=${MODEL_DIR}/model-00001-of-00004.safetensors" \
            "model_shard_2=${MODEL_DIR}/model-00002-of-00004.safetensors" \
            "model_shard_3=${MODEL_DIR}/model-00003-of-00004.safetensors" \
            "model_shard_4=${MODEL_DIR}/model-00004-of-00004.safetensors" \
            "model_config=${MODEL_DIR}/config.json" \
            "generation_config=${MODEL_DIR}/generation_config.json" \
            "tokenizer=${MODEL_DIR}/tokenizer.json" \
            "tokenizer_config=${MODEL_DIR}/tokenizer_config.json" \
            "vocab=${MODEL_DIR}/vocab.json" \
            "merges=${MODEL_DIR}/merges.txt" \
            "vision_weights=${VISION_TOWER}/model.safetensors" \
            "vision_config=${VISION_TOWER}/config.json" \
            "vision_preprocessor=${VISION_TOWER}/preprocessor_config.json"
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

if [[ -n "${SMOKE_EPISODES}" && "${DRY_RUN}" -eq 0 ]]; then
    "${PYTHON}" "${REPO_ROOT}/vln/scripts/validate_smoke_output.py" \
        --setting "${SETTING}" --result-root "${RESULT_ROOT}" \
        --manifest "${RUN_ORDER_DIR}/val_seen.json" \
        --expected-count "${SMOKE_EPISODES}"
fi
