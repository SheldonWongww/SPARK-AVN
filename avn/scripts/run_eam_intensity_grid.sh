#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run the pre-registered 72-job EAM adaptation-intensity grid.

Cartesian product:
  LR:              1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5
  UPDATE_INTERVAL: 1, 4, 16
  SCOPE:           head_only, transformer_ln_plus_head,
                   decoder_plus_head, full_transformer_plus_head

Fixed controls:
  SMT+Audio, single_source, sample actions, a=0.4, M=32, K=8,
  EPISODIC=False, STEPS=1, Adam betas=(0.9,0.999), weight_decay=0,
  max_grad_norm=0, seed=0, 2000 episodes.

Usage:
  bash avn/scripts/run_eam_intensity_grid.sh [options]

Options:
  --seed N              Episode-order/evaluation seed (default: 0)
  --gpus LIST           One or more distinct physical GPU ids (default: 0,1,2,3)
  --jobs-per-gpu N      Concurrent jobs per GPU (default: 2; maximum: 8)
  --episodes N          Episodes per job (default: 2000; maximum: 2000)
  --batch-id ID         Stable batch id (default: UTC timestamp)
  --resume              Resume a batch; skip validated completed jobs
  --allow-dirty         Permit tracked worktree changes (not recommended)
  --smoke               Run only job 3 (full transformer, LR=1e-8, interval=1)
  --dry-run             Print the selected plan without launching
  -h, --help            Show this help

Recommended detached launch after a short resource smoke test:
  screen -dmS eam_intensity \
    bash avn/scripts/run_eam_intensity_grid.sh \
      --gpus 0,1,2,3 --jobs-per-gpu 2 \
      --batch-id eam-intensity-smt-single-v1-seed0

Follow progress:
  tail -f avn/results/logs/eam_intensity_grid/\
eam-intensity-smt-single-v1-seed0/scheduler.log

Resume the same immutable batch:
  bash avn/scripts/run_eam_intensity_grid.sh \
    --gpus 0,1,2,3 --jobs-per-gpu 2 \
    --batch-id eam-intensity-smt-single-v1-seed0 --resume

Single-GPU, two-episode full-scope smoke test:
  bash avn/scripts/run_eam_intensity_grid.sh \
    --gpus 3 --jobs-per-gpu 1 --episodes 2 --smoke \
    --batch-id eam-smoke-full-scope-v1-seed0

This is a development/tuning grid. Do not reuse its episode stream as an
untouched final-test stream after selecting a configuration.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    if [[ -n "${PREFLIGHT_LOG:-}" ]]; then
        printf 'error: %s\n' "$*" >> "${PREFLIGHT_LOG}"
    fi
    exit 2
}

SEED=0
GPU_CSV="0,1,2,3"
JOBS_PER_GPU=2
EPISODES=2000
BATCH_ID=""
BATCH_ID_GIVEN=0
RESUME=0
ALLOW_DIRTY=0
SMOKE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)
            [[ $# -ge 2 ]] || die "--seed requires a value"
            SEED="$2"
            shift 2
            ;;
        --gpus)
            [[ $# -ge 2 ]] || die "--gpus requires a value"
            GPU_CSV="$2"
            shift 2
            ;;
        --jobs-per-gpu)
            [[ $# -ge 2 ]] || die "--jobs-per-gpu requires a value"
            JOBS_PER_GPU="$2"
            shift 2
            ;;
        --episodes)
            [[ $# -ge 2 ]] || die "--episodes requires a value"
            EPISODES="$2"
            shift 2
            ;;
        --batch-id)
            [[ $# -ge 2 ]] || die "--batch-id requires a value"
            BATCH_ID="$2"
            BATCH_ID_GIVEN=1
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --allow-dirty)
            ALLOW_DIRTY=1
            shift
            ;;
        --smoke)
            SMOKE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ "${SEED}" =~ ^[0-9]+$ ]] || die "seed must be a nonnegative integer"
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || die "episodes must be positive"
[[ ${EPISODES} -le 2000 ]] || die "canonical stream contains only 2000 episodes"
[[ "${JOBS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || \
    die "jobs-per-gpu must be positive"
[[ ${JOBS_PER_GPU} -le 8 ]] || die "jobs-per-gpu must not exceed 8"
if [[ ${RESUME} -eq 1 && ${BATCH_ID_GIVEN} -eq 0 ]]; then
    die "--resume requires --batch-id"
fi
if [[ -z "${BATCH_ID}" ]]; then
    BATCH_ID="eam-intensity-smt-single-v1-seed${SEED}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
[[ "${BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "batch-id may contain only letters, numbers, dot, underscore, and hyphen"

[[ "${GPU_CSV}" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
    die "--gpus must be a comma-separated list of numeric GPU ids"
IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
[[ ${#GPUS[@]} -ge 1 ]] || die "--gpus must contain at least one GPU id"
for gpu in "${GPUS[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU id: ${gpu}"
done
for ((i = 0; i < ${#GPUS[@]}; i++)); do
    for ((j = i + 1; j < ${#GPUS[@]}; j++)); do
        [[ "${GPUS[$i]}" != "${GPUS[$j]}" ]] || \
            die "GPU ids must be distinct"
    done
done

LRS=("1e-8" "1e-7" "3e-7" "1e-6" "3e-6" "1e-5")
UPDATE_INTERVALS=(1 4 16)
SCOPES=(
    "head_only"
    "transformer_ln_plus_head"
    "decoder_plus_head"
    "full_transformer_plus_head"
)
FULL_GRID_JOBS=$((${#LRS[@]} * ${#UPDATE_INTERVALS[@]} * ${#SCOPES[@]}))
[[ ${FULL_GRID_JOBS} -eq 72 ]] || \
    die "internal grid-size error: ${FULL_GRID_JOBS}"
if [[ ${SMOKE} -eq 1 ]]; then
    EXPECTED_JOBS=1
else
    EXPECTED_JOBS=${FULL_GRID_JOBS}
fi

job_selected() {
    local job_index="$1"
    [[ ${SMOKE} -eq 0 || ${job_index} -eq 3 ]]
}

# Frozen method controls. Keep every value explicit so config-default changes
# cannot silently alter a resumed or future run.
CONFIDENCE_SCALE="0.4"
MEMORY_SIZE=32
BATCH_SIZE=8
PARAM_SCOPE="module_prefixes"
EPISODIC="False"
STEPS=1
OPTIMIZER="Adam"
BETA1="0.9"
BETA2="0.999"
WEIGHT_DECAY="0.0"
MAX_GRAD_NORM="0.0"
ACTION_SELECTION="sample"

scope_prefixes() {
    case "$1" in
        head_only)
            printf '%s\n' \
                '["action_distribution"]'
            ;;
        transformer_ln_plus_head)
            printf '%s\n' \
                '["net.smt_state_encoder.transformer.encoder.layers.0.norm1","net.smt_state_encoder.transformer.encoder.layers.0.norm2","net.smt_state_encoder.transformer.encoder.norm","net.smt_state_encoder.transformer.decoder.layers.0.norm1","net.smt_state_encoder.transformer.decoder.layers.0.norm2","net.smt_state_encoder.transformer.decoder.layers.0.norm3","net.smt_state_encoder.transformer.decoder.norm","action_distribution"]'
            ;;
        decoder_plus_head)
            printf '%s\n' \
                '["net.smt_state_encoder.transformer.decoder","action_distribution"]'
            ;;
        full_transformer_plus_head)
            printf '%s\n' \
                '["net.smt_state_encoder.transformer","action_distribution"]'
            ;;
        *)
            die "unknown EAM scope: $1"
            ;;
    esac
}

scope_tensor_count() {
    case "$1" in
        head_only) printf '2\n' ;;
        transformer_ln_plus_head) printf '16\n' ;;
        decoder_plus_head) printf '22\n' ;;
        full_transformer_plus_head) printf '36\n' ;;
        *) die "unknown EAM scope: $1" ;;
    esac
}

scope_parameter_count() {
    case "$1" in
        head_only) printf '1028\n' ;;
        transformer_ln_plus_head) printf '4612\n' ;;
        decoder_plus_head) printf '660996\n' ;;
        full_transformer_plus_head) printf '1057284\n' ;;
        *) die "unknown EAM scope: $1" ;;
    esac
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/avn/scripts/eval_smt_audio.sh"
MANIFEST_VALIDATOR="${REPO_ROOT}/tools/validate_run_manifest.py"
CHECKPOINT="${REPO_ROOT}/avn/checkpoints/source/smt_audio/single_best_val.pth"
DATASET="${REPO_ROOT}/avn/data/datasets/tta_test/single_source/mp3d/v1/val/val.json.gz"
LOG_ROOT="${REPO_ROOT}/avn/results/logs/eam_intensity_grid/${BATCH_ID}"
JOBS_ROOT="${LOG_ROOT}/jobs"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf 'uncommitted')"

lr_slug() {
    local value="$1"
    value="${value//./p}"
    value="${value//+/p}"
    value="${value//-/m}"
    printf '%s\n' "${value}"
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print $1}'
    else
        shasum -a 256 "${path}" | awk '{print $1}'
    fi
}

job_tag() {
    local job_index="$1"
    local scope="$2"
    local lr="$3"
    local interval="$4"
    printf 'eamgrid-%s-j%03d-%s-lr%s-u%s\n' \
        "${BATCH_ID}" "${job_index}" "${scope}" \
        "$(lr_slug "${lr}")" "${interval}"
}

write_plan() {
    local job_index=0
    local li
    local ui
    local si
    local lr
    local interval
    local scope
    local gpu_index
    local gpu
    local tag

    printf '%s\n' \
        'job_id,run_tag,gpu,model,source_setting,method,scope,expected_tensors,expected_parameters,lr,update_interval,seed,episodes'
    for ((li = 0; li < ${#LRS[@]}; li++)); do
        lr="${LRS[$li]}"
        for ((ui = 0; ui < ${#UPDATE_INTERVALS[@]}; ui++)); do
            interval="${UPDATE_INTERVALS[$ui]}"
            for ((si = 0; si < ${#SCOPES[@]}; si++)); do
                scope="${SCOPES[$si]}"
                if job_selected "${job_index}"; then
                    # Rotate scope cost across however many GPUs were selected.
                    gpu_index=$(((li + ui + si) % ${#GPUS[@]}))
                    gpu="${GPUS[$gpu_index]}"
                    tag="$(job_tag "${job_index}" "${scope}" "${lr}" "${interval}")"
                    printf '%s,%s,%s,smt_audio,single_source,eam,%s,%s,%s,%s,%s,%s,%s\n' \
                        "${job_index}" "${tag}" "${gpu}" "${scope}" \
                        "$(scope_tensor_count "${scope}")" \
                        "$(scope_parameter_count "${scope}")" \
                        "${lr}" "${interval}" "${SEED}" "${EPISODES}"
                fi
                job_index=$((job_index + 1))
            done
        done
    done
}

printf 'AVN EAM adaptation-intensity Cartesian grid\n'
printf '  repository:       %s\n' "${REPO_ROOT}"
printf '  model/source:     smt_audio/single_source\n'
printf '  LRs:              1e-8,1e-7,3e-7,1e-6,3e-6,1e-5\n'
printf '  update intervals: 1,4,16\n'
printf '  scopes:           %s\n' "${SCOPES[*]}"
printf '  fixed EAM:        a=%s M=%s K=%s optimizer=%s clip=%s\n' \
    "${CONFIDENCE_SCALE}" "${MEMORY_SIZE}" "${BATCH_SIZE}" \
    "${OPTIMIZER}" "${MAX_GRAD_NORM}"
printf '  jobs:             %s selected from %s\n' \
    "${EXPECTED_JOBS}" "${FULL_GRID_JOBS}"
printf '  GPUs:             %s\n' "${GPU_CSV}"
printf '  jobs/GPU:         %s concurrent\n' "${JOBS_PER_GPU}"
printf '  mode:             %s\n' \
    "$([[ ${SMOKE} -eq 1 ]] && printf smoke || printf full_grid)"
printf '  seed:             %s\n' "${SEED}"
printf '  episodes/job:     %s\n' "${EPISODES}"
printf '  batch:            %s\n' "${BATCH_ID}"
printf '  logs:             %s\n' "${LOG_ROOT}"

if [[ ${DRY_RUN} -eq 1 ]]; then
    DRY_PLAN="$(mktemp "${TMPDIR:-/tmp}/navtta_eam_intensity.XXXXXX")"
    write_plan > "${DRY_PLAN}"
    cat "${DRY_PLAN}"
    PLAN_JOBS=$(($(wc -l < "${DRY_PLAN}") - 1))
    rm -f "${DRY_PLAN}"
    [[ ${PLAN_JOBS} -eq ${EXPECTED_JOBS} ]] || \
        die "dry-run plan has ${PLAN_JOBS} jobs, expected ${EXPECTED_JOBS}"
    printf '\nDry run complete: %s selected jobs, nothing launched.\n' \
        "${EXPECTED_JOBS}"
    exit 0
fi

PREFLIGHT_ROOT="${REPO_ROOT}/avn/results/logs/eam_intensity_grid"
mkdir -p "${PREFLIGHT_ROOT}"
PREFLIGHT_LOG="${PREFLIGHT_ROOT}/${BATCH_ID}.preflight.log"
printf 'preflight_started_at=%s\nbatch_id=%s\ngit_commit=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${BATCH_ID}" "${GIT_COMMIT}" \
    > "${PREFLIGHT_LOG}"

command -v python3 >/dev/null 2>&1 || die "python3 is unavailable"
command -v git >/dev/null 2>&1 || die "git is unavailable"
command -v tee >/dev/null 2>&1 || die "tee is unavailable"
command -v cmp >/dev/null 2>&1 || die "cmp is unavailable"
command -v pgrep >/dev/null 2>&1 || die "pgrep is unavailable"
[[ -f "${RUNNER}" ]] || die "missing evaluation runner: ${RUNNER}"
[[ -f "${MANIFEST_VALIDATOR}" ]] || \
    die "missing run-manifest validator: ${MANIFEST_VALIDATOR}"
[[ -f "${CHECKPOINT}" ]] || die "missing source checkpoint: ${CHECKPOINT}"
[[ -f "${DATASET}" ]] || die "missing single-source TTA dataset: ${DATASET}"

TRACKED_STATUS="$(
    git -C "${REPO_ROOT}" status --porcelain --untracked-files=no
)"
TRACKED_WORKTREE_DIRTY=0
if [[ -n "${TRACKED_STATUS}" ]]; then
    TRACKED_WORKTREE_DIRTY=1
    printf '%s\n' "${TRACKED_STATUS}" >> "${PREFLIGHT_LOG}"
    if [[ ${ALLOW_DIRTY} -eq 0 ]]; then
        die "tracked worktree changes detected; commit them first or use --allow-dirty"
    fi
fi

if ! fingerprints="$(
    python3 "${REPO_ROOT}/avn/scripts/fingerprint_episode_stream.py" \
        --dataset "${DATASET}" --seed "${SEED}" \
        2>> "${PREFLIGHT_LOG}"
)"; then
    die "episode-stream fingerprinting failed; inspect ${PREFLIGHT_LOG}"
fi
read -r STREAM_ORDER_SHA256 STREAM_CONTENT_SHA256 <<< "${fingerprints}"
for fingerprint in "${STREAM_ORDER_SHA256}" "${STREAM_CONTENT_SHA256}"; do
    [[ "${fingerprint}" =~ ^[0-9a-f]{64}$ ]] || \
        die "invalid episode-stream fingerprint: ${fingerprint}"
done
CHECKPOINT_SHA256="$(sha256_file "${CHECKPOINT}")"
[[ "${CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
    die "invalid checkpoint SHA256"
printf 'preflight_passed_at=%s\ntracked_worktree_dirty=%s\nstream_order_sha256=%s\nstream_content_sha256=%s\ncheckpoint_sha256=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TRACKED_WORKTREE_DIRTY}" \
    "${STREAM_ORDER_SHA256}" "${STREAM_CONTENT_SHA256}" \
    "${CHECKPOINT_SHA256}" \
    >> "${PREFLIGHT_LOG}"

validate_job_artifacts() {
    local manifest_path="$1"
    local tag="$2"
    local scope="$3"
    local lr="$4"
    local interval="$5"
    local expected_tensors="$6"
    local expected_parameters="$7"
    local prefixes_json="$8"

    [[ -f "${manifest_path}" ]] || return 1
    python3 "${MANIFEST_VALIDATOR}" \
        --manifest "${manifest_path}" \
        --run-tag "${tag}" \
        --model smt_audio \
        --method eam \
        --source-setting single_source \
        --seed "${SEED}" \
        --git-commit "${GIT_COMMIT}" \
        --checkpoint-sha256 "${CHECKPOINT_SHA256}" \
        --stream-order-sha256 "${STREAM_ORDER_SHA256}" \
        --stream-content-sha256 "${STREAM_CONTENT_SHA256}" >/dev/null || \
        return 1

    python3 - "${manifest_path}" "${SEED}" "${EPISODES}" \
        "${scope}" "${lr}" "${interval}" "${expected_tensors}" \
        "${expected_parameters}" "${prefixes_json}" <<'PY'
import json
import math
import os
import sys

(
    manifest_path,
    seed,
    expected_episodes_text,
    scope,
    lr_text,
    interval_text,
    expected_tensors_text,
    expected_parameters_text,
    prefixes_json,
) = sys.argv[1:]

run_dir = os.path.dirname(os.path.abspath(manifest_path))
stats_path = os.path.join(
    run_dir, "raw", "model", "tb", "val_stats_{}.json".format(seed)
)
diagnostics_path = os.path.join(
    run_dir, "raw", "model", "tb", "tta_diagnostics_{}.json".format(seed)
)
with open(stats_path, "r", encoding="utf-8") as handle:
    stats = json.load(handle)
with open(diagnostics_path, "r", encoding="utf-8") as handle:
    diagnostics = json.load(handle)

expected_episodes = int(expected_episodes_text)
interval = int(interval_text)
expected_tensors = int(expected_tensors_text)
expected_parameters = int(expected_parameters_text)
expected_prefixes = json.loads(prefixes_json)

def require(condition, message):
    if not condition:
        raise SystemExit(message)

require(isinstance(stats, dict), "episode statistics are not a dictionary")
require(len(stats) == expected_episodes, "episode statistics count mismatch")
reported_metrics = (
    "success",
    "spl",
    "softspl",
    "sna",
    "na",
    "distance_to_goal",
    "normalized_distance_to_goal",
)
for episode_key, episode_stats in stats.items():
    require(
        isinstance(episode_stats, dict),
        "episode statistics are invalid for {}".format(episode_key),
    )
    for metric in reported_metrics:
        require(
            metric in episode_stats,
            "episode {} is missing {}".format(episode_key, metric),
        )
        try:
            metric_value = float(episode_stats[metric])
        except (TypeError, ValueError):
            raise SystemExit(
                "episode {} has nonnumeric {}".format(episode_key, metric)
            )
        require(
            math.isfinite(metric_value),
            "episode {} has non-finite {}".format(episode_key, metric),
        )
require(diagnostics.get("episodes") == expected_episodes, "episode count mismatch")
require(diagnostics.get("optimizer") == "Adam", "EAM optimizer mismatch")
require(diagnostics.get("param_scope") == "module_prefixes", "scope mode mismatch")
require(diagnostics.get("trainable_prefixes") == expected_prefixes, "prefix list mismatch")
require(
    diagnostics.get("adapted_parameter_count") == expected_parameters,
    "adapted parameter count mismatch for {}".format(scope),
)
names = diagnostics.get("adapted_parameter_names")
require(isinstance(names, list), "adapted parameter names are missing")
require(len(names) == expected_tensors, "adapted tensor count mismatch")
require(
    all(
        any(name == prefix or name.startswith(prefix + ".") for prefix in expected_prefixes)
        for name in names
    ),
    "adapted parameter outside declared prefixes",
)
require(
    math.isclose(float(diagnostics.get("current_lr")), float(lr_text), rel_tol=1e-12),
    "EAM learning rate mismatch",
)
require(
    math.isclose(float(diagnostics.get("confidence_scale")), 0.4, rel_tol=1e-12),
    "EAM confidence scale mismatch",
)
require(diagnostics.get("memory_size_steps") == 32, "memory size mismatch")
require(diagnostics.get("batch_size_steps") == 8, "batch size mismatch")
require(diagnostics.get("update_interval") == interval, "update interval mismatch")
require(diagnostics.get("update_interval_unit") == "action_step", "interval unit mismatch")
require(diagnostics.get("replay_unit") == "action_step", "replay unit mismatch")
require(
    diagnostics.get("replay_sampling") == "updated_reservoir_including_current",
    "replay sampling semantics mismatch",
)
require(
    diagnostics.get("short_buffer_behavior") == "current_only_update",
    "short-buffer semantics mismatch",
)

action_steps = int(diagnostics.get("action_steps", 0))
attempts = int(diagnostics.get("update_attempts", -1))
updates = int(diagnostics.get("updates", -1))
seen_steps = int(diagnostics.get("seen_steps", -1))
require(action_steps > 0, "no EAM action steps were recorded")
require(seen_steps == action_steps, "seen/action step mismatch")
require(attempts == action_steps // interval, "update-attempt count mismatch")
require(0 <= updates <= attempts, "actual update count is invalid")
require(
    diagnostics.get("current_only_batches") == min(action_steps, 7),
    "current-only batch count mismatch",
)
require(
    diagnostics.get("replay_size") == min(action_steps, 32),
    "final replay size mismatch",
)
expected_replayed = sum(
    1 if step < 8 else 8
    for step in range(interval, action_steps + 1, interval)
)
require(
    diagnostics.get("replayed_steps") == expected_replayed,
    "replayed-step count mismatch",
)
for diagnostic_name in (
    "mean_entropy",
    "last_entropy",
    "last_grad_norm",
    "relative_param_drift",
    "mean_train_loss",
    "last_train_loss",
    "mean_max_action_probability",
):
    try:
        diagnostic_value = float(diagnostics[diagnostic_name])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(
            "missing or nonnumeric diagnostic: {}".format(diagnostic_name)
        )
    require(
        math.isfinite(diagnostic_value),
        "non-finite diagnostic: {}".format(diagnostic_name),
    )
PY
}

write_batch_spec() {
    printf 'batch_id=%s\n' "${BATCH_ID}"
    printf 'git_commit=%s\n' "${GIT_COMMIT}"
    printf 'tracked_worktree_dirty=%s\n' "${TRACKED_WORKTREE_DIRTY}"
    printf 'allow_dirty=%s\n' "${ALLOW_DIRTY}"
    printf 'experiment=eam_intensity_grid_v1\n'
    printf 'model=smt_audio\n'
    printf 'source_setting=single_source\n'
    printf 'method=eam\n'
    printf 'seed=%s\n' "${SEED}"
    printf 'episodes=%s\n' "${EPISODES}"
    printf 'gpus=%s\n' "${GPU_CSV}"
    printf 'jobs_per_gpu=%s\n' "${JOBS_PER_GPU}"
    printf 'smoke=%s\n' "${SMOKE}"
    printf 'expected_jobs=%s\n' "${EXPECTED_JOBS}"
    printf 'action_selection=%s\n' "${ACTION_SELECTION}"
    printf 'num_processes=1\n'
    printf 'eval_use_ckpt_config=False\n'
    printf 'lrs=1e-8,1e-7,3e-7,1e-6,3e-6,1e-5\n'
    printf 'update_intervals=1,4,16\n'
    printf 'scopes=%s\n' "$(IFS=,; printf '%s' "${SCOPES[*]}")"
    printf 'confidence_scale=%s\n' "${CONFIDENCE_SCALE}"
    printf 'memory_size=%s\n' "${MEMORY_SIZE}"
    printf 'batch_size=%s\n' "${BATCH_SIZE}"
    printf 'param_scope=%s\n' "${PARAM_SCOPE}"
    printf 'episodic=%s\n' "${EPISODIC}"
    printf 'steps=%s\n' "${STEPS}"
    printf 'optimizer=%s\n' "${OPTIMIZER}"
    printf 'beta1=%s\n' "${BETA1}"
    printf 'beta2=%s\n' "${BETA2}"
    printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
    printf 'max_grad_norm=%s\n' "${MAX_GRAD_NORM}"
    for scope in "${SCOPES[@]}"; do
        printf 'scope_%s_prefixes=%s\n' "${scope}" "$(scope_prefixes "${scope}")"
        printf 'scope_%s_tensors=%s\n' "${scope}" "$(scope_tensor_count "${scope}")"
        printf 'scope_%s_parameters=%s\n' "${scope}" "$(scope_parameter_count "${scope}")"
    done
    printf 'checkpoint_sha256=%s\n' "${CHECKPOINT_SHA256}"
    printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
    printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
}

if [[ ${RESUME} -eq 0 ]]; then
    [[ ! -e "${LOG_ROOT}" ]] || die "batch already exists: ${LOG_ROOT}"
    mkdir -p "${JOBS_ROOT}"
    write_batch_spec > "${LOG_ROOT}/batch.env"
    write_plan > "${LOG_ROOT}/grid.csv"
else
    [[ -d "${LOG_ROOT}" ]] || die "resume batch does not exist: ${LOG_ROOT}"
    [[ -f "${LOG_ROOT}/batch.env" ]] || die "resume batch is missing batch.env"
    [[ -f "${LOG_ROOT}/grid.csv" ]] || die "resume batch is missing grid.csv"
    RESUME_SPEC="$(mktemp "${TMPDIR:-/tmp}/navtta_eam_spec.XXXXXX")"
    RESUME_PLAN="$(mktemp "${TMPDIR:-/tmp}/navtta_eam_plan.XXXXXX")"
    write_batch_spec > "${RESUME_SPEC}"
    write_plan > "${RESUME_PLAN}"
    if ! cmp -s "${LOG_ROOT}/batch.env" "${RESUME_SPEC}"; then
        rm -f "${RESUME_SPEC}" "${RESUME_PLAN}"
        die "resume arguments or immutable inputs do not match batch.env"
    fi
    if ! cmp -s "${LOG_ROOT}/grid.csv" "${RESUME_PLAN}"; then
        rm -f "${RESUME_SPEC}" "${RESUME_PLAN}"
        die "resume arguments do not match grid.csv"
    fi
    rm -f "${RESUME_SPEC}" "${RESUME_PLAN}"
fi

LOCK_DIR="${LOG_ROOT}/.scheduler.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    die "batch is already running or has a stale lock: ${LOCK_DIR}"
fi
printf 'pid=%s\nhost=%s\nstarted_at=%s\n' \
    "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${LOCK_DIR}/owner"

GPU_ACTIVE=()
for ((GPU_INIT_INDEX = 0; GPU_INIT_INDEX < ${#GPUS[@]}; GPU_INIT_INDEX++)); do
    GPU_ACTIVE+=(0)
done
PIDS=()
PID_GPU_INDEX=()
PID_TAG=()
PID_JOB_DIR=()
LAUNCHED=0
SKIPPED=0
REAP_FAILURES=0
LOCK_REASON=""

collect_process_tree() {
    local parent_pid="$1"
    local child_pid
    for child_pid in $(pgrep -P "${parent_pid}" 2>/dev/null || true); do
        collect_process_tree "${child_pid}"
    done
    printf '%s\n' "${parent_pid}"
}

active_worker_exists() {
    local index
    local pid
    # Bash 3.2 with `set -u` treats an expansion of an empty array as an
    # unbound variable. Indexing by the recorded length is portable to the
    # server and the default macOS Bash.
    for ((index = 0; index < ${#PIDS[@]}; index++)); do
        pid="${PIDS[$index]}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

terminate_active_workers() {
    local index
    local pid
    local attempt
    local still_running
    local tree_pid
    local termination_pids=()

    # Snapshot every descendant before sending TERM. A worker subshell can
    # exit before a simulator child; retaining each PID lets the KILL fallback
    # still reach a child after it has been reparented.
    for ((index = 0; index < ${#PIDS[@]}; index++)); do
        pid="${PIDS[$index]}"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            while IFS= read -r tree_pid; do
                [[ -n "${tree_pid}" ]] && termination_pids+=("${tree_pid}")
            done < <(collect_process_tree "${pid}")
        fi
    done
    for ((index = 0; index < ${#termination_pids[@]}; index++)); do
        kill -TERM "${termination_pids[$index]}" 2>/dev/null || true
    done

    # Give Habitat workers a bounded grace period, then force termination so
    # an interrupted scheduler cannot hang forever or leave a stale lock.
    for ((attempt = 0; attempt < 10; attempt++)); do
        still_running=0
        for ((index = 0; index < ${#termination_pids[@]}; index++)); do
            tree_pid="${termination_pids[$index]}"
            if kill -0 "${tree_pid}" 2>/dev/null; then
                still_running=1
                break
            fi
        done
        [[ ${still_running} -eq 1 ]] || break
        sleep 1
    done

    for ((index = 0; index < ${#termination_pids[@]}; index++)); do
        tree_pid="${termination_pids[$index]}"
        if kill -0 "${tree_pid}" 2>/dev/null; then
            kill -KILL "${tree_pid}" 2>/dev/null || true
        fi
    done
    for ((index = 0; index < ${#PIDS[@]}; index++)); do
        pid="${PIDS[$index]}"
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
            PIDS[$index]=""
        fi
    done
}

cleanup_lock() {
    local scheduler_status=$?
    if active_worker_exists; then
        LOCK_REASON="unexpected_scheduler_exit_workers_terminated"
        terminate_active_workers
    fi
    if [[ -f "${LOCK_DIR}/owner" ]]; then
        printf 'status=%s\nscheduler_exit_code=%s\n' \
            "${LOCK_REASON:-finished}" "${scheduler_status}" \
            >> "${LOCK_DIR}/owner"
        rm -f "${LOCK_DIR}/owner"
    fi
    rmdir "${LOCK_DIR}" 2>/dev/null || true
}

handle_signal() {
    local status="$1"
    local reason="$2"
    LOCK_REASON="${reason}"
    trap - HUP INT TERM
    terminate_active_workers
    printf 'scheduler interrupted; workers stopped and lock released\n' >&2
    exit "${status}"
}

trap cleanup_lock EXIT
trap 'handle_signal 129 signal_hup' HUP
trap 'handle_signal 130 signal_int' INT
trap 'handle_signal 143 signal_term' TERM

SCHEDULER_LOG="${LOG_ROOT}/scheduler.log"
exec > >(tee -a "${SCHEDULER_LOG}") 2>&1

printf 'scheduler_started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'git_commit=%s\n' "${GIT_COMMIT}"
printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
printf 'resume=%s\n' "${RESUME}"

total_active() {
    local total=0
    local count
    for count in "${GPU_ACTIVE[@]}"; do
        total=$((total + count))
    done
    printf '%s\n' "${total}"
}

reap_finished() {
    local index
    local pid
    local gpu_index
    local tag
    local job_dir
    local status

    for ((index = 0; index < ${#PIDS[@]}; index++)); do
        pid="${PIDS[$index]}"
        [[ -n "${pid}" ]] || continue
        job_dir="${PID_JOB_DIR[$index]}"
        if [[ -f "${job_dir}/exitcode" ]] || ! kill -0 "${pid}" 2>/dev/null; then
            if wait "${pid}"; then
                status=0
            else
                status=$?
            fi
            gpu_index="${PID_GPU_INDEX[$index]}"
            tag="${PID_TAG[$index]}"
            GPU_ACTIVE[$gpu_index]=$((GPU_ACTIVE[$gpu_index] - 1))
            PIDS[$index]=""
            if [[ ${status} -ne 0 ]]; then
                REAP_FAILURES=$((REAP_FAILURES + 1))
            fi
            printf '%s finished tag=%s status=%s active=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${tag}" "${status}" \
                "$(total_active)"
        fi
    done
}

wait_for_gpu_slot() {
    local gpu_index="$1"
    while [[ ${GPU_ACTIVE[$gpu_index]} -ge ${JOBS_PER_GPU} ]]; do
        reap_finished
        if [[ ${GPU_ACTIVE[$gpu_index]} -ge ${JOBS_PER_GPU} ]]; then
            sleep 5
        fi
    done
}

launch_job() {
    local job_index="$1"
    local gpu_index="$2"
    local gpu="$3"
    local scope="$4"
    local lr="$5"
    local interval="$6"
    local tag="$7"
    local prefixes_json
    local expected_tensors
    local expected_parameters
    local job_dir="${JOBS_ROOT}/${tag}"
    local previous_status=""
    local previous_validation=""
    local previous_manifest=""
    local attempt
    local pid

    prefixes_json="$(scope_prefixes "${scope}")"
    expected_tensors="$(scope_tensor_count "${scope}")"
    expected_parameters="$(scope_parameter_count "${scope}")"
    mkdir -p "${job_dir}"

    if [[ -f "${job_dir}/exitcode" ]]; then
        previous_status="$(tr -d '[:space:]' < "${job_dir}/exitcode")"
        if [[ -f "${job_dir}/validation" ]]; then
            previous_validation="$(tr -d '[:space:]' < "${job_dir}/validation")"
        fi
        if [[ -f "${job_dir}/manifest.path" ]]; then
            previous_manifest="$(tr -d '[:space:]' < "${job_dir}/manifest.path")"
        fi
        if [[ ${RESUME} -eq 1 && "${previous_status}" == "0" && \
                "${previous_validation}" == "ok" && \
                -n "${previous_manifest}" ]] && \
                validate_job_artifacts "${previous_manifest}" "${tag}" \
                    "${scope}" "${lr}" "${interval}" \
                    "${expected_tensors}" "${expected_parameters}" \
                    "${prefixes_json}" >/dev/null 2>&1; then
            SKIPPED=$((SKIPPED + 1))
            printf '%s skip validated tag=%s\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${tag}"
            return
        fi
    fi

    if [[ -f "${job_dir}/exitcode" || -f "${job_dir}/runner_exitcode" || \
            -f "${job_dir}/validation" || -f "${job_dir}/console.log" || \
            -f "${job_dir}/manifest.path" ]]; then
        attempt="$(date -u +%Y%m%dT%H%M%SZ)"
        [[ ! -f "${job_dir}/exitcode" ]] || \
            mv "${job_dir}/exitcode" "${job_dir}/exitcode.previous.${attempt}"
        [[ ! -f "${job_dir}/runner_exitcode" ]] || \
            mv "${job_dir}/runner_exitcode" \
                "${job_dir}/runner_exitcode.previous.${attempt}"
        [[ ! -f "${job_dir}/validation" ]] || \
            mv "${job_dir}/validation" \
                "${job_dir}/validation.previous.${attempt}"
        [[ ! -f "${job_dir}/console.log" ]] || \
            mv "${job_dir}/console.log" \
                "${job_dir}/console.previous.${attempt}.log"
        [[ ! -f "${job_dir}/manifest.path" ]] || \
            mv "${job_dir}/manifest.path" \
                "${job_dir}/manifest.previous.${attempt}.path"
    fi

    wait_for_gpu_slot "${gpu_index}"
    {
        printf 'job_id=%s\n' "${job_index}"
        printf 'run_tag=%s\n' "${tag}"
        printf 'gpu=%s\n' "${gpu}"
        printf 'model=smt_audio\n'
        printf 'source_setting=single_source\n'
        printf 'method=eam\n'
        printf 'action_selection=%s\n' "${ACTION_SELECTION}"
        printf 'scope=%s\n' "${scope}"
        printf 'trainable_prefixes=%s\n' "${prefixes_json}"
        printf 'expected_tensors=%s\n' "${expected_tensors}"
        printf 'expected_parameters=%s\n' "${expected_parameters}"
        printf 'lr=%s\n' "${lr}"
        printf 'update_interval=%s\n' "${interval}"
        printf 'seed=%s\n' "${SEED}"
        printf 'episodes=%s\n' "${EPISODES}"
        printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
        printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
    } > "${job_dir}/parameters.env"

    (
        set +e
        printf '\n===== attempt %s =====\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${job_dir}/console.log"
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export PYTHONUNBUFFERED=1
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NAVTTA_RUN_TAG="${tag}"
        export NAVTTA_STREAM_ORDER_SHA256="${STREAM_ORDER_SHA256}"
        export NAVTTA_STREAM_CONTENT_SHA256="${STREAM_CONTENT_SHA256}"

        bash "${RUNNER}" single_source eam "${SEED}" \
            NUM_PROCESSES 1 \
            EVAL.USE_CKPT_CONFIG False \
            EVAL.ACTION_SELECTION "${ACTION_SELECTION}" \
            TTA.EPISODIC "${EPISODIC}" \
            TTA.STEPS "${STEPS}" \
            TTA.EAM.LR "${lr}" \
            TTA.EAM.CONFIDENCE_SCALE "${CONFIDENCE_SCALE}" \
            TTA.EAM.MEMORY_SIZE "${MEMORY_SIZE}" \
            TTA.EAM.BATCH_SIZE "${BATCH_SIZE}" \
            TTA.EAM.UPDATE_INTERVAL "${interval}" \
            TTA.EAM.PARAM_SCOPE "${PARAM_SCOPE}" \
            TTA.EAM.TRAINABLE_PREFIXES "${prefixes_json}" \
            TTA.EAM.OPTIMIZER "${OPTIMIZER}" \
            TTA.EAM.BETA1 "${BETA1}" \
            TTA.EAM.BETA2 "${BETA2}" \
            TTA.EAM.WEIGHT_DECAY "${WEIGHT_DECAY}" \
            TTA.EAM.MAX_GRAD_NORM "${MAX_GRAD_NORM}" \
            TEST_EPISODE_COUNT "${EPISODES}" \
            >> "${job_dir}/console.log" 2>&1
        status=$?
        composite_status="${status}"

        manifest_path="$(
            tr '\r' '\n' < "${job_dir}/console.log" | \
                sed -n '/\/manifest[.]json$/p' | tail -n 1
        )"
        if [[ -n "${manifest_path}" && -f "${manifest_path}" ]]; then
            printf '%s\n' "${manifest_path}" > "${job_dir}/manifest.path.tmp"
            mv "${job_dir}/manifest.path.tmp" "${job_dir}/manifest.path"
        fi

        validation="failed"
        if [[ ${status} -eq 0 && -n "${manifest_path}" ]] && \
                validate_job_artifacts "${manifest_path}" "${tag}" \
                    "${scope}" "${lr}" "${interval}" \
                    "${expected_tensors}" "${expected_parameters}" \
                    "${prefixes_json}" >> "${job_dir}/console.log" 2>&1; then
            validation="ok"
        elif [[ ${status} -eq 0 ]]; then
            composite_status=90
            printf 'artifact validation failed\n' >> "${job_dir}/console.log"
        fi

        printf '%s\n' "${status}" > "${job_dir}/runner_exitcode.tmp"
        mv "${job_dir}/runner_exitcode.tmp" "${job_dir}/runner_exitcode"
        printf '%s\n' "${validation}" > "${job_dir}/validation.tmp"
        mv "${job_dir}/validation.tmp" "${job_dir}/validation"
        printf '%s\n' "${composite_status}" > "${job_dir}/exitcode.tmp"
        mv "${job_dir}/exitcode.tmp" "${job_dir}/exitcode"
        exit "${composite_status}"
    ) &
    pid=$!

    PIDS+=("${pid}")
    PID_GPU_INDEX+=("${gpu_index}")
    PID_TAG+=("${tag}")
    PID_JOB_DIR+=("${job_dir}")
    GPU_ACTIVE[$gpu_index]=$((GPU_ACTIVE[$gpu_index] + 1))
    LAUNCHED=$((LAUNCHED + 1))
    printf '%s launched job=%s tag=%s gpu=%s active_on_gpu=%s total_active=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${job_index}" "${tag}" \
        "${gpu}" "${GPU_ACTIVE[$gpu_index]}" "$(total_active)"
}

JOB_INDEX=0
for ((LI = 0; LI < ${#LRS[@]}; LI++)); do
    LR="${LRS[$LI]}"
    for ((UI = 0; UI < ${#UPDATE_INTERVALS[@]}; UI++)); do
        UPDATE_INTERVAL="${UPDATE_INTERVALS[$UI]}"
        for ((SI = 0; SI < ${#SCOPES[@]}; SI++)); do
            SCOPE="${SCOPES[$SI]}"
            if job_selected "${JOB_INDEX}"; then
                GPU_INDEX=$(((LI + UI + SI) % ${#GPUS[@]}))
                GPU="${GPUS[$GPU_INDEX]}"
                RUN_TAG="$(job_tag "${JOB_INDEX}" "${SCOPE}" "${LR}" \
                    "${UPDATE_INTERVAL}")"
                launch_job "${JOB_INDEX}" "${GPU_INDEX}" "${GPU}" \
                    "${SCOPE}" "${LR}" "${UPDATE_INTERVAL}" "${RUN_TAG}"
            fi
            JOB_INDEX=$((JOB_INDEX + 1))
            reap_finished
        done
    done
done

while [[ $(total_active) -gt 0 ]]; do
    reap_finished
    [[ $(total_active) -eq 0 ]] || sleep 5
done

EXIT_FILES=$(find "${JOBS_ROOT}" -type f -name exitcode | wc -l | tr -d ' ')
SUCCESSFUL=0
FAILED=0
while IFS= read -r exit_file; do
    status="$(tr -d '[:space:]' < "${exit_file}")"
    if [[ "${status}" == "0" ]]; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done < <(find "${JOBS_ROOT}" -type f -name exitcode | sort)
MISSING=$((EXPECTED_JOBS - EXIT_FILES))
MANIFEST_POINTERS=$(find "${JOBS_ROOT}" -type f -name manifest.path | wc -l | tr -d ' ')
VALIDATED=0
while IFS= read -r validation_file; do
    validation_status="$(tr -d '[:space:]' < "${validation_file}")"
    if [[ "${validation_status}" == "ok" ]]; then
        VALIDATED=$((VALIDATED + 1))
    fi
done < <(find "${JOBS_ROOT}" -type f -name validation | sort)

{
    printf 'batch_id=%s\n' "${BATCH_ID}"
    printf 'git_commit=%s\n' "${GIT_COMMIT}"
    printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'tracked_worktree_dirty=%s\n' "${TRACKED_WORKTREE_DIRTY}"
    printf 'smoke=%s\n' "${SMOKE}"
    printf 'expected=%s\n' "${EXPECTED_JOBS}"
    printf 'launched_this_invocation=%s\n' "${LAUNCHED}"
    printf 'skipped_completed=%s\n' "${SKIPPED}"
    printf 'successful=%s\n' "${SUCCESSFUL}"
    printf 'failed=%s\n' "${FAILED}"
    printf 'missing=%s\n' "${MISSING}"
    printf 'manifest_pointers=%s\n' "${MANIFEST_POINTERS}"
    printf 'validated=%s\n' "${VALIDATED}"
    printf 'stream_order_sha256=%s\n' "${STREAM_ORDER_SHA256}"
    printf 'stream_content_sha256=%s\n' "${STREAM_CONTENT_SHA256}"
} | tee "${LOG_ROOT}/SUMMARY"

if [[ ${REAP_FAILURES} -ne 0 || ${FAILED} -ne 0 || ${MISSING} -ne 0 || \
        ${MANIFEST_POINTERS} -ne ${EXPECTED_JOBS} || \
        ${VALIDATED} -ne ${EXPECTED_JOBS} ]]; then
    printf 'Grid incomplete; inspect logs and resume with --batch-id %s --resume.\n' \
        "${BATCH_ID}" >&2
    exit 1
fi

printf 'All %s SMT+Audio single-source EAM intensity jobs completed successfully.\n' \
    "${EXPECTED_JOBS}"
