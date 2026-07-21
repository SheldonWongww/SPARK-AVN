#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Queue multiple Tent grids sequentially. A grid starts only after the previous
grid has produced four ALL_DONE markers and twelve successful exit-code files.

Usage:
  bash avn/scripts/queue_tent_grids.sh [options] <model:source> [...]

Grid specifications:
  smt_audio:single_source
  smt_audio:multi_source
  enmus:single_source
  enmus:multi_source

Options:
  --wait-batch ID    Wait for an already-running batch before starting the
                     queue. Use "latest" to select the newest batch directory.
  --seed N           Evaluation seed passed to every grid (default: 0)
  --gpus LIST        Four physical GPU ids (default: 0,1,2,3)
  --lrs LIST         Three learning rates (default: 1e-5,1e-6,1e-7)
  --episodes N       Episodes per run (default: 2000)
  --poll-seconds N   Queue polling interval (default: 60)
  --dry-run          Validate and print the queue without waiting or launching
  -h, --help         Show this help

Example while another grid is already running:
  screen -dmS tent_remaining_queue \
    bash avn/scripts/queue_tent_grids.sh --wait-batch latest \
      smt_audio:multi_source enmus:single_source enmus:multi_source

The queue inherits the active PATH/Conda environment. Its progress is visible
inside the supervisor screen and is also saved under results/logs/tent_grid/.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GRID_LAUNCHER="${SCRIPT_DIR}/run_tent_grid.sh"
GRID_ROOT="${REPO_ROOT}/avn/results/logs/tent_grid"

WAIT_BATCH=""
SEED=0
GPU_CSV="0,1,2,3"
LR_CSV="1e-5,1e-6,1e-7"
EPISODES=2000
POLL_SECONDS=60
DRY_RUN=0
SPECS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wait-batch)
            [[ $# -ge 2 ]] || die "--wait-batch requires a value"
            WAIT_BATCH="$2"
            shift 2
            ;;
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
        --lrs)
            [[ $# -ge 2 ]] || die "--lrs requires a value"
            LR_CSV="$2"
            shift 2
            ;;
        --episodes)
            [[ $# -ge 2 ]] || die "--episodes requires a value"
            EPISODES="$2"
            shift 2
            ;;
        --poll-seconds)
            [[ $# -ge 2 ]] || die "--poll-seconds requires a value"
            POLL_SECONDS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            die "unknown option: $1"
            ;;
        *)
            SPECS+=("$1")
            shift
            ;;
    esac
done

[[ ${#SPECS[@]} -gt 0 ]] || die "at least one model:source grid is required"
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    die "poll-seconds must be a positive integer"

for spec in "${SPECS[@]}"; do
    case "$spec" in
        smt_audio:single_source|smt_audio:multi_source|\
        enmus:single_source|enmus:multi_source)
            ;;
        *)
            die "invalid grid specification: $spec"
            ;;
    esac

    model="${spec%%:*}"
    source_setting="${spec#*:}"
    bash "$GRID_LAUNCHER" "$model" "$source_setting" \
        --seed "$SEED" \
        --gpus "$GPU_CSV" \
        --lrs "$LR_CSV" \
        --episodes "$EPISODES" \
        --batch-id "queue-validation" \
        --dry-run >/dev/null
done

latest_batch_dir() {
    local candidate
    local latest=""

    for candidate in "$GRID_ROOT"/*; do
        [[ -d "$candidate" ]] || continue
        if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
            latest="$candidate"
        fi
    done

    [[ -n "$latest" ]] || return 1
    printf '%s\n' "$latest"
}

count_files() {
    local path="$1"
    local pattern="$2"
    find "$path" -type f -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
}

wait_for_grid() {
    local path="$1"
    local batch_id="$2"
    local label="$3"
    local done_count=0
    local last_done=-1
    local exit_count
    local status_file
    local status

    [[ -d "$path" ]] || die "grid log directory does not exist: $path"
    printf 'waiting for %s (%s)\n' "$label" "$batch_id"

    while [[ $done_count -lt 4 ]]; do
        done_count="$(count_files "$path" ALL_DONE)"
        if [[ "$done_count" != "$last_done" ]]; then
            printf '%s progress %s: %s/4 scope workers complete\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label" "$done_count"
            last_done="$done_count"
        fi

        if [[ $done_count -lt 4 ]] && \
           ! screen -ls 2>/dev/null | grep -Fq "$batch_id"; then
            # Avoid racing the final marker write after the last screen exits.
            sleep 2
            done_count="$(count_files "$path" ALL_DONE)"
            if [[ $done_count -lt 4 ]]; then
                printf 'error: no screen for batch %s, but only %s/4 workers completed\n' \
                    "$batch_id" "$done_count" >&2
                return 1
            fi
        fi

        [[ $done_count -ge 4 ]] || sleep "$POLL_SECONDS"
    done

    exit_count="$(count_files "$path" '*.exitcode')"
    if [[ $exit_count -ne 12 ]]; then
        printf 'error: %s completed with only %s/12 exit-code files\n' \
            "$label" "$exit_count" >&2
        return 1
    fi

    while IFS= read -r status_file; do
        status="$(tr -d '[:space:]' < "$status_file")"
        if [[ "$status" != "0" ]]; then
            printf 'error: failed run (%s): exit code %s\n' \
                "$status_file" "$status" >&2
            return 1
        fi
    done < <(find "$path" -type f -name '*.exitcode' | sort)

    printf '%s completed successfully: %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$label"
}

printf 'Tent grid queue\n'
printf '  repository: %s\n' "$REPO_ROOT"
printf '  wait batch: %s\n' "${WAIT_BATCH:-none}"
printf '  grids:\n'
for spec in "${SPECS[@]}"; do
    printf '    - %s\n' "$spec"
done

if [[ $DRY_RUN -eq 1 ]]; then
    printf '  seed:       %s\n' "$SEED"
    printf '  GPUs:       %s\n' "$GPU_CSV"
    printf '  LRs:        %s\n' "$LR_CSV"
    printf '  episodes:   %s\n' "$EPISODES"
    printf '\nDry run complete; no batches were launched.\n'
    exit 0
fi

command -v screen >/dev/null 2>&1 || die "GNU screen is not installed"
mkdir -p "$GRID_ROOT"

QUEUE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
QUEUE_LOG="${GRID_ROOT}/queue_${QUEUE_ID}.log"
exec > >(tee -a "$QUEUE_LOG") 2>&1

printf 'queue_started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'queue_log=%s\n' "$QUEUE_LOG"

if [[ -n "$WAIT_BATCH" ]]; then
    if [[ "$WAIT_BATCH" == "latest" ]]; then
        WAIT_PATH="$(latest_batch_dir)" || die "no existing Tent batch was found"
        WAIT_BATCH="$(basename "$WAIT_PATH")"
    else
        WAIT_PATH="${GRID_ROOT}/${WAIT_BATCH}"
    fi

    wait_for_grid "$WAIT_PATH" "$WAIT_BATCH" "existing batch"
fi

QUEUE_INDEX=0
for spec in "${SPECS[@]}"; do
    QUEUE_INDEX=$((QUEUE_INDEX + 1))
    MODEL="${spec%%:*}"
    SOURCE_SETTING="${spec#*:}"
    BATCH_ID="queue-${QUEUE_ID}-${QUEUE_INDEX}-${MODEL}-${SOURCE_SETTING}"
    BATCH_PATH="${GRID_ROOT}/${BATCH_ID}/${MODEL}/${SOURCE_SETTING}"

    printf '%s launching %s as batch %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$spec" "$BATCH_ID"

    bash "$GRID_LAUNCHER" "$MODEL" "$SOURCE_SETTING" \
        --seed "$SEED" \
        --gpus "$GPU_CSV" \
        --lrs "$LR_CSV" \
        --episodes "$EPISODES" \
        --batch-id "$BATCH_ID"

    wait_for_grid "$BATCH_PATH" "$BATCH_ID" "$spec"
done

printf 'queue_completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'All queued Tent grids completed successfully.\n'
