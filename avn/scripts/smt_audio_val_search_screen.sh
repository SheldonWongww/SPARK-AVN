#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
BATCH_ID="${2:-}"
if [[ -z "${ACTION}" ]]; then
  printf 'usage: %s <start|resume|status|attach> [batch-id] [runner options ...]\n' "$0" >&2
  exit 2
fi
shift
if [[ $# -gt 0 ]]; then
  shift
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${REPO_ROOT}/avn/scripts/run_smt_audio_val_search.py"

if [[ -z "${BATCH_ID}" ]]; then
  if [[ "${ACTION}" == "start" ]]; then
    BATCH_ID="smt-four-val-v1-$(date -u +%Y%m%dT%H%M%SZ)"
  else
    printf '%s requires an explicit batch-id\n' "${ACTION}" >&2
    exit 2
  fi
fi
if [[ ! "${BATCH_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'invalid batch-id: %s\n' "${BATCH_ID}" >&2
  exit 2
fi
SESSION="avn-smt-${BATCH_ID}"
SESSION="${SESSION:0:78}"

cd "${REPO_ROOT}"
case "${ACTION}" in
  start)
    command -v screen >/dev/null || { printf 'screen is unavailable\n' >&2; exit 1; }
    screen -dmS "${SESSION}" \
      python3 "${RUNNER}" --batch-id "${BATCH_ID}" "$@"
    printf 'started screen=%s batch=%s\n' "${SESSION}" "${BATCH_ID}"
    ;;
  resume)
    command -v screen >/dev/null || { printf 'screen is unavailable\n' >&2; exit 1; }
    screen -dmS "${SESSION}" \
      python3 "${RUNNER}" --batch-id "${BATCH_ID}" --resume "$@"
    printf 'resumed screen=%s batch=%s\n' "${SESSION}" "${BATCH_ID}"
    ;;
  status)
    exec python3 "${RUNNER}" --batch-id "${BATCH_ID}" --status
    ;;
  attach)
    exec screen -r "${SESSION}"
    ;;
  *)
    printf 'unknown action: %s\n' "${ACTION}" >&2
    exit 2
    ;;
esac
