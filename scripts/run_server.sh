#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/run_server.sh inspect|verify|smoke|collect|resume|status|report [CLI arguments]' >&2
  exit 2
fi
exec "${PRECHECK_PYTHON:-python3}" -B -m frozen_repair "$@"
