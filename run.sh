#!/usr/bin/env bash
# One entry point; uses the GPU venv when present, otherwise your active Python.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == setup ]]; then
  shift
  exec ./scripts/setup_blackwell.sh "$@"
fi
if [[ -x "${KHA_VENV:-.venv-gpu}/bin/python" ]]; then
  PY="${KHA_VENV:-.venv-gpu}/bin/python"
  export PATH="$(cd "$(dirname "$PY")" && pwd):$PATH"
else
  PY=${PYTHON_BIN:-python3}
fi
export KHA_BUILD_CACHE="${KHA_BUILD_CACHE:-$PWD/runs/.build-cache}"
exec "$PY" -m kernel_agent.quickstart "$@"
