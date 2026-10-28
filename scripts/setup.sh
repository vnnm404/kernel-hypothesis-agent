#!/usr/bin/env bash
# Use the same Python environment as your GPU benchmark installation.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN=${PYTHON_BIN:-python3}
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 required"'
"$PYTHON_BIN" -m pip install -e '.[dev]'
echo 'Installed. Run: kha smoke --out runs/smoke'
echo 'GPU dependencies and agent authentication are managed separately; see docs/snowflake.md.'
