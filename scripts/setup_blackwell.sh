#!/usr/bin/env bash
# Install the B200 benchmark environment in a checkout-local virtualenv.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvcc >/dev/null || { echo 'CUDA 13 toolkit (nvcc) is required.' >&2; exit 1; }
command -v git >/dev/null || { echo 'git is required.' >&2; exit 1; }
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV=${KHA_VENV:-.venv-gpu}
"$PYTHON_BIN" -m venv "$VENV" --system-site-packages
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip
if ! "$PY" -c 'import torch; assert torch.cuda.is_available(); assert torch.version.cuda.startswith("13.")' 2>/dev/null; then
  "$PY" -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130
fi
"$PY" -m pip install -e '.[dev]' ninja 'nvidia-cutlass-dsl[cu13]==4.6.1' einops \
  typing_extensions 'apache-tvm-ffi>=0.1.12,<0.2' torch-c-dlpack-ext
# FA4/quack's development-version DSL pin conflicts with the working release above.
"$PY" -m pip install --no-deps 'quack-kernels==0.5.3'
FA_DIR=external/flash-attention
REV=df61ab6c4a0fb1f94f1f43b2a23479a0ab92b8ab
if [[ ! -d "$FA_DIR/.git" ]]; then
  mkdir -p "$FA_DIR"
  git -C "$FA_DIR" init -q
  git -C "$FA_DIR" remote add origin https://github.com/Dao-AILab/flash-attention.git
fi
git -C "$FA_DIR" fetch --depth 1 origin "$REV"
git -C "$FA_DIR" checkout --detach "$REV"
"$PY" -m pip install --no-deps "$FA_DIR/flash_attn/cute"
"$PY" - <<'PY'
import torch
from flash_attn.cute.interface import _flash_attn_fwd
assert torch.cuda.get_device_capability(0)[0] == 10, 'SM100 Blackwell GPU required'
assert _flash_attn_fwd is not None
print('Ready:', torch.cuda.get_device_name(0), 'torch', torch.__version__)
PY
echo 'Next: ./run.sh check (GPU validation only), or ./run.sh pilot --backend codex --model YOUR_MODEL'
