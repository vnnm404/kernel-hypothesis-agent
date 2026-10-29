#!/usr/bin/env bash
# Set up the benchmark locally, then run the installed Claude Code.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
CHECK=0 SMOKE=0
if [[ "${1:-}" == --check ]]; then CHECK=1; shift; elif [[ "${1:-}" == --smoke ]]; then SMOKE=1; shift; fi
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: ./run.sh [--check|--smoke] [existing-flash-cuda-path] [Claude options]'
  exit 0
fi
# An optional checkout path retains support for an already prepared environment.
FLASH=""
if [[ $# -gt 0 && "$1" != -* ]]; then FLASH=$(cd "$1" && pwd); shift; fi
GPU=${GPU:-0}
export PYTHONDONTWRITEBYTECODE=1
if [[ "$CHECK" == 0 ]]; then
  command -v claude >/dev/null || { echo 'Claude is not on PATH. Use your normal Claude environment.' >&2; exit 1; }
  # Default: Snowflake endpoint with the PAT file. Another endpoint needs ANTHROPIC_BASE_URL
  # set with its credentials, so a non-Snowflake key is never sent to Snowflake by default.
  SNOWFLAKE_URL=https://snowhouse.snowflakecomputing.com/api/v2/cortex/anthropic
  if [[ -z "${ANTHROPIC_BASE_URL:-}" && -n "${ANTHROPIC_API_KEY:-}${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
    echo 'An Anthropic key is set without ANTHROPIC_BASE_URL. Set ANTHROPIC_BASE_URL for that key,' >&2
    echo 'or unset the key to use the Snowflake route with ~/.snowhouse-pat.' >&2
    exit 1
  fi
  BASE_URL=${ANTHROPIC_BASE_URL:-$SNOWFLAKE_URL}
  KEY=${ANTHROPIC_API_KEY:-}
  PAT_FILE=${SNOWFLAKE_PAT_FILE:-$HOME/.snowhouse-pat}
  if [[ "$BASE_URL" == "$SNOWFLAKE_URL" && -z "$KEY${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
    # Reuse KDA's key only when its Cortex endpoint is explicitly configured.
    if [[ "${ANTHROPIC_BASE_URL:-}" == "$SNOWFLAKE_URL" || "${OPENAI_BASE_URL:-}" == https://snowhouse.snowflakecomputing.com/api/v2/cortex/v1 ]]; then
      KEY=${OPENAI_API_KEY:-}
    fi
    [[ -n "$KEY" || ! -s "$PAT_FILE" ]] || KEY=$(tr -d '[:space:]' < "$PAT_FILE")
    [[ -n "$KEY" ]] || { echo "The Snowflake route needs your PAT in $PAT_FILE." >&2; exit 1; }
  fi
  # KDA's Snowflake route uses this model; --model or ANTHROPIC_MODEL overrides it.
  if [[ "$BASE_URL" == "$SNOWFLAKE_URL" && -z "${ANTHROPIC_MODEL:-}" ]]; then
    MODEL=1
    for arg in "$@"; do [[ "$arg" == --model || "$arg" == --model=* ]] && MODEL=0; done
    [[ "$MODEL" == 0 ]] || set -- --model 'claude-opus-5[1m]' "$@"
  fi
fi
command -v nvcc >/dev/null || { echo 'CUDA 13 toolkit required: nvcc is not on PATH.' >&2; exit 1; }
if [[ -z "$FLASH" ]]; then
  for tool in git curl g++; do
    command -v "$tool" >/dev/null || { echo "Required system tool missing: $tool" >&2; exit 1; }
  done
  nvcc --version | grep -q 'release 13\.' || { echo 'This setup requires CUDA 13.' >&2; exit 1; }
  DEPS="$ROOT/.deps"
  mkdir -p "$DEPS"
  export UV_PYTHON_INSTALL_DIR="$DEPS/python" UV_CACHE_DIR="$DEPS/cache"
  # Managed downloads stay here; neither shell profiles nor system Python are changed.
  UV="$DEPS/bin/uv"
  if [[ ! -x "$UV" ]]; then
    curl -fsSL https://astral.sh/uv/0.11.29/install.sh -o "$DEPS/install-uv.sh"
    UV_UNMANAGED_INSTALL="$DEPS/bin" sh "$DEPS/install-uv.sh"
  fi
  checkout() {
    local name=$1 url=$2 commit=$3
    if [[ ! -d "$DEPS/$name/.git" ]]; then
      git init -q "$DEPS/$name"
      git -C "$DEPS/$name" remote add origin "$url"
    fi
    if [[ "$(git -C "$DEPS/$name" rev-parse HEAD 2>/dev/null || true)" != "$commit" ]]; then
      git -C "$DEPS/$name" fetch --depth 1 origin "$commit"
      git -C "$DEPS/$name" checkout --detach FETCH_HEAD
    fi
  }
  checkout flash-cuda https://github.com/Snowflake-AI-Research/flash-cuda.git 74af5ce7d2849f2c33f9017fa6f896b9c991a0e3
  checkout flash-attention https://github.com/Dao-AILab/flash-attention.git df61ab6c4a0fb1f94f1f43b2a23479a0ab92b8ab
  FLASH="$DEPS/flash-cuda"
  PY="$DEPS/venv/bin/python"
  if [[ ! -x "$PY" ]]; then "$UV" venv --python 3.12 "$DEPS/venv"; fi
  # Match the tested B200 stack. FA4/quack's upstream .dev0 CUTLASS pin is broken.
  "$UV" pip install --python "$PY" torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
  "$UV" pip install --python "$PY" ninja 'nvidia-cutlass-dsl[cu13]==4.6.1' einops typing_extensions 'apache-tvm-ffi>=0.1.12,<0.2' torch-c-dlpack-ext
  "$UV" pip install --python "$PY" --no-deps quack-kernels==0.5.3 "$DEPS/flash-attention/flash_attn/cute"
  export PATH="$DEPS/venv/bin:$PATH"
else
  PY=$(command -v "${PYTHON_BIN:-python3}")
fi
for file in bench/__init__.py bench/eval.py bench/scenarios.py kernels/fa4_blackwell.cu; do
  [[ -f "$FLASH/$file" ]] || { echo "Missing $FLASH/$file" >&2; exit 1; }
done
# Fail before calling the agent if this host cannot run the benchmark.
"$PY" - "$FLASH" "$GPU" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
try:
    import torch
    from bench import eval as E, scenarios as S
    torch.cuda.set_device(int(sys.argv[2]))
    assert torch.cuda.get_device_capability()[0] == 10, 'Blackwell GPU required'
    assert E.baseline_available(E.TARGETS['blackwell']), E.baseline_import_error(E.TARGETS['blackwell'])
    names = {c.name for c in S.get_configs('all')}
    assert all(n in names for n in ('attn_dense_hd128_causal', 'attn_dense_hd128_noncausal', 'attn_dense_hd128_gqa')), 'Expected benchmark cases are missing'
    print('Ready:', torch.cuda.get_device_name())
except Exception as e:
    sys.exit(f'Benchmark environment is not ready: {e}\nCheck the CUDA toolkit/driver and the setup output above.')
PY
mkdir -p "$ROOT/runs"
OUT=$(mktemp -d "$ROOT/runs/run-XXXXXXXX")
# Stage only the existing benchmark files, so its builds/reports stay in this run.
mkdir -p "$OUT/bench"
cp "$FLASH/bench/"{__init__,eval,scenarios}.py "$OUT/bench/"
[[ ! -f "$FLASH/LICENSE" ]] || cp "$FLASH/LICENSE" "$OUT/bench/LICENSE"
cp "$FLASH/kernels/fa4_blackwell.cu" "$OUT/seed.cu"
cp "$OUT/seed.cu" "$OUT/candidate.cu"
cp "$OUT/seed.cu" "$OUT/best.cu"
if [[ "$SMOKE" == 1 ]]; then
  cat > "$OUT/instructions.md" <<'SMOKE'
SMOKE TEST ONLY: do not optimize or run benchmarks. Read seed.log. Do not modify any
existing file. Write result.md starting with the line "Smoke test only; no optimization
attempted." followed by the correctness and perf lines copied exactly from seed.log.
SMOKE
else
  cat > "$OUT/instructions.md" <<'TASK'
Improve candidate.cu using this existing FlashAttention benchmark.

Start by reading seed.log and the source. Work only in this run directory.
Keep seed.cu, benchmark.sh, bench/, instructions.md, and existing logs unchanged.
Preserve the kernel API, supported inputs, and numerical behavior. Use CUDA and
inline PTX; do not replace the kernel with library calls or specialize on input values.

Use ./benchmark.sh to check correctness and measure all three workloads. It saves
each output in measurements/. Run measurements sequentially on the selected GPU.
Make at most three optimization attempts, keeping a source snapshot for each.
Before each change, write down what you think limits performance, the evidence,
and what result would disprove that explanation.

If two changes fail to improve performance, stop guessing. Name two competing
explanations and run the smallest test that distinguishes them. Record what the
test establishes, what remains uncertain, and how it changes your next step.
If profiling counters are unavailable, save the error and use another test.
Treat deliberately incorrect ablations only as diagnostics, never as improvements.

Recheck promising changes against the previous best using repeated runs. Keep a
change only if all cases pass, geometric-mean latency improves by at least 2%, and
no case gets more than 3% slower. Save the best verified version as best.cu;
leave it equal to seed.cu if no change qualifies. These decisions are your
responsibility; the wrapper only checks the seed and final best with the benchmark.

Write result.md with the tested explanations, changes, per-case measurements,
failed attempts, evidence filenames, and next experiment. Distinguish measured
results from guesses. Do not change the environment, install packages, or call
other agents. Stop after the three attempts and report honestly if blocked.
TASK
fi
git -C "$FLASH" rev-parse HEAD > "$OUT/source-commit.txt" 2>/dev/null || true
# The original evaluator stringifies FA3_LIB_NAME in TORCH_LIBRARY.
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$OUT"
  printf 'PY=%q\nGPU=%q\n' "$PY" "$GPU"
  cat <<'BENCH'
mkdir -p measurements
LOG=$(mktemp "$PWD/measurements/bench-XXXXXXXX")
"$PY" -m bench.eval --target blackwell --cand "${1:-$PWD/candidate.cu}" \
  --namespace FA3_LIB_NAME --gpu "$GPU" --set all --no-cudnn --no-flashinfer \
  --configs attn_dense_hd128_causal,attn_dense_hd128_noncausal,attn_dense_hd128_gqa | tee "$LOG"
# Upstream can exit zero for unsupported cases; require full correctness and timing.
grep -Eq '^correct: 3/3 +unsupported: 0 +wrong: 0' "$LOG" && \
  grep -q '^perf over 3 configs:' "$LOG" || { echo "Incomplete/failed benchmark: $LOG" >&2; exit 1; }
BENCH
} > "$OUT/benchmark.sh"
chmod +x "$OUT/benchmark.sh"
cd "$OUT"
export PYTHONPATH="$OUT${PYTHONPATH:+:$PYTHONPATH}"
echo "Results: $OUT"
./benchmark.sh | tee seed.log
if [[ "$CHECK" == 1 ]]; then echo "GPU check passed: $OUT"; exit 0; fi
echo "Running Claude. Trace: $OUT/trace.jsonl"
TOOLS='Read,Write,Edit,Bash,Glob,Grep'
[[ "$SMOKE" == 0 ]] || TOOLS='Read,Write'
# File-loaded credentials are passed only to Claude; they are never printed.
(
  export ANTHROPIC_BASE_URL=$BASE_URL
  [[ -z "$KEY" ]] || export ANTHROPIC_API_KEY=$KEY
  exec claude -p --verbose --output-format stream-json \
    --allowedTools "$TOOLS" "$@" < instructions.md > trace.jsonl
)
"$PY" - <<'PY'
import json
from pathlib import Path
results = [e for line in Path('trace.jsonl').read_text().splitlines()
           if line.strip() and (e := json.loads(line)).get('type') == 'result']
if not results or results[-1].get('is_error'):
    raise SystemExit('Claude did not finish successfully. See trace.jsonl.')
if not Path('result.md').is_file() or not Path('result.md').stat().st_size:
    raise SystemExit('Claude did not write result.md. See trace.jsonl.')
PY
if [[ "$SMOKE" == 1 ]] && ! { cmp -s seed.cu candidate.cu && cmp -s seed.cu best.cu; }; then
  echo 'Smoke test failed: Claude changed candidate.cu or best.cu.' >&2; exit 1
fi
# Same bytes: benchmark candidate.cu so an up-to-date build is reused; checks and timings rerun.
FINAL="$OUT/best.cu"
cmp -s best.cu candidate.cu && FINAL="$OUT/candidate.cu"
./benchmark.sh "$FINAL" | tee final.log
if [[ "$SMOKE" == 1 ]]; then
  echo "Smoke test passed (no optimization attempted). Read $OUT/result.md and $OUT/final.log"
else
  echo "Finished. Read $OUT/result.md (final verification: $OUT/final.log)"
fi
