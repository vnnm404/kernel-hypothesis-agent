#!/usr/bin/env bash
# Use the existing flash-cuda environment and installed Claude Code; install nothing.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
CHECK=0 SMOKE=0
if [[ "${1:-}" == --check ]]; then CHECK=1; shift; elif [[ "${1:-}" == --smoke ]]; then SMOKE=1; shift; fi
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
  echo 'Usage: ./run.sh [--check|--smoke] /path/to/flash-cuda [Claude options]'
  exit 0
fi
FLASH=$(cd "$1" && pwd); shift
PY=$(command -v "${PYTHON_BIN:-python3}")
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
command -v nvcc >/dev/null || { echo 'nvcc is not on PATH. Activate your working flash-cuda environment.' >&2; exit 1; }
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
    sys.exit(f'Benchmark environment is not ready: {e}\nUse the Python environment that already runs flash-cuda.')
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
  cp "$ROOT/prompt.md" "$OUT/instructions.md"
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
