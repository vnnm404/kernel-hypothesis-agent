#!/usr/bin/env bash
# Use the existing flash-cuda environment and Claude login; install nothing.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
CHECK=0
if [[ "${1:-}" == --check ]]; then CHECK=1; shift; fi
if [[ $# -eq 0 || "${1:-}" == --help ]]; then
  echo 'Usage: ./run.sh [--check] /path/to/flash-cuda [Claude options]'
  exit 0
fi
FLASH=$(cd "$1" && pwd); shift
PY=$(command -v "${PYTHON_BIN:-python3}")
GPU=${GPU:-0}
export PYTHONDONTWRITEBYTECODE=1
if [[ "$CHECK" == 0 ]]; then
  command -v claude >/dev/null || { echo 'Claude is not on PATH. Use your normal Claude environment.' >&2; exit 1; }
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
cp "$ROOT/prompt.md" "$OUT/instructions.md"
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
claude -p --verbose --output-format stream-json \
  --allowedTools 'Read,Write,Edit,Bash,Glob,Grep' "$@" < instructions.md > trace.jsonl
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
./benchmark.sh "$OUT/best.cu" | tee final.log
echo "Finished. Read $OUT/result.md (final verification: $OUT/final.log)"
