# Running on a Snowflake GPU host

This follows the KDA-on-FastKernels operational pattern: an existing authenticated
coding CLI, an explicit model, one task workspace per trial, GPU leases around
benchmarks/profiling, and artifacts for every attempt. It does not require a new
service, Humanize, Redis, or model training.

## Environment

Use a dedicated Linux GPU host or container. Install this checkout into the Python
environment that already runs flash-cuda or FastKernels:

```bash
python -m pip install -e '.[dev]'
kha smoke --out runs/smoke
```

For a CPU-only development environment, `uv sync --extra dev` followed by
`uv run kha smoke --out runs/smoke` works. A standalone uv environment does not
inherit your GPU Python environment; use `python -m pip install -e .` there instead.

- Blackwell: SM100 GPU, compatible NVIDIA driver, CUDA toolkit with nvcc, torch,
  ninja, and the FA4 CuteDSL reference. The source repo's setup uses CUDA 13,
  `nvidia-cutlass-dsl[cu13]==4.6.1`, and installs FA4 with `--no-deps` to avoid a
  conflicting development-version dependency. Use an existing validated flash-cuda
  environment first; see the pinned upstream setup linked in PROVENANCE.json.
- Hopper: SM90 GPU, CUDA toolkit >=12.3, torch, ninja, and a vLLM build providing
  `vllm.vllm_flash_attn._vllm_fa3_C`. Missing references are errors, never silent
  fp32-only performance results.
- Nsight Compute (`ncu`) and permission to read counters for profiling. This is
  useful but not required for a run; diagnostic failures remain recorded.
- FastKernels: install your pinned FastKernels checkout and use the same captures
  as the baseline campaign. The adapter uses its native correctness checks.

The vendored CUDA seed and evaluator are pinned; the GPU environment is not a
universal binary lock. Before an actual campaign save the environment:

```bash
mkdir -p runs/environment
python -m pip freeze > runs/environment/pip-freeze.txt
nvidia-smi > runs/environment/nvidia-smi.txt
nvcc --version > runs/environment/nvcc.txt
claude --version > runs/environment/claude.txt
```

## Provider authentication

Use the same approved CLI configuration as your KDA run. The runner inherits
credentials and endpoints from your environment; it does not read token files,
change endpoints, print credentials, or install user-level plugins.

For Cortex-backed Claude, set `ANTHROPIC_BASE_URL` and the authentication variables
required by your team's existing Claude configuration. For Cortex-backed Codex,
use the existing Codex provider/profile configuration and its approved endpoint.
Do not assume setting OPENAI_BASE_URL alone selects the intended Codex provider.
Pass a configured profile with `--agent-arg=--profile --agent-arg=PROFILE` if needed.

```bash
export KHA_MODEL='YOUR_APPROVED_EXPLICIT_MODEL_ID'
./run.sh pilot --backend claude --model "$KHA_MODEL"
```

Claude uses `-p`, JSONL output and pre-approved read/edit/shell tools; Codex uses
`exec --json --sandbox workspace-write`. The runner does not silently disable
sandboxing. If your cluster needs different CLI permissions or provider options,
pass reviewed flags using repeated `--agent-arg=...`; they are saved in manifests.
Never put credential values in those flags. All trials must use the same settings.

Installed skills, hooks and memories can affect a CLI. Use a clean dedicated user
or an identical approved configuration for both arms. The runner's per-call fresh
contexts are intentional; this is not a reproduction of an uninterrupted KDA or
Claude session. For research, record the provider/model revision as well as its ID.

## One-task pilot before the campaign

On B200, `./run.sh setup` installs the tested FA4 dependencies in `.venv-gpu`.
If using an existing approved GPU Python environment, skip that step.

```bash
./run.sh check
./run.sh pilot --backend claude --model "$KHA_MODEL" --rounds 1
```

The pilot is bounded to 15 minutes by default. Cold compilation can take several
minutes; compiled kernels are reused across evaluator processes, while correctness
and timing are always rerun. After the pilot works:

```bash
./run.sh campaign --backend claude --model "$KHA_MODEL"
```

See [validation](validation.md) for measured hardware coverage and remaining limits.

## FastKernels task (including GEMM)

Start from an existing bench report to freeze the exact workload identities. A
reference report may contain failures; it supplies identities only. The supplied
seed must independently pass every selected case before any agent call is made.

```bash
kha prepare-fk --fk "$FK_DIR" --captures "$CAPTURES" \
  --report /path/to/bench.json --level 1 --stem bmm \
  --seed /path/to/correct/bmm.py --out experiments/local-bmm.json
kha evaluate --config experiments/local-bmm.json --gpu 0 --out runs/bmm-seed
CONFIG=experiments/local-bmm.json OUT_DIR=runs/bmm-pilot ./scripts/run_experiment.sh
```

The seed must be a standalone FastKernels candidate module preserving all baseline
class names/signatures. Local task configs contain paths specific to the GPU host;
do not confuse them with portable benchmark artifacts. Cases are identified by
`op`, `shape`, and `dtype`; duplicate or changed coverage is rejected. FastKernels'
own input generation governs correctness seeds; our flash input-seed setting does
not control it.

## Sharing results

Every trial contains manifest.json, events.jsonl, raw CLI logs, registered plans,
diagnostic evidence, candidate snapshots, independent evaluations, and the selected
best source. Campaigns include summary.csv, trajectory.csv, and report.html.
Keep failed and timeout runs. Share the source with `./scripts/package.sh` and share
the relevant run directory separately through your approved internal channel.

The runner refuses to overwrite a run. Use a fresh output path after a failure;
recovery/resume of partially executed rounds is not implemented. GPU locks are
cooperative: other jobs must use the same KHA_GPU_LOCK_DIR and device identifier.
Prefer physical GPU UUIDs to indices when containers remap devices. Thinking does
not hold a GPU; external processes that ignore the lease can still introduce noise.
