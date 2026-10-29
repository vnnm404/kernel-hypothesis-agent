# Kernel Hypothesis Agent

A runnable experiment in escaping GPU-kernel optimization plateaus. It layers an
explicit hypothesis-testing procedure over Claude Code or Codex and compares it
against the same coding agent with the same tools and a normal optimization loop.

**This is an experimental agent, not a demonstrated improvement.** The CPU smoke test proves controller plumbing only. See
[validation notes](docs/validation.md) for exactly what has been exercised on real hardware.

## Quick start

On a **B200 Linux host with CUDA 13** (`nvcc` on PATH), Python 3.10+, and git:

```bash
git clone https://github.com/vnnm404/kernel-hypothesis-agent.git
cd kernel-hypothesis-agent
./run.sh setup
./run.sh check
./run.sh pilot --backend codex --model YOUR_APPROVED_MODEL_ID
```

`setup` installs the GPU dependencies and pinned FA4 reference into `.venv-gpu`.
`check` compiles the CUDA seed, checks all three workloads against the reference
with three input seeds, and measures performance. **It makes no model calls.**
`pilot` uses your already-authenticated Codex CLI; use `--backend claude` for Claude
Code. It runs at most three rounds within 15 minutes (including compilation and
agent calls). Both planning and implementation count toward that wall budget.
Use `--rounds 1 --minutes 10` for a smaller first trial. Results and an HTML report
are saved under `runs/`; the command prints the report path.

Once the pilot works, run the comparison:

```bash
./run.sh campaign --backend codex --model YOUR_APPROVED_MODEL_ID
```

That runs three repetitions of each arm using the larger budgets in the task JSON.
It can take hours. The short pilot is a functional check, not evidence of efficacy.
Scripts do not rent GPUs or install/authenticate a coding CLI for you.

Already have a working flash-cuda environment? Skip `setup`, install with
`python -m pip install -e .`, and set `PYTHON_BIN` to that environment's Python
(and `KHA_VENV` to a nonexistent path if `.venv-gpu` exists). On Hopper use
`--config experiments/hopper.json` with an existing FA3/vLLM environment;
**the automatic setup script currently targets B200 only**.

CPU-only check: `python -m pip install -e '.[dev]' && ./run.sh smoke`.
[Snowflake setup](docs/snowflake.md) covers existing Cortex authentication,
FastKernels tasks, profiling, and sharing results.

## What changes between arms

| Arm | Behavior |
| --- | --- |
| `baseline` | Inspect → plan → implement/profile → independent validation and timing → keep/reject. |
| `hypothesis` | Same loop; after two rounds without an accepted improvement, require a preregistered experiment distinguishing at least two explanations, then revise their status from saved evidence. |
| `rethink` | Same baseline with a generic “reconsider your assumptions” reminder when stalled. Optional attention/control ablation. |

The default campaign runs baseline and hypothesis. Add `--arms baseline hypothesis
rethink` for the third arm. Arms share the seed, workload list, model, resources,
evaluation thresholds and budget. Order is shuffled within each repetition using
a saved schedule. Every phase starts a fresh CLI context with explicit history.

A diagnostic round can be valuable without improving speed. Its code is **never
promoted**. Hypotheses retain their operating conditions and when they should be
revisited. The controller checks schema, records, and evidence existence; it does
not claim to automatically verify causal reasoning or the truth of agent-written
interpretations.

This is a **KDA-inspired baseline**, not an exact reproduction of KDA/Humanize. Both
arms use the same Nsight skill; neither uses the optional Humanize/Codex reviewer.
KernelWiki can be added to both via `resources` in the task JSON. No original 50%
autonomous run has been reconstructed, and no historical breakthrough answer is
included in the agent prompts.

## Measurements and acceptance

- Vendored flash-cuda CUDA seeds, scenario definitions and evaluator at a recorded
  commit. Adapter calls the actual correctness and timing functions; it does not
  scrape rounded console scores.
- Flash correctness across three input seeds by default. Every selected case must
  be correct, supported and have a working FA3/FA4 reference. Missing or skipped
  cases fail the evaluation rather than disappearing from a geometric mean.
- Compiled CUDA binaries are cached by source, target flags, compiler/PyTorch versions,
  and evaluator hashes. Cache hits verify the binary hash. Measurements and numerical
  checks are always rerun; delete `runs/.build-cache` to force a cold build.
- Independent evaluator subprocess and source snapshots. Parent and candidate are
  remeasured on the same selected GPU, with order alternated across rounds.
- Three timing repetitions; acceptance requires >=2% geometric-mean improvement
  over the remeasured parent and <=3% latency regression on every selected case.
  These are configurable pilot thresholds, **not a statistical significance test**.
- The metric includes the source harness's dispatch/launch behavior; it is not
  isolated kernel cycles. NCU runs are diagnostic evidence, not accepted timings.
- Round and wall limits are enforced; failed plans and evaluations consume budget.
  `--token-limit` is a soft cap checked between calls. Missing usage stops a
  token-capped run rather than reporting zero. No dollar-budget enforcement.

The supplied seed kernels are the published, already-optimized implementations.
They are useful for bring-up but are not the pre-breakthrough checkpoints needed
for a causal plateau study. Replace `seed` with a recovered checkpoint for that
experiment. Input seeds test numeric variation within selected shapes; they are
not a held-out workload suite. See [experiment design](docs/experiment.md).

## Outputs

```text
runs/b200-pilot/
  schedule.json
  summary.csv, trajectory.csv, report.html
  rep-00-hypothesis/
    manifest.json, events.jsonl, state.json, summary.json
    best.cu
    workspace/                 # candidate, task, history, shared tools
    rounds/001/                # plan, outcome, raw CLI events, evidence, candidate
    evaluations/               # independent logs, JSON results, source snapshots
```

The HTML chart groups trajectories by arm and marks diagnostic rounds. Synthetic
reports are prominently labeled. Failed runs are retained. Inspect raw observations
before treating an agent's “refuted” or “supported” label as scientific evidence.

```bash
kha report runs/b200-pilot
python -m pytest -q
./scripts/package.sh            # source archive in dist/, excludes runs and secrets
```

Run directories are never overwritten. Partial-run resume is not implemented;
start a new output directory. The controller is an experiment harness, **not an
adversarial isolation boundary**. Run on a dedicated machine/container; the same
user owns the agent and evaluator processes. Cooperative GPU locks do not prevent
unrelated jobs from using the device.

## Sources

- [flash-cuda](https://github.com/Snowflake-AI-Research/flash-cuda): vendored evaluator,
  workload definitions and CUDA seeds (Apache-2.0).
- [KDA](https://github.com/NVlabs/kda): workflow reference, archived prompt, and
  [ncu-report-skill](https://github.com/mit-han-lab/ncu-report-skill) (MIT).
- [KDA-on-FastKernels](https://github.com/sfc-gh-goliaro/kda-agent): setup and GPU-lease
  pattern reference. Its private wrapper code is not redistributed.

Exact revisions, hashes and attribution are in [PROVENANCE.json](PROVENANCE.json)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). First-party code is Apache-2.0;
vendored components retain their licenses.
