# Experiment and dataset format (schema version 1)

`./run.sh` compares two instruction conditions on the same pinned CUDA seed and
three fixed hd128 attention workloads: causal, noncausal, and GQA. Baseline receives
the common optimization task and logging instructions. Guided receives the same
text plus the instruction to state a falsifiable hypothesis and distinguish
competing explanations after two failed changes. This is an instrumented Claude
baseline, not KDA or the mentor's original autonomous harness.

Defaults: `PAIRS=6`, `BENCHMARKS=12`, `REPEATS=3`, `SECONDS_PER_RUN=3600`,
`ORDER_SEED=0`. Half the pairs run baseline first and half guided first; a seeded
shuffle assigns these orders. Odd pair counts have one extra baseline-first pair.
ORDER_SEED controls scheduling, not model sampling. Every session starts from the
same kernel in a fresh directory. Sessions are sequential, with fresh Claude
context, no session persistence, and safe mode disabling customizations. Managed
administrator policies still apply. Both arms have the same tools and resource
ceilings; actual tokens, duration, and benchmark usage can differ.

The 12-call search limit includes failed builds and repeated measurements. Calls
beyond it are rejected and logged. The one-hour cap covers the agent and its tool
calls, including compilation. Seed/final verification is outside that budget and
has a 15-minute limit per benchmark. Other diagnostic commands are allowed within
the time cap; only calls through `benchmark.sh` are scored and recorded as benchmarks.
This is cooperative instrumentation, not isolation against deliberate tampering.

## Files to return

```text
runs/experiment-<id>/
  experiment.json                configuration, order, source hashes, versions/hardware
  summary.csv                    one row per finished or failed trial
  measurements.jsonl             one JSON object per completed/interrupted benchmark
  pair-01-baseline/               analogous folders for every scheduled trial
    run.json                     identity, position, start times, agent exit/duration
    summary.json                 full machine-generated outcome and usage
    instructions.md              exact treatment prompt
    trace.jsonl                  raw Claude messages, tools/results, timestamps, usage
    agent.stderr                 CLI errors
    runner.log                   wrapper stdout/stderr
    result.md                    agent-written report; may be absent on failure
    notes.md                     agent-written experiment notes, when provided
    seed.cu / candidate.cu / best.cu
    seed.log / final.log          repeated verification output
    source-commit.txt             flash-cuda revision
    events.jsonl                 benchmark start/end and budget-rejection events
    measurements/0001/
      kernel.cu                  exact benchmark input snapshot
      benchmark.log              evaluator stdout AND stderr
      measurement.json           same end record as the aggregate JSONL
      notes.md                   notes as they existed before this benchmark, if any
```

The experiment folder is the dataset. Exclude `.fa4_trials/` (compiled binaries)
when transferring it. `.deps/` is outside the dataset and need not be returned.
Aggregate files update after each trial; raw per-trial events are written during
execution. If the whole wrapper is forcibly killed, retain the partial folder:
completed records still exist even if aggregates have not yet been updated.

## Measurement fields

Every benchmark-end object has these fields (JSON null means unavailable):

| Fields | Meaning |
|---|---|
| `schema_version`, `event` | `1`, `benchmark_end` |
| `trial_id`, `pair`, `condition` | Trial identifier, pair number, `baseline` or `guided` |
| `measurement_id`, `phase` | Zero-padded sequence within the trial; `seed`, `search`, or `final` |
| `started_at`, `finished_at` | UTC ISO-8601 timestamps |
| `elapsed_s`, `agent_elapsed_s` | Monotonic seconds at measurement start, relative to trial / agent start; agent time is null before agent launch |
| `duration_s`, `build_s` | Total benchmark time; compilation time reported by evaluator (when available) |
| `kernel_sha256`, `snapshot`, `log` | Input identity and file paths relative to the trial directory |
| `returncode`, `valid` | Evaluator exit code and whether all 3 workloads passed with positive timing data |
| `cases` | Per-workload objects described below; can be empty after interruption or missing source |
| `error` | Present for missing inputs or interrupted measurements |

Each `cases` object contains `case`, `correct`, `candidate_us`, `reference_us`,
and `speedup_vs_fa4`. The ratio is reference latency / candidate latency. A case
without timing has null latencies/ratio; missing case output has null correctness.
`valid=false` forbids treating the overall benchmark as a successful optimization,
even when some case measurements exist. Latencies are parsed from the upstream
printed aggregate timings (0.1 microsecond precision), not individual CUDA events.
Raw logs retain upstream diagnostics. `build_s` is absent on interrupted records.

Benchmark-start records contain the same identity/input/start fields, before
execution. Budget rejections have `event=budget_exhausted`, `timestamp`, and `phase`.
The runner marks started-but-unfinished benchmarks as interrupted after a trial
exits; it does not invent timings. Timeouts are also identified by agent exit code
124. Agent/tool trace events are Claude's native JSON schema, not normalized here.

For a performance-over-time chart, use search records with `valid=true`, matched
by kernel hash. Plot when the result became available: `agent_elapsed_s + duration_s`,
not just benchmark start. Notes snapshots and the trace explain the preceding
interaction. Counting search calls is a separate x-axis from elapsed time.

## Outcome fields

`summary.csv` has exactly these columns:

```text
trial_id,pair,condition,position,status,exit_code,actual_model,search_benchmarks,valid_search_benchmarks,final_speedup_vs_seed,worst_latency_ratio_vs_seed,qualifies,agent_duration_s,num_turns,reported_cost_usd
```

`position` is 1 or 2 within the pair. `status` is `completed` or `failed`; failures
are retained. Missing CSV values are empty, not zero. `summary.json` additionally
includes all run metadata, `finished_at`, `seed_median_us`, `final_median_us`, `usage`,
`model_usage`, and `agent_result_error`. Medians map workload names to microseconds.
`run.json` holds schema version, trial/pair/condition/position, seed hash, trial
start timestamps (`started_at`, Unix and monotonic seconds), and, once launched,
agent start timestamps, finish timestamp, duration and exit code.

For each workload, take the median of the 3 seed measurements and separately the
3 final measurements. `final_speedup_vs_seed` is the geometric mean of
seed median / final median across the 3 workloads. `worst_latency_ratio_vs_seed`
is the largest final median / seed median. `qualifies=true` requires a completed
run, >=1.02x geometric-mean improvement, and no workload >1.03x seed latency.
All repeated seed/final measurements must be valid; otherwise these scores are
null. `best.cu` is agent-selected; the wrapper measures it and marks qualification,
but does not silently replace a regressing selection with the seed.

Usage/cost/model fields come from Claude's trace. Reported dollar cost may be
absent or represent list prices rather than Snowflake billing. The manifest
records config, schedule, cases, seed SHA256, flash/wrapper Git revisions, hashes
of both runner files and all three evaluator files, Claude version/options,
GPU UUID/name/driver, selected GPU index and CUDA_VISIBLE_DEVICES mapping, nvcc
output, Python version, and installed package versions.
It does not dump environment variables or credentials.

## Expected volume and interpretation

For a completed default batch: **12 sessions, 6 per condition**, with 12 traces,
12 machine summaries, and up to 12 agent reports. With every search budget used:

- 144 search evaluations + 36 seed + 36 final = **216 benchmark records**.
- Each evaluates 3 workloads: **648 workload timing rows**, nested in `cases`.
- Up to **216 kernel snapshots**, plus the 36 top-level seed/candidate/best files.
- **12 CSV rows**, representing 6 paired comparisons.

Successful sessions that stop early still contribute 6 verification evaluations
but fewer search evaluations. Failures produce partial data and possibly missing
reports/timings. At the current ~574 KB seed size, the maximum snapshot and
top-level kernel copies occupy about **138 MiB uncompressed**, before traces,
logs, and build artifacts. Trace size is variable; no fixed archive size is promised.
Runtime is at most 12 agent-hours for the default batch, plus installation and
verification. Sessions may finish much sooner.

The independent replication is **6 sessions per condition**, not 648 independent
samples. These are repeated measurements of one already optimized kernel on three
workloads with the same benchmark input generation; timing repeats are not new
input-seed trials. They do not establish generalization to GEMM or reproduce the original
50% plateau. Compare paired outcomes, failures, and time/budget-to-improvement;
use the traces to annotate mechanisms. A two-arm result tests the added guidance
as a whole, not which individual question caused a breakthrough. Treat small or
single-measurement gains as exploratory until repeated. Keep the GPU otherwise idle.
