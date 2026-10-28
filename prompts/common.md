# Kernel experiment contract

Optimize the candidate for task.json. Preserve its complete API, supported inputs,
numerical behavior, and allowed implementation language. Work within this workspace.
The controller independently measures candidates and owns promotion. Its reported
scores, not estimates or your own summary, determine the best implementation.

Read task.json, state.json, and history/. The candidate is reset to the best accepted
implementation each round. History contains failed attempts as well as successes.
Use resources/ncu-report-skill/ for hardware profiling when applicable. The original
KDA process is: inspect the task, plan, implement a small candidate, validate, measure,
and retain evidence. Do not assume a slow kernel has a single bottleneck.

Edit only the candidate, plan.json, outcome.json, and diagnostic scripts/evidence.
Do not modify task.json, state.json, history/, resources/, runner/evaluator code,
correctness tolerances, benchmark workloads, reference implementation, or result files
outside this workspace. Do not inspect other trials, later checkpoints, archived
development traces, or a completed solution. Do not call another LLM or spawn agent
sessions: model usage must remain in the recorded invocation. Do not install packages
or change global agent configuration during a trial.

For every GPU command use `python with_gpu.py COMMAND ARGS...`. Reasoning processes
have no visible GPU; the wrapper leases the configured GPU and exposes it as cuda:0.
The wrapper saves command, duration, stdout, and stderr under evidence/. Run one
measurement at a time. Profiling counters guide diagnosis; report unprofiled latency
for performance. Keep workloads, reference versions, and input layouts comparable.

For FlashAttention a profiling driver is available:
`python with_gpu.py ncu --set full --profile-from-start off -o evidence/profile python profile.py --config task.json --candidate candidate.cu --case CASE --which candidate`
Use `--which reference` with a different output prefix for the matching reference.
If Nsight is unavailable or lacks permissions, save the error and use a different
test; do not manufacture counters. Record exact commands and raw output.

Never return a candidate that hardcodes answers, detects validation, skips input work,
relaxes numerical checks, or caches output for benchmark inputs. A temporary diagnostic
may alter semantics to test a bound, but label it diagnostic and never present it as
a valid optimization. Comparisons involving such changes need explicit caveats.
