# Validation

## B200 pilot — October 28, 2026

Tested on one rented NVIDIA B200 (SM100, 183359 MiB), driver 580.126.09,
CUDA toolkit 13.0.88, Python 3.12.3, PyTorch 2.9.1+cu130, CUTLASS DSL 4.6.1,
and the pinned FA4 reference at df61ab6c4a0fb1f94f1f43b2a23479a0ab92b8ab.

- `./run.sh setup` completed on a fresh `nvidia/cuda:13.0.1-devel-ubuntu24.04`
  container after installing Python/venv, git, and ninja system packages.
- `./run.sh check` compiled the unmodified published seed and passed all three
  selected workloads with input seeds 0, 1, 2, followed by three timing repetitions.
- The first real build revealed a namespace bug missed by the mocked API tests:
  `TORCH_LIBRARY(FA3_LIB_NAME, ...)` registers the literal `FA3_LIB_NAME` namespace.
  The adapter and profiling driver now use that name. Vendored kernels remain unchanged.
- A cold build plus evaluation took 264 seconds. The runner now reuses compiled
  binaries across fresh evaluator processes, verifies their hashes, and keys them
  by source, target flags, namespace, Python/PyTorch/CUDA/compiler versions and
  vendored evaluator hashes. Correctness and latency are always measured afresh.
- Nsight Compute 2025.3.1 launched through the GPU lease wrapper and profiling
  driver, but collection failed with `ERR_NVGPUCTRPERM`. This host does **not**
  validate hardware-counter collection; the failure was retained as evidence.
- Codex CLI 0.165.0 authenticated through a temporary SSH tunnel to a loopback
  bridge on the local Mac. Login credentials stayed on the Mac. The rented
  container cannot create the namespaces needed by Codex's default sandbox;
  the dedicated-container test used the explicit
  `--agent-arg=--dangerously-bypass-approvals-and-sandbox` option. The repository
  default remains `workspace-write`; do not assume every host needs that override.

A real Codex optimization trial completed one round: preregistered plan → source
edit → fresh parent/candidate correctness and timing → rejection → saved report.
The agent added three `prefetch.tensormap` hints. All nine candidate correctness
checks passed. Relative speedup versus the freshly measured parent was
**1.000491× (~0.049%)**, below the 2% threshold, so the incumbent was preserved.
The trial took 378 seconds; parent re-evaluation took 12.4 seconds (binary cache
hit), and the modified candidate took 268 seconds (cold build). The trial's saved
configuration explicitly requested one source optimization and disclosed unavailable
counters to exercise this path; it is not an unsteered efficacy comparison.

After the final fixes, all 36 tests passed on the B200 host, `./run.sh check` passed
again in 13.1 seconds, and both candidate/reference profiling drivers executed
successfully without NCU. The original diagnostic bundle also passed the revised
validator when replayed offline; its failed historical run was not rewritten.
Hardware-counter collection remains blocked by host permissions.

Raw measurement samples, the optimization plan/diff/decision, and validation
metadata are in [the B200 evidence bundle](validation/b200-2026-10-28.json).

The measured seed score is specific to these three small workloads and the installed
reference environment. It is neither a full FA4 performance claim nor an
agent-generated speedup. This is a functional pilot, not a controlled efficacy study.

The first diagnostic round exposed two additional integration issues: listing an
empty stderr file caused an otherwise substantive evidence bundle to be rejected,
and a generated `profile.py` shadowed Python's standard library during reference
imports. Evidence validation now permits empty ancillary logs while requiring at
least one nonempty file, and the driver is named `profile_kernel.py`. A run with
zero completed rounds now exits unsuccessfully instead of silently returning zero
when its round budget expires. The original failed trial is retained as a failure.

## Local checks

- 36 CPU tests pass: strict evaluator coverage and finite timings, per-case
  regressions, median-based promotion, diagnostic exclusion from promotion,
  preregistration, hypothesis updates, invalid seeds, token accounting,
  process-group cleanup, GPU lease contention, adapter ordering, vendor provenance,
  quick-start budgets and failures, binary-cache invalidation, legitimate empty stderr,
  Python profiler imports, and nonzero exit status when every round fails.
- Python lint/format checks and shell syntax checks pass.
- The three-arm synthetic campaign completes with logs, evidence, CSVs and an HTML
  report. Synthetic scores are fixtures and provide no GPU performance evidence.
- During initial bring-up, the report was rendered in Chromium and the source
  archive was extracted and independently exercised with the synthetic campaign.

The instance was destroyed after about 25 minutes and verified absent from the
Vast.ai instance list. Estimated GPU/container compute was $2.60, excluding small
storage/data-transfer charges; this is an estimate, not a billing invoice.

## Still unvalidated

- Hopper/FA3 and FastKernels GPU execution.
- Claude Code and Snowflake/Cortex provider integration.
- Successful Nsight counter collection on a host that grants permission.
- Generalization to other workloads, repeated multi-arm GPU campaigns, and any
  claim that hypothesis testing improves optimization efficiency or performance.
