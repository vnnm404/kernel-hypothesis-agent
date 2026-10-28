# Initial validation

Validated locally on macOS, October 27, 2026:

- 28 CPU tests passed: evaluator coverage and finite timings; per-case regressions;
  median-based promotion; diagnostic exclusion from promotion; preregistration;
  conditional updates; invalid seeds; token accounting; process-group timeout cleanup;
  lease contention; Flash adapter correctness-before-timing with a simulated GPU API;
  and vendored-file provenance.
- Three-arm synthetic campaign completed with saved plans, evidence, independent
  fixture evaluations, summaries, CSV trajectories, and HTML chart.
- Report rendered in Chromium with three trajectories, three trial rows and a
  prominent synthetic-data label.
- Source archive extracted into a separate temporary directory and independently
  ran its synthetic campaign. It excludes private runs, credentials, and virtualenvs.
- Skill frontmatter validation and Python lint/format checks passed.

Not validated here:

- CUDA compilation or runtime performance (no GPU/nvcc on this Mac).
- Actual FA3/FA4 imports and GPU-driver compatibility.
- Nsight collection and counter permissions on a Snowflake host.
- Paid Claude/Codex invocations, provider authentication, or model behavior.
- FastKernels GPU execution (adapter inspected against the recorded local version).
- Any claim that the hypothesis arm improves optimization efficiency or performance.

The first GPU-host pilot should run `kha doctor`, `kha evaluate` on the seed, and a
small agent trial before starting repeated campaigns. Keep and inspect failures;
do not treat the synthetic fixture's scores as research results.
