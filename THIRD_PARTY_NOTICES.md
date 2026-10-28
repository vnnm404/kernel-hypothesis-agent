# Third-party notices and reuse

## flash-cuda

Source: https://github.com/Snowflake-AI-Research/flash-cuda
Commit: `74af5ce7d2849f2c33f9017fa6f896b9c991a0e3`
License: Apache-2.0; included at `vendor/flash_cuda/LICENSE`.

The following files are copied byte-for-byte: `bench/eval.py`, `bench/scenarios.py`,
`bench/__init__.py`, `kernels/fa3_hopper.cu`, `kernels/fa4_blackwell.cu`.
Existing comments and notices are retained. The first-party adapter imports these
functions and applies additional strict coverage, input-seed, and timing checks.
The upstream evaluator's optional plotting, cuDNN and FlashInfer routes are not used.
CUDA kernels have not been modified by this project.

## ncu-report-skill

Source: https://github.com/mit-han-lab/ncu-report-skill
Commit: `d1887948c7d53690cfe6605f59c1329b8a1c6bb5` (KDA-pinned submodule).
Copyright (c) 2026 hanlab. MIT License.

Skill, reference documents and helpers are copied without modification under
`vendor/ncu-report-skill/`. The MIT license text distributed by KDA at
`third_party_licenses/MIT-hanlab.txt` is included as `vendor/ncu-report-skill/LICENSE`.

## KDA workflow prompt

Source: https://github.com/NVlabs/kda
Commit: `ef6ce61` (full revision in PROVENANCE.json).
The original `prompts/basic-flow.md` is archived at `vendor/kda/basic-flow.md`,
with the upstream license at `vendor/kda/LICENSE`. KDA licenses its first-party
documentation/prompts under CC BY 4.0. This project attributes the inspect/plan/
validate/record workflow to KDA. Our runtime prompt adapts that workflow and adds
a staged experiment protocol; it is not the original KDA prompt verbatim.

## KDA-on-FastKernels operational reference

Source: https://github.com/sfc-gh-goliaro/kda-agent
Commit: `a6f62e522448dc145c0a104ff403712d3de3cf6a`

Its local checkout was inspected for CLI invocation, existing provider setup,
FastKernels workspace layout, GPU leasing, and result records. No source files
from that wrapper are distributed: this project independently implements the
runner and lease utility. Employees with access can consult the original.

FastKernels is an external runtime dependency for the optional adapter, not vendored.
KernelWiki is optional external reference material and is not distributed here.
Claude Code, Codex, PyTorch, NVIDIA tools and FlashAttention reference packages are
external dependencies under their respective licenses.
