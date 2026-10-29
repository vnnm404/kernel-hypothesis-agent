# Kernel Hypothesis Agent

A small wrapper that asks Claude to optimize a CUDA attention kernel. When progress
stalls, it asks for a test of the explanation before trying another change.

On a B200 host, activate the Python environment you already use for **flash-cuda**,
with your normal Claude login, then run:

```bash
git clone https://github.com/vnnm404/kernel-hypothesis-agent.git
cd kernel-hypothesis-agent
./run.sh /path/to/flash-cuda
```

No package installation or new environment. The script checks the GPU and reference,
benchmarks the seed, calls Claude, and checks the final kernel. It works on copies;
your flash-cuda checkout is left unchanged. Run one job per GPU.

Results are under `runs/run-*/`: `best.cu`, `result.md`, benchmark logs, and the
conversation trace. Claude is asked to try at most three changes and save its best
verified result. This is a prompt-guided run; attempt limits and improvement
selection are not enforced by a separate controller.

Optional: `GPU=1 ./run.sh /path/to/flash-cuda` selects a GPU;
append `--model YOUR_MODEL` to choose a model instead of your configured default.
`./run.sh --check /path/to/flash-cuda` runs only the GPU check, without Claude.
Set `PYTHON_BIN` if your working environment uses a different Python executable.

The earlier full harness passed an end-to-end **Codex test on a rented B200**,
not Snowflake's host. That tested version and its evidence are preserved at
[`b200-harness-v1`](https://github.com/vnnm404/kernel-hypothesis-agent/tree/b200-harness-v1).
This smaller Claude wrapper has local orchestration tests; its complete Claude/GPU
path has not been rerun. It uses the existing
[flash-cuda](https://github.com/Snowflake-AI-Research/flash-cuda) benchmark.
