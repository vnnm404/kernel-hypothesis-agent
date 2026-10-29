# Kernel Hypothesis Agent

On a Linux B200 machine with CUDA 13 (`nvcc`), `g++`, `git`, `curl`, `claude`,
and your Snowflake PAT in `~/.snowhouse-pat`:

```bash
git clone https://github.com/vnnm404/kernel-hypothesis-agent.git
cd kernel-hypothesis-agent
./run.sh
```

The script downloads flash-cuda and sets up Python/dependencies automatically in `.deps/`,
benchmarks the seed, runs Claude, and checks the final kernel. First run takes several minutes.
Quick test without optimization: `./run.sh --smoke`.

Results: `runs/run-*/best.cu`, `result.md`, benchmark logs, and `trace.jsonl`.
