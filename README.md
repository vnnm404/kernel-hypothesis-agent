# Kernel Hypothesis Agent

On a B200, activate your working flash-cuda Python environment. Requires `claude`,
`nvcc`, and your Snowflake PAT in `~/.snowhouse-pat`.

```bash
git clone https://github.com/vnnm404/kernel-hypothesis-agent.git
cd kernel-hypothesis-agent
./run.sh /path/to/flash-cuda
```

For a quick test without optimization: `./run.sh --smoke /path/to/flash-cuda`.
The script benchmarks the seed, runs Claude, then independently checks the final kernel.

Results: `runs/run-*/best.cu`, `result.md`, benchmark logs, and `trace.jsonl`.
