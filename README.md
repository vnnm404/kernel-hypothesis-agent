# Kernel Hypothesis Agent

On a Linux B200 with CUDA 13 (`nvcc`), `g++`, `git`, `curl`, recent `claude`
(supporting `--safe-mode`), and your Snowflake PAT in `~/.snowhouse-pat`:

```bash
git clone https://github.com/vnnm404/kernel-hypothesis-agent.git
cd kernel-hypothesis-agent
./run.sh
```

Downloads flash-cuda and sets up Python automatically; allow 20 GB free disk.
Runs 6 baseline/guided pairs sequentially on GPU 0, with 12 search benchmarks and
1 hour maximum per agent session, plus 3 seed and 3 final measurements each.
Change scale: `PAIRS=12 BENCHMARKS=24 SECONDS_PER_RUN=7200 ./run.sh`.
Quick installation/agent check without optimization: `./run.sh --smoke`.

Results: `runs/experiment-*/summary.csv`, `measurements.jsonl`, and per-trial traces,
reports, and kernel snapshots. Return that experiment folder, excluding `.fa4_trials/`.
See [exact format and expected data volume](RESULTS.md). Use an otherwise idle GPU.
