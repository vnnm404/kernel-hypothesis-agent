Improve candidate.cu using this existing FlashAttention benchmark.

Start by reading seed.log and the source. Work only in this run directory.
Keep seed.cu, benchmark.sh, bench/, instructions.md, and existing logs unchanged.
Preserve the kernel API, supported inputs, and numerical behavior. Use CUDA and
inline PTX; do not replace the kernel with library calls or specialize on input values.

Use ./benchmark.sh to check correctness and measure all three workloads. It saves
each output in measurements/. Run measurements sequentially on the selected GPU.
Make at most three optimization attempts, keeping a source snapshot for each.
Before each change, write down what you think limits performance, the evidence,
and what result would disprove that explanation.

If two changes fail to improve performance, stop guessing. Name two competing
explanations and run the smallest test that distinguishes them. Record what the
test establishes, what remains uncertain, and how it changes your next step.
If profiling counters are unavailable, save the error and use another test.
Treat deliberately incorrect ablations only as diagnostics, never as improvements.

Recheck promising changes against the previous best using repeated runs. Keep a
change only if all cases pass, geometric-mean latency improves by at least 2%, and
no case gets more than 3% slower. Save the best verified version as best.cu;
leave it equal to seed.cu if no change qualifies. These decisions are your
responsibility; the wrapper only checks the seed and final best with the benchmark.

Write result.md with the tested explanations, changes, per-case measurements,
failed attempts, evidence filenames, and next experiment. Distinguish measured
results from guesses. Do not change the environment, install packages, or call
other agents. Stop after the three attempts and report honestly if blocked.
