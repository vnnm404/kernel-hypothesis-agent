# Experiment: does explicit hypothesis testing help escape plateaus?

## Claim and scope

Hypothesis: explicitly testing competing explanations after optimization stalls
reduces cost to a correct performance improvement, relative to the same agent,
tools and resources without the intervention. Faster kernels are a possible
outcome, not an assumption. A negative result is informative.

The historical traces motivate the mechanism. They are a selected observational
sample, with missing raw measurements, parallel work, and unreliable elapsed time
in some sessions. They cannot establish that human questions caused gains or that
an autonomous agent could not have found the same improvement. The original 50%
run has not been identified.

## Pilot protocol

1. Validate the GPU environment and seed independently. Freeze workload identities,
   correctness tolerances, reference versions, candidate language constraints and
   scoring before examining treatment outcomes.
2. Select checkpoints before known breakthroughs, including failures that did not
   lead to a breakthrough. Do not construct the benchmark solely from famous wins.
   Keep later code, retrospective notes and successful patches out of agent inputs.
3. Develop the procedure on some sessions; evaluate on held-out kernel families
   and independent tasks. Adjacent checkpoints of one trajectory are not independent
   samples. Published optimized seeds currently supplied here are bring-up tasks,
   not substitutes for this checkpoint dataset.
4. Run baseline/hypothesis and, when budget permits, rethink with identical model,
   tool access, reference access and limits. Default 3 repetitions is a pilot,
   not a power calculation. Use more repetitions and multiple tasks for inference.
5. Record all runs, including invalid seeds, errors, timeouts and no-improvement
   trajectories. Randomized order reduces simple drift confounding but does not
   eliminate contention, cache effects or model-service drift.

Both arms may profile, inspect assembly, and perform diagnostics voluntarily. The
treatment adds a mechanically triggered, preregistered diagnostic/revision step.
Its overhead is charged to the same time/round budget. That is the intervention
being tested. The KDA-inspired control omits Humanize in both arms; do not label it
as a full KDA benchmark. Test that separately if a paper claims superiority to KDA.

## Primary outputs

- Correct speedup at a fixed budget, with task and repetition distributions.
- Time/cost to reach a **predeclared** improvement target; unsuccessful runs are
  censored at the budget, not dropped. Config target_speedup is recorded, and raw
  trajectories allow this analysis; the runner does not early-stop upon reaching it.
- Fraction of runs reaching the target; fraction that finish with valid candidates.
- Diagnostic overhead, accepted improvements, repeated failed families, and whether
  reported hypothesis revisions agree with their evidence (requires human audit).

Token usage includes both planning and execution, with cache accounting normalized
where the CLI reports it. There is no reviewer in this version. Unknown usage stays
unknown. Fixed rounds and fixed wall-time answer different questions; analyze them
separately instead of presenting equal rounds as equal compute.

The generated HTML displays individual measured incumbent trajectories and does
not calculate confidence intervals or significance. For the full study, aggregate
within task, use task-level resampling or an appropriate hierarchical analysis,
and report variability instead of treating every checkpoint as independent.

## Follow-up ablations

- Generic rethink versus explicit hypotheses: controls for additional attention.
- Same intervention at a fixed schedule versus only at plateaus: tests trigger value.
- Conditional failure memory versus plain history: tests when old negatives should
  be revisited. Not separately implemented in the current three-arm pilot.
- Reviewer feedback with identical total model budget: tests the benefit of a second
  perspective beyond additional compute. Not included in this implementation.

## Limits to resolve before paper claims

Historical checkpoint recovery, a held-out evaluation suite, multiple kernel
families (e.g. FastKernels bmm/GEMM plus attention), hardware replication, repeat
measurements on idle devices, and independent outcome annotation remain required.
The numeric thresholds here are heuristic; two percent may be below or above
measurement noise on a particular host. Calibrate using repeated unchanged-seed
measurements first. Do not retune thresholds after seeing which arm wins.

The reference launch path affects short-kernel timings. Report kernel-only and
dispatch-inclusive results separately if making architectural claims. Input layout
assumptions, including contiguous page tables, need explicit coverage.
