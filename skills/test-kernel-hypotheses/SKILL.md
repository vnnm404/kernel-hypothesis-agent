---
name: test-kernel-hypotheses
description: Design and interpret discriminating experiments when GPU kernel optimization stalls. Use with the kernel-hypothesis-agent experiment workspace and its plan/outcome protocol.
---

# Test kernel hypotheses

The current implementation has stopped improving. Spend this round testing its
explanation before proposing another optimization. A diagnostic that rules out an
expensive dead end can be useful even if its measured latency does not improve.

Use the current evidence to name at least two plausible competing mechanisms.
Prefer explanations that predict different observable outcomes for one affordable
test. Choose the smallest comparison that can change the next decision: matched
workload/layout contrasts, source-to-generated-code differences, a controlled
ablation, isolated versus integrated components, or a bound on a suspected cost.
These are options, not a mandatory checklist. Keep numerical-invalid ablations
explicitly diagnostic; their timings are never eligible for promotion.

Before executing, write plan.json with kind=diagnostic and the usual question,
change, and expected_result fields, plus:

```json
{
  "hypotheses": [
    {
      "id": "h1",
      "mechanism": "A concrete causal explanation",
      "prediction": "The observable result if this explanation dominates",
      "falsifier": "The result that would weaken this explanation",
      "conditions": "Workload, layout, implementation and hardware where it applies"
    },
    {
      "id": "h2",
      "mechanism": "A different explanation of the same evidence",
      "prediction": "A distinguishable predicted result",
      "falsifier": "Evidence against this explanation",
      "conditions": "Relevant operating conditions"
    }
  ],
  "discriminating_test": "Exact comparison, controlled variables, and measured signals",
  "decision_rule": "How each possible outcome changes the next action"
}
```

After the execution invocation, outcome.json must update every hypothesis with:
id, status (supported/refuted/inconclusive), evidence, conditions, and revisit_when.
Cite saved raw evidence. Explain ambiguity rather than turning correlation into
certainty. If the test failed to run, record the error and mark the explanation
inconclusive. Do not claim an architectural ceiling just because several patches
failed. Revisit negative results when the implementation or bottleneck changes.

The next optimization round should act on the updated explanation. Preserve this
conditional memory in the outcome; the controller carries it into future history.
