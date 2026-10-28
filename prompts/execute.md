# This invocation: execute the registered plan

Read plan.json and execute it without rewriting the plan. For an optimization, leave
the modified candidate for the controller to evaluate. For a diagnostic, save raw
observations under evidence/; the controller discards candidate changes from that
round. If blocked, preserve the failure evidence and describe the limit honestly.

Write outcome.json:
```json
{
  "observation": "What was actually observed, or what remains unmeasured",
  "interpretation": "What the observation does and does not establish",
  "next_action": "What should be tried or measured next and why",
  "evidence_files": ["evidence/actual-output.log"],
  "hypothesis_updates": []
}
```
Optimization timing may remain unmeasured until the controller runs it. A diagnostic
requires real saved evidence. When the plan lists required competing hypotheses,
update each with id, status (supported/refuted/inconclusive), evidence, conditions,
and revisit_when. A tool failure usually means inconclusive. Distinguish a failed
implementation from evidence against the broader idea.
