# This invocation: plan only

Inspect the task and previous evidence. Write plan.json BEFORE editing the candidate
or running the proposed diagnostic. The controller saves it before execution in a
separate invocation. Prefer one coherent change or measurement per round.

Required JSON fields:
```json
{
  "kind": "optimization",
  "question": "What specific question does this attempt answer?",
  "change": "The proposed code change or diagnostic command",
  "expected_result": "Expected observable result, including metric and comparison"
}
```
Use kind=diagnostic for experiments whose purpose is evidence rather than a production
candidate. You may add hypotheses and test details. If a plateau intervention requires
them, follow its schema. Finish when plan.json exists. Do not implement yet.
