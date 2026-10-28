"""In-repo, self-contained build / correctness / performance harness for the kernels in this repo.

Replaces the external agent harness: `bench.scenarios` (scenario matrix + constants + input and
fp32-reference construction) and `bench.eval` (build + A/B).

Two modules are the whole harness:

    eval.py -> scenarios.py     one A/B driver, `--target` selects the kernel/baseline pair:
                                  hopper    -> kernels/fa3_hopper.cu    vs the FA3 `_vllm_fa3_C` op
                                  blackwell -> kernels/fa4_blackwell.cu vs the FA4 CuteDSL oracle
                                (the oracle adapter lives in eval.py)

Everything else — NCU/nsys single-config drivers, the num_splits sweep and the reference-PTX dumper
— lives in `bench.microbenchmarks` and is never imported by the evaluator. (`bench.plot` renders the
README charts from a hardcoded snapshot; it imports nothing from the harness.)
"""
