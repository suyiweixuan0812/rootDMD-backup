# Independent pair-reward DMD suite

This directory is separate from the existing RTDMD v30/v31 work.  The attached
paper's primary local baseline is the unchanged four-step SD3 DMD/AC-DMD run.
The required controls are:

* `dmd`: unchanged DMD, same prompts, seeds, CPS schedule and update count;
* `pair_lambda0`: same AC-DMD training with paired scoring but `lambda=0`, which
  isolates any effect of pair sampling/scoring from the cross term;
* `pair_lambda1`: the proposed objective
  `R=(S0+S1)/2 + S0*S1`.

Pair-GRPO, CMO/CAST, Flash-DMD and DMAD are external literature baselines. They
must be reported as external references unless their implementations and model,
data, seed and compute budgets are reproduced locally.

Run the mathematical validation (CPU or A800):

```bash
cd /rootDMD/ReFormFlow-main
python research/rtdmd_reward/validate_pair_objective.py
```

The validation is a prerequisite before adding a VLM scorer to the DMD loop;
it checks the analytic gradient and the `lambda=0` control exactly.
