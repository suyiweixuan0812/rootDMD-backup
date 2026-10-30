# Version C Teacher endpoint proposal vs the newly supplied DMD

This experiment follows the user's selected endpoint-proposal variant of `VersionC_TeacherEndpointProposal_Beamer_CN.pdf`. It does **not** run a multistep Teacher rollout. It does not add a mixture head, mixture fitting, Teacher variance calibration, LPIPS or a mean-correction loss.

## Exact estimator

The original C target remains a single diagonal Gaussian `q_s = N(m_s, C_s)`, with `m_s = x_t - t v_student(x_t,t)`. The existing C covariance procedure is retained without changes. The proposal is `q_k = 0.5 q_s + 0.5 N(m_teacher,C_s)`, where `m_teacher` is the frozen Teacher's clean readout at the same noisy query. Sample 32 endpoints from `q_k`; weight each by the **full-dimensional density ratio** `q_s/q_k`; normalize across particles and return their weighted endpoint mean. Construct the shared-noise bridge algebraically from the endpoint and query. There is no path-density ratio.

The Teacher component is a proposal, not a replacement target. In the infinite-sample limit the estimate still targets the Student Gaussian mean. Finite self-normalized importance sampling is biased and can have poor overlap in 65,536 dimensions. Training logs therefore include Teacher particle count, Teacher weight mass, ESS, maximum weight, and error relative to the analytic target mean. The experiment does not assume improved image quality.

## Controlled comparison

- Baseline: the user-supplied `ReFormFlow-main.zip`; `code/scripts/train_dmd_sd3.py` is preserved unchanged. `train_teacher_comparison.py` adds measurement and the C branch. AC is disabled.
- Both start from the same SD3 Medium base and freshly initialized LoRA, seed 42. No old trained DMD or C checkpoint is reused.
- Eight physical GPUs, batch 1 per GPU for both methods. This is an explicit common override of the supplied DMD preset's batch 9. No virtual 16-stream accumulation.
- Four CPS steps, shifted timesteps `[1000,750,500,250]`, noise level 0.9, random-stop training, Student CFG 1, Teacher CFG 4.5. LoRA rank 32/alpha 64, learning rate 3e-5, EMA 0.99 per outer step.
- DMD trains its independent fake-score adapter every outer step and the generator every 10 steps. C does **not** train or query that fake-score adapter; its source estimate is the Student Gaussian estimator above. The inactive adapter remains allocated for identical initial state structure.
- DMD runs 10,000 outer steps / 1,000 generator updates. C runs until the first complete generator update crossing DMD's measured total training time. C is not constrained to the same update count.
- Count actual training, text encoding, original C covariance refresh, and all proposal computation. Exclude model loading, development/preflight, evaluation, checkpoint saving and downtime. Report actual time differences, never substitute interpolated values for evaluations.
- Evaluate the fixed DrawBench file (200 distinct prompts x 5 images = 1,000 images), 512 px, same seed/RNG stream layout, initial point and DMD every 1,000 steps; C follows these measured time targets. Save all PNGs and all per-image scores. Display fixed indices 0..19 (20 distinct prompts) locally in SwanLab.
- Scores use the supplied scoring code: CLIP x100, HPSv2 x100, PickScore x26, Aesthetic unchanged. Both arms use the same saved PNGs and scorer weights. No new FID reference is introduced in this experiment.

## Files

- `code/flow_grpo/single_gaussian_teacher_proposal.py`: original C target adapter.
- `code/flow_grpo/teacher_endpoint_proposal.py`: proposal sampling and full-density importance weights.
- `code/scripts/train_teacher_comparison.py`: outer DMD update and measured-time control.
- `code/flow_grpo/gaussian_bridge_c.py`: retained original covariance mechanism.
- `code/config/teacher_comparison.py`: formal and smoke configurations.
- `code/checks/check_numerics.py`: independent density oracle, bridge, uniform-weight and high-dimensional overlap checks.
- `orchestration/run_region10.py`: isolated 8-GPU job, DDP preflight, then DMD and C sequentially. It refuses duplicate runs.
- `orchestration/verify_and_plot.py`: CPU recomputation, paired initial states/images, checkpoint hashes, actual-time curves, prompt-cluster bootstrap intervals.

## Validation and environment

The development comparison includes two runs of the untouched supplied trainer and one instrumented run. Bitwise equality did not hold even between the two untouched runs on the GPU backend. The instrumented differences are reported alongside that measured repeatability range; this is not claimed as bitwise equivalence. Original loss/save functions have identical ASTs, the initial DMD loss and random-stop sequence match, and both arms' initial image hashes match. Full checkpoints save model, optimizers, schedulers, scalers, EMA, per-rank RNG and C covariance. Eight-GPU preflight checks synchronized states before the formal run.

The installed HPSv2 package contains an unused `from turtle import forward`, which requires an unavailable GUI library. `headless_hps.py` removes only that unused import in memory, validates that the imported name is not read, and logs the original file hash. It does not edit shared packages or change any score computation.

Model weights, generated data, credentials and SSH keys are not included. Set deployment paths for your environment before launching. GPU runs are only intended for an allocated experimental node. Results are single-seed exploratory comparisons, not a proof of an exact SD3 posterior.
