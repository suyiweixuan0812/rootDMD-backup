"""Reproduce the preserved failed numerical implementation; never a quality trial."""
import runpy
import flow_grpo.conditional_path_is_inverse_diagnostic as diagnostic
import flow_grpo.diffusers_patch.sd3_random_walk_path_is as sampler
diagnostic.DIAGNOSTIC_OUTPUT="/rootDMD/ReFormFlow-main/logs/path-ratio-round1/failed-attempts/last-inverse-bounds-20261020T220837Z/failure-numeric-state.json"
sampler.lift_student_prefix=diagnostic.lift_student_prefix
runpy.run_path("scripts/train_last_inverse_diagnostic.py",run_name="__main__")
