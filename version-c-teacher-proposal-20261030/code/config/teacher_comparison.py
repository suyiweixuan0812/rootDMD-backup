"""Matched wall-time comparison using the newly supplied ReFormFlow DMD."""
import os
from pathlib import Path
from ml_collections import ConfigDict
from config.dmd import get_config as base

def get_config(name='dmd'):
    assert name in ('dmd','teacher_c','smoke_dmd','smoke_teacher_c','ddp_dmd','ddp_teacher_c')
    smoke=name.startswith('smoke_'); ddp=name.startswith('ddp_')
    arm=name.replace('smoke_','').replace('ddp_','')
    c=base('sd3m')
    c.pretrained.model='/personal_fast_storage/v29_assets/stable-diffusion-3-medium-diffusers'
    c.pretrained.variant=''
    c.run_name=name;c.method_name='DMD (new supplied code)' if arm=='dmd' else 'Version C Teacher proposal'
    c.save_dir=os.environ['EXPERIMENT_OUTPUT']
    c.train.batch_size=1
    c.num_epochs=20 if smoke or ddp else (10000 if arm=='dmd' else 2000000)
    c.eval_freq=c.save_freq=10 if smoke or ddp else 1000
    c.eval.num_prompts=c.eval.final_num_prompts=20 if smoke or ddp else 1000
    c.sample.test_batch_size=2 if smoke or ddp else 4
    c.eval.reward_fn={};c.logging.enabled=False;c.resume_from=''
    c.execution=ConfigDict(dict(physical_world_size=int(os.environ.get('WORLD_SIZE',1))))
    c.experiment=ConfigDict(dict(arm=arm,smoke=smoke,ddp_preflight=ddp,
        matched_time=not(smoke or ddp),kappa=.5,particles=32,
        reference_dir=str(Path(c.save_dir)/'dmd'),
        teacher_variance='same_as_original_C_target_covariance',
        score_images=not ddp))
    c.bridge=ConfigDict(dict(calibration_per_rank=4,heldout_per_rank=2,refresh_every=200))
    return c
