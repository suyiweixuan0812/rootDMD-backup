"""Keep literal Teacher/Student path ratios; development on separate prompts."""
import importlib.util
from pathlib import Path
from ml_collections import ConfigDict


def get_config(name='train025'):
    if name not in {'smoke','train000','train025','baseline_eval'}:raise ValueError(name)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('path_is_dmd',root/'config/dmd_sd3_pickscore.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config=module.get_config('smoke' if name=='smoke' else 'train')
    weight=0. if name=='train000' else .25
    config.method_name='rtdmd_conditional_path_is'
    config.run_name=f'sd3-medium-path-is-{name}'
    config.save_dir=str(root/'logs/path-ratio-round1/runs')
    config.path_is=ConfigDict(dict(enabled=True,num_particles=8,num_sweeps=2,
        base_scale=.05,block_size=16,proposal_cfg=1.,field_microbatch=1,
        target='Actual Student random-stop source posterior; shared Student terminal readout',
        weighting='Student/Teacher positive-CPS prefix ratio times Gaussian observation likelihood',
        initialization='Paired Student reference with conditional Teacher-innovation lift',
        particle_kernel='Auxiliary conditional importance resampling; not MH acceptance',
        proposal_innovation_dtype='float64; BF16 network and latent states unchanged',
        numerical_scope='Gaussian rounding-cell model; finite RNG and source FP32 multiply/add remain numerical approximations',
    ))
    config.posterior_consistency=ConfigDict(dict(enabled=True,weight=weight,
        role='regularization ablation' if weight==0 else 'primary detail setting'))
    config.eval.dataset=str(root/'dataset/drawbench')
    config.eval.num_prompts=2 if name=='smoke' else 8
    config.eval.final_num_prompts=2 if name=='smoke' else 64
    config.eval_freq=20 if name=='smoke' else 250
    config.save_freq=20 if name=='smoke' else 1000
    if name=='baseline_eval':
        config.method_name='dmd_checkpoint_evaluation_only'
        config.run_name='dmd-ema-dev64'
        config.num_epochs=1
        config.lora.generator_path=str(root/'logs/dmd/sd3-medium-pickscore-no-reward-train/checkpoints/checkpoint-1000/lora/generator')
    return config
