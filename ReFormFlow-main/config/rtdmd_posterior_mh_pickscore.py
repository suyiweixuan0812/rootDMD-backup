"""Teacher-guided posterior MH and the paper's same-observation Jacobian."""
import importlib.util
from pathlib import Path
from ml_collections import ConfigDict


def get_config(name='train'):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('baseline_pickscore_mh', root / 'config/dmd_sd3_pickscore.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config(name)
    config.method_name = 'rtdmd_posterior_mh'
    config.run_name = f'sd3-medium-rtdmd-posterior-mh-{name}'
    config.save_dir = str(root / 'logs/rtdmd-posterior-mh')
    config.posterior_mh = ConfigDict(dict(
        enabled=True, num_moves=8, teacher_shift=0.01, proposal_scale=2.38,
        scale_cycle=[0.5, 1.0, 2.0],
        gradient_site='same_observation_clean_prediction',
        initialization='paired_joint_source_trace',
        interpretation='Teacher-guided Gaussian-reference MH; not original reverse-CPS path IS',
    ))
    return config
