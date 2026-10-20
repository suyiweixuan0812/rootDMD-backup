"""v30 primary lambda=.25 and prespecified sensitivity lambda=.50."""
import importlib.util
from pathlib import Path
from ml_collections import ConfigDict


def get_config(name='train'):
    if name not in {'smoke', 'train', 'sensitivity'}:
        raise ValueError(name)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('previous_posterior_mh', root/'config/rtdmd_posterior_mh_pickscore.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config('smoke' if name == 'smoke' else 'train')
    weight = .50 if name == 'sensitivity' else .25
    label = '050' if name == 'sensitivity' else '025'
    config.method_name = 'rtdmd_pcmh'
    config.run_name = f'sd3-medium-rtdmd-pcmh-lambda{label}-' + ('smoke' if name=='smoke' else 'train')
    config.save_dir = str(root/'logs/rtdmd-pcmh')
    config.posterior_consistency = ConfigDict(dict(
        enabled=True, weight=weight,
        role='prespecified sensitivity' if name=='sensitivity' else 'prespecified primary',
        interpretation='Regularized posterior-MH surrogate; expected update differs from lambda=0',
    ))
    return config
