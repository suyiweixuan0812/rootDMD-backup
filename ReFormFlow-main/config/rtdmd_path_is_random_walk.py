"""Pure Teacher path mapping; symmetric auxiliary conditional importance."""
import importlib.util
from pathlib import Path


def get_config(name='random_walk'):
    if name not in {'random_walk', 'random_walk_time'}:
        raise ValueError(name)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('details', root/'config/rtdmd_path_is_details.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = module.get_config('time')
    c.method_name = 'rtdmd_teacher_random_walk_path_is'
    c.run_name = 'sd3-medium-path-is-'+name
    c.posterior_time_weight.mode = 'inverse_sigma' if name.endswith('_time') else 'none'
    c.path_is.high_noise_block_size = 4096
    c.path_is.high_noise_threshold = .9
    c.path_is.max_innovation_radius = .5
    c.path_is.weighting = 'Exact Student/Teacher path ratio times observation likelihood times Gaussian innovation prior correction phi(E)/phi(E_reference)'
    c.path_is.particle_kernel = 'Symmetric random-walk auxiliary: C|E=N(E,delta^2I), Eprime|C=N(C,delta^2I); no OU contraction, no bridge field, no MH acceptance'
    c.path_is.target = 'Same Student random-stop posterior; pure Teacher CPS mapping and shared Student readout'
    return c
