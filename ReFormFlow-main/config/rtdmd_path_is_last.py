"""Pure Teacher mapping and corrected symmetric i-SIR on final positive node."""
import importlib.util
from pathlib import Path


def get_config(name='last'):
    if name not in {'last', 'last_time'}:
        raise ValueError(name)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('random_walk', root/'config/rtdmd_path_is_random_walk.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = module.get_config('random_walk_time' if name.endswith('_time') else 'random_walk')
    c.method_name = 'rtdmd_teacher_last_innovation_path_is'
    c.run_name = 'sd3-medium-path-is-'+name
    c.path_is.bank_policy = 'last'
    c.path_is.prefix_cache = 'Exact field reuse only after unchanged-parent tensor equality checks'
    c.path_is.max_innovation_radius = .5
    c.path_is.particle_kernel = 'Symmetric conditional i-SIR on last positive innovation block; no change to Teacher path fields or Student/Teacher ratios'
    return c
