"""Two independent query views, same Teacher ratios and posterior kernels."""
import importlib.util
from pathlib import Path


def get_config(name='last_q2_time'):
    if name not in {'last_q2_time', 'paired_q2_time'}:
        raise ValueError(name)
    root = Path(__file__).resolve().parents[1]
    file = ('config/rtdmd_path_is_last.py' if name.startswith('last') else
            'config/rtdmd_path_is_details.py')
    preset = 'last_time' if name.startswith('last') else 'paired_time'
    spec = importlib.util.spec_from_file_location('query_base', root/file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = module.get_config(preset)
    if name.startswith('last'):
        c.method_name = 'rtdmd_teacher_last_query_average_path_is'
    c.run_name = 'sd3-medium-path-is-'+name
    c.path_is.query_views = 2
    c.path_is.query_view_policy = 'One source and sigma; two independent observations and particle RNG; mean gradient before one clipping/Adam update'
    c.path_is.query_view_cost = 'Twice the posterior/query evaluations; same prompt/source batch1, generator updates, LR, EMA and inference NFE'
    return c
