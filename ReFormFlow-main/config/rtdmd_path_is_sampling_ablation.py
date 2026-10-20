"""Sampling detail ablation; same original path weights and lambda=.10."""
import importlib.util
from pathlib import Path


def get_config(name='sweeps8'):
    if name!='sweeps8':raise ValueError(name)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('path_is_refined',root/'config/rtdmd_path_is_refined.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config=module.get_config('train010')
    config.run_name='sd3-medium-path-is-train010-sweeps8'
    config.path_is.num_sweeps=8
    return config
