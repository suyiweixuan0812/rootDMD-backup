"""Explicit numerical-control DMD; original DMD is separately preserved."""
import importlib.util
from pathlib import Path


def get_config(name="train"):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("original_dmd",root/"config/dmd_sd3_pickscore.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=module.get_config(name)
    config.method_name="dmd_fp32_observation_control"
    config.observation_noise_dtype="float32"
    return config
