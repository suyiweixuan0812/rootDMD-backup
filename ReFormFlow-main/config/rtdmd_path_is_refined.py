"""Detail ablations selected using independent DrawBench development prompts."""
import importlib.util
from pathlib import Path


def get_config(name="train010"):
    if name not in {"train010", "train_sigma"}:
        raise ValueError(name)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("path_is_base", root / "config/rtdmd_path_is_pickscore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config("train025")
    config.run_name = "sd3-medium-path-is-" + name
    config.posterior_consistency.weight = .1 if name == "train010" else .5
    config.posterior_consistency.schedule = "constant" if name == "train010" else "linear_alpha"
    config.posterior_consistency.role = "Independent-development regularization ablation; unchanged path sampler"
    return config
