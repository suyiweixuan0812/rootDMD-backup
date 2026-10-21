"""Standalone configs for the attached pair-reward paper."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("dmd", Path(__file__).with_name("dmd.py"))
_dmd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dmd)


def get_config(name="dmd"):
    if name not in {"dmd", "pair_lambda0", "pair_lambda1"}:
        raise ValueError(name)
    cfg = _dmd.sd3m()
    cfg.method_name = name
    cfg.run_name = f"paper-pair-{name}-seed42"
    cfg.save_dir = "logs/paper-pair"
    cfg.seed = 42
    cfg.sample.num_steps = 4
    cfg.sample.denoising_step_list = [1000, 750, 500, 250]
    # ConfigDict assignment merges nested mappings; set every field explicitly
    # so the dmd control cannot inherit a stale cross-term coefficient.
    cfg.pair_reward.enabled = name != "dmd"
    cfg.pair_reward.lambda_cross = 0.0 if name == "pair_lambda0" else (1.0 if name == "pair_lambda1" else 0.0)
    cfg.pair_reward.pair_size = 2
    cfg.pair_reward.scorer = "soft_tifa"
    cfg.pair_reward.score_range = (0.0, 1.0)
    return cfg


def dmd():
    return get_config("dmd")


def pair_lambda0():
    return get_config("pair_lambda0")


def pair_lambda1():
    return get_config("pair_lambda1")
