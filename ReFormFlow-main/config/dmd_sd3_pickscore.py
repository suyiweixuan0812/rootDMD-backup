"""Single-A100 SD3-Medium DMD; PickScore is evaluation-only."""

import importlib.util
import os
from pathlib import Path

from ml_collections import ConfigDict


def get_config(name="train"):
    if name not in {"smoke", "train"}:
        raise ValueError(f"Unknown preset: {name}")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("local_dmd", root / "config/dmd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.sd3m()
    config.pretrained.model = os.environ.get("SD3_MODEL", str(root / "models/sd3-medium"))
    config.pretrained.local_files_only = True
    # Public mirror uses the unsuffixed filenames; hashes match official fp16.
    config.pretrained.variant = ""
    config.dataset = str(root / "dataset/pickscore")
    config.reward_fn = ConfigDict()  # No reward enters the training objective.
    config.ac_dmd.enabled = False
    config.logging.enabled = False
    config.train.batch_size = 1
    config.train.gradient_checkpointing = True
    config.sample.test_batch_size = 1 if name == "smoke" else 4
    config.eval.dataset = config.dataset
    config.eval.reward_fn = ConfigDict({"pickscore": 1.0})
    config.eval.num_prompts = 4 if name == "smoke" else 16
    config.eval.final_num_prompts = 4 if name == "smoke" else 2048
    config.num_epochs = 20 if name == "smoke" else 1000
    config.eval_freq = 20 if name == "smoke" else 100
    config.save_freq = config.eval_freq
    config.save_dir = str(root / "logs/dmd")
    config.run_name = f"sd3-medium-pickscore-no-reward-{name}"
    return config
