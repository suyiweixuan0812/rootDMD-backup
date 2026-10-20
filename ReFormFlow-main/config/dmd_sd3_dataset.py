"""Original SD3-Medium, standard DMD without rewards, on OCR or GenEval."""

import importlib.util
from pathlib import Path


def get_config(name="ocr"):
    counts = {"ocr": 1018, "geneval": 2212}
    if name not in counts:
        raise ValueError(f"Unknown DMD dataset: {name!r}")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "local_dmd_pickscore", root / "config/dmd_sd3_pickscore.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config("train")
    config.dataset = str(root / "dataset" / name)
    config.eval.dataset = config.dataset
    config.eval.dataset_name = {"ocr": "OCR", "geneval": "GenEval"}[name]
    config.eval.final_num_prompts = counts[name]
    config.num_epochs = 10000
    config.eval_freq = 1000
    config.save_freq = 1000
    config.run_name = f"sd3-dmd-{name}-16gpu-10000"
    return config
