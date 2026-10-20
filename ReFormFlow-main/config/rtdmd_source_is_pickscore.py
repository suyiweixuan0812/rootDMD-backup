"""Source-posterior correction of Scheme A, derived from the local DMD preset."""
import importlib.util
from pathlib import Path
from ml_collections import ConfigDict


def get_config(name="train"):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("dmd_pickscore_baseline", root / "config/dmd_sd3_pickscore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.get_config(name)
    config.method_name = "rtdmd_source_is"
    config.run_name = f"sd3-medium-rtdmd-source-is-{name}"
    config.save_dir = str(root / "logs/rtdmd-source-is")
    config.source_is = ConfigDict(dict(
        enabled=True, num_particles=8, teacher_shift=0.01,
        interpretation="Corrected source-space SNIS; NOT reverse-CPS Scheme A",
    ))
    return config
