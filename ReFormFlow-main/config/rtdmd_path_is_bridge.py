"""Bridge proposal ablations; same Student posterior and explicit correction."""
import importlib.util
from pathlib import Path


def get_config(name="bridge"):
    if name not in {"bridge","bridge_time"}:raise ValueError(name)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("path_details",root/"config/rtdmd_path_is_details.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=module.get_config("time")
    config.method_name="rtdmd_bridged_conditional_path_is"
    config.run_name="sd3-medium-path-is-"+name
    config.posterior_time_weight.mode="none" if name=="bridge" else "inverse_sigma"
    config.path_is.bridge_strength=.9
    config.path_is.long_bridge_strength=.99
    config.path_is.long_bridge_min_stop=3
    config.path_is.high_noise_block_size=4096
    config.path_is.high_noise_threshold=.9
    config.path_is.max_innovation_radius=.5
    config.path_is.weighting="(Student/Teacher)*(Teacher/bridged-proposal)*Gaussian observation likelihood; effective Student/Q ratio computed directly"
    config.path_is.particle_kernel="Same auxiliary conditional i-SIR with deterministic noise-dependent block and radius bound"
    config.path_is.target="Unchanged actual Student random-stop source posterior; shared Student readout"
    return config
