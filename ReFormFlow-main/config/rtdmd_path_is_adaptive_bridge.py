"""History-calibrated bridge; target and exact proposal correction unchanged."""
import importlib.util
from pathlib import Path


def get_config(name="adaptive_bridge"):
    if name not in {"adaptive_bridge","adaptive_bridge_time"}:raise ValueError(name)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("fixed_bridge",root/"config/rtdmd_path_is_bridge.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=module.get_config("bridge_time" if name.endswith("_time") else "bridge")
    config.run_name="sd3-medium-path-is-"+name
    config.method_name="rtdmd_adaptive_bridged_conditional_path_is"
    config.path_is.adaptive_bridge=True
    config.path_is.bridge_kl_budget=.25
    config.path_is.bridge_history_decay=.95
    config.path_is.bridge_policy="gamma(J)=max(0,1-sqrt(budget/PAST per-node KL envelope)); frozen before current source sampling"
    config.path_is.bridge_strength=0.
    config.path_is.long_bridge_strength=0.
    return config
