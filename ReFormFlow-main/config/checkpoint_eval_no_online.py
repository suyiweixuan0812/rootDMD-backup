"""Skip redundant online PickScore; official four scores still use saved PNGs."""
import importlib.util
from pathlib import Path
from ml_collections import ConfigDict

def get_config(name='eval'):
    if name!='eval':raise ValueError(name)
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('original_eval_config',root/'config/rtdmd_path_is_pickscore.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    c=m.get_config('baseline_eval')
    c.eval.reward_fn=ConfigDict()
    return c
