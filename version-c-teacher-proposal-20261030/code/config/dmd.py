import importlib.util
import os
from pathlib import Path

from ml_collections import ConfigDict


_spec = importlib.util.spec_from_file_location("dmd_base", Path(__file__).with_name("base.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def get_config(name="sd35m"):
    presets = {"sd35m": sd35m, "sd3m": sd3m}
    if name not in presets:
        raise ValueError(f"Unknown DMD preset: {name!r}")
    return presets[name]()


def _get_config(model, variant):
    config = _base.get_config()
    config.method_name = "dmd"
    config.base_model = "sd3"
    config.pretrained.model = os.environ.get("SD3_MODEL", model)
    config.pretrained.variant = variant
    config.pretrained.local_files_only = True
    config.resolution = 512
    config.mixed_precision = "bf16"
    config.dataset = str(Path(__file__).resolve().parents[1] / "dataset/pickscore")
    config.prompt_fn = "general_ocr"
    config.save_dir = "logs/dmd"
    config.save_freq = 5000
    config.eval_freq = 1000
    config.sample.denoising_step_list = [1000, 750, 500, 250]
    config.sample.num_steps = len(config.sample.denoising_step_list)
    config.sample.random_stop_idx = True
    config.sample.noise_level = 0.9
    config.sample.test_batch_size = 4
    config.dmd = ConfigDict(dict(
        fake_update_ratio=10,
        gradient_normalization=True, real_guidance_scale_min=4.5,
        real_guidance_scale_max=4.5, fake_guidance_scale=1.0,
        fake_train_guidance_scale=1.0, min_step=20, max_step=980,
        fake_min_step=0, fake_max_step=999, fake_timestep_sampling="logit_normal",
        dmd_timestep_sampling="uniform", logit_mean=0.0, logit_std=1.0,
        dmd_logit_mean=0.0, dmd_logit_std=1.0,
    ))
    config.ac_dmd = ConfigDict(dict(
        enabled=False, end_timestep=1000.0,
        timestep_offset=10.0, min_renoise_step=20, max_renoise_step=980,
        all_include_terminal_when_no_random_stop=True,
        target_formula_move_scale_out=True, fake_loss_weight_clamp_min=0.0,
        fake_loss_weight_clamp_max=10.0, consistency_enabled=True,
        consistency_weight=0.01, consistency_epsilon_timestep=200.0,
        consistency_num_steps=1, consistency_use_two_sample_unbiased=True,
        consistency_stopgrad_transitions=True,
    ))
    # LoRA structure is fixed in the training script; paths are warm-start inputs only.
    config.lora = ConfigDict(dict(generator_path="", fake_score_path=""))
    # Each outer iteration consumes one prompt batch; both role updates reuse it.
    config.train.batch_size = 9
    config.train.learning_rate = 3e-5
    config.train.fake_learning_rate = 3e-5
    config.train.adam_weight_decay = 0.0
    config.train.warmup_steps = 0
    config.train.gradient_checkpointing = False
    config.train.log_interval = 1
    config.train.ema_decay = 0.99
    config.train.ema_every = 1
    config.train.max_sequence_length = 128
    config.logging = ConfigDict(dict(enabled=False, project="ReFormDMD"))
    config.eval = ConfigDict(dict(
        enabled=True, num_prompts=16,
        dataset=str(Path(__file__).resolve().parents[1] / "dataset/drawbench"),
        seed=42, use_ema=True, reward_fn={},
    ))
    return config


def sd35m():
    return _get_config("stabilityai/stable-diffusion-3.5-medium", "")


def sd3m():
    return _get_config("stabilityai/stable-diffusion-3-medium-diffusers", "fp16")
