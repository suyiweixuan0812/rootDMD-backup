import importlib.util
import os

base_path = os.path.join(os.path.dirname(__file__), "base.py")
spec = importlib.util.spec_from_file_location("base", base_path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def get_config(name):
    return globals()[name]()


def _get_config(base_model="sd3", n_gpus=8, gradient_step_per_epoch=1, dataset="pickscore", reward_fn=None, name="", num_image_per_prompt=24):
    config = base.get_config()
    assert base_model in ["sd3", "flux"]
    assert dataset in ["pickscore", "ocr", "geneval"]

    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    if base_model == "sd3":
        config.pretrained.model = os.environ.get("SD3_MODEL", "stabilityai/stable-diffusion-3.5-medium")
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.resolution = 512
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        bsz = 9
    elif base_model == "flux":
        config.pretrained.model = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
        config.mixed_precision = "bf16"
        config.sample.num_steps = 6
        config.sample.eval_num_steps = 28
        config.sample.guidance_scale = 3.5
        config.resolution = 512
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        bsz = 4

    config.sample.num_image_per_prompt = num_image_per_prompt
    num_groups = 48

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1
        
    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = 14 if dataset == "geneval" else 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"

    config.run_name = f"reformflow_{base_model}_{name}"
    config.save_dir = f"{config.logdir}/{base_model}/{name}"
    config.reward_fn = reward_fn
    config.train.adv_mode = "all"

    config.beta_weights = 40.0
    config.reward_clip = 5.0
    config.loss_in_fp32 = True
    config.group_center = True
    config.train.reform_residual_mode = "trajectory"

    config.use_lora = True
    config.mixed_precision = "bf16"
    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config


def sd3_pickscore():
    reward_fn = {
        "pickscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    return config


def sd3_geneval():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        dataset="geneval",
        reward_fn={"geneval": 1.0},
        name="geneval",
    )
    return config


def sd3_ocr():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=2,
        dataset="ocr",
        reward_fn={"ocr": 1.0},
        name="ocr",
    )
    return config


def sd3_hpsv2():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn={"hpsv2": 1.0},
        name="hpsv2",
    )
    return config


def sd3_ocr_group32():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=2,
        dataset="ocr",
        reward_fn={"ocr": 1.0},
        name="ocr_group32",
        num_image_per_prompt=32,
    )
    return config


def sd3_ocr_group8():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=2,
        dataset="ocr",
        reward_fn={"ocr": 1.0},
        name="ocr_group8",
        num_image_per_prompt=8,
    )
    return config


def sd3_pickscore_group32():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn={"pickscore": 1.0},
        name="pickscore_group32",
        num_image_per_prompt=32,
    )
    return config


def sd3_pickscore_group8():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn={"pickscore": 1.0},
        name="pickscore_group8",
        num_image_per_prompt=8,
    )
    return config


def sd3_multi_reward_geneval():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn={
            "pickscore": 1.0,
            "hpsv2": 1.0,
            "geneval": 1.0,
            "clipscore": 1.0,
        },
        name="multi_reward_geneval",
    )
    return config


def sd3_multi_reward_pickscore():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn={
            "pickscore": 1.0,
            "hpsv2": 1.0,
            "clipscore": 1.0,
        },
        name="multi_reward_pickscore",
    )
    return config


def sd3_multi_reward_ocr():
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=2,
        dataset="ocr",
        reward_fn={
            "pickscore": 1.0,
            "clipscore": 1.0,
            "ocr": 1.0,
            "hpsv2": 1.0,
        },
        name="multi_reward_ocr",
    )
    return config


# ===================== FLUX Configurations =====================


def flux_pickscore():
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn={"pickscore": 1.0},
        name="pickscore",
    )
    return config


def flux_geneval():
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn={"geneval": 1.0},
        name="geneval",
    )
    return config


def flux_ocr():
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=2,
        dataset="ocr",
        reward_fn={"ocr": 1.0},
        name="ocr",
    )
    return config


def flux_hpsv2():
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn={"hpsv2": 1.0},
        name="hpsv2",
    )
    return config


def flux_multi_reward_pickscore():
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn={
            "pickscore": 1.0,
            "hpsv2": 1.0,
            "clipscore": 1.0,
        },
        name="multi_reward_pickscore",
    )
    config.sample.num_steps = 10
    return config


def flux_multi_reward_geneval():
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn={
            "pickscore": 1.0,
            "hpsv2": 1.0,
            "geneval": 1.0,
            "clipscore": 1.0,
        },
        name="multi_reward_geneval",
    )
    config.sample.num_steps = 10
    return config


def flux_multi_reward_ocr():
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="ocr",
        reward_fn={
            "pickscore": 1.0,
            "clipscore": 1.0,
            "ocr": 1.0,
            "hpsv2": 1.0,
        },
        name="multi_reward_ocr",
    )
    config.sample.num_steps = 10
    return config
