from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking_reform import PerPromptStatTracker
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import swanlab
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import DataLoader
from flow_grpo.prompt_data import TextPromptDataset, GenevalPromptDataset, DistributedKRepeatSampler
from torch.utils.data.distributed import DistributedSampler
from flow_grpo.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.amp import GradScaler
from torch.amp import autocast

torch_autocast = partial(autocast, 'cuda')
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class LogZHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(256, in_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def build_train_timestep_schedule(pipeline, num_steps, timestep_fraction, device):
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(num_steps, device=device)
    scheduler_timesteps = scheduler.timesteps.to(device=device)

    scheduler_sigmas = getattr(scheduler, "sigmas", None)
    if scheduler_sigmas is not None:
        scheduler_sigmas = scheduler_sigmas.to(device=device, dtype=torch.float32)
        if scheduler_sigmas.numel() == scheduler_timesteps.numel() + 1:
            scheduler_sigmas = scheduler_sigmas[:-1]
        if scheduler_sigmas.numel() != scheduler_timesteps.numel():
            scheduler_sigmas = None

    if scheduler_sigmas is None:
        denom = float(getattr(scheduler.config, "num_train_timesteps", 1000))
        scheduler_sigmas = (scheduler_timesteps.float() / max(denom, 1.0)).clamp(0.0, 1.0)

    if scheduler_sigmas.numel() > 1 and scheduler_sigmas[-1].item() == 0:
        scheduler_sigmas = scheduler_sigmas[:-1]
        scheduler_timesteps = scheduler_timesteps[:scheduler_sigmas.numel()]

    num_train_timesteps = max(1, int(scheduler_sigmas.numel() * float(timestep_fraction)))
    train_timesteps = scheduler_timesteps[:num_train_timesteps].round().long()
    train_sigmas = scheduler_sigmas[:num_train_timesteps].float()
    return train_timesteps, train_sigmas


def build_flow_match_noisy(x0, sigma):
    if sigma.ndim == 0:
        sigma = sigma.expand(x0.shape[0])
    sigma = sigma.to(device=x0.device, dtype=x0.dtype)
    sigma_view = sigma.view(-1, *([1] * (x0.ndim - 1)))
    noise = torch.randn_like(x0)
    x_t = (1 - sigma_view) * x0 + sigma_view * noise
    return x_t, sigma_view


def expand_timestep_batch(timestep, batch_size, device):
    return torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def eval_fn(
    pipeline, test_dataloader, text_encoders, tokenizers, config,
    device, rank, world_size, global_step, reward_fn, executor,
    mixed_precision_dtype, ema, transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataloader.dataset.collate_fn,
        num_workers=0,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        cur_neg = sample_neg_prompt_embeds[:current_batch_size]
        cur_neg_pooled = sample_neg_pooled_prompt_embeds[:current_batch_size]

        with torch_autocast(enabled=(mixed_precision_dtype is not None), dtype=mixed_precision_dtype or torch.float32):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=cur_neg,
                    negative_pooled_prompt_embeds=cur_neg_pooled,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, _ = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}
        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            swanlab.log(
                {
                    "eval_images": [
                        swanlab.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                    **{f"eval_reward_{key}_std": np.std(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(save_dir, transformer_ddp, log_z_head, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler=None):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        transformer_ddp.module.save_pretrained(save_root_lora, selected_adapters=["default"])
        torch.save(log_z_head.module.state_dict(), os.path.join(save_root, "log_z_head.pt"))
        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None and scaler.is_enabled():
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)

        logger.info(f"Saved checkpoint to {save_root}")


def _checkpointed_transformer_forward(module, hidden_states, timestep, encoder_hidden_states, pooled_projections):
    return module(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        return_dict=False,
    )[0]


def main(_):
    config = FLAGS.config

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id

    config.save_freq = config.eval_freq
    config.save_dir = os.path.join(config.save_dir, config.run_name, unique_id)

    if is_main_process(rank):
        log_dir = config.save_dir
        os.makedirs(log_dir, exist_ok=True)
        _swanlab_resume_id = os.environ.get("SWANLAB_RESUME_ID", None)
        swanlab.init(
            project="ReForm", 
            experiment_name=config.run_name + '_' + unique_id,
            config=config.to_dict(), logdir=log_dir,
            resume=True if _swanlab_resume_id else False,
            id=_swanlab_resume_id,
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)

    reform_residual_mode = getattr(config.train, "reform_residual_mode", "trajectory")
    if reform_residual_mode not in ("trajectory", "per_timestep"):
        raise ValueError(
            f"Unsupported config.train.reform_residual_mode: {reform_residual_mode!r}; "
            "expected 'trajectory' or 'per_timestep'."
        )

    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=(mixed_precision_dtype == torch.float16))

    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1, disable=not is_main_process(rank),
        leave=False, desc="Timestep", dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if not config.use_lora:
        raise ValueError("This script requires config.use_lora=True")

    target_modules = [
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
        "attn.to_add_out", "attn.to_k", "attn.to_out.0",
        "attn.to_q", "attn.to_v",
    ]
    transformer_lora_config = LoraConfig(
        r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules,
    )
    if config.train.lora_path:
        transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
        transformer.set_adapter("default")
        if getattr(config.train, 'ref_resumed', False):
            transformer.load_adapter(config.train.lora_path, adapter_name="ref", is_trainable=False)
    else:
        transformer = get_peft_model(transformer, transformer_lora_config)

    transformer.set_adapter("default")

    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))

    train_timesteps, train_sigmas = build_train_timestep_schedule(
        pipeline=pipeline,
        num_steps=config.sample.num_steps,
        timestep_fraction=config.train.timestep_fraction,
        device=device,
    )
    num_train_timesteps = train_timesteps.numel()
    timestep_key_width = len(str(int(train_timesteps.max().item())))

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)

    log_z_head = LogZHead(in_dim=neg_pooled_prompt_embed.shape[-1]).to(device)
    log_z_head = DDP(log_z_head, device_ids=[local_rank], output_device=local_rank)

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=device)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            global_std=config.sample.global_std,
            reward_clip=float(getattr(config, 'reward_clip', 5.0)),
        )

    strict_trainable_parameters = transformer_trainable_parameters + list(log_z_head.module.parameters())
    optimizer = torch.optim.AdamW(
        strict_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError("Only general_ocr and geneval are supported")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.sample.test_batch_size,
        collate_fn=test_dataset.collate_fn, shuffle=False, num_workers=0, pin_memory=True,
    )

    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch

    logger.info("***** Running training (v2 — Trajectory Balance) *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Number of train timesteps = {num_train_timesteps}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):
            transformer_ddp.module.load_adapter(lora_path, adapter_name="default", is_trainable=True)
            if getattr(config.train, 'ref_resumed', False):
                transformer_ddp.module.load_adapter(lora_path, adapter_name="ref", is_trainable=False)
            transformer_ddp.module.set_adapter("default")
        else:
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        log_z_path = os.path.join(config.resume_from, "log_z_head.pt")
        if os.path.exists(log_z_path):
            log_z_head.module.load_state_dict(torch.load(log_z_path, map_location=device))

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and scaler.is_enabled():
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
            logger.info(f"Resumed global_step = {global_step}")
        except ValueError:
            logger.warning(f"Could not parse global_step from: {config.resume_from}. Starting from 0.")
            global_step = 0

    while True:
        #################### EVAL & SAVE ####################
        pipeline.transformer.eval()
        log_z_head.eval()

        if epoch % config.eval_freq == 0 and epoch > 0 and not config.debug:
            eval_fn(
                pipeline, test_dataloader, text_encoders, tokenizers, config,
                device, rank, world_size, global_step, eval_reward_fn, executor,
                mixed_precision_dtype, ema, transformer_trainable_parameters,
            )

        if epoch % config.save_freq == 0 and epoch > 0 and not config.debug:
            if world_size > 1:
                dist.barrier()
            if is_main_process(rank):
                save_ckpt(
                    config.save_dir, transformer_ddp, log_z_head, global_step, rank,
                    ema, transformer_trainable_parameters, config, optimizer, scaler,
                )

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        log_z_head.eval()
        samples = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device,
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt",
            ).input_ids.to(device)

            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype or torch.float32):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[:len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[:len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        deterministic=True,
                        solver="dpm2",
                        model_type="sd3",
                    )

            latents = torch.stack(latents, dim=1)
            x0 = latents[:, -1]

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples.append({
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "x0": x0,
                "rewards": rewards_future,
            })

        for sample in tqdm(
            samples, desc="Waiting for rewards",
            disable=not is_main_process(rank), position=0,
        ):
            rewards, _ = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=device).float()
                for key, value in rewards.items()
            }

        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()
            prompts_to_log = prompts
            rewards_to_log = samples["rewards"]["avg"][-len(images_to_log):].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):
                    img_data = images_to_log[idx].float()
                    pil = Image.fromarray(
                        (img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
                swanlab.log(
                    {
                        "images": [
                            swanlab.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]

        gathered_rewards = {
            key: gather_tensor_to_all(value, world_size).numpy()
            for key, value in samples["rewards"].items()
        }

        if is_main_process(rank):
            swanlab.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{key}": value.mean()
                        for key, value in gathered_rewards.items()
                        if '_strict_accuracy' not in key and '_accuracy' not in key
                    },
                    **{
                        f"reward_{key}_std": value.std()
                        for key, value in gathered_rewards.items()
                        if '_strict_accuracy' not in key and '_accuracy' not in key
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(samples["prompt_ids"], world_size)
            prompts_all = pipeline.tokenizer.batch_decode(prompt_ids_all.numpy(), skip_special_tokens=True)

            advantages, _, reward_hat = stat_tracker.compute_reform_targets(
                prompts_all, gathered_rewards['avg'], beta=1.0,
                center=getattr(config, 'group_center', True),
            )

            if is_main_process(rank):
                logger.info(f"len(prompts) = {len(prompts_all)}")
                logger.info(f"len(unique prompts) = {len(set(prompts_all))}")
                logger.info(f"reward_hat stats: mean={reward_hat.mean():.4f}, std={reward_hat.std():.4f}")

            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all, gathered_rewards)

            if is_main_process(rank):
                swanlab.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "reward_hat_mean": reward_hat.mean(),
                    },
                    step=global_step,
                )
            stat_tracker.clear()

            if getattr(config, 'filter_zero_advantage', False):
                _pa = np.array(prompts_all)
                _uniq, _inv = np.unique(_pa, return_inverse=True)
                group_std_flag = np.ones(len(prompts_all), dtype=np.float32)
                for _i in range(len(_uniq)):
                    if gathered_rewards['ori_avg'][_inv == _i].std() == 0:
                        group_std_flag[_inv == _i] = 0.0
            else:
                group_std_flag = np.ones(len(advantages), dtype=np.float32)
        else:
            advantages = (
                (gathered_rewards['avg'] - gathered_rewards['avg'].mean())
                / (gathered_rewards['avg'].std() + 1e-4)
            )
            reward_clip = float(getattr(config, "reward_clip", 5.0))
            reward_hat = np.clip(advantages, -reward_clip, reward_clip)
            group_std_flag = np.ones(len(advantages), dtype=np.float32)

        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        samples["advantages"] = (
            advantages.reshape(world_size, -1)[rank].to(device)
        )

        reward_hat = torch.as_tensor(reward_hat, dtype=torch.float32)
        samples["reward_hat"] = (
            reward_hat.reshape(world_size, -1)[rank].to(device)
        )

        group_std_flag_t = torch.as_tensor(group_std_flag, dtype=torch.float32)
        samples["group_std_flag"] = group_std_flag_t.reshape(world_size, -1)[rank].to(device)

        if is_main_process(rank):
            logger.info(f"advantages abs mean: {samples['advantages'].abs().mean():.4f}")

        del samples["rewards"]
        del samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch

        if getattr(config, 'filter_zero_advantage', False) and is_main_process(rank):
            n_effective = int(samples["group_std_flag"].sum().item())
            swanlab.log({"effective_sample_count": n_effective, "effective_batch_size": n_effective / num_batches}, step=global_step)

        total_batch_size = samples["x0"].shape[0]

        #################### TRAINING (Trajectory Balance) ####################
        effective_grad_accum_steps = config.train.gradient_accumulation_steps
        current_accumulated_steps = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=device)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(-1, total_batch_size // num_batches, *v.shape[1:])
                for k, v in samples.items()
            }
            samples_batched = [
                dict(zip(samples_batched, x))
                for x in zip(*samples_batched.values())
            ]

            transformer_ddp.train()
            log_z_head.train()
            info = defaultdict(list)

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                embeds = sample["prompt_embeds"]
                pooled_embeds = sample["pooled_prompt_embeds"]
                x0 = sample["x0"]
                B = x0.shape[0]
                group_weight = sample.get("group_std_flag", torch.ones(B, device=device))

                timestep_order = torch.randperm(num_train_timesteps, device=device)

                log_z = log_z_head(pooled_embeds.float())

                reward_hat_batch = sample["reward_hat"].detach()
                beta_reward = float(config.beta_weights) * reward_hat_batch

                delta_log_pi_sum = torch.zeros(B, device=device)
                step_delta_entries = []
                kl_sum = torch.zeros((), device=device)
                mse_theta_sum = torch.zeros((), device=device)
                mse_ref_sum = torch.zeros((), device=device)

                for step_idx, t_idx in enumerate(timestep_order):
                    timestep = train_timesteps[t_idx]
                    sigma = train_sigmas[t_idx]
                    timestep_batch = expand_timestep_batch(timestep, B, x0.device)
                    x_t, sigma_view = build_flow_match_noisy(x0, sigma)

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype or torch.float32):
                        v_pred = torch_checkpoint(
                            _checkpointed_transformer_forward,
                            transformer_ddp, x_t, timestep_batch, embeds, pooled_embeds,
                            use_reentrant=False,
                        )

                        with torch.no_grad():
                            if 'ref' in transformer_ddp.module.peft_config:
                                transformer_ddp.module.set_adapter("ref")
                                v_pred_ref = transformer_ddp(
                                    hidden_states=x_t,
                                    timestep=timestep_batch,
                                    encoder_hidden_states=embeds,
                                    pooled_projections=pooled_embeds,
                                    return_dict=False,
                                )[0]
                                transformer_ddp.module.set_adapter("default")
                            else:
                                with transformer_ddp.module.disable_adapter():
                                    v_pred_ref = transformer_ddp(
                                        hidden_states=x_t,
                                        timestep=timestep_batch,
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]

                    if config.loss_in_fp32:
                        x0_pred_theta = x_t.float() - sigma_view * v_pred.float()
                        x0_pred_ref = x_t.float() - sigma_view * v_pred_ref.float()
                        x0_loss = x0.float()
                    else:
                        x0_pred_theta = x_t - sigma_view * v_pred
                        x0_pred_ref = x_t - sigma_view * v_pred_ref
                        x0_loss = x0

                    spatial_dims = tuple(range(1, v_pred.ndim))

                    with torch.no_grad():
                        w_shared = (
                            torch.abs(x0_pred_ref.double() - x0_loss.double())
                            .mean(dim=spatial_dims, keepdim=True)
                            .clip(min=1e-5)
                            .float()
                        )

                    mse_theta = ((x0_pred_theta - x0_loss).pow(2) / w_shared).mean(dim=spatial_dims)
                    mse_ref = ((x0_pred_ref - x0_loss).pow(2) / w_shared).mean(dim=spatial_dims)

                    log_pi_theta = -mse_theta
                    log_pi_ref = -mse_ref

                    step_delta = log_pi_theta - log_pi_ref
                    delta_log_pi_sum = delta_log_pi_sum + step_delta
                    step_delta_entries.append((int(timestep.item()), step_delta))

                    if config.train.beta > 0:
                        if config.loss_in_fp32:
                            kl_diff = (v_pred.float() - v_pred_ref.float()) ** 2
                        else:
                            kl_diff = (v_pred - v_pred_ref) ** 2
                        kl_spatial = kl_diff.mean(dim=tuple(range(1, kl_diff.ndim)))
                        step_kl = (group_weight * kl_spatial).sum() / B
                        kl_sum = kl_sum + step_kl

                    with torch.no_grad():
                        mse_theta_sum += mse_theta.mean().detach()
                        mse_ref_sum += mse_ref.mean().detach()

                trajectory_residual = log_z + (delta_log_pi_sum / num_train_timesteps) - beta_reward

                with torch.no_grad():
                    residual_std = trajectory_residual.detach().std().clamp(min=1e-4)

                if reform_residual_mode == "trajectory":
                    policy_loss = (group_weight * (trajectory_residual / residual_std).pow(2)).sum() / B
                else:
                    local_residuals = torch.stack(
                        [log_z + step_delta - beta_reward for _, step_delta in step_delta_entries],
                        dim=0,
                    )
                    policy_loss = (
                        (group_weight * (local_residuals / residual_std).pow(2)).sum(dim=1) / B
                    ).mean()

                if config.train.beta > 0:
                    kl_loss = kl_sum / num_train_timesteps
                    loss = policy_loss + config.train.beta * kl_loss
                else:
                    kl_loss = None
                    loss = policy_loss

                scaled_loss = loss / effective_grad_accum_steps
                if mixed_precision_dtype == torch.float16:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                current_accumulated_steps += 1

                if is_main_process(rank) and i == 0:
                    logger.info(f"[ReForm] log_z: {log_z[:4]}")
                    logger.info(f"[ReForm] delta_log_pi_sum/T: {(delta_log_pi_sum / num_train_timesteps)[:4]}")
                    logger.info(f"[ReForm] beta_reward: {beta_reward[:4]}")
                    logger.info(f"[ReForm] trajectory_residual: {trajectory_residual[:4]}")
                    logger.info(f"[ReForm] residual_std: {residual_std:.4f}")
                    logger.info(f"[ReForm] policy_loss: {policy_loss.item():.4f}")
                    if kl_loss is not None:
                        logger.info(f"[ReForm] kl_loss: {kl_loss.item():.4f}")

                with torch.no_grad():
                    residual_detached = trajectory_residual.detach()
                    local_residuals_detached = torch.stack(
                        [log_z.detach() + step_delta.detach() - beta_reward for _, step_delta in step_delta_entries],
                        dim=0,
                    )
                    step_deltas_detached = torch.stack(
                        [step_delta.detach() for _, step_delta in step_delta_entries], dim=0
                    )
                    traj_raw_sq_loss = (group_weight * residual_detached.pow(2)).sum() / B
                    per_t_raw_sq_loss = (
                        (group_weight * local_residuals_detached.pow(2)).sum(dim=1) / B
                    ).mean()
                    step_delta_temporal_var = (
                        group_weight * step_deltas_detached.var(dim=0, unbiased=False)
                    ).sum() / B
                    decomposition_abs_error = (
                        per_t_raw_sq_loss - traj_raw_sq_loss - step_delta_temporal_var
                    ).abs()

                info["traj_raw_sq_loss"].append(traj_raw_sq_loss)
                info["per_t_raw_sq_loss"].append(per_t_raw_sq_loss)
                info["step_delta_temporal_var"].append(step_delta_temporal_var)
                info["decomposition_abs_error"].append(decomposition_abs_error)
                for (timestep_value, _), step_delta_row, local_residual_row in zip(
                    step_delta_entries, step_deltas_detached, local_residuals_detached
                ):
                    t_key = f"t{timestep_value:0{timestep_key_width}d}"
                    info[f"step_delta_mean/{t_key}"].append(step_delta_row.mean())
                    info[f"step_delta_std/{t_key}"].append(step_delta_row.std())
                    info[f"residual_abs_mean/{t_key}"].append(local_residual_row.abs().mean())

                info["policy_loss"].append(policy_loss.detach())
                info["loss"].append(loss.detach())
                info["trajectory_residual_abs"].append(trajectory_residual.abs().mean().detach())
                info["log_z"].append(log_z.mean().detach())
                info["delta_log_pi_per_t"].append((delta_log_pi_sum / num_train_timesteps).abs().mean().detach())
                info["beta_reward_abs"].append(beta_reward.abs().mean().detach())
                if kl_loss is not None:
                    info["kl_loss"].append(kl_loss.detach())

                if current_accumulated_steps % effective_grad_accum_steps == 0:
                    if mixed_precision_dtype == torch.float16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(strict_trainable_parameters, config.train.max_grad_norm)
                    if mixed_precision_dtype == torch.float16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                    log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info.items()}
                    info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                    dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                    reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}

                    if is_main_process(rank):
                        swanlab.log(
                            {"epoch": epoch, "inner_epoch": inner_epoch, **reduced_log_info},
                            step=global_step,
                        )
                    global_step += 1
                    info = defaultdict(list)

                    if config.train.ema and ema is not None:
                        ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        epoch += 1

    if is_main_process(rank):
        swanlab.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
