"""Standard DMD and optional AC-DMD training for the SD3 model family."""

from collections import defaultdict
from functools import partial
import json
import logging
import math
import os
import random
from pathlib import Path

from absl import app, flags
from ml_collections import config_flags
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model

from flow_grpo.prompt_data import TextPromptDataset, GenevalPromptDataset
from flow_grpo.train_utils import (
    setup_distributed, cleanup_distributed, is_main_process, set_seed,
    synchronize, capture_rng, restore_rng,
)
from flow_grpo.diffusers_patch.sd3_pipeline_dmd import generate_latents
from flow_grpo.diffusers_patch.solver import cps_step
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt


FLAGS = flags.FLAGS
if "config" not in FLAGS:
    config_flags.DEFINE_config_file("config", "config/dmd.py:sd35m", "DMD configuration.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def dmd_loss(latents, pred_fake_x0, pred_real_x0, normalize=True):
    with torch.no_grad():
        p_real = latents - pred_real_x0
        grad = pred_fake_x0 - pred_real_x0
        if normalize:
            grad = grad / p_real.abs().mean(dim=tuple(range(1, latents.ndim)), keepdim=True)
        grad = torch.nan_to_num(grad)
    return 0.5 * F.mse_loss(latents.double(), (latents - grad).detach().double())


def fake_loss(velocity, target, scale=None, clamp_min=0.0, clamp_max=10.0):
    if scale is None:
        return F.mse_loss(velocity.float(), target.float())
    raw = F.mse_loss(velocity.float() * scale.float(), target.float(), reduction="none")
    weight = (1 / scale.float().square().clamp(min=1e-12)).clamp(clamp_min, clamp_max)
    weight = torch.nan_to_num(weight, nan=0.0, posinf=clamp_max, neginf=clamp_min)
    return (weight * raw).mean()


def ac_loss(anchor, first, second=None):
    if second is None:
        loss = F.mse_loss(first, anchor)
    else:
        loss = ((first - anchor) * (second - anchor)).flatten(1).mean(1).mean()
    return torch.nan_to_num(loss.float(), nan=0.0, posinf=0.0, neginf=0.0)


def save_ckpt(
    save_dir, transformer, global_step, rank, ema, parameters, config, optimizers, scalers,
):
    if is_main_process(rank):
        root = Path(save_dir) / "checkpoints" / f"checkpoint-{global_step}"
        root.mkdir(parents=True, exist_ok=True)
        stored = {
            name: parameter.detach().clone()
            for name, parameter in parameters["generator"].items()
        }
        if ema is not None:
            for name, parameter in parameters["generator"].items():
                parameter.data.copy_(ema[name])
        transformer.save_pretrained(
            root / "lora",
            selected_adapters=["generator", "fake_score"],
            save_embedding_layers=False,
        )
        for name, parameter in parameters["generator"].items():
            parameter.data.copy_(stored[name])
        torch.save(
            {role: optimizer.state_dict() for role, optimizer in optimizers.items()},
            root / "optimizer.pt",
        )
        torch.save(
            {role: scaler.state_dict() for role, scaler in scalers.items()},
            root / "scaler.pt",
        )
        (root / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
        logger.info(f"Saved checkpoint to {root}")


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    setup_distributed(local_rank)
    device = torch.device("cuda", local_rank)

    # --- Random Seed and TF32 ---
    set_seed(config.seed, rank)
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    weight_dtype = mixed_precision_dtype if enable_amp else torch.float32
    torch_autocast = partial(
        torch.autocast,
        device.type,
        dtype=mixed_precision_dtype,
        enabled=enable_amp,
    )

    # --- Load Pipeline and Models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model,
        torch_dtype=weight_dtype,
        variant=config.pretrained.variant or None,
        revision=config.pretrained.revision or None,
        local_files_only=config.pretrained.local_files_only,
    )
    for component in (pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3):
        component.requires_grad_(False).eval()
    pipeline.vae.to(device=device, dtype=torch.float32)
    for component in (pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3):
        component.to(device=device, dtype=weight_dtype)

    # --- LoRA Setup ---
    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    transformer_lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipeline.transformer.requires_grad_(False)
    if config.train.gradient_checkpointing:
        pipeline.transformer.enable_gradient_checkpointing()
    transformer = get_peft_model(
        pipeline.transformer,
        transformer_lora_config,
        adapter_name="generator",
    )
    transformer.add_adapter("fake_score", transformer_lora_config)
    transformer.set_adapter("generator")
    generator_parameters = {
        name: parameter for name, parameter in transformer.named_parameters() if parameter.requires_grad
    }
    transformer.set_adapter("fake_score")
    fake_score_parameters = {
        name: parameter for name, parameter in transformer.named_parameters() if parameter.requires_grad
    }
    parameters = {"generator": generator_parameters, "fake_score": fake_score_parameters}

    # --- Load LoRA Weights ---
    # PEFT 0.10 casts existing LoRA layers when adding an adapter: register both before loading FP32 weights.
    for parameter in [*generator_parameters.values(), *fake_score_parameters.values()]:
        parameter.data = parameter.data.float()
    paths = {"generator": config.lora.generator_path, "fake_score": config.lora.fake_score_path}
    for role, path in paths.items():
        if path:
            transformer.load_adapter(
                path,
                adapter_name=role,
                is_trainable=True,
                local_files_only=config.pretrained.local_files_only,
            )
    transformer.set_adapter("generator")

    # --- DDP Setup ---
    # DDP must register both optimizable adapters, not only the currently active one.
    for parameter in [*generator_parameters.values(), *fake_score_parameters.values()]:
        parameter.requires_grad_(True)
    transformer.to(device)
    pipeline.transformer = transformer
    transformer_ddp = transformer
    if dist.is_initialized() and dist.get_world_size() > 1:
        transformer_ddp = DDP(
            transformer,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # --- Noise Schedule and Latent Shape ---
    total_steps = pipeline.scheduler.config.num_train_timesteps
    shift = pipeline.scheduler.config.shift
    sigmas = [
        round(shift * (step / total_steps) / (1 + (shift - 1) * (step / total_steps)), 6)
        for step in config.sample.denoising_step_list[:config.sample.num_steps]
    ]
    sigmas = torch.tensor(sigmas + [0.0], device=device, dtype=torch.float32)
    training_sigmas = pipeline.scheduler.sigmas.to(device).clone()
    size = config.resolution // pipeline.vae_scale_factor
    shape = (config.train.batch_size, transformer.config.in_channels, size, size)

    # --- Datasets and Dataloaders ---
    dataset_type = (
        GenevalPromptDataset
        if (Path(config.dataset) / "train_metadata.jsonl").is_file()
        else TextPromptDataset
    )
    train_dataset = dataset_type(config.dataset, "train")
    if len(train_dataset) < config.train.batch_size * world_size:
        raise ValueError("Prompt dataset must contain at least one full global batch")
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )
    batches_per_epoch = len(train_dataloader)
    train_iter = None
    logger.info(
        "DMD prompt sampling: shuffled epochs, %s batches per epoch, format=%s",
        batches_per_epoch, dataset_type.__name__,
    )

    # --- Optimizers and LR Schedulers ---
    optimizers, schedulers, scalers = {}, {}, {}
    for role in ("generator", "fake_score"):
        learning_rate = (
            config.train.learning_rate if role == "generator" else config.train.fake_learning_rate
        )
        optimizers[role] = torch.optim.AdamW(
            list(parameters[role].values()),
            lr=learning_rate,
            betas=(config.train.adam_beta1, config.train.adam_beta2),
            eps=config.train.adam_epsilon,
            weight_decay=config.train.adam_weight_decay,
        )
        warmup = config.train.warmup_steps
        schedulers[role] = torch.optim.lr_scheduler.LambdaLR(
            optimizers[role],
            lambda step, warmup=warmup: min(1.0, (step + 1) / max(1, warmup)),
        )
        scalers[role] = torch.amp.GradScaler(
            device.type,
            enabled=mixed_precision_dtype == torch.float16,
        )

    # --- Generator EMA ---
    ema = (
        {name: parameter.detach().clone() for name, parameter in parameters["generator"].items()}
        if config.train.ema else None
    )

    # --- Resume from Checkpoint ---
    outer_step, generator_step = 0, 0
    if config.resume_from:
        root = Path(config.resume_from)
        logger.info(f"Resuming from {root}")
        for role in ("generator", "fake_score"):
            transformer.load_adapter(root / "lora" / role, adapter_name=role, is_trainable=True)
        transformer.set_adapter("generator")
        optimizer_state = torch.load(root / "optimizer.pt", map_location=device, weights_only=False)
        scaler_state = torch.load(root / "scaler.pt", map_location=device, weights_only=False)
        for role in ("generator", "fake_score"):
            optimizers[role].load_state_dict(optimizer_state[role])
            scalers[role].load_state_dict(scaler_state[role])
        if ema is not None:
            for name, parameter in parameters["generator"].items():
                ema[name].copy_(parameter.detach())
        try:
            outer_step = int(root.name.split("-")[-1])
        except ValueError:
            logger.warning(f"Could not parse global_step from checkpoint name: {root}")

    # --- Output Directory and Logging ---
    output_dir = Path(config.save_dir) / (config.run_name or ("ac-dmd" if config.ac_dmd.enabled else "dmd"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process(rank):
        (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    tracker = None
    if config.logging.enabled and is_main_process(rank):
        import swanlab
        tracker = swanlab.init(
            project=config.logging.project,
            experiment_name=output_dir.name,
            config=config.to_dict(),
            logdir=str(output_dir),
        )

    # --- Prompt Embeddings ---
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    with torch.no_grad():
        negative, negative_pooled = encode_prompt(
            text_encoders,
            tokenizers,
            [""] * config.train.batch_size,
            config.train.max_sequence_length,
            device=device,
        )

    # --- Training ---
    while outer_step < config.num_epochs:
        # --- Data Epoch and Resume Position ---
        data_epoch, batch_offset = divmod(outer_step, batches_per_epoch)
        if train_iter is None or batch_offset == 0:
            train_sampler.set_epoch(data_epoch)
            train_iter = iter(train_dataloader)
            for _ in range(batch_offset):
                next(train_iter)

        # Iterator creation consumes RNG; reset the training seed after positioning it.
        set_seed(config.seed + outer_step * world_size, rank)
        transformer_ddp.train()

        # --- Prompt Batch ---
        prompts, _ = next(train_iter)
        with torch.no_grad():
            embeds, pooled = encode_prompt(
                text_encoders, tokenizers, prompts, config.train.max_sequence_length, device=device,
            )

        # --- Fake / Generator Updates ---
        metrics = defaultdict(list)
        update_generator = (outer_step + 1) % config.dmd.fake_update_ratio == 0
        for role in ("fake_score", "generator") if update_generator else ("fake_score",):
            optimizer, scaler = optimizers[role], scalers[role]
            optimizer.zero_grad(set_to_none=True)

            # --- Generator Sampling ---
            if role == "generator":
                cfg = float(
                    synchronize(
                        torch.empty(1, device=device).uniform_(
                            config.dmd.real_guidance_scale_min,
                            config.dmd.real_guidance_scale_max,
                        )
                    ).item()
                )
                metrics["generator/teacher_cfg"].append(cfg)
            count = len(sigmas) - 1
            stop = (
                int(synchronize(torch.randint(1, count + 1, (1,), device=device)).item())
                if config.sample.random_stop_idx else count
            )
            initial = torch.randn(shape, device=device, dtype=embeds.dtype)
            transformer.set_adapter("generator")
            x0, x_s = generate_latents(
                transformer_ddp if role == "generator" else transformer,
                initial,
                embeds,
                pooled,
                sigmas,
                total_steps,
                stop,
                config.sample.noise_level,
                mixed_precision_dtype=mixed_precision_dtype,
                retain_grad=role == "generator",
            )
            boundary = x_s if config.ac_dmd.enabled else x0

            # --- Renoising and Targets ---
            scale = None
            with torch.no_grad():
                if config.ac_dmd.enabled:
                    s_value = float(sigmas[stop])
                    s = torch.full((shape[0],), s_value, device=device, dtype=boundary.dtype)
                    start = s_value * total_steps + config.ac_dmd.timestep_offset
                    end = (
                        total_steps
                        if not config.sample.random_stop_idx and config.ac_dmd.all_include_terminal_when_no_random_stop
                        else config.ac_dmd.end_timestep
                    )
                    if end <= start:
                        end = start + 1.0
                    times = torch.rand(shape[0], device=device) * (end - start) + start
                    times = synchronize(
                        times.clamp(
                            max(config.ac_dmd.min_renoise_step, config.dmd.min_step),
                            min(config.ac_dmd.max_renoise_step, config.dmd.max_step, total_steps - 1),
                        )
                    )
                    sigma = (times / total_steps).to(boundary.dtype)
                    if torch.any(sigma <= s) or torch.any(s >= 1):
                        raise ValueError("AC configuration produced a non-forward or singular interval")
                    noise = torch.randn_like(boundary)
                    a = (1 - sigma) / (1 - s)
                    q = (sigma**2 - a**2 * s**2).clamp(min=1e-12).sqrt()
                    noisy = a[:, None, None, None] * boundary + q[:, None, None, None] * noise
                    aa = -1 / (1 - s)
                    bb = sigma + (1 - sigma) / (1 - s)**2 * s**2
                    if config.ac_dmd.target_formula_move_scale_out:
                        scale = q
                        aa = aa * q
                    else:
                        bb = bb / q
                        scale = torch.ones_like(aa)
                    target = aa[:, None, None, None] * boundary + bb[:, None, None, None] * noise
                    scale = scale[:, None, None, None]
                else:
                    lo, hi = (
                        (config.dmd.fake_min_step, config.dmd.fake_max_step)
                        if role == "fake_score" else (config.dmd.min_step, config.dmd.max_step)
                    )
                    sampling = (
                        config.dmd.fake_timestep_sampling
                        if role == "fake_score" else config.dmd.dmd_timestep_sampling
                    )
                    mean = config.dmd.logit_mean if role == "fake_score" else config.dmd.dmd_logit_mean
                    std = config.dmd.logit_std if role == "fake_score" else config.dmd.dmd_logit_std
                    if sampling == "logit_normal":
                        u = torch.normal(mean=mean, std=std, size=(shape[0],)).sigmoid()
                        indices = ((u * (hi - lo + 1)).long().clamp(0, hi - lo) + lo).to(device)
                    else:
                        indices = torch.randint(lo, hi + 1, (shape[0],), device=device)
                    if role == "fake_score":
                        sigma = training_sigmas[indices].to(embeds.dtype)
                        noise = torch.randn_like(boundary)
                        noisy = (
                            (1 - sigma[:, None, None, None]) * boundary
                            + sigma[:, None, None, None] * noise
                        )
                        target = noise - boundary
                        times = training_sigmas[indices] * total_steps
                    else:
                        sigma = training_sigmas.flip(0)[indices].reshape(-1, 1, 1, 1)
                        noisy = sigma * torch.randn_like(boundary) + (1 - sigma) * boundary.detach()
                        times = (sigma * total_steps).flatten()

            if role == "fake_score":
                # --- Fake Score Loss ---
                transformer.set_adapter("fake_score")
                fake_guidance_scale = config.dmd.fake_train_guidance_scale
                fake_cfg = fake_guidance_scale > 1.0
                fake_embeds = torch.cat([negative, embeds]) if fake_cfg else embeds
                fake_pooled = torch.cat([negative_pooled, pooled]) if fake_cfg else pooled
                with torch_autocast():
                    velocity = transformer_ddp(
                        hidden_states=torch.cat([noisy, noisy]) if fake_cfg else noisy,
                        timestep=torch.cat([times, times]) if fake_cfg else times,
                        encoder_hidden_states=fake_embeds,
                        pooled_projections=fake_pooled,
                        return_dict=False,
                    )[0]
                if fake_cfg:
                    uncond, cond = velocity.chunk(2)
                    velocity = uncond + fake_guidance_scale * (cond - uncond)
                base_loss = fake_loss(
                    velocity,
                    target,
                    scale,
                    config.ac_dmd.fake_loss_weight_clamp_min,
                    config.ac_dmd.fake_loss_weight_clamp_max,
                )

                # --- AC Consistency Loss ---
                consistency = base_loss.new_zeros(())
                if (
                    config.ac_dmd.enabled
                    and config.ac_dmd.consistency_enabled
                    and config.ac_dmd.consistency_weight > 0
                ):
                    anchor = noisy.float() - sigma[:, None, None, None].float() * velocity.float()
                    sequence = []
                    current = times.float()
                    for _ in range(config.ac_dmd.consistency_num_steps):
                        delta = synchronize(
                            torch.randint(
                                1,
                                max(1, round(config.ac_dmd.consistency_epsilon_timestep)) + 1,
                                current.shape,
                                device=device,
                            )
                        )
                        current = (current - delta).clamp(min=0)
                        sequence.append(current)
                    paths = []
                    for _ in range(2 if config.ac_dmd.consistency_use_two_sample_unbiased else 1):
                        sample, time, sigma_from = noisy, times, sigma
                        with torch.set_grad_enabled(not config.ac_dmd.consistency_stopgrad_transitions):
                            for index, time_to in enumerate(sequence):
                                if index == 0:
                                    pred = (
                                        velocity.detach()
                                        if config.ac_dmd.consistency_stopgrad_transitions else velocity
                                    )
                                else:
                                    transition_model = (
                                        transformer if config.ac_dmd.consistency_stopgrad_transitions else transformer_ddp
                                    )
                                    with torch_autocast():
                                        pred = transition_model(
                                            hidden_states=torch.cat([sample, sample]) if fake_cfg else sample,
                                            timestep=torch.cat([time, time]) if fake_cfg else time,
                                            encoder_hidden_states=fake_embeds,
                                            pooled_projections=fake_pooled,
                                            return_dict=False,
                                        )[0]
                                    if fake_cfg:
                                        uncond, cond = pred.chunk(2)
                                        pred = uncond + fake_guidance_scale * (cond - uncond)
                                sigma_to = (time_to / total_steps).to(noisy.dtype)
                                sample = cps_step(
                                    sample,
                                    pred,
                                    sigma_from,
                                    sigma_to,
                                    config.sample.noise_level,
                                    torch.randn_like(sample) if config.sample.noise_level > 0 else None,
                                )
                                time, sigma_from = time_to, sigma_to
                        paths.append((sample, time, sigma_from))
                    predictions = []
                    for sample, time, sigma_prime in paths:
                        with torch_autocast():
                            prediction = transformer_ddp(
                                hidden_states=torch.cat([sample, sample]) if fake_cfg else sample,
                                timestep=torch.cat([time, time]) if fake_cfg else time,
                                encoder_hidden_states=fake_embeds,
                                pooled_projections=fake_pooled,
                                return_dict=False,
                            )[0]
                        if fake_cfg:
                            uncond, cond = prediction.chunk(2)
                            prediction = uncond + fake_guidance_scale * (cond - uncond)
                        predictions.append(
                            sample.float() - sigma_prime[:, None, None, None].float() * prediction.float()
                        )
                    consistency = ac_loss(anchor, *predictions)

                loss = (
                    base_loss + config.ac_dmd.consistency_weight * consistency
                    if config.ac_dmd.enabled else base_loss
                )
                metrics["fake_score/loss_fake"].append(float(loss.detach()))
                metrics["fake_score/loss_fake_base"].append(float(base_loss.detach()))
                metrics["fake_score/loss_consistency"].append(float(consistency.detach()))
            else:
                # --- Generator DMD Loss ---
                with torch.no_grad():
                    transformer.eval()
                    teacher_cfg = cfg > 1.0
                    with transformer.disable_adapter(), torch_autocast():
                        real_velocity = transformer(
                            hidden_states=torch.cat([noisy, noisy]) if teacher_cfg else noisy,
                            timestep=torch.cat([times, times]) if teacher_cfg else times,
                            encoder_hidden_states=torch.cat([negative, embeds]) if teacher_cfg else embeds,
                            pooled_projections=torch.cat([negative_pooled, pooled]) if teacher_cfg else pooled,
                            return_dict=False,
                        )[0]
                    if teacher_cfg:
                        uncond, cond = real_velocity.chunk(2)
                        real_velocity = uncond + cfg * (cond - uncond)
                    transformer.train()
                    transformer.set_adapter("fake_score")
                    fake_guidance_scale = config.dmd.fake_guidance_scale
                    fake_cfg = fake_guidance_scale > 1.0
                    with torch_autocast():
                        fake_velocity = transformer(
                            hidden_states=torch.cat([noisy, noisy]) if fake_cfg else noisy,
                            timestep=torch.cat([times, times]) if fake_cfg else times,
                            encoder_hidden_states=torch.cat([negative, embeds]) if fake_cfg else embeds,
                            pooled_projections=torch.cat([negative_pooled, pooled]) if fake_cfg else pooled,
                            return_dict=False,
                        )[0]
                    if fake_cfg:
                        uncond, cond = fake_velocity.chunk(2)
                        fake_velocity = uncond + fake_guidance_scale * (cond - uncond)
                    if config.ac_dmd.enabled:
                        noise = torch.randn_like(noisy) if config.sample.noise_level > 0 else None
                        real = cps_step(noisy, real_velocity, sigma, s, config.sample.noise_level, noise)
                        fake = cps_step(noisy, fake_velocity, sigma, s, config.sample.noise_level, noise)
                    else:
                        real = (noisy - sigma * real_velocity).to(noisy.dtype)
                        fake = (noisy - sigma * fake_velocity).to(noisy.dtype)
                # Checkpoint recomputation must see the adapter used for the differentiable generation.
                transformer.set_adapter("generator")
                loss = dmd_loss(boundary, fake, real, config.dmd.gradient_normalization)
                metrics["generator/loss_dm"].append(float(loss.detach()))
            metrics[f"{role}/gen_steps"].append(stop)

            # --- Backward ---
            scaler.scale(loss).backward()

            # --- Optimizer Step ---
            scaler.unscale_(optimizer)
            max_norm = config.train.max_grad_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(parameters[role].values()),
                max_norm if max_norm > 0 else math.inf,
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                schedulers[role].step()
                if role == "generator":
                    generator_step += 1
            optimizer.zero_grad(set_to_none=True)
            metrics[f"{role}/grad_norm"].append(float(grad_norm))
            metrics[f"{role}/lr"].append(optimizer.param_groups[0]["lr"])
        outer_step += 1

        # --- EMA Update ---
        if ema is not None and outer_step % config.train.ema_every == 0:
            with torch.no_grad():
                for name, parameter in parameters["generator"].items():
                    ema[name].mul_(config.train.ema_decay).add_(
                        parameter.detach(), alpha=1 - config.train.ema_decay,
                    )

        # --- Training Metrics ---
        metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        if outer_step % config.train.log_interval == 0:
            if world_size > 1:
                values = torch.tensor(list(metrics.values()), device=device, dtype=torch.float64)
                dist.all_reduce(values)
                metrics = dict(zip(metrics, (values / world_size).tolist()))
            if is_main_process(rank):
                logger.info(f"step={outer_step} generator_step={generator_step} {metrics}")
                with (output_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(step=outer_step, generator_step=generator_step, **metrics)) + "\n")
                if tracker is not None:
                    tracker.log(metrics, step=outer_step)

        # --- Evaluation ---
        if config.eval.enabled and (outer_step % config.eval_freq == 0 or outer_step == config.num_epochs):
            rng = capture_rng()
            previous_adapter = transformer.active_adapter
            modes = [(module, module.training) for module in transformer.modules()]
            gradients = [(parameter, parameter.requires_grad) for parameter in transformer.parameters()]
            stored = {name: p.detach().clone() for name, p in parameters["generator"].items()}
            scores = defaultdict(list)
            transformer.set_adapter("generator")
            transformer.eval()
            if ema is not None and config.eval.use_ema:
                for name, parameter in parameters["generator"].items():
                    parameter.data.copy_(ema[name])
            torch.manual_seed(config.eval.seed + rank)
            random.seed(config.eval.seed + rank)
            np.random.seed(config.eval.seed + rank)
            path = Path(config.eval.dataset)
            if (path / "test_metadata.jsonl").is_file():
                records = [
                    json.loads(line)
                    for line in (path / "test_metadata.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                records = [dict(prompt=item["prompt"], metadata=item) for item in records]
            else:
                records = [
                    dict(prompt=line.strip(), metadata={})
                    for line in (path / "test.txt").read_text().splitlines()
                    if line.strip()
                ]
            if not records:
                raise ValueError(f"No prompts in {path} (test)")
            num_prompts = config.eval.num_prompts
            if outer_step == config.num_epochs:
                num_prompts = config.eval.get("final_num_prompts", num_prompts)
            records = list(enumerate(records[:num_prompts]))[rank::world_size]
            folder = output_dir / "eval" / f"step-{outer_step}" / f"rank-{rank}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "prompts.json").write_text(json.dumps([
                dict(index=index, **record) for index, record in records
            ], indent=2, ensure_ascii=False))
            reward = None
            if config.eval.reward_fn:
                from flow_grpo.rewards import multi_score
                reward = multi_score(device, config.eval.reward_fn)
            with torch.no_grad():
                for start in range(0, len(records), config.sample.test_batch_size):
                    batch = records[start:start + config.sample.test_batch_size]
                    prompts = [record["prompt"] for _, record in batch]
                    embeds, pooled = encode_prompt(
                        text_encoders, tokenizers, prompts, config.train.max_sequence_length, device=device,
                    )
                    initial = torch.randn(
                        len(batch), transformer.config.in_channels, size, size,
                        device=device, dtype=embeds.dtype,
                    )
                    x0, _ = generate_latents(
                        transformer,
                        initial,
                        embeds,
                        pooled,
                        sigmas,
                        total_steps,
                        len(sigmas) - 1,
                        config.sample.noise_level,
                        mixed_precision_dtype=mixed_precision_dtype,
                    )
                    vae_latents = (
                        x0.float() / pipeline.vae.config.scaling_factor
                        + pipeline.vae.config.shift_factor
                    )
                    images = (pipeline.vae.decode(vae_latents, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)
                    folder = output_dir / "eval" / f"step-{outer_step}" / f"rank-{rank}"
                    folder.mkdir(parents=True, exist_ok=True)
                    from PIL import Image
                    for (index, _), image in zip(batch, images):
                        array = image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                        Image.fromarray(array).save(folder / f"{index:06d}.png")
                    if reward is not None:
                        details, _ = reward(images, prompts, [record["metadata"] for _, record in batch])
                        for key, values in details.items():
                            scores[key].extend(np.asarray(values).reshape(-1).tolist())

            # --- Restore Training State ---
            for name, parameter in parameters["generator"].items():
                parameter.data.copy_(stored[name])
            transformer.set_adapter(previous_adapter)
            for module, mode in modes:
                module.training = mode
            for parameter, requires_grad in gradients:
                parameter.requires_grad_(requires_grad)
            restore_rng(rng)

            # --- Evaluation Metrics ---
            results = [None] * world_size
            local = {key: (sum(values), len(values)) for key, values in scores.items()}
            if dist.is_initialized():
                dist.all_gather_object(results, local)
            else:
                results[0] = local
            sums, counts = defaultdict(float), defaultdict(int)
            for result in results:
                for key, (value, count) in result.items():
                    sums[key] += value
                    counts[key] += count
            values = {f"eval/{key}": sums[key] / counts[key] for key in sums if counts[key]}
            if is_main_process(rank):
                logger.info(f"step={outer_step} {values}")
                evaluation = dict(step=outer_step, counts=dict(counts), **values)
                if "eval/pickscore" in values:
                    evaluation["eval/pickscore_raw"] = 26 * values["eval/pickscore"]
                (output_dir / "eval" / f"step-{outer_step}" / "metrics.json").write_text(
                    json.dumps(evaluation, indent=2)
                )
                if tracker is not None:
                    tracker.log(values, step=outer_step)

        # --- Save Checkpoint ---
        if outer_step % config.save_freq == 0 or outer_step == config.num_epochs:
            save_ckpt(
                output_dir, transformer, outer_step, rank, ema, parameters, config, optimizers, scalers,
            )
            if dist.is_initialized():
                dist.barrier()

    # --- Cleanup ---
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
