"""Path-ratio detail ablations: deterministic noise-conditioned consistency."""

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

from flow_grpo.posterior_consistency import consistency_surrogate
from flow_grpo.consistency_schedule import consistency_weight
from flow_grpo.query_time_weight import query_time_weight
from flow_grpo.diffusers_patch.sd3_last_innovation_path_is import make_fields, sample_prefix, conditional_path_mean
from flow_grpo.diffusers_patch.sd3_source_trace import replay_source
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
    config_flags.DEFINE_config_file("config", "config/rtdmd_path_is_last.py:last", "DMD configuration.")

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
            selected_adapters=["generator"],
            save_embedding_layers=False,
        )
        for name, parameter in parameters["generator"].items():
            parameter.data.copy_(stored[name])
        # Inference-only checkpoint: exact resume is explicitly disabled. Avoid
        # saving unused optimizer states on this disk-limited comparison host.
        (root / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
        logger.info(f"Saved checkpoint to {root}")


def main(_):
    config = FLAGS.config
    if not config.posterior_consistency.enabled:
        raise ValueError("Enable the documented consistency configuration, including weight=0")
    if config.ac_dmd.enabled:
        raise ValueError("Detail ablations require AC-DMD disabled")
    if not config.path_is.enabled and config.method_name != "rtdmd_paired_source_ablation":
        raise ValueError("Only the explicit paired-source control can disable particles")
    if config.resume_from:
        raise ValueError("Resume requires complete RNG/EMA state; start a fresh comparison")

    # --- Distributed Setup ---
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size != 1 or config.train.batch_size != 1:
        raise ValueError("Posterior trace replay currently supports one GPU and batch_size=1")
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
    if any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in transformer.modules()):
        raise ValueError("The observation Jacobian requires dropout-free Student predictions")
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
        update_generator = (outer_step + 1) % config.dmd.fake_update_ratio == 0
        # No Fake optimizer exists in this method. Preserve data position, seed
        # reset, EMA and update cadence, but encode text only when it is used.
        # Frozen eval-mode text encoders have no state updates on skipped rows.
        if update_generator:
            with torch.no_grad():
                embeds, pooled = encode_prompt(
                    text_encoders, tokenizers, prompts, config.train.max_sequence_length, device=device,
                )

        # --- Fake / Generator Updates ---
        metrics = defaultdict(list)
        for role in ("generator",) if update_generator else ():
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
            transformer.set_adapter("generator")
            transformer.eval()
            student_field, teacher_field = make_fields(transformer, embeds, pooled,
                negative, negative_pooled, total_steps, config.path_is.proposal_cfg,
                mixed_precision_dtype, config.path_is.field_microbatch)
            prefix, source = sample_prefix(shape, student_field, sigmas, stop,
                config.sample.noise_level, device, embeds.dtype)
            with torch.no_grad():
                lo, hi = config.dmd.min_step, config.dmd.max_step
                if config.dmd.dmd_timestep_sampling == "logit_normal":
                    u = torch.normal(mean=config.dmd.dmd_logit_mean,
                        std=config.dmd.dmd_logit_std, size=(shape[0],)).sigmoid()
                    indices = ((u * (hi - lo + 1)).long().clamp(0, hi - lo) + lo).to(device)
                else:
                    indices = torch.randint(lo, hi + 1, (shape[0],), device=device)
                sigma = training_sigmas.flip(0)[indices].reshape(-1, 1, 1, 1)
                noisy = sigma * torch.randn_like(source, dtype=torch.float32) + (1 - sigma) * source.float()
                times = (sigma * total_steps).flatten()
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
                real = noisy.float() - sigma * real_velocity.float()
                transformer.set_adapter("generator")
                if config.path_is.enabled:
                    fake, posterior_metrics = conditional_path_mean(noisy, sigma, prefix,
                        source, student_field, teacher_field, sigmas[:-1], config.sample.noise_level,
                        num_particles=config.path_is.num_particles, num_sweeps=config.path_is.num_sweeps,
                        base_scale=config.path_is.base_scale, block_size=config.path_is.block_size,
                        high_noise_block_size=config.path_is.high_noise_block_size,
                        high_noise_threshold=config.path_is.high_noise_threshold,
                        max_innovation_radius=config.path_is.max_innovation_radius)
                    metrics.update({key: [value] for key, value in posterior_metrics.items()})
                else:
                    # Control only: X*|y is a single posterior draw under paired
                    # generation. No Teacher paths, path weights or particles.
                    fake = source.float()
                    metrics["paired_source/no_particles_control"].append(1.)
                metrics["generator/query_sigma"].append(float(sigma.item()))
            # Eq.(2): J_theta D_theta(y,t,c), with y detached. This Jacobian
            # and DMD's normalization are deterministic conditional on (y,t,c),
            # avoiding the correlated-source-Jacobian bias of a paired chain.
            transformer.train()
            with torch_autocast():
                student_velocity = transformer_ddp(hidden_states=noisy.detach(),
                    timestep=times, encoder_hidden_states=embeds,
                    pooled_projections=pooled, return_dict=False)[0]
            boundary = noisy.detach().float() - sigma * student_velocity.float()
            effective_weight = consistency_weight(float(sigma.item()),
                config.posterior_consistency.weight,
                config.posterior_consistency.get("schedule", "constant"))
            loss = consistency_surrogate(boundary, fake, real,
                effective_weight, config.dmd.gradient_normalization)
            metrics["generator/loss_dm_unweighted"].append(float(loss.detach()))
            time_factor = query_time_weight(float(sigma.item()),
                config.posterior_time_weight.mode, config.posterior_time_weight.scale)
            loss = time_factor * loss
            metrics["generator/query_time_weight"].append(time_factor)
            metrics["generator/consistency_mse"].append(float((boundary.detach()-fake).square().mean()))
            metrics["generator/teacher_clean_mse"].append(float((boundary.detach()-real).square().mean()))
            metrics["generator/consistency_weight"].append(effective_weight)
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
