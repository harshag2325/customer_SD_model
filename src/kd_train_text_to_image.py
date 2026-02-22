#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import csv
import time
import copy

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False

check_min_version("0.15.0")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output
    return get_output_hook


def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))


def copy_weight_from_teacher(unet_stu, unet_tea, student_type):
    connect_info = {}
    if student_type in ["bk_base", "bk_small"]:
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.0.resnets.2.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.3.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.3.attentions.1.'] = 'up_blocks.3.attentions.2.'
    elif student_type in ["bk_tiny"]:
        connect_info['up_blocks.0.resnets.0.'] = 'up_blocks.1.resnets.0.'
        connect_info['up_blocks.0.attentions.0.'] = 'up_blocks.1.attentions.0.'
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.0.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.0.upsamplers.'] = 'up_blocks.1.upsamplers.'
        connect_info['up_blocks.1.resnets.0.'] = 'up_blocks.2.resnets.0.'
        connect_info['up_blocks.1.attentions.0.'] = 'up_blocks.2.attentions.0.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.1.upsamplers.'] = 'up_blocks.2.upsamplers.'
        connect_info['up_blocks.2.resnets.0.'] = 'up_blocks.3.resnets.0.'
        connect_info['up_blocks.2.attentions.0.'] = 'up_blocks.3.attentions.0.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.3.attentions.2.'
    else:
        raise NotImplementedError

    for k in unet_stu.state_dict().keys():
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])
                break
        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
        else:
            print(f"normal COPY {k_orig} -> {k}")
        unet_stu.state_dict()[k].copy_(unet_tea.state_dict()[k_orig])

    return unet_stu


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", default=False, action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--non_ema_revision", type=str, default=None, required=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")  # no-op, kept for CLI compatibility
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--unet_config_path", type=str, default="./src/unet_config")
    parser.add_argument("--unet_config_name", type=str, default="bk_small",
                        choices=["bk_base", "bk_small", "bk_tiny"])
    parser.add_argument("--lambda_sd", type=float, default=1.0)
    parser.add_argument("--lambda_kd_output", type=float, default=1.0)
    parser.add_argument("--lambda_kd_feat", type=float, default=1.0)
    parser.add_argument("--valid_prompt", type=str, default="a golden vase with different flowers")
    parser.add_argument("--valid_steps", type=int, default=500)
    parser.add_argument("--num_valid_images", type=int, default=2)
    parser.add_argument("--use_copy_weight_from_teacher", action="store_true")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    # ------------------------------------------------------------------ device
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    logger.info(f"Using device: {device}")
    if not cuda_available:
        logger.warning("CUDA is not available. Training on CPU will be very slow. "
                       "Please enable a GPU accelerator in your environment.")

    # --------------------------------------------------------- mixed precision
    # Only enable fp16 if CUDA is available; fp16 is not supported on CPU
    use_fp16 = (args.mixed_precision == "fp16") and cuda_available
    use_bf16 = (args.mixed_precision == "bf16") and cuda_available

    if args.mixed_precision == "fp16" and not cuda_available:
        logger.warning("fp16 mixed precision requested but CUDA is not available. "
                       "Falling back to fp32.")
    if args.mixed_precision == "bf16" and not cuda_available:
        logger.warning("bf16 mixed precision requested but CUDA is not available. "
                       "Falling back to fp32.")

    # GradScaler only works with CUDA + fp16
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    weight_dtype = torch.float32
    if use_fp16:
        weight_dtype = torch.float16
    elif use_bf16:
        weight_dtype = torch.bfloat16

    # -------------------------------------------------------------------- seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ----------------------------------------------------------- output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    val_img_dir = os.path.join(args.output_dir, 'val_img')
    os.makedirs(val_img_dir, exist_ok=True)

    # ------------------------------------------------------------ CSV logger
    csv_log_path = os.path.join(args.output_dir, 'log_loss.csv')
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'step', 'global_step',
                                'loss_total', 'loss_sd', 'loss_kd_output', 'loss_kd_feat',
                                'lr', 'lamb_sd', 'lamb_kd_output', 'lamb_kd_feat'])

    if args.allow_tf32 and cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = True

    # --------------------------------------------------------------- models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    # teacher
    unet_teacher = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision)

    # student
    config_student = UNet2DConditionModel.load_config(
        args.unet_config_path, subfolder=args.unet_config_name)
    unet = UNet2DConditionModel.from_config(config_student)

    if args.use_copy_weight_from_teacher:
        copy_weight_from_teacher(unet, unet_teacher, args.unet_config_name)

    # freeze everything except student unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_teacher.requires_grad_(False)

    # ------------------------------------------------------------------- EMA
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_config(config_student)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel,
                            model_config=ema_unet.config)
        ema_unet.to(device)

    # --------------------------------------------------------------- xformers
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly.")

    # ------------------------------------------------ gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ----------------------------------------------------------- scale lr
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size
        )

    # --------------------------------------------------------------- optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam: pip install bitsandbytes")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ---------------------------------------------------------------- dataset
    print("*** load dataset: start")
    t0 = time.time()
    dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, split="train")
    print(f"*** load dataset: end --- {time.time() - t0:.2f} sec")

    column_names = dataset.column_names
    image_column = column_names[0]
    caption_column = column_names[1]

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings.")
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids

    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    if args.max_train_samples is not None:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
    train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # ------------------------------------------------------- lr scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # ------------------------------------------- move models to device
    unet.to(device)
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet_teacher.to(device, dtype=weight_dtype)

    # ----------------------------------------------- feature KD hooks
    acts_tea = {}
    acts_stu = {}

    if args.unet_config_name in ["bk_base", "bk_small"]:
        mapping_layers_tea = [
            'up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3',
            'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3'
        ]
        mapping_layers_stu = copy.deepcopy(mapping_layers_tea)

    elif args.unet_config_name in ["bk_tiny"]:
        mapping_layers_tea = [
            'down_blocks.0', 'down_blocks.1', 'down_blocks.2.attentions.1.proj_out',
            'up_blocks.1', 'up_blocks.2', 'up_blocks.3'
        ]
        mapping_layers_stu = [
            'down_blocks.0', 'down_blocks.1', 'down_blocks.2.attentions.0.proj_out',
            'up_blocks.0', 'up_blocks.1', 'up_blocks.2'
        ]

    add_hook(unet_teacher, acts_tea, mapping_layers_tea)
    add_hook(unet, acts_stu, mapping_layers_stu)

    # --------------------------------------------------------- logging info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples         = {len(train_dataset)}")
    logger.info(f"  Num Epochs           = {args.num_train_epochs}")
    logger.info(f"  Batch size           = {args.train_batch_size}")
    logger.info(f"  Gradient accum steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optim steps    = {args.max_train_steps}")
    logger.info(f"  Mixed precision      = {args.mixed_precision} (active fp16={use_fp16}, bf16={use_bf16})")

    global_step = 0
    first_epoch = 0

    # ------------------------------------------------- resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            ckpt_dir = args.resume_from_checkpoint
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            ckpt_dir = os.path.join(args.output_dir, dirs[-1]) if dirs else None

        if ckpt_dir is None or not os.path.exists(ckpt_dir):
            logger.info("No checkpoint found. Starting fresh training run.")
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint: {ckpt_dir}")
            ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"), map_location=device)
            unet.load_state_dict(ckpt["unet"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            global_step = ckpt["global_step"]
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resumed at global step {global_step}")

    # ------------------------------------------------------------- training
    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()

        train_loss = 0.0
        train_loss_sd = 0.0
        train_loss_kd_output = 0.0
        train_loss_kd_feat = 0.0

        for step, batch in enumerate(train_dataloader):

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)

            # ---- forward pass with autocast (safe on CPU too) ----
            with torch.amp.autocast('cuda', enabled=use_fp16 and cuda_available):

                # encode images to latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # sample noise & timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # text embeddings
                encoder_hidden_states = text_encoder(input_ids)[0]

                # target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # student forward
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss_sd = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # teacher forward (no grad)
                with torch.no_grad():
                    model_pred_teacher = unet_teacher(
                        noisy_latents, timesteps, encoder_hidden_states).sample
                loss_kd_output = F.mse_loss(
                    model_pred.float(), model_pred_teacher.float(), reduction="mean")

                # feature KD loss
                losses_kd_feat = []
                for m_tea, m_stu in zip(mapping_layers_tea, mapping_layers_stu):
                    a_tea = acts_tea[m_tea]
                    a_stu = acts_stu[m_stu]
                    if isinstance(a_tea, tuple): a_tea = a_tea[0]
                    if isinstance(a_stu, tuple): a_stu = a_stu[0]
                    losses_kd_feat.append(
                        F.mse_loss(a_stu.float(), a_tea.detach().float(), reduction="mean"))
                loss_kd_feat = sum(losses_kd_feat)

                # total loss (scaled for gradient accumulation)
                loss = (args.lambda_sd * loss_sd
                        + args.lambda_kd_output * loss_kd_output
                        + args.lambda_kd_feat * loss_kd_feat)
                loss = loss / args.gradient_accumulation_steps

            # ---- backward ----
            if use_fp16 and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # accumulate metrics (store unscaled values)
            train_loss += loss.item()
            train_loss_sd += loss_sd.item() / args.gradient_accumulation_steps
            train_loss_kd_output += loss_kd_output.item() / args.gradient_accumulation_steps
            train_loss_kd_feat += loss_kd_feat.item() / args.gradient_accumulation_steps

            # ---- optimizer step after accumulation ----
            if (step + 1) % args.gradient_accumulation_steps == 0:

                if use_fp16 and scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                if use_fp16 and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

                if args.use_ema:
                    ema_unet.step(unet.parameters())

                progress_bar.update(1)
                global_step += 1
                current_lr = lr_scheduler.get_last_lr()[0]

                # console log
                logger.info(
                    f"epoch={epoch} step={step} global_step={global_step} "
                    f"loss={train_loss:.4f} loss_sd={train_loss_sd:.4f} "
                    f"loss_kd_output={train_loss_kd_output:.4f} "
                    f"loss_kd_feat={train_loss_kd_feat:.4f} lr={current_lr:.6f}"
                )

                # CSV log
                with open(csv_log_path, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([
                        epoch, step, global_step,
                        train_loss, train_loss_sd, train_loss_kd_output, train_loss_kd_feat,
                        current_lr,
                        args.lambda_sd, args.lambda_kd_output, args.lambda_kd_feat
                    ])

                # wandb log
                # wandb log
                if has_wandb and wandb.run is not None:
                    wandb.log({
                        "train_loss": train_loss,
                        "train_loss_sd": train_loss_sd,
                        "train_loss_kd_output": train_loss_kd_output,
                        "train_loss_kd_feat": train_loss_kd_feat,
                        "lr": current_lr,
                    }, step=global_step)

                # reset accumulators
                train_loss = 0.0
                train_loss_sd = 0.0
                train_loss_kd_output = 0.0
                train_loss_kd_feat = 0.0

                # # save checkpoint
                # if global_step % args.checkpointing_steps == 0:
                #     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #     os.makedirs(save_path, exist_ok=True)
                #     torch.save({
                #         "unet": unet.state_dict(),
                #         "optimizer": optimizer.state_dict(),
                #         "lr_scheduler": lr_scheduler.state_dict(),
                #         "global_step": global_step,
                #     }, os.path.join(save_path, "checkpoint.pt"))
                #     logger.info(f"Saved checkpoint to {save_path}")

                #     # prune old checkpoints if limit set
                #     if args.checkpoints_total_limit is not None:
                #         all_ckpts = sorted(
                #             [d for d in os.listdir(args.output_dir)
                #              if d.startswith("checkpoint")],
                #             key=lambda x: int(x.split("-")[1])
                #         )
                #         while len(all_ckpts) > args.checkpoints_total_limit:
                #             oldest = os.path.join(args.output_dir, all_ckpts.pop(0))
                #             shutil.rmtree(oldest)
                #             logger.info(f"Deleted old checkpoint: {oldest}")

            progress_bar.set_postfix({
                "step_loss": loss.detach().item(),
                "sd_loss": loss_sd.detach().item(),
                "kd_out": loss_kd_output.detach().item(),
                "kd_feat": loss_kd_feat.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            })

            # ------------------------------------------------- validation
            if (args.valid_prompt is not None) and (step % args.valid_steps == 0):
                logger.info(
                    f"Running validation... Generating {args.num_valid_images} images "
                    f"with prompt: '{args.valid_prompt}'")
                unet.eval()

                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    safety_checker=None,
                    revision=args.revision,
                ).to(device)
                pipeline.set_progress_bar_config(disable=True)

                generator = torch.Generator(device=device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)

                # teacher validation images (saved once)
                if not os.path.exists(os.path.join(val_img_dir, "teacher_0.png")):
                    for kk in range(args.num_valid_images):
                        image = pipeline(
                            args.valid_prompt, num_inference_steps=25, generator=generator
                        ).images[0]
                        image.save(os.path.join(val_img_dir, f"teacher_{kk}.png"))

                # student validation images
                pipeline.unet = unet
                for kk in range(args.num_valid_images):
                    image = pipeline(
                        args.valid_prompt, num_inference_steps=25, generator=generator
                    ).images[0]
                    tmp_name = os.path.join(
                        val_img_dir,
                        f"gstep{global_step}_epoch{epoch}_step{step}_{kk}.png")
                    image.save(tmp_name)
                    logger.info(f"Saved validation image: {tmp_name}")

                del pipeline
                torch.cuda.empty_cache()
                unet.train()

            if global_step >= args.max_train_steps:
                break

    # ------------------------------------------------------- save final model
    if args.use_ema:
        ema_unet.copy_to(unet.parameters())

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        revision=args.revision,
    )
    pipeline.save_pretrained(args.output_dir)
    logger.info(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
