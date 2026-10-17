"""EruDiff training on FLUX.1-dev (Algorithm 1).

    while not converged:
        # Phase I -- DK-DM
        sample prompt category c in {impl, expl, found} with probability p_c
        sample prompt triplet (y_impl, y_expl, y_found)
        construct the corresponding prompt pair
        sample z ~ N(0, I) and step count n ~ p(n)            (Eq. 3)
        execute n denoising steps to obtain x_tau_i = g_theta(z, tau_i)
        sample t ~ U(T_min, T_max) relative to [tau_i, tau_{i+1}]
        compute L_DK-DM                                       (Eq. 2)
        L_DK-DM <- EMAN(L_DK-DM)
        theta <- theta - eta * grad L_DK-DM

        # Phase II -- NO-RL
        sample failure x0 ~ D_loss and t ~ U[0, T]
        compute L_NO-RL                                       (Eq. 4)
        L_NO-RL <- EMAN(L_NO-RL)
        theta <- theta - eta * grad L_NO-RL

Per Sec. 3.3 the two objectives do not share a forward pass, so the loop below
backpropagates and steps them independently instead of summing the losses.

NO-RL needs a reward model to build D_loss, so `--reward_model_path` is
optional: omitting it trains with DK-DM alone.

Example (8 GPUs, DK-DM only):
    accelerate launch --num_processes 8 --multi_gpu train.py \
        --pretrained_model_name_or_path $FLUX --train_data_path data/train.jsonl \
        --output_dir runs/erudiff --rank 128 --lora_alpha 128 \
        --num_inference_steps 28 --guidance_scale 3.5 --max_train_steps 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import FluxPipeline
from peft import LoraConfig
from torch.utils.data import DataLoader

from src.data.dataset import PromptTripletDataset
from src.models.dkdm import compute_dkdm_loss
from src.models.generator import (
    ADAPTER_NAME,
    run_full,
    run_generator,
    sample_packed_noise,
    transformer_config,
)
from src.models.norl import compute_norl_loss
from src.utils.training import (
    EMANormalizer,
    FailureFilter,
    PromptMixer,
    TimestepCurriculum,
    decode_latents,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EruDiff on FLUX.1-dev")

    # --- paths ---
    p.add_argument("--pretrained_model_name_or_path", required=True,
                   help="FLUX.1-dev repo id or local snapshot directory")
    p.add_argument("--train_data_path", required=True,
                   help="JSONL of (implicit, explicit, sub) prompt triplets")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--reward_model_path", default=None,
                   help="CLIP-style reward model used to build D_loss. "
                        "Omit to train with DK-DM only.")

    # --- optimisation ---
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--adam_weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--train_batch_size", type=int, default=2,
                   help="per-device batch size")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=4000)

    # --- LoRA ---
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)

    # --- generation ---
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--max_sequence_length", type=int, default=512)

    # --- DK-DM / AF-KC / TA-CL ---
    p.add_argument("--curriculum_decay", type=float, default=0.1,
                   help="lambda in Eq. 3")
    p.add_argument("--p_impl", type=float, default=0.8)
    p.add_argument("--p_expl", type=float, default=0.1)
    p.add_argument("--p_found", type=float, default=0.1)
    p.add_argument("--dkdm_weight", type=float, default=1.0)

    # --- NO-RL ---
    p.add_argument("--norl_weight", type=float, default=1.0)
    p.add_argument("--norl_every", type=int, default=10,
                   help="run Phase II every N steps; it needs a full reference "
                        "sampling pass, so it is amortised")
    p.add_argument("--norl_beta", type=float, default=10.0)
    p.add_argument("--reward_threshold", type=float, default=None,
                   help="fixed failure threshold. Omit to calibrate from the "
                        "running mean of observed reward scores.")
    p.add_argument("--reward_warmup", type=int, default=64)

    # --- EMAN ---
    p.add_argument("--eman_decay", type=float, default=0.99)

    # --- bookkeeping ---
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--checkpoints_total_limit", type=int, default=3)

    args = p.parse_args()

    total = args.p_impl + args.p_expl + args.p_found
    if abs(total - 1.0) > 1e-6:
        p.error(f"--p_impl/--p_expl/--p_found must sum to 1 (got {total})")
    if args.num_inference_steps < 1:
        p.error("--num_inference_steps must be >= 1")
    return args


def collate(batch: list[dict]) -> dict[str, list]:
    return {k: [b[k] for b in batch] for k in batch[0]}


def build_pipeline(args, device, dtype) -> FluxPipeline:
    """Load FLUX, freeze everything, attach a single LoRA adapter.

    The adapter does double duty (Sec. 3.1): at scale 1 it is pi_theta, which
    provides s_impl; at scale 0 it recovers the frozen pi_ref, which provides
    s_expl.
    """
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path, dtype=dtype
    )
    pipeline.to(device)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.text_encoder_2.eval()

    # Sec. 5.1: LoRA on the attention layers.
    pipeline.transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
        adapter_name=ADAPTER_NAME,
    )
    pipeline.set_adapters([ADAPTER_NAME], adapter_weights=[1.0])

    for name, param in pipeline.transformer.named_parameters():
        param.requires_grad = "lora" in name.lower()

    # LoRA math in bf16 loses too much precision for a stable optimizer step.
    for param in pipeline.transformer.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    return pipeline


@torch.no_grad()
def encode(pipeline, prompts: list[str], device, max_seq_len: int):
    """Encode prompts with the frozen dual text encoders.

    Returns (prompt_embeds, pooled_prompt_embeds, text_ids), with text_ids
    unbatched as (seq, 3).
    """
    return pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_seq_len,
    )


def save_lora(pipeline, accelerator, out_dir: Path) -> None:
    """Export the adapter in diffusers' LoRA format for inference.

    `save_lora_weights` adds the `transformer.` prefix itself and
    `get_peft_model_state_dict` already returns keys relative to the
    transformer, so the keys are passed through unchanged.
    """
    from peft.utils import get_peft_model_state_dict

    out_dir.mkdir(parents=True, exist_ok=True)
    transformer = accelerator.unwrap_model(pipeline.transformer)
    state = get_peft_model_state_dict(transformer, adapter_name=ADAPTER_NAME)
    FluxPipeline.save_lora_weights(
        save_directory=str(out_dir), transformer_lora_layers=state,
        safe_serialization=True,
    )


def prune_checkpoints(root: Path, limit: int) -> None:
    if limit <= 0:
        return
    ckpts = sorted(
        (d for d in root.glob("checkpoint-*") if d.is_dir()),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    for stale in ckpts[:-limit]:
        shutil.rmtree(stale, ignore_errors=True)


def main() -> None:
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.WARNING,
    )
    out_root = Path(args.output_dir)
    if accelerator.is_main_process:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "args.json").write_text(json.dumps(vars(args), indent=2))
        logging.getLogger().addHandler(logging.FileHandler(out_root / "train.log"))

    # Every process must agree on the AF-KC mode and the TA-CL step count, which
    # change the shared computation graph, but should draw independent noise: the
    # samplers are seeded identically while torch's RNG is offset per rank.
    set_seed(args.seed + accelerator.process_index)

    dtype = {"no": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.mixed_precision]
    device = accelerator.device

    logger.info(f"loading FLUX from {args.pretrained_model_name_or_path}",
                main_process_only=True)
    pipeline = build_pipeline(args, device, dtype)
    lora_params = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    logger.info(
        f"LoRA rank={args.rank} alpha={args.lora_alpha}: "
        f"{sum(p.numel() for p in lora_params):,} trainable params "
        f"in {len(lora_params)} tensors",
        main_process_only=True,
    )

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    dataset = PromptTripletDataset(args.train_data_path, seed=args.seed)
    logger.info(f"dataset: {dataset.coverage}", main_process_only=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate,
        drop_last=True,
    )

    pipeline.transformer, optimizer, dataloader = accelerator.prepare(
        pipeline.transformer, optimizer, dataloader
    )

    reward_model = None
    if args.reward_model_path:
        from src.utils.reward import ClipReward
        reward_model = ClipReward(args.reward_model_path, device=device, dtype=dtype)
        logger.info(f"NO-RL enabled, reward model: {args.reward_model_path}",
                    main_process_only=True)
    else:
        logger.info("NO-RL disabled (no --reward_model_path): DK-DM only",
                    main_process_only=True)

    curriculum = TimestepCurriculum(args.num_inference_steps,
                                    decay=args.curriculum_decay, seed=args.seed)
    mixer = PromptMixer(args.p_impl, args.p_expl, args.p_found, seed=args.seed)
    eman = EMANormalizer(decay=args.eman_decay)
    failure_filter = FailureFilter(threshold=args.reward_threshold,
                                   warmup=args.reward_warmup)

    logger.info(
        f"TA-CL: p(1)={curriculum.probability(1):.4f} "
        f"p({args.num_inference_steps})="
        f"{curriculum.probability(args.num_inference_steps):.4f} "
        f"E[n]={curriculum.expected_steps():.2f}",
        main_process_only=True,
    )

    res = args.resolution
    global_step = 0
    started = time.time()
    stop = False

    while not stop:
        for batch in dataloader:
            if stop:
                break

            bsz = len(batch["impl"])

            # ---- Algorithm 1, lines 3-8: AF-KC prompt pair construction ----
            mode = mixer.sample_mode()
            sources, targets = [], []
            for i in range(bsz):
                src, tgt = mixer.resolve(
                    mode, batch["impl"][i], batch["expl"][i],
                    batch["found"][i] or None,
                )
                sources.append(src)
                targets.append(tgt)

            src_embeds, src_pooled, text_ids = encode(
                pipeline, sources, device, args.max_sequence_length)
            tgt_embeds, tgt_pooled, _ = encode(
                pipeline, targets, device, args.max_sequence_length)

            _, latent_image_ids = pipeline.prepare_latents(
                bsz, transformer_config(pipeline).in_channels // 4,
                res, res, dtype, device, None,
                torch.zeros(1, device=device),  # dummy: only the ids are wanted
            )

            metrics: dict[str, float] = {}

            # ---- Phase I: DK-DM ----
            with accelerator.accumulate(pipeline.transformer):
                num_steps = curriculum.sample()
                z = sample_packed_noise(pipeline, bsz, res, res, device, dtype)

                x_tau, sigma_cur, sigma_pre = run_generator(
                    pipeline, z, src_embeds, src_pooled, text_ids, latent_image_ids,
                    height=res, width=res,
                    num_inference_steps=args.num_inference_steps,
                    num_steps=num_steps,
                    guidance_scale=args.guidance_scale,
                )

                loss_dkdm, dkdm_metrics = compute_dkdm_loss(
                    pipeline, x_tau,
                    expl_prompt_embeds=tgt_embeds, expl_pooled_embeds=tgt_pooled,
                    impl_prompt_embeds=src_embeds, impl_pooled_embeds=src_pooled,
                    text_ids=text_ids, latent_image_ids=latent_image_ids,
                    sigma_cur=sigma_cur, sigma_pre=sigma_pre,
                    height=res, width=res, guidance_scale=args.guidance_scale,
                )
                metrics.update(dkdm_metrics)
                metrics["dkdm/loss"] = float(loss_dkdm.item())
                metrics["dkdm/num_steps"] = float(num_steps)

                scaled = eman.normalize("dkdm", loss_dkdm)
                accelerator.backward(scaled * args.dkdm_weight)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            del x_tau, loss_dkdm, scaled

            # ---- Phase II: NO-RL, a separate backward pass (Sec. 3.3) ----
            run_norl = (
                reward_model is not None
                and args.norl_weight > 0
                and global_step % args.norl_every == 0
            )
            if run_norl:
                with accelerator.accumulate(pipeline.transformer):
                    # D_loss is built from explicit-prompt renders.
                    with torch.no_grad():
                        z_ref = sample_packed_noise(pipeline, bsz, res, res,
                                                    device, dtype)
                        x0_expl = run_full(
                            pipeline, z_ref, tgt_embeds, tgt_pooled,
                            text_ids, latent_image_ids,
                            height=res, width=res,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            adapter_weight=0.0,  # pi_ref
                        )
                        images = decode_latents(pipeline, x0_expl)
                        # Scored against the implicit prompt: the question is
                        # whether the render satisfies the knowledge.
                        scores = reward_model(images, batch["impl"]).float()
                        del images

                        failure_filter.observe([s.item() for s in scores])
                        is_failure = failure_filter.is_failure(scores)

                    metrics["norl/score_mean"] = float(scores.mean().item())
                    thr = failure_filter.effective_threshold
                    metrics["norl/threshold"] = (float(thr) if thr is not None
                                                 else float("nan"))

                    loss_norl, norl_metrics = compute_norl_loss(
                        pipeline, x0_expl,
                        prompt_embeds=src_embeds, pooled_embeds=src_pooled,
                        text_ids=text_ids, latent_image_ids=latent_image_ids,
                        is_failure=is_failure,
                        height=res, width=res,
                        guidance_scale=args.guidance_scale,
                        beta=args.norl_beta,
                        accelerator=accelerator,
                    )
                    metrics.update(norl_metrics)
                    metrics["norl/loss"] = float(loss_norl.item())

                    if float(is_failure.sum()) > 0:
                        scaled = eman.normalize("norl", loss_norl)
                        accelerator.backward(scaled * args.norl_weight)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(lora_params,
                                                        args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        del scaled

                    del x0_expl, loss_norl

            # ---- logging / checkpointing ----
            if global_step % args.logging_steps == 0:
                parts = [
                    f"step {global_step}",
                    f"mode={mode}",
                    f"n={int(metrics.get('dkdm/num_steps', 0))}",
                    f"dkdm={metrics.get('dkdm/loss', float('nan')):.5f}",
                    f"sigma=[{metrics.get('dkdm/sigma_cur', 0):.3f},"
                    f"{metrics.get('dkdm/sigma_pre', 0):.3f}]",
                    f"gap={metrics.get('dkdm/score_gap', float('nan')):.4f}",
                ]
                if "norl/loss" in metrics:
                    parts += [
                        f"norl={metrics['norl/loss']:.5f}",
                        f"fail={int(metrics.get('norl/n_failure', 0))}/{bsz}",
                        f"score={metrics.get('norl/score_mean', float('nan')):.2f}",
                    ]
                logger.info("  ".join(parts), main_process_only=True)

            if (global_step > 0 and global_step % args.checkpointing_steps == 0
                    and accelerator.is_main_process):
                ckpt = out_root / f"checkpoint-{global_step}"
                save_lora(pipeline, accelerator, ckpt)
                prune_checkpoints(out_root, args.checkpoints_total_limit)
                logger.info(f"saved {ckpt}", main_process_only=True)

            global_step += 1
            if global_step >= args.max_train_steps:
                stop = True

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora(pipeline, accelerator, out_root / "final")
        logger.info(f"done: {global_step} steps in {time.time() - started:.0f}s "
                    f"-> {out_root / 'final'}", main_process_only=True)


if __name__ == "__main__":
    main()
