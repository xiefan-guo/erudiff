"""Truncated-gradient image generator g_theta used by DK-DM (paper Sec. 3.1).

Gradients are restricted to the final denoising step, and each iteration
denoises only `n` steps (n ~ p(n), Eq. 3), so the distribution is matched at the
intermediate point x_{tau_i} rather than at a fully denoised image.

FLUX shape conventions (diffusers >= 0.40):
  - the transformer consumes PACKED latents, (B, seq, C * 4) with
    seq = (h / 2) * (w / 2) over the latent grid
  - `_unpack_latents` maps packed latents back to (B, C, H_lat, W_lat)
  - `latent_image_ids` / `text_ids` are UNBATCHED, (seq, 3)
"""

from __future__ import annotations

import numpy as np
import torch

ADAPTER_NAME = "erudiff"


def transformer_config(pipeline):
    """Return the transformer config, looking through any DDP wrapper."""
    transformer = pipeline.transformer
    return getattr(transformer, "module", transformer).config


def set_adapter_scale(pipeline, weight: float) -> None:
    """Set the LoRA scale: 1.0 selects pi_theta, 0.0 recovers pi_ref."""
    inner = getattr(pipeline.transformer, "module", pipeline.transformer)
    inner.set_adapters([ADAPTER_NAME], weights=[weight])


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    """Resolution-dependent timestep shift mu used by the FLUX scheduler."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def latent_grid_size(height: int, width: int, vae_scale_factor: int) -> tuple[int, int]:
    """Latent-grid (h, w) for a pixel resolution, matching `prepare_latents`."""
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))
    return h, w


def sample_packed_noise(
    pipeline,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample z ~ N(0, I) in FLUX's packed layout, (B, seq, C * 4)."""
    num_channels_latents = transformer_config(pipeline).in_channels // 4
    h, w = latent_grid_size(height, width, pipeline.vae_scale_factor)
    latents = torch.randn(
        (batch_size, num_channels_latents, h, w),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return pipeline._pack_latents(latents, batch_size, num_channels_latents, h, w)


def _setup_schedule(pipeline, packed_latents: torch.Tensor, num_inference_steps: int,
                    device: torch.device):
    """Configure the scheduler as `FluxPipeline.__call__` does."""
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        packed_latents.shape[1],
        pipeline.scheduler.config.base_image_seq_len,
        pipeline.scheduler.config.max_image_seq_len,
        pipeline.scheduler.config.base_shift,
        pipeline.scheduler.config.max_shift,
    )
    pipeline.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    return pipeline.scheduler.timesteps


def _guidance(pipeline, batch_size: int, guidance_scale: float,
              device: torch.device) -> torch.Tensor | None:
    if not transformer_config(pipeline).guidance_embeds:
        return None
    return torch.full([batch_size], guidance_scale, device=device, dtype=torch.float32)


@torch.no_grad()
def run_full(
    pipeline,
    packed_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    adapter_weight: float = 0.0,
) -> torch.Tensor:
    """Denoise all the way to x_0 without gradients.

    `adapter_weight=0.0` renders with the frozen reference model pi_ref (used to
    build D_loss for NO-RL); 1.0 renders with the fine-tuned pi_theta.

    Returns unpacked latents, (B, C, H_lat, W_lat).
    """
    device = packed_latents.device
    latents = packed_latents
    timesteps = _setup_schedule(pipeline, latents, num_inference_steps, device)
    guidance = _guidance(pipeline, latents.shape[0], guidance_scale, device)

    set_adapter_scale(pipeline, adapter_weight)
    for t in timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipeline.transformer(
            hidden_states=latents,
            timestep=timestep / 1000.0,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)


def run_generator(
    pipeline,
    packed_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    height: int,
    width: int,
    num_inference_steps: int,
    num_steps: int,
    guidance_scale: float,
) -> tuple[torch.Tensor, float, float]:
    """Run g_theta for `num_steps` steps, keeping gradients only on the last one.

    Args:
        packed_latents: z ~ N(0, I) in packed layout, (B, seq, C * 4).
        num_steps: n in Eq. 3, 1-indexed; must lie in [1, num_inference_steps].

    Returns:
        (x_tau_i, sigma_cur, sigma_pre). x_tau_i is unpacked
        (B, C, H_lat, W_lat); sigma_cur is the noise level after the last
        executed step (tau_i) and sigma_pre the level before it (tau_{i+1}).
        DK-DM samples its matching timestep inside [sigma_cur, sigma_pre].
    """
    if not 1 <= num_steps <= num_inference_steps:
        raise ValueError(
            f"num_steps must be in [1, {num_inference_steps}], got {num_steps}"
        )

    device = packed_latents.device
    latents = packed_latents
    timesteps = _setup_schedule(pipeline, latents, num_inference_steps, device)
    guidance = _guidance(pipeline, latents.shape[0], guidance_scale, device)

    set_adapter_scale(pipeline, 1.0)

    for i, t in enumerate(timesteps[:num_steps]):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        is_last = i == num_steps - 1

        # Gradient truncation: only the final step stays in the autograd graph.
        with torch.enable_grad() if is_last else torch.no_grad():
            noise_pred = pipeline.transformer(
                hidden_states=latents,
                timestep=timestep / 1000.0,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            latents = pipeline.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]

    x_tau = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)

    # `sigmas` holds num_inference_steps + 1 entries, so after `num_steps` steps
    # the trajectory sits at sigmas[num_steps].
    sigma_cur = pipeline.scheduler.sigmas[num_steps].item()
    sigma_pre = pipeline.scheduler.sigmas[num_steps - 1].item()

    return x_tau, sigma_cur, sigma_pre
