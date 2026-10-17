"""Negative-Only Reinforcement Learning (NO-RL), paper Sec. 3.2.

NO-RL corrects the residual bias a pre-trained model exhibits when rendering
explicit prompts, using failure samples only, as a variant of KTO that needs
binary feedback rather than paired preferences.

Objective (Eq. 4):

    max_{pi_theta} E_{x0 ~ D_loss, t}[ U( -( beta * log(pi_theta(x_{t-1}|x_t)
                                            / pi_ref(x_{t-1}|x_t)) - Q_ref ) ) ]

with U a monotonically increasing value function (a sigmoid here) and the
reference point

    Q_ref = max(0, (1/m) * sum beta * log(pi_theta(x'_{t-1}|x'_t)
                                          / pi_ref(x'_{t-1}|x'_t)))

estimated over m unrelated (x'_{t-1}, x'_t) pairs.

The intractable per-step log-ratio is replaced by the negative diffusion loss,
log pi(x_{t-1}|x_t) ~= -L_diff, so the ratio becomes (L_ref - L_theta).
"""

from __future__ import annotations

import torch

from src.models.generator import set_adapter_scale, transformer_config


def sample_flow_sigma(
    batch_size: int,
    device: torch.device,
    sigma_min: float = 0.02,
    sigma_max: float = 0.98,
) -> torch.Tensor:
    """Sample t ~ U[0, T] as a rectified-flow noise level, clipped off the ends.

    The endpoints are excluded: at sigma = 0 the loss carries no signal and at
    sigma = 1 the sample is pure noise, so neither says anything about how the
    model renders this particular image.
    """
    u = torch.rand(batch_size, device=device)
    return sigma_min + u * (sigma_max - sigma_min)


def _diffusion_loss(
    pipeline,
    latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    height: int,
    width: int,
    guidance_scale: float,
    adapter_weight: float,
) -> torch.Tensor:
    """Per-sample rectified-flow matching loss, (B,).

    Stands in for -log pi(x_{t-1} | x_t): the flow-matching target is
    v* = noise - x_0, so the squared error against the predicted velocity is the
    per-step negative log-likelihood up to a constant.
    """
    bsz, channels, h, w = latents.shape
    sigma_b = sigma.view(bsz, 1, 1, 1).to(torch.float32)

    x0 = latents.to(torch.float32)
    x_noisy = (1.0 - sigma_b) * x0 + sigma_b * noise
    target = noise - x0

    packed = pipeline._pack_latents(x_noisy.to(latents.dtype), bsz, channels, h, w)

    guidance = None
    if transformer_config(pipeline).guidance_embeds:
        guidance = torch.full(
            [bsz], guidance_scale, device=latents.device, dtype=torch.float32
        )

    set_adapter_scale(pipeline, adapter_weight)
    pred = pipeline.transformer(
        hidden_states=packed,
        timestep=sigma.to(latents.dtype),
        guidance=guidance,
        pooled_projections=pooled_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]
    pred = pipeline._unpack_latents(pred, height, width, pipeline.vae_scale_factor)

    return ((pred.to(torch.float32) - target) ** 2).reshape(bsz, -1).mean(dim=1)


def compute_norl_loss(
    pipeline,
    latents: torch.Tensor,
    *,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    is_failure: torch.Tensor,
    height: int,
    width: int,
    guidance_scale: float,
    beta: float = 10.0,
    accelerator=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute L_NO-RL over a batch, counting failure samples only.

    Args:
        latents: clean latents x_0 of images rendered from explicit prompts,
            unpacked (B, C, h, w). No gradient is needed on the input itself.
        is_failure: (B,) mask, True where the reward model judged the render a
            failure; these samples form D_loss.
        beta: KL penalty strength from Eq. 4.
        accelerator: used to average Q_ref across ranks.

    Returns:
        (loss, metrics). The loss is a differentiable zero when the batch holds
        no failures.
    """
    bsz = latents.shape[0]
    device = latents.device
    is_failure = is_failure.to(device=device, dtype=torch.float32).reshape(-1)

    sigma = sample_flow_sigma(bsz, device)
    noise = torch.randn_like(latents, dtype=torch.float32)

    common = dict(
        latents=latents,
        noise=noise,
        sigma=sigma,
        text_ids=text_ids,
        latent_image_ids=latent_image_ids,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
    )

    # Policy pi_theta: gradients flow here.
    loss_theta = _diffusion_loss(
        pipeline, prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds,
        adapter_weight=1.0, **common,
    )

    # Reference pi_ref: frozen.
    with torch.no_grad():
        loss_ref = _diffusion_loss(
            pipeline, prompt_embeds=prompt_embeds, pooled_embeds=pooled_embeds,
            adapter_weight=0.0, **common,
        )

    log_ratio = loss_ref - loss_theta

    # Unrelated pairs: roll the prompt batch by one so each image is scored
    # against a different sample's prompt. Requires B >= 2.
    with torch.no_grad():
        if bsz > 1:
            shift = torch.roll(torch.arange(bsz, device=device), shifts=1)
            mism_prompt = prompt_embeds[shift]
            mism_pooled = pooled_embeds[shift]
            q_theta = _diffusion_loss(
                pipeline, prompt_embeds=mism_prompt, pooled_embeds=mism_pooled,
                adapter_weight=1.0, **common,
            )
            q_ref_loss = _diffusion_loss(
                pipeline, prompt_embeds=mism_prompt, pooled_embeds=mism_pooled,
                adapter_weight=0.0, **common,
            )
            q_ref = (beta * (q_ref_loss - q_theta)).mean()
        else:
            q_ref = torch.zeros((), device=device)

        if accelerator is not None and accelerator.num_processes > 1:
            q_ref = accelerator.reduce(q_ref.detach(), reduction="mean")
        q_ref = q_ref.clamp(min=0.0).detach()  # max(0, .) from Eq. 4

    set_adapter_scale(pipeline, 1.0)

    # Negative samples maximise U(-reward), so the loss is
    # 1 - sigmoid(-reward) = sigmoid(reward).
    implicit_reward = beta * log_ratio - q_ref
    neg_loss = torch.sigmoid(implicit_reward)

    n_failure = is_failure.sum()
    if float(n_failure) == 0.0:
        # Keep a graph edge so callers can backward() unconditionally.
        loss = (loss_theta * 0.0).sum()
    else:
        loss = (is_failure * neg_loss).sum() / n_failure

    metrics = {
        "norl/n_failure": float(n_failure.item()),
        "norl/frac_failure": float(is_failure.mean().item()),
        "norl/log_ratio": float(log_ratio.mean().item()),
        "norl/q_ref": float(q_ref.item()),
        "norl/implicit_reward": float(implicit_reward.mean().item()),
        "norl/loss_theta": float(loss_theta.mean().item()),
        "norl/loss_ref": float(loss_ref.mean().item()),
    }
    return loss, metrics
