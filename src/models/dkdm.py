"""Diffusion Knowledge Distribution Matching (DK-DM), paper Sec. 3.1.

DK-DM minimises E_t[ D_KL(p_impl,t || p_expl,t) ], aligning the distribution
induced by an implicit prompt with the one induced by its explicit anchor. The
gradient (Eq. 2) is

    grad L_DK-DM = -E_{t, tau_i, z}[ (s_expl(F(x_tau_i, t), t)
                                     - s_impl(F(x_tau_i, t), t)) d x_tau_i / d theta ]

where x_tau_i = g_theta(z, tau_i), s_expl is the frozen pre-trained score
conditioned on y_expl, s_impl the fine-tuned score conditioned on y_impl, and
t ~ U(T_min, T_max) with T_min = tau_i + 0.02 * (tau_{i+1} - tau_i) and
T_max = tau_i + 0.98 * (tau_{i+1} - tau_i).

Both scores come from the same weights: the LoRA adapter at scale 0 gives
s_expl (pi_ref) and at scale 1 gives s_impl (pi_theta), so no separately trained
fake denoiser is required.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.generator import set_adapter_scale, transformer_config


def flow_pred_to_x0(
    model_output: torch.Tensor,
    sigma: torch.Tensor,
    sample: torch.Tensor,
    sigma_target: float = 0.0,
) -> torch.Tensor:
    """Map a FLUX velocity prediction to the state at `sigma_target`.

    FLUX predicts v = (noise - x_0) along x_sigma = (1 - sigma) * x_0 +
    sigma * noise, hence x_{sigma_target} = x_sigma + (sigma_target - sigma) * v.
    `sigma_target` is non-zero here because full-timestep matching uses x_{tau_i}
    as the reference point rather than a fully denoised image.
    """
    sample = sample.to(torch.float32)
    model_output = model_output.to(torch.float32)
    return sample + (sigma_target - sigma) * model_output


def sample_matching_sigma(
    batch_size: int,
    sigma_cur: float,
    sigma_pre: float,
    device: torch.device,
    t_min_frac: float = 0.02,
    t_max_frac: float = 0.98,
) -> torch.Tensor:
    """Sample t ~ U(T_min, T_max) inside the interval [tau_i, tau_{i+1}]."""
    span = sigma_pre - sigma_cur
    t_min = sigma_cur + t_min_frac * span
    t_max = sigma_cur + t_max_frac * span
    u = torch.rand(batch_size, device=device)
    return t_min + u * (t_max - t_min)


def diffuse_from_x_tau(
    x_tau: torch.Tensor,
    sigma: torch.Tensor,
    sigma_cur: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Forward diffusion F(x_tau_i, t): renoise x_tau_i from sigma_cur to sigma.

    x_tau_i already carries noise level sigma_cur, so moving it to a higher level
    without knowing x_0 means interpolating toward fresh noise along the path:

        x_sigma = (1 - sigma) / (1 - sigma_cur) * x_tau_i
                  + (sigma - sigma_cur) / (1 - sigma_cur) * eps
    """
    denom = 1.0 - sigma_cur
    w_x = (1.0 - sigma) / denom
    w_n = (sigma - sigma_cur) / denom
    return w_x * x_tau + w_n * noise


def compute_dkdm_loss(
    pipeline,
    x_tau: torch.Tensor,
    *,
    expl_prompt_embeds: torch.Tensor,
    expl_pooled_embeds: torch.Tensor,
    impl_prompt_embeds: torch.Tensor,
    impl_pooled_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    sigma_cur: float,
    sigma_pre: float,
    height: int,
    width: int,
    guidance_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute L_DK-DM for one batch.

    Args:
        x_tau: g_theta(z, tau_i), unpacked (B, C, h, w), gradient-carrying.
        expl_*: embeddings of the explicit anchor y_expl (target distribution).
        impl_*: embeddings of the implicit prompt y_impl.
        text_ids / latent_image_ids: unbatched FLUX position ids, (seq, 3).
        sigma_cur / sigma_pre: the interval [tau_i, tau_{i+1}] from the generator.

    Returns:
        (loss, metrics). The loss backpropagates into theta through x_tau only.
    """
    bsz, channels, h, w = x_tau.shape
    device = x_tau.device

    sigma = sample_matching_sigma(bsz, sigma_cur, sigma_pre, device)
    sigma_b = sigma.view(bsz, 1, 1, 1).to(torch.float32)

    noise = torch.randn_like(x_tau, dtype=torch.float32)
    x_noisy = diffuse_from_x_tau(x_tau.to(torch.float32), sigma_b, sigma_cur, noise)
    x_noisy = x_noisy.to(x_tau.dtype)

    packed = pipeline._pack_latents(x_noisy, bsz, channels, h, w)
    timestep = sigma.to(x_tau.dtype)  # FLUX consumes sigma as timestep / 1000

    guidance = None
    if transformer_config(pipeline).guidance_embeds:
        guidance = torch.full([bsz], guidance_scale, device=device, dtype=torch.float32)

    def score_to_x_tau(prompt_embeds, pooled_embeds, adapter_weight: float):
        set_adapter_scale(pipeline, adapter_weight)
        pred = pipeline.transformer(
            hidden_states=packed,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        pred = pipeline._unpack_latents(pred, height, width, pipeline.vae_scale_factor)
        return flow_pred_to_x0(pred, sigma_b, x_noisy, sigma_target=sigma_cur)

    # Neither score needs gradients: the gradient enters the graph only through
    # x_tau, via the stop-gradient construction below.
    with torch.no_grad():
        pred_expl = score_to_x_tau(expl_prompt_embeds, expl_pooled_embeds, 0.0)
        pred_impl = score_to_x_tau(impl_prompt_embeds, impl_pooled_embeds, 1.0)

    set_adapter_scale(pipeline, 1.0)

    x_tau_f32 = x_tau.to(torch.float32)

    # Eq. 2, normalised by the mean displacement toward the target distribution
    # so the step size is invariant to the noise level.
    grad = pred_impl - pred_expl
    normalizer = (x_tau_f32 - pred_expl).abs().mean(dim=[1, 2, 3], keepdim=True)
    grad = grad / (normalizer + 1e-8)

    # d/dtheta 0.5 * ||x - sg(x - g)||^2 = g * dx/dtheta.
    target = (x_tau_f32 - grad).detach()
    loss = 0.5 * F.mse_loss(x_tau_f32, target, reduction="mean")

    metrics = {
        "dkdm/sigma_cur": float(sigma_cur),
        "dkdm/sigma_pre": float(sigma_pre),
        "dkdm/sigma_sampled": float(sigma.mean().item()),
        "dkdm/grad_abs_mean": float(grad.abs().mean().item()),
        "dkdm/normalizer": float(normalizer.mean().item()),
        "dkdm/score_gap": float((pred_impl - pred_expl).abs().mean().item()),
    }
    return loss, metrics
