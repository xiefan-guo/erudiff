"""Training strategies: TA-CL, AF-KC, EMAN and failure-sample filtering."""

from __future__ import annotations

import math
import random
from collections import deque

import torch


class TimestepCurriculum:
    """Timestep-aware curriculum learning (TA-CL), paper Eq. 3.

        p(n) = exp(-lambda * (n - 1)) / sum_{i=1}^{T_inference} exp(-lambda * (i - 1))

    p(n) is the probability that the generator performs only the first n
    denoising operations in an iteration. The decay prioritises early stages of
    the chain, which are more prompt-dependent and cascade into the matching of
    all later timesteps. The paper uses lambda = 0.1.
    """

    def __init__(self, num_inference_steps: int, decay: float = 0.1,
                 seed: int | None = None):
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if decay < 0:
            raise ValueError("decay must be >= 0")
        self.num_inference_steps = num_inference_steps
        self.decay = decay
        self._rng = random.Random(seed)
        self.steps = list(range(1, num_inference_steps + 1))
        weights = [math.exp(-decay * (n - 1)) for n in self.steps]
        total = sum(weights)
        self.probs = [w / total for w in weights]

    def sample(self) -> int:
        """Draw n ~ p(n), in [1, num_inference_steps]."""
        return self._rng.choices(self.steps, weights=self.probs, k=1)[0]

    def probability(self, n: int) -> float:
        if not 1 <= n <= self.num_inference_steps:
            raise ValueError(f"n must be in [1, {self.num_inference_steps}]")
        return self.probs[n - 1]

    def expected_steps(self) -> float:
        """E[n], the mean number of denoising steps per iteration."""
        return sum(n * p for n, p in zip(self.steps, self.probs))


class PromptMixer:
    """Anti-forgetting knowledge consolidation (AF-KC), paper Sec. 3.1.

    Each iteration runs one of three DK-DM variants, drawn with fixed
    probabilities (p_impl = 0.8, p_expl = 0.1, p_found = 0.1):

        impl  -> DK-DM(y_impl,  y_expl)   the knowledge transfer itself
        expl  -> DK-DM(y_expl,  y_expl)   explicit knowledge consolidation
        found -> DK-DM(y_found, y_found)  foundational knowledge consolidation

    The consolidation terms are self-matching, so their target distribution is
    the model's own pre-trained rendering. That pins existing renderings in
    place and prevents refactoring a composite fact from eroding its parts.
    """

    MODES = ("impl", "expl", "found")

    def __init__(self, p_impl: float = 0.8, p_expl: float = 0.1,
                 p_found: float = 0.1, seed: int | None = None):
        total = p_impl + p_expl + p_found
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1, got {total}")
        self.probs = (p_impl, p_expl, p_found)
        self._rng = random.Random(seed)

    def sample_mode(self) -> str:
        return self._rng.choices(self.MODES, weights=self.probs, k=1)[0]

    def resolve(self, mode: str, impl: str, expl: str,
                found: str | None) -> tuple[str, str]:
        """Map a mode to its (source, target) prompt pair.

        The source conditions the generator and s_impl, the target conditions
        s_expl. Samples without a y_found fall back to the `expl` mode.
        """
        if mode == "impl":
            return impl, expl
        if mode == "expl":
            return expl, expl
        if mode == "found":
            if not found:
                return expl, expl
            return found, found
        raise ValueError(f"unknown mode: {mode}")


class EMANormalizer:
    """Exponential moving average normalisation (EMAN), paper Sec. 3.3.

    DK-DM and NO-RL live at different scales and are backpropagated separately,
    so each loss is divided by its own EMA before the optimizer step (decay 0.99
    in the paper). The EMA is updated from the detached value, so this rescales
    the gradient without adding a spurious gradient path.
    """

    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = decay
        self.eps = eps
        self._ema: dict[str, float] = {}

    def normalize(self, name: str, loss: torch.Tensor) -> torch.Tensor:
        value = float(loss.detach().abs().item())
        if name not in self._ema:
            self._ema[name] = value
        else:
            self._ema[name] = self.decay * self._ema[name] + (1.0 - self.decay) * value
        return loss / (self._ema[name] + self.eps)

    def get(self, name: str) -> float:
        return self._ema.get(name, 0.0)


class FailureFilter:
    """Failure-sample filtering for D_loss, paper Appendix A.

    The paper renders images from randomly sampled implicit prompts with the
    pre-trained model and uses the mean of their reward scores as the threshold;
    samples scoring below it form the failure set. Pass a pre-computed
    `threshold` to reproduce that, or leave it None to calibrate online from the
    running mean of observed scores.
    """

    def __init__(self, threshold: float | None = None,
                 warmup: int = 64, history: int = 1000):
        self.threshold = threshold
        self.warmup = warmup
        self._scores: deque[float] = deque(maxlen=history)

    def observe(self, scores: list[float]) -> None:
        self._scores.extend(scores)

    @property
    def effective_threshold(self) -> float | None:
        if self.threshold is not None:
            return self.threshold
        if len(self._scores) < self.warmup:
            return None
        return sum(self._scores) / len(self._scores)

    def is_failure(self, scores: torch.Tensor) -> torch.Tensor:
        """Mark scores below the threshold as failures.

        Before calibration completes this returns all-False, so NO-RL
        contributes nothing rather than training on an arbitrary split.
        """
        thr = self.effective_threshold
        if thr is None:
            return torch.zeros_like(scores, dtype=torch.bool)
        return scores < thr


def decode_latents(pipeline, latents: torch.Tensor) -> torch.Tensor:
    """Decode unpacked FLUX latents to images in [0, 1], (B, 3, H, W)."""
    latents = latents.to(dtype=pipeline.vae.dtype)
    latents = (latents / pipeline.vae.config.scaling_factor) + \
        pipeline.vae.config.shift_factor
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    return pipeline.image_processor.postprocess(images, output_type="pt")
