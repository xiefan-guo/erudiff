"""CLIP-style reward model wrapper used to build D_loss for NO-RL.

Any CLIP-architecture checkpoint loadable through `AutoModel` works; the paper
uses SciScore for the Science-T2I setting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


class ClipReward(nn.Module):
    """Scaled image-text cosine similarity, higher is better."""

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device, dtype=dtype)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Score a batch of images against their prompts.

        Args:
            images: (B, C, H, W) float tensor in [0, 1].
            prompts: B prompt strings.

        Returns:
            (B,) tensor of similarity scores.
        """
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)

        image_inputs = self.processor(
            images=images, return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        image_embeds = F.normalize(self.model.get_image_features(**image_inputs), dim=-1)
        text_embeds = F.normalize(self.model.get_text_features(**text_inputs), dim=-1)

        similarity = (image_embeds * text_embeds).sum(dim=-1)
        return similarity * self.model.logit_scale.exp()
