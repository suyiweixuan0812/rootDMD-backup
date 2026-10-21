"""Differentiable local PickScore proxy for pair-reward training.

The paper names Soft-TIFA/Qwen3-VL as its scorer.  That model is not present in
the offline cache, while PickScore is available, so this adapter is explicitly
labelled as a proxy in experiment logs.
"""

from __future__ import annotations

import torch
import torchvision.transforms as T
from transformers import AutoModel, AutoProcessor


class DifferentiablePickScore(torch.nn.Module):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        super().__init__()
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device=device, dtype=dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        image_processor = self.processor.image_processor
        size = image_processor.size
        if isinstance(size, dict):
            edge = size.get("shortest_edge", size.get("height", 224))
        else:
            edge = size
        crop = image_processor.crop_size
        if isinstance(crop, dict):
            crop = crop.get("height", crop.get("width", edge))
        mean, std = image_processor.image_mean, image_processor.image_std
        self.image_transform = T.Compose([
            T.Resize(edge), T.CenterCrop(crop), T.Normalize(mean=mean, std=std),
        ])
        self.device = torch.device(device)
        self.dtype = dtype

    def forward(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be NCHW RGB tensors")
        pixels = self.image_transform(images.float()).to(device=self.device, dtype=self.dtype)
        text = self.processor(
            text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt",
        )
        text = {key: value.to(self.device) for key, value in text.items()}
        image_embs = self.model.get_image_features(pixel_values=pixels)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        with torch.no_grad():
            text_embs = self.model.get_text_features(**text)
            text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            scale = self.model.logit_scale.exp()
        # PickScore's published implementation uses the diagonal prompt/image score.
        return (scale * (text_embs.to(image_embs.dtype) * image_embs).sum(dim=-1)) / 26.0
