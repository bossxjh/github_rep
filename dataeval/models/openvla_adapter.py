# dataeval/models/openvla_adapter.py
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from PIL import Image

class OpenVLAAdapter:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 两个 ViT backbone
        self.vit_spatial_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
        self.vit_spatial = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        self.vit_spatial.eval()

        self.vit_temporal_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.vit_temporal = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)
        self.vit_temporal.eval()

    @torch.no_grad()
    def _extract_spatial(self, frames):
        images = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        inputs = self.vit_spatial_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.vit_spatial(**inputs).last_hidden_state.mean(dim=1)
        return feats.cpu().numpy().flatten()

    @torch.no_grad()
    def _extract_temporal(self, frames):
        images = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        inputs = self.vit_temporal_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 只用视觉分支
        feats = self.vit_temporal.vision_model(**inputs).last_hidden_state.mean(dim=1)
        return feats.cpu().numpy().flatten()

    def extract(self, frames):
        feat1 = self._extract_spatial(frames)
        feat2 = self._extract_temporal(frames)
        return np.concatenate([feat1, feat2], axis=-1)
