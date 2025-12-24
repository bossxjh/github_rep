# dataeval/models/openvla_adapter.py
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from PIL import Image

class OpenVLAAdapter:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.vit_spatial_processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
        self.vit_spatial = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        self.vit_spatial.eval()

        self.vit_temporal_processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224",
            use_fast=True
        )
        self.vit_temporal = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(self.device)
        self.vit_temporal.eval()

    @torch.no_grad()
    def extract_batch(self, batch_frames):
        """
        batch_frames: List[np.ndarray], each [T, H, W, 3]
        return: np.ndarray [B, D_total]
        """
        B = len(batch_frames)
        T = batch_frames[0].shape[0]

        images = []
        for frames in batch_frames:
            for f in frames:
                images.append(Image.fromarray(f.astype(np.uint8)))

        # ===== spatial =====
        inputs_s = self.vit_spatial_processor(images=images, return_tensors="pt")
        inputs_s = {k: v.to(self.device) for k, v in inputs_s.items()}
        feat_s = self.vit_spatial(**inputs_s).last_hidden_state.mean(dim=1)
        feat_s = feat_s.view(B, T, -1).mean(dim=1)  # [B, Ds]

        # ===== temporal (SigLIP vision only) =====
        inputs_t = self.vit_temporal_processor(images=images, return_tensors="pt")
        inputs_t = {k: v.to(self.device) for k, v in inputs_t.items()}
        feat_t = self.vit_temporal.vision_model(**inputs_t).last_hidden_state.mean(dim=1)
        feat_t = feat_t.view(B, T, -1).mean(dim=1)  # [B, Dt]

        feats = torch.cat([feat_s, feat_t], dim=-1)
        return feats.cpu().numpy()

    def extract(self, frames):
        return self.extract_batch([frames])[0]
