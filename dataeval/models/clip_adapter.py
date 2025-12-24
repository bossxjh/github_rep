# dataeval/models/clip_adapter.py
import clip
import torch
import numpy as np
from PIL import Image

class CLIPAdapter:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def extract(self, frames):
        images = [
            self.preprocess(Image.fromarray(f.astype(np.uint8)))
            .unsqueeze(0)
            .to(self.device)
            for f in frames
        ]
        with torch.no_grad():
            feats = torch.cat(
                [self.model.encode_image(img) for img in images], dim=0
            )
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().flatten()
