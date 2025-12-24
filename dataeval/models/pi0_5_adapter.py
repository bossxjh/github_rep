# dataeval/models/pi0_5_adapter.py
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
from PIL import Image

class Pi05Adapter:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 SigLIP-So400m 的视觉编码器
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _extract_vision(self, frames):
        images = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 只使用视觉编码器
        feats = self.model.vision_model(**inputs).last_hidden_state.mean(dim=1)
        return feats.cpu().numpy().flatten()

    def extract(self, frames):
        return self._extract_vision(frames)
