import clip
from PIL import Image
import torch
import numpy as np

def extract_with_clip(frames, model, preprocess, device="cuda"):
    from PIL import Image
    import torch

    images = [
        preprocess(Image.fromarray(f.astype(np.uint8))).unsqueeze(0).to(device)
        for f in frames
    ]
    with torch.no_grad():
        feats = torch.cat([model.encode_image(img) for img in images], dim=0)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().flatten()


EXTRACTOR_MAP = {
    "clip": extract_with_clip,
}
