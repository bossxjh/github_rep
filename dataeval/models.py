import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel


def load_model(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "clip":
        import clip
        model, processor = clip.load("ViT-B/32", device=device)
        return model, processor, device

    raise ValueError(f"Unsupported model: {model_name}")
