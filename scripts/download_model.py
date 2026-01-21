import os
from transformers import AutoModel, AutoProcessor

# 设置 Hugging Face 模型缓存路径
hf_cache_dir = "/mnt/shared-storage-user/xiaojiahao/trans/xiaojiahao/huggingface"
os.environ["HF_HOME"] = hf_cache_dir

# 模型名称
model_names = {
    "spatial": "facebook/dinov2-base",
    "temporal": "google/siglip-base-patch16-224"
}

# 下载模型并缓存到指定路径
def download_model(model_name):
    try:
        print(f"Downloading model {model_name}...")
        model = AutoModel.from_pretrained(model_name, cache_dir=hf_cache_dir)
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache_dir)
        print(f"Model {model_name} downloaded and cached successfully.")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

# 下载两个模型
for model_type, model_name in model_names.items():
    download_model(model_name)

print("All models downloaded successfully.")

# import os
# import torch
# import torchvision.models as models

# # # 1️⃣ 设置自定义缓存路径
# # cache_dir = "/mnt/shared-storage-user/xiaojiahao/trans/xiaojiahao/cache"
# # os.makedirs(cache_dir, exist_ok=True)
# # torch.hub.set_dir(cache_dir)

# # # 2️⃣ 下载并加载 ResNet18 权重
# # resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# # resnet18.eval()  # 切换到评估模式

# # print(f"ResNet18 权重已下载并缓存到: {cache_dir}")

# import os
# import clip
# import torch
# from transformers import AutoModel, AutoProcessor

# # ================== 环境变量（必须在 import HF 逻辑前）
# os.environ["TRANSFORMERS_NO_TF"] = "1"

# HF_HOME = "/mnt/shared-storage-user/xiaojiahao/trans/xiaojiahao/huggingface"
# os.environ["HF_HOME"] = HF_HOME

# # （可选）确保是“在线下载模式”
# os.environ.pop("HF_HUB_OFFLINE", None)

# device = "cpu"

# # ================== CLIP
# print("== Downloading CLIP ViT-B/32 ==")
# clip.load("ViT-B/32", device=device)

# # ================== HuggingFace models
# HF_MODELS = [
#     "facebook/dinov2-base",
#     "google/siglip-base-patch16-224",
#     "google/siglip-so400m-patch14-384",
# ]

# for model_id in HF_MODELS:
#     print(f"\n== Downloading {model_id} ==")

#     if "siglip" in model_id:
#         processor = AutoProcessor.from_pretrained(
#             model_id,
#             use_fast=True
#         )
#     else:
#         processor = AutoProcessor.from_pretrained(model_id)

#     model = AutoModel.from_pretrained(model_id)




# print("\nAll models & processors downloaded successfully.")
