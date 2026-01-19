import numpy as np
import random

# npz 文件路径
# npz_path = "/Volumes/T7/项目/具身数据评测/github_rep/feature/libero_spatial_clip_nf3_bs16.npz"
npz_path = "/Volumes/T7/项目/具身数据评测/features/Libero Spatial.npz"

# 加载 npz
data = np.load(npz_path, allow_pickle=True)

print("包含的 key:")
print(list(data.keys()))





# 查看一些基本信息
print("\nfeatures shape:", data['features'].shape)
print("task_ids shape:", data['task_ids'].shape)
print("task_lengths shape:", data['task_lengths'].shape)
print("demo_lengths shape:", data['demo_lengths'].shape)
print("task_descriptions shape:", data['task_descriptions'].shape)

# 随机抽取几个 demo
num_samples = 3
indices = random.sample(range(data['features'].shape[0]), num_samples)

print(f"\n随机抽取 {num_samples} 个 demo 的信息:")
for i in indices:
    print(f"\nDemo index: {i}")
    print("task_id:", data['task_ids'][i])
    print("task_length:", data['task_lengths'][i])
    print("demo_length:", data['demo_lengths'][i])
    print("task_description:", data['task_descriptions'][i])
    print("features shape:", data['features'][i].shape)
    print("features (前5维):", data['features'][i][:5])  # 只展示前5维方便查看
