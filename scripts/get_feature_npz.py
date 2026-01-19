from dataeval.api import extract_features_with_metadata
import os
import numpy as np
from tqdm import tqdm
import time

if __name__ == "__main__":
    dataset_path = "/Volumes/T7/数据集/具身/libero_spatial"
    save_dir = "/Volumes/T7/项目/具身数据评测/github_rep/feature"
    os.makedirs(save_dir, exist_ok=True)

    # 配置
    model_name = "clip"
    dataset_name = "libero"
    num_frames = 3
    batch_size = 1

    start_time = time.time()

    # 提取特征
    demo_features = []
    for feat_dict in tqdm(
        extract_features_with_metadata(model_name, dataset_name, dataset_path,
                                       num_frames=num_frames, batch_size=batch_size),
        desc="Extracting features"
    ):
        demo_features.append(feat_dict)

    total_time = time.time() - start_time

    # 打印信息
    print(f"demo 数量: {len(demo_features)}")
    print(f"demo 字典 keys: {demo_features[0].keys()}")
    print(f"第一个 demo 特征 shape: {demo_features[0]['features'].shape}")
    print(f"总耗时: {total_time:.2f} 秒")

    # 准备保存
    features_array = np.stack([d["features"] for d in demo_features], axis=0)
    task_ids = np.array([d["task_id"] for d in demo_features])
    task_lengths = np.array([d["task_length"] for d in demo_features])
    demo_lengths = np.array([d["demo_length"] for d in demo_features])
    task_descriptions = np.array([d["task_description"] for d in demo_features], dtype=object)

    # 保存文件名包含配置信息
    save_name = f"{dataset_name}_{model_name.replace(' ', '_')}_nf{num_frames}_bs{batch_size}.npz"
    save_path = os.path.join(save_dir, save_name)

    np.savez_compressed(
        save_path,
        features=features_array,
        task_ids=task_ids,
        task_lengths=task_lengths,
        demo_lengths=demo_lengths,
        task_descriptions=task_descriptions
    )
    print(f"保存完毕: {save_path}")