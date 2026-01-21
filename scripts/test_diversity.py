# 计算得到diversity
import numpy as np
from sklearn.metrics import pairwise_distances
from dataeval.api import extract_features
from dataeval.metric.diversity import compute_task_diversity_entropy
import os

def extract_features_and_compute_diversity(model_name, dataset_name, dataset_path, num_frames=3, batch_size=16, save_root_path=None):
    """
    提取特征并计算数据集的多样性熵，并将特征保存为 npz 文件。
    """
    # 收集所有特征
    features = []
    for feat in extract_features(model_name, dataset_name, dataset_path, num_frames=num_frames, batch_size=batch_size):
        features.append(feat.flatten())  # 展平特征，确保它们是二维的

    # 转换为 NumPy 数组
    features_array = np.vstack(features)  # 将所有特征堆叠成一个数组

    # 如果提供了保存路径，则保存特征到 .npz 文件
    if save_root_path:
        # 创建文件名，包含所有信息
        save_name = f"{dataset_name}_{model_name.replace(' ', '_')}_nf{num_frames}_bs{batch_size}.npz"
        save_path = os.path.join(save_root_path, save_name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, features=features_array)
        print(f"Features saved to: {save_path}")

    # 计算多样性熵
    H_hat, sigma = compute_task_diversity_entropy(features_array)

    print(f"Calculated diversity entropy: {H_hat:.4f}, Sigma: {sigma:.4f}")
    return H_hat, sigma


if __name__ == "__main__":
    dataset_path = "/mnt/shared-storage-user/xiaojiahao/tos2/xiaojiahao/具身/libero_100/libero_10"
    model_name = "clip"
    dataset_name = "libero-10"
    
    # 定义根目录保存路径
    save_root_path = "/mnt/shared-storage-user/xiaojiahao/trans/xiaojiahao/project/github_rep/feature/diversity"  # 根目录路径
    
    H_hat, sigma = extract_features_and_compute_diversity(model_name, dataset_name, dataset_path, num_frames=3, batch_size=32, save_root_path=save_root_path)
    print(f"Final diversity entropy: {H_hat:.4f}")
    
    