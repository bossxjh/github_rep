import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from scipy.stats import spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# compute_learning_ease_with_task_transfer
# │
# ├─ Step 0: 基本准备（task_ids, 比例 pi）
# │
# ├─ Step 1: 对每个 task 单独算一个 L_t（task 内部可学习性）
# │   ├─ task 内样本相似度 S_t
# │   ├─ 局部 entropy（h_local）
# │   ├─ 表征复杂度 R_t
# │   ├─ 密度 / 有效覆盖度 E_t
# │   └─ 得到 raw L_t
# │
# ├─ Step 2: 任务之间的相似度（task → task 迁移）
# │   ├─ task center
# │   ├─ task similarity matrix
# │   └─ 跨任务加权 L_t_adj
# │
# └─ 输出：
#     ├─ L_dataset
#     └─ 每个 task 的 L_t_adj


def covariance_entropy(X):
    """
    X: (N_t, D)
    返回归一化协方差熵
    """
    if X.shape[0] <= 1:
        return 0.0
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)  # (D, D)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    p = eigvals / eigvals.sum()
    H = -np.sum(p * np.log(p))
    # 归一化到 [0,1]
    # H_norm = H / np.log(len(p))
    H_norm = H
    return H_norm

def compute_learning_ease_with_task_transfer(task_groups, beta=0.8, sigma=None):
    """
    Task-centric version of compute_learning_ease_with_task_transfer.
    
    Args:
        task_groups: list of dicts, each dict has
            {
                "task_id": int,
                "features": (Nt, D) np.array,
                "demo_lengths": (Nt,) np.array,
                "task_length": float,
                "task_description": str
            }
        beta: trade-off coefficient
        sigma: kernel bandwidth (float or None)
    
    Returns:
        dataset_score: float
        task_scores: dict, task_id -> float
    """

    task_ids = [g["task_id"] for g in task_groups]
    n_tasks = len(task_groups)

    # Step 1: 计算每个 task 的 L_t 和 task center
    L_t_raw = {}
    task_centers = {}

    for i, g in enumerate(task_groups):
        X_t = g["features"]
        N_t = X_t.shape[0]
        if N_t <= 1:
            L_t_raw[g["task_id"]] = 0.0
            task_centers[g["task_id"]] = X_t.mean(axis=0) if N_t>0 else np.zeros(X_t.shape[1])
            continue

        # pairwise squared distances
        dists_t = pairwise_distances(X_t, X_t, metric="euclidean") ** 2
        sigma_t = 0.001 if sigma is None else sigma
        S_t = np.exp(-dists_t / (2 * sigma_t**2))
        P_t = S_t / S_t.sum(axis=1, keepdims=True)

        h_local = -np.sum(P_t * np.log(P_t + 1e-12), axis=1).mean()
        d_avg = np.mean(np.sqrt(dists_t[np.triu_indices(N_t, 1)]))

        # R_t 可以按原来的 covariance_entropy
        R_t = covariance_entropy(X_t) * np.tanh(d_avg / sigma_t)

        rho_t = S_t.mean(axis=1)
        E_t = rho_t.mean()
        E_t = E_t / np.log10(1 + g["task_length"])

        # L_t = (R_t**beta) * (E_t**(1-beta))
        L_t_raw[g["task_id"]] = (R_t**beta) * (E_t**(1-beta))
        print(L_t_raw)

        # task center
        task_centers[g["task_id"]] = X_t.mean(axis=0)

    # Step 2: 计算 task → task 相似度
    centers_array = np.stack([task_centers[t] for t in task_ids])
    center_dists = pairwise_distances(centers_array, centers_array, metric="euclidean") ** 2
    sigma_center = np.median(np.sqrt(center_dists)) if sigma is None else sigma
    S_task = np.exp(-center_dists / (2 * sigma_center**2))

    # Step 3: 跨任务加权 L_t_adj
    # 这里可以用 pi 权重或均匀权重
    task_scores = {}
    pi_first = 0.1
    pi_second = 0.1
    split_idx = n_tasks // 2
    pi_scale = 0.01698373

    for i, t in enumerate(task_ids):
        pi_t = pi_first if i < split_idx else pi_second
        pi_t = np.tanh(pi_t / pi_scale)

        L_t_adj = sum(S_task[i, j] * L_t_raw[task_ids[j]] for j in range(n_tasks))
        task_scores[t] = L_t_adj * pi_t

    # Step 4: dataset-level leanability
    dataset_score = np.mean(list(task_scores.values()))

    return dataset_score, task_scores




# def compute_task_diversity_entropy(X, sigma=0.1, eps=1e-12):
#     """
#     Kernel-based entropy estimator:
#       H_hat = - (1/N) * sum_i log( (1/N) * sum_j K_sigma(x_i, x_j) )
#     """
#     X = np.asarray(X)
#     N = X.shape[0]

#     # pairwise euclidean distances
#     dists = pairwise_distances(X, X, metric="euclidean")

#     # choose sigma if not provided: median of upper triangular distances
#     if sigma is None:
#         iu = np.triu_indices(N, k=1)
#         if iu[0].size > 0:
#             sigma = np.median(dists[iu])
#             if sigma == 0:
#                 nonzero = dists[iu][dists[iu] > 0]
#                 sigma = np.median(nonzero) if nonzero.size > 0 else 1.0
#         else:
#             sigma = 1.0

#     K = np.exp(-(dists**2) / (2.0 * (sigma ** 2)))
#     inner = K.mean(axis=1)
#     H_hat = - np.mean(np.log(inner + eps))
#     return H_hat, sigma

# def subsample_dataset(X, y, dataset_name):
#     """按照比例随机下采样，每个任务保留一定比例的样本"""
#     if dataset_name not in dataset_ratios:
#         return X, y

#     ratios = dataset_ratios[dataset_name]
#     unique_tasks = np.unique(y)

#     new_X_list, new_y_list = [], []
#     rng = np.random.default_rng(42)

#     n_tasks = len(unique_tasks)
#     split_idx = n_tasks // 2  # 前一半用 first，后一半用 second

#     for i, t in enumerate(unique_tasks):
#         X_t = X[y == t]
#         if i < split_idx:
#             ratio = ratios["first"]
#         else:
#             ratio = ratios["second"]

#         sample_size = int(np.round(len(X_t) * ratio))
#         sample_size = max(sample_size, 1)  # 至少保留一个样本

#         idx = rng.choice(len(X_t), size=sample_size, replace=False)
#         new_X_list.append(X_t[idx])
#         new_y_list.append(np.full(sample_size, t))

#     return np.vstack(new_X_list), np.concatenate(new_y_list)

# def covariance_entropy(X):
#     """
#     X: (N_t, D)
#     返回归一化协方差熵
#     """
#     if X.shape[0] <= 1:
#         return 0.0
#     X_centered = X - X.mean(axis=0)
#     cov = np.cov(X_centered, rowvar=False)  # (D, D)
#     eigvals = np.linalg.eigvalsh(cov)
#     eigvals = np.maximum(eigvals, 1e-12)
#     p = eigvals / eigvals.sum()
#     H = -np.sum(p * np.log(p))
#     # 归一化到 [0,1]
#     # H_norm = H / np.log(len(p))
#     H_norm = H
#     return H_norm

# def compute_learning_ease_with_task_transfer(X, y, task_lengths, dataset_name, beta=0.8, pi_scale=0.01698373, sigma=None):
#     """
#     Compute Learning Ease for a dataset with task-level transfer consideration.
    
#     Args:
#         X: np.array, shape (N, D), features
#         y: np.array, shape (N,), task ids
#         task_lengths: dict, 每个任务的长度
#         dataset_name: str, 当前数据集名字，用于查 pi 比例
#         beta: float, trade-off between robustness and overfitting ease
#         sigma: float or None, kernel bandwidth for similarity (if None, median distance is used)
#     """
#     task_ids = np.unique(y)
#     N = X.shape[0]

#     L_t_raw = {}
#     task_centers = {}

#     # 取当前数据集对应的比例
#     ratios = dataset_ratios.get(dataset_name, {"first": 0.1, "second": 0.1})
#     n_tasks = len(task_ids)
#     split_idx = n_tasks // 2

#     # Step 1: compute raw L_t per task
#     for i, t in enumerate(task_ids):
#         X_t = X[y == t]
#         N_t = X_t.shape[0]
#         if N_t <= 1:
#             continue
        
#         # pairwise squared distances
#         dists_t = pairwise_distances(X_t, X_t, metric="euclidean") ** 2
#         sigma_t = 0.001 if sigma is None else sigma
#         S_t = np.exp(-dists_t / (2 * sigma_t**2))
#         P_t = S_t / S_t.sum(axis=1, keepdims=True)

#         h_local = -np.sum(P_t * np.log(P_t + 1e-12), axis=1).mean()
#         d_avg = np.mean(np.sqrt(dists_t[np.triu_indices(N_t, 1)]))

#         R_t = covariance_entropy(X_t) * np.tanh(d_avg / sigma_t)

#         rho_t = S_t.mean(axis=1)
#         E_t = rho_t.mean()
#         if t in task_lengths:
#             E_t = E_t / np.log10(1 + task_lengths[t])

#         # 🔑 用固定比例替代 N_t/N
#         pi_t = ratios["first"] if i < split_idx else ratios["second"]
#         pi_scale = 0.01698373 #0.02016270
#         pi_t = np.tanh(pi_t / pi_scale)
#         L_t = (R_t**beta) * (E_t**(1 - beta))
#         L_t_raw[t] = L_t
#         task_centers[t] = X_t.mean(axis=0)

#     # Step 2: task similarity
#     centers_array = np.stack([task_centers[t] for t in task_ids])
#     center_dists = pairwise_distances(centers_array, centers_array, metric="euclidean") ** 2
#     sigma_center = np.median(np.sqrt(center_dists)) if sigma is None else 0.01
#     S_task = np.exp(-center_dists / (2 * sigma_center**2))

#     L_t_adjusted = {}
#     for i, t in enumerate(task_ids):
#         # print("sinma_center:", sigma_center)
#         L_t_adj = sum(S_task[i, j] * L_t_raw[task_ids[j]] for j in range(len(task_ids)))
#         L_t_adjusted[t] = L_t_adj*pi_t

#     L_dataset = np.mean(list(L_t_adjusted.values()))
#     return L_dataset, L_t_adjusted


# # ---------- (2) 相关性 ----------
# def compute_correlations(x, y):
#     srocc, _ = spearmanr(x, y)
#     krocc, _ = kendalltau(x, y)
#     plcc, _ = pearsonr(x, y)
#     return srocc, krocc, plcc

# # ---------- (3) 主程序 ----------
# # ---------- (4) 绘制拟合图 ----------
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_fit_vs_gt_multi(y_trues, y_preds, labels, markers=None, colors=None, title=None, fontsize=24):
#     """
#     y_trues, y_preds: list of np.arrays，每个元素对应一个数据集
#     labels: list of str，数据集名称
#     markers: list of str，点的样式
#     colors: list of str，点的颜色
#     fontsize: int，整体字体大小
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np

#     if markers is None:
#         markers = ['o', 's', 'D', '^', 'v', '*', 'x', '+', 'p', 'H', '8', '<', '>', '|', '_', '.', ','] * 3

#     if colors is None:
#         base_colors = plt.get_cmap('tab10').colors
#         colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

#     # 合并所有数据用于全局归一化
#     all_y_true = np.concatenate(y_trues)
#     all_y_pred = np.concatenate(y_preds)

#     # 全局归一化到 [20, 100] 用于绘图
#     y_min, y_max = all_y_pred.min(), all_y_pred.max()
#     all_y_pred_scaled = 20 + (all_y_pred - y_min) / (y_max - y_min) * (100 - 20)

#     # 三次多项式拟合
#     coefs = np.polyfit(all_y_true, all_y_pred_scaled, 3)
#     poly = np.poly1d(coefs)
#     x_fit = np.linspace(min(all_y_true), max(all_y_true), 200)
#     y_fit = poly(x_fit)

#     plt.figure(figsize=(10,4))
#     start_idx = 0
#     dataset_means = []

#     for i in range(len(y_trues)):
#         n = len(y_trues[i])
#         y_scaled = all_y_pred_scaled[start_idx:start_idx+n]

#         # 散点
#         plt.scatter(y_trues[i], y_scaled,
#                     label=labels[i], marker=markers[i], color=colors[i],
#                     s=70, alpha=0.8, edgecolor='None')

#         # 计算平均值
#         mean_pred = np.mean(y_scaled)
#         mean_gt = np.mean(y_trues[i])
#         dataset_means.append((mean_pred, mean_gt))

#         # 绘制水平虚线（预测平均值）
#         plt.hlines(mean_pred, xmin=min(y_trues[i]), xmax=max(y_trues[i]),
#                    colors=colors[i], linestyles='dashed', linewidth=1.3)
#         # 绘制垂直虚线（GT平均值）
#         plt.vlines(mean_gt, ymin=min(y_scaled), ymax=max(y_scaled),
#                    colors=colors[i], linestyles='dashed', linewidth=1.3)

#         # 标注预测均值，在水平虚线最左端
#         plt.text(min(y_trues[i]), mean_pred, f"{mean_pred:.1f}", color=colors[i],
#                  fontsize=fontsize - 10, verticalalignment='bottom', horizontalalignment='left')

#         # 标注GT均值，在竖线最下端
#         plt.text(mean_gt, min(y_scaled), f"{mean_gt:.1f}", color=colors[i],
#                  fontsize=fontsize - 10, verticalalignment='bottom', horizontalalignment='left')

#         start_idx += n

#     # 拟合曲线
#     plt.plot(x_fit, y_fit, color='red', linestyle='--', linewidth=2, label='Cubic Fit')

#     # 设置字体大小
#     plt.xlabel("Ground Truth", fontsize=fontsize-4)
#     plt.ylabel("Predicted", fontsize=fontsize-4)
#     if title:
#         plt.title(title, fontsize=fontsize + 2)
#     plt.xticks(fontsize=fontsize - 8)
#     plt.yticks(fontsize=fontsize - 8)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(fontsize=fontsize - 4, loc='lower right',bbox_to_anchor=(1.5, 0))

#     # 相关性指标用原始预测值计算
#     srocc, krocc, plcc = compute_correlations(all_y_true, all_y_pred)
#     plt.text(0.05, 0.95, f"SRCC={srocc:.3f}\nKRCC={krocc:.3f}\nPLCC={plcc:.3f}",
#              transform=plt.gca().transAxes, verticalalignment='top',
#              fontsize=fontsize - 9,
#              bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

#     plt.tight_layout()
#     plt.savefig("learning_ease_fit_multi.png", dpi=300)
#     plt.show()

#     return dataset_means  # 返回每个数据集的 (预测均值, GT均值)

