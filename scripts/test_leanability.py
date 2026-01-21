import numpy as np
from dataeval.metric.leanability import compute_leanability_from_npzdata

import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau

def compute_correlation(result, benchmark_name):
    """
    计算 SRCC, PLCC, KRCC
    :param result: 包含 leanability_per_task 的字典
    :param benchmark_name: 使用的 benchmark 名字，决定从 gts 中选择哪个目标（例如 'goal', 'object' 等）
    :return: 返回 SRCC, PLCC, KRCC
    """
    # 给定的 gt 数据
    gts = {
        "goal":    np.array([40,52,90,82,96,96,78,40,88,94]),#75.6
        "object":  np.array([88,70,90,84,72,86,88,94,90,86]),#84.8
        "spatial": np.array([94,96,76,96,82,84,88,90,80,72]),#85.8
        "ten":     np.array([52,34,48,20,62,58,66,44,38,68]),#49.0
        "ten-object":     np.array([54, 50, 24, 14, 48, 48, 74, 36, 30, 64,84, 66, 94, 88, 76, 76, 92, 88, 76, 92]),#44.2+83.2#都降低5/1
        "Goal-object":     np.array([64, 48, 88, 86, 94, 94, 74, 50, 70, 96,90, 80, 90, 86, 82, 92, 96, 92, 92, 92]),#76.4+89.2#都升高1/5
        "spatial-10":     np.array([94, 92, 82, 92, 84, 90, 90,100, 72, 76,50, 42, 38, 32, 54, 44, 66, 44, 60, 72]),#87.2+50.2#都升高2/1
        "spatial-goal":     np.array([90, 90, 72, 96, 84, 86, 86, 94, 84, 66,60, 44, 88, 90, 90, 94, 72, 42, 90, 94]),#84.8+76.4#一个降低一个升高1/1
        "Spatial-object":     np.array([94, 76, 68, 94, 80, 86, 94, 72, 88, 68,88, 72, 96, 88, 78, 84, 92, 96, 94, 92]),#82.0+88.0#一个降低一个升高3/4
        "goal+ten":     np.array([58, 52, 82, 84, 94, 98, 64, 38, 92, 98,52, 52, 58, 28, 70, 54, 78, 44, 56, 64]),#76.0+55.6#都升高0.4/6
    }

    # 提取 result 中的 leanability_per_task 和 task_ids
    leanability_per_task = result['leanability_per_task']
    task_ids = list(leanability_per_task.keys())
    leanability_values = [leanability_per_task[task_id] for task_id in task_ids]

    # 获取对应的 GT 数据
    gt_values = gts.get(benchmark_name)
    if gt_values is None:
        raise ValueError(f"Benchmark '{benchmark_name}' not found in GT data.")

    # 计算 SRCC, PLCC 和 KRCC
    srcc, _ = spearmanr(leanability_values, gt_values)
    plcc, _ = pearsonr(leanability_values, gt_values)
    krcc, _ = kendalltau(leanability_values, gt_values)

    # 返回结果
    return srcc, plcc, krcc
def main():
    npz_path = "/mnt/shared-storage-user/xiaojiahao/trans/xiaojiahao/project/github_rep/feature/libero-spatial_openvla_nf3_bs32.npz"  # 改成你的 npz 路径
    benchmark_name = "spatial"  # 你可以替换为 'object', 'spatial' 等

    npzdata = np.load(npz_path, allow_pickle=True)

    result = compute_leanability_from_npzdata(
        npzdata=npzdata,
        beta=0.5,
        sigma_task=0.001,               # 任务内样本相似度带宽（None 则用 0.001 保底）
        sigma_center=0.001, 
        pi_scale=0.01698373,
        # pi_scale=0.0000001,
    )

    print("\n===== Leanability Result =====")
    for k, v in result.items():
        print(f"{k}: {v}")

    srcc, plcc, krcc = compute_correlation(result, benchmark_name)
    print(f"SRCC: {srcc}")
    print(f"PLCC: {plcc}")
    print(f"KRCC: {krcc}")

if __name__ == "__main__":
    main()

