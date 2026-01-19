import numpy as np
from dataeval.metric.leanability import compute_leanability_from_npzdata

def main():
    npz_path = "/Volumes/T7/项目/具身数据评测/github_rep/feature/libero_spatial_clip_nf3_bs16.npz"  # 改成你的 npz 路径

    npzdata = np.load(npz_path, allow_pickle=True)

    result = compute_leanability_from_npzdata(
        npzdata=npzdata,
        beta=0.5,
        sigma=0.001,
    )

    print("\n===== Leanability Result =====")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

