import numpy as np

# ---------- 路径 ----------
npz1_path = "/Volumes/T7/项目/具身数据评测/github_rep/feature/libero_clip_nf3_bs1.npz"
npz2_path = "/Volumes/T7/项目/具身数据评测/github_rep/feature/libero_spatial_clip_nf3_bs16.npz"

#  ---------- 加载 npz1 ----------
# data1 = np.load(npz1_path, allow_pickle=True)
# X1, y1, task_lengths1 = data1["X"], data1["y"], data1["task_lengths"].item()



# ---------- 加载 npz2 ----------
data1 = np.load(npz1_path, allow_pickle=True)
X1 = data1["features"]
y1 = data1["task_ids"].astype(int)

task_lengths1= data1["task_lengths"]
if isinstance(task_lengths1, np.ndarray):
    task_lengths1 = task_lengths1[()]  # 取 object 内容，如果是 dict 形式

X1 = X1.reshape(X1.shape[0], -1)

# ---------- 加载 npz2 ----------
# ---------- 加载 npz2 ----------
data2 = np.load(npz2_path, allow_pickle=True)
X2 = data2["features"]
y2 = data2["task_ids"].astype(int)

task_lengths2 = data2["task_lengths"]
if isinstance(task_lengths2, np.ndarray):
    task_lengths2 = task_lengths2[()]  # 取 object 内容，如果是 dict 形式

X2 = X2.reshape(X2.shape[0], -1)

# ---------- 基本信息 ----------
print("X1 shape:", X1.shape, "X2 shape:", X2.shape)
print("y1 shape:", y1.shape, "y2 shape:", y2.shape)
print("task count npz1:", len(np.unique(y1)), "task count npz2:", len(np.unique(y2)))

# ---------- 检查整体特征是否一致 ----------
print("整体特征完全相等:", np.allclose(X1, X2))
print("整体标签完全相等:", np.all(y1 == y2))

# ---------- 每列平均差异 ----------
col_diff = np.mean(np.abs(X1 - X2), axis=0)
print("每列平均差异:", col_diff)

# ---------- 每行最大差异 ----------
row_diff = np.max(np.abs(X1 - X2), axis=1)
print("每行最大差异: min", row_diff.min(), "max", row_diff.max(), "mean", row_diff.mean())

# ---------- 每个任务中心差异 ----------
task_ids = np.unique(y1)
print("\n每个任务中心向量最大差异:")
for t in task_ids:
    center1 = X1[y1==t].mean(axis=0)
    center2 = X2[y2==t].mean(axis=0)
    print(f"task {t}: max center diff = {np.max(np.abs(center1 - center2)):.6e}, mean diff = {np.mean(np.abs(center1 - center2)):.6e}")

# ---------- 每个任务样本数 ----------
print("\n每个任务样本数差异:")
for t in task_ids:
    n1 = np.sum(y1==t)
    n2 = np.sum(y2==t)
    print(f"task {t}: npz1={n1}, npz2={n2}, diff={n1-n2}")