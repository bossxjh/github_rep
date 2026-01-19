# dataeval/metrics/leanability.py

import numpy as np
from .utils.task_grouping import group_by_task
from .learning_ease import compute_learning_ease_with_task_transfer


def compute_leanability_from_npzdata(
    npzdata,
    beta: float = 0.8,
    sigma=None,
):
    # -------- load & group --------
    task_groups = group_by_task(npzdata)

    # -------- DEBUG: inspect task_groups --------
    print(f"num_tasks = {len(task_groups)}")
    for i, g in enumerate(task_groups[:3]):
        print(f"\n[Task {i}]")
        print("  task_id:", g["task_id"])
        print("  task_length:", g["task_length"])
        print("  task_description:", g["task_description"])
        print("  features shape:", g["features"].shape)
        print("  demo_lengths shape:", g["demo_lengths"].shape)

    input("Press Enter to continue...")

    # -------- core leanability computation --------
    dataset_score, task_scores = compute_learning_ease_with_task_transfer(
        task_groups=task_groups,
        beta=beta,
        sigma=sigma,
    )

    return {
        "leanability_dataset": float(dataset_score),
        "leanability_per_task": {
            int(k): float(v) for k, v in task_scores.items()
        },
        "num_tasks": len(task_groups),
        "beta": beta,
        "sigma": sigma,
    }



# def compute_leanability_from_npzdata(
#     npzdata,
#     beta: float = 0.8,
#     sigma=None,
# ):
#     """
#     Compute dataset leanability from loaded npz data.

#     Args:
#         npzdata: result of np.load(..., allow_pickle=True)
#                  or a dict-like object with required fields
#         beta: trade-off coefficient
#         sigma: kernel bandwidth
#     """

#     # -------- basic fields --------
#     X = npzdata["features"]              # (N, D)
#     y = npzdata["task_ids"].astype(int)  # (N,)
#     task_lengths_demo = npzdata["task_lengths"]
#     dataset_name = str(npzdata["dataset_name"])

#     # -------- aggregate task-level lengths --------
#     task_length_dict = {}
#     for t in np.unique(y):
#         task_length_dict[int(t)] = float(
#             task_lengths_demo[y == t].mean()
#         )

#     # -------- call original leanability core --------
#     L_dataset, L_task = compute_learning_ease_with_task_transfer(
#         X=X,
#         y=y,
#         task_lengths=task_length_dict,
#         dataset_name=dataset_name,
#         beta=beta,
#         sigma=sigma,
#     )

#     # -------- package outputs --------
#     result = {
#         "dataset_score": float(L_dataset),
#         "task_scores": {int(k): float(v) for k, v in L_task.items()},
#         "beta": beta,
#         "sigma": sigma,
#         "num_demos": int(X.shape[0]),
#         "feature_dim": int(X.shape[1]),
#         "dataset_name": dataset_name,
#     }

#     return result
