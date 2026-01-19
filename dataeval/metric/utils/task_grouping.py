# dataeval/utils/task_grouping.py
import numpy as np


def group_by_task(npzdata):
    features = npzdata["features"]                  # (N, D)
    task_ids = npzdata["task_ids"].astype(int)      # (N,)
    task_lengths = npzdata["task_lengths"]          # (N,)
    demo_lengths = npzdata["demo_lengths"]          # (N,)
    task_descriptions = npzdata["task_descriptions"]# (N,)


    N, T, D = features.shape
    features = features.reshape(N, T*D)

    task_groups = []
    print("features shape", features.shape)

    for task_id in np.unique(task_ids):
        mask = task_ids == task_id

        task_groups.append({
            "task_id": int(task_id),
            # 取第一个就行，因为所有 demo 的值都是一样的
            "task_length": float(task_lengths[mask][0]),
            "task_description": task_descriptions[mask][0],
            "features": features[mask],          # (Nt, D)
            "demo_lengths": demo_lengths[mask],  # (Nt,)
        })

    return task_groups

