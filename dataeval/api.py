# dataeval/api.py
from dataeval.models import MODEL_ADAPTERS
from dataeval.datasets import DATASET_PARSERS
from dataeval.datasets_meta import DATASET_PARSERS_META

def extract_features_with_metadata(model_name, dataset_name, dataset_path, num_frames=3, batch_size=16):
    """
    提取整个数据集的 demo 级特征，保留 metadata。
    
    返回：
        demo_features: list of dict
        每个 dict 包含:
            - 'features': 模型提取的特征
            - 'task_id'
            - 'task_length'
            - 'demo_length'
            - 'task_description'
    """
    parser = DATASET_PARSERS_META[dataset_name]
    adapter = MODEL_ADAPTERS[model_name]()
    
    demo_features_list = parser(dataset_path, num_frames=num_frames)
    
    all_features = []
    batch_frames = []
    batch_meta = []

    for demo in demo_features_list:
        batch_frames.append(demo["frames"])
        batch_meta.append({
            "task_id": demo["task_id"],
            "task_length": demo["task_length"],
            "demo_length": demo["demo_length"],
            "task_description": demo["task_description"]
        })

        if len(batch_frames) == batch_size:
            feats = adapter.extract_batch(batch_frames)  # 输出形状 [B, F]
            for f, meta in zip(feats, batch_meta):
                all_features.append({**meta, "features": f})
            batch_frames.clear()
            batch_meta.clear()

    # 处理剩余 batch
    if batch_frames:
        feats = adapter.extract_batch(batch_frames)
        for f, meta in zip(feats, batch_meta):
            all_features.append({**meta, "features": f})

    return all_features





def extract_features(model_name, dataset_name, dataset_path,
                     num_frames=3, batch_size=16):

    parser = DATASET_PARSERS[dataset_name]
    adapter = MODEL_ADAPTERS[model_name]()

    batch = []
    for frames in parser(dataset_path, num_frames=num_frames):
        batch.append(frames)

        if len(batch) == batch_size:
            feats = adapter.extract_batch(batch)
            for f in feats:
                yield f
            batch.clear()

    if batch:
        feats = adapter.extract_batch(batch)
        for f in feats:
            yield f

# def extract_features(model_name, dataset_name, dataset_path, num_frames=3):
#     if dataset_name not in DATASET_PARSERS:
#         raise ValueError(f"Dataset '{dataset_name}' not supported.")
#     if model_name not in MODEL_ADAPTERS:
#         raise ValueError(f"Model '{model_name}' not supported.")

#     parser = DATASET_PARSERS[dataset_name]
#     adapter = MODEL_ADAPTERS[model_name]()  # ← 只 load 一次

#     for frames in parser(dataset_path, num_frames=num_frames):
#         yield adapter.extract(frames)
