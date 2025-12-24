# dataeval/api.py
from dataeval.models import MODEL_ADAPTERS
from dataeval.datasets import DATASET_PARSERS

def extract_features(model_name, dataset_name, dataset_path, num_frames=3):
    if dataset_name not in DATASET_PARSERS:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")
    if model_name not in MODEL_ADAPTERS:
        raise ValueError(f"Model '{model_name}' not supported.")

    parser = DATASET_PARSERS[dataset_name]
    adapter = MODEL_ADAPTERS[model_name]()  # ← 只 load 一次

    for frames in parser(dataset_path, num_frames=num_frames):
        yield adapter.extract(frames)