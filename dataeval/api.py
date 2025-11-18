from dataeval.extractors import EXTRACTOR_MAP
from dataeval.models import load_model
from dataeval.datasets import DATASET_PARSERS

def extract_features(model_name, dataset_name, dataset_path, num_frames=3):
    if dataset_name not in DATASET_PARSERS:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    parser = DATASET_PARSERS[dataset_name]
    model, processor, device = load_model(model_name)
    extractor = EXTRACTOR_MAP[model_name]

    for frames in parser(dataset_path, num_frames=num_frames):
        feat = extractor(frames, model, processor, device=device)
        yield feat
