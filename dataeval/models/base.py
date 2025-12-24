# dataeval/models/base.py

class ModelAdapter:
    def extract(self, frames):
        raise NotImplementedError