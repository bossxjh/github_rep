# dataeval/models/__init__.py
from .clip_adapter import CLIPAdapter
from .openvla_adapter import OpenVLAAdapter
from .pi0_5_adapter import Pi05Adapter

MODEL_ADAPTERS = {
    "clip": CLIPAdapter,
    "openvla": OpenVLAAdapter,
    "pi0.5": Pi05Adapter,
}