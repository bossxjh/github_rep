# dataeval/models/__init__.py
from .clip_adapter import CLIPAdapter
from .openvla_adapter import OpenVLAAdapter
from .pi0_5_adapter import Pi05Adapter
from .diffusion_policy_adapter import DPResNetAdapter

MODEL_ADAPTERS = {
    "clip": CLIPAdapter,
    "openvla": OpenVLAAdapter,
    "pi0.5": Pi05Adapter,
    "diffusion policy":DPResNetAdapter,
}