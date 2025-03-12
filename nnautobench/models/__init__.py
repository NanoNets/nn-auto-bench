from __future__ import annotations

from .claude35 import Claude35
from .claude37 import Claude37
from .dsv3 import DSv3
from .flash2 import Flash2
from .flash2v import Flash2V
from .gpt4o_model import GPT4oModel
from .gpt4v_model import GPT4VModel
from .gpto3mini import GPTo3MiniModel
from .mistral_large import MistralLarge
from .qwen2_model import Qwen2Model

available_models = {
    "qwen2": Qwen2Model,
    "gpt4v": GPT4VModel,
    "gpt4o": GPT4oModel,
    "dsv3": DSv3,
    "flash2v": Flash2V,
    "flash2": Flash2,
    "claude35": Claude35,
    "claude37": Claude37,
    "mistral-large": MistralLarge,
    "gpt-o3-mini": GPTo3MiniModel,
}


def get_model(model_name):
    if model_name not in available_models:
        raise ValueError(
            f"Unknown model: {model_name}. Available models are: {', '.join(available_models.keys())}",
        )
    return available_models[model_name]
