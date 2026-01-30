import importlib
from torch import nn

from .base_model import BaseModel

MODEL_REGISTRY = {}


def register_model(name):
    """
    Register a model class with a string name.
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate models ({})".format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseModel".format(name, cls.__name__)
            )
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls

    return register_model_cls


def try_import_model(model):
    if model not in MODEL_REGISTRY:
        if model in SUPPORTED_MODELS:
            importlib.import_module(SUPPORTED_MODELS[model])
        else:
            print(f"Failed to import {model} models.")
            return False
    return True


def build_model(model):
    if isinstance(model, nn.Module):
        if not hasattr(model, "build_model_from_args"):

            def build_model_from_args(args, hg):
                return model

            model.build_model_from_args = build_model_from_args
        return model
    if not try_import_model(model):
        exit(1)
    return MODEL_REGISTRY[model]


MODULE_PATH = __name__

SUPPORTED_MODELS = {
    'ROSA': f'{MODULE_PATH}.ROSA',
    'HGT_Multi': f'{MODULE_PATH}.HGT_Multi',
    'HAN_Multi': f'{MODULE_PATH}.HAN_Multi',
    'MAGNN_Multi': f'{MODULE_PATH}.MAGNN_Multi',
    'RGAT_Multi': f'{MODULE_PATH}.RGAT_Multi',
    'HAN': f'{MODULE_PATH}.HAN',
    'HGT': f'{MODULE_PATH}.HGT',
}

__all__ = [
    'BaseModel',
    'MODEL_REGISTRY',
    'SUPPORTED_MODELS',
    'register_model',
    'try_import_model',
    'build_model',
]
