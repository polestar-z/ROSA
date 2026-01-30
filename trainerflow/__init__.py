import importlib
from abc import ABC
from .base_flow import BaseFlow

FLOW_REGISTRY = {}


def register_flow(name):
    """
    Register a trainer flow with a string name.
    """

    def register_flow_cls(cls):
        if name in FLOW_REGISTRY:
            raise ValueError("Cannot register duplicate flow ({})".format(name))
        if not issubclass(cls, (BaseFlow, ABC)):
            raise ValueError("Flow ({}: {}) must extend BaseFlow or ABC".format(name, cls.__name__))
        FLOW_REGISTRY[name] = cls
        return cls

    return register_flow_cls


def try_import_flow(flow):
    if flow not in FLOW_REGISTRY:
        if flow in SUPPORTED_FLOWS:
            importlib.import_module(SUPPORTED_FLOWS[flow])
        else:
            print(f"Failed to import {flow} flows.")
            return False
    return True


def build_flow(args, flow_name):
    if not try_import_flow(flow_name):
        exit(1)
    return FLOW_REGISTRY[flow_name](args)


MODULE_PATH = __name__

SUPPORTED_FLOWS = {
    'node_classification': f'{MODULE_PATH}.node_classification',
    'rosa_node_classification': f'{MODULE_PATH}.rosa_node_classification',
    'aoa_nc_trainer': f'{MODULE_PATH}.aoa_trainer',
    'coatrainer': f'{MODULE_PATH}.coa_trainer',
}

__all__ = [
    'BaseFlow',
    'FLOW_REGISTRY',
    'SUPPORTED_FLOWS',
    'register_flow',
    'try_import_flow',
    'build_flow',
]
