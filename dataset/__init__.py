import importlib
from dgl.data import DGLDataset

from .base_dataset import BaseDataset

DATASET_REGISTRY = {}
CLASS_DATASETS = {}
SUPPORTED_DATASETS = {
    "node_classification": "openhgnn.dataset.NodeClassificationDataset",
}


def register_dataset(name):
    """
    New dataset types can be added with the :func:`register_dataset` decorator.
    """

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        if not issubclass(cls, BaseDataset):
            raise ValueError(
                "Dataset ({}: {}) must extend BaseDataset".format(name, cls.__name__)
            )
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


def try_import_task_dataset(task):
    if task not in DATASET_REGISTRY:
        if task in SUPPORTED_DATASETS:
            importlib.import_module(SUPPORTED_DATASETS[task])
        else:
            print(f"Failed to import {task} dataset.")
            return False
    return True


def build_dataset_GB(dataset, *args, **kwargs):
    return None


def build_dataset_v2(dataset, task):
    raise NotImplementedError("build_dataset_v2 is not available in this trimmed dataset module.")


def build_dataset(dataset, task, *args, **kwargs):
    args = kwargs.get("args", None)

    if isinstance(dataset, DGLDataset):
        return dataset

    dataset_name = dataset
    if dataset_name == "your_dataset_filtered":
        dataset_name = "my_custom_node_classification_filtered"

    if not try_import_task_dataset(task):
        raise ValueError(f"Unsupported task dataset: {task}")
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return DATASET_REGISTRY[dataset_name](logger=kwargs.get("logger"), args=args)


from .NodeClassificationDataset import (              
    NodeClassificationDataset,
    IMDBNodeClassification,
    PersonaNodeClassification,
    ChemistryDatasetFiltered,
)

__all__ = [
    "BaseDataset",
    "register_dataset",
    "build_dataset",
    "build_dataset_GB",
    "build_dataset_v2",
    "try_import_task_dataset",
    "DATASET_REGISTRY",
    "CLASS_DATASETS",
    "NodeClassificationDataset",
    "IMDBNodeClassification",
    "PersonaNodeClassification",
    "ChemistryDatasetFiltered",
]

classes = __all__
