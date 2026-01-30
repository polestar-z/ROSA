import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BaseTask, register_task
from ..dataset import build_dataset
from ..utils import Evaluator


@register_task("node_classification")
class NodeClassification(BaseTask):
    r"""
    Node classification task.
    """

    def __init__(self, args):
        super(NodeClassification, self).__init__()
        self.logger = args.logger
        self.dataset = build_dataset(
            args.dataset,
            "node_classification",
            logger=self.logger,
            args=args,
        )

        if hasattr(args, "validation"):
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split(args.validation)
        else:
            self.train_idx, self.val_idx, self.test_idx = self.dataset.get_split()

        self.evaluator = Evaluator(args.seed)
        self.labels = self.dataset.get_labels()
        self.multi_label = self.dataset.multi_label

        self.threshold = getattr(args, "threshold", 0.5)
        self.evaluation_metric = getattr(args, "evaluation_metric", "f1")

    def get_graph(self):
        return self.dataset.g

    def get_loss_fn(self):
        if self.multi_label:
            return nn.BCEWithLogitsLoss()
        return F.cross_entropy

    def get_evaluator(self, name):
        if name == "acc":
            return self.evaluator.cal_acc
        if name == "f1_lr":
            return self.evaluator.nc_with_LR
        if name == "f1":
            return self.evaluator.f1_node_classification
        raise ValueError("The evaluation metric is not supported!")

    def evaluate(self, logits, mode="test", info=True):
        if mode == "test":
            mask = self.test_idx
        elif mode == "valid":
            mask = self.val_idx
        elif mode == "train":
            mask = self.train_idx
        else:
            raise ValueError("Unsupported evaluation mode")

        if self.multi_label:
            probs = torch.sigmoid(logits[mask])
            pred = (probs.cpu().numpy() > self.threshold).astype(int)
        else:
            pred = logits[mask].argmax(dim=1).cpu().numpy()

        if isinstance(self.labels, torch.Tensor):
            labels_np = self.labels[mask].cpu().numpy()
        else:
            labels_np = self.labels[mask]

        if self.evaluation_metric == "acc":
            acc = self.evaluator.cal_acc(labels_np, pred)
            return dict(Accuracy=acc)
        if self.evaluation_metric == "f1":
            f1_dict = self.evaluator.f1_node_classification(labels_np, pred)
            return f1_dict
        raise ValueError("The evaluation metric is not supported!")

    def downstream_evaluate(self, logits, evaluation_metric):
        if evaluation_metric == "f1_lr":
            micro_f1, macro_f1 = self.evaluator.nc_with_LR(
                logits, self.labels, self.train_idx, self.test_idx
            )
            return dict(Macro_f1=macro_f1, Mirco_f1=micro_f1)
        raise ValueError("The evaluation metric is not supported!")

    def get_split(self):
        return self.train_idx, self.val_idx, self.test_idx

    def get_labels(self):
        return self.labels
