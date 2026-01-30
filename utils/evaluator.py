from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class Evaluator:
    def __init__(self, seed):
        self.seed = seed

    def f1_node_classification(self, y_label, y_pred):
        if len(y_label.shape) > 1 and y_label.shape[1] > 1:
            macro_f1 = f1_score(y_label, y_pred, average="macro", zero_division=0)
            micro_f1 = f1_score(y_label, y_pred, average="micro", zero_division=0)
            samples_f1 = f1_score(y_label, y_pred, average="samples", zero_division=0)
            return dict(Macro_f1=macro_f1, Micro_f1=micro_f1, Samples_f1=samples_f1)
        macro_f1 = f1_score(y_label, y_pred, average="macro")
        micro_f1 = f1_score(y_label, y_pred, average="micro")
        return dict(Macro_f1=macro_f1, Micro_f1=micro_f1)

    def cal_acc(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def nc_with_LR(self, emd, labels, train_idx, test_idx):
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        lr_model = LogisticRegression(max_iter=10000)
        x_train = emd[train_idx]
        x_test = emd[test_idx]
        lr_model.fit(x_train, y_train)
        y_pred = lr_model.predict(x_test)
        f1_dict = self.f1_node_classification(y_test, y_pred)
        return f1_dict["Micro_f1"], f1_dict["Macro_f1"]
