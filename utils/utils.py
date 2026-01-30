import copy
import random

import dgl
import numpy as np
import torch as th

from .best_config import BEST_CONFIGS


def set_best_config(args):
    configs = BEST_CONFIGS.get(args.task)
    if configs is None:
        print("The task: {} do not have a best_config!".format(args.task))
        return args
    if args.model not in configs:
        print("The model: {} is not in the best config.".format(args.model))
        return args
    configs = configs[args.model]
    for key, value in configs.get("general", {}).items():
        args.__setattr__(key, value)
    if args.dataset not in configs:
        print(
            "The dataset: {} is not in the best config of model: {}.".format(
                args.dataset, args.model
            )
        )
        return args
    for key, value in configs[args.dataset].items():
        args.__setattr__(key, value)
    print(
        "Load the best config of model: {} for dataset: {}.".format(
            args.model, args.dataset
        )
    )
    return args


class EarlyStopping(object):
    def __init__(self, patience=10, save_path=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.early_stop = False
        if save_path is None:
            self.best_model = None
        self.save_path = save_path

    def step(self, loss, score, model):
        if isinstance(score, tuple):
            score = score[0]
        if self.best_loss is None:
            self.best_score = score
            self.best_loss = loss
            self.save_model(model)
        elif (loss > self.best_loss) and (score < self.best_score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (score >= self.best_score) and (loss <= self.best_loss):
                self.save_model(model)

            self.best_loss = np.min((loss, self.best_loss))
            self.best_score = np.max((score, self.best_score))
            self.counter = 0
        return self.early_stop

    def step_score(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if score >= self.best_score:
                self.save_model(model)

            self.best_score = np.max((score, self.best_score))
            self.counter = 0
        return self.early_stop

    def loss_step(self, loss, model):
        if isinstance(loss, th.Tensor):
            loss = loss.item()
        if self.best_loss is None:
            self.best_loss = loss
            self.save_model(model)
        elif loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if loss < self.best_loss:
                self.save_model(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_model(self, model):
        if self.save_path is None:
            self.best_model = copy.deepcopy(model)
        else:
            model.eval()
            th.save(model.state_dict(), self.save_path)

    def load_model(self, model):
        if self.save_path is None:
            return self.best_model
        model.load_state_dict(th.load(self.save_path))
        return model


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        emb[ntype] = node_embed[ntype][nid]
    return emb


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    dgl.seed(seed)


def extract_metapaths(category, canonical_etypes, self_loop=False):
    meta_paths_dict = {}
    for etype in canonical_etypes:
        if etype[0] in category:
            for dst_e in canonical_etypes:
                if etype[0] == dst_e[2] and etype[2] == dst_e[0]:
                    if self_loop:
                        mp_name = "mp" + str(len(meta_paths_dict))
                        meta_paths_dict[mp_name] = [etype, dst_e]
                    else:
                        if etype[0] != etype[2]:
                            mp_name = "mp" + str(len(meta_paths_dict))
                            meta_paths_dict[mp_name] = [etype, dst_e]
    return meta_paths_dict


def to_hetero_feat(h, type, name):
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[th.where(type == index)]
    return h_dict


def to_hetero_idx(g, hg, idx):
    input_nodes_dict = {}
    for i in idx:
        if not hg.ntypes[g.ndata["_TYPE"][i]] in input_nodes_dict:
            a = g.ndata["_ID"][i].cpu()
            a = np.expand_dims(a, 0)
            a = th.tensor(a)
            input_nodes_dict[hg.ntypes[g.ndata["_TYPE"][i]]] = a
        else:
            a = input_nodes_dict[hg.ntypes[g.ndata["_TYPE"][i].cpu()]]
            b = g.ndata["_ID"][i].cpu()
            b = np.expand_dims(b, 0)
            b = th.tensor(b)
            input_nodes_dict[hg.ntypes[g.ndata["_TYPE"][i]]] = th.cat((a, b), 0)
    return input_nodes_dict


def to_homo_feature(ntypes, h_dict):
    h = None
    for ntype in ntypes:
        if ntype in h_dict:
            if h is None:
                h = h_dict[ntype]
            else:
                h = th.cat((h, h_dict[ntype]), dim=0)
    return h


def to_homo_idx(ntypes, num_nodes_dict, idx_dict):
    idx = None
    start_idx = [0]
    for i, num_nodes in enumerate([num_nodes_dict[ntype] for ntype in ntypes]):
        if i < len(ntypes) - 1:
            start_idx.append(num_nodes + start_idx[i])
    for i, ntype in enumerate(ntypes):
        if ntype in idx_dict and th.is_tensor(idx_dict[ntype]):
            if idx is None:
                idx = th.add(idx_dict[ntype], start_idx[i])
            else:
                idx = th.cat((idx, th.add(idx_dict[ntype], start_idx[i])), dim=0)
    return idx


def get_ntypes_from_canonical_etypes(canonical_etypes=None):
    ntypes = set()
    for etype in canonical_etypes:
        src = etype[0]
        dst = etype[2]
        ntypes.add(src)
        ntypes.add(dst)
    return ntypes
