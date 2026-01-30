import dgl
import torch as th



class Budget(object):
    def __init__(self, hg, n_types, NS):
        self.n_types = {}
        for key, value in n_types.items():
            self.n_types[key] = th.zeros(value)
        self.NS = NS
        self.hg = hg
    def update(self, dst_type, idxs):
        for etype in self.hg.canonical_etypes:
            if dst_type == etype[2]:
                src_type = etype[0]
                                                              
                for i in idxs:
                    src_idx = self.hg.predecessors(i, etype=etype)
                                             
                    len = src_idx.shape[0]
                    if src_type in self.NS.keys():
                        src_idx = th.tensor([i for i in src_idx if i not in self.NS[src_type]])
                    if src_idx.shape[0] > 0:
                        self.n_types[src_type][src_idx] += 1 / len

    def pop(self, type, idx):
        self.n_types[type][idx] = 0


class COAsampler(object):
    def __init__(self, hg, category, num_nodes_per_type, num_steps):
        self.n_types = {}
        for n in hg.ntypes:
            self.n_types[n] = hg.num_nodes(n)
        self.category = category
        self.num_nodes_per_type = num_nodes_per_type
        self.num_steps = num_steps
        self.hg = hg

    def sampler_subgraph(self, seed_nodes):
        OS = {self.category: th.stack(seed_nodes)}
        NS = OS
        B = Budget(self.hg, self.n_types, NS)
        for type, idxs in OS.items():
            B.update(type, idxs)
        for i in range(self.num_steps):
            prob = {}
            for src_type, p in B.n_types.items():
                                
                if p.max() > 0:
                    prob[src_type] = p / th.sum(p)
                    sampled_idx = th.multinomial(prob[src_type], self.num_nodes_per_type, replacement=False)
                    if not OS.__contains__(src_type):
                        OS[src_type] = sampled_idx
                    else:
                        OS[src_type] = th.cat((OS[src_type], sampled_idx))
                    B.update(src_type, sampled_idx)
                    B.pop(src_type, sampled_idx)
        sg = self.hg.subgraph(OS)
        return sg, OS
