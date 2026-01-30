import torch as th


def get_binary_mask(total_size, indices):
    mask = th.zeros(total_size)
    mask[indices] = 1
    return mask.to(th.bool)
