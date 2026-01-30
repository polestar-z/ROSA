import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        pt0 = xs_pos * y + self.eps
        pt1 = xs_neg * (1 - y) + self.eps

        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                asymmetric_w = (1 - pt0) ** self.gamma_pos
                asymmetric_w_neg = pt1 ** self.gamma_neg
            loss = -los_pos * asymmetric_w - los_neg * asymmetric_w_neg
        else:
            loss = -los_pos * ((1 - pt0) ** self.gamma_pos) - los_neg * (pt1 ** self.gamma_neg)

        return loss.mean()

    def __repr__(self):
        return (
            f"AsymmetricLoss(gamma_neg={self.gamma_neg}, gamma_pos={self.gamma_pos}, "
            f"clip={self.clip}, eps={self.eps}, disable_torch_grad={self.disable_torch_grad_focal_loss})"
        )
