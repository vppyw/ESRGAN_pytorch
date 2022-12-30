import torch
import torch.nn as nn
import torch.nn.functional as F

def AffinityMatrix(fm, mode="spatial"):
    if fm.dim() == 4:
        fm = fm.view(*fm.shape[:-2], -1)

    if mode == "spatial":
        fm_n = fm.div(fm.norm(p=2,
                              dim=-1,
                              keepdim=True).expand_as(fm) \
                      + 1e-12)
        return fm_n.transpose(-2, -1).bmm(fm_n)
    elif mode == "channel":
        fm_n = fm.div(fm.norm(p=2,
                              dim=-2,
                              keepdim=True).expand_as(fm) \
                      + 1e-12)
        return fm_n.transpose(-2, -1).bmm(fm_n)
    elif mode == "instance":
        fm = fm.view(fm.size(0), -1)
        Q = fm.mm(fm.transpose(0, 1))
        return Q.div(Q.norm(p=2, dim=1, keepdim=True).expand_as(Q) \
                     + 1e-12)
    else:
        raise NotImplementedError

class FALoss(nn.Module):
    def __init__(self, mode="spatial", reduction: str ="mean"):
        super().__init__()
        self.mode = mode
        self.reduction = reduction

    def forward(self, input_features, target_features):
        """
        input: b * C * W * H
        target: b * C * W * H
        """
        if input_features.dim() != 3 and input_features.dim() != 4:
            print("input dimention should be 3D(b * C * L) or 4D(b * C * W * H)")
            raise RuntimeError
        if target_features.dim() != 3 and target_features.dim() != 4:
            print("target dimention should be 3D(b * C * L) or 4D(b * C * W * H)")
            raise RuntimeError

        input_fa = AffinityMatrix(input_features)
        target_fa = AffinityMatrix(target_features)
        return F.mse_loss(input_fa,
                          target_fa,
                          reduction=self.reduction)

class ATLoss(nn.Module):
    def __init__(self, mode="sum", p:int=1):
        super().__init__()
        self.mode = mode
        self.p = p
    
    def forward(self, input_features, target_features):
        if self.mode == "sum":
            input_atmap = input_features\
                            .pow(self.p)\
                            .sum(dim=1)\
                            .reshape(input_features.size(0), -1)
            target_atmap = target_features\
                            .pow(self.p)\
                            .sum(dim=1)\
                            .reshape(target_features.size(0), -1)
        elif self.mode == "max":
            input_atmap, _ = input_features\
                                .max()\
                                .reshape(input_features.size(0), -1)
            target_atmap, _ = target_features\
                                .max()\
                                .reshape(target_features.size(0), -1)
        else:
            raise NotImplementedError
        input_atmap = input_atmap.norm(p=2, dim=-1, keepdim=True)
        target_atmap = target_atmap.norm(p=2, dim=-1, keepdim=True)
        return F.pairwise_distance(input_atmap,
                                   target_atmap,
                                   p=self.p,
                                   eps=1e-12).mean()
