"""Original focal and Dice segmentation losses."""
import torch
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.ignore_index = ignore_index
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets,
                             weight=self.weight.to(logits.device) if self.weight is not None else None,
                             ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce)
        fl = (1-pt)**self.gamma * ce
        return fl[targets != self.ignore_index].mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth; self.ignore_index = ignore_index
    def forward(self, logits, targets):
        B,C,H,W = logits.shape
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index)
        tgt   = targets.clone(); tgt[~valid] = 0
        oh    = F.one_hot(tgt, C).permute(0,3,1,2).float() * valid.unsqueeze(1)
        p     = probs * valid.unsqueeze(1)
        inter = (p * oh).sum(dim=(0,2,3))
        union = p.sum(dim=(0,2,3)) + oh.sum(dim=(0,2,3))
        return 1 - ((2*inter + self.smooth) / (union + self.smooth)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, gamma=2.0, weights=None, ignore_index=255):
        super().__init__()
        self.focal = FocalLoss(gamma, weights, ignore_index)
        self.dice  = DiceLoss(ignore_index=ignore_index)
    def forward(self, logits, targets):
        return 0.5*self.focal(logits,targets) + 0.5*self.dice(logits,targets)

