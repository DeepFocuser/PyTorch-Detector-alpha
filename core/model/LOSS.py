import torch
import torch.nn.functional as F
from torch.nn import Module


class HeatmapFocalLoss(Module):

    def __init__(self, from_sigmoid=True, alpha=2, beta=4):
        super(HeatmapFocalLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._alpha = alpha
        self._beta = beta

    def forward(self, pred, label):
        if not self._from_sigmoid:
            pred = F.sigmoid(pred)

        # a penalty-reduced pixelwise logistic regression with focal loss
        condition = label == 1
        loss = torch.where(condition=condition,
                           x=torch.pow(1 - pred, self._alpha) * torch.log(pred),
                           y=torch.pow(1 - label, self._beta) * torch.pow(pred, self._alpha) * torch.log(1 - pred))

        loss = -torch.sum(loss, dim=[1,2,3]).mean()
        norm = torch.sum(condition).clamp(1, 1e30)
        return loss / norm


class NormedL1Loss(Module):

    def __init__(self):
        super(NormedL1Loss, self).__init__()

    def forward(self, pred, label, mask):

        # HeatmapFocalLoss 의 condition 은 mask와 같다.
        loss = torch.abs(label * mask - pred * mask)
        loss = torch.sum(loss, dim=[1,2,3]).mean()

        norm = torch.sum(mask).clamp(1, 1e30)
        return loss / norm
