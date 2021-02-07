import torch
from torch.nn import Module


class TripletLoss(Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self._margin = margin

    def forward(self, anchor, positive, negative):

        ap_loss = torch.sum(torch.mul(anchor, positive), dim=1)
        an_loss = torch.sum(torch.mul(anchor, negative), dim=1)
        loss = torch.clamp(torch.acos(ap_loss) - torch.acos(an_loss) + self._margin, min=0.0)
        return torch.mean(loss)