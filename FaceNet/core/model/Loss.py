import torch
from torch.nn import Module
from torch.nn.modules.distance import PairwiseDistance

class TripletLoss(Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.PDLoss = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):

        ap_loss = self.PDLoss(anchor, positive)
        an_loss = self.PDLoss(anchor, negative)
        loss = torch.clamp(ap_loss - an_loss + self.margin, min=0.0)
        return torch.mean(loss)