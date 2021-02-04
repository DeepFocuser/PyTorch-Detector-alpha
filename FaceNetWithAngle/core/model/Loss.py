import torch
from torch.nn import CosineSimilarity
from torch.nn import Module


class TripletLoss(Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cossimloss = CosineSimilarity()

    def forward(self, anchor, positive, negative):

        ap_loss = self.cossimloss(anchor, positive)
        an_loss = self.cossimloss(anchor, negative)
        loss = torch.clamp(torch.acos(ap_loss) - torch.acos(an_loss) + self.margin, min=0.0)
        
        #loss = torch.clamp(ap_loss - an_loss + torch.cos(self.margin), min=0.0)
        return torch.mean(loss)