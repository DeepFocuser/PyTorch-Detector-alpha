import torch
from torch.nn import Module

class SoftmaxCrossEntropyLoss(Module):

    def __init__(self, axis=1, from_logits=False):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self._axis = axis
        self._from_logits = from_logits

    def forward(self, pred, label):

        if not self._from_logits:
            pred = torch.log_softmax(pred, dim=self._axis)

        loss = -(pred * label).sum(dim=1)
        return loss.mean()
