import torch
from torch.nn import Module


class SoftmaxCrossEntropyLoss(Module):

    def __init__(self, axis=-1, sparse_label=True, from_logits=False):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def forward(self, pred, label):

        if not self._from_logits:
            pred = torch.log_softmax(pred, dim=self._axis)
        if self._sparse_label:
            loss = -torch.index_select(pred, self._axis, label)
        else:
            loss = -(pred * label).sum(dim=-1, keepdim=True)

        return loss.mean()
