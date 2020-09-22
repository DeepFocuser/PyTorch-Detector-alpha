import torch
from torch.nn import Module

class Yolov3Loss(Module):

    def __init__(self, sparse_label = True,
                 from_sigmoid=False,
                 batch_axis=None,
                 num_classes=5,
                 reduction="sum"):

        super(Yolov3Loss, self).__init__()
        self._sparse_label = sparse_label
        self._from_sigmoid = from_sigmoid
        self._batch_axis = batch_axis
        self._num_classes = num_classes
        self._reduction = reduction.upper()
        self._num_pred = 5 + num_classes

        self._sigmoid_ce = SigmoidBinaryCrossEntropyLoss(from_sigmoid=from_sigmoid,
                                                         batch_axis=batch_axis,
                                                         reduction=reduction)
        self._l2loss = L2Loss(batch_axis=batch_axis,
                              reduction=reduction)

    def forward(self, output1, output2, output3, xcyc_target, wh_target, objectness, class_target, weights):

        #1. prediction 쪼개기
        b, _, _, _ = output1.shape
        pred = torch.cat([out.reshape(b, -1, self._num_pred) for out in [output1, output2, output3]], dim=1)

        xcyc_pred = pred[:,:,0:2]
        wh_pred = pred[:,:,2:4]
        objectness_pred = pred[:,:,4:5]
        class_pred = pred[:,:,5:]

        #2. loss 구하기
        object_mask = objectness == 1
        noobject_mask = objectness == 0

        # coordinates loss
        if not self._from_sigmoid:
            xcyc_pred = torch.sigmoid(xcyc_pred)

        xcyc_loss = self._l2loss(xcyc_pred, xcyc_target, object_mask*weights)
        wh_loss =self._l2loss(wh_pred, wh_target, object_mask*weights)

        # object loss + noboject loss
        obj_loss = self._sigmoid_ce(objectness_pred, objectness, object_mask)
        noobj_loss = self._sigmoid_ce(objectness_pred, objectness, noobject_mask)
        object_loss = torch.add(noobj_loss, obj_loss)

        if self._sparse_label:
            class_target = class_target.to(torch.int64)
            class_target = torch.nn.functional.one_hot(class_target+1, self._num_classes + 1)
            class_target = class_target[:, :, 1:]

        # class loss
        class_loss = self._sigmoid_ce(class_pred, class_target, object_mask)

        return xcyc_loss, wh_loss, object_loss, class_loss

class L2Loss(Module):

    def __init__(self, batch_axis=0, reduction="sum"):
        super(L2Loss, self).__init__()

        self._batch_axis = batch_axis
        self._reduction = reduction.upper()

    def forward(self, pred, label, sample_weight=None):
        loss = torch.square(label - pred)
        if sample_weight is not None:
            loss = torch.mul(loss, sample_weight)
        if self._reduction == "SUM":
            return torch.sum(loss, dim=[1,2]).mean()
        elif self._reduction == "MEAN":
            return torch.mean(loss, dim=[1,2]).mean()
        else:
            raise NotImplementedError

class SigmoidBinaryCrossEntropyLoss(Module):

    def __init__(self, from_sigmoid=False, batch_axis=0, reduction="sum"):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._batch_axis = batch_axis
        self._reduction = reduction.upper()

    def forward(self, pred, label, sample_weight=None, pos_weight=None):

        if not self._from_sigmoid:
            if pos_weight is None:
                # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
                loss = torch.nn.functional.relu(pred) - pred * label + \
                       torch.log(1 + torch.exp(-torch.abs(pred)))
            else:
                # We use the stable formula: x - x * z + (1 + z * pos_weight - z) * \
                #    (log(1 + exp(-abs(x))) + max(-x, 0))
                log_weight = 1 + torch.mul(pos_weight - 1, label)
                loss = pred - pred * label + log_weight * \
                       (torch.log(1 + torch.exp(-torch.abs(pred))) + torch.nn.functional.relu(-pred))
        else:
            eps = 1e-7
            if pos_weight is None:
                loss = -(torch.log(pred + eps) * label
                         + torch.log(1. - pred + eps) * (1. - label))
            else:
                loss = -(torch.mul(torch.log(pred + eps) * label, pos_weight)
                         + torch.log(1. - pred + eps) * (1. - label))
        if sample_weight is not None:
            loss = torch.mul(loss, sample_weight)

        if self._reduction == "SUM":
            return torch.sum(loss, dim=[1,2]).mean()
        elif self._reduction == "MEAN":
            return torch.mean(loss, dim=[1,2]).mean()
        else:
            raise NotImplementedError
