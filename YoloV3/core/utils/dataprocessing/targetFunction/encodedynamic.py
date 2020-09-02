import numpy as np
import torch
from torch.nn import Module

from core.utils.dataprocessing.targetFunction.matching import Matcher


class BBoxBatchIOU(Module):

    def __init__(self, axis=-1):
        super(BBoxBatchIOU, self).__init__()
        self._axis=axis

    def forward(self, a, b):
        """Compute IOU for each batch
        Parameters
        ----------
        a : torch.Tensor
            (B, N, 4) first input.
        b : torch.Tensor
            (B, M, 4) second input.
        Returns
        -------
        torch.Tensor
            (B, N, M) array of IOUs.
        """
        al, at, ar, ab = torch.split(a, 1, dim=self._axis)
        bl, bt, br, bb = torch.split(b, 1, dim=self._axis)

        al = al.squeeze(-1)
        at = at.squeeze(-1)
        ar = ar.squeeze(-1)
        ab = ab.squeeze(-1)

        bl = bl.squeeze(-1)
        bt = bt.squeeze(-1)
        br = br.squeeze(-1)
        bb = bb.squeeze(-1)

        # (B, N, M)
        left = torch.max(al.unsqueeze(-1), bl.unsqueeze(-2))
        right = torch.min(ar.unsqueeze(-1), br.unsqueeze(-2))
        top = torch.max(at.unsqueeze(-1), bt.unsqueeze(-2))
        bot = torch.min(ab.unsqueeze(-1), bb.unsqueeze(-2))

        # clip with (0, float16.max)
        iw = torch.clamp(right - left, min=0, max=6.55040e+04)
        ih = torch.clamp(bot - top, min=0, max=6.55040e+04)
        i = iw * ih

        # areas
        area_a = ((ar - al) * (ab - at)).unsqueeze(-1)
        area_b = ((br - bl) * (bb - bt)).unsqueeze(-2)
        union = torch.add(area_a, area_b) - i
        return torch.true_divide(i, union)

class BBoxCornerToCenter(Module):
    def __init__(self, axis=-1):
        super(BBoxCornerToCenter, self).__init__()
        self._axis = axis

    def forward(self, x):
        xmin, ymin, xmax, ymax = torch.split(x, 1, dim=self._axis)
        width = xmax - xmin
        height = ymax - ymin
        x_center = torch.true_divide(xmin + width, 2)
        y_center = torch.true_divide(ymin + height, 2)
        return x_center, y_center, width, height

class Encoderdynamic(Module):

    def __init__(self, ignore_threshold=0.7, from_sigmoid=False):
        super(Encoderdynamic, self).__init__()
        self._cornertocenter = BBoxCornerToCenter(axis=-1)
        self._batch_iou = BBoxBatchIOU(axis=-1)
        self._from_sigmoid = from_sigmoid
        self._ignore_threshold = ignore_threshold

    def forward(self, matches, ious, outputs, anchors, gt_boxes, gt_ids, input_size):

        in_height = input_size[0]
        in_width = input_size[1]
        feature_size = []
        anchor_size = []
        strides = []

        for i, out, anchor in zip(range(len(outputs)), outputs, anchors):
            _, h, w, ac = out.shape
            _, _, a, _ = anchor.shape
            feature_size.append([h, w])
            anchor_size.append(a)
            stride = np.reshape([in_width // w, in_height // h], (1, 1, 1, 2))
            strides.append(stride)

        self._num_pred = np.divide(ac, a).astype(int)
        all_anchors = torch.cat([anchor.reshape(-1, 2) for anchor in anchors], dim=0)
        num_anchors = np.cumsum(anchor_size)  # ex) (3, 6, 9)
        num_offsets = np.cumsum([np.prod(feature) for feature in feature_size])  # ex) (338, 1690, 3549)
        offsets = [0] + num_offsets.tolist()

        # target 공간 만들어 놓기
        xcyc_targets = torch.zeros(gt_boxes.shape[0],
                                   num_offsets[-1],
                                   num_anchors[-1], 2,
                                   device=gt_boxes.device,
                                   dtype=gt_boxes.dtype)  # (batch, 3549, 9, 2)가 기본 요소
        wh_targets = torch.zeros_like(xcyc_targets)
        weights = torch.zeros_like(xcyc_targets)
        objectness = torch.zeros_like(xcyc_targets.split(1, dim=-1)[0])
        class_targets = torch.zeros_like(objectness)

        all_gtx, all_gty, all_gtw, all_gth = self._cornertocenter(gt_boxes)

        np_gtx, np_gty, np_gtw, np_gth = [x.cpu().numpy().copy() for x in [all_gtx, all_gty, all_gtw, all_gth]]
        np_anchors = all_anchors.cpu().numpy().copy().astype(float)
        np_gt_ids = gt_ids.cpu().numpy().copy().astype(int)

        # 가장 큰것에 할당하고, target network prediction 비교해서 0.7 이상인것들 무시하기
        batch, anchorN, objectN = ious.shape
        for b in range(batch):
            for a in range(anchorN):
                for o in range(objectN):
                    nlayer = np.where(num_anchors > a)[0][0]
                    out_height = outputs[nlayer].shape[1]
                    out_width = outputs[nlayer].shape[2]

                    gtx, gty, gtw, gth = (np_gtx[b, o, 0], np_gty[b, o, 0],
                                          np_gtw[b, o, 0], np_gth[b, o, 0])
                    ''' 
                        matching 단계에서 image만 들어온 데이터들도 matching이 되기때문에 아래와 같이 걸러줘야 한다.
                        image만 들어온 데이터 or padding 된것들은 noobject이다. 
                    '''
                    if gtx == -1.0 and gty == -1.0 and gtw == 0.0 and gth == 0.0:
                        continue
                    # compute the location of the gt centers
                    loc_x = int(gtx / in_width * out_width)
                    loc_y = int(gty / in_height * out_height)
                    # write back to targets
                    index = offsets[nlayer] + loc_y * out_width + loc_x
                    if a == matches[b, o]:  # 최대인 값은 제외
                        xcyc_targets[b, index, a, 0] = gtx / in_width * out_width - loc_x
                        xcyc_targets[b, index, a, 1] = gty / in_height * out_height - loc_y
                        '''
                        if gtx == -1.0 and gty == -1.0 and gtw == 0.0 and gth == 0.0: 
                            continue
                        에서 처리를 해주었으나, 그래도 한번 더 대비 해놓자.
                        '''
                        wh_targets[b, index, a, 0] = np.log(
                            max(gtw, 1) / np_anchors[a, 0])  # max(gtw,1)? gtw,gth가 0일경우가 있다.
                        wh_targets[b, index, a, 1] = np.log(max(gth, 1) / np_anchors[a, 1])
                        weights[b, index, a, :] = 2.0 - gtw * gth / in_width / in_height
                        objectness[b, index, a, 0] = 1
                        class_targets[b, index, a, 0] = np_gt_ids[b, o, 0]

        xcyc_targets = self._slice(xcyc_targets, num_anchors, num_offsets)
        wh_targets = self._slice(wh_targets, num_anchors, num_offsets)
        weights = self._slice(weights, num_anchors, num_offsets)
        objectness = self._slice(objectness, num_anchors, num_offsets)
        class_targets = self._slice(class_targets, num_anchors, num_offsets)
        class_targets = class_targets.squeeze(-1)

        # dynamic
        box_preds = []
        for out, an, st in zip(outputs, anchors, strides):
            box_preds.append(self._boxdecoder(out, an, st))
        box_preds = torch.cat(box_preds, dim=1)
        batch_ious = self._batch_iou(box_preds, gt_boxes)  # (b, N, M)
        ious_max, _ = batch_ious.max(dim=-1, keepdim=True)  # (b, N, 1)
        objectness_dynamic = (ious_max > self._ignore_threshold) * -1.0  # ignore

        # objectness 와 objectness_dynamic 조합하기
        objectness = torch.where(objectness > 0, objectness, objectness_dynamic)

        # # threshold 바꿔가며 개수 세어보기
        # print((objectness == 1).sum().item())
        # print((objectness == 0).sum().item())
        # print((objectness == -1).sum().item())

        return xcyc_targets, wh_targets, objectness, class_targets, weights

    def _slice(self, x, num_anchors, num_offsets):

        anchors = [0] + num_anchors.tolist()
        offsets = [0] + num_offsets.tolist()
        ret = []
        for i in range(len(num_anchors)):
            y = x[:, offsets[i]:offsets[i + 1], anchors[i]:anchors[i + 1], :]
            b, f, a, _ = y.shape
            ret.append(y.reshape(b, f*a, -1))
        return torch.cat(ret, dim=1)

    def _boxdecoder(self, output, anchor, stride):

        batch, height, width, _ = output.shape
        stride = torch.as_tensor(stride, device=output.device)
        # grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        grid_y, grid_x = np.mgrid[:height, :width]
        offset = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)  # (13,13,2)
        offset = np.reshape(offset, (1, -1, 1, 2))  # (1, 169, 1, 2)
        offset = torch.as_tensor(offset, device=output.device)

        # 자르기
        output = output.reshape((batch, height*width, -1, self._num_pred))  # (b, 169, 3, 10)
        xy_pred = output[:, :, : ,0:2] # (b, 169, 3, 2)
        wh_pred = output[:, :, : ,2:4] # (b, 169, 3, 2)
        if not self._from_sigmoid:
            xy_pred = torch.sigmoid(xy_pred)
        xy_preds = torch.mul(torch.add(xy_pred, offset), stride)
        wh_preds = torch.mul(torch.exp(wh_pred), anchor)
        # center to corner
        wh = torch.true_divide(wh_preds, 2.0)
        bbox = torch.cat([xy_preds - wh, xy_preds + wh], dim=-1)  # (b, 169, 3, 4)
        return bbox.reshape(batch, -1, 4)


# test
if __name__ == "__main__":
    from core import Yolov3, YoloTrainTransform, DetectionDataset
    import os

    input_size = (416, 416)
    device = torch.device("cuda")
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    transform = YoloTrainTransform(input_size[0], input_size[1])
    dataset = DetectionDataset(path='/home/jg/Desktop/mountain/valid', transform=transform)
    num_classes = dataset.num_class

    image, label, _ = dataset[0]

    net = Yolov3(base=18,
                 input_size=input_size,
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=num_classes,  # foreground만
                 pretrained=False)
    net.to(device)

    matcher = Matcher()
    encoder = Encoderdynamic(ignore_threshold=0.7, from_sigmoid=False)

    # batch 형태로 만들기
    image = image[None,:,:]
    label = label[None,:,:]
    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    output1, output2, output3, anchor1, anchor2, anchor3, _, _, _, _, _, _ = net(
        image.to(device))

    matches, ious = matcher([anchor1, anchor2, anchor3], gt_boxes.to(device))
    xcyc_targets, wh_targets, objectness, class_targets, weights = encoder(matches, ious, [output1, output2, output3],
                                                                           [anchor1, anchor2, anchor3], gt_boxes.to(device),
                                                                           gt_ids.to(device),
                                                                           input_size)

    print(f"< input size(height, width) : {input_size} >")
    print(f"xcyc_targets shape : {xcyc_targets.shape}")
    print(f"wh_targets shape : {wh_targets.shape}")
    print(f"objectness shape : {objectness.shape}")
    print(f"class_targets shape : {class_targets.shape}")
    print(f"weights shape : {weights.shape}")
    '''
    < input size(height, width) : (416, 416) >
    xcyc_targets shape : torch.Size([1, 10647, 2])
    wh_targets shape : torch.Size([1, 10647, 2])
    objectness shape : torch.Size([1, 10647, 1])
    class_targets shape : torch.Size([1, 10647])
    weights shape : torch.Size([1, 10647, 2])
    '''
