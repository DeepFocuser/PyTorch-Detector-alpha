import torch
from torch.nn import Module

class BBoxCenterToCorner(Module):

    def __init__(self, axis=-1):
        super(BBoxCenterToCorner, self).__init__()
        self._axis = axis

    def forward(self, x):

        x, y, width, height = torch.split(x, 1, dim=self._axis)
        # half_w = torch.true_divide(width, 2)
        half_w = torch.true_divide(width, 2)
        half_h = torch.true_divide(height, 2)
        xmin = x - half_w
        ymin = y - half_h
        xmax = x + half_w
        ymax = y + half_h
        return torch.cat([xmin, ymin, xmax, ymax], dim=self._axis)

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

class Matcher(Module):

    def __init__(self):
        super(Matcher, self).__init__()
        self._batchiou = BBoxBatchIOU(axis=-1)
        self._cornertocenter = BBoxCornerToCenter(axis=-1)
        self._centertocorner = BBoxCenterToCorner(axis=-1)

    def forward(self, anchors, gt_boxes):

        '''
        gt_box : 중심이 0인 공간으로 gt를 mapping하는 방법
        -> grid cell 기반이라서 이러한 방법으로 matching 가능
         anchor와 gt의 중심점은 공유된다.
        '''
        gtx, gty, gtw, gth = self._cornertocenter(gt_boxes)  # 1. gt를 corner -> center로 변경하기
        shift_gt_boxes =torch.cat([-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth], dim=-1)  # 중심점이 0,0인 corner로 바꾸기
        '''
        anchor는 width, height를 알고 있으니 중심점이 0, 0 을 가리키도록 한다. 
        '''
        all_anchors = torch.cat([a.reshape(-1, 2) for a in anchors], dim=0)
        anchor_boxes = torch.cat([0 * all_anchors, all_anchors], dim=-1)  # zero center anchors / (9, 4)
        anchor_boxes = self._centertocorner(anchor_boxes).unsqueeze(0)

        # anchor_boxes : (1, 9, 4) / gt_boxes : (Batch, N, 4) -> (Batch, 9, N)
        ious = self._batchiou(anchor_boxes, shift_gt_boxes)

        # numpy로 바꾸기
        matches = ious.argmax(axis=1).cpu().numpy().copy().astype(int)  # (Batch, N) / 가장 큰것 하나만 뽑는다.
        ious = ious.cpu().numpy().copy()

        return matches, ious


# test
if __name__ == "__main__":
    from core import Yolov3, YoloTrainTransform,DetectionDataset
    import os

    input_size = (416, 416)
    device = torch.device("cuda")
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    transform = YoloTrainTransform(input_size[0], input_size[1])
    dataset = DetectionDataset(path=os.path.join(root, 'valid'), transform=transform)
    num_classes = dataset.num_class

    image, label, _ = dataset[0]

    net = Yolov3(base=18,
                 input_size=input_size,
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=num_classes,  # foreground만
                 pretrained=False,)
    net.to(device)

    centertocorner = BBoxCenterToCorner(axis=-1)
    cornertocenter = BBoxCornerToCenter(axis=-1)
    matcher = Matcher()

    # batch 형태로 만들기
    image = image[None,:,:]
    label = label[None,:,:]

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]

    _, _, _, anchor1, anchor2, anchor3, _, _, _, _, _, _ = net(image.to(device))
    matches, ious = matcher([anchor1, anchor2, anchor3], gt_boxes.to(device))
    print(f"match shape : {matches.shape}")
    print(f"iou shape : {ious.shape}")
    '''
    match shape : (1, 1)
    iou shape : (1, 9, 1)
    '''
