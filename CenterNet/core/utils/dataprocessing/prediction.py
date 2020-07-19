import torch
from torch.nn import Module, MaxPool2d

class Prediction(Module):

    def __init__(self, batch_size=1, topk=100, scale=4.0, nms=False, except_class_thresh=0.01, nms_thresh=0.5):
        super(Prediction, self).__init__()
        self._batch_size = batch_size
        self._topk = topk
        self._scale = scale
        self._heatmap_nms = MaxPool2d(3, stride=1, padding=1)
        self._nms = nms
        self._nms_thresh = nms_thresh
        self._except_class_thresh = except_class_thresh

    def non_maximum_suppression(self, ids, scores, bboxes, valid_thresh=0.2, overlap_thresh=0.5):

        '''
         # ex) thresh=0.01 이상인것만 뽑기
        mask = scores > valid_thresh
        ids = torch.where(mask, ids, torch.ones_like(ids) * -1)
        scores = F.where(mask, scores, torch.ones_like(scores) * -1)
        '''
        # 1. score 내림차순 정렬 후, id score valid_thresh 보다 적은 것 -1로 채우기
        # 2. box에 nms 적용

        return ids, scores, bboxes

    def forward(self, heatmap, offset, wh):
        '''
        The peak keypoint extraction serves
        as a sufficient NMS alternative and can be implemented efficiently on device using a 3 × 3 max pooling operation.
        '''
        keep = self._heatmap_nms(heatmap) == heatmap
        heatmap = torch.mul(keep, heatmap)
        _, _, height, width = heatmap.shape

        # 상위 self._topk개만 뽑아내기
        heatmap_resize=heatmap.reshape((0, -1))
        indices = heatmap_resize.argsort(dim=-1, descending=True)  #(batch, channel * height * width) / int64
        indices = indices[:,:self._topk]

        scores = heatmap_resize.topk(k=self._topk, dim=-1, largest=True, sorted=True)  #(batch, channel * height * width) / int64
        scores = scores[:,:,None]
        ids = torch.floor_divide(indices, (height * width)) # 몫 만 구하기
        ids = ids.float()  # c++에서 float으로 받아오기 때문에!!! 형 변환 필요
        ids = ids[:,:,None]

        '''
        박스 복구
        To limit the computational burden, we use a single size prediction  WxHx2
        for all object categories. 
        offset, wh에 해당 
        '''
        offset = offset.permute(0, 2, 3, 1).reshape(
            (0, -1, 2))  # (batch, x, y, channel) -> (batch, height*width, 2)
        wh = wh.permute(0, 2, 3, 1).reshape((0, -1, 2))  # (batch, width, height, channel) -> (batch, height*width, 2)
        topk_indices = torch.fmod(indices, (height * width))  # 클래스별 index

        # 2차원 복구
        topk_ys = torch.floor_divide(topk_indices, width)  # y축 index
        topk_xs = torch.fmod(topk_indices, width)  # x축 index

        # https://mxnet.apache.org/api/python/docs/api/ndarray/ndarray.html?highlight=gather_nd#mxnet.ndarray.gather_nd
        # offset 에서 offset_xs를 index로 보고 뽑기 - gather_nd를 알고 나니 상당히 유용한 듯.
        # x index가 0번에 있고, y index가 1번에 있으므로!!!
        batch_indices = torch.arange(self._batch_size)
        batch_indices = batch_indices[:offset.shape[0], None]
        batch_indices = batch_indices.repeat_interleave(self._topk, dim=-1) # (batch, self._topk)

        offset_xs_indices = torch.zeros_like(batch_indices, dtype=torch.int64)
        offset_ys_indices = torch.ones_like(batch_indices, dtype=torch.int64)

        offset_xs = torch.cat((batch_indices, topk_indices, offset_xs_indices), dim=0).reshape((3, -1))
        offset_ys = torch.cat((batch_indices, topk_indices, offset_ys_indices), dim=0).reshape((3, -1))

        # 핵심
        # (batch, height*width, 2) / (3(각 인덱스), self_batch_size*self._topk)
        xs = offset[offset_xs[0], offset_xs[1], offset_xs[2]].reshape((-1, self._topk))
        ys = offset[offset_ys[0], offset_ys[1], offset_ys[2]].reshape((-1, self._topk))
        topk_xs = topk_xs.float() + xs
        topk_ys = topk_ys.float() + ys

        # (batch, height*width, 2) / (3(각 인덱스), self_batch_size*self._topk)
        w = wh[offset_xs[0], offset_xs[1], offset_xs[2]].reshape((-1, self._topk))
        h = wh[offset_ys[0], offset_ys[1], offset_ys[2]].reshape((-1, self._topk))

        half_w = w / 2
        half_h = h / 2
        bboxes = [topk_xs - half_w, topk_ys - half_h, topk_xs + half_w, topk_ys + half_h]  # 각각 (batch, self._topk)
        bboxes = torch.cat([bbox[:,:,None] for bbox in bboxes], dim=-1)  # (batch, self._topk, 1) ->  (batch, self._topk, 4)

        if self._nms:
            if self._nms_thresh > 0 and self._nms_thresh < 1:
                ids, scores, bboxes = self.non_maximum_suppression(ids, scores, bboxes * self._scale,
                                                                   valid_thresh=self._except_class_thresh,
                                                                   overlap_thresh=self._nms_thresh,
                                                                   topk=self._topk)
            return ids, scores, bboxes
        else:
            return ids, scores, bboxes * self._scale


# test
if __name__ == "__main__":
    import os
    from collections import OrderedDict
    from core import CenterNet

    input_size = (512, 512)
    scale_factor = 4
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    '''
    heatmap의 bias가 -2.19 인 이유는??? retinanet의 식과 같은데... 흠..
    For the final conv layer of the classification subnet, we set the bias initialization to b = − log((1 − π)/π),
    where π specifies that at the start of training every anchor should be labeled as foreground with confidence of ∼π.
    We use π = .01 in all experiments, although results are robust to the exact value. As explained in §3.3, 
    this initialization prevents the large number of background anchors from generating a large, 
    destabilizing loss value in the first iteration of training
    '''
    net = CenterNet(base=18,
                    heads=OrderedDict([
                        ('heatmap', {'num_output': 3, 'bias': -2.19}),
                        ('offset', {'num_output': 2}),
                        ('wh', {'num_output': 2})
                    ]),
                    head_conv_channel=64,
                    pretrained=False,
                    root=os.path.join(root, 'models'),
                    use_dcnv2=False)


    prediction = Prediction(batch_size=8, topk=100, scale=scale_factor)
    heatmap, offset, wh = net(torch.rand(2, 3, input_size[0], input_size[1]))
    ids, scores, bboxes = prediction(heatmap, offset, wh)

    print(f"< input size(height, width) : {input_size} >")
    print(f"topk class id shape : {ids.shape}")
    print(f"topk class scores shape : {scores.shape}")
    print(f"topk box predictions shape : {bboxes.shape}")
    '''
    < input size(height, width) : (512, 512) >
    topk class id shape : (2, 100, 1)
    topk class scores shape : (2, 100, 1)
    topk box predictions shape : (2, 100, 4)
    '''
