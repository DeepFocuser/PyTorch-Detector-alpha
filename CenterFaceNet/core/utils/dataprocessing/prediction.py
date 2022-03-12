import torch
import torch.nn as nn

class Prediction(nn.Module):

    def __init__(self, unique_ids=["smoke"], topk=100, scale=4.0, nms=False, except_class_thresh=0.01, nms_thresh=0.5):
        super(Prediction, self).__init__()
        self._unique_ids = [-1] + [ i for i in range(len(unique_ids))]
        self._topk = topk
        self._scale = scale
        self._heatmap_nms = nn.MaxPool2d(3, stride=1, padding=1)
        self._nms = nms
        self._nms_thresh = nms_thresh
        self._except_class_thresh = except_class_thresh

    def _non_maximum_suppression(self, ids, scores, bboxes, landmarks):

        # https://gist.github.com/mkocabas/a2f565b27331af0da740c11c78699185
        '''
        1. 내림차순 정렬 하기
        2. non maximum suppression 후 빈 박스들은  -1로 채우기

        :param ids: (object number, 1)
        :param scores: (object number, 1)
        :param bboxes: (object number, 4)
        :param landmarks: (object number, 10)
        :return: ids, scores, bboxes, landmarks
        '''

        if ids.shape[0] == 1:
            return ids, scores, bboxes, landmarks
        
        # You should not write a stable argument.
        _, indices = torch.sort(scores, dim=0, descending=True) # 내림차순 정렬
        indices = indices[:,0]
        
        ids = ids[indices]
        scores = scores[indices]
        xmin = bboxes[:,0:1][indices]
        ymin = bboxes[:,1:2][indices]
        xmax = bboxes[:,2:3][indices]
        ymax = bboxes[:,3:4][indices]

        lx1 = landmarks[:,0:1][indices]
        ly1 = landmarks[:,1:2][indices]
        lx2 = landmarks[:,2:3][indices]
        ly2 = landmarks[:,3:4][indices]
        lx3 = landmarks[:,4:5][indices]
        ly3 = landmarks[:,5:6][indices]
        lx4 = landmarks[:,6:7][indices]
        ly4 = landmarks[:,7:8][indices]
        lx5 = landmarks[:,8:9][indices]
        ly5 = landmarks[:,9:10][indices]

        # nms 알고리즘
        x1 = xmin[:, 0]
        y1 = ymin[:, 0]
        x2 = xmax[:, 0]
        y2 = ymax[:, 0]

        mask = torch.ones_like(x1)
        i = 0
        while i < len(ids)-1:

            xx1 = torch.max(x1[i], x1[i+1:])
            yy1 = torch.max(y1[i], y1[i+1:])
            xx2 = torch.min(x2[i], x2[i+1:])
            yy2 = torch.min(y2[i], y2[i+1:])
            w = xx2 - xx1 + 1
            h = yy2 - yy1 + 1

            box1_area = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1)
            boxn_area = (x2[i+1:] - x1[i+1:] + 1) * (y2[i+1:] - y1[i+1:] + 1)
            overlap = (w * h) / (box1_area + boxn_area - (w * h))
            mask[i+1:] = torch.where(overlap > self._nms_thresh, torch.ones_like(overlap)*-1, torch.ones_like(overlap))
            i+=1

        # nms 한 것들 mask 씌우기
        mask = mask[:,None]
        ids = ids * mask
        scores = scores * mask
        xmin = xmin * mask
        ymin = ymin * mask
        xmax = xmax * mask
        ymax = ymax * mask

        lx1 = lx1 * mask
        ly1 = ly1 * mask
        lx2 = lx2 * mask
        ly2 = ly2 * mask
        lx3 = lx3 * mask
        ly3 = ly3 * mask
        lx4 = lx4 * mask
        ly4 = ly4 * mask
        lx5 = lx5 * mask
        ly5 = ly5 * mask

        # nms 한 것들 mask 씌우기
        ids = torch.where(ids<0, torch.ones_like(ids)*-1, ids) # 0 : nms / -1 : 배경 -> -1 로 표현
        scores = torch.where(scores<0, torch.ones_like(scores)*-1, scores)
        xmin = torch.where(xmin<0, torch.ones_like(xmin)*-1, xmin)
        ymin = torch.where(ymin<0, torch.ones_like(ymin)*-1, ymin)
        xmax = torch.where(xmax<0, torch.ones_like(xmax)*-1, xmax)
        ymax = torch.where(ymax<0, torch.ones_like(ymax)*-1, ymax)
        bboxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
        landmarks = torch.cat([lx1, ly1,
                               lx2, ly2,
                               lx3, ly3,
                               lx4, ly4,
                               lx5, ly5
                               ], dim=-1)

        return ids, scores, bboxes, landmarks

    def forward(self, heatmap, offset, wh, landmark):
        '''
        The peak keypoint extraction serves
        as a sufficient NMS alternative and can be implemented efficiently on device using a 3 × 3 max pooling operation.
        '''
        keep = self._heatmap_nms(heatmap) == heatmap
        heatmap = torch.mul(keep, heatmap)
        batch, channel, height, width = heatmap.shape

        # 상위 self._topk개만 뽑아내기
        heatmap_resize = heatmap.reshape((batch, -1))
        scores, indices = heatmap_resize.topk(k=self._topk, dim=-1, largest=True, sorted=True)  #(batch, channel * height * width) / int64

        scores = scores[:,:,None]
        ids = torch.div(indices, (height * width), rounding_mode="floor") # 몫 만 구하기
        ids = ids.float()  # c++에서 float으로 받아오기 때문에!!! 형 변환 필요
        ids = ids[:,:,None]

        '''
        박스 복구
        To limit the computational burden, we use a single size prediction  WxHx2
        for all object categories. 
        offset, wh에 해당 
        '''
        offset = offset.permute(0, 2, 3, 1).reshape(
            (batch, -1, 2))  # (batch, x, y, channel) -> (batch, height*width, 2)
        wh = wh.permute(0, 2, 3, 1).reshape((batch, -1, 2))  # (batch, width, height, channel) -> (batch, height*width, 2)
        landmark = landmark.permute(0, 2, 3, 1).reshape((batch, -1, landmark.shape[1])) # (batch, width, height, channel) -> (batch, height*width, 10)
        landmark_split = torch.split(landmark, 2, dim=-1) # 각각 (batch, height*width, 2)
        
        topk_indices = torch.remainder(indices, (height * width))

        # 2차원 복구
        topk_ys = torch.div(topk_indices, width, rounding_mode="floor")  # y축 index
        topk_xs = torch.remainder(topk_indices, width)  # x축 index
     
        batch_indices = torch.arange(batch, device=ids.device).unsqueeze(dim=-1)
        batch_indices = batch_indices.repeat(1, self._topk) # (batch, self._topk)

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

        landmark1_x = (landmark_split[0][offset_xs[0], offset_xs[1], offset_xs[2]]).reshape((-1, self._topk, 1))
        landmark1_y = (landmark_split[0][offset_ys[0], offset_ys[1], offset_ys[2]]).reshape((-1, self._topk, 1))

        landmark2_x = (landmark_split[1][offset_xs[0], offset_xs[1], offset_xs[2]]).reshape((-1, self._topk, 1))
        landmark2_y = (landmark_split[1][offset_ys[0], offset_ys[1], offset_ys[2]]).reshape((-1, self._topk, 1))

        landmark3_x = (landmark_split[2][offset_xs[0], offset_xs[1], offset_xs[2]]).reshape((-1, self._topk, 1))
        landmark3_y = (landmark_split[2][offset_ys[0], offset_ys[1], offset_ys[2]]).reshape((-1, self._topk, 1))

        landmark4_x = (landmark_split[3][offset_xs[0], offset_xs[1], offset_xs[2]]).reshape((-1, self._topk, 1))
        landmark4_y = (landmark_split[3][offset_ys[0], offset_ys[1], offset_ys[2]]).reshape((-1, self._topk, 1))

        landmark5_x = (landmark_split[4][offset_xs[0], offset_xs[1], offset_xs[2]]).reshape((-1, self._topk, 1))
        landmark5_y = (landmark_split[4][offset_ys[0], offset_ys[1], offset_ys[2]]).reshape((-1, self._topk, 1))

        # landmarks 복구
        topk_xs = topk_xs[:,:,None]
        topk_ys = topk_ys[:,:,None]
        landmark1_x = landmark1_x + topk_xs
        landmark1_y = landmark1_y + topk_ys

        landmark2_x = landmark2_x + topk_xs
        landmark2_y = landmark2_y + topk_ys

        landmark3_x = landmark3_x + topk_xs
        landmark3_y = landmark3_y + topk_ys

        landmark4_x = landmark4_x + topk_xs
        landmark4_y = landmark4_y + topk_ys

        landmark5_x = landmark5_x + topk_xs
        landmark5_y = landmark5_y + topk_ys

        except_mask = scores > self._except_class_thresh
        ids = torch.where(except_mask, ids, torch.ones_like(ids) * -1)
        scores = torch.where(except_mask, scores, torch.ones_like(scores) * -1)
        xmin = torch.where(except_mask, bboxes[:, :, 0:1], torch.ones_like(bboxes[:, :, 0:1]) * -1)
        ymin = torch.where(except_mask, bboxes[:, :, 1:2], torch.ones_like(bboxes[:, :, 1:2]) * -1)
        xmax = torch.where(except_mask, bboxes[:, :, 2:3], torch.ones_like(bboxes[:, :, 2:3]) * -1)
        ymax = torch.where(except_mask, bboxes[:, :, 3:4], torch.ones_like(bboxes[:, :, 3:4]) * -1)
        bboxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1)

        landmark1_x = torch.where(except_mask, landmark1_x, torch.ones_like(landmark1_x) * -1)
        landmark1_y = torch.where(except_mask, landmark1_y, torch.ones_like(landmark1_y) * -1)
        landmark2_x = torch.where(except_mask, landmark2_x, torch.ones_like(landmark2_x) * -1)
        landmark2_y = torch.where(except_mask, landmark2_y, torch.ones_like(landmark2_y) * -1)
        landmark3_x = torch.where(except_mask, landmark3_x, torch.ones_like(landmark3_x) * -1)
        landmark3_y = torch.where(except_mask, landmark3_y, torch.ones_like(landmark3_y) * -1)
        landmark4_x = torch.where(except_mask, landmark4_x, torch.ones_like(landmark4_x) * -1)
        landmark4_y = torch.where(except_mask, landmark4_y, torch.ones_like(landmark4_y) * -1)
        landmark5_x = torch.where(except_mask, landmark5_x, torch.ones_like(landmark5_x) * -1)
        landmark5_y = torch.where(except_mask, landmark5_y, torch.ones_like(landmark5_y) * -1)

        landmark_list = [landmark1_x, landmark1_y, landmark2_x, landmark2_y, landmark3_x, landmark3_y, landmark4_x, landmark4_y, landmark5_x, landmark5_y]
        landmarks = torch.cat(landmark_list, dim=-1)  # (batch, self._topk, 1) ->  (batch, self._topk, 10)


        if self._nms:
            if self._nms_thresh > 0 and self._nms_thresh < 1:

                ids_list = []
                scores_list = []
                bboxes_list = []
                llmarks_list = []

                # batch 별로 나누기
                #for id, score, x_min, y_min, x_max, y_max in zip(ids, scores, xmin, ymin, xmax, ymax):
                for id, score, box, lmark in zip(ids, scores, bboxes, landmarks):
                    id_list = []
                    score_list = []
                    bbox_list = []
                    llmark_list = []
                    # id별로 나누기
                    for uid in self._unique_ids:
                        indices = id==uid
                        bbox = torch.cat([box[:,0:1][indices, None], box[:,1:2][indices, None], box[:, 2:3][indices, None], box[:,3:4][indices, None]], dim=-1)
                        llmark = torch.cat([lmark[:, 0:1][indices, None],
                                            lmark[:, 1:2][indices, None],
                                            lmark[:, 2:3][indices, None],
                                            lmark[:, 3:4][indices, None],
                                            lmark[:, 4:5][indices, None],
                                            lmark[:, 5:6][indices, None],
                                            lmark[:, 6:7][indices, None],
                                            lmark[:, 7:8][indices, None],
                                            lmark[:, 8:9][indices, None],
                                            lmark[:, 9:10][indices, None]], dim=-1)
                        if uid < 0: # 배경인 경우
                            id_part, score_part, bbox_part, llmark_part = id[indices, None], score[indices, None], bbox, llmark
                        else:
                            id_part, score_part, bbox_part, llmark_part = self._non_maximum_suppression(id[indices, None], score[indices, None], bbox, llmark)

                        id_list.append(id_part)
                        score_list.append(score_part)
                        bbox_list.append(bbox_part)
                        llmark_list.append(llmark_part)

                    id_concat = torch.cat(id_list, dim=0)
                    score_concat = torch.cat(score_list, dim=0)
                    bbox_concat = torch.cat(bbox_list, dim=0)
                    llmark_concat = torch.cat(llmark_list, dim=0)

                    ids_list.append(id_concat)
                    scores_list.append(score_concat)
                    bboxes_list.append(bbox_concat)
                    llmarks_list.append(llmark_concat)

                # batch 차원
                ids = torch.stack(ids_list, dim=0)
                scores = torch.stack(scores_list, dim=0)
                bboxes = torch.stack(bboxes_list, dim=0)
                landmarks = torch.stack(llmarks_list, dim=0)

            return ids, scores, bboxes * self._scale, landmarks * self._scale
        else:
            return ids, scores, bboxes * self._scale, landmarks * self._scale

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
                    input_frame_number=1,
                    heads=OrderedDict([
                        ('heatmap', {'num_output': 1, 'bias': -2.19}),
                        ('offset', {'num_output': 2}),
                        ('wh', {'num_output': 2}),
                        ('landmark', {'num_output': 10})
                    ]),
                    head_conv_channel=64,
                    pretrained=False)


    prediction = Prediction(unique_ids=["faces"], topk=100, scale=scale_factor, nms=True, except_class_thresh=0.1, nms_thresh=0.5)

    with torch.no_grad():
        heatmap, offset, wh, landmark = net(torch.rand(1, 3, input_size[0], input_size[1]))
        ids, scores, bboxes, landmarks = prediction(heatmap, offset, wh, landmark)

    print(f"< input size(height, width) : {input_size} >")
    print(f"topk class id shape : {ids.shape}")
    print(f"topk class scores shape : {scores.shape}")
    print(f"topk box predictions shape : {bboxes.shape}")
    print(f"topk landmarks predictions shape : {landmarks.shape}")
    '''
    < input size(height, width) : (512, 512) >
    topk class id shape : torch.Size([1, 100, 1])
    topk class scores shape : torch.Size([1, 100, 1])
    topk box predictions shape : torch.Size([1, 100, 4])
    topk landmark predictions shape : torch.Size([1, 100, 10])
    '''
