import torch
from torch.nn import Module

from core.utils.dataprocessing.predictFunction.decoder import Decoder

'''
    Prediction 클래스를 hybridBlock 으로 만든이유?
    net + decoder + nms =  complete object network로 만들어서 json, param 파일로 저장하기 위함
    -> 장점은? mxnet c++에서 inference 할 때, image(RGB) 넣으면 ids, scores, bboxes 가 바로 출력
'''


class Prediction(Module):

    def __init__(self,
                 unique_ids=["smoke"],
                 from_sigmoid=False,
                 num_classes=3,
                 nms_thresh=0.5,
                 nms_topk=500,
                 except_class_thresh=0.05,
                 multiperclass=True):
        super(Prediction, self).__init__()

        self._unique_ids = [-1] + [ i for i in range(len(unique_ids))]
        self._except_class_thresh = except_class_thresh
        self._decoder = Decoder(from_sigmoid=from_sigmoid, num_classes=num_classes, thresh=except_class_thresh,
                                multiperclass=multiperclass)
        self._nms_thresh = nms_thresh
        self._nms_topk = nms_topk

    def _non_maximum_suppression(self, ids, scores, bboxes):

        # https://gist.github.com/mkocabas/a2f565b27331af0da740c11c78699185
        '''
        1. 내림차순 정렬 하기
        2. non maximum suppression 후 빈 박스들은  -1로 채우기

        :param ids: (object number, 1)
        :param scores: (object number, 1)
        :param bboxes: (object number, 4)
        :return: ids, scores, bboxes
        '''

        if ids.shape[0] == 1:
            return ids, scores, bboxes

        # 내림차순 정렬
        indices = scores.argsort(dim=0, descending=True)[:,0] # 내림차순 정렬
        ids = ids[indices]
        scores = scores[indices]
        xmin = bboxes[:,0:1][indices]
        ymin = bboxes[:,1:2][indices]
        xmax = bboxes[:,2:3][indices]
        ymax = bboxes[:,3:4][indices]

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

        # nms 한 것들 mask 씌우기
        ids = torch.where(ids<0, torch.ones_like(ids)*-1, ids) # 0 : nms / -1 : 배경 -> -1 로 표현
        scores = torch.where(scores<0, torch.ones_like(scores)*-1, scores)
        xmin = torch.where(xmin<0, torch.ones_like(xmin)*-1, xmin)
        ymin = torch.where(ymin<0, torch.ones_like(ymin)*-1, ymin)
        xmax = torch.where(xmax<0, torch.ones_like(xmax)*-1, xmax)
        ymax = torch.where(ymax<0, torch.ones_like(ymax)*-1, ymax)
        bboxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1)

        return ids, scores, bboxes

    def forward(self, output1, output2, output3,
                anchor1, anchor2, anchor3,
                offset1, offset2, offset3,
                stride1, stride2, stride3):

        results = []
        for out, an, off, st in zip([output1, output2, output3],
                                    [anchor1, anchor2, anchor3],
                                    [offset1, offset2, offset3],
                                    [stride1, stride2, stride3]):
            results.append(self._decoder(out, an, off, st))
        results = torch.cat(results, dim=1)

        ids = results[:,:,0:1]
        scores = results[:,:,1:2]
        bboxes = results[:,:,2:]

        if self._nms_thresh > 0 and self._nms_thresh < 1:

            ids_list = []
            scores_list = []
            bboxes_list = []

            # batch 별로 나누기
            # for id, score, x_min, y_min, x_max, y_max in zip(ids, scores, xmin, ymin, xmax, ymax):
            for id, score, box in zip(ids, scores, bboxes):
                id_list = []
                score_list = []
                bbox_list = []
                # id별로 나누기
                for uid in self._unique_ids:
                    indices = id == uid
                    bbox = torch.cat(
                        [box[:, 0:1][indices, None], box[:, 1:2][indices, None], box[:, 2:3][indices, None],
                         box[:, 3:4][indices, None]], dim=-1)
                    if uid < 0:  # 배경인 경우
                        id_part, score_part, bbox_part = id[indices, None], score[indices, None], bbox
                    else:
                        id_part, score_part, bbox_part = self._non_maximum_suppression(id[indices, None],
                                                                                       score[indices, None], bbox)

                    id_list.append(id_part)
                    score_list.append(score_part)
                    bbox_list.append(bbox_part)

                id_concat = torch.cat(id_list, dim=0)
                score_concat = torch.cat(score_list, dim=0)
                bbox_concat = torch.cat(bbox_list, dim=0)

                ids_list.append(id_concat)
                scores_list.append(score_concat)
                bboxes_list.append(bbox_concat)

            # batch 차원
            ids = torch.stack(ids_list, dim=0)
            scores = torch.stack(scores_list, dim=0)
            bboxes = torch.stack(bboxes_list, dim=0)

        return ids, scores, bboxes


# test
if __name__ == "__main__":
    from core import Yolov3, YoloTrainTransform, DetectionDataset
    import os

    input_size = (416, 416)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = YoloTrainTransform(input_size[0], input_size[1])
    dataset = DetectionDataset(path=os.path.join(root, 'valid'), transform=transform, sequence_number=1)
    num_classes = dataset.num_class

    image, label, _ = dataset[0]

    net = Yolov3(base=18,
                 input_size=input_size,
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=num_classes,  # foreground만
                 pretrained=False,)

    prediction = Prediction(
        from_sigmoid=False,
        num_classes=num_classes,
        nms_thresh=0.5,
        nms_topk=-1,
        except_class_thresh=0.05,
        multiperclass=False)

    # batch 형태로 만들기
    image = image[None, :, :, :]
    label = label[None, : , :]

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
        image)
    ids, scores, bboxes = prediction(output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3,
                                     stride1, stride2, stride3)

    print(f"nms class id shape : {ids.shape}")
    print(f"nms class scores shape : {scores.shape}")
    print(f"nms box predictions shape : {bboxes.shape}")
    '''
    multiperclass = True 일 때,
    nms class id shape : torch.Size([1, 10647, 1])
    nms class scores shape : torch.Size([1, 10647, 1])
    nms box predictions shape : torch.Size([1, 10647, 4])

    multiperclass = False 일 때,
    nms class id shape : torch.Size([1, 10647, 1])
    nms class scores shape : torch.Size([1, 10647, 1])
    nms box predictions shape : torch.Size([1, 10647, 4])

    '''
