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
    dataset = DetectionDataset(path=os.path.join(root, 'valid'), transform=transform, sequence_number=2)
    num_classes = dataset.num_class

    image, label, _ = dataset[0]

    net = Yolov3(Darknetlayer=53,
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
        multiperclass=True)

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
    nms class id shape : (1, 53235, 1)
    nms class scores shape : (1, 53235, 1)
    nms box predictions shape : (1, 53235, 4)

    multiperclass = False 일 때,
    nms class id shape : (1, 10647, 1)
    nms class scores shape : (1, 10647, 1)
    nms box predictions shape : (1, 10647, 4)
    '''
