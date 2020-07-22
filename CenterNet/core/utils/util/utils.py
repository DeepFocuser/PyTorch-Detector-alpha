import logging
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

# test시 nms 통과후 적용
def plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, reverse_rgb=False, absolute_coordinates=True,
              image_show=False, image_save=False, image_save_path=None, image_name=None, heatmap=None):

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if image_save:
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)

    if isinstance(img, torch.Tensor):

        img = img.type(torch.uint8)
        img = img.detach().cpu().numpy().copy()

    img = img.astype(np.uint8)

    if len(bboxes) < 1:
        if image_save:
            cv2.imwrite(os.path.join(image_save_path, image_name + ".jpg"), img)
        if image_show:
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
        return img
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy().copy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy().copy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy().copy()

        if reverse_rgb:
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]

        copied_img = img.copy()

        if not absolute_coordinates:
            # convert to absolute coordinates using image shape
            height = img.shape[0]
            width = img.shape[1]
            bboxes[:, (0, 2)] *= width
            bboxes[:, (1, 3)] *= height

        # use random colors if None is provided
        if colors is None:
            colors = dict()

        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.ravel()[i] < thresh:  # threshold보다 작은 것 무시
                continue
            if labels is not None and labels.ravel()[i] < 0:  # 0이하 인것들 인것 무시
                continue

            cls_id = int(labels.ravel()[i]) if labels is not None else -1
            if cls_id not in colors:
                if class_names is not None and cls_id != -1:
                    colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
                else:
                    colors[cls_id] = (random.random(), random.random(), random.random())

            denorm_color = [x * 255 for x in colors[cls_id]]
            bbox[np.isinf(bbox)] = 0
            bbox[bbox < 0] = 0
            xmin, ymin, xmax, ymax = [int(np.rint(x)) for x in bbox]

            try:
                '''
                colors[cls_id] -> 기본적으로 list, tuple 자료형에 동작함

                numpy인 경우 float64만 동작함 - 나머지 동작안함
                다른 자료형 같은 경우는 tolist로 바꾼 다음 넣어줘야 동작함.
                '''
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), denorm_color, thickness=3)
            except Exception as E:
                logging.info(E)

            if class_names is not None and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ''

            score = '{:.2f}'.format(scores.ravel()[i]) if scores is not None else ''

            if class_name or score:
                cv2.putText(copied_img,
                            text='{} {}'.format(class_name, score), \
                            org=(xmin + 7, ymin + 20), \
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, \
                            fontScale=0.5, \
                            color=[255, 255, 255], \
                            thickness=1, bottomLeftOrigin=False)

        result = cv2.addWeighted(img, 0.5, copied_img, 0.5, 0)

        if heatmap is not None:
            result = np.concatenate([result, heatmap], axis=1)
        if image_save:
            cv2.imwrite(os.path.join(image_save_path, image_name + ".jpg"), result)
        if image_show:
            cv2.imshow(image_name, result)
            cv2.waitKey(0)

        return result

class PrePostNet(nn.Module):

    def __init__(self, net=None, auxnet=None, input_frame_number=2):
        super(PrePostNet, self).__init__()

        mean = torch.as_tensor([123.675, 116.28, 103.53]*input_frame_number).reshape((1, 1, 1, 3*input_frame_number))
        scale = torch.as_tensor([58.395, 57.12, 57.375]*input_frame_number).reshape((1, 1, 1, 3*input_frame_number))
        self._init_mean = torch.nn.Parameter(data=mean, requires_grad=False)
        self._init_scale = torch.nn.Parameter(data=scale, requires_grad=False)
        self._net = net
        self._auxnet = auxnet

    def forward(self, x):
        x = torch.sub(x, self._init_mean)
        x = torch.div(x, self._init_scale)
        x = x.permute(0, 3, 1, 2)
        heatmap_pred, offset_pred, wh_pred = self._net(x)
        return self._auxnet(heatmap_pred, offset_pred, wh_pred)
