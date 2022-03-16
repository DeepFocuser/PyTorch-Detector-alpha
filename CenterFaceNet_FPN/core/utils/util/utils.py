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
def plot_bbox(img, bboxes, landmarks=None, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, reverse_rgb=False, absolute_coordinates=True,
              image_show=False, image_save=False, image_save_path=None, image_name=None, heatmap=None):

    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if image_save:
        image_name = image_name if image_name else "image"
        image_save_path = image_save_path if image_save_path else "images"
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
            image_name = image_name if image_name else "image"
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
        return img
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy().copy()
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.detach().cpu().numpy().copy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy().copy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy().copy()

        if reverse_rgb:
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]

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
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), denorm_color, thickness=1)
            except Exception as E:
                logging.info(E)

            # landmark 그리기
            try:
                for j in range(0, len(landmarks[i]), 2):
                    cv2.line(img, landmarks[i][j:j+2].astype(np.int), landmarks[i][j:j+2].astype(np.int), denorm_color, thickness=2)
            except Exception as E:
                logging.info(E)

            if class_names is not None and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ''

            score = '{:.2f}'.format(scores.ravel()[i]) if scores is not None else ''

            if class_name or score:
                cv2.putText(img,
                            text='{} {}'.format(class_name, score), \
                            org=(xmin - 22 , ymin - 7), \
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, \
                            fontScale=0.4, \
                            color=denorm_color, \
                            thickness=1, bottomLeftOrigin=False)

        if heatmap is not None:
            img = np.concatenate([img, heatmap], axis=1)
        if image_save:
            cv2.imwrite(os.path.join(image_save_path, image_name + ".jpg"), img)
        if image_show:
            image_name = image_name if image_name else "image"
            cv2.imshow(image_name, img)
            cv2.waitKey(0)

        return img

class PrePostNet(nn.Module):

    def __init__(self, net=None, auxnet=None, input_frame_number=1):
        super(PrePostNet, self).__init__()

        self._mean = torch.as_tensor([123.675, 116.28, 103.53]*input_frame_number).reshape((1, 1, 1, 3*input_frame_number))
        self._scale = torch.as_tensor([58.395, 57.12, 57.375]*input_frame_number).reshape((1, 1, 1, 3*input_frame_number))
        self._net = net
        self._auxnet = auxnet

    def forward(self, x):
        x = torch.sub(x, self._mean.to(x.device))
        x = torch.div(x, self._scale.to(x.device))
        x = x.permute(0, 3, 1, 2)
        heatmap_pred, offset_pred, wh_pred, landmark_pred = self._net(x)
        return self._auxnet(heatmap_pred, offset_pred, wh_pred, landmark_pred)
