import logging
import os

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
def plot_bbox(img, score=None, label=None, class_names=None, colors=None, reverse_rgb=False,
              image_show=False, image_save=False, image_save_path=None, image_name=None, gt=False):
    if image_save:
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)

    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy().copy()
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy().copy()

    if isinstance(img, torch.Tensor):
        img = img.type(torch.uint8)
        img = img.detach().cpu().numpy().copy()

    img = img.astype(np.uint8)

    if label:

        id = None
        for i, cn in enumerate(class_names):
            if cn == label:
                id = i
                break

        if reverse_rgb:
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]

        copied_img = img.copy()
        h, w, _ = img.shape
        # use random colors if None is provided
        if colors is None:
            colors = dict()

        if gt:
            denorm_color = [0, 255, 0]
        else:
            colors[id] = plt.get_cmap('hsv')(id / len(class_names))
            denorm_color = [x * 255 for x in colors[id]]

        cv2.rectangle(img, (0, 0), (w, h), denorm_color, thickness=3)

        if score == None:
            score=''

        cv2.putText(copied_img,
                    text='{} {}'.format(label, score), \
                    org=(5, 14), \
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, \
                    fontScale=0.5, \
                    color=denorm_color, \
                    thickness=1, bottomLeftOrigin=False)

        result = cv2.addWeighted(img, 0.5, copied_img, 0.5, 0)

        if image_save:
            cv2.imwrite(os.path.join(image_save_path, image_name + ".jpg"), result)
        if image_show:
            cv2.imshow(image_name, result)
            cv2.waitKey(0)

        return result

class PrePostNet(nn.Module):

    def __init__(self, net=None, input_frame_number=1):
        super(PrePostNet, self).__init__()

        self._mean = torch.as_tensor([123.675, 116.28, 103.53]*input_frame_number).reshape((1, 1, 1, 3*input_frame_number))
        self._scale = torch.as_tensor([58.395, 57.12, 57.375]*input_frame_number).reshape((1, 1, 1, 3*input_frame_number))
        self._net = net

    def forward(self, x):
        x = torch.sub(x, self._mean.to(x.device))
        x = torch.div(x, self._scale.to(x.device))
        x = x.permute(0, 3, 1, 2)
        x = self._net(x)
        x = torch.softmax(x, dim=-1)
        return x
