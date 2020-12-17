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

def face_aligner(images, boxes, landmarks, margin_xyxy=(21, 21, 21, 21), RotationMatrix_Center="boxcenter", reverse_rgb=True, image_show=True):

    '''

    Parameters
    ----------
    images : shape -> (height, width, 3)
    label : shape -> (N(face number), 5) / (xmin, ymin, xmax, ymax, label)
    landmarks : shape -> (N(face number), 10) (x1, y2, x2, y3, ...)
    margin_xyxy
    RotationMatrix_Center -> "eyecenter" or "nosecenter" or "lipscenter" or "boxcenter"
    reverse_rgb : bgr -> rgb or rgb -> bgr

    Returns
    split images -> N number image list
    -------
    '''

    origin = []
    output = []
    oh, ow, _ = images.shape

    if margin_xyxy:
        mx1, my1, mx2, my2 = margin_xyxy
    else:
        mx1, my1, mx2, my2 = 0 ,0, 0, 0

    if reverse_rgb:
        images[:, :, (0, 1, 2)] = images[:, :, (2, 1, 0)]

    '''
        1. images에서 박스 영역 잘라내기 
        2. 왼쪽 오른쪽 눈 좌표를 이용해서 잘라진 박스영역 x축에 수평하게끔 돌리기
    '''
    for box, landmark in zip(boxes, landmarks):

        box = [int(x) for x in box]
        xmin, ymin, xmax, ymax = box

        if xmin==-1 or ymin == -1 or xmax == -1 or ymax == -1:
            continue
        else:
            slice_image = images[ymin:ymax, xmin:xmax, :]
            origin.append(slice_image)

            right_eye_x, right_eye_y, left_eye_x, left_eye_y = landmark[0:4]
            nose_x, nose_y = landmark[4:6]
            right_lips_x, right_lips_y, left_lips_x, left_lips_y = landmark[6:]
            if right_eye_x==-1 or right_eye_y == -1 or left_eye_x==-1 or left_eye_y==-1 or nose_x == -1 or nose_y == -1 or \
                    right_lips_x == -1 or right_lips_y== -1 or left_lips_x == -1 or left_lips_y == -1:
                output.append(slice_image)
            else:
                if ymin - my1 >= 0:
                    nymin = ymin - my1
                else:
                    nymin = 0
                    my1 = ymin

                if xmin-mx1 >= 0:
                    nxmin = xmin - mx1
                else:
                    nxmin = 0
                    mx1 = xmin

                if ymax+my2 <= oh:
                    nymax = ymax+my2
                else:
                    nymax = oh
                    my2 = oh - ymax

                if xmax+mx2 <= ow:
                    nxmax = xmax+mx2
                else:
                    nxmax = ow
                    mx2 = ow - xmax

                slice_image = images[nymin:nymax, nxmin:nxmax, :]
                h, w, _ = slice_image.shape

                x = left_eye_x - right_eye_x
                y = left_eye_y - right_eye_y

                if RotationMatrix_Center == "eyecenter":
                    center = ((right_eye_x+left_eye_x)/2, (right_eye_y+left_eye_y)/2)  # 왼쪽 오른쪽 눈의 중심
                elif RotationMatrix_Center == "nosecenter":
                    center = (nose_x/2, nose_y/2) # 박스의 중심
                elif RotationMatrix_Center == "lipscenter":
                    center = ((right_lips_x+left_lips_x)/2, (right_lips_y+left_lips_y)/2) # 박스의 중심
                elif RotationMatrix_Center == "boxcenter":
                    center = (w/2, h/2) # 박스의 중심
                else:
                    raise AttributeError

                angle = np.arctan2(y, x) * 180 / np.pi
                M = cv2.getRotationMatrix2D(center, angle, 1)

                slice_image = cv2.warpAffine(slice_image, M, dsize=(w, h))

                slice_image = slice_image[my1:h-my2, mx1:w-mx2,:]
                output.append(slice_image)

    if origin and output:
        if image_show:
            for ori, out in zip(origin, output):
                temp = cv2.hconcat([ori, out])
                cv2.imshow("before_and_after",temp)
                cv2.waitKey(0)
    else:
        logging.info("background image")
    return output
