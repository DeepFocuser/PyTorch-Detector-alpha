import os

import cv2
import torch
import torchvision

from core.utils.util.box_utils import *
from core.utils.util.image_utils import *


class YoloTrainTransform(object):

    def __init__(self, height, width, input_frame_number=1, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225], augmentation=False):

        self._height = height
        self._width = width
        self._input_frame_number = input_frame_number
        self._mean = torch.as_tensor(mean*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._std = torch.as_tensor(std*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()
        self._augmentation = augmentation

    def __call__(self, img, bbox, name):

        if self._augmentation:

            distortion = np.random.choice([False, True], p=[0.5, 0.5])
            if distortion:
                seq_img_list=[]
                seq_img = np.split(img, self._input_frame_number, axis=-1)
                for si in seq_img:
                    si = image_random_color_distort(si, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                                                    saturation_low=0.5, saturation_high=1.5, hue_delta=0.21).astype(np.uint8)
                    seq_img_list.append(si)
                img = np.concatenate(seq_img_list, axis=-1)

            # random cropping
            crop = np.random.choice([False, True], p=[0.5, 0.5])
            if crop:
                seq_img_list = []
                seq_img = np.split(img, self._input_frame_number, axis=-1)
                h, w, _ = img.shape
                bbox, crop = box_random_crop_with_constraints(bbox, (w, h),
                                                              min_scale=0.5,
                                                              max_scale=1.0,
                                                              max_aspect_ratio=2,
                                                              constraints=None,
                                                              max_trial=30)

                x0, y0, w, h = crop
                for si in seq_img:
                    si = si[y0:y0 + h, x0:x0 + w]
                    seq_img_list.append(si)
                img = np.concatenate(seq_img_list, axis=-1)

            # random horizontal flip with probability of 0.5
            h, w, _ = img.shape
            img, flips = random_flip(img, px=0.5)
            bbox = box_flip(bbox, (w, h), flip_x=flips[0])

            # # random vertical flip with probability of 0.5
            # img, flips = random_flip(img, py=0.5)
            # bbox = box_flip(bbox, (w, h), flip_y=flips[1])

            # random translation
            translation = np.random.choice([False, True], p=[0.5, 0.5])
            if translation:
                x_offset = np.random.randint(-7, high=7)
                y_offset = np.random.randint(-7, high=7)
                M = np.float64([[1, 0, x_offset], [0, 1, y_offset]])
                img = cv2.warpAffine(img, M, (w,h))
                bbox = box_translate(bbox, x_offset=x_offset, y_offset=y_offset, shape=(h, w))

            # resize with random interpolation
            h, w, _ = img.shape
            interp = np.random.randint(0, 3)
            img = cv2.resize(img, (self._width, self._height), interpolation=interp)
            bbox = box_resize(bbox, (w, h), (self._width, self._height))

        else:
            h, w, _ = img.shape
            img = cv2.resize(img, (self._width, self._height), interpolation=1)
            bbox = box_resize(bbox, (w, h), (self._width, self._height))

        img = self._toTensor(img)  # 0 ~ 1 로 바꾸기
        img = torch.sub(img, self._mean)
        img = torch.div(img, self._std)
        bbox = torch.as_tensor(bbox)

        return img, bbox, name

class YoloValidTransform(object):

    def __init__(self, height, width, input_frame_number=2, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self._height = height
        self._width = width
        self._mean = torch.as_tensor(mean*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._std = torch.as_tensor(std*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img, bbox, name):

        h, w, _ = img.shape
        img = cv2.resize(img, (self._width, self._height), interpolation=1)
        bbox = box_resize(bbox, (w, h), (self._width, self._height))

        img = self._toTensor(img)  # 0 ~ 1 로 바꾸기
        img = torch.sub(img, self._mean)
        img = torch.div(img, self._std)
        bbox = torch.as_tensor(bbox)

        return img, bbox, name

# test
if __name__ == "__main__":
    import random
    from core.utils.dataprocessing.dataset import DetectionDataset

    input_size = (320, 640)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = YoloTrainTransform(input_size[0], input_size[1], input_frame_number=1, augmentation=True)
    dataset = DetectionDataset(path=os.path.join(root, 'valid'), transform=transform, sequence_number=1)
    length = len(dataset)
    image, label, _ = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('image shape:', image.shape)
    print('label shape:', label.shape)
    '''
    images length: 1150
    image shape: (6, 320, 640)
    label shape: (3, 5)
    '''

