import os

import cv2
import torch
import torchvision

from core.utils.util.image_utils import *


class CenterTrainTransform(object):

    def __init__(self, input_size, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), augmentation=True):

        self._width = input_size[1]
        self._height = input_size[0]
        self._mean = torch.as_tensor(mean).reshape((3, 1, 1))
        self._std = torch.as_tensor(std).reshape((3, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()
        self._augmentation = augmentation

    def __call__(self, anchor, positive, negative):

        if self._augmentation:

            # anchor
            distortion = np.random.choice([False, True], p=[0.5, 0.5])
            if distortion:
                anchor = image_random_color_distort(anchor, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                                                    saturation_low=0.5, saturation_high=1.5, hue_delta=0.21)

            # random horizontal flip with probability of 0.5
            anchor, flips = random_flip(anchor, px=0.5)

            # # random vertical flip with probability of 0.5
            # anchor, flips = random_flip(anchor, py=0.5)
            #
            # # rotate
            # select = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
            # anchor = cv2.rotate(anchor,  random.choice(select))

            # resize with random interpolation
            interp = np.random.randint(0, 5)
            anchor = cv2.resize(anchor, (self._width, self._height), interpolation=interp)

            # positive
            distortion = np.random.choice([False, True], p=[0.5, 0.5])
            if distortion:
                positive = image_random_color_distort(positive, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                                                      saturation_low=0.5, saturation_high=1.5, hue_delta=0.21)

            # random horizontal flip with probability of 0.5
            positive, flips = random_flip(positive, px=0.5)

            # # random vertical flip with probability of 0.5
            # positive, flips = random_flip(positive, py=0.5)
            #
            # # rotate
            # select = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
            # positive = cv2.rotate(positive,  random.choice(select))

            # resize with random interpolation
            interp = np.random.randint(0, 5)
            positive = cv2.resize(positive, (self._width, self._height), interpolation=interp)

            # negative
            distortion = np.random.choice([False, True], p=[0.5, 0.5])
            if distortion:
                negative = image_random_color_distort(negative, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                                                      saturation_low=0.5, saturation_high=1.5, hue_delta=0.21)

            # random horizontal flip with probability of 0.5
            negative, flips = random_flip(negative, px=0.5)

            # # random vertical flip with probability of 0.5
            # negative, flips = random_flip(negative, py=0.5)

            # # rotate
            # select = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
            # negative = cv2.rotate(negative,  random.choice(select))

            # resize with random interpolation
            interp = np.random.randint(0, 5)
            negative = cv2.resize(negative, (self._width, self._height), interpolation=interp)

        else:
            anchor = cv2.resize(anchor, (self._width, self._height), interpolation=1)
            positive = cv2.resize(positive, (self._width, self._height), interpolation=1)
            negative = cv2.resize(negative, (self._width, self._height), interpolation=1)

        anchor = self._toTensor(anchor)  # 0 ~ 1 로 바꾸기
        anchor = torch.sub(anchor, self._mean)
        anchor = torch.div(anchor, self._std)

        positive = self._toTensor(positive)  # 0 ~ 1 로 바꾸기
        positive = torch.sub(positive, self._mean)
        positive = torch.div(positive, self._std)

        negative = self._toTensor(negative)  # 0 ~ 1 로 바꾸기
        negative = torch.sub(negative, self._mean)
        negative = torch.div(negative, self._std)

        return anchor, positive, negative



class CenterValidTransform(object):

    def __init__(self, input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

        self._width = input_size[1]
        self._height = input_size[0]
        self._mean = torch.as_tensor(mean).reshape((3, 1, 1))
        self._std = torch.as_tensor(std).reshape((3, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()

    def __call__(self, anchor, positive, negative):

        anchor = cv2.resize(anchor, (self._width, self._height), interpolation=1)
        anchor = self._toTensor(anchor)  # 0 ~ 1 로 바꾸기
        anchor = torch.sub(anchor, self._mean)
        anchor = torch.div(anchor, self._std)

        positive = cv2.resize(positive, (self._width, self._height), interpolation=1)
        positive = self._toTensor(positive)  # 0 ~ 1 로 바꾸기
        positive = torch.sub(positive, self._mean)
        positive = torch.div(positive, self._std)

        negative = cv2.resize(negative, (self._width, self._height), interpolation=1)
        negative = self._toTensor(negative)  # 0 ~ 1 로 바꾸기
        negative = torch.sub(negative, self._mean)
        negative = torch.div(negative, self._std)

        return anchor, positive, negative

# test
if __name__ == "__main__":
    import random
    from core.utils.dataprocessing.dataset import FaceDataset

    input_size = (256, 256)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = CenterTrainTransform(input_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    dataset = FaceDataset(path=os.path.join('D:\\CASIA-WebFace', 'valid'), transform=transform)
    length = len(dataset)

    anchor, positive, negative, anchor_path, positive_path, negative_path = dataset[0]

    print(f'images length: {length}')

    print(f'anchor shape: {anchor.shape}')
    print(f'anchor path: {anchor_path}\n')

    print(f'positive shape: {positive.shape}')
    print(f'positive path: {positive_path}\n')

    print(f'negative shape: {negative.shape}')
    print(f'negative path: {negative_path}')

    '''
    images length: 1087
    anchor shape: torch.Size([3, 256, 256])
    anchor path: D:\CASIA-WebFace\valid\0004844\001.jpg
    
    positive shape: torch.Size([3, 256, 256])
    positive path: D:\CASIA-WebFace\valid\0004844\008.jpg
    
    negative shape: torch.Size([3, 256, 256])
    negative path: D:\CASIA-WebFace\valid\5875151\005.jpg
    '''
