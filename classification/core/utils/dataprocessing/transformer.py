import os

import cv2
import torch
import torchvision

from core.utils.util.image_utils import *


class CenterTrainTransform(object):

    def __init__(self, input_size, input_frame_number=1, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), augmentation=True):

        self._width = input_size[1]
        self._height = input_size[0]
        self._input_frame_number = input_frame_number
        self._mean = torch.as_tensor(mean*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._std = torch.as_tensor(std*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()
        self._augmentation = augmentation

    def __call__(self, img, label, name):

        if self._augmentation:

            distortion = np.random.choice([False, True], p=[0.5, 0.5])
            if distortion:
                seq_img_list=[]
                seq_img = np.split(img, self._input_frame_number, axis=-1)
                for si in seq_img:
                    si = image_random_color_distort(si, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                                                    saturation_low=0.5, saturation_high=1.5, hue_delta=0.21)
                    seq_img_list.append(si)
                img = np.concatenate(seq_img, axis=-1)

            # random crop을 할만한 크기가 아니다

            # random horizontal flip with probability of 0.5
            img, flips = random_flip(img, px=0.5)

            # random vertical flip with probability of 0.5
            img, flips = random_flip(img, py=0.5)

            # rotate
            select = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
            img = cv2.rotate(img,  random.choice(select))

            # resize with random interpolation
            interp = np.random.randint(0, 5)
            img = cv2.resize(img, (self._width, self._height), interpolation=interp)

        else:
            img = cv2.resize(img, (self._width, self._height), interpolation=1)

        img = self._toTensor(img)  # 0 ~ 1 로 바꾸기
        img = torch.sub(img, self._mean)
        img = torch.div(img, self._std)

        label = torch.as_tensor(label)
        return img, label, name



class CenterValidTransform(object):

    def __init__(self, input_size, input_frame_number=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self._width = input_size[1]
        self._height = input_size[0]
        self._mean = torch.as_tensor(mean*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._std = torch.as_tensor(std*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img, label, name):

        h, w, _ = img.shape
        img = cv2.resize(img, (self._width, self._height), interpolation=1)

        img = self._toTensor(img)  # 0 ~ 1 로 바꾸기
        img = torch.sub(img, self._mean)
        img = torch.div(img, self._std)

        label = torch.as_tensor(label)
        return img, label, name

# test
if __name__ == "__main__":
    import random
    from core.utils.dataprocessing.dataset import DetectionDataset

    input_size = (256, 256)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = CenterTrainTransform(input_size, input_frame_number=1, mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset_syphonic', 'valid'), transform=transform, sequence_number=1)
    length = len(dataset)

    image, label, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('image shape:', image.shape)
    print('label shape:', label.shape)
    '''
    image shape: torch.Size([3, 960, 1280])
    label shape: torch.Size([2])
    '''
