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
        self._mean = torch.as_tensor(mean * input_frame_number).reshape((3 * input_frame_number, 1, 1))
        self._std = torch.as_tensor(std * input_frame_number).reshape((3 * input_frame_number, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()
        self._augmentation = augmentation

    def __call__(self, img, mask, name):

        if self._augmentation:

            distortion = np.random.choice([False, True], p=[0.5, 0.5])
            if distortion:
                seq_img_list = []
                seq_img = np.split(img, self._input_frame_number, axis=-1)
                for si in seq_img:
                    si = image_random_color_distort(si, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                                                    saturation_low=0.5, saturation_high=1.5, hue_delta=0.21)
                    seq_img_list.append(si)
                img = np.concatenate(seq_img, axis=-1)

            # random horizontal flip with probability of 0.3
            img, mask, flips = random_flip(img, mask, px=0.3)

            # random vertical flip with probability of 0.3
            img, mask, flips = random_flip(img, mask, py=0.3)

            # rotate
            candidate = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]
            select = random.choice(candidate)
            img = cv2.rotate(img, select)
            mask = cv2.rotate(mask, select)

            # resize with random interpolation
            interp = np.random.randint(0, 3)  # 4,5 중 채널축을 사용하는 resize 방법이 있어서 오류발생하므로, 안쓴다.
            img = cv2.resize(img, (self._width, self._height), interpolation=interp)
            mask = cv2.resize(mask, (self._width, self._height), interpolation=interp)

        else:
            img = cv2.resize(img, (self._width, self._height), interpolation=1)
            mask = cv2.resize(mask, (self._width, self._height), interpolation=1)

        # origin_img=img.copy()
        img = self._toTensor(img)  # 0 ~ 1 로 바꾸기
        img = torch.sub(img, self._mean)
        img = torch.div(img, self._std)

        mask_foreground = np.where(mask >= 127.5, 1, 0)
        mask_background = np.where(mask < 127.5, 1, 0)

        # if self._input_frame_number == 1:
        #     temp = cv2.resize(origin_img,dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        #     temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
        #     cv2.imshow("input image", temp)
        #     cv2.waitKey(0)
        #
        # elif self._input_frame_number == 2:
        #     before, after = np.split(origin_img, 2, axis=-1)
        #     temp = np.concatenate([before, after], axis=1)
        #     temp = cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
        #     temp = cv2.resize(temp,dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        #     cv2.imshow("input image", temp)
        #     cv2.waitKey(0)

        # temp = np.concatenate([mask.astype(np.uint8), mask_foreground.astype(np.uint8)*255, mask_background.astype(np.uint8)*255], axis=1)
        # temp = cv2.resize(temp,dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # cv2.imshow("mask_fore_back", temp)
        # cv2.waitKey(0)

        # foreground, background 순서
        mask = np.stack([mask_foreground, mask_background], axis=0)
        mask = torch.as_tensor(mask, dtype=img.dtype)

        return img, mask, name


class CenterValidTransform(object):

    def __init__(self, input_size, input_frame_number=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self._width = input_size[1]
        self._height = input_size[0]
        self._mean = torch.as_tensor(mean * input_frame_number).reshape((3 * input_frame_number, 1, 1))
        self._std = torch.as_tensor(std * input_frame_number).reshape((3 * input_frame_number, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()

    def __call__(self, img, mask, name):
        h, w, _ = img.shape
        img = cv2.resize(img, (self._width, self._height), interpolation=1)
        mask = cv2.resize(mask, (self._width, self._height), interpolation=1)

        img = self._toTensor(img)  # 0 ~ 1 로 바꾸기
        img = torch.sub(img, self._mean)
        img = torch.div(img, self._std)

        mask_foreground = np.where(mask >= 127.5, 1, 0)
        mask_background = np.where(mask < 127.5, 1, 0)
        mask = np.stack([mask_foreground, mask_background], axis=0)
        mask = torch.as_tensor(mask, dtype=img.dtype)

        return img, mask, name


# test
if __name__ == "__main__":
    import random
    from core.utils.dataprocessing.dataset import SegmentationDataset

    input_size = (512, 512)
    input_frame_number = 2
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = CenterTrainTransform(input_size, input_frame_number=input_frame_number, mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    dataset = SegmentationDataset(path="C:/Users/user/Downloads/P3M-10k/train/blurred_image", transform=transform,
                               sequence_number=input_frame_number)
    length = len(dataset)
    image, mask, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('image shape:', image.shape)
    print('mask shape:', mask.shape)
    print("file name:", file_name)

    '''
    images length: 9421
    image shape: torch.Size([3, 720, 1280])
    mask shape: torch.Size([2, 720, 1280])
    '''
