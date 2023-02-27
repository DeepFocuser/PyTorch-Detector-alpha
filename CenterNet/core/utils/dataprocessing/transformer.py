import os

import cv2
import torch
import torchvision

from core.utils.dataprocessing.target import TargetGenerator
from core.utils.util.box_utils import *
from core.utils.util.image_utils import *


class CenterTrainTransform(object):

    def __init__(self, input_size, input_frame_number=1, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), scale_factor=4, augmentation=True, make_target=False, num_classes=3):

        self._width = input_size[1]
        self._height = input_size[0]
        self._input_frame_number = input_frame_number
        self._mean = torch.as_tensor(mean*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._std = torch.as_tensor(std*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()
        self._scale_factor = scale_factor
        self._augmentation = augmentation
        self._make_target = make_target
        if self._make_target:
            self._target_generator = TargetGenerator(num_classes=num_classes)
        else:
            self._target_generator = None

    def __call__(self, img, bbox, name):

        output_w = self._width // self._scale_factor
        output_h = self._height // self._scale_factor

        if self._augmentation:

            distortion = np.random.choice([False, True], p=[0.5, 0.5])
            if distortion:
                seq_img_list=[]
                seq_img = np.split(img, self._input_frame_number, axis=-1)
                for si in seq_img:
                    si = image_random_color_distort(si, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                                                    saturation_low=0.5, saturation_high=1.5, hue_delta=0.21)
                    seq_img_list.append(si)
                img = np.concatenate(seq_img_list, axis=-1).astype(np.uint8)

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
            bbox = box_resize(bbox, (w, h), (output_w, output_h))

        else:
            h, w, _ = img.shape
            img = cv2.resize(img, (self._width, self._height), interpolation=1)
            bbox = box_resize(bbox, (w, h), (output_w, output_h))

        # heatmap 기반이기 때문에 제한 해줘야 한다.
        bbox[:, 0] = np.clip(bbox[:, 0], 0, output_w-1)
        bbox[:, 1] = np.clip(bbox[:, 1], 0, output_h-1)
        bbox[:, 2] = np.clip(bbox[:, 2], 0, output_w-1)
        bbox[:, 3] = np.clip(bbox[:, 3], 0, output_h-1)

        img = self._toTensor(img)  # 0 ~ 1 로 바꾸기
        img = torch.sub(img, self._mean)
        img = torch.div(img, self._std)

        if self._make_target:
            bbox = bbox[np.newaxis, :, :]
            bbox = torch.as_tensor(bbox)
            heatmap, offset_target, wh_target, mask_target = self._target_generator(bbox[:, :, :4], bbox[:, :, 4:5],
                                                                                    output_w, output_h, img.device)
            return img, bbox[0], heatmap[0], offset_target[0], wh_target[0], mask_target[0], name
        else:
            bbox = torch.as_tensor(bbox)
            return img, bbox, name


class CenterValidTransform(object):

    def __init__(self, input_size, input_frame_number=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale_factor=4,
                 make_target=False, num_classes=3):
        self._width = input_size[1]
        self._height = input_size[0]
        self._mean = torch.as_tensor(mean*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._std = torch.as_tensor(std*input_frame_number).reshape((3*input_frame_number, 1, 1))
        self._toTensor = torchvision.transforms.ToTensor()
        self._scale_factor = scale_factor
        self._make_target = make_target
        if self._make_target:
            self._target_generator = TargetGenerator(num_classes=num_classes)
        else:
            self._target_generator = None


    def __call__(self, img, bbox, name):

        output_w = self._width // self._scale_factor
        output_h = self._height // self._scale_factor

        h, w, _ = img.shape
        img = cv2.resize(img, (self._width, self._height), interpolation=1)
        bbox = box_resize(bbox, (w, h), (output_w, output_h))

        img = self._toTensor(img)  # 0 ~ 1 로 바꾸기
        img = torch.sub(img, self._mean)
        img = torch.div(img, self._std)

        # heatmap 기반이기 때문에 제한 해줘야 한다.
        bbox[:, 0] = np.clip(bbox[:, 0], 0, output_w-1)
        bbox[:, 1] = np.clip(bbox[:, 1], 0, output_h-1)
        bbox[:, 2] = np.clip(bbox[:, 2], 0, output_w-1)
        bbox[:, 3] = np.clip(bbox[:, 3], 0, output_h-1)

        if self._make_target:
            bbox = bbox[np.newaxis, :, :]
            bbox = torch.as_tensor(bbox)
            heatmap, offset_target, wh_target, mask_target = self._target_generator(bbox[:, :, :4], bbox[:, :, 4:5],
                                                                                    output_w, output_h, img.device)

            return img, bbox[0], heatmap[0], offset_target[0], wh_target[0], mask_target[0], name
        else:
            bbox = torch.as_tensor(bbox)
            return img, bbox, name


# test
if __name__ == "__main__":
    import random
    from core.utils.dataprocessing.dataset import DetectionDataset

    input_size = (960, 1280)
    scale_factor = 4
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = CenterTrainTransform(input_size, input_frame_number=2, mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225),
                                     scale_factor=scale_factor)
    dataset = DetectionDataset(path=os.path.join(root, 'valid'), transform=transform, sequence_number=2)
    length = len(dataset)
    image, label, file_name, _, _ = dataset[random.randint(0, length - 1)]

    print('images length:', length)
    print('image shape:', image.shape)
    print('label shape:', label.shape)
    '''
    images length: 1500
    image shape: torch.Size([6, 960, 1280])
    label shape: torch.Size([1, 5])
    '''
