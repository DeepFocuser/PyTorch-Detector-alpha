import glob
import logging
import os

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

# P3M-10K Dataset
class SegmentationDataset(Dataset):

    def __init__(self, path='Dataset/train', transform=None, sequence_number=1, test=False):
        super(SegmentationDataset, self).__init__()
        if sequence_number < 1 and isinstance(sequence_number, float):
            logging.error(f"{sequence_number} Must be greater than 0")
            return

        self._name = os.path.basename(path)
        self._sequence_number = sequence_number
        self._image_path_list = sorted(glob.glob(os.path.join(path, "*")), key=lambda path: self.key_func(path))

        self._transform = transform
        self._items = []
        self._itemname = []
        self._test = test
        self._make_item_list()

    def key_func(self, path):
        return path

    def _make_item_list(self):

        if self._image_path_list:
            for image_path in self._image_path_list:
                mask_path = image_path.replace(self._name, "mask")
                mask_path = mask_path.replace(".jpg", ".png")
                self._items.append((image_path, mask_path))
                base_image = os.path.basename(image_path)
                self._itemname.append(base_image)
        else:
            logging.info("The dataset does not exist")

    def __getitem__(self, idx):

        image_path, mask_path = self._items[idx]

        image = cv2.imread(image_path, flags=-1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, flags=0) # grayscale

        if self._transform:
            result = self._transform(image, mask, self._itemname[idx])
            return result[0], result[1], result[2]
        else:
            return image, mask, self._itemname[idx]

    def __str__(self):
        return self._name + " " + "dataset"

    def __len__(self):
        return len(self._items)

if __name__ == "__main__":
    import random
    from core.utils.util.utils import plot_bbox

    sequence_number = 2
    # root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), sequence_number=sequence_number)
    # root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = SegmentationDataset(path='C:/Users/user/Downloads/VideoMatte240K_JPEG_HD/train', sequence_number=sequence_number)
    length = len(dataset)
    sequence_image, mask, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('sequence image shape:', sequence_image.shape)

    if sequence_number > 1:
        sequence_image = sequence_image[:,:,3*(sequence_number-1):]

    plot_bbox(sequence_image, reverse_rgb=True,
              image_show=True, image_save=False, image_save_path="result", image_name=os.path.basename(file_name))
    plot_bbox(mask, reverse_rgb=True,
              image_show=True, image_save=False, image_save_path="result", image_name=os.path.basename(file_name))
    '''
    images length: 9421
    sequence image shape: (1080, 1620, 3)
    '''
