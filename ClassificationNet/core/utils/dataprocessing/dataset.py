import glob
import logging
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


class DetectionDataset(Dataset):
    """
    Parameters
    ----------
    path : str(jpg)
        Path to input image directory.
    transform : object
    """
    CLASSES = ['ng', 'ok']

    def __init__(self, path='Dataset/train', transform=None, sequence_number=1, test=False):
        super(DetectionDataset, self).__init__()
        if sequence_number < 1 and isinstance(sequence_number, float):
            logging.error(f"{sequence_number} Must be greater than 0")
            return

        self._name = os.path.basename(path)
        self._sequence_number = sequence_number
        self._class_path_List = sorted(glob.glob(os.path.join(path, "*")), key=lambda path: self.key_func(path))

        self._transform = transform
        self._items = []
        self._itemname = []
        self._test = test
        self._make_item_list()

    def key_func(self, path):
        return path

    def _make_item_list(self):

        if self._class_path_List:
            for path in self._class_path_List:
                class_name = os.path.basename(path)
                image_path_list = sorted(glob.glob(os.path.join(path, "*")), key=lambda path: self.key_func(path))
                for i in tqdm(range(len(image_path_list) - (self._sequence_number - 1))):
                    image_path = image_path_list[i:i + self._sequence_number]
                    self._items.append((image_path, class_name))
                    base_image = os.path.basename(image_path[-1])
                    self._itemname.append(base_image)
        else:
            logging.info("The dataset does not exist")

    def __getitem__(self, idx):

        images = []
        image_sequence_path, label = self._items[idx]
        for image_path in image_sequence_path:
            image = cv2.imread(image_path, flags=-1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        images = np.concatenate(images, axis=-1)
        origin_images = images.copy()

        if self._transform:
            one_hot_label = self._one_hot(label)
            result = self._transform(images, one_hot_label, self._itemname[idx])
            if self._test:
                return result[0], result[1], result[2], torch.as_tensor(origin_images)
            else:
                return result[0], result[1], result[2]
        else:
            return origin_images, label, self._itemname[idx]

    def _one_hot(self, label):

        unit_matrix = np.eye(len(self.CLASSES))
        if label == 'ng':
            label=unit_matrix[0]
        elif label == 'ok':
            label=unit_matrix[1]
        return label

    @property
    def classes(self):
        return self.CLASSES

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.CLASSES)

    def __str__(self):
        return self._name + " " + "dataset"

    def __len__(self):
        return len(self._items)


# test
if __name__ == "__main__":
    import random
    from core.utils.util.utils import plot_bbox

    sequence_number = 1
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), sequence_number=sequence_number)

    length = len(dataset)
    sequence_image, label, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('sequence image shape:', sequence_image.shape)

    if sequence_number > 1:
        sequence_image = sequence_image[:,:,3*(sequence_number-1):]
        file_name = file_name[-1]

    plot_bbox(sequence_image, score=None, label=label,
              class_names=dataset.classes, colors=None, reverse_rgb=True,
              image_show=True, image_save=False, image_save_path="result", image_name=os.path.basename(file_name), gt=True)
    '''
    images length: 1499
    sequence image shape: (720, 1280, 9)
    '''
