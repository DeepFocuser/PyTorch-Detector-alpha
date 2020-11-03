import logging
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

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
    CLASSES = ['faces']

    def __init__(self, path='Dataset/train', transform=None, sequence_number=1):
        super(DetectionDataset, self).__init__()
        if sequence_number < 1 and isinstance(sequence_number, float):
            logging.error(f"{sequence_number} Must be greater than 0")
            return

        self._name = os.path.basename(path)
        self._sequence_number = sequence_number

        self._image_path = os.path.join(path, "images")
        self._label_txt = os.path.join(self._image_path.replace("images", "labels"), "label.txt")

        self._transform = transform
        self._items = []
        self._itemname = []
        self._make_item_list()

    def key_func(self, path):
        return path

    def _make_item_list(self):

        if os.path.exists(self._label_txt):

            image_path_list = []
            label_dict = defaultdict(list)

            with open(self._label_txt, mode='r') as f:
                lines = f.readlines()
                count = -1
                for line in lines:
                    if line[0] == "#":
                        line = line.lstrip("# ") # #+공백 제거
                        line = line.strip("\n")
                        line = os.path.join(self._image_path, line)

                        image_path_list.append(line)
                        count += 1
                    else:
                        line = line.strip("\n")
                        label_dict[count].append(line)

            for i, image_path in enumerate(image_path_list):
                self._items.append(([image_path], label_dict[i]))

                # 이름 저장
                base_image = os.path.basename(image_path)
                #name = os.path.splitext(base_image)[0]
                self._itemname.append(base_image)
        else:
            logging.info("The dataset does not exist")

    def __getitem__(self, idx):

        images = []
        image_sequence_path, label_string = self._items[idx]
        for image_path in image_sequence_path:
            image = cv2.imread(image_path, flags=-1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        images = np.concatenate(images, axis=-1)
        origin_images = images.copy()

        label = self._parsing(label_string)
        origin_label = label.copy()

        if self._transform:
            result = self._transform(images, label, self._itemname[idx])
            if len(result) == 3:
                return result[0], result[1], result[2], torch.as_tensor(origin_images), torch.as_tensor(origin_label)
            else:
                return result[0], result[1], result[2], result[3], result[4], result[5], result[
                    6], result[7]
        else:
            return images, label, self._itemname[idx]

    def _parsing(self, label_string):

        label_list = []
        if label_string:
            for label in label_string:
                label = label.split(" ")

                # train
                if len(label) > 4:
                    label = label[:-2]
                    label = [float(lb) for lb in label]
                    del label[6]
                    del label[8]
                    del label[10]
                    del label[12]
                else: # valid
                    label = [float(lb) for lb in label]
                    label = label + [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                label.insert(4, 0) # face 클래스
                label[2] = label[0] + label[2]
                label[3] = label[1] + label[3]
                label_list.append(label)
        else:
            label_list.append((-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1))
        return np.array(label_list, dtype="float32")  # 반드시 numpy여야함.

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
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'valid'), sequence_number=sequence_number)

    length = len(dataset)
    sequence_image, label, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('sequence image shape:', sequence_image.shape)

    if sequence_number > 1:
        sequence_image = sequence_image[:,:,3*(sequence_number-1):]
        file_name = file_name[-1]

    plot_bbox(sequence_image, label[:, :4], landmarks=label[:, 5:],
              scores=None, labels=label[:, 4:5],
              class_names=dataset.classes, colors=None, reverse_rgb=True, absolute_coordinates=True,
              image_show=True, image_save=False, image_save_path="result", image_name=os.path.basename(file_name))
    '''
    images length: 1499
    sequence image shape: (720, 1280, 9)
    '''
