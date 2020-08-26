import glob
import json
import logging
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸

if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

class DetectionDataset(Dataset):

    CLASSES = ['smoke']

    def __init__(self, path='valid', transform=None, sequence_number=2, test=False):
        super(DetectionDataset, self).__init__()

        if sequence_number < 1 and isinstance(sequence_number, float):
            logging.error(f"{sequence_number} Must be greater than 0")
            return

        self._name = os.path.basename(path)
        self._sequence_number = sequence_number
        self._camera_list = glob.glob(os.path.join(path, "images", "*"))
        self._transform = transform
        self._items = []
        self._itemname = []
        self._make_item_list()

    def key_func(self, path):

        base_path = os.path.basename(path)
        except_format = os.path.splitext(base_path)[0]
        split_path = except_format.split("_")
        number = int(split_path[-1])
        return number

    def _make_item_list(self):
        if self._camera_list:
            for camera_list in self._camera_list:
                for camera in glob.glob(os.path.join(camera_list, "*")):
                    image_path_list = sorted(glob.glob(os.path.join(camera, "*.jpg")), key=lambda path: self.key_func(path))
                    for i in range(len(image_path_list) - (self._sequence_number - 1)):
                        image_path = image_path_list[i:i + self._sequence_number]
                        label_path = image_path[-1].replace("images", "labels").replace(".jpg", ".json")
                        self._items.append((image_path, label_path))
                        # base_image = os.path.basename(image_path[-1])
                        # name = os.path.splitext(base_image)[0]
                        # self._itemname.append(name)
                        self._itemname.append(image_path[-1])
        else:
            logging.info("The dataset does not exist")

    def __getitem__(self, idx):

        images = []
        image_sequence_path, label_path = self._items[idx]
        for image_path in image_sequence_path:
            image = cv2.imread(image_path, flags=-1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        images = np.concatenate(images, axis=-1)

        origin_images = images.copy()
        label = self._parsing(label_path)  # dtype을 float 으로 해야 아래 단계에서 편하다
        origin_label = label.copy()

        if self._transform:
            result = self._transform(images, label, self._itemname[idx])
            if self._test:
                # test - batch size = 1 일 때를 위함
                return result[0], result[1], result[2], torch.as_tensor(origin_images), torch.as_tensor(origin_label)
            else:
                # train, valid를 위함
                return result[0], result[1], result[2]
        else:
            return images, label, self._itemname[idx]

    def _parsing(self, path):
        json_list = []
        # json파일 parsing - 순서 -> topleft_x, topleft_y, bottomright_x, bottomright_y, center_x, center_y
        try:
            with open(path, mode='r') as json_file:
                dict = json.load(json_file)
                for i in range(len(dict["landmarkAttr"])):
                    if "attributes" in list(dict["landmarkAttr"][i].keys()):
                        xmin = int(dict["landmarkAttr"][i]["box"][0]['x'])
                        ymin = int(dict["landmarkAttr"][i]["box"][0]['y'])
                        xmax = int(dict["landmarkAttr"][i]["box"][1]['x'])
                        ymax = int(dict["landmarkAttr"][i]["box"][1]['y'])
                        category_id = dict["landmarkAttr"][i]["attributes"][0]['selected']

                        if isinstance(category_id, (list, tuple)):
                            category_id = category_id[0]

                        if category_id == "0":
                            classes = 0
                        # elif category_id == "1":
                        #     classes = 1
                        elif category_id == 0:
                            classes = 0
                        # elif category_id == 1:
                        #     classes = 1
                        elif category_id == "smoke":
                            classes = 0
                        # elif category_id == "smoke":
                        #     classes = 1
                        else:
                            xmin, ymin, xmax, ymax, classes = -1, -1, -1, -1, -1
                        json_list.append((xmin, ymin, xmax, ymax, classes))
                    else:
                        print(f"only image : {path}")
                        json_list.append((-1, -1, -1, -1, -1))
        except Exception:
            # print(f"only image or json crash : {path}")
            json_list.append((-1, -1, -1, -1, -1))
            return np.array(json_list, dtype="float32")  # 반드시 numpy여야함.
        else:
            return np.array(json_list, dtype="float32")  # 반드시 numpy여야함.

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

    sequence_number = 3
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = DetectionDataset(path=os.path.join(root, 'valid'), sequence_number=sequence_number)

    length = len(dataset)
    sequence_image, label, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('sequence image shape:', sequence_image.shape)

    if sequence_number > 1:
        sequence_image = sequence_image[:,:,3*(sequence_number-1):]
        file_name = file_name[-1]

    plot_bbox(sequence_image, label[:, :4],
              scores=None, labels=label[:, 4:5],
              class_names=dataset.classes, colors=None, reverse_rgb=True, absolute_coordinates=True,
              image_show=True, image_save=False, image_save_path="result", image_name=os.path.basename(file_name))
    '''
    images length: 1499
    sequence image shape: (720, 1280, 9)
    '''
