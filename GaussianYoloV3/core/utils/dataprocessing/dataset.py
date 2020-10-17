import glob
import logging
import os
from xml.etree.ElementTree import parse

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸

if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

class DetectionDataset(Dataset):

    CLASSES = ['meerkat', 'otter', 'panda', 'raccoon', 'pomeranian']

    def __init__(self, path='valid', transform=None, sequence_number=1, test=False):
        super(DetectionDataset, self).__init__()

        if sequence_number < 1 and isinstance(sequence_number, float):
            logging.error(f"{sequence_number} Must be greater than 0")
            return

        self._name = os.path.basename(path)
        self._sequence_number = sequence_number
        self._image_path_List = sorted(glob.glob(os.path.join(path, "*.jpg")), key=lambda path: self.key_func(path))
        self._transform = transform
        self._items = []
        self._itemname = []
        self._test = test
        self._make_item_list()

    def key_func(self, path):
        return path

    def _make_item_list(self):

        if self._image_path_List:
            for i in range(len(self._image_path_List) - (self._sequence_number - 1)):
                image_path = self._image_path_List[i:i + self._sequence_number]
                xml_path = image_path[-1].replace(".jpg", ".xml")
                self._items.append((image_path, xml_path))
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
        xml_list = []
        try:
            tree = parse(path)
            root = tree.getroot()
            object = root.findall("object")
            for ob in object:
                if ob.find("bndbox") != None:
                    bndbox = ob.find("bndbox")
                    xmin, ymin, xmax, ymax = [int(pos.text) for i, pos in enumerate(bndbox.iter()) if i > 0]

                    # or
                    # xmin = int(bndbox.findtext("xmin"))
                    # ymin = int(bndbox.findtext("ymin"))
                    # xmax = int(bndbox.findtext("xmax"))
                    # ymax = int(bndbox.findtext("ymax"))

                    select = ob.findtext("name")
                    if select == "meerkat":
                        classes = 0
                    elif select == "otter":
                        classes = 1
                    elif select == "panda":
                        classes = 2
                    elif select == "raccoon":
                        classes = 3
                    elif select == "pomeranian":
                        classes = 4
                    else:
                        xmin, ymin, xmax, ymax, classes = -1, -1, -1, -1, -1
                    xml_list.append((xmin, ymin, xmax, ymax, classes))
                else:
                    '''
                        image만 있고 labeling 없는 데이터에 대비 하기 위함 - ssd, retinanet loss에는 아무런 영향이 없음.
                        yolo 대비용임
                    '''
                    print(f"only image : {path}")
                    xml_list.append((-1, -1, -1, -1, -1))

        except Exception:
            print(f"only image or json crash : {path}")
            xml_list.append((-1, -1, -1, -1, -1))
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.
        else:
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.

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

    plot_bbox(sequence_image, label[:, :4],
              scores=None, labels=label[:, 4:5],
              class_names=dataset.classes, colors=None, reverse_rgb=True, absolute_coordinates=True,
              image_show=True, image_save=False, image_save_path="result", image_name=os.path.basename(file_name))
    '''
    images length: 1499
    sequence image shape: (720, 1280, 9)
    '''
