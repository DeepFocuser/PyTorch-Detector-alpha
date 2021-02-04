import glob
import logging
import os
import random

import cv2
from torch.utils.data import Dataset

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


class FaceDataset(Dataset):

    """
    Parameters
    ----------
    path : str(jpg)
        Path to input image directory.
    transform : object
    """
    def __init__(self, path='Dataset/train', same_identity_per_batch=1, transform=None):
        super(FaceDataset, self).__init__()

        self._path = path
        self._name = os.path.basename(path)
        self._folder_list = glob.glob(os.path.join(path, "*"))
        self._same_identity_per_batch = same_identity_per_batch
        self._transform = transform
        self._items = []
        self._make_item_list()
        self._count = 0
        self._pin = None

    def key_func(self, path):
        return path

    def _make_item_list(self):

        if self._folder_list:
            for folder in self._folder_list:
                image_list = glob.glob(os.path.join(folder, "*"))
                for image in image_list:
                    self._items.append(image)
        else:
            logging.info("The dataset does not exist")

    def __getitem__(self, idx):

        '''
        In our experiments we sample the training data
        such that around 40 faces are selected per identity per minibatch.
        Additionally, randomly sampled negative faces are
        added to each mini-batch.
        '''

        if self._count % self._same_identity_per_batch == 0:
            self._pin = idx

        anchor_positive_folder, _ = os.path.split(self._items[self._pin])
        anchor_positive_candidate_list = glob.glob(os.path.join(anchor_positive_folder, "*"))
        anchor_path, positive_path = random.sample(anchor_positive_candidate_list, 2)

        dataset_folder, _ = os.path.split(anchor_positive_folder)
        dataset_path_list = glob.glob(os.path.join(dataset_folder, "*"))
        random.shuffle(dataset_path_list)

        for dataset_path in dataset_path_list:
            if dataset_path == anchor_positive_folder:
                continue
            else:
                negative_folder = dataset_path
                break

        negative_path = random.choice(glob.glob(os.path.join(negative_folder, "*")))

        anchor = cv2.imread(anchor_path, flags=-1)
        positive = cv2.imread(positive_path, flags=-1)
        negative = cv2.imread(negative_path, flags=-1)

        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
        positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)
        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

        self._count+=1

        if self._transform:
            return self._transform(anchor, positive, negative) + (anchor_path, positive_path, negative_path)
        else:
            return anchor, positive, negative, anchor_path, positive_path, negative_path

    def __str__(self):
        return self._name + " " + "dataset"

    def __len__(self):
        return len(self._items)


# test
if __name__ == "__main__":

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = FaceDataset(path=os.path.join('D:\CASIA-WebFace', 'valid'))

    length = len(dataset)
    anchor, positive, negative, anchor_path, positive_path, negative_path = dataset[0]
    print(f'images length: {length}')

    print(f'anchor shape: {anchor.shape}')
    print(f'anchor path: {anchor_path}\n')

    print(f'positive shape: {positive.shape}')
    print(f'positive path: {positive_path}\n')

    print(f'negative shape: {negative.shape}')
    print(f'negative path: {negative_path}')

    '''images length: 1087
    anchor shape: (250, 250, 3)
    anchor path: D:\CASIA-WebFace\valid\0004844\001.jpg
    positive shape: (250, 250, 3)
    positive path: D:\CASIA-WebFace\valid\0004844\024.jpg
    negative shape: (250, 250, 3)
    negative path: D:\CASIA-WebFace\valid\1708957\003.jpg
    '''
