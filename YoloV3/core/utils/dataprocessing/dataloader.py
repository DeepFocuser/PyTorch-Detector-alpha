import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from core.utils.dataprocessing.dataset import DetectionDataset
from core.utils.dataprocessing.transformer import YoloTrainTransform, YoloValidTransform


class Tuple(object):

    def __init__(self, fn, *args, dataset = None, interval = 10, train_transform=None):

        self._counter = 0
        self._dataset = dataset
        self._interval = interval
        self._train_transform = train_transform
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn,) + args
        for i, ele_fn in enumerate(self._fn):
            assert hasattr(ele_fn, '__call__'), 'Batchify functions must be callable! ' \
                                                'type(fn[%d]) = %s' % (i, str(type(ele_fn)))

    def __call__(self, data):

        self._counter+=1
        if self._interval == self._counter:
            train_transform = random.choice(self._train_transform)
        else:
            train_transform = self._train_transform[-1] # 원본사이즈 transform을 마지막 리스트의 요소로 놓기
        data_transform =[train_transform(ele) for ele in data]

        assert len(data_transform[0]) == len(self._fn), \
            'The number of attributes in each data sample should contains' \
            ' {} elements, given {}.'.format(len(self._fn), len(data_transform[0]))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data_transform]))

        if self._counter >= len(self._dataset) // len(data):
            self._counter = 0

        return ret


def _pad_arrs_to_max_length(arrs, pad_axis, pad_val):
    if not isinstance(arrs[0], (torch.Tensor, np.ndarray)):
        arrs = [np.asarray(ele) for ele in arrs]
    original_length = [ele.shape[pad_axis] for ele in arrs]
    max_size = max(original_length)
    ret_shape = list(arrs[0].shape)
    ret_shape[pad_axis] = max_size
    ret_shape = (len(arrs),) + tuple(ret_shape)
    ret = torch.full(size=ret_shape, fill_value=pad_val, dtype=arrs[0].dtype)
    original_length = torch.as_tensor(original_length, dtype=torch.int32)

    # arrs -> (batch, max object number, 5)
    for i, arr in enumerate(arrs):
        if arr.shape[pad_axis] == max_size:
            ret[i] = arr
        else:
            ret[i:i + 1, 0:arr.shape[pad_axis], :] = arr
    return ret, original_length


class Pad(object):

    def __init__(self, axis=0, pad_val=0, ret_length=False):
        self._axis = axis
        assert isinstance(axis, int), 'axis must be an integer! ' \
                                      'Received axis=%s, type=%s.' % (str(axis),
                                                                      str(type(axis)))
        self._pad_val = pad_val
        self._ret_length = ret_length

    def __call__(self, data):

        if isinstance(data[0], (torch.Tensor, np.ndarray, list)):
            padded_arr, original_length = _pad_arrs_to_max_length(data, self._axis,
                                                                  self._pad_val)
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError


class Stack(object):

    def __call__(self, batch):
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, dim=0)
        elif isinstance(batch[0], str):  # str
            return batch
        else:
            out = np.asarray(batch)
            return torch.as_tensor(out)

def traindataloader(multiscale=False, factor_scale=[10, 9], augmentation=True, path="Dataset/train",
                    input_size=(512, 512), input_frame_number=2, batch_size=8, pin_memory=True, batch_interval=10, num_workers=4, shuffle=True,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    num_workers = 0 if pin_memory else num_workers

    dataset = DetectionDataset(path=path, sequence_number=input_frame_number, test=False)

    if multiscale:
        init = factor_scale[0]
        end = init + factor_scale[1] + 1
        train_transform = [YoloTrainTransform(x * 32, x * 32, input_frame_number=input_frame_number, mean=mean, std=std, augmentation = augmentation) for x in range(init, end)]
    else:
        train_transform = [YoloTrainTransform(input_size[0], input_size[1],
                                              input_frame_number = input_frame_number,
                                              mean=mean, std=std,
                                              augmentation=augmentation)]

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Tuple(Stack(),
                         Pad(pad_val=-1),
                         Stack(),

                         # multiscale을 위한 구현
                         dataset = dataset,
                         interval = batch_interval,
                         train_transform = train_transform),
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers)

    return dataloader, dataset

def validdataloader(path="Dataset/valid",
                    input_size=(512, 512), input_frame_number=2, batch_size=8, pin_memory=True, num_workers=4, shuffle=True,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    num_workers = 0 if pin_memory else num_workers

    transform = YoloValidTransform(input_size[0], input_size[1], input_frame_number, mean=mean, std=std)
    dataset = DetectionDataset(path=path, transform=transform, sequence_number=input_frame_number, test=False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Tuple(Stack(),
                         Pad(pad_val=-1),
                         Stack()),
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return dataloader, dataset

def testdataloader(path="Dataset/test", input_size=(512, 512), input_frame_number=2, pin_memory=True,
                   num_workers=4, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    num_workers = 0 if pin_memory else num_workers

    transform = YoloValidTransform(input_size[0], input_size[1], input_frame_number=input_frame_number, mean=mean, std=std)
    dataset = DetectionDataset(path=path, transform=transform, sequence_number=input_frame_number, test=True)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=Tuple(Stack(),
                         Pad(pad_val=-1),
                         Stack(),
                         Stack(),
                         Pad(pad_val=-1)),
        pin_memory=pin_memory,
        num_workers=num_workers)
    return dataloader, dataset


# test
if __name__ == "__main__":
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataloader, dataset = traindataloader(path=os.path.join(root, 'valid'),
                                          input_size=(320, 640))

    # for문 돌리기 싫으므로, iterator로 만든
    dataloader_iter = iter(dataloader)
    data, label, name = next(dataloader_iter)

    # 첫번째 이미지만 가져옴
    image = data[0]
    label = label[0]
    name = name[0]

    print(f"image shape : {image.shape}")
    print(f"label shape : {label.shape}")
    print(f"name : {name}")

