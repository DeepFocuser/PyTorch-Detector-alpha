import numpy as np
import torch
from torch.utils.data import DataLoader

from core.utils.dataprocessing.dataset import DetectionDataset
from core.utils.dataprocessing.transformer import CenterTrainTransform, CenterValidTransform


class Tuple(object):

    def __init__(self, fn, *args):
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

        assert len(data[0]) == len(self._fn), \
            'The number of attributes in each data sample should contains' \
            ' {} elements, given {}.'.format(len(self._fn), len(data[0]))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
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


def traindataloader(augmentation=True, path="Dataset/train",
                    input_size=(512, 512), input_frame_number=2, batch_size=8, pin_memory=True, num_workers=4, shuffle=True,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale_factor=4, make_target=True):

    num_workers = 0 if pin_memory else num_workers

    transform = CenterTrainTransform(input_size, input_frame_number=input_frame_number, mean=mean, std=std, scale_factor=scale_factor,
                                     augmentation=augmentation, make_target=make_target,
                                     num_classes=DetectionDataset(path=path).num_class)
    dataset = DetectionDataset(path=path, transform=transform, sequence_number=input_frame_number)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Tuple(Stack(),
                         Pad(pad_val=-1),
                         Stack(),
                         Stack(),
                         Stack(),
                         Stack(),
                         Stack(),
                         Stack(),
                         Stack()),
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=num_workers)

    return dataloader, dataset


def validdataloader(path="Dataset/valid", input_size=(512, 512), input_frame_number=1,
                    batch_size=1, pin_memory=True, num_workers=4, shuffle=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                    scale_factor=4, make_target=True):

    num_workers = 0 if pin_memory else num_workers

    transform = CenterValidTransform(input_size, input_frame_number=input_frame_number, mean=mean, std=std, scale_factor=scale_factor, make_target=make_target,
                                     num_classes=DetectionDataset(path=path).num_class)
    dataset = DetectionDataset(path=path, transform=transform, sequence_number=input_frame_number)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Tuple(Stack(),
                         Pad(pad_val=-1),
                         Stack(),
                         Stack(),
                         Stack(),
                         Stack(),
                         Stack(),
                         Stack(),
                         Stack()),
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers)

    return dataloader, dataset


def testdataloader(path="Dataset/test", input_size=(512, 512), input_frame_number=2, pin_memory=True,
                   num_workers=4, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale_factor=4):

    num_workers = 0 if pin_memory else num_workers

    transform = CenterValidTransform(input_size, input_frame_number=input_frame_number, mean=mean, std=std, scale_factor=scale_factor, make_target=False)
    dataset = DetectionDataset(path=path, transform=transform, sequence_number=input_frame_number)

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

    '''
    https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader
    It is generally not recommended to return CUDA tensors in multi-process loading because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing 
    (see CUDA in multiprocessing). Instead, we recommend using automatic memory pinning (i.e., setting pin_memory=True),
    which enables fast data transfer to CUDA-enabled GPUs.
    '''
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataloader, dataset = validdataloader(path=os.path.join(root, 'Dataset_WIDER', "valid"), input_size=(512, 512),
                                          batch_size=4, pin_memory=True, num_workers=4, shuffle=True, mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    # for문 돌리기 싫으므로, iterator로 만든
    dataloader_iter = iter(dataloader)
    data, label, _, _, _, _, _, _, name = next(dataloader_iter)

    print(f"images shape : {data.shape}")
    print(f"labels shape : {label.shape}")
    print(f"name : {name}")

    '''
    images shape : torch.Size([4, 3, 512, 512])
    labels shape : torch.Size([4, 9, 15])
    name : ['1_Handshaking_Handshaking_1_313', '13_Interview_Interview_Sequences_13_40', '4_Dancing_Dancing_4_253', '56_Voter_peoplevoting_56_819']
    '''