from torch.utils.data import DataLoader

from core.utils.dataprocessing.dataset import DetectionDataset
from core.utils.dataprocessing.transformer import CenterTrainTransform, CenterValidTransform


def traindataloader(augmentation=True, path="Dataset/train",
                    input_size=(512, 512), input_frame_number=1, batch_size=8, pin_memory=True, num_workers=4, shuffle=True,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    num_workers = 0 if pin_memory else num_workers

    transform = CenterTrainTransform(input_size, input_frame_number=input_frame_number, mean=mean, std=std,
                                     augmentation=augmentation)
    dataset = DetectionDataset(path=path, transform=transform, sequence_number=input_frame_number)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=num_workers)

    return dataloader, dataset


def validdataloader(path="Dataset/valid", input_size=(512, 512), input_frame_number=1,
                    batch_size=1, pin_memory=True, num_workers=4, shuffle=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    num_workers = 0 if pin_memory else num_workers

    transform = CenterValidTransform(input_size, input_frame_number=input_frame_number, mean=mean, std=std)
    dataset = DetectionDataset(path=path, transform=transform, sequence_number=input_frame_number)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers)

    return dataloader, dataset


def testdataloader(path="Dataset/test", input_size=(512, 512), input_frame_number=1, pin_memory=True,
                   num_workers=4, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    num_workers = 0 if pin_memory else num_workers

    transform = CenterValidTransform(input_size, input_frame_number=input_frame_number, mean=mean, std=std)
    dataset = DetectionDataset(path=path, transform=transform, sequence_number=input_frame_number, test=True)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
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
    dataloader, dataset = validdataloader(path=os.path.join(root, 'Dataset', "valid"), input_size=(256, 256),
                                          batch_size=8, pin_memory=True, num_workers=4, shuffle=True, mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    # for문 돌리기 싫으므로, iterator로 만든
    dataloader_iter = iter(dataloader)
    data, label, name = next(dataloader_iter)

    print(f"images shape : {data.shape}")
    print(f"labels shape : {label.shape}")
    print(f"name : {name}")

    '''
    images shape : torch.Size([8, 3, 256, 256])
    labels shape : torch.Size([8, 2])
    name : ['2.tif', 'ng_03.tif', 'ng_04.tif', '14.tif', '5.tif', '4.tif', '0.tif', '3.tif']
    '''