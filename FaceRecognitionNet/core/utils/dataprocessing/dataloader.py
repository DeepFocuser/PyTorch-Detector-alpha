from torch.utils.data import DataLoader

from core.utils.dataprocessing.dataset import FaceDataset
from core.utils.dataprocessing.transformer import CenterTrainTransform, CenterValidTransform


def traindataloader(augmentation=True, path="Dataset/train",
                    input_size=(512, 512), batch_size=8, pin_memory=True, num_workers=4, shuffle=True,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    num_workers = 0 if pin_memory else num_workers

    transform = CenterTrainTransform(input_size, mean=mean, std=std,
                                     augmentation=augmentation)
    dataset = FaceDataset(path=path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=num_workers)

    return dataloader, dataset


def validdataloader(path="Dataset/valid", input_size=(512, 512), batch_size=1, pin_memory=True, num_workers=4, shuffle=True,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    num_workers = 0 if pin_memory else num_workers

    transform = CenterValidTransform(input_size, mean=mean, std=std)
    dataset = FaceDataset(path=path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers)

    return dataloader, dataset


def testdataloader(path="Dataset/test", input_size=(512, 512), pin_memory=True,
                   num_workers=4, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    num_workers = 0 if pin_memory else num_workers

    transform = CenterValidTransform(input_size, mean=mean, std=std)
    dataset = FaceDataset(path=path, transform=transform)

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
    # dataloader, dataset = validdataloader(path=os.path.join(root, 'Dataset', "valid"), input_size=(256, 256),
    #                                       batch_size=8, pin_memory=True, num_workers=4, shuffle=True, mean=[0.485, 0.456, 0.406],
    #                                       std=[0.229, 0.224, 0.225])

    dataloader, dataset = validdataloader(path=os.path.join('D:\\CASIA-WebFace', 'valid'), input_size=(256, 256),
                                          batch_size=8, pin_memory=True, num_workers=4, shuffle=True, mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    # for문 돌리기 싫으므로, iterator로 만든
    dataloader_iter = iter(dataloader)
    anchor, positive, negative, anchor_path, positive_path, negative_path = next(dataloader_iter)

    print(f'anchor shape: {anchor.shape}')
    print(f'anchor path: {anchor_path}\n')

    print(f'positive shape: {positive.shape}')
    print(f'positive path: {positive_path}\n')

    print(f'negative shape: {negative.shape}')
    print(f'negative path: {negative_path}')

    '''
    anchor shape: torch.Size([8, 3, 256, 256])
    anchor path: ['D:\\CASIA-WebFace\\valid\\0740535\\051.jpg', 'D:\\CASIA-WebFace\\valid\\1706767\\279.jpg', 'D:\\CASIA-WebFace\\valid\\6234845\\016.jpg', 'D:\\CASIA-WebFace\\valid\\0740535\\033.jpg', 'D:\\CASIA-WebFace\\valid\\5559738\\022.jpg', 'D:\\CASIA-WebFace\\valid\\0740535\\025.jpg', 'D:\\CASIA-WebFace\\valid\\1706767\\257.jpg', 'D:\\CASIA-WebFace\\valid\\6152976\\013.jpg']
    
    positive shape: torch.Size([8, 3, 256, 256])
    positive path: ['D:\\CASIA-WebFace\\valid\\0740535\\005.jpg', 'D:\\CASIA-WebFace\\valid\\1706767\\131.jpg', 'D:\\CASIA-WebFace\\valid\\6234845\\017.jpg', 'D:\\CASIA-WebFace\\valid\\0740535\\027.jpg', 'D:\\CASIA-WebFace\\valid\\5559738\\007.jpg', 'D:\\CASIA-WebFace\\valid\\0740535\\022.jpg', 'D:\\CASIA-WebFace\\valid\\1706767\\260.jpg', 'D:\\CASIA-WebFace\\valid\\6152976\\035.jpg']
    
    negative shape: torch.Size([8, 3, 256, 256])
    negative path: ['D:\\CASIA-WebFace\\valid\\5657590\\020.jpg', 'D:\\CASIA-WebFace\\valid\\0004883\\036.jpg', 'D:\\CASIA-WebFace\\valid\\5559738\\020.jpg', 'D:\\CASIA-WebFace\\valid\\1701859\\005.jpg', 'D:\\CASIA-WebFace\\valid\\6252408\\020.jpg', 'D:\\CASIA-WebFace\\valid\\6234845\\003.jpg', 'D:\\CASIA-WebFace\\valid\\5476978\\021.jpg', 'D:\\CASIA-WebFace\\valid\\3848020\\009.jpg']
    '''