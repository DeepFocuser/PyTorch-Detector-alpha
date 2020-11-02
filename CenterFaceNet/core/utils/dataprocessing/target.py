import numpy as np
import torch
import torch.nn as nn


# object size-adaptive standard deviation 구하기
# https://en.wikipedia.org/wiki/Gaussian_function
def gaussian_radius(height=512, width=512, min_overlap=0.7):
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)

    temp = max(0, b1 ** 2 - 4 * a1 * c1)
    sq1 = np.sqrt(temp)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height

    temp = max(0, b2 ** 2 - 4 * a2 * c2)
    sq2 = np.sqrt(temp)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height

    temp = max(0, b3 ** 2 - 4 * a3 * c3)
    sq3 = np.sqrt(temp)
    r3 = (b3 + sq3) / 2

    return max(0, int(min(r1, r2, r3)))


def gaussian_2d(shape=(10, 10), sigma=1):
    m, n = [s // 2 for s in shape]
    y, x = np.mgrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h


def draw_gaussian(heatmap, center_x, center_y, radius, k=1):
    diameter = 2 * radius + 1  # 홀수
    gaussian = gaussian_2d(shape=(diameter, diameter), sigma=diameter / 6)

    # 경계선에서 어떻게 처리 할지
    height, width = heatmap.shape[0:2]
    left, right = min(center_x, radius), min(width - center_x, radius + 1)
    top, bottom = min(center_y, radius), min(height - center_y, radius + 1)

    masked_heatmap = heatmap[center_y - top: center_y + bottom, center_x - left: center_x + right]
    masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
    '''
    https://rfriend.tistory.com/290
    Python Numpy의 배열 indexing, slicing에서 유의해야할 것이 있다. 
    배열을 indexing 해서 얻은 객체는 복사(copy)가 된 독립된 객체가 아니며, 
    단지 원래 배열의 view 일 뿐이라는 점이다.  
    따라서 view를 새로운 값으로 변경시키면 원래의 배열의 값도 변경이 된다.
    따라서 아래와 같은 경우 masked_heatmap은 view 일뿐.
    '''
    # inplace 연산
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


# https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
class TargetGenerator(nn.Module):

    def __init__(self, num_classes=3):
        super(TargetGenerator, self).__init__()
        self._num_classes = num_classes

    def forward(self, gt_boxes, gt_ids, gt_landmarks, output_width, output_height, device):

        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.detach().cpu().numpy()
        if isinstance(gt_landmarks, torch.Tensor):
            gt_landmarks = gt_landmarks.detach().cpu().numpy()
        if isinstance(gt_ids, torch.Tensor):
            gt_ids = gt_ids.detach().cpu().numpy()

        batch_size = gt_boxes.shape[0]
        heatmap = np.zeros((batch_size, self._num_classes, output_height, output_width),
                           dtype=np.float32)
        offset_target = np.zeros((batch_size, 2, output_height, output_width), dtype=np.float32)
        wh_target = np.zeros((batch_size, 2, output_height, output_width), dtype=np.float32)

        '''
            for face five(x,y) points landmark
            중심으로부터의 offset을 계산하자
        '''
        landmark_target = np.zeros((batch_size, 2, output_height, output_width), dtype=np.float32)
        mask_target = np.zeros((batch_size, 2, output_height, output_width), dtype=np.float32)
        landmark_mask_target = np.zeros((batch_size, 2, output_height, output_width), dtype=np.float32)

        for batch, gt_box, gt_id, gt_landmark in zip(range(len(gt_boxes)), gt_boxes, gt_ids, gt_landmarks):
            for bbox, id, landmark in zip(gt_box, gt_id, gt_landmark):

                # background인 경우
                if bbox[0] == -1 or bbox[1] == -1 or bbox[2] == -1 or bbox[3] == -1 or id == -1:
                    continue

                box_h, box_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                center = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    dtype=np.float32)
                center_int = center.astype(np.int32)
                center_x, center_y = center_int

                # data augmentation으로 인해 범위가 넘어갈수 가 있음.
                center_x = np.clip(center_x, 0, output_width - 1)
                center_y = np.clip(center_y, 0, output_height - 1)

                landmark1 = landmark[:2]
                landmark2 = landmark[2:4]
                landmark3 = landmark[4:6]
                landmark4 = landmark[6:8]
                landmark5 = landmark[8:10]

                landmark1_int = landmark1.astype(np.int32)
                landmark2_int = landmark2.astype(np.int32)
                landmark3_int = landmark3.astype(np.int32)
                landmark4_int = landmark4.astype(np.int32)
                landmark5_int = landmark5.astype(np.int32)

                landmark1_x, landmark1_y = landmark1_int
                landmark2_x, landmark2_y = landmark2_int
                landmark3_x, landmark3_y = landmark3_int
                landmark4_x, landmark4_y = landmark4_int
                landmark5_x, landmark5_y = landmark5_int

                landmark1_y = np.clip(landmark1_y, 0, output_height - 1)
                landmark1_x = np.clip(landmark1_x, 0, output_width - 1)
                landmark2_y = np.clip(landmark2_y, 0, output_height - 1)
                landmark2_x = np.clip(landmark2_x, 0, output_width - 1)
                landmark3_y = np.clip(landmark3_y, 0, output_height - 1)
                landmark3_x = np.clip(landmark3_x, 0, output_width - 1)
                landmark4_y = np.clip(landmark4_y, 0, output_height - 1)
                landmark4_x = np.clip(landmark4_x, 0, output_width - 1)
                landmark5_y = np.clip(landmark5_y, 0, output_height - 1)
                landmark5_x = np.clip(landmark5_x, 0, output_width - 1)

                # heatmap
                # C:\ProgramData\Anaconda3\Lib\site-packages\gluoncv\model_zoo\center_net\target_generator.py
                radius = gaussian_radius(height=box_h, width=box_w)
                radius = max(0, int(radius))

                # 가우시안 그리기 - inplace 연산(np.maximum)
                draw_gaussian(heatmap[batch, int(id), ...], center_x, center_y,
                              radius)
                # wh
                box = np.array([box_w, box_h], dtype=np.float32)
                wh_target[batch, :, center_y, center_x] = box

                # center offset
                offset_target[batch, :, center_y, center_x] = center - center_int

                # landmark - center
                landmark_target[batch, :, landmark1_y, landmark1_x] = center - landmark1
                landmark_target[batch, :, landmark2_y, landmark2_x] = center - landmark2
                landmark_target[batch, :, landmark3_y, landmark3_x] = center - landmark3
                landmark_target[batch, :, landmark4_y, landmark4_x] = center - landmark4
                landmark_target[batch, :, landmark5_y, landmark5_x] = center - landmark5

                # mask
                mask_target[batch, :, center_y, center_x] = 1.0

                # landmark mask
                landmark_mask_target[batch, :, landmark1_y, landmark1_x] = 1.0
                landmark_mask_target[batch, :, landmark2_y, landmark2_x] = 1.0
                landmark_mask_target[batch, :, landmark3_y, landmark3_x] = 1.0
                landmark_mask_target[batch, :, landmark4_y, landmark4_x] = 1.0
                landmark_mask_target[batch, :, landmark5_y, landmark5_x] = 1.0

        return tuple([torch.as_tensor(ele, device=device) for ele in (heatmap, offset_target, wh_target, landmark_target, mask_target, landmark_mask_target)])


# test
if __name__ == "__main__":

    from core.utils.dataprocessing.dataset import DetectionDataset
    from core.utils.dataprocessing.transformer import CenterTrainTransform
    import os

    input_size = (768, 1280)
    scale_factor = 4
    sequence_number = 1
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = CenterTrainTransform(input_size, input_frame_number=sequence_number, mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225),
                                     scale_factor=4)
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'valid'), transform=transform, sequence_number=sequence_number)

    num_classes = dataset.num_class
    image, label, _, _, _ = dataset[0]
    targetgenerator = TargetGenerator(num_classes=num_classes)

    # batch 형태로 만들기
    label = label[None,:, :]
    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    gt_landmarks = label[:, :, 5:]
    heatmap_target, offset_target, wh_target, landmark_target, mask_target, landmark_mask_target = targetgenerator(gt_boxes, gt_ids, gt_landmarks,
                                                                                                                   input_size[1] // scale_factor,
                                                                                                                   input_size[0] // scale_factor, image.device)

    print(f"heatmap_targets shape : {heatmap_target.shape}")
    print(f"offset_targets shape : {offset_target.shape}")
    print(f"wh_targets shape : {wh_target.shape}")
    print(f"landmark_target shape : {landmark_target.shape}")
    print(f"mask_targets shape : {mask_target.shape}")
    print(f"landmark_mask_target shape : {landmark_mask_target.shape}")
    '''
    heatmap_targets shape : torch.Size([1, 1, 192, 320])
    offset_targets shape : torch.Size([1, 2, 192, 320])
    wh_targets shape : torch.Size([1, 2, 192, 320])
    landmark_target shape : torch.Size([1, 2, 192, 320])
    mask_targets shape : torch.Size([1, 2, 192, 320])
    landmark_mask_target shape : torch.Size([1, 2, 192, 320])
    '''
