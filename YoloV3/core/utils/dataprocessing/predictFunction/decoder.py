import torch
from torch.nn import Module

class Decoder(Module):

    def __init__(self, from_sigmoid=False, num_classes=5, thresh=0.01, multiperclass=True):
        super(Decoder, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._num_classes = num_classes
        self._num_pred = 5 + num_classes
        self._thresh = thresh
        self._multiperclass = multiperclass

    def forward(self, output, anchor, offset, stride):

        # 자르기
        batch_size, h, w, ac = output.shape
        out = output.reshape((batch_size, h*w, -1, self._num_pred))  # (b, 169, 3, 10)

        xy_pred = out[:, :, :, 0:2] # (b, 169, 3, 2)
        wh_pred = out[:, :, :, 2:4] # (b, 169, 3, 2)
        objectness = out[:, :, :, 4:5] # (b, 169, 3, 1)
        class_pred = out[:, :, :, 5:]  # (b, 169, 3, 5)

        if not self._from_sigmoid:
            xy_pred = torch.sigmoid(xy_pred)
            objectness = torch.sigmoid(objectness)
            class_pred = torch.sigmoid(class_pred)

        # 복구하기
        '''
        offset이 output에 따라 변하는 값이기 때문에, 
        네트워크에서 출력할 때 충분히 크게 만들면,
        c++에서 inference 할 때 어떤 값을 넣어도 정상적으로 동작하게 된다. 
        '''
        offset = offset[:, :w, :h, :, :]
        offset = offset.reshape((1, -1, 1, 2))

        xy_preds = torch.mul(torch.add(xy_pred, offset), stride)
        wh_preds = torch.mul(torch.exp(wh_pred), anchor)
        class_pred = torch.mul(class_pred, objectness)  # (b, 169, 3, 5)

        # center to corner
        wh = torch.true_divide(wh_preds, 2.0)
        bbox = torch.cat([xy_preds - wh, xy_preds + wh], dim=-1)  # (b, 169, 3, 4)

        # prediction per class
        if self._multiperclass:

            bbox = bbox.unsqueeze(dim=0)
            bbox = torch.repeat_interleave(bbox, self._num_classes, dim=0)  # (5, b, 169, 3, 4)
            class_pred = class_pred.permute(3, 0, 1, 2).unsqueeze(dim=-1)  # (5, b, 169, 3, 1)

            id = torch.add(class_pred * 0,
                           torch.arange(0, self._num_classes, device=class_pred.device).reshape(
                               (self._num_classes, 1, 1, 1, 1)))  # (5, b, 169, 3, 1)

            # ex) thresh=0.01 이상인것만 뽑기
            mask = class_pred > self._thresh
            id = torch.where(mask, id, torch.ones_like(id) * -1)
            score = torch.where(mask, class_pred, torch.zeros_like(class_pred))

            # reshape to (b, -1, 6)
            results = torch.cat([id, score, bbox], dim=-1)  # (5, b, 169, 3, 6)
            results = results.permute(1, 0, 2, 3, 4) # (5, b, 169, 3, 6) -> (b, 5, 169, 3, 6)
        else:  # prediction multiclass
            class_pred, id = torch.max(class_pred, dim = -1, keepdim=True)
            id = id.to(class_pred.dtype)
            # ex) thresh=0.01 이상인것만 뽑기
            mask = class_pred > self._thresh
            id = torch.where(mask, id, torch.ones_like(id) * -1)
            score = torch.where(mask, class_pred, torch.zeros_like(class_pred))

            results = torch.cat([id, score, bbox], dim=-1)  # (b, 169, 3, 6)

        return torch.reshape(results, shape=(batch_size, -1, 6))  # (b, -1, 6)


# test
if __name__ == "__main__":
    from core import Yolov3, YoloTrainTransform, DetectionDataset
    import os

    input_size = (608, 608)
    device = torch.device("cuda")
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

    transform = YoloTrainTransform(input_size[0], input_size[1])
    dataset = DetectionDataset(path='/home/jg/Desktop/mountain/valid', transform=transform)
    num_classes = dataset.num_class

    image, label, _ = dataset[0]

    net = Yolov3(base=18,
                 input_frame_number=1,
                 input_size=input_size,
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=num_classes,  # foreground만
                 pretrained=False,)

    net.to(device)
    # batch 형태로 만들기
    image = image[None,:,:]
    label = label[None,:,:]

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
        image.to(device))

    results = []
    decoder = Decoder(from_sigmoid=False, num_classes=num_classes, thresh=0.01, multiperclass=False)
    for out, an, off, st in zip([output1, output2, output3], [anchor1, anchor2, anchor3], [offset1, offset2, offset3],
                                [stride1, stride2, stride3]):
        results.append(decoder(out, an, off, st))
    results = torch.cat(results, dim=1)
    print(f"decoder shape : {results.shape}")
    '''
    multiperclass=True 일 때 
    decoder shape : torch.Size([1, 10647, 6])
    
    multiperclass=False 일 때 
    decoder shape : torch.Size([1, 10647, 6])
    '''
