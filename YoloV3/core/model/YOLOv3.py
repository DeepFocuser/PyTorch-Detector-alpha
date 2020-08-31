import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d

from core.model.backbone.ResNet import get_resnet

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

class YoloAnchorGenerator(object):

    def __init__(self, anchor, feature, stride, alloc_size):
        super(YoloAnchorGenerator, self).__init__()

        fwidth, fheight = feature
        aheight, awidth = alloc_size
        width = max(fwidth, awidth)
        height = max(fheight, aheight)

        self._anchor = np.reshape(anchor, (1, 1, -1, 2))
        self._anchor = torch.as_tensor(self._anchor)

        # grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        grid_y, grid_x = np.mgrid[:height, :width]
        offset = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)  # (13, 13, 2)
        offset = np.expand_dims(offset, axis=0)  # (1, 13, 13, 2)
        offset = np.expand_dims(offset, axis=3)  # (1, 13, 13, 1, 2)
        stride = np.reshape(stride, (1, 1, 1, 2))
        self._offset = torch.as_tensor(offset)
        self._stride = torch.as_tensor(stride)

    def __call__(self, device, dtype):
        return self._anchor.to(device = device, dtype = dtype), self._offset.to(device = device, dtype = dtype), self._stride.to(device = device, dtype = dtype)

class Yolov3(Module):

    def __init__(self, base=18,
                 input_frame_number=1,
                 input_size=(416, 416),
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=1,  # foreground만
                 pretrained=True,
                 alloc_size=(64, 64)):
        super(Yolov3, self).__init__()

        in_height, in_width = input_size
        features = []
        strides = []
        anchors = OrderedDict(anchors)
        anchors = list(anchors.values())[::-1]
        self._numoffst = len(anchors)
        self._resnet = get_resnet(base, pretrained=pretrained, input_frame_number=input_frame_number)
        output = self._resnet(torch.rand(1, input_frame_number*3, in_height, in_width))
        in_channels = []
        for out in output:
            _ , out_channel ,out_height, out_width = out.shape
            in_channels.append(out_channel)
            features.append([out_width, out_height])
            strides.append([in_width // out_width, in_height // out_height])  # w, h

        in_channels = in_channels[::-1]
        features = features[::-1]
        strides = strides[::-1]  # deep -> middle -> shallow 순으로 !!!
        self._num_classes = num_classes
        self._num_pred = 5 + num_classes  # 고정

        head_init_num_channel = 512
        trans_init_num_channel = 256

        head = []
        transition = []
        self._anchor_generators = []

        # output
        for j in range(len(anchors)):
            if j == 0:
                factor = 1
            else:
                factor = 2
                in_channels[j] = in_channels[j]*2

            head_init_num_channel = head_init_num_channel // factor
            head.append(Conv2d(in_channels[j], head_init_num_channel,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True,
                               ))
            head.append(BatchNorm2d(head_init_num_channel, eps=1e-5, momentum=0.9))
            head.append(LeakyReLU(negative_slope=0.1))

            head.append(Conv2d(head_init_num_channel, head_init_num_channel * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True))
            head.append(BatchNorm2d(head_init_num_channel * 2, eps=1e-5, momentum=0.9))
            head.append(LeakyReLU(negative_slope=0.1))

            head.append(Conv2d(head_init_num_channel * 2, head_init_num_channel,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True,
                               ))
            head.append(BatchNorm2d(head_init_num_channel, eps=1e-5, momentum=0.9))
            head.append(LeakyReLU(negative_slope=0.1))

            head.append(Conv2d(head_init_num_channel, head_init_num_channel * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True))
            head.append(BatchNorm2d(head_init_num_channel * 2, eps=1e-5, momentum=0.9))
            head.append(LeakyReLU(negative_slope=0.1))

            head.append(Conv2d(head_init_num_channel * 2, head_init_num_channel,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True,
                               ))
            head.append(BatchNorm2d(head_init_num_channel, eps=1e-5, momentum=0.9))
            head.append(LeakyReLU(negative_slope=0.1))

            head.append(Conv2d(head_init_num_channel, head_init_num_channel * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True))
            head.append(BatchNorm2d(head_init_num_channel * 2, eps=1e-5, momentum=0.9))
            head.append(LeakyReLU(negative_slope=0.1))

            head.append(Conv2d(head_init_num_channel * 2, len(anchors[j]) * self._num_pred,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=True
                               ))

        # for upsample - transition
        for i in range(len(anchors) - 1):
            if i == 0:
                factor = 1
            else:
                factor = 2
            trans_init_num_channel = trans_init_num_channel // factor
            transition.append(Conv2d(trans_init_num_channel*2, trans_init_num_channel,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     bias=True,
                                     ))
            transition.append(BatchNorm2d(trans_init_num_channel, eps=1e-5, momentum=0.9))
            transition.append(LeakyReLU(negative_slope=0.1))

        for i, anchor, feature, stride in zip(range(len(anchors)), anchors, features, strides):
            self._anchor_generators.append(
                YoloAnchorGenerator(anchor, feature, stride, (alloc_size[0] * (2 ** i), alloc_size[1] * (2 ** i))))

        self._head = Sequential(*head)
        self._transition = Sequential(*transition)

        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.normal_(m.weight, mean=0., std=0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        logging.info(f"{self.__class__.__name__} Head weight init 완료")

    def forward(self, x):

        feature_36, feature_61, feature_74 = self._resnet(x)
        # first

        transition = self._head[:15](feature_74)  # darknet 기준 75 ~ 79
        output82 = self._head[15:19](transition)  # darknet 기준 79 ~ 82

        # second
        transition = self._transition[0:3](transition)

        transition = torch.nn.functional.interpolate(transition, scale_factor=2, mode='nearest')

        transition = torch.cat((transition, feature_61), dim=1)

        transition = self._head[19:34](transition)  # darknet 기준 75 ~ 91

        output94 = self._head[34:38](transition)  # darknet 기준 91 ~ 82

        # third
        transition = self._transition[3:](transition)

        transition = torch.nn.functional.interpolate(transition, scale_factor=2, mode='nearest')

        transition = torch.cat((transition, feature_36), dim=1)
        output106 = self._head[38:](transition)  # darknet 기준 91 ~ 106

        output82 = output82.permute(0, 2, 3, 1)
        output94 = output94.permute(0, 2, 3, 1)
        output106 = output106.permute(0, 2, 3, 1)

        # (batch size, height, width, len(anchors), (5 + num_classes)
        anchors = []
        offsets = []
        strides = []

        for i in range(self._numoffst):
            anchor, offset, stride = self._anchor_generators[i](x.device, x.dtype)
            anchors.append(anchor)
            offsets.append(offset)
            strides.append(stride)

        return output82, output94, output106, \
               anchors[0], anchors[1], anchors[2], \
               offsets[0], offsets[1], offsets[2], \
               strides[0], strides[1], strides[2],


if __name__ == "__main__":

    input_size = (608, 608)
    device = torch.device("cuda")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net = Yolov3(base=34,
                 input_frame_number=1,
                 input_size=input_size,
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=5,  # foreground만
                 pretrained=False,
                 alloc_size=(64, 64))
    net.to(device)
    output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(torch.rand(1, 3, input_size[0],input_size[1], device=device))
    print(f"< input size(height, width) : {input_size} >")
    for i, pred in enumerate([output1, output2, output3]):
        print(f"prediction {i + 1} : {pred.shape}")
    for i, anchor in enumerate([anchor1, anchor2, anchor3]):
        print(f"anchor {i + 1} w, h 순서 : {anchor.shape}")
    for i, offset in enumerate([offset1, offset2, offset3]):
        print(f"offset {i + 1} w, h 순서 : {offset.shape}")
    for i, stride in enumerate([stride1, stride2, stride3]):
        print(f"stride {i + 1} w, h 순서 : {stride.shape}")
    '''
    < input size(height, width) : (608, 608) >
    prediction 1 : (1, 19, 19, 30)
    prediction 2 : (1, 38, 38, 30)
    prediction 3 : (1, 76, 76, 30)
    anchor 1 w, h 순서 : (1, 1, 3, 2)
    anchor 2 w, h 순서 : (1, 1, 3, 2)
    anchor 3 w, h 순서 : (1, 1, 3, 2)
    offset 1 w, h 순서 : (1, 64, 64, 1, 2)
    offset 2 w, h 순서 : (1, 128, 128, 1, 2)
    offset 3 w, h 순서 : (1, 256, 256, 1, 2)
    stride 1 w, h 순서 : (1, 1, 1, 2)
    stride 2 w, h 순서 : (1, 1, 1, 2)
    stride 3 w, h 순서 : (1, 1, 1, 2)
    '''
