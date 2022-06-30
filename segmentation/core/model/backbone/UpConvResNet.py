import logging
import math
import os

import torch
import torch.nn as nn

from core.model.backbone.ResNet import get_resnet

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


class UpConvResNet(nn.Module):

    def __init__(self, base=18,
                 input_frame_number = 2,
                 deconv_channels=(256, 128, 64),
                 deconv_kernels=(4, 4, 4),
                 pretrained=True):

        super(UpConvResNet, self).__init__()
        self._resnet = get_resnet(base, pretrained=pretrained, input_frame_number=input_frame_number)
        _, in_channels , _, _ = self._resnet(torch.rand(1, input_frame_number*3, 512, 512)).shape

        upconv = []
        for out_channels, kernel in zip(deconv_channels, deconv_kernels):
            kernel, padding, output_padding = self._get_conv_argument(kernel)
            upconv.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
            upconv.append(nn.BatchNorm2d(out_channels, momentum=0.9))
            upconv.append(nn.ReLU(inplace=True))
            upconv.append(nn.ConvTranspose2d(out_channels, out_channels, kernel, stride=2, padding=padding, output_padding=output_padding, bias=False))
            upconv.append(nn.BatchNorm2d(out_channels, momentum=0.9))
            upconv.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self._upconv = nn.Sequential(*upconv)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                self._bilinear_init(m)

        logging.info(f"{self.__class__.__name__} weight init 완료")

    def _bilinear_init(self, m):

        w = m.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def _get_conv_argument(self, kernel):

        """Get the upconv configs using presets"""
        if kernel == 4:
            padding = 1
            output_padding = 0
        elif kernel == 3:
            padding = 1
            output_padding = 1
        elif kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError('Unsupported deconvolution kernel: {}'.format(kernel))
        return kernel, padding, output_padding

    def forward(self, x):
        x = self._resnet(x)
        x = self._upconv(x)
        return x


def get_upconv_resnet(base=18, pretrained=False, input_frame_number=2):
    net = UpConvResNet(base=base,
                       input_frame_number=input_frame_number,
                       # deconv_channels=(256, 128, 64, 2),
                       deconv_channels=(128, 64, 2),
                       deconv_kernels=(4, 4, 4),
                       pretrained=pretrained)
    return net

if __name__ == "__main__":

    input_size = (960, 1280)
    device = torch.device("cuda")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    net = get_upconv_resnet(base=18, pretrained=False, input_frame_number=1)
    net.to(device)
    output = net(torch.rand(1, 3, input_size[0],input_size[1], device=device))
    print(f"< input size(height, width) : {input_size} >")
    print(f"< output shape : {output.shape} >")
    '''
    < input size(height, width) : (960, 1280) >
    < output shape : torch.Size([1, 2, 240, 320]) >
    '''