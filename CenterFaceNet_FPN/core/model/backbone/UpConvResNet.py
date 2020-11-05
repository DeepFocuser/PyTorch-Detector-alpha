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

        in_channels_list = []
        for layer in self._resnet(torch.rand(1, input_frame_number*3, 512, 512)):
            _, in_channels , _, _ = layer.shape
            in_channels_list.append(in_channels)

        upconv1 = []
        upconv2 = []
        upconv3 = []

        transition_conv1 = []
        transition_conv2 = []
        transition_conv3 = []

        #transition1
        transition_conv1.append(nn.Conv2d(in_channels_list[2], deconv_channels[0],
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True,
                                          ))
        transition_conv1.append(nn.BatchNorm2d(deconv_channels[0], eps=1e-5, momentum=0.9, track_running_stats=False))
        transition_conv1.append(nn.ReLU(inplace=True))

        #transition2
        transition_conv2.append(nn.Conv2d(in_channels_list[1], deconv_channels[1],
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True,
                                          ))
        transition_conv2.append(nn.BatchNorm2d(deconv_channels[1], eps=1e-5, momentum=0.9, track_running_stats=False))
        transition_conv2.append(nn.ReLU(inplace=True))

        #transition3
        transition_conv3.append(nn.Conv2d(in_channels_list[0], deconv_channels[2],
                                          kernel_size=1,
                                          stride=1,
                                          padding=0,
                                          bias=True,
                                          ))
        transition_conv3.append(nn.BatchNorm2d(deconv_channels[2], eps=1e-5, momentum=0.9, track_running_stats=False))
        transition_conv3.append(nn.ReLU(inplace=True))


        # unconv1
        kernel, padding, output_padding = self._get_conv_argument(deconv_kernels[0])
        out_channels = deconv_channels[0]
        upconv1.append(nn.Conv2d(in_channels_list[-1], out_channels, 3, stride=1, padding=1, bias=False))
        upconv1.append(nn.BatchNorm2d(out_channels, momentum=0.9))
        upconv1.append(nn.ReLU(inplace=True))
        upconv1.append(nn.ConvTranspose2d(out_channels, out_channels, kernel, stride=2, padding=padding,
                                          output_padding=output_padding, bias=False))
        upconv1.append(nn.BatchNorm2d(out_channels, momentum=0.9))
        upconv1.append(nn.ReLU(inplace=True))

        in_channels = out_channels*2

        # unconv2
        kernel, padding, output_padding = self._get_conv_argument(deconv_kernels[1])
        out_channels = deconv_channels[1]
        upconv2.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
        upconv2.append(nn.BatchNorm2d(out_channels, momentum=0.9))
        upconv2.append(nn.ReLU(inplace=True))
        upconv2.append(nn.ConvTranspose2d(out_channels, out_channels, kernel, stride=2, padding=padding,
                                          output_padding=output_padding, bias=False))
        upconv2.append(nn.BatchNorm2d(out_channels, momentum=0.9))
        upconv2.append(nn.ReLU(inplace=True))

        in_channels = out_channels*2

        # unconv3
        kernel, padding, output_padding = self._get_conv_argument(deconv_kernels[2])
        out_channels = deconv_channels[2]
        upconv3.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
        upconv3.append(nn.BatchNorm2d(out_channels, momentum=0.9))
        upconv3.append(nn.ReLU(inplace=True))
        upconv3.append(nn.ConvTranspose2d(out_channels, out_channels, kernel, stride=2, padding=padding,
                                          output_padding=output_padding, bias=False))
        upconv3.append(nn.BatchNorm2d(out_channels, momentum=0.9))
        upconv3.append(nn.ReLU(inplace=True))

        self._upconv1 = nn.Sequential(*upconv1)
        self._upconv2 = nn.Sequential(*upconv2)
        self._upconv3 = nn.Sequential(*upconv3)
        self._transition_conv1 = nn.Sequential(*transition_conv1)
        self._transition_conv2 = nn.Sequential(*transition_conv2)
        self._transition_conv3 = nn.Sequential(*transition_conv3)

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

        layer1, layer2, layer3, layer4 = self._resnet(x)

        transition_conv1 = self._transition_conv1(layer3)

        upconv1 = self._upconv1(layer4) # [1, 256, 32, 32]
        upconv1 = torch.cat([upconv1, transition_conv1], dim=1)

        transition_conv2 = self._transition_conv2(layer2)

        upconv2 = self._upconv2(upconv1) # [1, 128, 64, 64]
        upconv2 = torch.cat([upconv2, transition_conv2], dim=1)

        transition_conv3 = self._transition_conv3(layer1)

        upconv3 = self._upconv3(upconv2) # [1, 64, 128, 128]
        result = torch.cat([upconv3, transition_conv3], dim=1)

        return result


def get_upconv_resnet(base=18, pretrained=False, input_frame_number=2):
    net = UpConvResNet(base=base,
                       input_frame_number=input_frame_number,
                       deconv_channels=(256, 128, 64),
                       deconv_kernels=(4, 4, 4),
                       pretrained=pretrained)
    return net


if __name__ == "__main__":
    input_size = (1024, 1024)
    device = torch.device("cuda")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    net = get_upconv_resnet(base=34, pretrained=False, input_frame_number=1)
    net.to(device)
    output = net(torch.rand(1, 3, input_size[0],input_size[1], device=device))
    print(f"< input size(height, width) : {input_size} >")
    print(f"< output shape : {output.shape} >")
    '''
    < input size(height, width) : (512, 512) >
    < output shape : torch.Size([1, 64, 128, 128]) >
    '''
