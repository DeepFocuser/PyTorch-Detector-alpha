import logging
import os

import torch
import torch.nn as nn
from core.model.backbone.UpConvResNet import get_upconv_resnet

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

class SegmentationNet(nn.Module):

    def __init__(self, base=18, input_frame_number=1,pretrained=True):
        super(SegmentationNet, self).__init__()

        self._base_network = get_upconv_resnet(base=base, pretrained=pretrained, input_frame_number=input_frame_number)
        _, in_channels, _, _ = self._base_network(torch.rand(1, input_frame_number*3, 960, 1280)).shape
        self._upsample = nn.Upsample(size=None, scale_factor=4, mode='bilinear', align_corners=None, recompute_scale_factor=None)

    def forward(self,  x):

        output = self._base_network(x)
        output = self._upsample(output)
        # torch.softmax(output, axis=-1) # jit or onnx 만들 때

        return output


if __name__ == "__main__":

    input_size = (512, 512)
    device = torch.device("cuda")
    net = SegmentationNet(base=18,
                    input_frame_number=1,
                    pretrained=False)
    net.to(device)
    output = net(torch.rand(1, 3, input_size[0],input_size[1], device=device))
    print(f"< input size(height, width) : {input_size} >")
    print(f"output prediction shape : {output.shape}")

    '''
    < input size(height, width) : (960, 1280) >
    output prediction shape : torch.Size([1, 2, 960, 1280])
    '''
