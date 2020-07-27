import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from core.model.backbone.UpConvResNet import get_upconv_resnet

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

class CenterNet(nn.Module):

    def __init__(self, base=18, input_frame_number=1, heads=OrderedDict(), head_conv_channel=64, pretrained=True):
        super(CenterNet, self).__init__()

        self._base_network = get_upconv_resnet(base=base, pretrained=pretrained, input_frame_number=input_frame_number)
        _, in_channels, _, _ = self._base_network(torch.rand(1, input_frame_number*3, 512, 512)).shape

        heatmap = []
        offset = []
        wh = []

        # heatmap
        num_output = heads['heatmap']["num_output"]
        bias = heads['heatmap'].get('bias', 0.0)
        heatmap.append(nn.Conv2d(in_channels, head_conv_channel, kernel_size=3, padding=1, bias=True))
        heatmap.append(nn.ReLU(inplace=True))
        temp = nn.Conv2d(head_conv_channel, num_output, kernel_size=1, bias=True)
        temp.bias.data.fill_(bias)
        heatmap.append(temp)

        # offset
        num_output = heads['offset']["num_output"]
        bias = heads['offset'].get('bias', 0.0)
        offset.append(nn.Conv2d(in_channels, head_conv_channel, kernel_size=3, padding=1, bias=True))
        offset.append(nn.ReLU(inplace=True))
        temp = nn.Conv2d(head_conv_channel, num_output, kernel_size=1, bias=True)
        temp.bias.data.fill_(bias)
        offset.append(temp)

        # wh
        num_output = heads['wh']["num_output"]
        bias = heads['wh'].get('bias', 0.0)
        wh.append(nn.Conv2d(in_channels, head_conv_channel, kernel_size=3, padding=1, bias=True))
        wh.append(nn.ReLU(inplace=True))
        temp = nn.Conv2d(head_conv_channel, num_output, kernel_size=1, bias=True)
        temp.bias.data.fill_(bias)
        wh.append(temp)

        self._heatmap = nn.Sequential(*heatmap)
        self._offset = nn.Sequential(*offset)
        self._wh = nn.Sequential(*wh)

    def forward(self,  x):
        feature = self._base_network(x)

        heatmap = self._heatmap(feature)
        offset = self._offset(feature)
        wh = self._wh(feature)

        heatmap = torch.sigmoid(heatmap)
        return heatmap, offset, wh


if __name__ == "__main__":
    input_size = (512, 512)
    device = torch.device("cuda")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    '''
    heatmap의 bias가 -2.19 인 이유는??? retinanet의 식과 같은데... 흠..
    For the final conv layer of the classification subnet, we set the bias initialization to b = − log((1 − π)/π),
    where π specifies that at the start of training every anchor should be labeled as foreground with confidence of ∼π.
    We use π = .01 in all experiments, although results are robust to the exact value. As explained in §3.3, 
    this initialization prevents the large number of background anchors from generating a large, 
    destabilizing loss value in the first iteration of training
    '''
    net = CenterNet(base=18,
                    input_frame_number=2,
                    heads=OrderedDict([
                        ('heatmap', {'num_output': 5, 'bias': -2.19}),
                        ('offset', {'num_output': 2}),
                        ('wh', {'num_output': 2})
                    ]),
                    head_conv_channel=64,
                    pretrained=False)
    net.to(device)
    heatmap, offset, wh = net(torch.rand(1, 6, input_size[0],input_size[1], device=device))
    print(f"< input size(height, width) : {input_size} >")
    print(f"heatmap prediction shape : {heatmap.shape}")
    print(f"offset prediction shape : {offset.shape}")
    print(f"width height prediction shape : {wh.shape}")
    '''
    < input size(height, width) : (512, 512) >
    heatmap prediction shape : torch.Size([1, 5, 128, 128])
    offset prediction shape : torch.Size([1, 2, 128, 128])
    width height prediction shape : torch.Size([1, 2, 128, 128])
    '''
