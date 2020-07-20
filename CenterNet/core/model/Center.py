import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.backbone.UpConvResNet import get_upconv_resnet

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

class CenterNet(nn.Module):

    def __init__(self, base=18, heads=OrderedDict(), head_conv_channel=64, pretrained=True):
        super(CenterNet, self).__init__()

        with self.name_scope():
            self._base_network = get_upconv_resnet(base=base, pretrained=pretrained)
            _, in_channels, _, _ = self._base_network(torch.rand(1, 3, 512, 512)).shape
            heads = []
            for name, values in heads.items():
                num_output = values['num_output']
                bias = values.get('bias', 0.0)
                heads.append(nn.Conv2d(in_channels ,head_conv_channel, kernel_size=3, padding=1, bias=True))
                heads.append(nn.ReLU(inplace=True))
                heads.append(nn.Conv2d(head_conv_channel, num_output, kernel_size=1, bias=True, bias_initializer=mx.init.Constant(bias)))
            self._heads = nn.Sequential(*heads)

    def forward(self,  x):
        feature = self._base_network(x)
        heatmap, offset, wh = [head(feature) for head in self._heads]
        heatmap = F.sigmoid(heatmap)
        return heatmap, offset, wh


if __name__ == "__main__":
    input_size = (768, 1280)
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
                    heads=OrderedDict([
                        ('heatmap', {'num_output': 5, 'bias': -2.19}),
                        ('offset', {'num_output': 2}),
                        ('wh', {'num_output': 2})
                    ]),
                    head_conv_channel=256,
                    pretrained=False)

    heatmap, offset, wh = net(torch.rand(1, 3, input_size[0],input_size[1]))
    print(f"< input size(height, width) : {input_size} >")
    print(f"heatmap prediction shape : {heatmap.shape}")
    print(f"offset prediction shape : {offset.shape}")
    print(f"width height prediction shape : {wh.shape}")
    '''
    heatmap prediction shape : (1, 3, 128, 128)
    offset prediction shape : (1, 2, 128, 128)
    width height prediction shape : (1, 2, 128, 128)
    '''
