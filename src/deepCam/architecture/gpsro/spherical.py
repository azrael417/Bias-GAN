import os
import sys

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.spherical import SphericalConv

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class SphericalConvUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels

        # build network
        self.conv1 = SphericalConv(lmax = 8,
                                   num_in_channels = num_input_channels,
                                   num_out_channels = num_output_channels)

    def forward(self, r, phi, theta, area, data):

        # apply spherical conv
        v = self.conv1(phi, theta, area, data)

        return r, phi, theta, area, v
        
