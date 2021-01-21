import os
import sys

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.spherical import SphericalConv


class SphericalConvUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels

        # build network
        self.conv1 = SphericalConv(lmax = 8,
                                   num_in_channels = num_input_channels,
                                   num_out_channels = num_output_channels,
                                   normalizer = nn.BatchNorm1d)

    def forward(self, r, theta, phi, area, data):

        # apply spherical conv
        v = self.conv1(theta, phi, area, data)

        return r, theta, phi, area, v
        
