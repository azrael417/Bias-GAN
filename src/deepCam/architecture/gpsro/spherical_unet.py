import os
import sys

import math
from collections import OrderedDict 

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.spherical import SphericalFT, InverseSphericalFT, SphericalConv, SphericalConvSpectral


class SphericalConvBlock(nn.Module):
    def __init__(self, lmax, num_channels, n_conv = 3,
                 normalizer = None, activation = nn.LeakyReLU):
        super(SphericalConvBlock).__init__()
        self.lmax = lmax
        self.num_channels = num_channels
        self.normalizer = normalizer
        self.activation = activation

        # build network
        self.sft = SphericalFT(lmax = lmax)
        self.isft = InverseSphericalFT(lmax = lmax)

        # middle
        layers = []
        for n in range(n_conv):
            layers.append((f"conv_{n}", SphericalConvSpectral(lmax = lmax, num_channels, num_channels,
                                                              normalizer = normalizer, activation = activation)))
            
        # define sequence
        self.block = nn.Sequential(OrderedDict(layers))

        
    def forward(self, theta, phi, area, data):
        sdata = self.sft(theta, phi, area, data)
        sdata = self.block(sdata)
        out = self.isft(theta, phi, area, sdata)
        out = torch.cat(out, data, dim=-1)
        return out

    
class SphericalConvUNet(nn.Module):
    def __init__(self, lmax, num_input_channels, num_output_channels):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels

        # build network
        self.conv1 = SphericalConv(lmax = lmax,
                                   num_in_channels = num_input_channels,
                                   num_out_channels = num_input_channels,
                                   normalizer = nn.BatchNorm1d,
                                   activation = nn.LeakyReLU)
        
        self.block1 = SphericalConvBlock(lmax = lmax, num_input_channels, n_conv = 3,
                                         normalizer = nn.BatchNorm1d, activation = nn.LeakyReLU)
        

    def forward(self, r, theta, phi, area, data):

        # apply spherical conv
        v = self.conv1(theta, phi, area, data)
        v = self.block1(theta, phi, area, v)

        return r, theta, phi, area, v
        
