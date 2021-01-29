import os
import sys

import math
from collections import OrderedDict 

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.spherical import SphericalFT, InverseSphericalFT, SphericalConv, SphericalConvSpectral


def init_weights(m):
    if (type(m) == SphericalConv) or (type(m) == SphericalConvSpectral):
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        [torch.nn.init.xavier_uniform_(w, gain = gain) for w in m.weights]
    elif type(m) == nn.BatchNorm1d:
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class SphericalConvBlock(nn.Module):
    def __init__(self, lmax, num_channels, n_conv = 3,
                 normalizer = None, activation = nn.LeakyReLU):
        super(SphericalConvBlock, self).__init__()
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
            layers.append((f"conv_{n}", SphericalConvSpectral(lmax = lmax,
                                                              num_in_channels = num_channels,
                                                              num_out_channels = num_channels,
                                                              normalizer = normalizer,
                                                              activation = activation)))
            
        # define sequence
        self.block = nn.Sequential(OrderedDict(layers))

        
    def forward(self, theta, phi, area, data):
        print("block_in: ", data.cpu().detach().numpy())
        
        # transform data
        sdata = self.sft(theta, phi, area, data)

        # backup for skip
        skip = sdata

        print("block_skip: ", skip.cpu().detach().numpy())
        
        # apply block
        sdata = self.block(sdata)

        print("block_process: ", sdata.cpu().detach().numpy())

        # add skip to block output
        sdata = sdata + skip

        print("block_add: ", sdata.cpu().detach().numpy())

        # transform back
        out = self.isft(theta, phi, sdata)

        print("block_out: ", data.cpu().detach().numpy())
        
        return out

    
class SphericalConvUNet(nn.Module):
    def __init__(self, lmax, num_input_channels, num_output_channels):
        super(SphericalConvUNet, self).__init__()

        # backup parameters
        self.lmax = lmax
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels

        # build network
        # encoder
        self.conv1 = SphericalConv(lmax = lmax,
                                   num_in_channels = num_input_channels,
                                   num_out_channels = num_input_channels,
                                   normalizer = nn.BatchNorm1d,
                                   activation = nn.LeakyReLU)
        
        self.block1 = SphericalConvBlock(lmax = lmax,
                                         num_channels = num_input_channels,
                                         n_conv = 3,
                                         normalizer = nn.BatchNorm1d,
                                         activation = nn.LeakyReLU)

        self.conv2 = SphericalConv(lmax = lmax,
                                   num_in_channels = num_input_channels,
                                   num_out_channels = 2 * num_input_channels,
                                   normalizer = nn.BatchNorm1d,
                                   activation = nn.LeakyReLU)

        self.block2 = SphericalConvBlock(lmax = lmax // 2,
                                         num_channels = 2 * num_input_channels,
                                         n_conv = 3,
                                         normalizer = nn.BatchNorm1d,
                                         activation = nn.LeakyReLU)

        self.conv3 = SphericalConv(lmax = lmax // 2,
                                   num_in_channels = 2 * num_input_channels,
                                   num_out_channels = 4 * num_input_channels,
                                   normalizer = nn.BatchNorm1d,
                                   activation = nn.LeakyReLU)

        self.block3 = SphericalConvBlock(lmax = lmax // 4,
                                         num_channels = 4 * num_input_channels,
                                         n_conv = 3,
                                         normalizer = nn.BatchNorm1d,
                                         activation = nn.LeakyReLU)

        # decoder
        self.deconv1 = SphericalConv(lmax = lmax // 4,
                                     num_in_channels = 4 * num_input_channels,
                                     num_out_channels = 2 * num_input_channels,
                                     normalizer = nn.BatchNorm1d,
                                     activation = nn.LeakyReLU)

        self.deconv2 = SphericalConv(lmax = lmax // 2,
                                     num_in_channels = 4 * num_input_channels,
                                     num_out_channels = num_input_channels,
                                     normalizer = nn.BatchNorm1d,
                                     activation = nn.LeakyReLU)

        self.deconv3 = SphericalConv(lmax = lmax,
                                     num_in_channels = 2 * num_input_channels,
                                     num_out_channels = num_output_channels,
                                     normalizer = nn.BatchNorm1d,
                                     activation = nn.LeakyReLU)

        #debug
        self.sft = SphericalFT(lmax = lmax)
        self.isft = InverseSphericalFT(lmax = lmax)
        #debug
        
        # init weights
        self.apply(init_weights)
        

    def forward(self, r, theta, phi, area, data):
        
        # encoder
        # step 1
        print("net_input: ", data.cpu().detach().numpy()[0, :10, 0])
        sdata = self.sft(theta, phi, area, data)
        print("sft: ", sdata.cpu().detach().numpy()[0, :10, 0])
        out = self.isft(theta, phi, sdata)
        print("net_output: ", out.cpu().detach().numpy()[0, :10, 0])
        #v1 = self.conv1(theta, phi, area, data)
        #print("v1_pre: ", v1.cpu().detach().numpy())
        #v1 = self.block1(theta, phi, area, v1)
        #x1 = v1

        #print("v1: ", v1.cpu().detach().numpy())
        
        ## step 2
        #v2 = self.conv2(theta, phi, area, v1)
        #v2 = self.block2(theta, phi, area, v2)
        #x2 = v2

        #print("v2: ", v2.cpu().detach().numpy())
        
        ## step 3
        #v3 = self.conv3(theta, phi, area, v2)
        #v3 = self.block3(theta, phi, area, v3)

        #print("v3: ", v3.cpu().detach().numpy())

        ## decoder
        ## step 1
        #dv1 = self.deconv1(theta, phi, area, v3)        
        #dv1 = torch.cat([dv1,x2], dim=-1)

        #print("dv1: ", dv1.cpu().detach().numpy())
        
        ## step 2
        #dv2 = self.deconv2(theta, phi, area, dv1)
        #dv2 = torch.cat([dv2,x1], dim=-1)

        #print("dv2: ", dv2.cpu().detach().numpy())

        ## step 3
        #dv3 = self.deconv3(theta, phi, area, dv2)
        
        return r, theta, phi, area, out
        
