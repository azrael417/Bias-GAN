import os
import sys

import math
from collections import OrderedDict 

import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.chebyshev_conv import ChebConv


def init_weights(m):
    if type(m) == ChebConv:
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        torch.nn.init.xavier_uniform_(m.weight, gain = gain)
        if hasattr(m, "bias"):
            m.bias.data.zero_()
    elif type(m) == nn.BatchNorm1d:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

        
class GConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size = 3,
                 normalizer = None, activation = None):
        super(GConv, self).__init__()

        self.norm = None
        if normalizer is not None:
            self.norm = normalizer(num_features = out_channels)

        self.activation = None
        if activation is not None:
            self.activation = activation()
        
        self.conv = ChebConv(in_channels,
                             out_channels,
                             kernel_size,
                             bias = False if self.norm is not None else True)
        

    def forward(self, lap, x):
        # conv
        x = self.conv(lap, x) # B x V x Fout

        # norm
        if self.norm is not None:
            xt = x.permute((0, 2, 1)) # B x Fout x V
            x = self.norm(xt).permute((0, 2, 1)) # B x V x Fout

        # activation
        if self.activation is not None:
            x = self.activation(x)

        return lap, x

        
class GConvBlock(nn.Module):
    def __init__(self, num_channels, n_conv = 3,
                 normalizer = None, activation = nn.LeakyReLU):
        super(GConvBlock, self).__init__()
        self.num_channels = num_channels
        self.normalizer = normalizer
        self.activation = activation

        # build network
        layers = []
        for n in range(n_conv):
            layers.append(GConv(in_channels = num_channels,
                                out_channels = num_channels,
                                normalizer = normalizer,
                                activation = activation))
            
        # define sequence
        self.block = nn.ModuleList(layers)
        
    def forward(self, lap, x):
        # backup for skip
        skip = x

        # apply block
        for layer in self.block:
            lap, x = layer(lap, x)

        # add skip to block output
        out = x + skip

        return lap, out

    
class GraphConvUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels,
                 conv_sizes=[64, 128, 256, 512]):
        super(GraphConvUNet, self).__init__()

        # backup parameters
        self.conv_sizes = conv_sizes
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        
        # build network
        # encoder
        self.input_conv = GConv(self.num_input_channels,
                                self.conv_sizes[0],
                                kernel_size = 3,
                                normalizer = None,
                                activation = nn.LeakyReLU)
        
        encoder_layers = []
        for idc in range(1, len(self.conv_sizes)):
            encoder_layers.append(GConvBlock(num_channels = self.conv_sizes[idc-1],
                                             n_conv = 3,
                                             normalizer = None,
                                             activation = nn.LeakyReLU))
            
            # expand
            encoder_layers.append(GConv(self.conv_sizes[idc-1],
                                        self.conv_sizes[idc],
                                        kernel_size = 3,
                                        normalizer = None,
                                        activation = nn.LeakyReLU))

        # assemble encoder
        self.encoder = nn.ModuleList(encoder_layers)

        # decoder
        conv_sizes_inv = self.conv_sizes[::-1]
        decoder_layers = [GConv(conv_sizes_inv[0],
                                conv_sizes_inv[1],
                                kernel_size = 3,
                                normalizer = None,
				activation = nn.LeakyReLU)]
        
        for idc in range(1, len(conv_sizes_inv)-1):
            decoder_layers.append(GConv(2*conv_sizes_inv[idc],
                                        conv_sizes_inv[idc+1],
                                        kernel_size = 3,
                                        normalizer = None,
                                        activation = nn.LeakyReLU))

        self.decoder = nn.ModuleList(decoder_layers)

        self.output_conv = GConv(2*conv_sizes_inv[-1],
                                 self.num_output_channels,
                                 kernel_size = 3,
                                 normalizer = None,
                                 activation = None)
        
        # init weights
        self.apply(init_weights)
        

    def forward(self, laplacian, inputs):

        # process inputs
        lap, x = self.input_conv(laplacian, inputs)

        # skip list
        skips = []
        
        # encoder
        for i in range(0, len(self.encoder), 2):
            lap, x = self.encoder[i](lap, x)
            skips.append(x)
            lap, x = self.encoder[i+1](lap, x)

        # decoder
        skips = skips[::-1]
        for i in range(0, len(self.decoder)):
            lap, x = self.decoder[i](lap, x)
            x = torch.cat([x, skips[i]], axis=-1)

        # finalize
        lap, out = self.output_conv(lap, x)
        
        return lap, out

