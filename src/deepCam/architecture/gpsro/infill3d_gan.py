import os
import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common.partialconv3d import PartialConv3d

from infill3d import PCBActiv3d, PConvUNet3d as Generator

class Discriminator(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, normalizer=nn.BatchNorm3d):
        super().__init__()
        self.layer_size = layer_size
        self.enc_1 = PCBActiv3d(input_channels, 64, sample='down-3', normalizer=None)
        self.enc_2 = PCBActiv3d(64, 128, sample='down-3', normalizer=normalizer)
        self.enc_3 = PCBActiv3d(128, 256, sample='down-3', normalizer=normalizer)
        self.enc_4 = PCBActiv3d(256, 512, sample='down-3', normalizer=normalizer)
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv3d(512, 512, sample='down-3', normalizer=normalizer))

        # critic layer
        self.linear = nn.Linear(512,1, bias = False)
        self.sigmoid = nn.Sigmoid()

        # init weights
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input, input_mask):
        h_dict = {}  # for the output of enc_N
        h_mask_dict = {}  # for the output of enc_N

        h_dict['h_0'], h_mask_dict['h_0'] = input, input_mask

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # final layer
        enc_h_key = 'h_{:d}'.format(i - 1)
        x = torch.mean(h_dict[enc_h_key], dim=[2, 3, 4])
        logits = self.linear(x)
            
        return logits, self.sigmoid(logits)


class GAN(object):
    def __init__(self, input_channels=3, output_channels=3, 
                 gen_layer_size=7, disc_layer_size=7, 
                 upsampling_mode='nearest', 
                 gen_normalizer=nn.BatchNorm3d, disc_normalizer=nn.BatchNorm3d):
        # we need these guys
        self.generator = Generator(layer_size=gen_layer_size,
                                   input_channels=input_channels, output_channels=output_channels, 
                                   upsampling_mode='nearest', normalizer=gen_normalizer)
        self.discriminator = Discriminator(layer_size=disc_layer_size,
                                           input_channels=input_channels, normalizer=disc_normalizer)
        
    def generate(self, inp, mask):
        x, _ = self.generator(inp, mask)
        return x
        
    def discriminate(self, inp, mask):
        x = self.discriminator(inp, mask)
        return x


def gradient_penalty(critic, images_fake, images_real):
    # interpolate between fake and real input
    uniform = torch.distributions.uniform.Uniform(0., 1.)
    eta = uniform.rsample( (images_fake.shape[0], 1, 1, 1, 1) ).to(images_fake.device)
    images_interpol = eta * images_fake + (1. - eta) * images_real
    
    # we need to to that so that we can take the derivative                                                                                                    
    images_interpol = Variable(images_interpol, requires_grad=True)
    logits_interpol, _ = critic(images_interpol)

    # get the gradients
    grad_outputs = torch.ones(logits_interpol.size()).to(logits_interpol.device)
    gradients = torch.autograd.grad(outputs = logits_interpol, inputs = images_interpol,
				    grad_outputs = grad_outputs)[0]  #, create_graph=True, retain_graph=True)[0]
    
    # flatten gradients, (batch_size, rest)
    gradients = gradients.view(gradients.shape[0], -1)

    # compute norm
    gradients_norm = gradients.norm(2, dim=1)
    grad_penalty = ((gradients_norm - 1.) ** 2).mean()

    return grad_penalty
