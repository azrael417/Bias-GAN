import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

#import deeplab components
from .deeplab3d import *


class Discriminator(nn.Module):
    def __init__(self, n_input=3, os=16, pretrained=False, normalizer=nn.LayerNorm):
        super(Discriminator, self).__init__()
                
        # Atrous Conv
        self.xception_features = Xception3d(n_input, os, pretrained, normalizer)
        
        # MLP
        # single connected layer
        #self.linear = nn.Linear(36864, 1)
        # fully conv, reduce remaining dims before
        self.linear = nn.Linear(2048, 1) 
        self.sigmoid = nn.Sigmoid()

        # Init weights
        self.__init_weight()


    def forward(self, input):
        x, _ = self.xception_features(input)

        # reduce remaining spatial dimensions
        x = torch.mean(x, dim=[2, 3, 4])
        
        # reshape
        x_rs = torch.reshape(x, (x.shape[0], -1))

        # apply MLP
        logits = self.linear(x_rs)
        prediction = self.sigmoid(logits)
        
        # MLP
        return logits, prediction

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                gain = nn.init.calculate_gain('leaky_relu', 0.2)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1.)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, n_input, n_output, upsampler_type, noise_type, noise_dimensions, os=16, pretrained=False, normalizer=nn.BatchNorm3d):
        super(Generator, self).__init__()
        
        # set up noise vector
        self.noise_dimensions = noise_dimensions
        self.dist = None
        if noise_type == "Uniform":
            self.dist = torch.distributions.uniform.Uniform(0., 1.)
        elif noise_type == "Normal":
            self.dist = torch.distributions.normal.Normal(0., 1.)
        else:
            raise NotImplementedError("Error, noise type {} not supported.".format(noise_type))
        
        # setup u-net
        self.model = DeepLab3d(n_input = (n_input + noise_dimensions), n_output = n_output, os = os, 
                               upsampler_type = upsampler_type, pretrained = pretrained, normalizer = normalizer)

        # Init weight is called by DeepLabv3+ constructor
        
        
    def forward(self, input_raw):

        # merge the noise to the input
        if self.noise_dimensions > 0:
            input_noise = self.dist.rsample( (input_raw.shape[0], self.noise_dimensions, input_raw.shape[2], input_raw.shape[3], input_raw.shape[4]) ).to(input_raw.device)
            input = torch.cat((input_raw, input_noise), dim = 1)
        else:
            input = input_raw
        
        return self.model(input)

    

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
