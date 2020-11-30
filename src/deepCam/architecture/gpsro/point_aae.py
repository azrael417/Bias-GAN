import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from itertools import chain 
from collections import OrderedDict, namedtuple

__all__ = ['AAE3d']

class Generator(nn.Module):
    def __init__(self, num_points, num_features, latent_dim, generator_filters, 
                 use_generator_bias, generator_relu_slope, init_weights=None):
        super().__init__()

        # copy some parameters
        self.num_points = num_points
        self.num_features = num_features
        self.z_size = latent_dim
        self.use_bias = use_generator_bias
        self.relu_slope = generator_relu_slope

        # select activation
        self.activ_args = {"inplace": True}
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU
            self.activ_args["negative_slope"] = self.relu_slope
        else:
            self.activation = nn.ReLU
            
        # first layer
        layers = OrderedDict([('gen_linear1', nn.Linear(in_features = self.z_size,
                                                    out_features = generator_filters[0],
                                                    bias=self.use_bias)),
                              ('gen_relu1', self.activation(**self.activ_args))])
        
        # intermediate layers
        for idx in range(1, len(generator_filters)):
            layers.update({'gen_linear{}'.format(idx+1) : nn.Linear(in_features = generator_filters[idx - 1],
                                                                out_features = generator_filters[idx],
                                                                bias=self.use_bias)})
            layers.update({'gen_relu{}'.format(idx+1) : self.activation(**self.activ_args)})

        # last layer
        layers.update({'gen_linear{}'.format(idx+2) : nn.Linear(in_features = generator_filters[-1],
                                                            out_features = self.num_points * (3 + self.num_features),
                                                            bias=self.use_bias)})

        # construct model
        self.model = nn.Sequential(layers)
        
        # init weights
        self.init_weights(init_weights)
    
    def _init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
            nn.init.xavier_uniform_(m.weight)
    
    def init_weights(self, init_weights):
        if init_weights is None:
            self.model.apply(self._init_weights)
        elif init_weights.endswith('.pt'):
            checkpoint = torch.load(init_weights, map_location='cpu')
            self.load_state_dict(checkpoint['generator_state_dict'])
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, input):
        output = self.model(input.squeeze())
        output = output.view(-1, (3 + self.num_features), self.num_points)
        return output


class Discriminator(nn.Module):
    def __init__(self, latent_dim, discriminator_filters,
                 use_discriminator_bias, discriminator_relu_slope, init_weights=None):
        super().__init__()

        self.z_size = latent_dim
        self.use_bias = use_discriminator_bias
        self.relu_slope = discriminator_relu_slope

        # select activation
        self.activ_args = {"inplace": True}
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU
            self.activ_args["negative_slope"] = self.relu_slope
        else:
            self.activation = nn.ReLU

        # first layer
        layers = OrderedDict([('dis_linear1', nn.Linear(in_features = self.z_size,
                                                        out_features = discriminator_filters[0],
                                                        bias = self.use_bias)),
                              ('dis_relu1', self.activation(**self.activ_args))])

        # intermediate layers
        for idx in range(1, len(hparams.discriminator_filters)):
            layers.update({'dis_linear{}'.format(idx+1) : nn.Linear(in_features = discriminator_filters[idx - 1],
                                                                    out_features = discriminator_filters[idx],
                                                                    bias = self.use_bias)})
            layers.update({'dis_relu{}'.format(idx+1) : self.activation(**self.activ_args)})

        # final layer
        layers.update({'dis_linear{}'.format(idx+2) : nn.Linear(in_features = discriminator_filters[-1],
                                                                out_features = 1,
                                                                bias = self.use_bias)})

        # construct model
        self.model = nn.Sequential(layers)
        
        # init weights
        self.init_weights(init_weights)
        
    def _init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
            nn.init.xavier_uniform_(m.weight)
        
    def init_weights(self, init_weights):
        if init_weights is None:
            self.model.apply(self._init_weights)
        elif init_weights.endswith('.pt'):
            checkpoint = torch.load(init_weights, map_location='cpu')
            self.load_state_dict(checkpoint['discriminator_state_dict'])
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        logit = self.model(x)
        return logit


class Encoder(nn.Module):
    def __init__(self, num_points, num_features, latent_dim, encoder_filters,
                 use_encoder_bias, encoder_relu_slope, init_weights=None):
        super().__init__()

        # copy some parameters
        self.num_points = num_points
        self.num_features = num_features
        self.z_size = latent_dim
        self.use_bias = use_encoder_bias
        self.relu_slope = encoder_relu_slope
        
        # select activation
        self.activ_args = {"inplace": True}
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU
            self.activ_args["negative_slope"] = self.relu_slope
        else:
            self.activation = nn.ReLU

        # first layer
        layers = OrderedDict([('enc_conv1', nn.Conv1d(in_channels = (3 + self.num_features),
                                                      out_channels = encoder_filters[0],
                                                      kernel_size = encoder_kernel_sizes[0],
                                                      bias = self.use_bias)),
                              ('enc_relu1', self.activation(**self.activ_args))])

        # intermediate layers
        for idx in range(1, len(hparams.encoder_filters)-1):
            layers.update({'enc_conv{}'.format(idx+1) : nn.Conv1d(in_channels = encoder_filters[idx - 1],
                                                                  out_channels = encoder_filters[idx],
                                                                  kernel_size = encoder_kernel_sizes[idx],
                                                                  bias = self.use_bias)})
            layers.update({'enc_relu{}'.format(idx+1) : self.activation(**self.activ_args)})
            
        # final layer
        layers.update({'enc_conv{}'.format(idx+2) : nn.Conv1d(in_channels = encoder_filters[-2],
                                                              out_channels = encoder_filters[-1],
                                                              kernel_size = encoder_kernel_sizes[-1],
                                                              bias = self.use_bias)})

        # construct model
        self.conv = nn.Sequential(layers)

        self.fc = nn.Sequential(
            nn.Linear(encoder_filters[-1],
                      encoder_filters[-2],
                      bias=True),
            self.activation(**self.activ_args)
        )

        self.mu_layer = nn.Linear(encoder_filters[-2], self.z_size, bias=True)
        self.std_layer = nn.Linear(encoder_filters[-2], self.z_size, bias=True)
        
        # init model
        self.init_weights(init_weights)
    
    def _init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
            nn.init.xavier_uniform_(m.weight)
    
    def init_weights(self, init_weights):
        if init_weights is None:
            self.conv.apply(self._init_weights)
            self.fc.apply(self._init_weights)
            self.mu_layer.apply(self._init_weights)
            self.std_layer.apply(self._init_weights)
        elif init_weights.endswith('.pt'):
            checkpoint = torch.load(init_weights, map_location='cpu')
            self.load_state_dict(checkpoint['encoder_state_dict'])
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)[1]

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim = 2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)[0]


class AAE3dModel(nn.Module):
    def __init__(self, num_points, num_features, latent_dim,
                 encoder_filters, use_encoder_bias, encoder_relu_slope
                 generator_filters, use_generator_bias, generator_relu_slope,
                 discriminator_filters, use_discriminator_bias, discriminator_relu_slope,
                 init_weights = None):
        super(AAE3dModel, self).__init__()
        
        # instantiate encoder, generator and discriminator
        self.encoder = Encoder(num_points, num_features, latent_dim, encoder_filters,
                               use_encoder_bias, encoder_relu_slope,
                               init_weights)
        self.generator = Generator(num_points, num_features, latent_dim, generator_filters,
                                   use_generator_bias, generator_relu_slope,
                                   init_weights)
        self.discriminator = Discriminator(latent_dim, discriminator_filters,
                                           use_discriminator_bias, discriminator_relu_slope, 
                                           init_weights)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x = self.generator(z)
        return x, mu, logvar
        
    def encode(self, x):
        z, mu, logvar = self.encoder(x)
        return z, mu, logvar

    def generate(self, z):
        x = self.generator(z)
        return x
        
    def discriminate(self, z):
        p = self.discriminator(z)
        return p
    
    def save_weights(self, enc_path, gen_path, disc_path):
        self.encoder.save_weights(enc_path)
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)

    def load_weights(self, enc_path, dec_path):
        self.encoder.load_weights(enc_path)
        self.generator.load_weights(gen_path)
        self.discriminator.load_weights(disc_path)
