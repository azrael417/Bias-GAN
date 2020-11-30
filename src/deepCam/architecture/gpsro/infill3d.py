import os
import sys

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.partialconv3d import PartialConv3d

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


#class PartialConv3D(nn.Module):
#    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                 padding=0, dilation=1, groups=1, bias=True):
#        super().__init__()
#        self.input_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
#                                    stride, padding, dilation, groups, bias)
#        self.mask_conv = nn.Conv3d(in_channels, out_channels, kernel_size,
#                                   stride, padding, dilation, groups, False)
#        self.input_conv.apply(weights_init('kaiming'))
#
#        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
#
#        # mask is not updated
#        for param in self.mask_conv.parameters():
#            param.requires_grad = False
#
#    def forward(self, input, mask):
#        # http://masc.cs.gmu.edu/wiki/partialconv
#        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
#        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
#
#        output = self.input_conv(input * mask)
#        if self.input_conv.bias is not None:
#            output_bias = self.input_conv.bias.view(1, -1, 1, 1, 1).expand_as(
#                output)
#        else:
#            output_bias = torch.zeros_like(output)
#
#        with torch.no_grad():
#            output_mask = self.mask_conv(mask)
#
#        no_update_holes = output_mask == 0
#        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
#
#        output_pre = (output - output_bias) / mask_sum + output_bias
#        output = output_pre.masked_fill_(no_update_holes, 0.0)
#
#        new_mask = torch.ones_like(output)
#        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
#
#        return output, new_mask


class PCBActiv3d(nn.Module):
    def __init__(self, in_ch, out_ch, normalizer=nn.BatchNorm3d, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv3d(in_ch, out_ch, 5, 2, 2, bias=conv_bias,
                                      multi_channel = True, return_mask = True, eps=1e-6)
        elif sample == 'down-7':
            self.conv = PartialConv3d(in_ch, out_ch, 7, 2, 3, bias=conv_bias,
                                      multi_channel = True, return_mask = True, eps=1e-6)
        elif sample == 'down-3':
            self.conv = PartialConv3d(in_ch, out_ch, 3, 2, 1, bias=conv_bias,
                                      multi_channel = True, return_mask = True, eps=1e-6)
        elif sample == "point-1":
            self.conv = PartialConv3d(in_ch, out_ch, 1, 1, 0, bias=conv_bias,
                                      multi_channel = True, return_mask = True, eps=1e-6)
        else:
            self.conv = PartialConv3d(in_ch, out_ch, 3, 1, 1, bias=conv_bias,
                                      multi_channel = True, return_mask = True, eps=1e-6)

        if normalizer is not None:
            self.bn = normalizer(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

    
class PCDropout3d(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout3d(p = p)
        self.scale = (1. - p)

    def forward(self, input, mask):
        if self.training:
            # drop mask: be safe here, better round
            mask_d = torch.round(self.dropout(mask) * self.scale)
            # extract what got dropped
            drop_vals = (mask - mask_d)
            # drop input too
            input_d = input * (1. - drop_vals)
            #input_d[drop_vals > 0.] = 0.
            input_d /= self.scale
        else:
            input_d = input
            mask_d = mask
        return input_d, mask_d

    
class PConvUNet3d(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, output_channels=3,
                 upsampling_mode='nearest', normalizer=nn.BatchNorm3d, dropout_p = 0.):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        self.enc_1 = PCBActiv3d(input_channels, 64, sample='down-3', normalizer=None)
        self.enc_2 = PCBActiv3d(64, 128, sample='down-3', normalizer=normalizer)
        self.enc_3 = PCBActiv3d(128, 256, sample='down-3', normalizer=normalizer)
        self.enc_4 = PCBActiv3d(256, 512, sample='down-3', normalizer=normalizer)
        for i in range(4, self.layer_size):
            name = 'enc_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv3d(512, 512, sample='down-3', normalizer=normalizer))

        for i in range(4, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            setattr(self, name, PCBActiv3d(512 + 512, 512, activ='leaky', normalizer=normalizer))
        self.dec_4 = PCBActiv3d(512 + 256, 256, activ='leaky', normalizer=normalizer)
        self.dec_3 = PCBActiv3d(256 + 128, 128, activ='leaky', normalizer=normalizer)
        self.dec_2 = PCBActiv3d(128 + 64, 64, activ='leaky', normalizer=normalizer)
        self.dec_1 = PCBActiv3d(64 + input_channels, 32, activ='leaky', normalizer=normalizer)

        # dropout
        self.dropout = PCDropout3d(p = dropout_p) if dropout_p > 0. else None
        
        # for 1x1 resolution
        self.last_conv = PCBActiv3d(32, output_channels, activ=None, normalizer=None,
                                    sample='point-1', conv_bias=True)

        # init weights
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                #gain = nn.init.calculate_gain('leaky_relu', 0.2)
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
            elif isinstance(m, nn.BatchNorm3d):
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

            # conv
            h_dict[h_key], h_mask_dict[h_key] = getattr(self, l_key)(
                h_dict[h_key_prev], h_mask_dict[h_key_prev])
            
            # dropout
            if self.dropout is not None:
                h_dict[h_key], h_mask_dict[h_key] = self.dropout(h_dict[h_key], h_mask_dict[h_key])

            # continue
            h_key_prev = h_key

        h_key = 'h_{:d}'.format(self.layer_size)
        h, h_mask = h_dict[h_key], h_mask_dict[h_key]

        # concat upsampled output of h_enc_N-1 and dec_N+1, then do dec_N
        # (exception)
        #                            input         dec_2            dec_1
        #                            h_enc_7       h_enc_8          dec_8

        for i in range(self.layer_size, 0, -1):
            enc_h_key = 'h_{:d}'.format(i - 1)
            dec_l_key = 'dec_{:d}'.format(i)

            # better give the concrete target shapes
            h_shape = h_dict[enc_h_key].shape
            h = F.interpolate(h,
                              size=(h_shape[2], h_shape[3], h_shape[4]), mode=self.upsampling_mode)
            hm_shape = h_mask_dict[enc_h_key].shape
            h_mask = F.interpolate(h_mask,
                                   size=(hm_shape[2], hm_shape[3], hm_shape[4]), mode='nearest')
            
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            h_mask = torch.cat([h_mask, h_mask_dict[enc_h_key]], dim=1)

            # conv
            h, h_mask = getattr(self, dec_l_key)(h, h_mask)

            # dropout
            if self.dropout is not None:
                h, h_mask = self.dropout(h, h_mask)

        # required for 1x1 resolution
        #hin, hin_mask = self.input_enc_1(input, input_mask)
        #hlast = torch.cat([h, hin], dim=1)
        #hlast_mask = torch.cat([h_mask, hin_mask], dim=1)
        h, h_mask = self.last_conv(h, h_mask)
            
        return h, h_mask
