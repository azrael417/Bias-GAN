import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.model_zoo as model_zoo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.partialconv2d import PartialConv2d


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.PartialConv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias,
                               multi_channel = True, return_mask = True, eps=1e-6)
        self.pointwise = nn.PartialConv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias,
                                          multi_channel = True, return_mask = True, eps=1e-6)

    def forward(self, x, h):
        x, h = self.conv1(x, h)
        x, h = self.pointwise(x, h)
        return x, h


def fixed_padding(inputs, masks, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    padded_masks = F.pad(masks, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs, masks


class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        self.conv1 = nn.PartialConv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                                      groups=inplanes, bias=bias,
                                      multi_channel = True, return_mask = True, eps=1e-6)
        self.pointwise = nn.PartialConv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias,
                                          multi_channel = True, return_mask = True, eps=1e-6)

    def forward(self, x, h):
        x, h = fixed_padding(x, h, self.conv1.kernel_size[0], rate=self.conv1.dilation[0])
        x, h = self.conv1(x, h)
        x, h = self.pointwise(x, h)
        return x, h

class PCBlockHelper(nn.Module):
    def __init__(self, in_filters, out_filters, stride=1, dilation=1, activation="leaky-relu", normalizer=nn.BatchNorm2d):
        super(PCBlockHelper, self).__init__()
        
        if activation == "leaky-relu":
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activ = nn.ReLU(inplace=True)
        else:
            self.activ = None
        
        # conv
        self.conv = SeparableConv2d_same(in_filters, out_filters, 3, stride=stride, dilation=dilation)
        
        if normalizer is not None:
            self.bn = normalizer(out_filters)
        else:
            self.bn = None


    def forward(self, x, h):
        if self.activ is not None:
            x = self.activ(x)
        x, h = self.conv(x, h)
        if self.bn is not None:
            x = self.bn(x)
        
        return x, h


class PCBlock(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False, normalizer=nn.BatchNorm2d):
        super(PCBlock, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.PartialConv2d(inplanes, planes, 1, stride=stride, bias=False,
                                         multi_channel = True, return_mask = True, eps=1e-6)
            self.skipbn = normalizer(planes)
        else:
            self.skip = None

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        rep = []

        #filters = inplanes
        #if grow_first:
        #    rep.append(self.relu)
        #    rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
        #    rep.append(normalizer(planes))
        #    filters = planes
        
        filters = inplanes
        if grow_first:
            rep.append(BlockHelper(inplanes, planes, 
                                    stride = 1,
                                    activation='leaky-relu' if start_with_relu else None,
                                    normalizer = normalizer))
            filters = planes

        #for i in range(reps - 1):
        #    rep.append(self.relu)
        #    rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
        #    rep.append(normalizer(filters))
        
        for i in range(reps - 1):
            rep.append(BlockHelper(filters, filters, 
                                   stride = 1,
                                   activation = None if (not start_with_relu and i==0 and not grow_first) else "leaky-relu",
                                   normalizer = normalizer))

        #if not grow_first:
        #    rep.append(self.relu)
        #    rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
        #    rep.append(normalizer(planes))
        
        if not grow_first:
            rep.append(BlockHelper(inplanes, planes, 
                                   stride = 1,
                                   activation = "leaky-relu",
                                   normalizer = normalizer))

        #if not start_with_relu:
        #    rep = rep[1:]

        #if stride != 1:
        #    rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))

        if stride != 1:
            rep.append(BlockHelper(planes, planes, stride = 2, activation = None, normalizer = None))

       # if stride == 1 and is_last:
       #    rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))
       
       if stride == 1 and is_last:
           rep.append(BlockHelper(planes, planes, stride = 1, activation = None, normalizer = None))

        self.rep = nn.Sequential(*rep)


    def forward(self, inp, mask):
        x, h = self.rep(inp, mask)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip

        return x, h


class PCXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16, normalizer=nn.BatchNorm2d):
        super(PCXception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        #self.conv1 = nn.Conv2d(inplanes, 32, 3, stride=2, padding=1, bias=False)
        #self.bn1 = normalizer(32)
        #self.relu = nn.LeakyReLU(0.2, inplace=True)

        #self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        #self.bn2 = normalizer(64)

        #self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False, normalizer=normalizer)


        # Modified entry flow
        # We should use more channels here
        self.conv1 = nn.PartialConv2d(inplanes, 128, 3, stride=2, padding=1, bias=False,
                                      multi_channel = True, return_mask = True, eps=1e-6)
        self.bn1 = normalizer(128)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.PartialConv2d(128, 128, 3, stride=1, padding=1, bias=False,
                                      multi_channel = True, return_mask = True, eps=1e-6)
        self.bn2 = normalizer(128) 

        self.block1 = PCBlock(128, 128, reps=2, stride=2, start_with_relu=False, normalizer=normalizer)

        
        # Original entry flow remainder
        self.block2 = PCBlock(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block3 = PCBlock(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            is_last=True, normalizer=normalizer)

        # Middle flow
        self.block4  = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block5  = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block6  = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block7  = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block8  = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block9  = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block10 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block11 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block12 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block13 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block14 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block15 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block16 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block17 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block18 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block19 = PCBlock(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)

        # Exit flow
        self.block20 = PCBlock(728, 1024, reps=2, stride=1, dilation=exit_block_rates[0],
                             start_with_relu=True, grow_first=False, is_last=True, normalizer=normalizer)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn3 = normalizer(1536)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn4 = normalizer(1536)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_rates[1])
        self.bn5 = normalizer(2048)

        # Init weights
        self.__init_weight()

    def forward(self, x, h):
        # Entry flow
        x, h = self.conv1(x, h)
        x = self.bn1(x)
        x = self.relu(x)

        x, h = self.conv2(x, h)
        x = self.bn2(x)
        x = self.relu(x)

        x, h = self.block1(x, h)
        low_level_feat = x
        low_level_mask = h
        x, h = self.block2(x, h)
        x, h = self.block3(x, h)

        # Middle flow
        x, h = self.block4(x, h)
        x, h = self.block5(x, h)
        x, h = self.block6(x, h)
        x, h = self.block7(x, h)
        x, h = self.block8(x, h)
        x, h = self.block9(x, h)
        x, h = self.block10(x, h)
        x, h = self.block11(x, h)
        x, h = self.block12(x, h)
        x, h = self.block13(x, h)
        x, h = self.block14(x, h)
        x, h = self.block15(x, h)
        x, h = self.block16(x, h)
        x, h = self.block17(x, h)
        x, h = self.block18(x, h)
        x, h = self.block19(x, h)

        # Exit flow
        x, h = self.block20(x, h)
        x, h = self.conv3(x, h)
        x = self.bn3(x)
        x = self.relu(x)

        x, h = self.conv4(x, h)
        x = self.bn4(x)
        x = self.relu(x)

        x, h = self.conv5(x, h)
        x = self.bn5(x)
        x = self.relu(x)

        return x, h, low_level_feat, low_level_mask

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.PartialConv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, normalizer=nn.BatchNorm2d):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.PartialConv2d(inplanes, planes, kernel_size=kernel_size,
                                                   stride=1, padding=padding, dilation=rate, bias=False,
                                                   multi_channel = True, return_mask = True, eps=1e-6)
        self.bn = normalizer(planes)
        self.relu = nn.LeakyReLU(0.2)

        self.__init_weight()

    def forward(self, x, h):
        x, h = self.atrous_convolution(x, h)
        x = self.bn(x)
        x = self.relu(x)

        return x, h

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.PartialConv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

                
#class InterpolationUpsampler(nn.Module):
#    def __init__(self, n_output, normalizer=nn.BatchNorm2d):
#        super(InterpolationUpsampler, self).__init__()
#        
#        #last conv layer
#        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                       normalizer(256),
#                                       nn.LeakyReLU(0.2, inplace=True),
#                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                       normalizer(256),
#                                       nn.LeakyReLU(0.2, inplace=True),
#                                       nn.Conv2d(256, n_output, kernel_size=1, stride=1))
#
#        # Init weights
#        self.__init_weight()
#
#    def forward(self, x, low_level_features, input_size):
#        x = F.interpolate(x, size=(int(math.ceil(input_size[-2]/4)),
#                                   int(math.ceil(input_size[-1]/4))), mode='bilinear', align_corners=True)
#        x = torch.cat((x, low_level_features), dim=1)
#        x = self.last_conv(x)
#        x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)
#
#        return x
#
#    def __init_weight(self):
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d) or isinstance(m, nn.PartialConv2d):
#                gain = nn.init.calculate_gain('leaky_relu', 0.2)
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
#                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.ones_(m.weight)
#                nn.init.zeros_(m.bias)
#            elif isinstance(m, nn.LayerNorm):
#                nn.init.ones_(m.weight)
#                nn.init.zeros_(m.bias)
#
#        
#class DeconvUpsampler(nn.Module):
#    def __init__(self, n_output, normalizer=nn.BatchNorm2d, nn_pooling=True):
#        super(DeconvUpsampler, self).__init__()
#
#        # AvgPool2d can help with unwanted checkerboard patterns
#        nnpooler = nn.Identity if not nn_pooling else nn.AvgPool2d
#        
#        # deconvs
#        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(0,1), bias=False),
#                                     normalizer(256),
#                                     nnpooler(2, stride=1, padding=0),
#                                     nn.LeakyReLU(0.2, inplace=True))
#
#        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=0, output_padding=(1,0), bias=False),
#                                     normalizer(256),
#                                     nnpooler(2, stride=1, padding=0),
#                                     nn.LeakyReLU(0.2, inplace=True))
#
#        self.conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                   normalizer(256),
#                                   nn.LeakyReLU(0.2, inplace=True),
#                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                   normalizer(256),
#                                   nn.LeakyReLU(0.2, inplace=True),
#                                   nn.Conv2d(256, 256, kernel_size=1, stride=1))
#
#        #using 128 intermediate channels because of memory requirements
#        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1,0), bias=False),
#                                     normalizer(128),
#                                     nnpooler(2, stride=1, padding=0),
#                                     nn.LeakyReLU(0.2, inplace=True))
#
#        #no bias on the last deconv
#        self.last_deconv = nn.Sequential(nn.ConvTranspose2d(128, n_output, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False))
#
#        # Init weights
#        self.__init_weight()
#
#    def forward(self, x, low_level_features, input_size):
#        x = self.deconv1(x)
#        x = self.deconv2(x)
#        x = torch.cat((x, low_level_features), dim=1)
#        x = self.conv1(x)
#        x = self.deconv3(x)
#        x = self.last_deconv(x)
#        
#        return x
#
#    def __init_weight(self):
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                gain = nn.init.calculate_gain('leaky_relu', 0.2)
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
#                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
#            elif isinstance(m, nn.ConvTranspose2d):
#                gain = nn.init.calculate_gain('leaky_relu', 0.2)
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
#                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.ones_(m.weight)
#                nn.init.zeros_(m.bias)
#            elif isinstance(m, nn.LayerNorm):
#                nn.init.ones_(m.weight)
#                nn.init.zeros_(m.bias)
#
#
#class UpsamplerExtension(nn.Module):
#    def __init__(self, n_input, n_output, normalizer=nn.BatchNorm2d, nn_pooling=True):
#        super(UpsamplerExtension, self).__init__()
#
#        # AvgPool2d can help with unwanted checkerboard patterns
#        nnpooler = nn.Identity if not nn_pooling else nn.AvgPool2d
#        
#        self.init_norm = nn.Sequential(normalizer(128),
#                                       nnpooler(2, stride=1, padding=1),
#                                       nn.LeakyReLU(0.2, inplace=True))
#        
#        self.conv1 = nn.Sequential(nn.Conv2d(n_input, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                                   normalizer(64),
#                                   nn.LeakyReLU(0.2, inplace=True),
#                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                                   normalizer(128),
#                                   nn.LeakyReLU(0.2, inplace=True))
#                                   
#        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                                   normalizer(64),
#                                   nn.LeakyReLU(0.2, inplace=True),
#                                   nn.Conv2d(64, n_output, kernel_size=3, stride=1, padding=1, bias=False))
#
#        # Init weights
#        self.__init_weight()
#
#    def forward(self, input, x):
#        
#        skip = self.conv1(input)
#        #we need to BN x here:
#        x = self.init_norm(x)
#        x = torch.cat((x,skip), dim=1)
#        x = self.conv2(x)
#        
#        return x
#        
#    def __init_weight(self):
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                gain = nn.init.calculate_gain('leaky_relu', 0.2)
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
#                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
#            elif isinstance(m, nn.ConvTranspose2d):
#                gain = nn.init.calculate_gain('leaky_relu', 0.2)
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
#                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.ones_(m.weight)
#                nn.init.zeros_(m.bias)
#            elif isinstance(m, nn.LayerNorm):
#                nn.init.ones_(m.weight)
#                nn.init.zeros_(m.bias)
#
#
#class DeconvUpsamplerLocalExtension(nn.Module):
#    def __init__(self, input_size, n_output, normalizer=nn.BatchNorm2d, nn_pooling=True):
#        super(DeconvUpsamplerLocalExtension, self).__init__()
#
#        # AvgPool2d can help with unwanted checkerboard patterns
#        nnpooler = nn.Identity if not nn_pooling else nn.AvgPool2d
#
#        n_input = input_size[1]
#        self.init_norm = nn.Sequential(normalizer(128),
#                                       nnpooler(2, stride=1, padding=1),
#                                       nn.LeakyReLU(0.2, inplace=True))
#        
#        self.conv1 = nn.Sequential(nn.Conv2d(n_input, 64, kernel_size=3, stride=1, padding=1, bias=False),
#                                   normalizer(64),
#                                   nn.LeakyReLU(0.2, inplace=True),
#                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#                                   normalizer(128),
#                                   nn.LeakyReLU(0.2, inplace=True))
#
#        self.conv2 = nn.Sequential(nn.Conv2d(256, n_output, kernel_size=3, stride=1, padding=1, bias=False),
#                                   normalizer(n_output),
#                                   nn.LeakyReLU(0.2, inplace=True))
#        
#        self.local_conv = Conv2dLocal(kernel_size=3, input_size = input_size)
#        
#        # Init weights
#        self.__init_weight()
#        
#    def forward(self, input, x):
#        skip = self.conv1(input)
#        
#        #we need to BN x here:
#        x = self.init_norm(x)
#        x = torch.cat((x,skip), dim=1)
#        x = self.conv2(x)
#        x = torch.transpose(x, 1, 3)
#        x = self.local_conv
#        x = torch.transpose(x, 1, 3)
#        
#        return x
#
#    def __init_weight(self):
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                gain = nn.init.calculate_gain('leaky_relu', 0.2)
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
#                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
#            elif isinstance(m, Conv2dLocal):
#                gain = nn.init.calculate_gain('leaky_relu', 0.2)
#                n = m.kernel_size * m.kernel_size * m.in_channels
#                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
#            elif isinstance(m, nn.ConvTranspose2d):
#                gain = nn.init.calculate_gain('leaky_relu', 0.2)
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
#                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.ones_(m.weight)
#                nn.init.zeros_(m.bias)
#            elif isinstance(m, nn.LayerNorm):
#                nn.init.ones_(m.weight)

class DeepLabv3_plus(nn.Module):
    def __init__(self, n_input=3, n_output=21, os=16,
                 upsampler_type="Deconv", pretrained=False, _print=True,
                 normalizer=nn.BatchNorm2d, nn_pooling=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of output channels: {}".format(n_output))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(n_input))
        super(DeepLabv3_plus, self).__init__()
        
        #which upsampler are we using
        self.upsampler_type = upsampler_type

        #select nnpooling
        self.nn_pooling = nn_pooling
        nnpooler = nn.Identity if not self.nn_pooling else nn.AvgPool2d
        
        # Atrous Conv
        self.xception_features = PCXception(n_input, os, pretrained, normalizer)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPP_module(2048, 256, rate=rates[0], normalizer=normalizer)
        self.aspp2 = ASPP_module(2048, 256, rate=rates[1], normalizer=normalizer)
        self.aspp3 = ASPP_module(2048, 256, rate=rates[2], normalizer=normalizer)
        self.aspp4 = ASPP_module(2048, 256, rate=rates[3], normalizer=normalizer)

        self.relu = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             normalizer(256),
                                             nn.LeakyReLU(0.2))

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = normalizer(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = normalizer(48)
        
        # upsampling
        if self.upsampler_type == "Interpolate":
            self.upsample = InterpolationUpsampler(n_output, normalizer)
        elif self.upsampler_type.startswith("Deconv"):
            self.upsample = DeconvUpsampler(n_output = 128 if self.upsampler_type == "Deconv1x" else n_output,
                                            normalizer = normalizer,
                                            nn_pooling = nn_pooling)
        else:
            raise NotImplementedError("Error, upsampler {} not implemented.".format(upsampler_type))
            
        # add a final couple of layers if we want 1x1 resolution
        if (self.upsampler_type == "Deconv1x") or (self.upsampler_type == "Interpolate1x"):
            self.upsample_extension = UpsamplerExtension(n_input,
                                                         n_output,
                                                         normalizer,
                                                         nn_pooling)
        elif self.upsampler_type == "Deconv":
            # we need one more avg pool here
            self.final_pool = nnpooler(2, stride=1, padding=1)

        
    def forward(self, input, masks):
        x, h, low_level_features, low_level_masks = self.xception_features(input, masks)

        # ASPP step
        x1, h1 = self.aspp1(x, h)
        x2, h2 = self.aspp2(x, h)
        x3, h3 = self.aspp3(x, h)
        x4, h4 = self.aspp4(x, h)
        x5, h5 = self.global_avg_pool(x, h)
        #h5 = torch.ones((1), requires_grad=False)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        h5 = F.interpolate(h5, size=h4.size()[2:], mode='nearest', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        h = torch.cat((h1, h2, h3, h4, h5), dim=1)

        x, h = self.conv1(x, h)
        x = self.bn1(x)
        x = self.relu(x)
        
        # low level feature processing
        low_level_features, low_level_features = self.conv2(low_level_features, low_level_mask)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)            
        
        # decoder / upsampling logic
        x = self.upsample(x, h, low_level_features, low_level_mask, input.size())
        
        # we need to add this if we want 1x1 resolution
        if self.upsampler_type == "Deconv1x":
            x, h = self.upsample_extension(input, x, h)
        elif self.upsampler_type == "Deconv":
            x = self.final_pool(x)

        return x, h

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.PartialConv2d):
                gain = nn.init.calculate_gain('leaky_relu', 0.2)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
