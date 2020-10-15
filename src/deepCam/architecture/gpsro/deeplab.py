import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.model_zoo as model_zoo
import conv2d_local

class Conv2dLocalFunction(Function):
    """Function for Conv2dLocal"""
    @staticmethod
    def forward(ctx, input_data, weights, kernel_size):
        output = conv2d_local.forward(input_data, weights, kernel_size)
        variables = [input_data, weights, torch.as_tensor(kernel_size)]
        ctx.save_for_backward(*variables)
        return output
    
    @staticmethod
    def backward(ctx, grad_o):
        outputs = conv2d_local.backward(grad_o.contiguous(), *ctx.saved_variables)
        d_input, d_weights = outputs
        return d_input, d_weights, None
                                                                                

class Conv2dLocal(nn.Module):
    r"""Applies a 2D local (aka unshared/untied) depthwise convolution
    over an input signal composed of several input planes.

    There are number of assumptions made by the current implementation:

    * Input must be in HWC format.
    * Kernel is square and have an odd dimension.
    * Kernel can be incomplete e.g. 24 values instead of 25 for 5x5 kernel.
    * Unlike ordinary convolution, this particular implementation requires
      separate set of filters for each sample in a minibatch. That is, given
      input of size :math:`(N, H, W, C)`, corresponding weights tensor
      should be :math:`(N, H, W, K*K)` (last dimension can be less than K*K).
    """

    def __init__(self, input_size, kernel_size):
        super(Conv2dLocal, self).__init__()
        self.kernel_size = int(kernel_size)
        self.in_channels = int(input_size[1])
        self.weights = torch.zeros([input_size[0], input_size[2], input_size[3], self.kernel_size * self.kernel_size],
                                   dtype=torch.float32)
        
    def forward(self, input_data):
        return Conv2dLocalFunction.apply(
            input_data.contiguous(), self.weights.contiguous(), self.kernel_size)
                                            

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv2d_same, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], rate=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False, normalizer=nn.BatchNorm2d):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = normalizer(planes)
        else:
            self.skip = None

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(normalizer(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(filters, filters, 3, stride=1, dilation=dilation))
            rep.append(normalizer(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inplanes, planes, 3, stride=1, dilation=dilation))
            rep.append(normalizer(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=2))

        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(planes, planes, 3, stride=1))


        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16, pretrained=False, normalizer=nn.BatchNorm2d):
        super(Xception, self).__init__()

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
        self.conv1 = nn.Conv2d(inplanes, 128, 3, stride=2, padding=1, bias=False)
        self.bn1 = normalizer(128)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn2 = normalizer(128) 

        self.block1 = Block(128, 128, reps=2, stride=2, start_with_relu=False, normalizer=normalizer)

        
        # Original entry flow remainder
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, start_with_relu=True, grow_first=True,
                            is_last=True, normalizer=normalizer)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_rate, start_with_relu=True, grow_first=True, normalizer=normalizer)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_rates[0],
                             start_with_relu=True, grow_first=False, is_last=True, normalizer=normalizer)

        self.conv3 = SeparableConv2d_same(1024, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn3 = normalizer(1536)

        self.conv4 = SeparableConv2d_same(1536, 1536, 3, stride=1, dilation=exit_block_rates[1])
        self.bn4 = normalizer(1536)

        self.conv5 = SeparableConv2d_same(1536, 2048, 3, stride=1, dilation=exit_block_rates[1])
        self.bn5 = normalizer(2048)

        # Init weights
        self.__init_weight()

        # Load pretrained model
        if pretrained:
            self.__load_xception_pretrained()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
                
    def __load_xception_pretrained(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            print(k)
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('block11'):
                    model_dict[k.replace('block11', 'block12')] = v
                elif k.startswith('conv3'):
                    model_dict[k] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, normalizer=nn.BatchNorm2d):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = normalizer(planes)
        self.relu = nn.LeakyReLU(0.2)

        self.__init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

                
class InterpolationUpsampler(nn.Module):
    def __init__(self, n_output, normalizer=nn.BatchNorm2d):
        super(InterpolationUpsampler, self).__init__()
        
        #last conv layer
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       normalizer(256),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       normalizer(256),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Conv2d(256, n_output, kernel_size=1, stride=1))

        # Init weights
        self.__init_weight()

    def forward(self, x, low_level_features, input_size):
        x = F.interpolate(x, size=(int(math.ceil(input_size[-2]/4)),
                                   int(math.ceil(input_size[-1]/4))), mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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

        
class DeconvUpsampler(nn.Module):
    def __init__(self, n_output, normalizer=nn.BatchNorm2d, nn_pooling=True):
        super(DeconvUpsampler, self).__init__()

        # AvgPool2d can help with unwanted checkerboard patterns
        nnpooler = nn.Identity if not nn_pooling else nn.AvgPool2d
        
        # deconvs
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=(0,1), bias=False),
                                     normalizer(256),
                                     nnpooler(2, stride=1, padding=0),
                                     nn.LeakyReLU(0.2, inplace=True))

        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=0, output_padding=(1,0), bias=False),
                                     normalizer(256),
                                     nnpooler(2, stride=1, padding=0),
                                     nn.LeakyReLU(0.2, inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   normalizer(256),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                   normalizer(256),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=1, stride=1))

        #using 128 intermediate channels because of memory requirements
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(1,0), bias=False),
                                     normalizer(128),
                                     nnpooler(2, stride=1, padding=0),
                                     nn.LeakyReLU(0.2, inplace=True))

        #no bias on the last deconv
        self.last_deconv = nn.Sequential(nn.ConvTranspose2d(128, n_output, kernel_size=3, stride=2, padding=1, output_padding=(1,1), bias=False))

        # Init weights
        self.__init_weight()

    def forward(self, x, low_level_features, input_size):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.conv1(x)
        x = self.deconv3(x)
        x = self.last_deconv(x)
        
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = nn.init.calculate_gain('leaky_relu', 0.2)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
            elif isinstance(m, nn.ConvTranspose2d):
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


class UpsamplerExtension(nn.Module):
    def __init__(self, n_input, n_output, normalizer=nn.BatchNorm2d, nn_pooling=True):
        super(UpsamplerExtension, self).__init__()

        # AvgPool2d can help with unwanted checkerboard patterns
        nnpooler = nn.Identity if not nn_pooling else nn.AvgPool2d
        
        self.init_norm = nn.Sequential(normalizer(128),
                                       nnpooler(2, stride=1, padding=1),
                                       nn.LeakyReLU(0.2, inplace=True))
        
        self.conv1 = nn.Sequential(nn.Conv2d(n_input, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   normalizer(64),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                   normalizer(128),
                                   nn.LeakyReLU(0.2, inplace=True))
                                   
        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   normalizer(64),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, n_output, kernel_size=3, stride=1, padding=1, bias=False))

        # Init weights
        self.__init_weight()

    def forward(self, input, x):
        
        skip = self.conv1(input)
        #we need to BN x here:
        x = self.init_norm(x)
        x = torch.cat((x,skip), dim=1)
        x = self.conv2(x)
        
        return x
        
    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = nn.init.calculate_gain('leaky_relu', 0.2)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
            elif isinstance(m, nn.ConvTranspose2d):
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


class DeconvUpsamplerLocalExtension(nn.Module):
    def __init__(self, input_size, n_output, normalizer=nn.BatchNorm2d, nn_pooling=True):
        super(DeconvUpsamplerLocalExtension, self).__init__()

        # AvgPool2d can help with unwanted checkerboard patterns
        nnpooler = nn.Identity if not nn_pooling else nn.AvgPool2d

        n_input = input_size[1]
        self.init_norm = nn.Sequential(normalizer(128),
                                       nnpooler(2, stride=1, padding=1),
                                       nn.LeakyReLU(0.2, inplace=True))
        
        self.conv1 = nn.Sequential(nn.Conv2d(n_input, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   normalizer(64),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                   normalizer(128),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(256, n_output, kernel_size=3, stride=1, padding=1, bias=False),
                                   normalizer(n_output),
                                   nn.LeakyReLU(0.2, inplace=True))
        
        self.local_conv = Conv2dLocal(kernel_size=3, input_size = input_size)
        
        # Init weights
        self.__init_weight()
        
    def forward(self, input, x):
        skip = self.conv1(input)
        
        #we need to BN x here:
        x = self.init_norm(x)
        x = torch.cat((x,skip), dim=1)
        x = self.conv2(x)
        x = torch.transpose(x, 1, 3)
        x = self.local_conv
        x = torch.transpose(x, 1, 3)
        
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = nn.init.calculate_gain('leaky_relu', 0.2)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
            elif isinstance(m, Conv2dLocal):
                gain = nn.init.calculate_gain('leaky_relu', 0.2)
                n = m.kernel_size * m.kernel_size * m.in_channels
                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
            elif isinstance(m, nn.ConvTranspose2d):
                gain = nn.init.calculate_gain('leaky_relu', 0.2)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0., std=gain/math.sqrt(n))
                #nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2)/np.sqrt(2.))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)

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
        self.xception_features = Xception(n_input, os, pretrained, normalizer)

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

        
    def forward(self, input):
        x, low_level_features = self.xception_features(input)

        # ASPP step
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # low level feature processing
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)            
        
        # decoder / upsampling logic
        x = self.upsample(x, low_level_features, input.size())
        
        # we need to add this if we want 1x1 resolution
        if self.upsampler_type == "Deconv1x":
            x = self.upsample_extension(input, x)
        elif self.upsampler_type == "Deconv":
            x = self.final_pool(x)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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