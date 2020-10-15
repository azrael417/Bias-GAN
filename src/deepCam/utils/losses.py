import torch
from torch.autograd import Variable
from torch import nn
import numpy as np

def fp_loss(logit, target, weight, fpw_1=0, fpw_2=0):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    
    #later should use cuda
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), reduction='none')
    losses = criterion(logit, target.long())
    
    preds = torch.max(logit, 1)[1]
    
    #is fp 1
    is_fp_one = (torch.eq(preds, 1) & torch.ne(preds, 1)).float()
    fp_matrix_one = (is_fp_one * fpw_1) + 1
    losses = torch.mul(fp_matrix_one, losses)
        
    #is fp 1
    is_fp_two = (torch.eq(preds, 2) & torch.ne(preds, 2)).float()
    fp_matrix_two = (is_fp_two * fpw_2) + 1
    losses = torch.mul(fp_matrix_two, losses)
    
    loss = torch.mean(losses)

    return loss


#inpainting loss, taken from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/loss.py
def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor=None):
        super().__init__()
        self.l1 = nn.L1Loss()
        #self.l1w = L1LossWeighted()
        self.extractor = extractor
        
    def forward(self, input, output, gt, mask):
        loss_dict = {}

        # these two are simple
        loss_dict['hole'] = self.l1( (1. - mask) * output, (1. - mask) * gt )
        loss_dict['valid'] = self.l1( mask * output, mask * gt )
        #loss_dict['hole'] = self.l1w(output, gt, 1.-mask)
        #loss_dict['valid'] = self.l1w(output, gt, mask)

        # we need that for variational loss
        output_comp = mask * input + (1 - mask) * output
        
        if self.extractor is not None:
            if output.shape[1] == 3:
                feat_output_comp = self.extractor(output_comp)
                feat_output = self.extractor(output)
                feat_gt = self.extractor(gt)
            elif output.shape[1] == 1:
                feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
                feat_output = self.extractor(torch.cat([output]*3, 1))
                feat_gt = self.extractor(torch.cat([gt]*3, 1))
            else:
                raise ValueError('only gray an')
            
            loss_dict['prc'] = 0.0
            for i in range(3):
                loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
                loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])
            
            loss_dict['style'] = 0.0
            for i in range(3):
                loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                              gram_matrix(feat_gt[i]))
                loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                              gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)
        
        return loss_dict
                

class L1LossWeighted:

    def __init__(self, normalize = False, eps = 1.e-8, smooth = False):
        self.l1 = nn.L1Loss(reduction='none') if not smooth else nn.SmoothL1Loss(reduction='none')
        self.eps = eps
        self.normalize = normalize

    def __call__(self, prediction, target, weights):
        if self.normalize:
            return torch.sum( self.l1(prediction, target) * weights ) / ( torch.sum(weights) + self.eps )
        else:
            return torch.mean( self.l1(prediction, target) * weights )

        
class L2LossWeighted:

    def __init__(self, normalize = False, eps = 1.e-8):
        self.l2 = nn.MSELoss(reduction='none')
        self.eps = eps
        self.normalize = normalize

    def __call__(self, prediction, target, weights):
        if self.normalize:
            return torch.sum( self.l2(prediction, target) * weights ) / ( torch.sum(weights) + self.eps )
        else:
            return torch.mean( self.l2(prediction, target) * weights )

        
class GANLoss:
    
    def __init__(self, mode, batch_size, device):

        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        
        if self.mode == "ModifiedMinMax":
            self.criterion = nn.BCEWithLogitsLoss()
            self.label_real = torch.ones ( (self.batch_size, 1) ).to(self.device)
            self.label_fake = torch.zeros( (self.batch_size, 1) ).to(self.device)
            self.dist_real = torch.distributions.uniform.Uniform(0.8, 1.0)
            self.dist_fake = torch.distributions.uniform.Uniform(0.0, 0.2)
            self.dist_swap = torch.distributions.uniform.Uniform(0.0, 1.0)
        elif self.mode == "Wasserstein":
            pass
        else:
            raise NotImplementedError("Error, {} loss not implemented".format(self.mode))
        
        
    def d_loss(self, logits_real, logits_fake):
        if self.mode == "ModifiedMinMax":
            # noise the labels for more stability
            label_fake = self.dist_fake.rsample( self.label_fake.shape ).to(self.label_fake.device)
            label_real = self.dist_real.rsample( self.label_real.shape ).to(self.label_real.device)
            # swap labels with 5% probability
            if (self.dist_swap.sample() < 0.05):
                result = 0.5 * ( self.criterion(logits_fake, label_real) + self.criterion(logits_real, label_fake) )
            else:
                result = 0.5 * ( self.criterion(logits_fake, label_fake) + self.criterion(logits_real, label_real) )
            
        elif self.mode == "Wasserstein":
            result = torch.mean(logits_fake - logits_real)

        return result

        
    def g_loss(self, logits_fake):
        if self.mode == "ModifiedMinMax":
            # we do not need to add noise here
            return self.criterion(logits_fake, self.label_real)
        elif self.mode == "Wasserstein":
            return -1. * torch.mean(logits_fake)

