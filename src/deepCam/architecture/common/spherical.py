import math
import torch
import torch.nn.functional as F
from torch import nn, cuda

# this should be automatically differentiable
def LegendreP(l, x):
    if l == 0:
        return torch.Tensor.new_full(x.shape, 1, dtype = x.dtype, device = x.device)
    elif l == 1:
        return x
    elif l == 2:
        return 1.5 * x * x - 0.5
    elif l == 3:
        return 2.5 * x * x * x - 1.5 * x
    elif l == 4:
        x2 = x * x
        return 4.375 * x2 * x2 - 2.75 * x2 + 0.375
    elif l == 5:
        x3 = x * x * x
        return 7.875 * x3 * x * x - 8.75 * x3 + 1.875 * x
    elif l == 6:
        x2 = x * x
        return 14.4375 * x2 * x2 * x2 - 19.6875 * x2 * x2 + 6.5625 * x2 - 2.1875
    else:
        return ( (2.*l-1.) * x * LegendreP(l-1, x) - (l-1.) * LegendreP(l-2, x) ) / torch.float(l)

    
def SphYCoeff(l, m):
    numerator = (2.*l+1.) * math.factorial(l-m)
    denominator = 4. * math.pi * math.factorial(l+m)
    return (-1.)**m * math.sqrt(numerator / denominator)
    

class SphericalFFT(nn.Module):

    def __init__(self, lmax, num_in_channels, num_out_channels, theta_out=None, phi_out=None):
        super(SphericalFFT, self).__init__()
        self.lmax = lmax
        self.n_in = num_in_channels
        self.n_out = num_out_channels
        assert(self.lmax >= 0, "Error, SphericalFFT only supports l >= 0.")

        # build lookup tables:
        self.coeffs = {}
        self.coeffs[(0,0)] = 2.*math.pi*math.sqrt(4.*math.pi) * SphYCoeff(0, 0)**2
        self.elem_count = 1
        for l in range(1, self.lmax+1):
            coeff = 2.*math.pi*math.sqrt(4.*math.pi / (2.*l+1.))
            for m in range(0, l+1):
                self.coeffs[(l,m)] = coeff * SphYCoeff(l, m)**2
                self.elem_count	+= 1
        
        # output arrays
        self.theta_out = theta_out
        self.phi_out = phi_out

        # init weights: only m=0 are relevant
        self.weights = nn.ParameterList([torch.randn((self.n_in, self.n_out)) for i in range(self.lmax)])

    def forward(self, theta, phi, areas, values):
        # expects the following input:
        # theta: N x num_points
        # phi: N x num_points
        # areas: N x num_points
        # values: N x num_points x num_in_channels

        # reshape input
        theta = torch.unsqueeze(theta, dim=2)
        phi = torch.unsqueeze(phi, dim=2)
        areas = torch.unsqueeze(areas, dim=2)
        
        # prereqs
        # azimuth
        exp_mimphi_in = {}
        for m in range(1, self.lmax+1):
            cos_mphi = torch.cos(m*phi)
            sin_mphi = torch.sin(m*phi)
            exp_mimphi_in[m] = torch.complex(cos_mphi, -sin_mphi)
            
        # if resampling is requested, do it here
        if self.phi_out is not None:
            exp_mimphi_out = {}
            for m in range(1, self.lmax+1):
                cos_mphi = torch.cos(m*self.phi_out)
                sin_mphi = torch.sin(m*self.phi_out)
                exp_mimphi_out[m] = torch.complex(cos_mphi, -sin_mphi)
        else:
            exp_mimphi_out = exp_mimphi_in

        # polar
        cos_theta = torch.cos(theta)
        leg_l_in = {}
        for l in range(0, self.lmax+1):
            leg_l_in[l] = LegendreP(l, cos_theta)
        
        # if resampling is requested, do it here
        if self.theta_out is not None:
            cos_theta = torch.cos(self.theta_out)
            leg_l_out = {}
            for l in range(0, self.lmax+1):
                leg_l_out[l] = LegendreP(l, cos_theta)
        else:
            leg_l_out = leg_l_in
        
        # init result
        results = []

        # l = m = 0
        prod = areas * leg_l_in[0] * torch.matmul(values, self.weights[0])
        results.append(self.coeffs[(0,0)] * torch.sum(prod, dims=1))
        
        # compute the SFT
        count = 1
        for l in range(1, self.lmax+1):
            # legendre polynomial
            leg = leg_l_in[l]

            # m = 0
            prod = areas * leg * torch.matmul(values, self.weights[l])
            results.append(self.coeffs[(l,0)] * torch.sum(prod, dims=1))
            count += 1
            
            # m > 0
            for m in range(1, l+1):
                results.append(self.coeffs[(l,m)] * torch.sum(prod * exp_mimphi_in[m], dims=1))
                count += 1

        # compute the ISFT: coefficients are already taken care of
        # l = m = 0
        result = results[0] * leg_l_out[0]
        count = 1
        for l in range(1, self.lmax+1):
            # legendre polynomial
            leg = leg_l_out[l]

            # m = 0
            result += results[count] * leg
            count += 1

            # m > 0
            for m in range(1, l+1):
                # we have a + instead of a minus because we are using Ybar_lm not Y_lm
                tmp = torch.real(results[count]) * torch.real(exp_mimphi_out[m]) + torch.imag(results[count]) * torch.imag(exp_mimphi_out[m])
                result += leg * 2. * tmp
                    
        return result


