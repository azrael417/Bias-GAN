import math
import torch
import torch.nn.functional as F
from torch import nn, cuda

# this should be automatically differentiable
def LegendreP(l, m, x):
    assert(abs(m) <= l), "Error, please specify a valid combination of l and m."
    if l == 0:
        return 1.
    elif l == 1:
        if m == 1:
            return -torch.sqrt(1-x*x)
        if m ==0:
            return x
        if m == -1:
            return 0.5 * torch.sqrt(1-x*x)
    elif l == 2:
        if m == 2:
            return 3. * (1-x*x)
        if m == 1:
            return -3. * x * torch.sqrt(1-x*x)
        if m == 0:
            return 0.5 * (3.*x*x-1.)
        if m == -1:
            return 0.5 * x * torch.sqrt(1-x*x)
        if m == -2:
            return 0.125 * (1.-x*x)
    else:
        # make m positive
        if m < 0:
            return (-1)**m * math.factorial(l+m) / math.factorial(l-m) * LegendreP(l, -m, x)
        
        # use recurrence relation
        if l == m:
            return -(2.*l-1.) * torch.sqrt(1.-x*x) * LegendreP(l-1, l-1, x)
        elif l == m+1:
            return x * (2.*l-1.) * LegendreP(l-1, l-1, x)
        else:
            prefac = 1./float(l-m)
            return prefac * ((2.*l-1.) * x * LegendreP(l-1, m, x) - (l-1.+m) * LegendreP(l-2, m, x))

    
def SphYCoeff(l, m):
    numerator = (2.*l+1.) * math.factorial(l-m)
    denominator = 4. * math.pi * math.factorial(l+m)
    return (-1.)**m * math.sqrt(numerator / denominator)


def SphericalHarmonicY(l, m, theta, phi):
    exp_imphi = torch.complex(torch.cos(m*phi), torch.sin(m*phi))
    return SphYCoeff(l, m) * LegendreP(l, m, torch.cos(theta)) * exp_imphi


class SphericalFT(nn.Module):

    def __init__(self, lmax):
        super(SphericalFT, self).__init__()
        self.lmax = lmax
        assert (self.lmax >= 0), "Error, SphericalFFT only supports l >= 0."

        # building lookup tables:
        self.coeffs = {}
        self.coeffs[(0,0)] = SphYCoeff(0, 0)
        self.elem_count = 1
        for l in range(1, self.lmax+1):
            for m in range(0, l+1):
                self.coeffs[(l,m)] = SphYCoeff(l, m)
                self.elem_count += 1

                
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

        # polar
        cos_theta = torch.cos(theta)
        leg_l_in = {}
        leg_l_in[(0,0)] = LegendreP(0, 0, cos_theta)
        for l in range(1, self.lmax+1):
            for m in range(0, l+1):
                leg_l_in[(l,m)] = LegendreP(l, m, cos_theta)

        # init result
        results = []

        # l = m = 0
        prod = areas * leg_l_in[(0,0)] * values
        results.append(self.coeffs[(0,0)] * torch.sum(prod, dim=1, keepdim=True))

        # compute the SFT
        for l in range(1, self.lmax+1):

            # m = 0
            leg = leg_l_in[(l,0)]
            prod = areas * values
            results.append(self.coeffs[(l,0)] * torch.sum(leg * prod, dim=1, keepdim=True))

            # m > 0
            for m in range(1, l+1):
                leg = leg_l_in[(l,m)]
                tmp_res = self.coeffs[(l,m)] * torch.sum(leg * prod * exp_mimphi_in[m], dim=1, keepdim=True)
                results.append(torch.real(tmp_res))
                results.append(torch.imag(tmp_res))

        # cat output
        output = torch.cat(results, dim=1)

        return output


class InverseSphericalFT(nn.Module):

    def __init__(self, lmax):
        super(InverseSphericalFT, self).__init__()
        self.lmax = lmax
        assert (self.lmax >= 0), "Error, InverseSphericalFFT only supports l >= 0."

        # building lookup tables:
        self.coeffs = {}
        self.coeffs[(0,0)] = SphYCoeff(0, 0)
        self.elem_count = 1
        for l in range(1, self.lmax+1):
            for m in range(0, l+1):
                self.coeffs[(l,m)] = SphYCoeff(l, m)
                self.elem_count += 1
                if m > 0:
                    self.elem_count += 1

                
    def forward(self, theta_out, phi_out, values):
        # sanity check
        assert(values.shape[1] == self.elem_count), \
            f"Error, expected {self.elem_count} elements in dim=1 but got {values.shape[1]}"
        
        # expects the following input:
        # theta: N x num_points
        # phi: N x num_points
        # values: N x num_components x num_out_channels

        # reshape input                                                                                                                                  
        theta_out = torch.unsqueeze(theta_out, dim=2)
        phi_out = torch.unsqueeze(phi_out, dim=2)

        # prereqs
        #azimuth
        exp_imphi_out = {}
        for m in range(1, self.lmax+1):
            cos_mphi = torch.cos(m*phi_out)
            sin_mphi = torch.sin(m*phi_out)
            exp_imphi_out[m] = torch.complex(cos_mphi, sin_mphi)

        # polar
        cos_theta = torch.cos(theta_out)
        leg_l_out = {}
        leg_l_out[(0,0)] = LegendreP(0, 0, cos_theta)
        for l in range(1, self.lmax+1):
            for m in range(0, l+1):
                leg_l_out[(l, m)] = LegendreP(l, m, cos_theta)

        # compute ISFT
        results = torch.split(values, 1, dim=1)

        # compute the ISFT:
	# l = m = 0
        result = results[0] * self.coeffs[(0,0)] * leg_l_out[(0,0)]
        count = 1
        for l in range(1, self.lmax+1):

            # m = 0
            leg = leg_l_out[(l,0)]
            result = result + results[count] * self.coeffs[(l,0)] * leg
            count += 1

            # m > 0
            for m in range(1, l+1):
                leg = leg_l_out[(l,m)]
                # extract real and imag parts
                tmp_re = results[count] * torch.real(exp_imphi_out[m])
                count += 1
                tmp_im = results[count] * torch.imag(exp_imphi_out[m])
                count += 1
                # we have a - instead of a + because we are using Y_lm not Ybar_lm as in the FW transform
                result = result + self.coeffs[(l,m)] * leg * 2. * (tmp_re - tmp_im)

        return result

    
# fused SFT + conv + bn + activation + ISFT
class SphericalConv(nn.Module):

    def __init__(self, lmax, num_in_channels, num_out_channels,
                 normalizer = None, activation = nn.LeakyReLU):
        super(SphericalConv, self).__init__()
        self.lmax = lmax
        self.n_in = num_in_channels
        self.n_out = num_out_channels
        assert (self.lmax >= 0), "Error, SphericalFFT only supports l >= 0."

        # build lookup tables:
        self.coeffs = {}
        self.coeffs[(0,0)] = 2.*math.pi*math.sqrt(4.*math.pi) * SphYCoeff(0, 0)**2
        self.elem_count = 1
        for l in range(1, self.lmax+1):
            coeff = 2.*math.pi*math.sqrt(4.*math.pi / (2.*l+1.))
            for m in range(0, l+1):
                self.coeffs[(l,m)] = coeff * SphYCoeff(l, m)**2
                self.elem_count	+= 1

        # init weights: only m=0 are relevant
        self.weights = nn.ParameterList([nn.Parameter(torch.randn((self.n_in, self.n_out))) for i in range(self.lmax+1)])
        
        # normalizer if requested
        self.norm = None
        if normalizer is not None:
            self.norm = normalizer(num_features = self.n_out)

        # activation
        self.activation = None
        if activation is not None:
            self.activation = activation()

            
    def forward(self, theta, phi, areas, values, theta_out = None, phi_out = None):
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
        if phi_out is not None:
            phi_out = torch.unsqueeze(phi_out, dim=2)
            exp_mimphi_out = {}
            for m in range(1, self.lmax+1):
                cos_mphi = torch.cos(m*phi_out)
                sin_mphi = torch.sin(m*phi_out)
                exp_mimphi_out[m] = torch.complex(cos_mphi, -sin_mphi)
        else:
            exp_mimphi_out = exp_mimphi_in

        # polar
        cos_theta = torch.cos(theta)
        leg_l_in = {}
        leg_l_in[(0,0)] = LegendreP(0, 0, cos_theta)
        for l in range(1, self.lmax+1):
            for m in range(0, l+1):
                leg_l_in[(l,m)] = LegendreP(l, m, cos_theta)
        
        # if resampling is requested, do it here
        if theta_out is not None:
            theta_out = torch.unsqueeze(theta_out, dim=2)
            cos_theta = torch.cos(theta_out)
            leg_l_out = {}
            leg_l_out[(0,0)] = LegendreP(0, 0, cos_theta)
            for l in range(1, self.lmax+1):
                for m in range(0, l+1):
                    leg_l_out[(l, m)] = LegendreP(l, m, cos_theta)
        else:
            leg_l_out = leg_l_in
        
        # init result
        results = []

        # l = m = 0
        prod = areas * leg_l_in[(0,0)] * torch.matmul(values, self.weights[0])
        results.append(self.coeffs[(0,0)] * torch.sum(prod, dim=1, keepdim=True))
        
        # compute the SFT
        for l in range(1, self.lmax+1):
            
            # m = 0
            leg = leg_l_in[(l,0)]
            prod = areas * torch.matmul(values, self.weights[l])
            results.append(self.coeffs[(l,0)] * torch.sum(leg * prod, dim=1, keepdim=True))
            
            # m > 0
            for m in range(1, l+1):
                leg = leg_l_in[(l,m)]
                tmp_res = self.coeffs[(l,m)] * torch.sum(leg * prod * exp_mimphi_in[m], dim=1, keepdim=True)
                results.append(torch.real(tmp_res))
                results.append(torch.imag(tmp_res))

        # prepare for activation
        results = torch.cat(results, dim=1)
                
        # normalize if requested
        if self.norm is not None:
            results = torch.transpose(results, 1, 2)
            results = torch.transpose(self.norm(results), 1, 2)

        # fire activation
        if self.activation is not None:
            results = self.activation(results)

        # postprocess
        results = torch.split(results, 1, dim=1)
        
        # compute the ISFT: coefficients are already taken care of
        # l = m = 0
        result = results[0] * leg_l_out[(0,0)]
        count = 1
        for l in range(1, self.lmax+1):

            # m = 0
            leg = leg_l_out[(l,0)]
            result = result + results[count] * leg
            count += 1

            # m > 0
            for m in range(1, l+1):
                leg = leg_l_out[(l,m)]
                # extract real and imag parts
                tmp_re = results[count] * torch.real(exp_mimphi_out[m])
                count += 1
                tmp_im = results[count] * torch.imag(exp_mimphi_out[m])
                count += 1
                # we have a + instead of a minus because we are using Ybar_lm not Y_lm
                result = result + leg * 2. * (tmp_re + tmp_im)
                
        return result


