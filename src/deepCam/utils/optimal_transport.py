import numpy as np
import torch

# implementation of parts of
# https://arxiv.org/pdf/1803.05573.pdf
def get_cmat(output_1, output_2, eps = 1.e-8):
    # C_{ij} = 1. - c(x_i) * c(x_j) / ( ||c(x_i)|| * ||c(x_j)|| )
    # output_N are of size batch_size x num_features,
    # C_{ij} pf size batch_size * batch_size
    # normalize input first
    output_1_norm = output_1 / ( output_1.norm(2, dims=1, keepdim=True) + eps )
    output_2_norm = output_2 / ( output_2.norm(2, dims=1, keepdim=True) + eps )
    cmat = 1. - torch.matmul(output_1_norm, torch.transpose(output_2_norm, 0,1))
    return cmat


# sinkhorn algorithm implementation from
# https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport.pdf
# extension of https://www.cerfacs.fr/algor/reports/2006/TR_PA_06_42.pdf
def sinkhorn(cmat, max_iter=10000, eps=1.e-7, lmbda=1.):
    # start matrices
    rvec = torch.ones((cmat.shape[0]), dtype = cmat.dtype,
                      device = cmat.device, requires_grad = False)
    amat = cmat.detach()

    # regularization step
    #kmat = torch.exp(-lmbda * amat)

    # do the fix point iteration
    it = 0
    diff = 20. * eps
    while (it < max_iter) and (diff > eps):
        cvec = 1. / torch.matmul(rvec, amat)
        rvec_old = rvec.clone()
        rvec = 1. / torch.matmul(amat, cvec)
        diff = torch.norm(rvec - rvec_old, 2)
        it += 1

    #verbose
    print("Sinkhorn: {} iterations, diff = {}".format(it, diff))
        
    # construct the matching matrix: M = diag(r) A diag(c)
    mmat = torch.matmul( torch.diag(rvec) , torch.matmul( amat, torch.diag(cvec) ) )
    return mmat


def ot_loss(output_1, output_2):
    cmat = get_cmat(output_1, output_2)
    mmat = sinkhorn(cmat)
    return torch.mean(mmat * cmat)


def main():
    # test mat
    cmat = torch.zeros((6,6), dtype=torch.float, requires_grad=False).uniform_(0.1, 1.5)

    mmat = sinkhorn(cmat)

    print(cmat, mmat)
    print(torch.sum(mmat, dim=0), torch.sum(mmat, dim=1))


if __name__ =="__main__":
    main()
