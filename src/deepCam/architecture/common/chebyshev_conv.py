r"""
PyTorch implementation of a convolutional neural network on graphs based on
Chebyshev polynomials of the graph Laplacian.
See https://arxiv.org/abs/1606.09375 for details.
Copyright 2018 Michaël Defferrard.
Released under the terms of the MIT license.
"""


from scipy import sparse
import scipy.sparse.linalg
import torch


#def prepare_laplacian(laplacian):
#    r"""Prepare a graph Laplacian to be fed to a graph convolutional layer."""
#
#    def estimate_lmax(laplacian, tol=5e-3):
#        r"""Estimate the largest eigenvalue of an operator."""
#        lmax = sparse.linalg.eigsh(laplacian, k=1, tol=tol,
#                                   ncv=min(laplacian.shape[0], 10),
#                                   return_eigenvectors=False)
#        lmax = lmax[0]
#        lmax *= 1 + 2*tol  # Be robust to errors.
#        return lmax
#
#    def scale_operator(L, lmax, scale=1):
#        r"""Scale the eigenvalues from [0, lmax] to [-scale, scale]."""
#        I = sparse.identity(L.shape[0], format=L.format, dtype=L.dtype)
#        L *= 2 * scale / lmax
#        L -= I
#        return L
#
#    lmax = estimate_lmax(laplacian)
#    laplacian = scale_operator(laplacian, lmax)
#
#    laplacian = sparse.coo_matrix(laplacian)
#
#    # PyTorch wants a LongTensor (int64) as indices (it'll otherwise convert).
#    indices = np.empty((2, laplacian.nnz), dtype=np.int64)
#    np.stack((laplacian.row, laplacian.col), axis=0, out=indices)
#    indices = torch.from_numpy(indices)
#
#    laplacian = torch.sparse_coo_tensor(indices, laplacian.data, laplacian.shape)
#    laplacian = laplacian.coalesce()  # More efficient subsequent operations.
#    return laplacian


# State-less function.
def cheb_conv(laplacian, x, weight):
    B, V, Fin = x.shape
    Fin, K, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials (kenel size)

    # transform to Chebyshev basis
    x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin*B])              # V x Fin*B
    x = x0.unsqueeze(0)                   # 1 x V x Fin*B

    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)     # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for _ in range(2, K):
        x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
        x0, x1 = x1, x2

    x = x.view([K, V, Fin, B])              # K x V x Fin x B
    x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    x = x.view([B*V, Fin*K])                # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin*K, Fout)
    x = x.matmul(weight)      # B*V x Fout
    x = x.view([B, V, Fout])  # B x V x Fout

    return x


def cheb_conv_padded(laplacian, x, weight):
    B, V, Fin = x.shape
    Fin, K, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials (kenel size)
    
    # transform to Chebyshev basis
    x0 = x.contiguous()  # B x V x Fin
    x0 = x0.view([B*V, Fin]) # B*V x Fin
    x = x0.unsqueeze(0) # 1 x B*V x Fin
    
    if K > 1:
        x1 = torch.sparse.mm(laplacian, x0)     # B*V x Fin
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x B*V x Fin
    for _ in range(2, K):
        x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin
        x0, x1 = x1, x2

    x = x.view([K, B, V, Fin]) # K x B x V x Fin
    x = x.permute(1, 2, 3, 0).contiguous()  # B x V x Fin x K
    x = x.view([B*V, Fin*K])                # B*V x Fin*K
    
    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin*K, Fout)
    x = torch.matmul(x, weight) # B*V x Fout
    x = x.view([B, V, Fout])  # B x V x Fout

    return x
    

# State-full class.
class ChebConv(torch.nn.Module):
    """Graph convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Number of trainable parameters per filter, which is also the size of
        the convolutional kernel.
        The order of the Chebyshev polynomials is kernel_size - 1.

        * A kernel_size of 1 won't take the neighborhood into account.
        * A kernel_size of 2 will look up to the 1-neighborhood (1 hop away).
        * A kernel_size of 3 will look up to the 2-neighborhood (2 hops away).

        A kernel_size of 0 is equivalent to not having a graph (or an empty
        adjacency matrix). All the vertices are treated independently and form
        a set. Every element of that set is given to a fully connected layer
        with a weight matrix of size (out_channels x in_channels).
    bias : bool
        Whether to add a bias term.
    conv : callable
        Function which will perform the actual convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 bias = True, conv=cheb_conv_padded):
        super(ChebConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = conv
        self.use_bias = bias

        # shape = (kernel_size, out_channels, in_channels)
        shape = (in_channels, kernel_size, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def extra_repr(self):
        s = '{in_channels} -> {out_channels}, kernel_size={kernel_size}'
        s += ', bias=' + str(self.use_bias)
        return s.format(**self.__dict__)

    
    def forward(self, laplacian, inputs):
        r"""Forward graph convolution.

        Parameters
        ----------
        laplacian : sparse matrix of shape n_vertices x n_vertices
            Encode the graph structure.
        inputs : tensor of shape n_signals x n_vertices x n_features
            Data, i.e., features on the vertices.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.use_bias:
            outputs += self.bias

        return outputs
