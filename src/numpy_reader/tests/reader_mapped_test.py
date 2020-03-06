import numpy as np
import torch
import numpy_reader as nr

#instantiate reader class
#npr = nr.numpy_reader((3,5), torch.int64, )

print("CPU Test")
npr = nr.numpy_reader(-1)

#linear tensor
filename = "/data/numpy_tests/arr_linear.npy"
t_lin = npr.get_sample(filename)
a_lin = np.load(filename)
print("linear tensor: ", t_lin)
print("linear array:", a_lin)
print("linear difference:", np.linalg.norm(t_lin.numpy() - a_lin) )

#zero tensor
filename = "/data/numpy_tests/arr_zeros.npy"
t_zero = npr.get_sample(filename)
a_zero = np.load(filename)
print("zero tensor: ", t_zero)
print("zero array:", a_zero)
print("zero difference:", np.linalg.norm(t_zero.numpy() - a_zero) )


#big tensor
filename = "/data/numpy_tests/arr_big.npy"
t_big = npr.get_sample(filename)
a_big = np.load(filename)
print("big tensor norm:", np.linalg.norm(t_big.numpy()) )
print("big array norm:", np.linalg.norm(a_big) )
print("big difference:", np.linalg.norm(t_big.numpy() - a_big) )


print("\n\n")
print("GPU Test")
npr = nr.numpy_reader(0)

#linear tensor
filename = "/data/numpy_tests/arr_linear.npy"
t_lin = npr.get_sample(filename).to("cpu")
a_lin = np.load(filename)
print("linear tensor: ", t_lin.to("cpu"))
print("linear array:", a_lin)
print("linear difference:", np.linalg.norm(t_lin.numpy() - a_lin) )

#zero tensor
filename = "/data/numpy_tests/arr_zeros.npy"
t_zero = npr.get_sample(filename).to("cpu")
a_zero = np.load(filename)
print("zero tensor: ", t_zero.to("cpu"))
print("zero array:", a_zero)
print("zero difference:", np.linalg.norm(t_zero.numpy() - a_zero) )

#big tensor
filename = "/data/numpy_tests/arr_big.npy"
t_big = npr.get_sample(filename).to("cpu")
a_big = np.load(filename)
print("big tensor norm:", np.linalg.norm(t_big.numpy()) )
print("big array norm:", np.linalg.norm(a_big) )
print("big difference:", np.linalg.norm(t_big.numpy() - a_big) )
