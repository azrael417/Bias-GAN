from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='numpy_reader',
      ext_modules=[CUDAExtension(name='numpy_reader',
                                 sources=['cpp/numpy_reader.cpp'],
                                 libraries=["cufile"],
                                 extra_compile_args={'cxx': ['-g', '-O2'], 'nvcc': ['-g', '-O2']})
      ],
      cmdclass={'build_ext': BuildExtension})
