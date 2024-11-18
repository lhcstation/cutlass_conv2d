from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, library_paths
import os

torch_lib_path = library_paths()[0]

setup(
    name='custom_conv_cuda',
    ext_modules=[
        CUDAExtension(
            name='custom_conv_cuda',
            sources=['custom_conv.cpp'],
            extra_compile_args={
                'cxx': [
                    '-O3',
                    '-std=c++17',
                    '-fpermissive',
                    '-I/home/lhc/codes/hw2/cutlass/include',
                    '-I/home/lhc/codes/hw2/cutlass/tools/util/include'
                ],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-arch=sm_89',
                    '-Xcompiler', '-fpermissive',
                    '-Xcompiler', '-std=c++17',
                    '-I/home/lhc/codes/hw2/cutlass/include',
                    '-I/home/lhc/codes/hw2/cutlass/tools/util/include',
                ]
            },
            extra_link_args=[
                '-L' + torch_lib_path,
                '-Wl,-rpath,' + torch_lib_path,
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
