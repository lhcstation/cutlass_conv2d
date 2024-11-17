from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_cutlass_conv',
    ext_modules=[
        CUDAExtension(
            'custom_cutlass_conv',
            ['custom_conv.cu'],
            include_dirs=[
                '/home/lhc/codes/hw2/cutlass/include/',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--gpu-architecture=compute_75',  # Adjust for your GPU
                    '--generate-line-info',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-std=c++17'
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
