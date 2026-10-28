"""Optional CUDA extension build: python setup_cuda.py build_ext --inplace.

Requires CUDA Toolkit/nvcc, a CUDA-enabled PyTorch wheel, and a C++ toolchain.
Normal pip installation and CPU tests do not build the extension.
"""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if not (Path('cuda/fused_ops.cpp').is_file() and Path('cuda/fused_ops.cu').is_file()):
    raise SystemExit('Run from the repository root')

setup(name='ltx-gui-cuda', version='0.5.0',
      ext_modules=[CUDAExtension('ltx_gui._cuda_kernels',
                   ['cuda/fused_ops.cpp', 'cuda/fused_ops.cu'],
                   extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']})],
      cmdclass={'build_ext': BuildExtension})
