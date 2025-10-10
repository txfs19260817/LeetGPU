import torch
import triton
import triton.language as tl

from torch.utils.cpp_extension import load

# PyTorch implementation
# A, B, C are tensors on the GPU
def solve_pytorch(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    torch.add(A, B, out=C)

# Load from torch_kernel.cpp
vector_add = load(name="vector_add", sources=["torch_kernel.cpp", "src_cuda.cu"], verbose=True)
def solve_pytorch_cuda_vector_add(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int):
    vector_add.torch_launch_vector_add(A, B, C, N)

# Triton implementation
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_idx = tl.program_id(0)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a = tl.load(a_ptr + offset, mask=mask, other=0.0)
    b = tl.load(b_ptr + offset, mask=mask, other=0.0)
    c = a + b
    tl.store(c_ptr + offset, c, mask=mask)
   
# a, b, c are tensors on the GPU
def solve_triton(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)


if __name__ == "__main__":
    N = 1024
    A = torch.randn(N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, dtype=torch.float32, device="cuda")
    C_solve_pytorch = torch.zeros(N, dtype=torch.float32, device="cuda")
    C_solve_pytorch_cuda_vector_add = torch.zeros(N, dtype=torch.float32, device="cuda")
    C_solve_triton = torch.zeros(N, dtype=torch.float32, device="cuda")
    solve_pytorch(A, B, C_solve_pytorch, N)
    solve_pytorch_cuda_vector_add(A, B, C_solve_pytorch_cuda_vector_add, N)
    solve_triton(A, B, C_solve_triton, N)

    assert torch.allclose(C_solve_pytorch, C_solve_pytorch_cuda_vector_add)
    assert torch.allclose(C_solve_pytorch, C_solve_triton)
    print("All tests passed")


'''
python src_python.py --compiler jit
W1007 21:59:44.048000 4913 torch/utils/cpp_extension.py:2425] TORCH_CUDA_ARCH_LIST is not set, all archs for visible cards are included for compilation. 
W1007 21:59:44.048000 4913 torch/utils/cpp_extension.py:2425] If this is not desired, please set os.environ['TORCH_CUDA_ARCH_LIST'] to specific architectures.
[1/3] /usr/local/cuda/bin/nvcc --generate-dependencies-with-compile --dependency-output src_cuda.cuda.o.d -DTORCH_EXTENSION_NAME=vector_add -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1018\" -isystem /mnt/d/Projects/LeetGPU/.venv/lib/python3.12/site-packages/torch/include -isystem /mnt/d/Projects/LeetGPU/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.12 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 --compiler-options '-fPIC' -std=c++17 -c /mnt/d/Projects/LeetGPU/001-vector-addition/src_cuda.cu -o src_cuda.cuda.o 
[2/3] c++ -MMD -MF torch_kernel.o.d -DTORCH_EXTENSION_NAME=vector_add -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1018\" -isystem /mnt/d/Projects/LeetGPU/.venv/lib/python3.12/site-packages/torch/include -isystem /mnt/d/Projects/LeetGPU/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda/include -isystem /usr/include/python3.12 -fPIC -std=c++17 -c /mnt/d/Projects/LeetGPU/001-vector-addition/torch_kernel.cpp -o torch_kernel.o 
[3/3] c++ torch_kernel.o src_cuda.cuda.o -shared -L/mnt/d/Projects/LeetGPU/.venv/lib/python3.12/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o vector_add.so
All tests passed
'''