# LeetGPU

CUDA, PyTorch, and Triton implementations for selected
[LeetGPU](https://leetgpu.com/) exercises.

Each exercise directory owns its implementations and metadata:

- `src_cuda.cu`: CUDA implementations.
- `cuda_registry.cuh`: implementation names, test cases, benchmark axes.
- `test_cuda.cu`: GoogleTest correctness tests.
- `benchmark_cuda.cu`: NVBench benchmarks.
- `src_python.py` / `src_triton.py`: Python framework implementations when available.

## Requirements

- CUDA Toolkit 12.8+ or 13.x.
- CMake 3.30+.
- A C++17 host compiler supported by CUDA.
- Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/).

This repo is tested on WSL2 Ubuntu 24.04 with a GeForce RTX 3070 Laptop GPU and
CUDA 13.0.

## Linux / WSL

```bash
uv sync
make build
make test
make py-test
```

Release benchmarks:

```bash
make build-release
make bench
```

Run a single benchmark implementation with NVBench axes:

```bash
./build/001e_vector_addition_benchmark --axis "implementation=basic" --axis "N=1048576"
./build/001m_reduction_benchmark --axis "implementation=cub" --axis "N=16777216"
./build/002e_matrix_multiplication_benchmark --axis "implementation=smem_2d" --axis "case=square_512"
```

CMake presets are also available:

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
```

## Windows Visual Studio

Install CUDA Toolkit, Visual Studio 2022 with C++ build tools, CMake, and Python
3.12. From a Developer PowerShell:

```powershell
cmake --preset windows-vs-debug
cmake --build --preset windows-vs-debug
ctest --preset windows-vs-debug
```

Open `build/windows-vs/LeetGPU.sln` to debug individual `*_test` or
`*_benchmark` targets.

## Python

```bash
uv sync
uv run pytest -rs
uv run ruff check .
```

The default Python tests cover PyTorch and Triton. The PyTorch C++/CUDA extension
path is optional because it invokes just-in-time compilation:

```bash
LEETGPU_TEST_TORCH_EXTENSION=1 uv run pytest 001e-vector-addition -rs
```

## Validation

Last validated in WSL2 with CUDA 13.0:

```bash
make clean && make build
ctest --test-dir build --output-on-failure
make py-test
uv run ruff check .
uv run ruff format --check .
```

The CUDA suite currently discovers 136 tests across vector addition, reduction,
and matrix multiplication.

## Nsight Compute

Generated `.ncu-rep` files are ignored by git. Example:

```bash
mkdir -p out/profiles
ncu --set=full \
  -f \
  --kernel-name-base demangled \
  --kernel-name 'regex:vector_add' \
  -o out/profiles/vector_add \
  ./build/001e_vector_addition_benchmark \
  --profile \
  --axis "implementation=basic" \
  --axis "N=67108864"
```

Open the report in Nsight Compute GUI on Windows, or inspect it with:

```bash
ncu -i out/profiles/vector_add.ncu-rep
```
