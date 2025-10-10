# LeetGPU

A collection of exercises from [LeetGPU](https://leetgpu.com/) ([GitHub](https://github.com/AlphaGPU/leetgpu-challenges)), featuring implementations in CUDA, PyTorch, and Triton.

## Prerequisites

Tested on WSL2 Ubuntu 24.04.

### System Requirements
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
- [CMake 4](https://cmake.org/download/)
- Python 3.12+, GTest, NVBench
- [uv](https://docs.astral.sh/uv/getting-started/)

```bash
sudo apt-get install -y python3.12-dev libgtest-dev
uv sync
```

## Usages

### Run CUDA Tests

```bash
make build && make test
```

### Run CUDA Benchmarks

```bash
make build-release && make bench
```

### Run Python Tests

```bash
make py-sync && make py-test
```

### Clean

```bash
make clean
```

### NCU

1. Follow (the Windows section for WSL2) in [NVIDIA Developer Tools Solutions: Permission Issue with Performance Counters](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) to grant access to the GPU performance counters to all users.
2. Restart WSL in powershell by running `wsl --shutdown`
3. Run `ncu` (without sudo):
  ```bash
  ncu \
    --set=full \ # Most comprehensive profiling
    -f \ # Force overwrite output files if they already exist
    --kernel-name-base demangled \ # Use human-readable kernel names in output
    --kernel-name 'regex:vector_add' \ # Only profile kernels matching the regex pattern "vector_add"
    -o vector_add \ # Output results to files with "vector_add" prefix (creates .ncu-rep files)
    ./001_vector_addition_benchmark \ # The executable to profile. Here is a nvbench program. Flags for nvbench program can be found in https://github.com/NVIDIA/nvbench/blob/main/docs/cli_help.md
    --profile \ # Run once only
    --axis "N=67108864" # Run the benchmark with N=67108864
  ```

This will generate `vector_add.ncu-rep` which can be opened in:
- **Nsight Compute GUI** (Windows): For interactive analysis with charts and recommendations
- **Command line**: `ncu -i vector_add.ncu-rep` for text-based analysis