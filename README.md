# LeetGPU

A collection of exercises from [LeetGPU](https://leetgpu.com/), featuring implementations in CUDA, PyTorch, and Triton.

## Prerequisites

Tested on WSL2 Ubuntu 24.04.

### System Requirements
- CUDA Toolkit
- CMake 3.18+
- Python 3.12+

### Required Libraries

#### CUDA Toolkit

[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

#### Python Dependencies

Managed by `uv` (https://docs.astral.sh/uv/getting-started/):

```bash
uv sync
```

#### System Dependencies (Ubuntu/Debian)
```bash
sudo apt-get install -y python3.12-dev libgtest-dev libbenchmark-dev
```

## Usage Examples

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
