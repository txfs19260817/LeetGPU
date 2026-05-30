from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def solve_pytorch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, n: int) -> None:
    torch.add(a[:n], b[:n], out=c[:n])


@lru_cache(maxsize=1)
def _vector_add_extension():
    source_dir = Path(__file__).resolve().parent
    return load(
        name="leetgpu_vector_add",
        sources=[str(source_dir / "torch_kernel.cpp"), str(source_dir / "src_cuda.cu")],
        verbose=False,
    )


def solve_pytorch_cuda_vector_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, n: int
) -> None:
    _vector_add_extension().torch_launch_vector_add(a, b, c, n)


if __name__ == "__main__":
    n = 1024
    a = torch.randn(n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, dtype=torch.float32, device="cuda")
    c = torch.empty_like(a)
    solve_pytorch(a, b, c, n)
    assert torch.allclose(c, a + b)
    print("All tests passed")
