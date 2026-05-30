import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

torch = pytest.importorskip("torch", reason="PyTorch not installed")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")

IMPLS = [
    ("pytorch", "src_python", "solve_pytorch"),
    ("triton", "src_triton", "solve"),
]

if os.getenv("LEETGPU_TEST_TORCH_EXTENSION") == "1":
    IMPLS.append(("torch_cuda_ext", "src_python", "solve_pytorch_cuda_vector_add"))


@pytest.mark.parametrize("impl_name,module_name,func_name", IMPLS, ids=[impl[0] for impl in IMPLS])
@pytest.mark.parametrize("n", [1, 3, 1 << 16, 1 << 20], ids=lambda n: f"N{n}")
def test_vector_add_python(impl_name, module_name, func_name, n):
    mod = pytest.importorskip(module_name, reason=f"{impl_name} dependencies not installed")
    fn = getattr(mod, func_name, None)
    if fn is None:
        pytest.skip(f"{func_name} not found in {module_name}.py")

    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    try:
        fn(a, b, c, n)
    except Exception as exc:
        if impl_name == "torch_cuda_ext":
            pytest.skip(f"torch CUDA extension is unavailable: {exc}")
        raise

    expect = a + b
    assert c.shape == expect.shape
    assert torch.allclose(c, expect, rtol=1e-5, atol=1e-6)
