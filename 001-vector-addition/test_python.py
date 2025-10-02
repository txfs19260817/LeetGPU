import sys
import pathlib
import importlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

torch = pytest.importorskip("torch", reason="PyTorch not installed")
triton = pytest.importorskip("triton", reason="Triton not installed")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU not available"
)

mod = importlib.import_module("src_python")

IMPLS = ["pytorch", "triton"]

@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("N", [1 << 16, 1 << 20])
def test_vector_add_python(impl, N):
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    if impl == "pytorch":
        fn = getattr(mod, "solve_pytorch", None)
        if fn is None:
            pytest.skip("solve_pytorch not found in src_python.py")
        fn(a, b, c, N)
    elif impl == "triton":
        fn = getattr(mod, "solve_triton", None)
        if fn is None:
            pytest.skip("solve_triton not found in src_python.py")
        fn(a, b, c, N)
    else:
        pytest.skip(f"unknown impl: {impl}")

    expect = a + b
    assert c.shape == expect.shape
    assert torch.allclose(c, expect, rtol=1e-5, atol=1e-6)
