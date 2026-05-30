# 001e Vector Addition

Implementations:

- CUDA: `basic`, `grid_stride`, `float4_loop`, `float4_direct`
- Python: PyTorch
- Triton: `src_triton.py`

The CUDA implementation registry lives in `cuda_registry.cuh`; tests and
benchmarks both use it so implementation names stay consistent.

Run only this exercise:

```bash
cmake --build build --target 001e_vector_addition_test 001e_vector_addition_benchmark
ctest --test-dir build -R 001e_vector_addition --output-on-failure
./build/001e_vector_addition_benchmark --axis "implementation=basic" --axis "N=1048576"
uv run pytest 001e-vector-addition -rs
```
