#include "cuda_registry.cuh"

#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include <cstddef>

namespace {

void bench_matrix_multiplication(nvbench::state &state)
{
  const auto &impl =
      leetgpu::matrix_multiplication::find_implementation(state.get_string("implementation"));
  const auto &matrix_case =
      leetgpu::matrix_multiplication::find_benchmark_case(state.get_string("case"));

  const int m = matrix_case.m;
  const int n = matrix_case.n;
  const int k = matrix_case.k;
  const auto size_a = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
  const auto size_b = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
  const auto size_c = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  NVBENCH_CUDA_CALL(cudaMalloc(&d_a, size_a * sizeof(float)));
  NVBENCH_CUDA_CALL(cudaMalloc(&d_b, size_b * sizeof(float)));
  NVBENCH_CUDA_CALL(cudaMalloc(&d_c, size_c * sizeof(float)));
  NVBENCH_CUDA_CALL(cudaMemset(d_a, 0, size_a * sizeof(float)));
  NVBENCH_CUDA_CALL(cudaMemset(d_b, 0, size_b * sizeof(float)));

  state.add_element_count(size_c);
  state.add_global_memory_reads<float>(size_a + size_b);
  state.add_global_memory_writes<float>(size_c);

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch &) { impl.func(d_a, d_b, d_c, m, n, k); });

  NVBENCH_CUDA_CALL(cudaFree(d_a));
  NVBENCH_CUDA_CALL(cudaFree(d_b));
  NVBENCH_CUDA_CALL(cudaFree(d_c));
}

} // namespace

NVBENCH_BENCH(bench_matrix_multiplication)
    .set_name("matrix_multiplication")
    .add_string_axis("implementation", leetgpu::matrix_multiplication::implementation_ids())
    .add_string_axis("case", leetgpu::matrix_multiplication::benchmark_case_ids());
