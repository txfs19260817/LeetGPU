#include "cuda_registry.cuh"

#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include <cstddef>

namespace {

void bench_vector_add(nvbench::state &state)
{
  const auto &impl = leetgpu::vector_addition::find_implementation(state.get_string("implementation"));
  const int n = static_cast<int>(state.get_int64("N"));
  const auto bytes = static_cast<std::size_t>(n) * sizeof(float);

  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  NVBENCH_CUDA_CALL(cudaMalloc(&d_a, bytes));
  NVBENCH_CUDA_CALL(cudaMalloc(&d_b, bytes));
  NVBENCH_CUDA_CALL(cudaMalloc(&d_c, bytes));
  NVBENCH_CUDA_CALL(cudaMemset(d_a, 0, bytes));
  NVBENCH_CUDA_CALL(cudaMemset(d_b, 0, bytes));

  state.add_element_count(static_cast<std::size_t>(n));
  state.add_global_memory_reads<float>(static_cast<std::size_t>(n) * 2);
  state.add_global_memory_writes<float>(static_cast<std::size_t>(n));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) { impl.func(d_a, d_b, d_c, n); });

  NVBENCH_CUDA_CALL(cudaFree(d_a));
  NVBENCH_CUDA_CALL(cudaFree(d_b));
  NVBENCH_CUDA_CALL(cudaFree(d_c));
}

} // namespace

NVBENCH_BENCH(bench_vector_add)
    .set_name("vector_add")
    .add_string_axis("implementation", leetgpu::vector_addition::implementation_ids())
    .add_int64_axis("N", leetgpu::vector_addition::benchmark_sizes());
