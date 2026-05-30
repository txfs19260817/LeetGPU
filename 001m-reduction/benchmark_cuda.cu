#include "cuda_registry.cuh"

#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

#include <cstddef>

namespace {

void bench_reduction(nvbench::state &state)
{
  const auto &impl = leetgpu::reduction::find_implementation(state.get_string("implementation"));
  const int n = static_cast<int>(state.get_int64("N"));
  const auto input_bytes = static_cast<std::size_t>(n) * sizeof(float);

  float *d_input = nullptr;
  float *d_output = nullptr;
  NVBENCH_CUDA_CALL(cudaMalloc(&d_input, input_bytes));
  NVBENCH_CUDA_CALL(cudaMalloc(&d_output, sizeof(float)));
  NVBENCH_CUDA_CALL(cudaMemset(d_input, 0, input_bytes));
  NVBENCH_CUDA_CALL(cudaMemset(d_output, 0, sizeof(float)));

  state.add_element_count(static_cast<std::size_t>(n));
  state.add_global_memory_reads<float>(static_cast<std::size_t>(n));
  state.add_global_memory_writes<float>(1);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) { impl.func(d_input, d_output, n); });

  NVBENCH_CUDA_CALL(cudaFree(d_input));
  NVBENCH_CUDA_CALL(cudaFree(d_output));
}

} // namespace

NVBENCH_BENCH(bench_reduction)
    .set_name("reduction_sum")
    .add_string_axis("implementation", leetgpu::reduction::implementation_ids())
    .add_int64_axis("N", leetgpu::reduction::benchmark_sizes());
