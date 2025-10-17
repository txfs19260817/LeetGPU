#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <random>
#include <vector>

extern "C" void solve(const float *input, float *output, int N);
extern "C" void solve1(const float* input, float* output, int N);
extern "C" void solve2(const float* input, float* output, int N);
extern "C" void solve3(const float* input, float* output, int N);
extern "C" void solve4(const float* input, float* output, int N);
extern "C" void solve5(const float* input, float* output, int N);
extern "C" void solve6(const float* input, float* output, int N);
extern "C" void solve_cub(const float* input, float* output, int N);

template <auto KernelFunc>
static void bench_reduction_impl(nvbench::state &state) {
  // Get the array size for this specific benchmark run
  const int N = static_cast<int>(state.get_int64("N"));

  // Generate random input data on the host
  std::vector<float> h_input(N);
  std::mt19937 rng(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (int i = 0; i < N; ++i) {
    h_input[i] = dist(rng);
  }

  // Allocate memory on the GPU
  float *d_input, *d_output;
  NVBENCH_CUDA_CALL(cudaMalloc(&d_input, N * sizeof(float)));
  NVBENCH_CUDA_CALL(cudaMalloc(&d_output, sizeof(float)));

  // Copy random input to GPU
  NVBENCH_CUDA_CALL(cudaMemcpy(d_input, h_input.data(), N * sizeof(float),
                               cudaMemcpyHostToDevice));
  NVBENCH_CUDA_CALL(cudaMemset(d_output, 0, sizeof(float)));

  // state.exec() calls the specific KernelFunc that was passed in
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    KernelFunc(d_input, d_output, N);
  });

  // Release GPU memory
  NVBENCH_CUDA_CALL(cudaFree(d_input));
  NVBENCH_CUDA_CALL(cudaFree(d_output));
}

static void bench_solve(nvbench::state &state) {
  bench_reduction_impl<solve>(state);
}
static void bench_solve_1(nvbench::state &state) {
    bench_reduction_impl<solve1>(state);
}
static void bench_solve_2(nvbench::state &state) {
    bench_reduction_impl<solve2>(state);
}
static void bench_solve_3(nvbench::state &state) {
    bench_reduction_impl<solve3>(state);
}
static void bench_solve_4(nvbench::state &state) {
    bench_reduction_impl<solve4>(state);
}
static void bench_solve_5(nvbench::state &state) {
    bench_reduction_impl<solve5>(state);
}
static void bench_solve_6(nvbench::state &state) {
    bench_reduction_impl<solve6>(state);
}
static void bench_solve_cub(nvbench::state &state) {
    bench_reduction_impl<solve_cub>(state);
}

// --- nvbench Benchmark Registration ---

const std::vector<nvbench::int64_t> ns = {
  10000,
  65535,
  1 << 16, // 65,536
  1 << 20, // 1,048,576 (1M)
  1 << 24, // 16,777,216 (16M)
  1 << 26, // 67,108,864 (64M)
  1 << 28, // 268,435,456 (256M)
  1 << 30  // 1,073,741,824 (1G)
};

NVBENCH_BENCH(bench_solve).set_name("reduction_baseline").add_int64_axis("N", ns);
NVBENCH_BENCH(bench_solve_1).set_name("reduction_v1").add_int64_axis("N", ns);
NVBENCH_BENCH(bench_solve_2).set_name("reduction_v2").add_int64_axis("N", ns);
NVBENCH_BENCH(bench_solve_3).set_name("reduction_v3").add_int64_axis("N", ns);
NVBENCH_BENCH(bench_solve_4).set_name("reduction_v4").add_int64_axis("N", ns);
NVBENCH_BENCH(bench_solve_5).set_name("reduction_v5").add_int64_axis("N", ns);
NVBENCH_BENCH(bench_solve_6).set_name("reduction_v6").add_int64_axis("N", ns);
NVBENCH_BENCH(bench_solve_cub).set_name("reduction_cub").add_int64_axis("N", ns);
