#include <cstddef> // std::size
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <vector>

extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K);
extern "C" void solve2(const float *A, const float *B, float *C, int M, int N,
                       int K);
extern "C" void solve3(const float *A, const float *B, float *C, int M, int N,
                       int K);
extern "C" void solve4(const float *A, const float *B, float *C, int M, int N,
                       int K);

// --- Test cases ---
struct MatrixTestCase {
  int M;            // Rows of A and C
  int N;            // Columns of A, rows of B
  int K;            // Columns of B and C
  const char *desc; // Description
};

static const MatrixTestCase TEST_CASES[] = {
    {2, 3, 2, "2*3 X 3*2 case"},
    {128, 128, 128, "128*128 X 128*128 case"},
    {1024, 1024, 1024, "1024*1024 X 1024*1024 case"},
    {2048, 4096, 2048, "2048*4096 X 4096*2048 case"},
};

// Helper to build axis values and pretty names.
static std::vector<int64_t> make_case_indices() {
  std::vector<int64_t> v;
  v.reserve(std::size(TEST_CASES));
  for (int i = 0; i < static_cast<int>(std::size(TEST_CASES)); ++i)
    v.push_back(i);
  return v;
}

template <auto KernelFunc>
static void bench_matrix_mul_impl(nvbench::state &state) {
  // Pick the test case by index chosen on the axis.
  const int case_idx = static_cast<int>(state.get_int64("case"));
  const auto &tc = TEST_CASES[case_idx];
  const int M = tc.M, N = tc.N, K = tc.K;

  float *d_A, *d_B, *d_C;
  NVBENCH_CUDA_CALL(
      cudaMalloc(&d_A, static_cast<size_t>(M) * N * sizeof(float)));
  NVBENCH_CUDA_CALL(
      cudaMalloc(&d_B, static_cast<size_t>(N) * K * sizeof(float)));
  NVBENCH_CUDA_CALL(
      cudaMalloc(&d_C, static_cast<size_t>(M) * K * sizeof(float)));
  NVBENCH_CUDA_CALL(
      cudaMemset(d_A, 0, static_cast<size_t>(M) * N * sizeof(float)));
  NVBENCH_CUDA_CALL(
      cudaMemset(d_B, 0, static_cast<size_t>(N) * K * sizeof(float)));

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch &) { KernelFunc(d_A, d_B, d_C, M, N, K); });

  NVBENCH_CUDA_CALL(cudaFree(d_A));
  NVBENCH_CUDA_CALL(cudaFree(d_B));
  NVBENCH_CUDA_CALL(cudaFree(d_C));
}

static void bench_solve(nvbench::state &state) {
  bench_matrix_mul_impl<solve>(state);
}
static void bench_solve_sliced(nvbench::state &state) {
  bench_matrix_mul_impl<solve2>(state);
}
static void bench_solve_sliced_vec4(nvbench::state &state) {
  bench_matrix_mul_impl<solve3>(state);
}
static void bench_solve_sliced_v1(nvbench::state &state) {
  bench_matrix_mul_impl<solve4>(state);
}

// --- nvbench Benchmark Registration ---

NVBENCH_BENCH(bench_solve)
    .set_name("[solve]matrix_multiplication_kernel")
    .add_int64_axis("case", make_case_indices());

NVBENCH_BENCH(bench_solve_sliced)
    .set_name("[solve2]matrix_multiplication_kernel_sliced")
    .add_int64_axis("case", make_case_indices());

NVBENCH_BENCH(bench_solve_sliced_vec4)
    .set_name("[solve3]matrix_multiplication_kernel_sliced_vec4")
    .add_int64_axis("case", make_case_indices());

NVBENCH_BENCH(bench_solve_sliced_v1)
    .set_name("[solve4]matrix_multiplication_kernel_sliced_v1")
    .add_int64_axis("case", make_case_indices());
