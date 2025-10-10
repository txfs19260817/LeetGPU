#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>

extern "C" void solve(const float* A, const float* B, float* C, int N);
extern "C" void solve_stride(const float* A, const float* B, float* C, int N);
extern "C" void solve_vec(const float* A, const float* B, float* C, int N);
extern "C" void solve_vec2(const float* A, const float* B, float* C, int N);

template <auto KernelFunc>
static void bench_vector_add_impl(nvbench::state& state) {
    // Get the vector size for this specific benchmark run
    const int N = static_cast<int>(state.get_int64("N"));

    // Allocate memory on the GPU (this code is now shared)
    float *d_A, *d_B, *d_C;
    NVBENCH_CUDA_CALL(cudaMalloc(&d_A, N * sizeof(float)));
    NVBENCH_CUDA_CALL(cudaMalloc(&d_B, N * sizeof(float)));
    NVBENCH_CUDA_CALL(cudaMalloc(&d_C, N * sizeof(float)));
    NVBENCH_CUDA_CALL(cudaMemset(d_A, 0, N * sizeof(float)));
    NVBENCH_CUDA_CALL(cudaMemset(d_B, 0, N * sizeof(float)));

    // state.exec() calls the specific KernelFunc that was passed in
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        KernelFunc(d_A, d_B, d_C, N);
    });

    // Release GPU memory (also shared)
    NVBENCH_CUDA_CALL(cudaFree(d_A));
    NVBENCH_CUDA_CALL(cudaFree(d_B));
    NVBENCH_CUDA_CALL(cudaFree(d_C));
}

static void bench_solve(nvbench::state &state) { bench_vector_add_impl<solve>(state); }
static void bench_solve_stride(nvbench::state &state) { bench_vector_add_impl<solve_stride>(state); }
static void bench_solve_vec(nvbench::state &state) { bench_vector_add_impl<solve_vec>(state); }
static void bench_solve_vec2(nvbench::state &state) { bench_vector_add_impl<solve_vec2>(state); }

// --- nvbench Benchmark Registration ---

const std::vector<nvbench::int64_t> ns = {1024, 2048 + 3, 1 << 10, 1 << 16, 1 << 20, 1 << 26, 1 << 28};

NVBENCH_BENCH(bench_solve)
    .set_name("vector_add")
    .add_int64_axis("N", ns);

NVBENCH_BENCH(bench_solve_stride)
    .set_name("vector_add_stride")
    .add_int64_axis("N", ns);

NVBENCH_BENCH(bench_solve_vec)
    .set_name("vector_add_vectorized")
    .add_int64_axis("N", ns);

NVBENCH_BENCH(bench_solve_vec2)
    .set_name("vector_add_vectorized2")
    .add_int64_axis("N", ns);
