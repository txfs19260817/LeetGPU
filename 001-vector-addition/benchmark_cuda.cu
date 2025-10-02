#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

extern "C" void solve(const float *A, const float *B, float *C, int N);
extern "C" void solve_stride(const float *A, const float *B, float *C, int N);

static void BM_Solve(benchmark::State& state, void(*fn)(const float*, const float*, float*, int)) {
    int N = state.range(0);
    size_t size = N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    for (auto _ : state) {
        fn(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

static void BM_SolveBasic(benchmark::State& state) { BM_Solve(state, solve); }
static void BM_SolveStride(benchmark::State& state) { BM_Solve(state, solve_stride); }

BENCHMARK(BM_SolveBasic)->Arg(1<<20);
BENCHMARK(BM_SolveStride)->Arg(1<<20);

BENCHMARK_MAIN();
