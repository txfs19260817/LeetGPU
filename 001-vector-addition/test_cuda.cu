#include <gtest/gtest.h>
#include <cuda_runtime.h>

// N = 1M elements
constexpr int N = 1 << 20;

extern "C" void solve(const float *A, const float *B, float *C, int N);
extern "C" void solve_stride(const float *A, const float *B, float *C, int N);

static void run_and_check(void (*solve_fn)(const float *, const float *, float *, int), int N)
{
    size_t size = N * sizeof(float);
    std::vector<float> h_A(N), h_B(N), h_C(N);

    for (int i = 0; i < N; i++)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    // CUDA event profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solve_fn(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

    // correctness check
    for (int i = 0; i < N; i++)
    {
        ASSERT_FLOAT_EQ(h_C[i], h_A[i] + h_B[i]);
    }

    std::cout << "Kernel execution time: " << ms << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(VectorAddTest, BasicKernel)
{
    run_and_check(solve, N);
}

TEST(VectorAddTest, StrideKernel)
{
    run_and_check(solve_stride, N);
}
