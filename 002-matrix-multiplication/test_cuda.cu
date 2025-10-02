#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K);

static void run_matrix_mul_and_check(int M, int N, int K)
{
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C(M * K);

    // Initialize A, B
    for (int i = 0; i < M * N; i++)
    {
        h_A[i] = static_cast<float>(i % 5 + 1); // Loop small number
    }
    for (int i = 0; i < N * K; i++)
    {
        h_B[i] = static_cast<float>((i % 7) + 1);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    // Calculate CPU reference
    std::vector<float> ref(M * K, 0.0f);
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < K; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < N; i++)
            {
                sum += h_A[row * N + i] * h_B[i * K + col];
            }
            ref[row * K + col] = sum;
        }
    }

    // Verify
    for (int i = 0; i < M * K; i++)
    {
        ASSERT_NEAR(h_C[i], ref[i], 1e-4) << "Mismatch at index " << i;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(MatrixMultiplicationTest, SmallCase)
{
    run_matrix_mul_and_check(2, 3, 2);
}

TEST(MatrixMultiplicationTest, MediumCase)
{
    run_matrix_mul_and_check(8, 8, 8);
}

TEST(MatrixMultiplicationTest, RectangularCase)
{
    run_matrix_mul_and_check(4, 6, 5);
}

TEST(MatrixMultiplicationTest, LargerCase)
{
    run_matrix_mul_and_check(32, 64, 16);
}
