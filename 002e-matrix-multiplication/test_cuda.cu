#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K);
extern "C" void solve_sliced(const float *A, const float *B, float *C, int M, int N, int K);
extern "C" void solve_double_buffer(const float *A, const float *B, float *C, int M, int N, int K);

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _err = (call);                                                  \
    if (_err != cudaSuccess) {                                                  \
      GTEST_FAIL() << "CUDA error: " << cudaGetErrorString(_err)                \
                   << " at " << __FILE__ << ":" << __LINE__;                    \
    }                                                                           \
  } while (0)

struct MatrixTestCase {
  int M;                // Rows of A and C
  int N;                // Columns of A, rows of B
  int K;                // Columns of B and C
  const char *desc;     // Description
};

static const MatrixTestCase TEST_CASES[] = {
    {2, 3, 2, "2*3 X 3*2 case"},
    {8, 8, 8, "8*8 X 8*8 case"},
    {4, 6, 5, "4*6 X 6*5 case"},
    {32, 64, 16, "32*64 X 64*16 case"},
    {64, 64, 64, "64*64 X 64*64 case"},
    {128, 128, 128, "128*128 X 128*128 case"},
    {256, 256, 256, "256*256 X 256*256 case"},
    {512, 512, 512, "512*512 X 512*512 case"},
    {1024, 1024, 1024, "1024*1024 X 1024*1024 case"},
    {4096, 8192, 4096, "4096*8192 X 8192*4096 case"},
};

// RAII deleter for device pointers
struct CudaDeleter {
  void operator()(float *p) const noexcept { if (p) cudaFree(p); }
};

using DevicePtr = std::unique_ptr<float, CudaDeleter>;

// Generic test runner that accepts a kernel function pointer
using KernelFunc = void (*)(const float *, const float *, float *, int, int, int);

static void run_matrix_mul_and_check(int M, int N, int K, KernelFunc kernel_func) {
  const size_t sizeA = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
  const size_t sizeB = static_cast<size_t>(N) * static_cast<size_t>(K) * sizeof(float);
  const size_t sizeC = static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float);

  std::vector<float> h_A(static_cast<size_t>(M) * static_cast<size_t>(N));
  std::vector<float> h_B(static_cast<size_t>(N) * static_cast<size_t>(K));
  std::vector<float> h_C(static_cast<size_t>(M) * static_cast<size_t>(K));

  // Initialize inputs with small repeating patterns
  for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = static_cast<float>(i % 7 + 1);

  // Device allocations (RAII-managed)
  float *rawA = nullptr, *rawB = nullptr, *rawC = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&rawA), sizeA));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&rawB), sizeB));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&rawC), sizeC));
  DevicePtr d_A(rawA), d_B(rawB), d_C(rawC);

  // H2D copies
  CUDA_CHECK(cudaMemcpy(d_A.get(), h_A.data(), sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B.get(), h_B.data(), sizeB, cudaMemcpyHostToDevice));

  // Launch kernel
  kernel_func(d_A.get(), d_B.get(), d_C.get(), M, N, K);
  // Check kernel launch and runtime errors
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // D2H copy
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.get(), sizeC, cudaMemcpyDeviceToHost));

  // CPU reference
  std::vector<float> ref(static_cast<size_t>(M) * static_cast<size_t>(K), 0.0f);
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < K; ++col) {
      float sum = 0.0f;
      for (int i = 0; i < N; ++i) {
        sum += h_A[static_cast<size_t>(row) * N + i] * h_B[static_cast<size_t>(i) * K + col];
      }
      ref[static_cast<size_t>(row) * K + col] = sum;
    }
  }

  // Verify with useful context on failure
  SCOPED_TRACE(testing::Message() << "M=" << M << ", N=" << N << ", K=" << K);
  for (size_t i = 0; i < h_C.size(); ++i) {
    EXPECT_NEAR(h_C[i], ref[i], 1e-4f) << "Mismatch at index " << i;
  }
}

// -----------------------------------------------------------------------------
// Test suite for basic kernel
// -----------------------------------------------------------------------------
TEST(BasicKernelTest, SmallCase) {
  run_matrix_mul_and_check(TEST_CASES[0].M, TEST_CASES[0].N, TEST_CASES[0].K, solve);
}

TEST(BasicKernelTest, MediumCase) {
  run_matrix_mul_and_check(TEST_CASES[1].M, TEST_CASES[1].N, TEST_CASES[1].K, solve);
}

TEST(BasicKernelTest, RectangularCase) {
  run_matrix_mul_and_check(TEST_CASES[2].M, TEST_CASES[2].N, TEST_CASES[2].K, solve);
}

TEST(BasicKernelTest, LargerCase) {
  run_matrix_mul_and_check(TEST_CASES[3].M, TEST_CASES[3].N, TEST_CASES[3].K, solve);
}

TEST(BasicKernelTest, XLargerCase) {
  run_matrix_mul_and_check(TEST_CASES[4].M, TEST_CASES[4].N, TEST_CASES[4].K, solve);
}

// -----------------------------------------------------------------------------
// Test suite for sliced kernel
// -----------------------------------------------------------------------------

TEST(SlicedKernelTest, SmallCase) {
  run_matrix_mul_and_check(TEST_CASES[0].M, TEST_CASES[0].N, TEST_CASES[0].K, solve_sliced);
}

TEST(SlicedKernelTest, MediumCase) {
  run_matrix_mul_and_check(TEST_CASES[1].M, TEST_CASES[1].N, TEST_CASES[1].K, solve_sliced);
}

TEST(SlicedKernelTest, RectangularCase) {
  run_matrix_mul_and_check(TEST_CASES[2].M, TEST_CASES[2].N, TEST_CASES[2].K, solve_sliced);
}

TEST(SlicedKernelTest, LargerCase) {
  run_matrix_mul_and_check(TEST_CASES[3].M, TEST_CASES[3].N, TEST_CASES[3].K, solve_sliced);
}

TEST(SlicedKernelTest, XLargerCase) {
  run_matrix_mul_and_check(TEST_CASES[4].M, TEST_CASES[4].N, TEST_CASES[4].K, solve_sliced);
}

// -----------------------------------------------------------------------------
// Test suite for double buffer kernel
// -----------------------------------------------------------------------------
TEST(DoubleBufferKernelTest, SmallCase) {
  run_matrix_mul_and_check(TEST_CASES[0].M, TEST_CASES[0].N, TEST_CASES[0].K, solve_double_buffer);
}

TEST(DoubleBufferKernelTest, MediumCase) {
  run_matrix_mul_and_check(TEST_CASES[1].M, TEST_CASES[1].N, TEST_CASES[1].K, solve_double_buffer);
}

TEST(DoubleBufferKernelTest, RectangularCase) {
  run_matrix_mul_and_check(TEST_CASES[2].M, TEST_CASES[2].N, TEST_CASES[2].K, solve_double_buffer);
}

TEST(DoubleBufferKernelTest, LargerCase) {
  run_matrix_mul_and_check(TEST_CASES[3].M, TEST_CASES[3].N, TEST_CASES[3].K, solve_double_buffer);
}

TEST(DoubleBufferKernelTest, XLargerCase) {
  run_matrix_mul_and_check(TEST_CASES[4].M, TEST_CASES[4].N, TEST_CASES[4].K, solve_double_buffer);
}
