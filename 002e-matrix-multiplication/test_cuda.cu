#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// -----------------------------------------------------------------------------
// DUT symbols
// -----------------------------------------------------------------------------
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K);
extern "C" void solve2(const float *A, const float *B, float *C, int M, int N,
                       int K);
extern "C" void solve3(const float *A, const float *B, float *C, int M, int N,
                       int K);
extern "C" void solve4(const float *A, const float *B, float *C, int M, int N,
                       int K);

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      GTEST_FAIL() << "CUDA error: " << cudaGetErrorString(_err) << " at "     \
                   << __FILE__ << ":" << __LINE__;                             \
    }                                                                          \
  } while (0)

struct CudaDeleter {
  void operator()(float *p) const noexcept {
    if (p)
      cudaFree(p);
  }
};
using DevicePtr = std::unique_ptr<float, CudaDeleter>;

// -----------------------------------------------------------------------------
// Parameter specs
// -----------------------------------------------------------------------------
using KernelFunc = void (*)(const float *, const float *, float *, int, int,
                            int);

struct KernelSpec {
  const char *name;
  KernelFunc func;
  float atol =
      std::numeric_limits<float>::quiet_NaN(); // optional per-kernel tol
};

struct MatCase {
  int M, N, K;
  const char *desc; // human-friendly
};

inline std::ostream &operator<<(std::ostream &os, const KernelSpec &k) {
  return os << k.name;
}
inline std::ostream &operator<<(std::ostream &os, const MatCase &c) {
  return os << c.desc << " (M=" << c.M << ",N=" << c.N << ",K=" << c.K << ")";
}

// -----------------------------------------------------------------------------
// Core runner (shared by all parameter combinations)
// -----------------------------------------------------------------------------
static void run_matrix_mul_and_check(int M, int N, int K,
                                     KernelFunc kernel_func, float atol) {
  const size_t sizeA =
      static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
  const size_t sizeB =
      static_cast<size_t>(N) * static_cast<size_t>(K) * sizeof(float);
  const size_t sizeC =
      static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float);

  std::vector<float> h_A(static_cast<size_t>(M) * static_cast<size_t>(N));
  std::vector<float> h_B(static_cast<size_t>(N) * static_cast<size_t>(K));
  std::vector<float> h_C(static_cast<size_t>(M) * static_cast<size_t>(K));

  // Deterministic small
  for (size_t i = 0; i < h_A.size(); ++i)
    h_A[i] = static_cast<float>(i % 5 + 1);
  for (size_t i = 0; i < h_B.size(); ++i)
    h_B[i] = static_cast<float>(i % 7 + 1);

  float *rawA = nullptr, *rawB = nullptr, *rawC = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&rawA), sizeA));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&rawB), sizeB));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&rawC), sizeC));
  DevicePtr d_A(rawA), d_B(rawB), d_C(rawC);

  CUDA_CHECK(cudaMemcpy(d_A.get(), h_A.data(), sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B.get(), h_B.data(), sizeB, cudaMemcpyHostToDevice));

  // Launch and sync
  kernel_func(d_A.get(), d_B.get(), d_C.get(), M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.get(), sizeC, cudaMemcpyDeviceToHost));

  // CPU reference
  std::vector<float> ref(static_cast<size_t>(M) * static_cast<size_t>(K), 0.0f);
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < K; ++c) {
      float sum = 0.0f;
      for (int i = 0; i < N; ++i) {
        sum += h_A[static_cast<size_t>(r) * N + i] *
               h_B[static_cast<size_t>(i) * K + c];
      }
      ref[static_cast<size_t>(r) * K + c] = sum;
    }
  }

  SCOPED_TRACE(testing::Message() << "M=" << M << ", N=" << N << ", K=" << K);
  const float tol = std::isnan(atol) ? 1e-4f : atol;
  for (size_t i = 0; i < h_C.size(); ++i) {
    EXPECT_NEAR(h_C[i], ref[i], tol) << "Mismatch at linear index " << i;
  }
}

// -----------------------------------------------------------------------------
// Parameter sets
// -----------------------------------------------------------------------------
static const KernelSpec kKernels[] = {
    {"solve", &solve, 1e-4f},
    {"solve2", &solve2, 1e-4f},
    {"solve3", &solve3, 1e-4f},
    {"solve4", &solve4, 1e-4f},
};

static const MatCase kCases[] = {
    {2, 3, 2, "2x3 * 3x2"},
    {8, 8, 8, "8x8 * 8x8"},
    {4, 6, 5, "4x6 * 6x5"},
    {32, 64, 16, "32x64 * 64x16"},
    {64, 64, 64, "64x64 * 64x64"},
    {128, 128, 128, "128x128 * 128x128"},
    {256, 256, 256, "256x256 * 256x256"},
    {512, 512, 512, "512x512 * 512x512"},
    // {1024, 1024, 1024, "1024x1024 * 1024x1024"},
};

// -----------------------------------------------------------------------------
// TEST: one body, runs across (kernel, case) combinations
// -----------------------------------------------------------------------------
using ParamT = std::tuple<KernelSpec, MatCase>;

class MatmulTest : public ::testing::TestWithParam<ParamT> {
protected:
  const KernelSpec &kernel() const { return std::get<0>(GetParam()); }
  const MatCase &mcase() const { return std::get<1>(GetParam()); }
};

TEST_P(MatmulTest, Correctness) {
  const auto &k = kernel();
  const auto &c = mcase();
  run_matrix_mul_and_check(c.M, c.N, c.K, k.func, k.atol);
}

// Name generator to make gtest output/filters friendly
static std::string PrettyNameGen(const testing::TestParamInfo<ParamT> &info) {
  const auto &k = std::get<0>(info.param);
  const auto &c = std::get<1>(info.param);
  std::ostringstream oss;
  oss << k.name << "__M" << c.M << "_N" << c.N << "_K" << c.K;
  return oss.str();
}

INSTANTIATE_TEST_SUITE_P(KernelsAndSizes, MatmulTest,
                         ::testing::Combine(::testing::ValuesIn(kKernels),
                                            ::testing::ValuesIn(kCases)),
                         PrettyNameGen);
