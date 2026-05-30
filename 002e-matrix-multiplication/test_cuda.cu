#include "cuda_registry.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                                        \
  do {                                                                                          \
    cudaError_t err = (call);                                                                   \
    if (err != cudaSuccess) {                                                                   \
      GTEST_FAIL() << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":"  \
                   << __LINE__;                                                                 \
    }                                                                                           \
  } while (0)

struct CudaDeleter {
  void operator()(float *p) const noexcept
  {
    if (p) {
      cudaFree(p);
    }
  }
};

using DevicePtr = std::unique_ptr<float, CudaDeleter>;
using Implementation = leetgpu::matrix_multiplication::CudaImplementation;
using MatrixCase = leetgpu::matrix_multiplication::MatrixCase;
using Param = std::tuple<Implementation, MatrixCase>;

void run_matrix_mul_and_check(const Implementation &impl, const MatrixCase &matrix_case)
{
  const int m = matrix_case.m;
  const int n = matrix_case.n;
  const int k = matrix_case.k;
  const auto size_a = static_cast<std::size_t>(m) * static_cast<std::size_t>(k);
  const auto size_b = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
  const auto size_c = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

  std::vector<float> h_a(size_a);
  std::vector<float> h_b(size_b);
  std::vector<float> h_c(size_c);

  for (std::size_t i = 0; i < h_a.size(); ++i) {
    h_a[i] = static_cast<float>(i % 5 + 1);
  }
  for (std::size_t i = 0; i < h_b.size(); ++i) {
    h_b[i] = static_cast<float>(i % 7 + 1);
  }

  float *raw_a = nullptr;
  float *raw_b = nullptr;
  float *raw_c = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raw_a), size_a * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raw_b), size_b * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raw_c), size_c * sizeof(float)));
  DevicePtr d_a(raw_a), d_b(raw_b), d_c(raw_c);

  CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), size_a * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), size_b * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c.get(), 0, size_c * sizeof(float)));

  impl.func(d_a.get(), d_b.get(), d_c.get(), m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.get(), size_c * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> ref(size_c, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        sum += h_a[static_cast<std::size_t>(row) * k + inner] *
               h_b[static_cast<std::size_t>(inner) * n + col];
      }
      ref[static_cast<std::size_t>(row) * n + col] = sum;
    }
  }

  SCOPED_TRACE(testing::Message() << "implementation=" << impl.id << ", case=" << matrix_case.id
                                  << " (M=" << m << ", N=" << n << ", K=" << k << ")");
  for (std::size_t i = 0; i < h_c.size(); ++i) {
    EXPECT_NEAR(h_c[i], ref[i], impl.atol) << "Mismatch at linear index " << i;
  }
}

class MatmulCudaTest : public ::testing::TestWithParam<Param> {};

TEST_P(MatmulCudaTest, Correctness)
{
  const auto &[impl, matrix_case] = GetParam();
  run_matrix_mul_and_check(impl, matrix_case);
}

std::string pretty_name(const testing::TestParamInfo<Param> &info)
{
  const auto &[impl, matrix_case] = info.param;
  std::ostringstream oss;
  oss << impl.id << "__" << matrix_case.id << "__M" << matrix_case.m << "_N" << matrix_case.n
      << "_K" << matrix_case.k;
  return oss.str();
}

INSTANTIATE_TEST_SUITE_P(
    ImplementationsAndCases,
    MatmulCudaTest,
    ::testing::Combine(::testing::ValuesIn(leetgpu::matrix_multiplication::kCudaImplementations),
                       ::testing::ValuesIn(leetgpu::matrix_multiplication::kTestCases)),
    pretty_name);

} // namespace
