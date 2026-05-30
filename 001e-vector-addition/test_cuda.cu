#include "cuda_registry.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <ostream>
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
using Implementation = leetgpu::vector_addition::CudaImplementation;
using Param = std::tuple<Implementation, int>;

void run_and_check(const Implementation &impl, int n)
{
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  std::vector<float> h_a(n), h_b(n), h_c(n);

  for (int i = 0; i < n; i++) {
    h_a[static_cast<std::size_t>(i)] = static_cast<float>(i);
    h_b[static_cast<std::size_t>(i)] = static_cast<float>(i * 2);
  }

  float *raw_a = nullptr;
  float *raw_b = nullptr;
  float *raw_c = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raw_a), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raw_b), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raw_c), bytes));
  DevicePtr d_a(raw_a), d_b(raw_b), d_c(raw_c);

  CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c.get(), 0, bytes));

  impl.func(d_a.get(), d_b.get(), d_c.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));

  SCOPED_TRACE(testing::Message() << "implementation=" << impl.id << ", N=" << n);
  for (int i = 0; i < n; i++) {
    ASSERT_FLOAT_EQ(h_c[static_cast<std::size_t>(i)],
                    h_a[static_cast<std::size_t>(i)] + h_b[static_cast<std::size_t>(i)])
        << "Mismatch at index " << i;
  }
}

class VectorAddCudaTest : public ::testing::TestWithParam<Param> {};

TEST_P(VectorAddCudaTest, Correctness)
{
  const auto &[impl, n] = GetParam();
  run_and_check(impl, n);
}

std::string pretty_name(const testing::TestParamInfo<Param> &info)
{
  const auto &[impl, n] = info.param;
  std::ostringstream oss;
  oss << impl.id << "__N" << n;
  return oss.str();
}

INSTANTIATE_TEST_SUITE_P(
    ImplementationsAndSizes,
    VectorAddCudaTest,
    ::testing::Combine(::testing::ValuesIn(leetgpu::vector_addition::kCudaImplementations),
                       ::testing::ValuesIn(leetgpu::vector_addition::kTestSizes)),
    pretty_name);

} // namespace
