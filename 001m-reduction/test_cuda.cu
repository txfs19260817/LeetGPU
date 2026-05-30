#include "cuda_registry.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
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
using Implementation = leetgpu::reduction::CudaImplementation;

float cpu_reference_sum(const std::vector<float> &h_in)
{
  long double acc = 0.0L;
  for (float x : h_in) {
    acc += static_cast<long double>(x);
  }
  return static_cast<float>(acc);
}

void run_reduce_and_check(const std::vector<float> &h_in, const Implementation &impl)
{
  const int n = static_cast<int>(h_in.size());
  ASSERT_GT(n, 0) << "Input must be non-empty.";
  const auto bytes_in = static_cast<std::size_t>(n) * sizeof(float);

  float *raw_in = nullptr;
  float *raw_out = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raw_in), bytes_in));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raw_out), sizeof(float)));
  DevicePtr d_in(raw_in), d_out(raw_out);

  CUDA_CHECK(cudaMemcpy(d_in.get(), h_in.data(), bytes_in, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.get(), 0, sizeof(float)));

  impl.func(d_in.get(), d_out.get(), n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  float got = 0.0f;
  CUDA_CHECK(cudaMemcpy(&got, d_out.get(), sizeof(float), cudaMemcpyDeviceToHost));

  const float ref = cpu_reference_sum(h_in);
  SCOPED_TRACE(testing::Message() << "implementation=" << impl.id << ", N=" << n);
  EXPECT_NEAR(got, ref, impl.atol) << "Expected: " << ref << ", Got: " << got
                                   << ", Abs. Error: " << std::abs(got - ref)
                                   << ", Tolerance: " << impl.atol;
}

std::vector<float> seq(std::initializer_list<float> xs)
{
  return {xs};
}

std::vector<float> zeros(int n)
{
  return std::vector<float>(static_cast<std::size_t>(n), 0.0f);
}

std::vector<float> ones(int n)
{
  return std::vector<float>(static_cast<std::size_t>(n), 1.0f);
}

std::vector<float> uniform(int n, float lo, float hi, std::uint64_t seed)
{
  std::vector<float> v(static_cast<std::size_t>(n));
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (int i = 0; i < n; ++i) {
    v[static_cast<std::size_t>(i)] = dist(rng);
  }
  return v;
}

class ReductionCudaTest : public ::testing::TestWithParam<Implementation> {
protected:
  const Implementation &impl() const { return GetParam(); }
};

TEST_P(ReductionCudaTest, BasicExample)
{
  run_reduce_and_check(seq({1, 2, 3, 4, 5, 6, 7, 8}), impl());
}

TEST_P(ReductionCudaTest, NegativeNumbers)
{
  run_reduce_and_check(seq({-2.5f, 1.5f, -1.0f, 2.0f}), impl());
}

TEST_P(ReductionCudaTest, SingleElement)
{
  run_reduce_and_check(seq({42.0f}), impl());
}

TEST_P(ReductionCudaTest, AllZeros1024)
{
  run_reduce_and_check(zeros(1024), impl());
}

TEST_P(ReductionCudaTest, AllOnes1024)
{
  run_reduce_and_check(ones(1024), impl());
}

TEST_P(ReductionCudaTest, NonPowerOfTwo)
{
  run_reduce_and_check(seq({1, 2, 3, 4, 5}), impl());
}

TEST_P(ReductionCudaTest, LargeRandom10k)
{
  run_reduce_and_check(uniform(10'000, -1000.0f, 1000.0f, 123), impl());
}

TEST_P(ReductionCudaTest, LargeRandom1M)
{
  run_reduce_and_check(uniform(1 << 20, -100.0f, 100.0f, 321), impl());
}

INSTANTIATE_TEST_SUITE_P(
    Implementations,
    ReductionCudaTest,
    ::testing::ValuesIn(leetgpu::reduction::kCudaImplementations),
    [](const testing::TestParamInfo<ReductionCudaTest::ParamType> &info) {
      return std::string(info.param.id);
    });

} // namespace
