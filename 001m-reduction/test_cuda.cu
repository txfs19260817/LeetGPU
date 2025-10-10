#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

// DUT symbols
extern "C" void solve(const float *input, float *output, int N);
// extern "C" void solve_2(const float* input, float* output, int N);
// extern "C" void solve_atomic(const float* input, float* output, int N);

// Utilities
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

static float cpu_reference_sum(const std::vector<float> &h_in) {
  long double acc = 0.0L;
  for (float x : h_in)
    acc += static_cast<long double>(x);
  return static_cast<float>(acc);
}

using KernelFunc = void (*)(const float *, float *, int);

static void run_reduce_and_check(const std::vector<float> &h_in,
                                 KernelFunc kernel_func, float atol = 1e-4f) {
  const int N = static_cast<int>(h_in.size());
  ASSERT_GT(N, 0) << "Input must be non-empty.";
  const size_t bytes_in = static_cast<size_t>(N) * sizeof(float);

  // Device allocations (RAII-managed)
  float *raw_in = nullptr, *raw_out = nullptr;
  CUDA_CHECK(cudaMalloc((void **)&raw_in, bytes_in));
  CUDA_CHECK(cudaMalloc((void **)&raw_out, sizeof(float)));
  DevicePtr d_in(raw_in), d_out(raw_out);

  // Copy input from host to device
  CUDA_CHECK(
      cudaMemcpy(d_in.get(), h_in.data(), bytes_in, cudaMemcpyHostToDevice));
  // (Optional) poison the output so we know kernel overwrites it
  const uint32_t poison = 0x7f800001u; // a signaling-NaN-ish pattern
  CUDA_CHECK(
      cudaMemcpy(d_out.get(), &poison, sizeof(float), cudaMemcpyHostToDevice));

  // Launch kernel
  kernel_func(d_in.get(), d_out.get(), N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy output from device to host
  float got = 0.0f;
  CUDA_CHECK(
      cudaMemcpy(&got, d_out.get(), sizeof(float), cudaMemcpyDeviceToHost));

  // Verify with CPU reference
  float ref = cpu_reference_sum(h_in);
  SCOPED_TRACE(testing::Message() << "N=" << N);
  EXPECT_NEAR(got, ref, atol) << "Expected: " << ref << ", Got: " << got << ", Abs. Error: " << std::abs(got - ref) << ", Tolerance: " << atol;
}

// Simple input builders
static std::vector<float> seq(std::initializer_list<float> xs) { return {xs}; }
static std::vector<float> zeros(int N) {
  return std::vector<float>((size_t)N, 0.0f);
}
static std::vector<float> ones(int N) {
  return std::vector<float>((size_t)N, 1.0f);
}
static std::vector<float> uniform(int N, float lo, float hi,
                                  uint64_t seed = 42) {
  std::vector<float> v((size_t)N);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (int i = 0; i < N; ++i)
    v[(size_t)i] = dist(rng);
  return v;
}

// Parameterization over kernels
struct KernelSpec {
  const char *name;
  KernelFunc func;
  float atol = std::numeric_limits<float>::quiet_NaN();
};

std::ostream &operator<<(std::ostream &os, const KernelSpec &k) {
  return os << k.name; // so names appear nicely in test output
}

class ReduceKernelTest : public ::testing::TestWithParam<KernelSpec> {
protected:
  KernelFunc kernel() const { return GetParam().func; }
  float atol() const { return GetParam().atol; }
};

// Register the kernels you want to test here:
static const KernelSpec kKernels[] = {
    {"solve", &solve, 100.0f},  // `solve` is error prone due to “floating-point atomics are non-deterministic”
    // {"solve_2", &solve_2},
    // {"solve_atomic", &solve_atomic}, // include if you have it
};

INSTANTIATE_TEST_SUITE_P(
    Kernels, ReduceKernelTest, ::testing::ValuesIn(kKernels),
    [](const testing::TestParamInfo<ReduceKernelTest::ParamType> &info) {
      return std::string(info.param.name); // test name suffix = kernel name
    });

// -----------------------------------------------------------------------------
// Write each test ONCE — it runs for every kernel above
// -----------------------------------------------------------------------------
TEST_P(ReduceKernelTest, BasicExample) {
  run_reduce_and_check(seq({1, 2, 3, 4, 5, 6, 7, 8}), kernel(), (std::isnan(atol())) ? 1e-6f : atol());
}

TEST_P(ReduceKernelTest, NegativeNumbers) {
  run_reduce_and_check(seq({-2.5f, 1.5f, -1.0f, 2.0f}), kernel(), (std::isnan(atol())) ? 1e-6f : atol());
}

TEST_P(ReduceKernelTest, SingleElement) {
  run_reduce_and_check(seq({42.0f}), kernel(), (std::isnan(atol())) ? 0.0f : atol());
}

TEST_P(ReduceKernelTest, AllZeros_1024) {
  run_reduce_and_check(zeros(1024), kernel(), (std::isnan(atol())) ? 0.0f : atol());
}

TEST_P(ReduceKernelTest, AllOnes_1024) {
  run_reduce_and_check(ones(1024), kernel(), (std::isnan(atol())) ? 1e-6f : atol());
}

TEST_P(ReduceKernelTest, NonPowerOfTwo) {
  run_reduce_and_check(seq({1, 2, 3, 4, 5}), kernel(), (std::isnan(atol())) ? 1e-6f : atol());
}

TEST_P(ReduceKernelTest, LargeRandom_10k) {
  run_reduce_and_check(uniform(10000, -1000.f, 1000.f, 123), kernel(), (std::isnan(atol())) ? 1e-3f : atol());
}

TEST_P(ReduceKernelTest, LargeRandom_15M) {
  run_reduce_and_check(uniform(15'000'000, -1000.f, 1000.f, 321), kernel(), (std::isnan(atol())) ? 1e-2f : atol());
}
