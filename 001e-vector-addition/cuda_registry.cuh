#pragma once

#include <array>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" void solve(const float *A, const float *B, float *C, int N);
extern "C" void solve_stride(const float *A, const float *B, float *C, int N);
extern "C" void solve_vec(const float *A, const float *B, float *C, int N);
extern "C" void solve_vec2(const float *A, const float *B, float *C, int N);

namespace leetgpu::vector_addition {

using KernelFunc = void (*)(const float *, const float *, float *, int);

struct CudaImplementation {
  const char *id;
  const char *description;
  KernelFunc func;
};

inline const std::array<CudaImplementation, 4> kCudaImplementations{{
    {"basic", "one thread per output element", &solve},
    {"grid_stride", "grid-stride loop", &solve_stride},
    {"float4_loop", "float4 loop with scalar tail", &solve_vec},
    {"float4_direct", "direct float4 loads with scalar tail", &solve_vec2},
}};

inline void PrintTo(const CudaImplementation &impl, std::ostream *os)
{
  *os << impl.id;
}

inline const std::array<int, 6> kTestSizes{
    1,
    2,
    3,
    1024,
    2048 + 3,
    1 << 20,
};

inline const std::array<std::int64_t, 6> kBenchmarkSizes{
    1024,
    2048 + 3,
    1 << 16,
    1 << 20,
    1 << 26,
    1 << 28,
};

inline std::vector<std::string> implementation_ids()
{
  std::vector<std::string> ids;
  ids.reserve(kCudaImplementations.size());
  for (const auto &impl : kCudaImplementations) {
    ids.emplace_back(impl.id);
  }
  return ids;
}

inline std::vector<std::int64_t> benchmark_sizes()
{
  return {kBenchmarkSizes.begin(), kBenchmarkSizes.end()};
}

inline const CudaImplementation &find_implementation(std::string_view id)
{
  for (const auto &impl : kCudaImplementations) {
    if (impl.id == id) {
      return impl;
    }
  }
  throw std::invalid_argument("unknown vector-add CUDA implementation: " + std::string{id});
}

} // namespace leetgpu::vector_addition
