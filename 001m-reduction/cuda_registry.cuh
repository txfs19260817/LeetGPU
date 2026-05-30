#pragma once

#include <array>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" void solve(const float *input, float *output, int N);
extern "C" void solve1(const float *input, float *output, int N);
extern "C" void solve2(const float *input, float *output, int N);
extern "C" void solve3(const float *input, float *output, int N);
extern "C" void solve4(const float *input, float *output, int N);
extern "C" void solve5(const float *input, float *output, int N);
extern "C" void solve6(const float *input, float *output, int N);
extern "C" void solve_cub(const float *input, float *output, int N);

namespace leetgpu::reduction {

using KernelFunc = void (*)(const float *, float *, int);

struct CudaImplementation {
  const char *id;
  const char *description;
  KernelFunc func;
  float atol;
};

inline const std::array<CudaImplementation, 8> kCudaImplementations{{
    {"baseline", "interleaved addressing with warp divergence", &solve, 0.5f},
    {"sequential_addressing", "sequential addressing", &solve1, 0.5f},
    {"halving", "sequential halving reduction", &solve2, 0.5f},
    {"two_loads", "two input elements per thread", &solve3, 0.5f},
    {"warp_unroll", "shared-memory warp tail unroll", &solve4, 0.5f},
    {"warp_shuffle", "warp shuffle tail reduction", &solve5, 0.5f},
    {"cooperative_groups", "cooperative groups warp tile", &solve6, 0.5f},
    {"cub", "CUB DeviceReduce::Sum", &solve_cub, 0.5f},
}};

inline void PrintTo(const CudaImplementation &impl, std::ostream *os)
{
  *os << impl.id;
}

inline const std::array<std::int64_t, 6> kBenchmarkSizes{
    10'000,
    65'535,
    1 << 20,
    1 << 24,
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
  throw std::invalid_argument("unknown reduction CUDA implementation: " + std::string{id});
}

} // namespace leetgpu::reduction
