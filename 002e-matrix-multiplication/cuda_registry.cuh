#pragma once

#include <array>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" void solve0(const float *A, const float *B, float *C, int M, int N, int K);
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K);
extern "C" void solve2(const float *A, const float *B, float *C, int M, int N, int K);
extern "C" void solve3(const float *A, const float *B, float *C, int M, int N, int K);
extern "C" void solve4(const float *A, const float *B, float *C, int M, int N, int K);
extern "C" void solve5(const float *A, const float *B, float *C, int M, int N, int K);

namespace leetgpu::matrix_multiplication {

using KernelFunc = void (*)(const float *, const float *, float *, int, int, int);

struct CudaImplementation {
  const char *id;
  const char *description;
  KernelFunc func;
  float atol;
};

struct MatrixCase {
  const char *id;
  int m;
  int n;
  int k;
  const char *description;
};

inline const std::array<CudaImplementation, 6> kCudaImplementations{{
    {"uncoalesced_2d", "2D global-memory baseline with poor coalescing", &solve0, 1e-4f},
    {"gmem_2d", "2D global-memory baseline", &solve, 1e-4f},
    {"gmem_1d", "1D-thread global-memory baseline", &solve2, 1e-4f},
    {"smem_2d", "shared-memory tiling", &solve3, 1e-4f},
    {"smem_1d", "shared-memory tiling with 1D thread indexing", &solve4, 1e-4f},
    {"smem_float4", "shared-memory tiling with float4 global loads", &solve5, 1e-4f},
}};

inline void PrintTo(const CudaImplementation &impl, std::ostream *os)
{
  *os << impl.id;
}

inline const std::array<MatrixCase, 8> kTestCases{{
    {"tiny_rect", 2, 2, 3, "2x3 * 3x2"},
    {"small_square", 8, 8, 8, "8x8 * 8x8"},
    {"small_rect", 4, 5, 6, "4x6 * 6x5"},
    {"wide_rect", 32, 16, 64, "32x64 * 64x16"},
    {"tile_tail", 65, 33, 17, "65x17 * 17x33"},
    {"large_rect", 96, 80, 64, "96x64 * 64x80"},
    {"square_128", 128, 128, 128, "128x128 * 128x128"},
    {"square_256", 256, 256, 256, "256x256 * 256x256"},
}};

inline void PrintTo(const MatrixCase &matrix_case, std::ostream *os)
{
  *os << matrix_case.id << "(M=" << matrix_case.m << ",N=" << matrix_case.n
      << ",K=" << matrix_case.k << ")";
}

inline const std::array<MatrixCase, 4> kBenchmarkCases{{
    {"tiny_rect", 2, 2, 3, "2x3 * 3x2"},
    {"square_128", 128, 128, 128, "128x128 * 128x128"},
    {"square_512", 512, 512, 512, "512x512 * 512x512"},
    {"square_1024", 1024, 1024, 1024, "1024x1024 * 1024x1024"},
}};

inline std::vector<std::string> implementation_ids()
{
  std::vector<std::string> ids;
  ids.reserve(kCudaImplementations.size());
  for (const auto &impl : kCudaImplementations) {
    ids.emplace_back(impl.id);
  }
  return ids;
}

inline std::vector<std::string> benchmark_case_ids()
{
  std::vector<std::string> ids;
  ids.reserve(kBenchmarkCases.size());
  for (const auto &matrix_case : kBenchmarkCases) {
    ids.emplace_back(matrix_case.id);
  }
  return ids;
}

inline const CudaImplementation &find_implementation(std::string_view id)
{
  for (const auto &impl : kCudaImplementations) {
    if (impl.id == id) {
      return impl;
    }
  }
  throw std::invalid_argument("unknown matrix-multiplication CUDA implementation: " +
                              std::string{id});
}

inline const MatrixCase &find_benchmark_case(std::string_view id)
{
  for (const auto &matrix_case : kBenchmarkCases) {
    if (matrix_case.id == id) {
      return matrix_case;
    }
  }
  throw std::invalid_argument("unknown matrix-multiplication benchmark case: " +
                              std::string{id});
}

} // namespace leetgpu::matrix_multiplication
