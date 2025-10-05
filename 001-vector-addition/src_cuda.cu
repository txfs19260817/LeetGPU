#include <cuda_runtime.h>

__global__ void vector_add(const float *A, const float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
}

// Vector add with stride
__global__ void vector_add_stride(const float *A, const float *B, float *C,
                                  int N) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    C[i] = A[i] + B[i];
  }
}

// Vector add with vectorized access via casting to float4
__global__ void vector_add_vectorized(const float *A, const float *B, float *C,
                                      int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int vecN = N / 4;

  for (int j = i; j < vecN; j += gridDim.x * blockDim.x) {
    float4 a = reinterpret_cast<const float4 *>(A)[j];
    float4 b = reinterpret_cast<const float4 *>(B)[j];
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    reinterpret_cast<float4 *>(C)[j] = c;
  }

  // Only the first thread in the block will add the remaining elements
  int leftover = N % 4;
  int base = N - leftover;
  if (i == 0)
    for (int j = 0; j < leftover; j++)
      C[base + j] = A[base + j] + B[base + j];
}

__global__ void vector_add_vectorized2(const float *A, const float *B, float *C,
                                       int N) {
  int i = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
  if (i < N) {
    float4 a = reinterpret_cast<const float4 *>(&(A[i]))[0];
    float4 b = reinterpret_cast<const float4 *>(&(B[i]))[0];
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    reinterpret_cast<float4 *>(&(C[i]))[0] = c;
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  cudaDeviceSynchronize();
}

extern "C" void solve_stride(const float *A, const float *B, float *C, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  vector_add_stride<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  cudaDeviceSynchronize();
}

extern "C" void solve_vec(const float *A, const float *B, float *C, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  vector_add_vectorized<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  cudaDeviceSynchronize();
}

extern "C" void solve_vec2(const float *A, const float *B, float *C, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  vector_add_vectorized2<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
  cudaDeviceSynchronize();
}