#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 1024

// ===== Reduction Kernels =====

// Baseline: warp-divergence kernel
__global__ void reduce0(const float *__restrict__ d_in,
                        float *__restrict__ d_out) {
  // Each thread loads 1 element from GMEM to SMEM
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = d_in[i];
  __syncthreads();

  // Do reduction in SMEM
  // Problem: odd number threads don't participate in the reduction but only
  // wait for the even number threads to finish
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // Write result for this block back to GMEM
  if (tid == 0)
    d_out[blockIdx.x] = sdata[0];
}

__global__ void reduce0_final_stage(const float *d_in, float *d_out, int N) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = (i < N) ? d_in[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(d_out, sdata[0]);
}

// V1: warp-divergence free but bank-conflict kernel
__global__ void reduce1(const float *__restrict__ d_in,
                        float *__restrict__ d_out) {
  // Each thread loads 1 element from GMEM to SMEM
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = d_in[i];
  __syncthreads();

  // Do reduction in SMEM
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (int index = 2 * s * tid; index < blockDim.x) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }

  // Write result for this block back to GMEM
  if (tid == 0)
    d_out[blockIdx.x] = sdata[0];
}

__global__ void reduce1_final_stage(const float *d_in, float *d_out, int N) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = (i < N) ? d_in[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(d_out, sdata[0]);
}

// V2: warp-divergence free and bank-conflict free kernel
__global__ void reduce2(const float *__restrict__ d_in,
                        float *__restrict__ d_out) {
  // Each thread loads 1 element from GMEM to SMEM
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  sdata[tid] = d_in[i];
  __syncthreads();

  // Do reduction in SMEM
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // Write result for this block back to GMEM
  if (tid == 0)
    d_out[blockIdx.x] = sdata[0];
}

__global__ void reduce2_final_stage(const float *d_in, float *d_out, int N) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  sdata[tid] = (i < N) ? d_in[i] : 0.0f;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(d_out, sdata[0]);
}

// V3: idle thread free kernel
__global__ void reduce3(const float *__restrict__ d_in,
                        float *__restrict__ d_out) {
  // Each thread loads 2 elements from GMEM to SMEM
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
  sdata[tid] = d_in[i] + d_in[i + blockDim.x];
  __syncthreads();

  // Do reduction in SMEM
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // Write result for this block back to GMEM
  if (tid == 0)
    d_out[blockIdx.x] = sdata[0];
}

__global__ void reduce3_final_stage(const float *d_in, float *d_out, int N) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = (blockDim.x) * blockIdx.x + threadIdx.x;

  float val = 0.0f;
  for (; i < N; i += blockDim.x * gridDim.x)
    val += d_in[i];
  sdata[tid] = val;
  __syncthreads();

  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(d_out, sdata[0]);
}

// V4: warp-reduce saving __syncthreads() on the last round
__device__ void warp_reduce(volatile float *cache, int tid) {
  cache[tid] += cache[tid + 32];
  cache[tid] += cache[tid + 16];
  cache[tid] += cache[tid + 8];
  cache[tid] += cache[tid + 4];
  cache[tid] += cache[tid + 2];
  cache[tid] += cache[tid + 1];
}

__global__ void reduce4(const float *__restrict__ d_in,
                        float *__restrict__ d_out) {
  extern __shared__ float sdata[];

  // Each thread loads 2 elements from GMEM to SMEM
  unsigned int tid = threadIdx.x;
  unsigned int i = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
  sdata[tid] = d_in[i] + d_in[i + blockDim.x];
  __syncthreads();

  // Do reduction in SMEM: loop ends when s == 32
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // 32 threads within a warp are synchronized so
  // __syncthreads(); is unnecessary
  if (tid < 32)
    warp_reduce(sdata, tid);

  // Write result for this block back to GMEM
  if (tid == 0)
    d_out[blockIdx.x] = sdata[tid];
}

__global__ void reduce4_final_stage(const float *d_in, float *d_out, int N) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  float val = 0.0f;
  for (; i < N; i += blockDim.x * gridDim.x)
    val += d_in[i];
  sdata[tid] = val;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // 32 threads within a warp are synchronized so
  // __syncthreads(); is unnecessary
  if (tid < 32)
    warp_reduce(sdata, tid);

  if (tid == 0)
    atomicAdd(d_out, sdata[0]);
}

// V5: warp-reduce with __shfl_down_sync
__device__ __forceinline__ float warp_reduce_sum(float v, unsigned mask) {
  v += __shfl_down_sync(mask, v, 16);  // add the value from thread with ID 16
  v += __shfl_down_sync(mask, v, 8);  // add the value from thread with ID 8
  v += __shfl_down_sync(mask, v, 4);  // add the value from thread with ID 4
  v += __shfl_down_sync(mask, v, 2);  // add the value from thread with ID 2
  v += __shfl_down_sync(mask, v, 1);  // add the value from thread with ID 1
  return v;
}

__global__ void reduce5(const float *__restrict__ d_in,
                        float *__restrict__ d_out) {
  extern __shared__ float sdata[];

  // Each thread loads 2 elements from GMEM to SMEM
  unsigned int tid = threadIdx.x;
  unsigned int i = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
  sdata[tid] = d_in[i] + d_in[i + blockDim.x];
  __syncthreads();

  // Do reduction in SMEM
  // Note: s >= 32, unlike kernel4 s > 32
  for (unsigned int s = blockDim.x / 2; s >= 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  unsigned mask = __activemask();
  float sum = sdata[tid];           // SMEM -> register
  sum = warp_reduce_sum(sum, mask); // register-only reduction
  // Write result for this block back to GMEM
  if (tid == 0)
    d_out[blockIdx.x] = sum;
}

__global__ void reduce5_final_stage(const float *d_in, float *d_out, int N) {
  extern __shared__ float sdata[];

  // Each thread loads 2 elements from GMEM to SMEM
  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  float val = 0.0f;
  for (; i < N; i += blockDim.x * gridDim.x)
    val += d_in[i];
  sdata[tid] = val;
  __syncthreads();

  // Do reduction in SMEM
  // Note: s >= 32, unlike kernel4 s > 32
  for (unsigned int s = blockDim.x / 2; s >= 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  unsigned mask = __activemask();
  float sum = sdata[tid];           // SMEM -> register
  sum = warp_reduce_sum(sum, mask); // register-only reduction
  // Write result for this block back to GMEM
  if (tid == 0)
    atomicAdd(d_out, sum);
}

// V6: warp-reduce with tiled_partition
__global__ void reduce6(const float *__restrict__ d_in,
                        float *__restrict__ d_out) {
  extern __shared__ float sdata[];

  // Each thread loads 2 elements from GMEM to SMEM
  unsigned int tid = threadIdx.x;
  unsigned int i = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
  sdata[tid] = d_in[i] + d_in[i + blockDim.x];
  __syncthreads();

  // Do reduction in SMEM
  for (unsigned int s = blockDim.x / 2; s >= 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  cooperative_groups::thread_block_tile<32> g =
      cooperative_groups::tiled_partition<32>(
          cooperative_groups::this_thread_block());

  float sum = sdata[tid]; // SMEM -> register
  sum += g.shfl_down(sum, 16);
  sum += g.shfl_down(sum, 8);
  sum += g.shfl_down(sum, 4);
  sum += g.shfl_down(sum, 2);
  sum += g.shfl_down(sum, 1);
  // Write result for this block back to GMEM
  if (tid == 0)
    d_out[blockIdx.x] = sum;
}

__global__ void reduce6_final_stage(const float *d_in, float *d_out, int N) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  float val = 0.0f;
  for (; i < N; i += blockDim.x * gridDim.x)
    val += d_in[i];
  sdata[tid] = val;
  __syncthreads();

  // Do reduction in SMEM
  for (unsigned int s = blockDim.x / 2; s >= 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  cooperative_groups::thread_block_tile<32> g =
      cooperative_groups::tiled_partition<32>(
          cooperative_groups::this_thread_block());

  float sum = sdata[tid]; // SMEM -> register
  sum += g.shfl_down(sum, 16);
  sum += g.shfl_down(sum, 8);
  sum += g.shfl_down(sum, 4);
  sum += g.shfl_down(sum, 2);
  sum += g.shfl_down(sum, 1);
  // Write result for this block back to GMEM
  if (tid == 0)
    atomicAdd(d_out, sum);
}

// ===== Solvers =====

// input, output are device pointers
template <int ThreadsPerBlock = THREAD_PER_BLOCK, int BlockStride = 1,
          typename ReduceKernel, typename ReduceFinalKernel>
static void solve_impl(const float *input, float *output, int N,
                       ReduceKernel reduce_kernel,
                       ReduceFinalKernel reduce_final_kernel) {
  cudaMemset(output, 0, sizeof(float));

  const int full_blocks = N / (BlockStride * ThreadsPerBlock);
  const int tail = N % (BlockStride * ThreadsPerBlock);
  const size_t shmem = ThreadsPerBlock * sizeof(float);

  if (full_blocks == 0) {
    const int blocks = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    reduce_final_kernel<<<blocks, ThreadsPerBlock, shmem>>>(input, output, N);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
    return;
  }

  float *d_partial = nullptr;
  cudaMalloc(&d_partial, full_blocks * sizeof(float));

  reduce_kernel<<<full_blocks, ThreadsPerBlock, shmem>>>(input, d_partial);
  cudaPeekAtLastError();
  cudaDeviceSynchronize();

  {
    const int num_partial = full_blocks;
    const int blocks = (num_partial + ThreadsPerBlock - 1) / ThreadsPerBlock;
    reduce_final_kernel<<<blocks, ThreadsPerBlock, shmem>>>(d_partial, output,
                                                            num_partial);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
  }

  if (tail > 0) {
    const float *tail_ptr =
        input + (size_t)full_blocks * (BlockStride * ThreadsPerBlock);

    reduce_final_kernel<<<1, ThreadsPerBlock, shmem>>>(tail_ptr, output, tail);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();
  }

  cudaFree(d_partial);
}

extern "C" void solve(const float *input, float *output, int N) {
  solve_impl<THREAD_PER_BLOCK>(input, output, N, reduce0, reduce0_final_stage);
}

extern "C" void solve1(const float *input, float *output, int N) {
  solve_impl<THREAD_PER_BLOCK>(input, output, N, reduce1, reduce1_final_stage);
}

extern "C" void solve2(const float *input, float *output, int N) {
  solve_impl<THREAD_PER_BLOCK>(input, output, N, reduce2, reduce2_final_stage);
}

extern "C" void solve3(const float *input, float *output, int N) {
  solve_impl<THREAD_PER_BLOCK, 2>(input, output, N, reduce3,
                                  reduce3_final_stage);
}

extern "C" void solve4(const float *input, float *output, int N) {
  solve_impl<THREAD_PER_BLOCK, 2>(input, output, N, reduce4,
                                  reduce4_final_stage);
}

extern "C" void solve5(const float *input, float *output, int N) {
  solve_impl<THREAD_PER_BLOCK, 2>(input, output, N, reduce5,
                                  reduce5_final_stage);
}

extern "C" void solve6(const float *input, float *output, int N) {
  solve_impl<THREAD_PER_BLOCK, 2>(input, output, N, reduce6,
                                  reduce6_final_stage);
}

extern "C" void solve_cub(const float *input, float *output, int N) {
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // 1. First call, get the required temporary storage size
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, output, N);

  // 2. Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // 3. Second call, execute the actual reduction calculation
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input, output, N);

  // Release temporary storage
  cudaFree(d_temp_storage);
}