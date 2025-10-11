#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// Baseline: warp-divergence kernel
__global__ void reduce0(const float* __restrict__ d_in,
                        float* __restrict__ d_out,
                        int N) {
    // Each thread loads 1 element from GMEM to SMEM
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    sdata[tid] = (i < N) ? d_in[i] : .0f;  // Load valid data or 0 if out of range
    __syncthreads();

    // Do reduction in SMEM
    // Problem: odd number threads don't participate in the reduction but only wait for the even number threads to finish
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if (tid % (2 * s) == 0) {
        sdata[tid] += sdata[tid + s];
      }
      __syncthreads();
    }

    // Write result for this block back to GMEM
    // Note that it has precision issue, leading to big error when N is large.
    if (tid == 0) atomicAdd(d_out, sdata[0]);
}

// V1: warp-divergence free but bank-conflict kernel
__global__ void reduce1(const float* __restrict__ d_in,
                        float* __restrict__ d_out,
                        int N) {
    // Each thread loads 1 element from GMEM to SMEM
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    sdata[tid] = (i < N) ? d_in[i] : .0f;  // Load valid data or 0 if out of range
    __syncthreads();

    // Do reduction in SMEM
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if (int index = 2 * s * tid; index < blockDim.x) {
        sdata[index] += sdata[index + s];
      }
      __syncthreads();
    }

    // Write result for this block back to GMEM
    // Note that it has precision issue, leading to big error when N is large.
    if (tid == 0) atomicAdd(d_out, sdata[0]);
}

// V2: warp-divergence free and bank-conflict free kernel
__global__ void reduce2(const float* __restrict__ d_in,
                        float* __restrict__ d_out,
                        int N) {
    // Each thread loads 1 element from GMEM to SMEM
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    sdata[tid] = (i < N) ? d_in[i] : .0f;  // Load valid data or 0 if out of range
    __syncthreads();

    // Do reduction in SMEM
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }

    // Write result for this block back to GMEM
    // Note that it has precision issue, leading to big error when N is large.
    if (tid == 0) atomicAdd(d_out, sdata[0]);
}

// V3: idle thread free kernel
__global__ void reduce3(const float* __restrict__ d_in,
                        float* __restrict__ d_out,
                        int N) {
    // Each thread loads 1 element from GMEM to SMEM
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
    sdata[tid] = (i < N) ? (d_in[i] + d_in[i+blockDim.x]) : .0f;  // Load valid data or 0 if out of range
    __syncthreads();

    // Do reduction in SMEM
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }

    // Write result for this block back to GMEM
    // Note that it has precision issue, leading to big error when N is large.
    if (tid == 0) atomicAdd(d_out, sdata[0]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  cudaMemsetAsync(output, 0, sizeof(float));
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  int sharedMemorySize = threadsPerBlock*sizeof(float);

  reduce0<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(input, output, N);
  cudaDeviceSynchronize();
}

extern "C" void solve1(const float* input, float* output, int N) {
  cudaMemsetAsync(output, 0, sizeof(float));
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  int sharedMemorySize = threadsPerBlock*sizeof(float);

  reduce1<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(input, output, N);
  cudaDeviceSynchronize();
}

extern "C" void solve2(const float* input, float* output, int N) {
  cudaMemsetAsync(output, 0, sizeof(float));
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  int sharedMemorySize = threadsPerBlock*sizeof(float);

  reduce2<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(input, output, N);
  cudaDeviceSynchronize();
}

extern "C" void solve3(const float* input, float* output, int N) {
  cudaMemsetAsync(output, 0, sizeof(float));
  int threadsPerBlock = THREADS_PER_BLOCK;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  int sharedMemorySize = threadsPerBlock*sizeof(float);

  reduce3<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(input, output, N);
  cudaDeviceSynchronize();
}