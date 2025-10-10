#include <cuda_runtime.h>

// Block reduction kernel: grid-stride + shared-memory reduce + one atomic per block
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
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  int sharedMemorySize = threadsPerBlock*sizeof(float);

  reduce0<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(input, output, N);
  cudaDeviceSynchronize();
}
