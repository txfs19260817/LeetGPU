#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col)) // row-major order
#define FLOAT4(ptr)                                                            \
  (reinterpret_cast<float4 *>(&(ptr))[0]) // Load 4 floats at once
#define CEIL(a, b) ((a + b - 1) / b)
#define TILE_SIZE 16 // Basic tile size for a thread block

// Baseline (A[M][N] * B[N][K] = C[M][K])
__global__ void matrix_multiplication_kernel(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C, int M,
                                             int N, int K) {
  int row = blockDim.y * blockIdx.y + threadIdx.y; // 0..M-1
  int col = blockDim.x * blockIdx.x + threadIdx.x; // 0..K-1

  // Note: C is MxK, so the column bound is K, the row bound is M
  if (row < M && col < K) {
    float sum = 0.0f;
#pragma unroll
    for (int p = 0; p < N; ++p) { // The accumulation dimension is N
      // A: MxN -> ld = N (A[row, p]); B: NxK -> ld = K (B[p, col])
      sum += A[OFFSET(row, p, N)] * B[OFFSET(p, col, K)];
    }
    // C: MxK -> ld = K (C[row, col])
    C[OFFSET(row, col, K)] = sum;
  }
}

__global__ void matrix_multiplication_kernel_sliced(const float *__restrict__ A,
                                                    const float *__restrict__ B,
                                                    float *__restrict__ C,
                                                    int M, int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  // Sliced: Iterate through the N dimension of the current tile
  for (int t = 0; t < CEIL(N, TILE_SIZE); t++) {
    // Load a tile of matrix A (M*T) into shared memory
    As[threadIdx.y][threadIdx.x] =
        (row < M && t * TILE_SIZE + threadIdx.x < N)
            ? A[row * N + t * TILE_SIZE + threadIdx.x]
            : 0.0f;
    // Explanation:
    // - row = blockIdx.y * _TILE_SIZE + threadIdx.y (global row index in A)
    // - t * _TILE_SIZE + threadIdx.x = global column index in A
    //
    // Boundary checks:
    //   • row < M: Ensure we don't exceed A's row count
    //   • t * _TILE_SIZE + threadIdx.x < N: Ensure we don't exceed A's column
    //   count
    //     (N is the shared dimension; t is the tile number along this
    //     dimension)
    //
    // Index calculation for A[row * N + t * _TILE_SIZE + threadIdx.x]:
    //   • row * N: Row-major offset to reach the correct row
    //   • t * _TILE_SIZE: Offset to the current tile's starting column
    //   • threadIdx.x: Offset within the tile (0-15)

    // Load a tile of matrix B (T*K) into shared memory
    Bs[threadIdx.y][threadIdx.x] =
        (col < K && t * TILE_SIZE + threadIdx.y < N)
            ? B[(t * TILE_SIZE + threadIdx.y) * K + col]
            : 0.0f;
    // Explanation:
    // - t * _TILE_SIZE + threadIdx.y = global row index in B
    // - col = blockIdx.x * _TILE_SIZE + threadIdx.x (global column index in B)
    //
    // Boundary checks:
    //   • col < K: Ensure we don't exceed B's column count
    //   • t * _TILE_SIZE + threadIdx.y < N: Ensure we don't exceed B's row
    //   count
    //     (N is the shared dimension; note threadIdx.y for rows this time)
    //
    // Index calculation for B[(t * _TILE_SIZE + threadIdx.y) * K + col]:
    //   • (t * _TILE_SIZE + threadIdx.y) * K: Row-major offset to reach the
    //   correct row
    //     - t * _TILE_SIZE: Starting row of current tile
    //     - threadIdx.y: Offset within the tile (0-15)
    //   • col: Column index within that row

    __syncthreads(); // Wait for all threads to finish loading

    // Calculate the partial product of this tile
    for (int i = 0; i < TILE_SIZE; i++) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads(); // Wait for all threads to finish using this tile
  }

  // Write the result back to global memory
  if (row < M && col < K) {
    C[row * K + col] = sum;
  }
}

__global__ void matrix_multiplication_kernel_sliced_vec4(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  const int ty = threadIdx.y;
  const int tx = threadIdx.x;

  const int row = blockIdx.y * TILE_SIZE + ty; // 负责的 C 的行
  const int col = blockIdx.x * TILE_SIZE + tx; // 负责的 C 的列

  float sum = 0.f;

  // 能否做对齐良好的 float4 访问（行主序下需要 ld 是4的倍数）
  const bool can_vec4_A = (N & 3) == 0; // A 的行跨度 N 是否是 4 的倍数
  const bool can_vec4_B = (K & 3) == 0; // B 的行跨度 K 是否是 4 的倍数

  // 沿着“内积维 N”分块循环
  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // ---------------------
    // gmem -> smem: A 的 tile
    // 每4列为一组；仅 x%4==0 的线程做一次 float4
    // 加载，其他线程不加载（但都会参与计算）
    // ---------------------
    const int a_col_base =
        t * TILE_SIZE + (tx & ~3); // 向下取整到4的倍数（组内起点）
    if ((tx & 3) == 0 && row < M) {
      if (can_vec4_A && a_col_base + 3 < N) {
        // 对齐良好，直接 vector load
        const float4 v = *reinterpret_cast<const float4 *>(
            &A[row * N + a_col_base]); // N%4==0保证16B对齐
        // smem 按标量写，避免 smem 对齐/向量写的额外注意事项
        As[ty][tx + 0] = v.x;
        As[ty][tx + 1] = v.y;
        As[ty][tx + 2] = v.z;
        As[ty][tx + 3] = v.w;
      } else {
// 尾块/不对齐：逐元素兜底
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int cc = a_col_base + u;
          As[ty][tx + u] = (cc < N) ? A[row * N + cc] : 0.f;
        }
      }
    }

    // ---------------------
    // gmem -> smem: B 的 tile
    // B 的“行”是内积维，按行连续；同样按4列打包 float4
    // ---------------------
    const int b_row = t * TILE_SIZE + ty; // B 的行（=内积维）
    const int b_col_base =
        blockIdx.x * TILE_SIZE + (tx & ~3); // 列起点，4的倍数
    if ((tx & 3) == 0 && b_row < N) {
      if (can_vec4_B && b_col_base + 3 < K) {
        const float4 v = *reinterpret_cast<const float4 *>(
            &B[b_row * K + b_col_base]); // K%4==0保证16B对齐
        Bs[ty][tx + 0] = v.x;
        Bs[ty][tx + 1] = v.y;
        Bs[ty][tx + 2] = v.z;
        Bs[ty][tx + 3] = v.w;
      } else {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          const int cc = b_col_base + u;
          Bs[ty][tx + u] = (cc < K) ? B[b_row * K + cc] : 0.f;
        }
      }
    }

    __syncthreads();

// ---------------------
// 计算阶段：和原 sliced 完全一样
// ---------------------
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += As[ty][i] * Bs[i][tx];
    }

    __syncthreads();
  }

  if (row < M && col < K) {
    C[row * K + col] = sum;
  }
}

__global__ void matrix_multiplication_kernel_sliced_v1(
    const float *__restrict__ A, const float *__restrict__ B,
    float *__restrict__ C, int M, int N, int K) {
  const int BM = 128, BN = 128, BK = 8;
  const int TM = 8, TN = 8;

  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];

  float r_c[TM][TN] = {0};

  int load_a_smem_m = tid >> 1;
  int load_a_smem_k = (tid & 1) << 2; // 0 or 4
  int load_b_smem_k = tid >> 5;
  int load_b_smem_n = (tid & 31) << 2; // 0..124 step 4

  int base_a_m = by * BM + load_a_smem_m; // global row of A
  int base_b_n = bx * BN + load_b_smem_n; // global col of B

  for (int bk = 0; bk < (N + BK - 1) / BK; ++bk) {
    int a_col = bk * BK + load_a_smem_k; // global col of A
    int b_row = bk * BK + load_b_smem_k; // global row of B

    // ===== gmem -> smem : 修正 ld =====
    // A: ld = N
    int a_gmem = OFFSET(base_a_m, a_col, N);
    // B: ld = K
    int b_gmem = OFFSET(b_row, base_b_n, K);

    // （可选：边界掩码，满足对齐且完全在界内才走 float4）
    // For float4 loads, we need 16-byte alignment, which requires N%4==0 for A
    // and K%4==0 for B
    bool a_in = (base_a_m < M) && (a_col + 3 < N) && (N % 4 == 0);
    bool b_in = (b_row < N) && (base_b_n + 3 < K) && (K % 4 == 0);

    if (a_in)
      FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) =
          FLOAT4(const_cast<float &>(A[a_gmem]));
    else {
// 标量填充 0，避免越界
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        int cc = load_a_smem_k + t;
        s_a[load_a_smem_m][cc] = (base_a_m < M && (bk * BK + cc) < N)
                                     ? A[OFFSET(base_a_m, bk * BK + cc, N)]
                                     : 0.f;
      }
    }

    if (b_in)
      FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) =
          FLOAT4(const_cast<float &>(B[b_gmem]));
    else {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        int cc = load_b_smem_n + t;
        s_b[load_b_smem_k][cc] =
            ((bk * BK + load_b_smem_k) < N && (base_b_n + t) < K)
                ? B[OFFSET(bk * BK + load_b_smem_k, base_b_n + t, K)]
                : 0.f;
      }
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          int comp_a_m = ty * TM + m;
          int comp_b_n = tx * TN + n;
          r_c[m][n] += s_a[comp_a_m][k] * s_b[k][comp_b_n];
        }
      }
    }
    __syncthreads();
  }

// ===== smem(accum) -> gmem : 修正 ld = K，并加边界保护 =====
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int out_m = by * BM + ty * TM + i;
    if (out_m >= M)
      break;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      int out_n = bx * BN + tx * TN + j;
      // Need K%4==0 for aligned float4 writes
      if ((out_n + 3 < K) && (K % 4 == 0)) {
        int c_gmem = OFFSET(out_m, out_n, K); // ✅ C 的 ld = K
        FLOAT4(C[c_gmem]) = FLOAT4(r_c[i][j]);
      } else {
// 尾部不满 4 的情况做标量写
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          if (out_n + t < K) {
            C[OFFSET(out_m, out_n + t, K)] = r_c[i][j + t];
          }
        }
      }
    }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid(CEIL(K, threadsPerBlock.x), CEIL(M, threadsPerBlock.y));

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}

extern "C" void solve2(const float *A, const float *B, float *C, int M, int N,
                       int K) {
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid(CEIL(K, threadsPerBlock.x), CEIL(M, threadsPerBlock.y));

  matrix_multiplication_kernel_sliced<<<blocksPerGrid, threadsPerBlock>>>(
      A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

extern "C" void solve3(const float *A, const float *B, float *C, int M, int N,
                       int K) {
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid(CEIL(K, threadsPerBlock.x), CEIL(M, threadsPerBlock.y));

  matrix_multiplication_kernel_sliced_vec4<<<blocksPerGrid, threadsPerBlock>>>(
      A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

extern "C" void solve4(const float *A, const float *B, float *C, int M, int N,
                       int K) {
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid(CEIL(K, threadsPerBlock.x), CEIL(M, threadsPerBlock.y));

  matrix_multiplication_kernel_sliced_v1<<<blocksPerGrid, threadsPerBlock>>>(
      A, B, C, M, N, K);
  cudaDeviceSynchronize();
}