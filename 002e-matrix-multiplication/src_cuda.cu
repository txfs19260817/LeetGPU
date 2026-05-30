#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))         // row-major order
#define FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0]) // Load 4 floats at once
#define CEIL(a, b) ((a + b - 1) / b)
#define TILE_SIZE 32 // Basic tile size for a thread block

// Bad baseline (A[M][K] * B[K][N] = C[M][N])
__global__ void matrix_multiplication_kernel_bad(const float *__restrict__ A, const float *__restrict__ B,
                                                 float *__restrict__ C, int M, int N, int K)
{
    // Coordinate in C that this thread is responsible for
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    // Note: C is MxN, so the column bound is N, the row bound is M
    if (row < M && col < N)
    {
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < K; ++p)
        {
            // The accumulation dimension is K
            // A: MxK -> ld = K (A[row, p]); B: KxN -> ld = N (B[p, col])
            sum += A[OFFSET(row, p, K)] * B[OFFSET(p, col, N)];
        }
        // C: MxN -> ld = N (C[row, col])
        C[OFFSET(row, col, N)] = sum;
    }
}

// Baseline (A[M][K] * B[K][N] = C[M][N]) (Global Memory Coalescing)
__global__ void matrix_multiplication_kernel(const float *__restrict__ A, const float *__restrict__ B,
                                             float *__restrict__ C, int M, int N, int K)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y; // 0..M-1
    int col = blockDim.x * blockIdx.x + threadIdx.x; // 0..N-1

    // Note: C is MxN, so the column bound is N, the row bound is M
    if (row < M && col < N)
    {
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < K; ++p)
        { // The accumulation dimension is K
            // A: MxK -> ld = K (A[row, p]); B: KxN -> ld = N (B[p, col])
            sum += A[OFFSET(row, p, K)] * B[OFFSET(p, col, N)];
        }
        // C: MxN -> ld = N (C[row, col])
        C[OFFSET(row, col, N)] = sum;
    }
}

// 1D-version of baseline (Global Memory Coalescing)
__global__ void matrix_multiplication_kernel_1d(const float *__restrict__ A, const float *__restrict__ B,
                                                float *__restrict__ C, int M, int N, int K)
{
    int row = blockIdx.y * TILE_SIZE + (threadIdx.x / TILE_SIZE);
    int col = blockIdx.x * TILE_SIZE + (threadIdx.x % TILE_SIZE);

    // Note: C is MxN, so the column bound is N, the row bound is M
    if (row < M && col < N)
    {
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < K; ++p)
        { // The accumulation dimension is K
            // A: MxK -> ld = K (A[row, p]); B: KxN -> ld = N (B[p, col])
            sum += A[OFFSET(row, p, K)] * B[OFFSET(p, col, N)];
        }
        // C: MxN -> ld = N (C[row, col])
        C[OFFSET(row, col, N)] = sum;
    }
}

__global__ void matrix_multiplication_kernel_smem(const float *__restrict__ A, const float *__restrict__ B,
                                                  float *__restrict__ C, int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Sliced: Iterate through the K dimension of the current tile
    for (int t = 0; t < CEIL(K, TILE_SIZE); t++)
    {
        // Load a tile of matrix A (M*T) into shared memory
        As[threadIdx.y][threadIdx.x] =
            (row < M && t * TILE_SIZE + threadIdx.x < K) ? A[row * K + t * TILE_SIZE + threadIdx.x] : 0.0f;
        // Explanation:
        // - row = blockIdx.y * _TILE_SIZE + threadIdx.y (global row index in A)
        // - t * _TILE_SIZE + threadIdx.x = global column index in A
        //
        // Boundary checks:
        //   • row < M: Ensure we don't exceed A's row count
        //   • t * _TILE_SIZE + threadIdx.x < K: Ensure we don't exceed A's column
        //   count
        //     (K is the shared dimension; t is the tile number along this
        //     dimension)
        //
        // Index calculation for A[row * K + t * _TILE_SIZE + threadIdx.x]:
        //   • row * K: Row-major offset to reach the correct row
        //   • t * _TILE_SIZE: Offset to the current tile's starting column
        //   • threadIdx.x: Offset within the tile (0-15)

        // Load a tile of matrix B (T*N) into shared memory
        Bs[threadIdx.y][threadIdx.x] =
            (col < N && t * TILE_SIZE + threadIdx.y < K) ? B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;
        // Explanation:
        // - t * _TILE_SIZE + threadIdx.y = global row index in B
        // - col = blockIdx.x * _TILE_SIZE + threadIdx.x (global column index in B)
        //
        // Boundary checks:
        //   • col < N: Ensure we don't exceed B's column count
        //   • t * _TILE_SIZE + threadIdx.y < K: Ensure we don't exceed B's row
        //   count
        //     (K is the shared dimension; note threadIdx.y for rows this time)
        //
        // Index calculation for B[(t * _TILE_SIZE + threadIdx.y) * N + col]:
        //   • (t * _TILE_SIZE + threadIdx.y) * N: Row-major offset to reach the
        //   correct row
        //     - t * _TILE_SIZE: Starting row of current tile
        //     - threadIdx.y: Offset within the tile (0-15)
        //   • col: Column index within that row

        __syncthreads(); // Wait for all threads to finish loading

        // Calculate the partial product of this tile
        for (int i = 0; i < TILE_SIZE; i++)
        {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads(); // Wait for all threads to finish using this tile
    }

    // Write the result back to global memory
    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

// Shared Memory 1D Threading Matrix Multiplication
// A[M×K] × B[K×N] = C[M×N]
// Uses 1D thread indexing for better global memory coalescing
__global__ void matrix_multiplication_kernel_smem_1d(const float *__restrict__ A, const float *__restrict__ B,
                                                     float *__restrict__ C, int M, int N, int K)
{
    const uint cRow = blockIdx.y; // Block row index in output matrix C
    const uint cCol = blockIdx.x; // Block column index in output matrix C

    // Convert 1D thread index to 2D tile coordinates
    const uint threadCol = threadIdx.x % TILE_SIZE; // Column within tile [0, TILE_SIZE-1]
    const uint threadRow = threadIdx.x / TILE_SIZE; // Row within tile [0, TILE_SIZE-1]

    __shared__ float As[TILE_SIZE * TILE_SIZE]; // Tile cache for A
    __shared__ float Bs[TILE_SIZE * TILE_SIZE]; // Tile cache for B

    // Advance to starting position for this block
    A += cRow * TILE_SIZE * K;                    // A: Start at row cRow*TILE_SIZE, col 0
    B += cCol * TILE_SIZE;                        // B: Start at row 0, col cCol*TILE_SIZE
    C += cRow * TILE_SIZE * N + cCol * TILE_SIZE; // C: Start at row cRow*TILE_SIZE, col cCol*TILE_SIZE

    // Iterate over K dimension with stride of TILE_SIZE
    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += TILE_SIZE)
    {
        // Load Tile from A into Shared Memory with Boundary Checks
        int globalRow = cRow * TILE_SIZE + threadRow; // Global row in A
        int globalColA = bkIdx + threadCol;           // Global col in A (advances each iteration)
        As[threadRow * TILE_SIZE + threadCol] = (globalRow < M && globalColA < K) ? A[threadRow * K + threadCol] : 0.0f;

        // Load Tile from B into Shared Memory with Boundary Checks
        int globalRowB = bkIdx + threadRow;           // Global row in B (advances each iteration)
        int globalCol = cCol * TILE_SIZE + threadCol; // Global col in B
        Bs[threadRow * TILE_SIZE + threadCol] = (globalRowB < K && globalCol < N) ? B[threadRow * N + threadCol] : 0.0f;

        // Synchronize: Wait for all threads to finish loading tiles
        __syncthreads();

        // Advance pointers to next tile in K dimension
        A += TILE_SIZE;
        B += TILE_SIZE * N;

        // Compute Partial Dot Product for Current Tiles
        for (int dotIdx = 0; dotIdx < TILE_SIZE; ++dotIdx)
        {
            tmp += As[threadRow * TILE_SIZE + dotIdx] * Bs[dotIdx * TILE_SIZE + threadCol];
        }

        // Synchronize: Ensure all threads finish computation before loading next tiles
        __syncthreads();
    }

    // Write Result to Global Memory with Boundary Check
    int globalRow = cRow * TILE_SIZE + threadRow;
    int globalCol = cCol * TILE_SIZE + threadCol;
    if (globalRow < M && globalCol < N)
    {
        C[threadRow * N + threadCol] = tmp;
    }
}

__global__ void matrix_multiplication_kernel_sliced_vec4(const float *__restrict__ A, const float *__restrict__ B,
                                                         float *__restrict__ C, int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int row = blockIdx.y * TILE_SIZE + ty; // 负责的 C 的行
    const int col = blockIdx.x * TILE_SIZE + tx; // 负责的 C 的列

    float sum = 0.f;

    // 能否做对齐良好的 float4 访问（行主序下需要 ld 是4的倍数）
    const bool can_vec4_A = (K & 3) == 0; // A 的行跨度 K 是否是 4 的倍数
    const bool can_vec4_B = (N & 3) == 0; // B 的行跨度 N 是否是 4 的倍数

    // 沿着"内积维 K"分块循环
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // ---------------------
        // gmem -> smem: A 的 tile
        // 每4列为一组；仅 x%4==0 的线程做一次 float4
        // 加载，其他线程不加载（但都会参与计算）
        // ---------------------
        const int a_col_base = t * TILE_SIZE + (tx & ~3); // 向下取整到4的倍数（组内起点）
        if ((tx & 3) == 0)
        {
            if (row < M && can_vec4_A && a_col_base + 3 < K)
            {
                // 对齐良好，直接 vector load
                const float4 v = *reinterpret_cast<const float4 *>(&A[row * K + a_col_base]); // K%4==0保证16B对齐
                // smem 按标量写，避免 smem 对齐/向量写的额外注意事项
                As[ty][tx + 0] = v.x;
                As[ty][tx + 1] = v.y;
                As[ty][tx + 2] = v.z;
                As[ty][tx + 3] = v.w;
            }
            else
            {
// 尾块/不对齐：逐元素兜底
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    const int cc = a_col_base + u;
                    As[ty][tx + u] = (row < M && cc < K) ? A[row * K + cc] : 0.f;
                }
            }
        }

        // ---------------------
        // gmem -> smem: B 的 tile
        // B 的"行"是内积维，按行连续；同样按4列打包 float4
        // ---------------------
        const int b_row = t * TILE_SIZE + ty;                      // B 的行（=内积维）
        const int b_col_base = blockIdx.x * TILE_SIZE + (tx & ~3); // 列起点，4的倍数
        if ((tx & 3) == 0)
        {
            if (b_row < K && can_vec4_B && b_col_base + 3 < N)
            {
                const float4 v = *reinterpret_cast<const float4 *>(&B[b_row * N + b_col_base]); // N%4==0保证16B对齐
                Bs[ty][tx + 0] = v.x;
                Bs[ty][tx + 1] = v.y;
                Bs[ty][tx + 2] = v.z;
                Bs[ty][tx + 3] = v.w;
            }
            else
            {
#pragma unroll
                for (int u = 0; u < 4; ++u)
                {
                    const int cc = b_col_base + u;
                    Bs[ty][tx + u] = (b_row < K && cc < N) ? B[b_row * N + cc] : 0.f;
                }
            }
        }

        __syncthreads();

// ---------------------
// 计算阶段：和原 sliced 完全一样
// ---------------------
#pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
        {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve0(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // y dimension → columns → needs to cover N elements
    // x dimension → rows → needs to cover M elements
    dim3 blocksPerGrid(CEIL(M, threadsPerBlock.x), CEIL(N, threadsPerBlock.y));

    matrix_multiplication_kernel_bad<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // x dimension → columns → needs to cover N elements
    // y dimension → rows → needs to cover M elements
    dim3 blocksPerGrid(CEIL(N, threadsPerBlock.x), CEIL(M, threadsPerBlock.y));

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

extern "C" void solve2(const float *A, const float *B, float *C, int M, int N, int K)
{
    // Make blockDim 1-dimensional, but don't change number of threads
    dim3 threadsPerBlock(TILE_SIZE * TILE_SIZE);
    dim3 blocksPerGrid(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE));

    matrix_multiplication_kernel_1d<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

extern "C" void solve3(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(CEIL(N, threadsPerBlock.x), CEIL(M, threadsPerBlock.y));

    matrix_multiplication_kernel_smem<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

extern "C" void solve4(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(TILE_SIZE * TILE_SIZE);
    dim3 blocksPerGrid(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE));

    matrix_multiplication_kernel_smem_1d<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

extern "C" void solve5(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(CEIL(N, threadsPerBlock.x), CEIL(M, threadsPerBlock.y));

    matrix_multiplication_kernel_sliced_vec4<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
