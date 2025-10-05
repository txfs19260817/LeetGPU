#include <cuda_runtime.h>

#define CEIL(a, b) ((a + b - 1) / b)
#define TM 8 // Number of rows in the result matrix processed by each thread
#define TK 8 // Number of columns in the result matrix processed by each thread
#define TILE_SIZE 16        // Basic tile size for a thread block
#define BM (TM * TILE_SIZE) // Block size in the M dimension
#define BK (TK * TILE_SIZE) // Block size in the K dimension


__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < M && col < K) {
    float sum = .0f;
    #pragma unroll
    for (size_t i = 0; i < N; i++) {
      sum += A[row * N + i] * B[i * K + col];
    }
    C[row * K + col] = sum;
  }
}

__global__ void matrix_multiplication_kernel_sliced(const float *A,
                                                    const float *B, float *C,
                                                    int M, int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  // Sliced: Iterate through the N dimension of the current tile
  for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
    // Load a tile of matrix A into shared memory
    if (row < M && t * TILE_SIZE + threadIdx.x < N) {
      As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Load a tile of matrix B into shared memory
    if (col < K && t * TILE_SIZE + threadIdx.y < N) {
      Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads(); // Wait for all threads to finish loading

    // Calculate the partial product of this tile
    for (int i = 0; i < TILE_SIZE; i++) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads(); // Wait for all threads to finish using this tile, then enter the next iteration
  }

  // Write the result back to global memory
  if (row < M && col < K) {
    C[row * K + col] = sum;
  }
}

__global__ void matrix_multiplication_kernel_double_buffer(const float* A, const float* B, float* C, int M, int N, int K) {
  // Use double buffering; declare two tile buffers for As and Bs respectively.
  // As[buffer_id][BM][TILE_SIZE]: Stores a tile of matrix A, size 128x16
  // Bs[buffer_id][TILE_SIZE][BK]: Stores a tile of matrix B, size 16x128
  __shared__ float As[2][BM][TILE_SIZE];
  __shared__ float Bs[2][TILE_SIZE][BK];

  // Get the coordinates of the current thread within the block.
  int tx = threadIdx.x;  // Thread's column index within the block (0-15)
  int ty = threadIdx.y;  // Thread's row index within the block (0-15)

  // Calculate the starting position of the current block in the global matrix.
  int row_base = blockIdx.y * BM;  // Starting row position for this block in matrix A/C
  int col_base = blockIdx.x * BK;  // Starting column position for this block in matrix B/C

  // Each thread maintains a small TMxTK result matrix, initialized to 0.
  // val[8][8]: Each thread is responsible for calculating an 8x8 result block.
  float val[TM][TK] = {{0.0f}};

  // Calculate how many tiles need to be processed (along the N dimension).
  // num_tiles: The number of iterations needed, equal to N/TILE_SIZE rounded up.
  int num_tiles = CEIL(N, TILE_SIZE);

  // Double buffer index management.
  int load_buf_idx = 0;  // Index for the buffer currently used for loading data (0 or 1)
  int calc_buf_idx = 1;  // Index for the buffer currently used for computation (0 or 1)

  // ==================== Initial Data Load ====================
  // Before the first computation, load the first tile into buffer 0.
  
  // Load the first tile of matrix A into shared memory As[0].
  // Each thread is responsible for loading specific elements from TM rows.
  #pragma unroll
  for (int i = 0; i < TM; i++) {
      // Calculate global row index: block start row + i-th row handled by the thread + thread's row offset within the tile.
      int row = row_base + i * TILE_SIZE + ty;
      // Calculate the column index within the current tile.
      int k_idx = tx;
      
      // Boundary check: ensure no out-of-bounds access.
      if (row < M && (0 * TILE_SIZE + k_idx) < N) {
          // Load data from matrix A into the shared memory As buffer indicated by load_buf_idx.
          // As[buffer][row index][column index] = A[global row][global column]
          As[load_buf_idx][i * TILE_SIZE + ty][tx] = A[row * N + 0 * TILE_SIZE + k_idx];
      } else {
          // Fill out-of-bounds positions with 0.
          As[load_buf_idx][i * TILE_SIZE + ty][tx] = 0.0f;
      }
  }

  // Load the first tile of matrix B into shared memory Bs[0].
  // Each thread is responsible for loading specific elements from TK columns.
  #pragma unroll
  for (int i = 0; i < TK; i++) {
      // Calculate global column index: block start column + i-th column handled by the thread + thread's column offset within the tile.
      int col = col_base + i * TILE_SIZE + tx;
      // Calculate the row index within the current tile.
      int k_idx = ty;
      
      // Boundary check.
      if (col < K && (0 * TILE_SIZE + k_idx) < N) {
          // Load data from matrix B into the shared memory Bs buffer indicated by load_buf_idx.
          // Bs[buffer][row index][column index] = B[global row][global column]
          Bs[load_buf_idx][ty][i * TILE_SIZE + tx] = B[(0 * TILE_SIZE + k_idx) * K + col];
      } else {
          // Fill out-of-bounds positions with 0.
          Bs[load_buf_idx][ty][i * TILE_SIZE + tx] = 0.0f;
      }
  }

  // Ensure all threads have finished the initial data load.
  __syncthreads();

  // Switch buffer indices to prepare for the first computation.
  // Initial load was into buffer 0, so the next load will be into buffer 1, and the current computation uses buffer 0.
  load_buf_idx = 1;
  calc_buf_idx = 0;

  // ==================== Main Computation Loop (Double Buffering Core) ====================
  // Loop to process all tiles, each of size TILE_SIZE.
  for (int t = 0; t < num_tiles; t++) {
      
      // 1. Start loading data for the next iteration (if it's not the last one).
      //    This part executes in parallel with the current iteration's computation, overlapping computation and memory access.
      if (t < num_tiles - 1) {
          // Load the next tile of matrix A into the buffer indicated by load_buf_idx.
          #pragma unroll
          for (int i = 0; i < TM; i++) {
              // Calculate the global row index.
              int row = row_base + i * TILE_SIZE + ty;
              // Current column index.
              int k_idx = tx;
              // Calculate the starting position of the next tile in the N dimension.
              int next_n_tile = (t + 1) * TILE_SIZE;
              
              // Check boundaries and load data.
              if (row < M && (next_n_tile + k_idx) < N) {
                  As[load_buf_idx][i * TILE_SIZE + ty][tx] = 
                      A[row * N + next_n_tile + k_idx];
              } else {
                  As[load_buf_idx][i * TILE_SIZE + ty][tx] = 0.0f;
              }
          }

          // Load the next tile of matrix B into the buffer indicated by load_buf_idx.
          #pragma unroll
          for (int i = 0; i < TK; i++) {
              // Calculate the global column index.
              int col = col_base + i * TILE_SIZE + tx;
              // Current row index.
              int k_idx = ty;
              // Calculate the starting position of the next tile in the N dimension.
              int next_n_tile = (t + 1) * TILE_SIZE;
              
              // Check boundaries and load data.
              if (col < K && (next_n_tile + k_idx) < N) {
                  Bs[load_buf_idx][ty][i * TILE_SIZE + tx] = 
                      B[(next_n_tile + k_idx) * K + col];
              } else {
                  Bs[load_buf_idx][ty][i * TILE_SIZE + tx] = 0.0f;
              }
          }
      }

      // 2. Perform the computation for the current iteration using the buffer pointed to by calc_buf_idx.
      //    Calculate matrix multiplication: val[i][j] += sum_k(As[...][k] * Bs[k][...]).
      //    This is the standard matrix multiplication calculation: C[i][j] += A[i][k] * B[k][j].
      for (int k = 0; k < TILE_SIZE; k++) {      // Iterate through the K dimension of the current tile (0-15).
          for (int i = 0; i < TM; i++) {          // Iterate through the rows of the result matrix (0-7).
              for (int j = 0; j < TK; j++) {      // Iterate through the columns of the result matrix (0-7).
                  // Accumulate the result: val[i][j] += A element * B element.
                  // As[current computation buffer][i-th row handled by the thread + offset][k dimension index]
                  // Bs[current computation buffer][k dimension index][j-th column handled by the thread + offset]
                  val[i][j] += 
                      As[calc_buf_idx][i * TILE_SIZE + ty][k] * Bs[calc_buf_idx][k][j * TILE_SIZE + tx];
              }
          }
      }

      // 3. Wait for the current iteration's computation and the next iteration's data load to complete.
      //    Ensure all threads have finished computation and data loading to avoid race conditions.
      __syncthreads();

      // 4. Swap buffer indices: prepare for the next iteration.
      //    The next computation will use the buffer that was just loaded, and the next load will use the buffer that was just used for computation.
      int tmp = load_buf_idx;
      load_buf_idx = calc_buf_idx;
      calc_buf_idx = tmp;
  }

  // ==================== Write Results Back to Global Memory ====================
  // Write the TMxTK result calculated by each thread back to the global memory matrix C.
  for (int i = 0; i < TM; i++) {
      for (int j = 0; j < TK; j++) {
          // Calculate the position of the result in the global matrix C.
          int row = row_base + i * TILE_SIZE + ty;  // Global row index
          int col = col_base + j * TILE_SIZE + tx;  // Global column index
          
          // Check boundaries and write back the result.
          if (row < M && col < K) {
              C[row * K + col] = val[i][j];   // C[row][column] = calculated result
          }
      }
  }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

extern "C" void solve_sliced(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrix_multiplication_kernel_sliced<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}

extern "C" void solve_double_buffer(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matrix_multiplication_kernel_double_buffer<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
  cudaDeviceSynchronize();
}