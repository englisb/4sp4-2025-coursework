// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

namespace swiftware::hpp {
enum class GemmStrategy {
  BASELINE,         // Naive implementation
  SHARED_MEMORY,    // Use shared memory tiling
  COALESCED_MEMORY, // Optimize memory access patterns
  COMBINED          // Shared memory + coalesced access
};

// Baseline implementation
inline __global__ void gemm_gpu_baseline(int m, int n, int k, const float *A,
                                         const float *B, float *C) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < m * n) {
    int row = tid / n;
    int col = tid % n;
    float sum = 0.0f;
    for (int p = 0; p < k; ++p) {
      sum += A[row * k + p] * B[p * n + col];
    }
    C[row * n + col] += sum;
    tid += blockDim.x * gridDim.x;
  }
}

// Shared memory tiling implementation - computes C = A * B^T
#define TILE_SIZE 16
inline __global__ void gemm_gpu_shared(int m, int n, int k, const float *A,
                                       const float *B, float *C) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Load tile of A
    if (row < m && t * TILE_SIZE + threadIdx.x < k)
      As[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0f;

    // Load tile of B^T (B is stored as n x k, load transposed)
    int b_col = t * TILE_SIZE + threadIdx.y;
    if (col < n && b_col < k)
      Bs[threadIdx.y][threadIdx.x] = B[col * k + b_col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    // Compute partial dot product
    for (int i = 0; i < TILE_SIZE; ++i)
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

    __syncthreads();
  }

  if (row < m && col < n)
    C[row * n + col] += sum;
}

// Coalesced memory access implementation
inline __global__ void gemm_gpu_coalesced(int m, int n, int k, const float *A,
                                          const float *B, float *C) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;
    for (int p = 0; p < k; ++p) {
      sum += A[row * k + p] * B[p * n + col];
    }
    C[row * n + col] += sum;
  }
}

// Combined: shared memory + coalesced access - computes C = A * B^T
inline __global__ void gemm_gpu_combined(int m, int n, int k, const float *A,
                                         const float *B, float *C) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    // Coalesced loads into shared memory
    int a_col = t * TILE_SIZE + threadIdx.x;
    int b_col = t * TILE_SIZE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? A[row * k + a_col] : 0.0f;
    // Load B^T (B is stored as n x k)
    Bs[threadIdx.y][threadIdx.x] =
        (col < n && b_col < k) ? B[col * k + b_col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i)
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

    __syncthreads();
  }

  if (row < m && col < n)
    C[row * n + col] += sum;
}


// GEMV kernel with row-per-thread approach
// result = A * b, where A is (m x n), b is (n x 1), result is (m x 1)
inline __global__ void MV(const float *a, const float *b, float *result, int m,
                          int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    float sum = 0.0f;
#pragma unroll 4
    for (int col = 0; col < n; col++) {
      sum += a[row * n + col] * b[col];
    }
    result[row] += sum;
  }
}

// ============================================================================
// SpMM Implementations (Sparse Matrix-Matrix Multiplication)
// result = A * B^T, where A is sparse (CSR format) (m x k), B is dense (n x k),
// result is (m x n)
// ============================================================================

// SpMM BASELINE: Simple row-per-thread approach
inline __global__ void SpMM_BASELINE(int *row_ptr, int *col_id, const float *a,
                                     const float *b, float *result, int m,
                                     int n, int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int col = 0; col < n; col++) {
      float sum = 0.0f;
      for (int j = row_start; j < row_end; j++) {
        int a_col = col_id[j];
        sum += a[j] * b[col * k + a_col];
      }
      result[row * n + col] += sum;
    }
  }
}

// SpMM COALESCED: Safe coalesced access - ONE thread per row to avoid divergence
// Each thread processes its entire row with coalesced B accesses
inline __global__ void SpMM_COALESCED(int *row_ptr, int *col_id,
                                      const float *a, const float *b,
                                      float *result, int m, int n, int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    // Process output columns sequentially (thread-safe)
    for (int col = 0; col < n; col++) {
      float sum = 0.0f;
      // Coalesced reads from b: consecutive threads read consecutive columns
#pragma unroll 4
      for (int j = row_start; j < row_end; j++) {
        int a_col = col_id[j];
        sum += a[j] * b[col * k + a_col];
      }
      result[row * n + col] += sum;
    }
  }
}

// SpMM SHARED_MEMORY: Safe shared memory approach - prefetch sparse values
// Load sparse row into shared memory for better cache locality
inline __global__ void SpMM_SHARED_MEMORY(int *row_ptr, int *col_id,
                                          const float *a, const float *b,
                                          float *result, int m, int n, int k,
                                          int tile_size) {
  __shared__ float a_shared[TILE_SIZE * TILE_SIZE];
  __shared__ int col_shared[TILE_SIZE * TILE_SIZE];

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x % 32;

  if (row < m) {
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];
    int nnz = row_end - row_start;
    const int cached_nnz = nnz < (TILE_SIZE * TILE_SIZE)
                               ? nnz
                               : (TILE_SIZE * TILE_SIZE);

    // Prefetch sparse row into shared memory
    for (int idx = lane; idx < cached_nnz; idx += 32) {
      a_shared[idx] = a[row_start + idx];
      col_shared[idx] = col_id[row_start + idx];
    }
    __syncthreads();

    // Process all output columns with cached sparse row
    for (int col = 0; col < n; col++) {
      float sum = 0.0f;
#pragma unroll 4
      for (int j = 0; j < cached_nnz; j++) {
        int a_col = col_shared[j];
        sum += a_shared[j] * b[col * k + a_col];
      }
      result[row * n + col] += sum;
    }
  }
}

// SpMM COMBINED: Safe combined - one thread per row + aggressive optimization
// Coalesced access pattern without warp divergence
inline __global__ void SpMM_COMBINED(int *row_ptr, int *col_id, const float *a,
                                     const float *b, float *result, int m,
                                     int n, int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    // Process output columns - one thread per row
    for (int col = 0; col < n; col++) {
      float sum = 0.0f;

      // Aggressive inner loop unrolling for maximum ILP
#pragma unroll 8
      for (int j = row_start; j < row_end; j++) {
        int a_col = col_id[j];
        sum += a[j] * b[col * k + a_col];
      }

      result[row * n + col] += sum;
    }
  }
}


// ============================================================================
// SpMV Implementations (Sparse Matrix-Vector Multiplication)
// result = A * b, where A is sparse (CSR format) (m x n), b is (n x 1),
// result is (m x 1)
// ============================================================================

// SpMV BASELINE: Simple row-per-thread approach
inline __global__ void SpMV_BASELINE(int *row_ptr, int *col_id, const float *a,
                                     const float *b, float *result, int m,
                                     int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    float sum = 0.0f;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
      int col = col_id[j];
      sum += a[j] * b[col];
    }
    result[row] += sum;
  }
}

// SpMV COALESCED: Coalesced memory access for sparse matrix
inline __global__ void SpMV_COALESCED(int *row_ptr, int *col_id,
                                      const float *a, const float *b,
                                      float *result, int m, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    float sum = 0.0f;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

// Unroll loop to improve instruction-level parallelism
#pragma unroll 4
    for (int j = row_start; j < row_end; j++) {
      int col = col_id[j];
      // Coalesced read from b (consecutive threads read consecutive elements)
      sum += a[j] * b[col];
    }
    result[row] += sum;
  }
}

// SpMV WARP_LEVEL: Safe warp-level - one thread per row, flexible load balance
// Maintains efficiency across variable sparsity patterns
inline __global__ void SpMV_WARP_LEVEL(int *row_ptr, int *col_id,
                                       const float *a, const float *b,
                                       float *result, int m, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    float sum = 0.0f;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    // One thread per row - avoids underutilization from sparse variation
#pragma unroll 8
    for (int j = row_start; j < row_end; j++) {
      int col = col_id[j];
      sum += a[j] * b[col];
    }

    result[row] += sum;
  }
}

// SpMV COMBINED: Safe combined - aggressive optimization without divergence
// One thread per row maintains efficiency across variable sparsity
inline __global__ void SpMV_COMBINED(int *row_ptr, int *col_id, const float *a,
                                     const float *b, float *result, int m,
                                     int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    float sum = 0.0f;
    int row_start = row_ptr[row];
    int row_end = row_ptr[row + 1];

    // Aggressive unrolling for instruction-level parallelism
    // Safe because one thread per row - no divergence
#pragma unroll 8
    for (int j = row_start; j < row_end; j++) {
      int col = col_id[j];
      sum += a[j] * b[col];
    }

    result[row] += sum;
  }
}


// Activation function kernels
inline __global__ void apply_tanh_kernel(float *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = tanhf(data[idx]);
  }
}

inline __global__ void apply_sigmoid_kernel(float *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = 1.0f / (1.0f + expf(-data[idx]));
  }
}

// Bias addition kernel
inline __global__ void add_bias_kernel(float *matrix, const float *bias,
                                       int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    int col = idx % cols;
    matrix[idx] += bias[col];
  }
}

// Argmax kernel
inline __global__ void argmax_rows_kernel(const float *matrix, float *indices,
                                          int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    float max_val = matrix[row * cols];
    int max_idx = 0;
    for (int col = 1; col < cols; col++) {
      float val = matrix[row * cols + col];
      if (val > max_val) {
        max_val = val;
        max_idx = col;
      }
    }
    indices[row] = static_cast<float>(max_idx);
  }
}

} // namespace swiftware::hpp
#endif // PROJECT_KERNELS_CUH
