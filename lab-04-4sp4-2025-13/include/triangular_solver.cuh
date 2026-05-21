// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB04_TRIANGULAR_SOLVER_CUH
#define LAB04_TRIANGULAR_SOLVER_CUH

#include "def.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cusparse.h>
#else
/* allow parsing with non-CUDA compilers / IDEs */
#define __global__
#endif
namespace swiftware::hpp
{

  template<typename T>
  __global__ void sparse_csr_parallel_gpu(T *val, int *col_ind, int *row_ptr, T *x, T *b, int n, int *ready)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;
    
    // Each thread handles one row
    int row = tid;
    
    // Check if this row is already computed
    if (ready[row] == 1) return;
    
    // Check if all dependencies are ready
    bool can_compute = true;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      int col = col_ind[j];
      if (col < row && ready[col] == 0) {
        can_compute = false;
        break;
      }
    }
    
    if (!can_compute) return;
    
    // Compute the solution for this row
    T sum = b[row];
    T diag = 1.0;
    
    // Process all non-zero elements in this row
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      int col = col_ind[j];
      if (col < row) {
        // Off-diagonal element: subtract contribution from already computed x[col]
        sum -= val[j] * x[col];
      } else if (col == row) {
        // Diagonal element
        diag = val[j];
      }
    }
    
    // Compute x[row] = sum / diagonal
    if (diag != 0.0) {
      x[row] = sum / diag;
    } else {
      x[row] = sum; // Handle zero diagonal
    }
    
    // Mark this row as ready
    ready[row] = 1;
  }

  // Level-set preprocessing (CPU) - separate scheduling time from solve time
  inline void compute_level_sets(int *col_ind, int *row_ptr, int n, 
                                  std::vector<int>& level_ptr, std::vector<int>& level_set) {
    std::vector<int> level(n, -1);  // Level for each row
    level_ptr.clear();
    level_set.clear();
    level_ptr.push_back(0);
    
    int max_level = 0;
    // Compute level for each row
    for (int row = 0; row < n; ++row) {
      int row_level = 0;
      // Find max level of dependencies
      for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
        int col = col_ind[j];
        if (col < row && level[col] >= 0) {
          row_level = std::max(row_level, level[col] + 1);
        }
      }
      level[row] = row_level;
      max_level = std::max(max_level, row_level);
    }
    
    // Build level sets
    std::vector<std::vector<int>> levels(max_level + 1);
    for (int row = 0; row < n; ++row) {
      levels[level[row]].push_back(row);
    }
    
    for (int l = 0; l <= max_level; ++l) {
      for (int row : levels[l]) {
        level_set.push_back(row);
      }
      level_ptr.push_back(level_set.size());
    }
  }

  // Level-set based implementation for better parallelism
  template<typename T>
  __global__ void sparse_csr_parallel_gpu_leveled(T *val, int *col_ind, int *row_ptr, T *x, T *b, 
                                                   int n, int *level_ptr, int *level_set, int current_level)
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Get the range of rows in this level
    int level_start = level_ptr[current_level];
    int level_end = level_ptr[current_level + 1];
    int level_size = level_end - level_start;
    
    if (tid >= level_size) return;
    
    // Get the actual row number from level_set
    int row = level_set[level_start + tid];
    
    if (row >= n) return;
    
    // Compute the solution for this row
    T sum = b[row];
    T diag = 1.0;
    
    // Process all non-zero elements in this row
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
      int col = col_ind[j];
      if (col < row) {
        // Off-diagonal element
        sum -= val[j] * x[col];
      } else if (col == row) {
        // Diagonal element
        diag = val[j];
      }
    }
    
    // Compute x[row] = sum / diagonal
    if (diag != 0.0) {
      x[row] = sum / diag;
    } else {
      x[row] = sum;
    }
  }

  // GPU Sequential Implementation (single-threaded on GPU)
  template<typename T>
  __global__ void sparse_csr_sequential_gpu(T *val, int *col_ind, int *row_ptr, T *x, T *b, int n)
  {
    // Only thread 0 does all the work
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      // Sequential forward substitution for lower triangular matrix
      for (int row = 0; row < n; ++row) {
        T sum = b[row];
        T diag = 1.0;
        
        // Process all non-zero elements in this row
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
          int col = col_ind[j];
          if (col < row) {
            // Off-diagonal element: subtract contribution
            sum -= val[j] * x[col];
          } else if (col == row) {
            // Diagonal element
            diag = val[j];
          }
        }
        
        // Compute x[row] = sum / diagonal
        if (diag != 0.0) {
          x[row] = sum / diag;
        } else {
          x[row] = sum;
        }
      }
    }
  }

}

#endif //LAB04_TRIANGULAR_SOLVER_CUH
