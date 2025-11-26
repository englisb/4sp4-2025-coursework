// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB03_TRIANGULAR_SOLVER_H
#define LAB03_TRIANGULAR_SOLVER_H
#include "def.h"
#include <omp.h>

namespace swiftware::hpp
{
  template<typename T>
  void sptrsv_csr(T *val, int *col_ind, int *row_ptr, T *x, T *b, int n, ScheduleParams *SP) {
    // Sequential sparse lower-triangular solve (forward substitution)
    // A is assumed lower-triangular with explicit diagonal stored in CSR.
    // Solves A x = b.
    (void)SP; // unused in sequential implementation

    for (int i = 0; i < n; ++i) {
      T sum = b[i];
      T diag = static_cast<T>(0);

      const int row_start = row_ptr[i];
      const int row_end   = row_ptr[i + 1];

      for (int idx = row_start; idx < row_end; ++idx) {
        const int j = col_ind[idx];
        const T aij = val[idx];

        if (j < i) {
          sum -= aij * x[j];
        } else if (j == i) {
          diag = aij;
        }
        // entries with j > i are zero in a proper lower-triangular matrix
      }

      // Non-zero diagonal assumed
      x[i] = sum / diag;
    }
  }


  template<typename T>
  void sptrsv_csr_parallel(T *val, int *col_ind, int *row_ptr, T *x, T *b, int n, ScheduleParams *SP) {
    // Level-scheduled parallel forward substitution for lower-triangular CSR
    
    if (n <= 0) return;

    // Compute level for each row (maximum level of dependencies + 1)
    std::vector<int> level(n, 0);
    for (int i = 0; i < n; ++i) {
      int max_dep_level = -1;
      for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
        int j = col_ind[idx];
        if (j < i) {
          max_dep_level = std::max(max_dep_level, level[j]);
        }
      }
      level[i] = max_dep_level + 1;
    }

    // Find number of levels
    int num_levels = 0;
    for (int i = 0; i < n; ++i) {
      num_levels = std::max(num_levels, level[i] + 1);
    }

    // Group rows by level
    std::vector<std::vector<int>> level_sets(num_levels);
    for (int i = 0; i < n; ++i) {
      level_sets[level[i]].push_back(i);
    }

    // Calculate average level size for scheduling decisions
    double avg_level_size = static_cast<double>(n) / num_levels;
    
    int num_threads = 1;
    if (SP && SP->NumThreads > 0) {
      num_threads = SP->NumThreads;
      omp_set_num_threads(num_threads);
    }

    // Clear and prepare timing data if collection is enabled
    if (SP && SP->collect_level_times) {
      SP->level_times.clear();
      SP->level_times.reserve(num_levels);
    }
    
    // Process each level in order with adaptive scheduling
    for (int lvl = 0; lvl < num_levels; ++lvl) {
      const std::vector<int>& current = level_sets[lvl];
      int level_size = static_cast<int>(current.size());
      
      double level_start_time = 0.0;
      if (SP && SP->collect_level_times) {
        level_start_time = omp_get_wtime();
      }
      
      // Choose scheduling strategy based on level size and matrix characteristics
      // For larger average level sizes, use static scheduling
      // For medium levels, use dynamic scheduling
      // For smaller levels, use guided scheduling
      
      if (avg_level_size > 1000) {
        // Large well-balanced levels: use static scheduling with moderate chunk
        int chunk_size = std::max(1, level_size / (num_threads * 4));
        #pragma omp parallel for schedule(static, chunk_size)
        for (std::size_t t = 0; t < current.size(); ++t) {
          int i = current[t];
          T sum = b[i];
          T diag = static_cast<T>(0);
          for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            const int j = col_ind[idx];
            const T aij = val[idx];
            if (j < i) sum -= aij * x[j];
            else if (j == i) diag = aij;
          }
          x[i] = sum / diag;
        }
        if (SP && SP->collect_level_times) {
          SP->level_times.push_back(omp_get_wtime() - level_start_time);
        }
      } else if (level_size > 64) {
        // Medium-sized levels with potential imbalance: use dynamic scheduling
        int chunk_size = std::max(1, level_size / (num_threads * 8));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (std::size_t t = 0; t < current.size(); ++t) {
          int i = current[t];
          T sum = b[i];
          T diag = static_cast<T>(0);
          for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            const int j = col_ind[idx];
            const T aij = val[idx];
            if (j < i) sum -= aij * x[j];
            else if (j == i) diag = aij;
          }
          x[i] = sum / diag;
        }
        if (SP && SP->collect_level_times) {
          SP->level_times.push_back(omp_get_wtime() - level_start_time);
        }
      } else {
        // Smaller levels: use guided scheduling for adaptive load balancing
        #pragma omp parallel for schedule(guided)
        for (std::size_t t = 0; t < current.size(); ++t) {
          int i = current[t];
          T sum = b[i];
          T diag = static_cast<T>(0);
          for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
            const int j = col_ind[idx];
            const T aij = val[idx];
            if (j < i) sum -= aij * x[j];
            else if (j == i) diag = aij;
          }
          x[i] = sum / diag;
        }
        if (SP && SP->collect_level_times) {
          SP->level_times.push_back(omp_get_wtime() - level_start_time);
        }
      }
    }
  }
}

#endif //LAB03_TRIANGULAR_SOLVER_H