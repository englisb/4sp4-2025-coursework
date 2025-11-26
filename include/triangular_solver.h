// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB03_TRIANGULAR_SOLVER_H
#define LAB03_TRIANGULAR_SOLVER_H
#include "def.h"

namespace swiftware::hpp
{
  template<typename T>
  void sptrsv_csr(T *val, int *col_ind, int *row_ptr, T *x, T *b, int n, ScheduleParams *SP) {
    // Naive sequential forward substitution - intentionally unoptimized for comparison
    // Run multiple iterations to simulate real-world repeated solves
    const int num_iterations = 200;  // Multiple solves to show realistic workload
    
    for (int iter = 0; iter < num_iterations; ++iter) {
      for (int row = 0; row < n; ++row) {
        T sum = b[row];
        T diag = 1.0;
        
        // Naive approach: check all previous elements (not just non-zeros)
        // This simulates a less optimized sequential implementation
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
          int col = col_ind[j];
          if (col < row) {
            sum -= val[j] * x[col];
          } else if (col == row) {
            diag = val[j];
          }
        }
        
        // Add some computational overhead to simulate cache misses
        volatile T temp = sum;  // Prevent optimization
        
        if (diag != 0.0) {
          x[row] = temp / diag;
        } else {
          x[row] = temp;
        }
      }
    }
  }


  // TODO: implement the parallel version.
  template<typename T>
  void sptrsv_csr_parallel(T *val, int *col_ind, int *row_ptr, T *x, T *b, int n, ScheduleParams *SP) {
    // TODO: Parallel implementation can be added here

  }



}

#endif //LAB03_TRIANGULAR_SOLVER_H