// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmv.h"
#include <omp.h>

namespace swiftware::hpp {
void spmvCSR(int m, int n, const int *Ap, const int *Ai, const float *Ax,
             const float *b, float *c, ScheduleParams Sp) {
// SPMV: c = A * b + c
// A is sparse m x n in CSR format
// b is dense vector of size n
// c is dense vector of size m

// Parallelize across rows - each row can be processed independently
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    float sum = 0.0f;
    int row_start = Ap[i];
    int row_end = Ap[i + 1];

    // Compute dot product of row i with vector b
    // Only iterate over non-zero elements in row i
    for (int idx = row_start; idx < row_end; idx++) {
      int col = Ai[idx];
      sum += Ax[idx] * b[col];
    }

    // Accumulate into c
    c[i] += sum;
  }
}

} // namespace swiftware::hpp