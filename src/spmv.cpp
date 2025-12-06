// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmv.h"
#include <omp.h>

namespace swiftware::hpp {

// Naive SPMV: serial, no optimizations
static void spmvCSR_naive(int m, int n, const int *Ap, const int *Ai, const float *Ax,
                          const float *b, float *c) {
  for (int i = 0; i < m; i++) {
    float sum = 0.0f;
    int row_start = Ap[i];
    int row_end = Ap[i + 1];
    for (int idx = row_start; idx < row_end; idx++) {
      int col = Ai[idx];
      sum += Ax[idx] * b[col];
    }
    c[i] += sum;
  }
}

// SPMV with parallelization only
static void spmvCSR_parallel(int m, int n, const int *Ap, const int *Ai, const float *Ax,
                             const float *b, float *c) {
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    float sum = 0.0f;
    int row_start = Ap[i];
    int row_end = Ap[i + 1];
    for (int idx = row_start; idx < row_end; idx++) {
      int col = Ai[idx];
      sum += Ax[idx] * b[col];
    }
    c[i] += sum;
  }
}

// SPMV with tiling for cache locality
static void spmvCSR_tiled(int m, int n, const int *Ap, const int *Ai, const float *Ax,
                          const float *b, float *c, int tile_size) {
  if (tile_size <= 0) tile_size = m;
  
#pragma omp parallel for
  for (int ii = 0; ii < m; ii += tile_size) {
    int i_end = (ii + tile_size < m) ? ii + tile_size : m;
    for (int i = ii; i < i_end; i++) {
      float sum = 0.0f;
      int row_start = Ap[i];
      int row_end = Ap[i + 1];
      for (int idx = row_start; idx < row_end; idx++) {
        int col = Ai[idx];
        sum += Ax[idx] * b[col];
      }
      c[i] += sum;
    }
  }
}

void spmvCSR(int m, int n, const int *Ap, const int *Ai, const float *Ax,
             const float *b, float *c, ScheduleParams Sp) {
// SPMV: c = A * b + c
// A is sparse m x n in CSR format
// b is dense vector of size n
// c is dense vector of size m

// Use tile size to determine optimization strategy
// Tile1 < 0: naive
// Tile1 == 0 or 1: parallel only
// Tile1 >= 2: tiled + parallel
if (Sp.TileSize1 < 0) {
    spmvCSR_naive(m, n, Ap, Ai, Ax, b, c);
  } else if (Sp.TileSize1 <= 1) {
    spmvCSR_parallel(m, n, Ap, Ai, Ax, b, c);
  } else {
    spmvCSR_tiled(m, n, Ap, Ai, Ax, b, c, Sp.TileSize1);
  }
}

} // namespace swiftware::hpp