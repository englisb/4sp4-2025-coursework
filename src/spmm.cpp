// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmm.h"
#include "immintrin.h"
#include <omp.h>

namespace swiftware::hpp {

void spmmCSR(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax,
             const float *B, float *C, ScheduleParams Sp) {
  // SPMM: C = A * B + C
  // A is sparse m x k in CSR format
  // B is dense k x n (row-major)
  // C is dense m x n (row-major)

  const int simd_width = 8; // AVX processes 8 floats at a time
  const int n_simd = n / simd_width;

  // Determine tile sizes for cache blocking
  int tile_m = (Sp.TileSize1 > 0) ? Sp.TileSize1 : m;
  int tile_n = (Sp.TileSize2 > 0) ? Sp.TileSize2 : n;

  // Use tiling if tile sizes are reasonable
  if (tile_m < m || tile_n < n) {
    // Tiled SPMM with cache blocking
#pragma omp parallel for collapse(2)
    for (int ii = 0; ii < m; ii += tile_m) {
      for (int jj = 0; jj < n; jj += tile_n) {
        int i_end = (ii + tile_m < m) ? ii + tile_m : m;
        int j_end = (jj + tile_n < n) ? jj + tile_n : n;

        // Process tile [ii:i_end, jj:j_end]
        for (int i = ii; i < i_end; i++) {
          float *C_row = C + i * n;
          int row_start = Ap[i];
          int row_end = Ap[i + 1];

          // Handle unaligned columns at the start
          int j_start = jj;
          int j_start_aligned =
              ((jj + simd_width - 1) / simd_width) * simd_width;
          if (j_start_aligned > j_end) {
            j_start_aligned = j_end;
          }

          // Scalar code for unaligned start
          for (int j = j_start; j < j_start_aligned; j++) {
            float sum = C_row[j];
            for (int idx = row_start; idx < row_end; idx++) {
              int col = Ai[idx];
              sum += Ax[idx] * B[col * n + j];
            }
            C_row[j] = sum;
          }

          // SIMD-vectorized blocks
          for (int j = j_start_aligned; j + simd_width <= j_end;
               j += simd_width) {
            __m256 c_vec = _mm256_loadu_ps(C_row + j);

            // For each non-zero in row i of A
            for (int idx = row_start; idx < row_end; idx++) {
              int col = Ai[idx];
              float a_val = Ax[idx];
              __m256 a_broadcast = _mm256_set1_ps(a_val);
              __m256 b_vec = _mm256_loadu_ps(B + col * n + j);
              c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
            }

            _mm256_storeu_ps(C_row + j, c_vec);
          }

          // Handle remainder columns at the end
          int j_remainder_start =
              j_start_aligned +
              ((j_end - j_start_aligned) / simd_width) * simd_width;
          for (int j = j_remainder_start; j < j_end; j++) {
            float sum = C_row[j];
            for (int idx = row_start; idx < row_end; idx++) {
              int col = Ai[idx];
              sum += Ax[idx] * B[col * n + j];
            }
            C_row[j] = sum;
          }
        }
      }
    }
  } else {
    // Non-tiled version (when tile sizes cover full matrix or are invalid)
    // Parallelize across rows
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
      float *C_row = C + i * n;
      int row_start = Ap[i];
      int row_end = Ap[i + 1];

      // Process SIMD-aligned column blocks
      for (int j = 0; j < n_simd; j++) {
        __m256 c_vec = _mm256_loadu_ps(C_row + j * simd_width);

        // For each non-zero in row i of A
        for (int idx = row_start; idx < row_end; idx++) {
          int col = Ai[idx];
          float a_val = Ax[idx];
          __m256 a_broadcast = _mm256_set1_ps(a_val);
          __m256 b_vec = _mm256_loadu_ps(B + col * n + j * simd_width);
          c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
        }

        _mm256_storeu_ps(C_row + j * simd_width, c_vec);
      }

      // Handle remainder columns
      for (int j = n_simd * simd_width; j < n; j++) {
        float sum = C_row[j];
        for (int idx = row_start; idx < row_end; idx++) {
          int col = Ai[idx];
          sum += Ax[idx] * B[col * n + j];
        }
        C_row[j] = sum;
      }
    }
  }
}

} // namespace swiftware::hpp