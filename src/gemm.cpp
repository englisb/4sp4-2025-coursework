// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gemm.h"
#include <immintrin.h>
#include <iostream>
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp {

void gemm(int m, int n, int k, const float *A, const float *B, float *C,
          ScheduleParams Sp) {
  // GEMM: C = A * B + C
  // A is m x k (row-major), B is k x n (row-major), C is m x n (row-major)

  const int simd_width = 8; // AVX processes 8 floats at a time
  const int n_simd = n / simd_width;

  // Determine tile sizes (use defaults if not provided or invalid)
  // For GEMM, we can use tiling to improve cache locality
  int tile_m = (Sp.TileSize1 > 0) ? Sp.TileSize1 : m;
  int tile_n = (Sp.TileSize2 > 0) ? Sp.TileSize2 : n;

  // Use tiling if tile sizes are reasonable
  if (tile_m < m || tile_n < n) {
// Tiled GEMM with cache blocking
#pragma omp parallel for collapse(2)
    for (int ii = 0; ii < m; ii += tile_m) {
      for (int jj = 0; jj < n; jj += tile_n) {
        int i_end = (ii + tile_m < m) ? ii + tile_m : m;
        int j_end = (jj + tile_n < n) ? jj + tile_n : n;

        // Process tile [ii:i_end, jj:j_end]
        for (int i = ii; i < i_end; i++) {
          const float *A_row = A + i * k;
          float *C_row = C + i * n;

          // Handle unaligned columns at the start
          int j_start = jj;
          int j_start_aligned =
              ((jj + simd_width - 1) / simd_width) * simd_width;
          if (j_start_aligned > j_end) {
            j_start_aligned = j_end;
          }

          for (int j = j_start; j < j_start_aligned; j++) {
            float sum = C_row[j];
            for (int l = 0; l < k; l++) {
              sum += A_row[l] * B[l * n + j];
            }
            C_row[j] = sum;
          }

          // Process SIMD-aligned blocks in this tile
          for (int j = j_start_aligned; j + simd_width <= j_end;
               j += simd_width) {
            __m256 c_vec = _mm256_loadu_ps(C_row + j);

            // Compute dot product: C[i][j:j+8] += A[i][:] * B[:][j:j+8]
            for (int l = 0; l < k; l++) {
              __m256 a_broadcast = _mm256_set1_ps(A_row[l]);
              __m256 b_vec = _mm256_loadu_ps(B + l * n + j);
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
            for (int l = 0; l < k; l++) {
              sum += A_row[l] * B[l * n + j];
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
      const float *A_row = A + i * k;
      float *C_row = C + i * n;

      // For each column block in C
      for (int j = 0; j < n_simd; j++) {
        __m256 c_vec = _mm256_loadu_ps(C_row + j * simd_width);

        // Compute dot product: C[i][j:j+8] += A[i][:] * B[:][j:j+8]
        for (int l = 0; l < k; l++) {
          __m256 a_broadcast = _mm256_set1_ps(A_row[l]);
          __m256 b_vec = _mm256_loadu_ps(B + l * n + j * simd_width);
          c_vec = _mm256_fmadd_ps(a_broadcast, b_vec, c_vec);
        }

        _mm256_storeu_ps(C_row + j * simd_width, c_vec);
      }

      // Handle remainder columns
      for (int j = n_simd * simd_width; j < n; j++) {
        float sum = C_row[j];
        for (int l = 0; l < k; l++) {
          sum += A_row[l] * B[l * n + j];
        }
        C_row[j] = sum;
      }
    }
  }
}

#ifdef USE_MKL
void gemmMKL(int m, int n, int k, const float *A, const float *B, float *C,
             ScheduleParams Sp) {
  // MKL GEMM: C = alpha*A*B + beta*C
  // Since C may have initial values, we use beta=1.0 to accumulate
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              1.0f,  // alpha
              A, k,  // A is m x k
              B, n,  // B is k x n
              1.0f,  // beta (accumulate into C)
              C, n); // C is m x n
}
#endif

} // namespace swiftware::hpp