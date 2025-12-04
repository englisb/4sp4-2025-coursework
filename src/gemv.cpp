// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gemv.h"
#include <immintrin.h>
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp {

// Naive implementation - no optimizations
void gemv_naive(int m, int n, const float *A, const float *x, float *y,
                ScheduleParams Sp) {
  // GEMV: y = A * x + y
  // A is m x n (row-major), x is n x 1, y is m x 1
  
  for (int i = 0; i < m; i++) {
    const float *A_row = A + i * n;
    float sum = 0.0f;
    
    for (int j = 0; j < n; j++) {
      sum += A_row[j] * x[j];
    }
    
    y[i] += sum;
  }
}

// SIMD-optimized implementation (no parallelization)
void gemv_simd(int m, int n, const float *A, const float *x, float *y,
               ScheduleParams Sp) {
  // GEMV: y = A * x + y
  // A is m x n (row-major), x is n x 1, y is m x 1

  const int simd_width = 8; // AVX processes 8 floats at a time
  const int n_simd = n / simd_width;

  for (int i = 0; i < m; i++) {
    const float *A_row = A + i * n;
    float sum = 0.0f;

    // SIMD-vectorized dot product
    __m256 sum_vec = _mm256_setzero_ps();

    // Process 8 elements at a time
    for (int j = 0; j < n_simd; j++) {
      __m256 a_vec = _mm256_loadu_ps(A_row + j * simd_width);
      __m256 x_vec = _mm256_loadu_ps(x + j * simd_width);
      sum_vec = _mm256_fmadd_ps(a_vec, x_vec, sum_vec);
    }

    // Horizontal sum of the 8 elements in sum_vec
    // Reduce 8 -> 4
    __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);

    // Reduce 4 -> 2
    __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));

    // Reduce 2 -> 1
    __m128 sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x55));
    sum = _mm_cvtss_f32(sum_32);

    // Handle remainder elements
    for (int j = n_simd * simd_width; j < n; j++) {
      sum += A_row[j] * x[j];
    }

    // Accumulate into y
    y[i] += sum;
  }
}

// SIMD + OpenMP parallel implementation (current default)
void gemv_simd_parallel(int m, int n, const float *A, const float *x, float *y,
                        ScheduleParams Sp) {
  // GEMV: y = A * x + y
  // A is m x n (row-major), x is n x 1, y is m x 1

  const int simd_width = 8; // AVX processes 8 floats at a time
  const int n_simd = n / simd_width;

// Parallelize across rows
#pragma omp parallel for
  for (int i = 0; i < m; i++) {
    const float *A_row = A + i * n;
    float sum = 0.0f;

    // SIMD-vectorized dot product
    __m256 sum_vec = _mm256_setzero_ps();

    // Process 8 elements at a time
    for (int j = 0; j < n_simd; j++) {
      __m256 a_vec = _mm256_loadu_ps(A_row + j * simd_width);
      __m256 x_vec = _mm256_loadu_ps(x + j * simd_width);
      sum_vec = _mm256_fmadd_ps(a_vec, x_vec, sum_vec);
    }

    // Horizontal sum of the 8 elements in sum_vec
    // Reduce 8 -> 4
    __m128 sum_low = _mm256_extractf128_ps(sum_vec, 0);
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);

    // Reduce 4 -> 2
    __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));

    // Reduce 2 -> 1
    __m128 sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x55));
    sum = _mm_cvtss_f32(sum_32);

    // Handle remainder elements
    for (int j = n_simd * simd_width; j < n; j++) {
      sum += A_row[j] * x[j];
    }

    // Accumulate into y
    y[i] += sum;
  }
}

// This stub is kept for header compatibility
void gemv(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp) {
  gemv_simd_parallel(m, n, A, x, y, Sp);
}

#ifdef USE_MKL
void gemvMKL(int m, int n, const float *A, const float *x, float *y,
             ScheduleParams Sp) {
  // MKL GEMV: y = alpha*A*x + beta*y
  // Since y may have initial values, we use beta=1.0 to accumulate
  cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n,
              1.0f,  // alpha
              A, n,  // A is m x n, leading dimension n
              x, 1,  // x vector with increment 1
              1.0f,  // beta (accumulate into y)
              y, 1); // y vector with increment 1
}
#endif

} // namespace swiftware::hpp