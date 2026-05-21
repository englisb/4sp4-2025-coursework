// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include <cmath>
#ifdef __AVX__
#include <immintrin.h>
#endif
#include "cholesky.h"
namespace swiftware::hpp{

  void cholesky_decomposition(double **A, double **L, int n) {
    // Zero initialize L (keep upper triangle clean)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) L[i][j] = 0.0;
    }

    for (int j = 0; j < n; ++j) {
      // Diagonal element L[j][j]
      double sumjj = 0.0;
      for (int k = 0; k < j; ++k) {
        double v = L[j][k];
        sumjj += v * v;
      }
      double diag = A[j][j] - sumjj;
      if (diag < 0.0) diag = 0.0; // guard small negative due to round-off
      double Ljj = std::sqrt(diag);
      L[j][j] = Ljj;

      // Off-diagonal elements L[i][j], i > j
      if (Ljj != 0.0) {
        for (int i = j + 1; i < n; ++i) {
          double s = 0.0;
          for (int k = 0; k < j; ++k) {
            s += L[i][k] * L[j][k];
          }
          L[i][j] = (A[i][j] - s) / Ljj;
        }
      } else {
        // Degenerate (should not happen for SPD); keep below-diagonal zeros
        for (int i = j + 1; i < n; ++i) L[i][j] = 0.0;
      }
      // Upper triangle already zero-initialized
    }
  }

  void cholesky_decomposition_vectorized(double **A, double **L, int n) {
    // Zero initialize L
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) L[i][j] = 0.0;
    }

    #ifdef __AVX__
        // AVX path
        for (int j = 0; j < n; ++j) {
          double* Lj = L[j];

          // sum(L[j][k]^2 for k in 0..j-1) using AVX
          double sumjj = 0.0;
          {
            __m256d acc = _mm256_setzero_pd();
            int k = 0;
            for (; k + 4 <= j; k += 4) {
              __m256d v = _mm256_loadu_pd(Lj + k);
              __m256d sq = _mm256_mul_pd(v, v);
              acc = _mm256_add_pd(acc, sq);
            }
            double buf[4];
            _mm256_storeu_pd(buf, acc);
            sumjj = buf[0] + buf[1] + buf[2] + buf[3];
            for (; k < j; ++k) sumjj += Lj[k] * Lj[k];
          }

          double diag = A[j][j] - sumjj;
          if (diag < 0.0) diag = 0.0;
          double Ljj = std::sqrt(diag);
          Lj[j] = Ljj;

          if (Ljj != 0.0) {
            for (int i = j + 1; i < n; ++i) {
              double* Li = L[i];
              double s = 0.0;
              {
                __m256d acc = _mm256_setzero_pd();
                int k = 0;
                for (; k + 4 <= j; k += 4) {
                  __m256d vi = _mm256_loadu_pd(Li + k);
                  __m256d vj = _mm256_loadu_pd(Lj + k);
                  __m256d prod = _mm256_mul_pd(vi, vj);
                  acc = _mm256_add_pd(acc, prod);
                }
                double buf[4];
                _mm256_storeu_pd(buf, acc);
                s = buf[0] + buf[1] + buf[2] + buf[3];
                for (; k < j; ++k) s += Li[k] * Lj[k];
              }
              Li[j] = (A[i][j] - s) / Ljj;
            }
          } else {
            for (int i = j + 1; i < n; ++i) L[i][j] = 0.0;
          }
        }
    #else
        // Scalar path
        for (int j = 0; j < n; ++j) {
          double* Lj = L[j];

          // sum(L[j][k]^2 for k in 0..j-1) using scalar
          double sumjj = 0.0;
          for (int k = 0; k < j; ++k) sumjj += Lj[k] * Lj[k];

          double diag = A[j][j] - sumjj;
          if (diag < 0.0) diag = 0.0;
          double Ljj = std::sqrt(diag);
          Lj[j] = Ljj;

          if (Ljj != 0.0) {
            for (int i = j + 1; i < n; ++i) {
              double* Li = L[i];
              double s = 0.0;
              for (int k = 0; k < j; ++k) s += Li[k] * Lj[k];
              Li[j] = (A[i][j] - s) / Ljj;
            }
          } else {
            for (int i = j + 1; i < n; ++i) L[i][j] = 0.0;
          }
        }
    #endif
  }

#ifdef USE_MKL
  int cholesky_decomposition_mkl(double *A, int n) {
    // call chol from mkl
    auto info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A, n);
    return info;
  }
#endif

}
