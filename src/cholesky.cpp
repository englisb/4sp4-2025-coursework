// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include <cmath>
#include "cholesky.h"
namespace swiftware::hpp{

  void cholesky_decomposition(double **A, double **L, int n) {
    // TODO: Implement the non-vectorized Cholesky decomposition
  }

  void cholesky_decomposition_vectorized(double **A, double **L, int n) {
    // TODO: Implement the vectorized Cholesky decomposition using AVX
  }

#ifdef USE_MKL
  int cholesky_decomposition_mkl(double *A, int n) {
    // call chol from mkl
    auto info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A, n);
    return info;
  }
#endif

}
