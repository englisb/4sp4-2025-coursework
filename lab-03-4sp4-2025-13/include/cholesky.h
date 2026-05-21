// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB03_CHOLESKY_H
#define LAB03_CHOLESKY_H

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp
{

  //
  void cholesky_decomposition(double **A, double **L, int n);

  void cholesky_decomposition_vectorized(double **A, double **L, int n);

  // In-place blocked Cholesky decomposition similar to LAPACK's DPOTRF
  void cholesky_decomposition_blocked_inplace(double *A, int n, int blockSize);

#ifdef USE_MKL
  // see here for details: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/potrf.html
  int cholesky_decomposition_mkl(double *A, int n);
#endif

}

#endif //LAB03_CHOLESKY_H
