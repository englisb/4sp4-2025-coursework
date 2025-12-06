// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmv.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

namespace swiftware::hpp {
// TODO: Add more test

// Test basic spmv
TEST(SpmvTest, BasicTest) {
  int m = 4, n = 4;

  // Create a sparse matrix A (4x4) with diagonal pattern
  // A = [[1, 0, 0, 0],
  //      [0, 2, 0, 0],
  //      [0, 0, 3, 0],
  //      [0, 0, 0, 4]]
  auto *A_dense = new DenseMatrix(m, n);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);
  A_dense->data[0 * n + 0] = 1.0f; // A[0,0] = 1
  A_dense->data[1 * n + 1] = 2.0f; // A[1,1] = 2
  A_dense->data[2 * n + 2] = 3.0f; // A[2,2] = 3
  A_dense->data[3 * n + 3] = 4.0f; // A[3,3] = 4

  auto *A_csr = denseToCSR(A_dense);

  // Create vector b = [1, 2, 3, 4]
  float b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  ScheduleParams sp(-1, -1);
  spmvCSR(m, n, A_csr->Ap.data(), A_csr->Ai.data(), A_csr->Ax.data(), b, c, sp);

  // Expected result: c = A * b = [1*1, 2*2, 3*3, 4*4] = [1, 4, 9, 16]
  EXPECT_NEAR(c[0], 1.0f, 1e-5);
  EXPECT_NEAR(c[1], 4.0f, 1e-5);
  EXPECT_NEAR(c[2], 9.0f, 1e-5);
  EXPECT_NEAR(c[3], 16.0f, 1e-5);

  delete A_dense;
  delete A_csr;
}

} // namespace swiftware::hpp