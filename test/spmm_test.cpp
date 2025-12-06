// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmm.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

namespace swiftware::hpp {
// TODO: Add more test

// Test basic spmm
TEST(SpmmTest, BasicTest) {
  int m = 4, n = 4, k = 4;

  // Create a sparse matrix A (4x4) with diagonal pattern
  // A = [[1, 0, 0, 0],
  //      [0, 2, 0, 0],
  //      [0, 0, 3, 0],
  //      [0, 0, 0, 4]]
  auto *A_dense = new DenseMatrix(m, k);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);
  A_dense->data[0 * k + 0] = 1.0f; // A[0,0] = 1
  A_dense->data[1 * k + 1] = 2.0f; // A[1,1] = 2
  A_dense->data[2 * k + 2] = 3.0f; // A[2,2] = 3
  A_dense->data[3 * k + 3] = 4.0f; // A[3,3] = 4

  auto *A_csr = denseToCSR(A_dense);

  // Create dense matrix B (4x4) = identity matrix
  // B = [[1, 0, 0, 0],
  //      [0, 1, 0, 0],
  //      [0, 0, 1, 0],
  //      [0, 0, 0, 1]]
  float B[16] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  float C[16] = {0.0f};

  ScheduleParams sp(-1, -1);
  spmmCSR(m, n, k, A_csr->Ap.data(), A_csr->Ai.data(), A_csr->Ax.data(), B, C,
          sp);

  // Expected result: C = A * B = A (since B is identity)
  // C should be [[1, 0, 0, 0],
  //              [0, 2, 0, 0],
  //              [0, 0, 3, 0],
  //              [0, 0, 0, 4]]
  EXPECT_NEAR(C[0 * n + 0], 1.0f, 1e-5);
  EXPECT_NEAR(C[1 * n + 1], 2.0f, 1e-5);
  EXPECT_NEAR(C[2 * n + 2], 3.0f, 1e-5);
  EXPECT_NEAR(C[3 * n + 3], 4.0f, 1e-5);

  // Check that off-diagonal elements are zero
  EXPECT_NEAR(C[0 * n + 1], 0.0f, 1e-5);
  EXPECT_NEAR(C[0 * n + 2], 0.0f, 1e-5);
  EXPECT_NEAR(C[1 * n + 0], 0.0f, 1e-5);

  delete A_dense;
  delete A_csr;
}

} // namespace swiftware::hpp