// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "gtest/gtest.h"
#include "cholesky.h"
#include <vector>
#include <cmath>

using namespace swiftware::hpp;

// Test 1: Identity matrix should yield identity lower-triangular L
TEST(CholeskyTest, IdentityMatrix) {
  const int n = 4;
  std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
  std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
  for (int i = 0; i < n; ++i) A[i][i] = 1.0;

  std::vector<double*> Ap(n), Lp(n);
  for (int i = 0; i < n; ++i) { Ap[i] = A[i].data(); Lp[i] = L[i].data(); }

  cholesky_decomposition(Ap.data(), Lp.data(), n);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) {
        EXPECT_NEAR(L[i][j], 1.0, 1e-12);
      } else if (i < j) {
        EXPECT_NEAR(L[i][j], 0.0, 1e-12);
      } else {
        // below diagonal can be anything, but for identity it should be zero
        EXPECT_NEAR(L[i][j], 0.0, 1e-12);
      }
    }
  }
}

// Test 2: Known 3x3 SPD matrix with closed-form Cholesky
// A = [ [  4,  12, -16],
//       [ 12,  37, -43],
//       [-16, -43,  98] ]
// L = [ [ 2, 0, 0],
//       [ 6, 1, 0],
//       [-8, 5, 3] ]
TEST(CholeskyTest, Known3x3SPD) {
  const int n = 3;
  std::vector<std::vector<double>> A = {
      { 4.0, 12.0, -16.0 },
      {12.0, 37.0, -43.0 },
      {-16.0,-43.0, 98.0 }
  };
  std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

  std::vector<double*> Ap(n), Lp(n);
  for (int i = 0; i < n; ++i) { Ap[i] = A[i].data(); Lp[i] = L[i].data(); }

  cholesky_decomposition(Ap.data(), Lp.data(), n);

  const double expected[3][3] = {
      { 2.0, 0.0, 0.0 },
      { 6.0, 1.0, 0.0 },
      {-8.0, 5.0, 3.0 }
  };

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_NEAR(L[i][j], expected[i][j], 1e-9);
    }
  }
}

// Test 3: Non-positive-definite symmetric matrix should not crash and will zero-out where needed
// Using A = [[1, 2], [2, 1]] which is symmetric but indefinite.
// The implementation guards negative diagonals by clamping to zero.
TEST(CholeskyTest, NonPositiveDefiniteSymmetric) {
  const int n = 2;
  std::vector<std::vector<double>> A = {
      {1.0, 2.0},
      {2.0, 1.0}
  };
  std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

  std::vector<double*> Ap(n), Lp(n);
  for (int i = 0; i < n; ++i) { Ap[i] = A[i].data(); Lp[i] = L[i].data(); }

  // Should not throw or crash
  cholesky_decomposition(Ap.data(), Lp.data(), n);

  // For this A, the first step gives L00 = 1, L10 = 2.
  // Second diagonal would be sqrt(max(0, 1 - 2^2)) = 0; ensure it's clamped to 0.
  EXPECT_NEAR(L[0][0], 1.0, 1e-12);
  EXPECT_NEAR(L[1][0], 2.0, 1e-12);
  EXPECT_NEAR(L[1][1], 0.0, 1e-12);

  // Upper triangle stays zeroed by implementation contract
  EXPECT_NEAR(L[0][1], 0.0, 1e-12);
}