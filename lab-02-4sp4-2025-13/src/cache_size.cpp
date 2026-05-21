// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab
// Distribution of the code is not
// allowed in any form without permission
// from SwiftWare Lab.

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace swiftware {
namespace hpp {

// Function to measure cache size
void copy_func_2(double *A, int m, int REP) {
  // Allocate destination array B
  double *B = new double[m];

  // Perform repeated sequential copy from A to B
  // volatile is used here to prevent compiler optimization
  volatile double sum = 0.0;

  for (int rep = 0; rep < REP; ++rep) {
    // Sequential copy which will touch every cache line
    for (int i = 0; i < m; ++i) {
      B[i] = A[i];
      sum += B[i];
    }
  }

  delete[] B;
}

} // namespace hpp
} // namespace swiftware
