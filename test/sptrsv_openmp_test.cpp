// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include <vector>
#include <gtest/gtest.h>
#include <numeric>

#include "triangular_solver.h"


TEST(SpTRSVOMPTest, SmallMatrix) {
    // Define a small lower triangular matrix in CSR format
    // Matrix:
    // [2 0 0]
    // [3 5 0]
    // [1 4 6]
    std::vector<double> values = {2.0, 3.0, 5.0, 1.0, 4.0, 6.0};
    std::vector<int> col_indices = {0, 0, 1, 0, 1, 2};
    std::vector<int> row_pointer = {0, 1, 3, 6};
    int n = 3;

    // Right-hand side vector b
    std::vector<double> b = {2.0, 8.0, 32.0};

    // Solution vector x
    std::vector<double> x(n, 0.0);

    // Schedule parameters
    swiftware::hpp::ScheduleParams SP(-1, -1, 10, 1);

    // Call the parallel SpTRSV function
    swiftware::hpp::sptrsv_csr_parallel<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    // Expected solution is x = [1.0, 1.0, 4.0]
    std::vector<double> expected_x = {1.0, 1.0, 4.0};

    // Verify the solution
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], expected_x[i], 1e-5);
    }
}
//TODO : Add more tests for larger matrices and edge cases

