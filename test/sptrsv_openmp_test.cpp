// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include <vector>
#include <gtest/gtest.h>
#include <numeric>
#include <cmath>

#include "triangular_solver.h"


// Test sequential solver with a small matrix
TEST(SpTRSVSeqTest, SmallMatrix) {
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
    swiftware::hpp::ScheduleParams SP(-1, -1, 1, 1);

    // Call the sequential SpTRSV function
    swiftware::hpp::sptrsv_csr<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    // Expected solution: Ax = b
    // x[0] = b[0]/A[0,0] = 2.0/2.0 = 1.0
    // x[1] = (b[1] - A[1,0]*x[0])/A[1,1] = (8.0 - 3.0*1.0)/5.0 = 1.0
    // x[2] = (b[2] - A[2,0]*x[0] - A[2,1]*x[1])/A[2,2] = (32.0 - 1.0*1.0 - 4.0*1.0)/6.0 = 27.0/6.0 = 4.5
    std::vector<double> expected_x = {1.0, 1.0, 4.5};

    // Verify the solution
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], expected_x[i], 1e-10);
    }
}


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
    swiftware::hpp::ScheduleParams SP(-1, -1, 4, 1);

    // Call the parallel SpTRSV function
    swiftware::hpp::sptrsv_csr_parallel<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    // Expected solution: corrected to 4.5 (was incorrectly 4.0)
    std::vector<double> expected_x = {1.0, 1.0, 4.5};

    // Verify the solution
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], expected_x[i], 1e-10);
    }
}


// Test with identity matrix (simplest case)
TEST(SpTRSVSeqTest, IdentityMatrix) {
    // Identity matrix 3x3
    // [1 0 0]
    // [0 1 0]
    // [0 0 1]
    std::vector<double> values = {1.0, 1.0, 1.0};
    std::vector<int> col_indices = {0, 1, 2};
    std::vector<int> row_pointer = {0, 1, 2, 3};
    int n = 3;

    std::vector<double> b = {5.0, 10.0, 15.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 1, 1);
    swiftware::hpp::sptrsv_csr<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    // For identity matrix, x should equal b
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], b[i], 1e-10);
    }
}


TEST(SpTRSVOMPTest, IdentityMatrix) {
    // Identity matrix 3x3
    std::vector<double> values = {1.0, 1.0, 1.0};
    std::vector<int> col_indices = {0, 1, 2};
    std::vector<int> row_pointer = {0, 1, 2, 3};
    int n = 3;

    std::vector<double> b = {5.0, 10.0, 15.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 4, 1);
    swiftware::hpp::sptrsv_csr_parallel<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    // For identity matrix, x should equal b
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], b[i], 1e-10);
    }
}


// Test with a larger matrix with multiple dependency levels
TEST(SpTRSVSeqTest, LargerMatrix) {
    // 5x5 lower triangular matrix with varying dependencies
    // [1 0 0 0 0]
    // [2 1 0 0 0]
    // [0 3 1 0 0]
    // [1 0 2 1 0]
    // [0 1 0 3 1]
    std::vector<double> values = {1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0};
    std::vector<int> col_indices = {0, 0, 1, 1, 2, 0, 2, 3, 1, 3, 4};
    std::vector<int> row_pointer = {0, 1, 3, 5, 8, 11};
    int n = 5;

    std::vector<double> b = {1.0, 3.0, 4.0, 5.0, 8.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 1, 1);
    swiftware::hpp::sptrsv_csr<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    // Manually compute expected solution:
    // x[0] = 1.0/1.0 = 1.0
    // x[1] = (3.0 - 2.0*1.0)/1.0 = 1.0
    // x[2] = (4.0 - 3.0*1.0)/1.0 = 1.0
    // x[3] = (5.0 - 1.0*1.0 - 2.0*1.0)/1.0 = 2.0
    // x[4] = (8.0 - 1.0*1.0 - 3.0*2.0)/1.0 = 1.0
    std::vector<double> expected_x = {1.0, 1.0, 1.0, 2.0, 1.0};

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], expected_x[i], 1e-10);
    }
}


TEST(SpTRSVOMPTest, LargerMatrix) {
    // Same 5x5 matrix as sequential test
    std::vector<double> values = {1.0, 2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0};
    std::vector<int> col_indices = {0, 0, 1, 1, 2, 0, 2, 3, 1, 3, 4};
    std::vector<int> row_pointer = {0, 1, 3, 5, 8, 11};
    int n = 5;

    std::vector<double> b = {1.0, 3.0, 4.0, 5.0, 8.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 8, 1);
    swiftware::hpp::sptrsv_csr_parallel<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    std::vector<double> expected_x = {1.0, 1.0, 1.0, 2.0, 1.0};

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], expected_x[i], 1e-10);
    }
}


// Test with all ones solution
TEST(SpTRSVSeqTest, AllOnesCase) {
    // Create a matrix where the solution is all ones
    // [2 0 0 0]
    // [1 3 0 0]
    // [2 1 4 0]
    // [1 2 3 5]
    std::vector<double> values = {2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 1.0, 2.0, 3.0, 5.0};
    std::vector<int> col_indices = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
    std::vector<int> row_pointer = {0, 1, 3, 6, 10};
    int n = 4;

    // RHS computed so solution is [1, 1, 1, 1]
    std::vector<double> b = {2.0, 4.0, 7.0, 11.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 1, 1);
    swiftware::hpp::sptrsv_csr<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    std::vector<double> expected_x = {1.0, 1.0, 1.0, 1.0};

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], expected_x[i], 1e-10);
    }
}


TEST(SpTRSVOMPTest, AllOnesCase) {
    // Same test as sequential
    std::vector<double> values = {2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 1.0, 2.0, 3.0, 5.0};
    std::vector<int> col_indices = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
    std::vector<int> row_pointer = {0, 1, 3, 6, 10};
    int n = 4;

    std::vector<double> b = {2.0, 4.0, 7.0, 11.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 8, 1);
    swiftware::hpp::sptrsv_csr_parallel<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    std::vector<double> expected_x = {1.0, 1.0, 1.0, 1.0};

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], expected_x[i], 1e-10);
    }
}


// Test consistency: sequential and parallel should give same results
TEST(SpTRSVConsistencyTest, SeqVsParallel) {
    // Test with a moderately sized matrix
    int n = 10;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_pointer = {0};

    // Create a random lower triangular matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            values.push_back(1.0 + i + j * 0.5);
            col_indices.push_back(j);
        }
        row_pointer.push_back(values.size());
    }

    // RHS vector
    std::vector<double> b(n);
    for (int i = 0; i < n; i++) {
        b[i] = i + 1.0;
    }

    // Solve sequentially
    std::vector<double> x_seq(n, 0.0);
    swiftware::hpp::ScheduleParams SP_seq(-1, -1, 1, 1);
    swiftware::hpp::sptrsv_csr<double>(values.data(), col_indices.data(), row_pointer.data(), x_seq.data(), b.data(), n, &SP_seq);

    // Solve in parallel
    std::vector<double> x_par(n, 0.0);
    swiftware::hpp::ScheduleParams SP_par(-1, -1, 8, 1);
    swiftware::hpp::sptrsv_csr_parallel<double>(values.data(), col_indices.data(), row_pointer.data(), x_par.data(), b.data(), n, &SP_par);

    // Results should match
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x_seq[i], x_par[i], 1e-8);
    }
}


// Test with fractional values
TEST(SpTRSVSeqTest, FractionalValues) {
    // Matrix with fractional coefficients
    // [0.5 0   0  ]
    // [0.2 0.8 0  ]
    // [0.1 0.3 1.5]
    std::vector<double> values = {0.5, 0.2, 0.8, 0.1, 0.3, 1.5};
    std::vector<int> col_indices = {0, 0, 1, 0, 1, 2};
    std::vector<int> row_pointer = {0, 1, 3, 6};
    int n = 3;

    std::vector<double> b = {1.0, 2.0, 3.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 1, 1);
    swiftware::hpp::sptrsv_csr<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    // x[0] = 1.0/0.5 = 2.0
    // x[1] = (2.0 - 0.2*2.0)/0.8 = 1.6/0.8 = 2.0
    // x[2] = (3.0 - 0.1*2.0 - 0.3*2.0)/1.5 = (3.0 - 0.8)/1.5 = 1.466...
    std::vector<double> expected_x = {2.0, 2.0, 22.0/15.0};

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], expected_x[i], 1e-10);
    }
}


TEST(SpTRSVOMPTest, FractionalValues) {
    // Same fractional test
    std::vector<double> values = {0.5, 0.2, 0.8, 0.1, 0.3, 1.5};
    std::vector<int> col_indices = {0, 0, 1, 0, 1, 2};
    std::vector<int> row_pointer = {0, 1, 3, 6};
    int n = 3;

    std::vector<double> b = {1.0, 2.0, 3.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 4, 1);
    swiftware::hpp::sptrsv_csr_parallel<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    std::vector<double> expected_x = {2.0, 2.0, 22.0/15.0};

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x[i], expected_x[i], 1e-10);
    }
}


// Edge case: single element matrix
TEST(SpTRSVSeqTest, SingleElement) {
    std::vector<double> values = {3.0};
    std::vector<int> col_indices = {0};
    std::vector<int> row_pointer = {0, 1};
    int n = 1;

    std::vector<double> b = {6.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 1, 1);
    swiftware::hpp::sptrsv_csr<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    EXPECT_NEAR(x[0], 2.0, 1e-10);
}


TEST(SpTRSVOMPTest, SingleElement) {
    std::vector<double> values = {3.0};
    std::vector<int> col_indices = {0};
    std::vector<int> row_pointer = {0, 1};
    int n = 1;

    std::vector<double> b = {6.0};
    std::vector<double> x(n, 0.0);

    swiftware::hpp::ScheduleParams SP(-1, -1, 2, 1);
    swiftware::hpp::sptrsv_csr_parallel<double>(values.data(), col_indices.data(), row_pointer.data(), x.data(), b.data(), n, &SP);

    EXPECT_NEAR(x[0], 2.0, 1e-10);
}

