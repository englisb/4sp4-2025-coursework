// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "gtest/gtest.h"
#include "triangular_solver.cuh"
#include "sparse_io.h"
#include <numeric>
#include <cmath>

#define CUDA_CHECK_TEST(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    FAIL() << "CUDA error: " << cudaGetErrorString(e); \
  } \
} while(0)

TEST(SpTRSVGPUTest, SmallDiagonalMatrix) {
    const size_t n = 4;
    // Simple diagonal matrix: each row has only diagonal element
    // [ 2  0  0  0 ]   [1]   [2]
    // [ 0  3  0  0 ] * [1] = [3]
    // [ 0  0  4  0 ]   [1]   [4]
    // [ 0  0  0  5 ]   [1]   [5]
    
    std::vector<double> values = {2.0, 3.0, 4.0, 5.0};
    std::vector<int> col_indices = {0, 1, 2, 3};
    std::vector<int> row_ptr = {0, 1, 2, 3, 4};
    std::vector<double> b = {2.0, 3.0, 4.0, 5.0};
    std::vector<double> x(n, 0.0);
    
    // Allocate device memory
    double *d_val, *d_x, *d_b;
    int *d_col_ind, *d_row_ptr, *d_wave;
    
    CUDA_CHECK_TEST(cudaMalloc(&d_val, values.size() * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&d_col_ind, col_indices.size() * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&d_wave, n * sizeof(int)));
    
    // Copy to device
    CUDA_CHECK_TEST(cudaMemcpy(d_val, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_col_ind, col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(d_x, 0, n * sizeof(double)));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    swiftware::hpp::sparse_csr_parallel_gpu<double><<<blocksPerGrid, threadsPerBlock>>>(
        d_val, d_col_ind, d_row_ptr, d_x, d_b, n, d_wave
    );
    CUDA_CHECK_TEST(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK_TEST(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Verify: solution should be [1, 1, 1, 1]
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], 1.0, 1e-6) << "Mismatch at index " << i;
    }
    
    // Cleanup
    CUDA_CHECK_TEST(cudaFree(d_val));
    CUDA_CHECK_TEST(cudaFree(d_col_ind));
    CUDA_CHECK_TEST(cudaFree(d_row_ptr));
    CUDA_CHECK_TEST(cudaFree(d_x));
    CUDA_CHECK_TEST(cudaFree(d_b));
    CUDA_CHECK_TEST(cudaFree(d_wave));
}

TEST(SpTRSVGPUTest, LowerTriangularMatrix) {
    const size_t n = 4;
    // Lower triangular matrix:
    // [ 2  0  0  0 ]   [1]   [ 2]
    // [ 1  3  0  0 ] * [1] = [ 4]
    // [ 0  2  4  0 ]   [1]   [ 6]
    // [ 0  0  3  5 ]   [1]   [ 8]
    
    std::vector<double> values = {2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0};
    std::vector<int> col_indices = {0, 0, 1, 1, 2, 2, 3};
    std::vector<int> row_ptr = {0, 1, 3, 5, 7};
    std::vector<double> b = {2.0, 4.0, 6.0, 8.0};
    std::vector<double> x(n, 0.0);
    
    // Allocate device memory
    double *d_val, *d_x, *d_b;
    int *d_col_ind, *d_row_ptr, *d_wave;
    
    CUDA_CHECK_TEST(cudaMalloc(&d_val, values.size() * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&d_col_ind, col_indices.size() * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&d_wave, n * sizeof(int)));
    
    // Copy to device
    CUDA_CHECK_TEST(cudaMemcpy(d_val, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_col_ind, col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(d_x, 0, n * sizeof(double)));
    
    // Launch kernel iteratively
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    for (int iter = 0; iter < n; ++iter) {
        swiftware::hpp::sparse_csr_parallel_gpu<double><<<blocksPerGrid, threadsPerBlock>>>(
            d_val, d_col_ind, d_row_ptr, d_x, d_b, n, d_wave
        );
        CUDA_CHECK_TEST(cudaDeviceSynchronize());
    }
    
    // Copy result back
    CUDA_CHECK_TEST(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Verify: solution should be [1, 1, 1, 1]
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], 1.0, 1e-6) << "Mismatch at index " << i;
    }
    
    // Cleanup
    CUDA_CHECK_TEST(cudaFree(d_val));
    CUDA_CHECK_TEST(cudaFree(d_col_ind));
    CUDA_CHECK_TEST(cudaFree(d_row_ptr));
    CUDA_CHECK_TEST(cudaFree(d_x));
    CUDA_CHECK_TEST(cudaFree(d_b));
    CUDA_CHECK_TEST(cudaFree(d_wave));
}

TEST(SpTRSVGPUTest, MediumMatrix) {
    const size_t n = 100;
    // Create a larger lower triangular matrix with diagonal = 2, subdiagonal = -1
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    row_ptr.push_back(0);
    
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            values.push_back(-1.0);
            col_indices.push_back(i - 1);
        }
        values.push_back(2.0);
        col_indices.push_back(i);
        row_ptr.push_back(values.size());
    }
    
    // Build RHS for solution = all ones
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) {
        b[i] = 2.0;
        if (i > 0) b[i] -= 1.0;
    }
    
    std::vector<double> x(n, 0.0);
    
    // Allocate device memory
    double *d_val, *d_x, *d_b;
    int *d_col_ind, *d_row_ptr, *d_wave;
    
    CUDA_CHECK_TEST(cudaMalloc(&d_val, values.size() * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&d_col_ind, col_indices.size() * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&d_row_ptr, row_ptr.size() * sizeof(int)));
    CUDA_CHECK_TEST(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK_TEST(cudaMalloc(&d_wave, n * sizeof(int)));
    
    // Copy to device
    CUDA_CHECK_TEST(cudaMemcpy(d_val, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_col_ind, col_indices.data(), col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_row_ptr, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK_TEST(cudaMemset(d_x, 0, n * sizeof(double)));
    
    // Launch kernel iteratively
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    for (int iter = 0; iter < n; ++iter) {
        swiftware::hpp::sparse_csr_parallel_gpu<double><<<blocksPerGrid, threadsPerBlock>>>(
            d_val, d_col_ind, d_row_ptr, d_x, d_b, n, d_wave
        );
        CUDA_CHECK_TEST(cudaDeviceSynchronize());
    }
    
    // Copy result back
    CUDA_CHECK_TEST(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Verify: solution should be all ones
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], 1.0, 1e-5) << "Mismatch at index " << i;
    }
    
    // Cleanup
    CUDA_CHECK_TEST(cudaFree(d_val));
    CUDA_CHECK_TEST(cudaFree(d_col_ind));
    CUDA_CHECK_TEST(cudaFree(d_row_ptr));
    CUDA_CHECK_TEST(cudaFree(d_x));
    CUDA_CHECK_TEST(cudaFree(d_b));
    CUDA_CHECK_TEST(cudaFree(d_wave));
}

TEST(COOtoCSRTest, ConversionTest) {
    // Test COO to CSR conversion
    swiftware::hpp::Matrix<double> coo;
    coo.rows = 4;
    coo.cols = 4;
    coo.non_zeros = 7;
    coo.row_indices = {0, 1, 1, 2, 2, 3, 3};
    coo.col_indices = {0, 0, 1, 1, 2, 2, 3};
    coo.values = {2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0};
    coo.type = swiftware::hpp::MatrixType::General;
    
    auto csr = swiftware::hpp::COO_to_CSR(coo);
    
    EXPECT_EQ(csr.rows, 4);
    EXPECT_EQ(csr.cols, 4);
    EXPECT_EQ(csr.non_zeros, 7);
    EXPECT_EQ(csr.row_pointer.size(), 5);
    EXPECT_EQ(csr.col_indices.size(), 7);
    EXPECT_EQ(csr.values.size(), 7);
    
    // Verify row_pointer
    std::vector<int> expected_row_ptr = {0, 1, 3, 5, 7};
    for (size_t i = 0; i < expected_row_ptr.size(); ++i) {
        EXPECT_EQ(csr.row_pointer[i], expected_row_ptr[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}