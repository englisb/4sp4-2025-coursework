// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "def.h"
#include <cuda_runtime.h>

#ifndef LAB01_GPU_DENSE_NN_CUH
#define LAB01_GPU_DENSE_NN_CUH

namespace swiftware::hpp
{
    /// \brief Matrix-matrix multiplication on GPU
    /// \param m Number of rows of A and C
    /// \param n Number of columns of B and C
    /// \param k Number of columns of A and rows of B
    /// \param A Matrix A
    /// \param B Matrix B
    /// \param C Matrix C

    enum class GemmStrategy;

    __host__ void gemm_gpu(int m, int n, int k, const float *A, const float *B, float *C, GemmStrategy strategy);
    __global__ void gemm_gpu_baseline(int m, int n, int k, const float *A, const float *B, float *C);
    __global__ void gemm_gpu_shared(int m, int n, int k, const float *A, const float *B, float *C);
    __global__ void gemm_gpu_coalesced(int m, int n, int k, const float *A, const float *B, float *C);
    __global__ void gemm_gpu_combined(int m, int n, int k, const float *A, const float *B, float *C);

    __global__ void dense_nn_gpu(int batch_size, int input_size, int output_size,
                                 const float *input, const float *weights, const float *bias,
                                 float *output);
}
#endif //LAB01_GPU_DENSE_NN_CUH