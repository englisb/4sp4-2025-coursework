// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "include/gpu_dense_nn.cuh"

namespace swiftware::hpp {
    enum class GemmStrategy {
        BASELINE,           // Naive implementation
        SHARED_MEMORY,      // Use shared memory tiling
        COALESCED_MEMORY,   // Optimize memory access patterns
        COMBINED            // Shared memory + coalesced access
    };

    // Baseline implementation (original)
    __global__ void gemm_gpu_baseline(int m, int n, int k, const float *A, const float *B, float *C) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        while(tid < m * n) {
            int row = tid / n;
            int col = tid % n;
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[row * k + p] * B[p * n + col];
            }
            C[row * n + col] += sum;
            tid += blockDim.x * gridDim.x;
        }
    }

    // Shared memory tiling implementation
    #define TILE_SIZE 16
    __global__ void gemm_gpu_shared(int m, int n, int k, const float *A, const float *B, float *C) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        float sum = 0.0f;

        for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            // Load tiles into shared memory
            if (row < m && t * TILE_SIZE + threadIdx.x < k)
                As[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
            else
                As[threadIdx.y][threadIdx.x] = 0.0f;

            if (col < n && t * TILE_SIZE + threadIdx.y < k)
                Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
            else
                Bs[threadIdx.y][threadIdx.x] = 0.0f;

            __syncthreads();

            // Compute partial dot product
            for (int i = 0; i < TILE_SIZE; ++i)
                sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

            __syncthreads();
        }

        if (row < m && col < n)
            C[row * n + col] += sum;
    }

    // Coalesced memory access implementation
    __global__ void gemm_gpu_coalesced(int m, int n, int k, const float *A, const float *B, float *C) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < n) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[row * k + p] * B[p * n + col];
            }
            C[row * n + col] += sum;
        }
    }

    // Combined: shared memory + coalesced access
    __global__ void gemm_gpu_combined(int m, int n, int k, const float *A, const float *B, float *C) {
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        float sum = 0.0f;

        for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            // Coalesced loads into shared memory
            int a_col = t * TILE_SIZE + threadIdx.x;
            int b_row = t * TILE_SIZE + threadIdx.y;

            As[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? A[row * k + a_col] : 0.0f;
            Bs[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? B[b_row * n + col] : 0.0f;

            __syncthreads();

            #pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i)
                sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

            __syncthreads();
        }

        if (row < m && col < n)
            C[row * n + col] += sum;
    }

    // Main dispatcher function
    /// \param m Number of rows of A and C
    /// \param n Number of columns of B and C
    /// \param k Number of columns of A and rows of B
    /// \param strategy Optimization strategy to use
    __host__ void gemm_gpu(int m, int n, int k, const float *A, const float *B, float *C, GemmStrategy strategy) {
        switch(strategy) {
            case GemmStrategy::BASELINE:
                gemm_gpu_baseline<<<gridDim, blockDim>>>(m, n, k, A, B, C);
                break;
            case GemmStrategy::SHARED_MEMORY:
                gemm_gpu_shared<<<gridDim, blockDim>>>(m, n, k, A, B, C);
                break;
            case GemmStrategy::COALESCED_MEMORY:
                gemm_gpu_coalesced<<<gridDim, blockDim>>>(m, n, k, A, B, C);
                break;
            case GemmStrategy::COMBINED:
                gemm_gpu_combined<<<gridDim, blockDim>>>(m, n, k, A, B, C);
                break;
        }
    }

    __global__ void dense_nn_gpu(int batch_size, int input_size, int output_size,
                                 const float *input, const float *weights, const float *bias,
                                 float *output) {
        // Dense NN layer: output = input * weights + bias
        // input is batch_size x input_size (row-major)
        // weights is input_size x output_size (row-major)
        // bias is output_size
        // output is batch_size x output_size (row-major)
    }
} // namespace swiftware