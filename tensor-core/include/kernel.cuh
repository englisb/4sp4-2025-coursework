// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef TUT03_KERNEL_CUH
#define TUT03_KERNEL_CUH

#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

namespace swiftware::hpp
{

#define M 16
#define K 16
// Kernel utilizing standard CUDA Cores for D = A * B where A is 16x16, B is 16xN, C is 16xN
  __global__ void Naive_CUDA_Core_GEMM(const half* A, const half* B, float* C, int N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) { // M=16 is fixed
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) { // K=16 is fixed
        // A is M x K (16x16), B is K x N (16xN)
        sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
      }
      C[row * N + col] = sum;
    }
  }



// WMMA fragment dimensions (M x K x N = 16 x 16 x 16)
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N_TILE = 16;
  constexpr int WMMA_K = 16;

  // Kernel utilizing WMMA for D = A * B + C
  __global__ void WMMA_Tensor_Core_GEMM(const half* A, const half* B, float* C_out, int N) {

    // Fragment Declarations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N_TILE, WMMA_K, half, wmma::row_major> a_frag;
    // B is stored row-major (K x N), so use row_major here to match memory layout
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N_TILE, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N_TILE, WMMA_K, float> acc_frag;

    int tile_idx_n = blockIdx.x;

    if (tile_idx_n * WMMA_N_TILE < N) {
      int tile_n = tile_idx_n * WMMA_N_TILE; // N dimension steps by 16
      // Initialize Accumulator
      wmma::fill_fragment(acc_frag, 0.0f);
      // Load A and B
      const half* A_start = A;                      // A is M x K, row-major
      wmma::load_matrix_sync(a_frag, A_start, K);   // leading dim = K
      const half* B_start = B + tile_n;             // top-left of B tile (row-major)
      wmma::load_matrix_sync(b_frag, B_start, N);   // leading dim = N (full N)
      // Matrix Multiplication and Accumulation
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      // Store Result C (16x16 fragment)
      float* C_tile_start = C_out + tile_n;
      wmma::store_matrix_sync(C_tile_start, acc_frag, N, wmma::mem_row_major);
    }
  }

}
#endif //TUT03_KERNEL_CUH