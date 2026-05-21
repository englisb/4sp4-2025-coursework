// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <random>

namespace swiftware::hpp {

// takes three arrays and allocate device arrays and copy data to device
  void allocate_and_copy_to_device(float* h_A, float* h_B, float* h_C,
                                   float** d_A, float** d_B, float** d_C, int n) {
    cudaMalloc((void**)d_A, n * sizeof(float));
    cudaMalloc((void**)d_B, n * sizeof(float));
    cudaMalloc((void**)d_C, n * sizeof(float));
    cudaMemcpy(*d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_C, h_C, n * sizeof(float), cudaMemcpyHostToDevice);
  }

  void free_buffers(float* d_a, float* d_b, float* d_out)
  {
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
  }

} // namespace swiftware::hpp