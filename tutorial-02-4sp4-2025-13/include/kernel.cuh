// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef TUT02_KERNEL_CUH
#define TUT02_KERNEL_CUH

namespace swiftware::hpp
{

  template<typename T>
  __global__ void mulAddKernel_v1(const T* a, const T* b, T* out, size_t n)
  {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) out[i] = a[i] * b[i] + out[i];
  }

  template<typename T>
  __global__ void mulAddKernel_v2(const T* a, const T* b, T* out, size_t n)
  {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i] + out[i];
  }

  template<typename T>
  __global__ void mulAddKernel_v3(const T* a, const T* b, T* out, size_t n)
  {
    // Dynamic mapping with thread coarsening: each thread processes exactly 4 elements
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = tid * 4;

    // Fast path when a full group of 4 fits
    if (idx + 3 < n) {
#pragma unroll
      for (size_t k = 0; k < 4; ++k) {
        const size_t j = idx + k;
        out[j] = a[j] * b[j] + out[j];
      }
    } else if (idx < n) {
      // Tail handling for the last partial chunk
      for (size_t j = idx; j < n; ++j) {
        out[j] = a[j] * b[j] + out[j];
      }
    }
  }

}
#endif //TUT02_KERNEL_CUH