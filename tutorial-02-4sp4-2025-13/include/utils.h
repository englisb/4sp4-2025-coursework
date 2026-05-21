// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef TUT02_UTILS_H
#define TUT02_UTILS_H
namespace swiftware::hpp {
  // create three randome arrays of size n and fill them with random float numbers
  void generate_random_float_arrays(float* A, float* B, float* C, float *Correct, int n);

  // takes three arrays and allocate device arrays and copy data to device
  void allocate_and_copy_to_device(float* h_A, float* h_B, float* h_C,
                                   float** d_A, float** d_B, float** d_C, int n);

  void free_buffers(float* d_a, float* d_b, float* d_out);

  inline void cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
      std::abort();
    }
  }

} // namespace swiftware::hpp

#endif //TUT02_UTILS_H