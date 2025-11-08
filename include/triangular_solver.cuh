// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB04_TRIANGULAR_SOLVER_CUH
#define LAB04_TRIANGULAR_SOLVER_CUH

#include "def.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cusparse.h>
#else
/* allow parsing with non-CUDA compilers / IDEs */
#define __global__
#endif
namespace swiftware::hpp
{

  template<typename T>
  __global__ void sparse_csr_parallel_gpu(T *val, int *col_ind, int *row_ptr, T *x, T *b, int n, int *wave)
  {
    // TODO : implement sparse triangular solve in parallel
  }


}

#endif //LAB04_TRIANGULAR_SOLVER_CUH
