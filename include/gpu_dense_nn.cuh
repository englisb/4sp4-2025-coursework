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

#include "def.h"

namespace swiftware::hpp
{
    DenseMatrix *gpu_dense_nn_gemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, 
                                    DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);
    
    DenseMatrix *gpu_dense_nn_gemv(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, 
                                    DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);
}
#endif //LAB01_GPU_DENSE_NN_CUH