// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gemm.h"
#include <immintrin.h>
#include <iostream>
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp {

    void gemm(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
       // TODO Implement GEMM: C = A * B + C
    }



#ifdef USE_MKL
    void gemmMKL(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
        // MKL GEMM: C = alpha*A*B + beta*C
        // Since C may have initial values, we use beta=1.0 to accumulate
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    1.0f,        // alpha
                    A, k,        // A is m x k
                    B, n,        // B is k x n
                    1.0f,        // beta (accumulate into C)
                    C, n);       // C is m x n
    }
#endif

}