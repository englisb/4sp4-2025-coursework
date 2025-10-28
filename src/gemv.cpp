// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gemv.h"
#include <immintrin.h>
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp {

    void gemv(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp) {
        //TODO Implement GEMV: y = A * x + y
    }



#ifdef USE_MKL
    void gemvMKL(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp) {
        // MKL GEMV: y = alpha*A*x + beta*y
        // Since y may have initial values, we use beta=1.0 to accumulate
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    m, n,
                    1.0f,        // alpha
                    A, n,        // A is m x n, leading dimension n
                    x, 1,        // x vector with increment 1
                    1.0f,        // beta (accumulate into y)
                    y, 1);       // y vector with increment 1
    }
#endif

}