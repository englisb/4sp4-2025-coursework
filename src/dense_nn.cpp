// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "dense_nn.h"
#include <cmath>
#include <algorithm>
#include <chrono>


namespace swiftware::hpp {


    // TODO Implement Dense NN with GEMM
    DenseMatrix *dense_nn_gemm(DenseMatrix *InData,
                               DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
        int batchSize = InData->m;
        DenseMatrix *pred = new DenseMatrix(batchSize, 1);


        return pred;
    }

    //TODO: Implement Dense NN with GEMV
    DenseMatrix *dense_nn_gemv(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp){
        int batchSize = InData->m;
        DenseMatrix *pred = new DenseMatrix(batchSize, 1);


        return pred;
    }

#ifdef USE_MKL
    void gemmMKL(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp);
    void gemvMKL(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp);
    
    DenseMatrix *dense_nn_mkl_gemm(DenseMatrix *InData,
                              DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
        int batchSize = InData->m;
        DenseMatrix *pred = new DenseMatrix(batchSize, 1);


        return pred;
    }
    
    DenseMatrix *dense_nn_mkl_gemv(DenseMatrix *InData,
                                   DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
        int batchSize = InData->m;
        DenseMatrix *pred = new DenseMatrix(batchSize, 1);


        return pred;
    }
#endif

}