// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "utils.h"
#include "dense_nn.h"
#include "gemv.h"
#include "spmv.h"
#include "spmm.h"
#include <iostream>
#include <cstdlib>

static void BM_GEMM(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    for (int i = 0; i < m * k; ++i) {
        A->data[i] = 1.0;
    }
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0;
    }

    for (auto _: state) {
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete A;
    delete B;
    delete C;

}

// TODO add more benchmarks for SPMV and SpMV

static void BM_GEMV(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);
    
    for (int i = 0; i < m * n; ++i) {
        A->data[i] = 1.0;
    }
    for (int i = 0; i < n; ++i) {
        x->data[i] = 1.0;
    }

    for (auto _: state) {
        swiftware::hpp::gemv(m, n, A->data.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }
    
    delete A;
    delete x;
    delete y;
}



// For baseline and simd where tile sizes are not used
BENCHMARK(BM_GEMM)
    ->Args({64, 64, 64, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

// GEMM with various tile sizes (must be < matrix dims)
BENCHMARK(BM_GEMM)
    ->Args({64, 64, 64, 8, 8})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({128, 128, 128, 16, 16})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({256, 256, 256, 32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({512, 512, 512, 64, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({1024, 1024, 1024, 128, 128})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

// Optional larger tiles only for bigger sizes
BENCHMARK(BM_GEMM)
    ->Args({1024, 1024, 1024, 256, 256})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

// Asymmetric tiling
BENCHMARK(BM_GEMM)
    ->Args({512, 512, 512, 64, 128})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({512, 512, 512, 128, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({256, 256, 256, 32, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({256, 256, 256, 64, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

// GEMV benchmark
BENCHMARK(BM_GEMV)
    ->Args({4096, 4096})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);



BENCHMARK_MAIN();
