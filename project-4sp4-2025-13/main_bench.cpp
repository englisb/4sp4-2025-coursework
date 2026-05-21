// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

// Control flags for quick vs comprehensive benchmarking
// Uncomment QUICK_RUN for faster execution (excludes slow 4096 and naive benchmarks at 1024+)
#define QUICK_RUN

#include "benchmark/benchmark.h"
#include "gemm.h"
#include "gemv.h"
#include "spmm.h"
#include "spmv.h"
#include "utils.h"

// Macro to generate powers of 2 tile size registrations (64, 128, 256, 512, 1024)
// Usage: REG_POW2_TILES(BM_GEMM, 64, 64, 64)
#define REG_POW2_TILES(FN, M, N, K, REPS) \
  BENCHMARK(FN)->Args({M, N, K, 8, 8})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS); \
  BENCHMARK(FN)->Args({M, N, K, 16, 16})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS); \
  BENCHMARK(FN)->Args({M, N, K, 32, 32})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS); \
  BENCHMARK(FN)->Args({M, N, K, 64, 64})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS);

// Macro for asymmetric tile variants
#define REG_ASYMM_TILES(FN, M, N, K, T1, T2, REPS) \
  BENCHMARK(FN)->Args({M, N, K, T1, T2})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS);

// Macro for single tile registration
#define REG_TILE(FN, M, N, K, T1, T2, REPS) \
  BENCHMARK(FN)->Args({M, N, K, T1, T2})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS);

// Macro for SPMM tile sizes (powers of 2: 8, 16, 32, 64, 128)
#define REG_SPMM_POW2(FN, M, N, K, REPS) \
  BENCHMARK(FN)->Args({M, N, K, 8, 8})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS); \
  BENCHMARK(FN)->Args({M, N, K, 16, 16})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS); \
  BENCHMARK(FN)->Args({M, N, K, 32, 32})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS); \
  BENCHMARK(FN)->Args({M, N, K, 64, 64})->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(REPS);

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

  for (auto _ : state) {
    swiftware::hpp::gemm_tiled_simd_parallel(m, n, k, A->data.data(), B->data.data(),
                         C->data.data(),
                         swiftware::hpp::ScheduleParams(t1, t2));
  }
  delete A;
  delete B;
  delete C;
}

// Variant-specific GEMM benchmarks
static void BM_GEMM_NAIVE(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    std::fill(A->data.begin(), A->data.end(), 1.0f);
    std::fill(B->data.begin(), B->data.end(), 1.0f);
    for (auto _ : state) {
        swiftware::hpp::gemm_naive(m, n, k, A->data.data(), B->data.data(),
                                                             C->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }
    delete A; delete B; delete C;
}

static void BM_GEMM_SIMD(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    std::fill(A->data.begin(), A->data.end(), 1.0f);
    std::fill(B->data.begin(), B->data.end(), 1.0f);
    for (auto _ : state) {
        swiftware::hpp::gemm_simd(m, n, k, A->data.data(), B->data.data(),
                                                            C->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }
    delete A; delete B; delete C;
}

static void BM_GEMM_SIMD_PAR(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    std::fill(A->data.begin(), A->data.end(), 1.0f);
    std::fill(B->data.begin(), B->data.end(), 1.0f);
    for (auto _ : state) {
        swiftware::hpp::gemm_simd_parallel(m, n, k, A->data.data(), B->data.data(),
                                                                             C->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }
    delete A; delete B; delete C;
}

static void BM_GEMM_TILED_SIMD_PAR(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    std::fill(A->data.begin(), A->data.end(), 1.0f);
    std::fill(B->data.begin(), B->data.end(), 1.0f);
    for (auto _ : state) {
        swiftware::hpp::gemm_tiled_simd_parallel(m, n, k, A->data.data(), B->data.data(),
                                                                                         C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete A; delete B; delete C;
}

// SPMM Naive variant (t1=-1)
static void BM_SPMM_NAIVE(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int sparsity = state.range(3);

  auto *A_dense = new swiftware::hpp::DenseMatrix(m, k);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);
  for (int i = 0; i < m && i < k; i++) {
    A_dense->data[i * k + i] = 1.0f;
    if (i + 1 < k) A_dense->data[i * k + (i + 1)] = 0.5f;
    if (i > 0) A_dense->data[i * k + (i - 1)] = 0.5f;
  }
  auto *A_csr = swiftware::hpp::denseToCSR(A_dense);
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);
  for (int i = 0; i < k * n; ++i) B->data[i] = 1.0;
  std::fill(C->data.begin(), C->data.end(), 0.0f);

  for (auto _ : state) {
    swiftware::hpp::spmmCSR(m, n, k, A_csr->Ap.data(), A_csr->Ai.data(),
                            A_csr->Ax.data(), B->data.data(), C->data.data(),
                            swiftware::hpp::ScheduleParams(-1, 0));
  }
  delete A_dense; delete A_csr; delete B; delete C;
}

// SPMM SIMD variant (t1=0)
static void BM_SPMM_SIMD(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int sparsity = state.range(3);

  auto *A_dense = new swiftware::hpp::DenseMatrix(m, k);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);
  for (int i = 0; i < m && i < k; i++) {
    A_dense->data[i * k + i] = 1.0f;
    if (i + 1 < k) A_dense->data[i * k + (i + 1)] = 0.5f;
    if (i > 0) A_dense->data[i * k + (i - 1)] = 0.5f;
  }
  auto *A_csr = swiftware::hpp::denseToCSR(A_dense);
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);
  for (int i = 0; i < k * n; ++i) B->data[i] = 1.0;
  std::fill(C->data.begin(), C->data.end(), 0.0f);

  for (auto _ : state) {
    swiftware::hpp::spmmCSR(m, n, k, A_csr->Ap.data(), A_csr->Ai.data(),
                            A_csr->Ax.data(), B->data.data(), C->data.data(),
                            swiftware::hpp::ScheduleParams(0, 0));
  }
  delete A_dense; delete A_csr; delete B; delete C;
}

// SPMM SIMD+Parallel variant (t1=1)
static void BM_SPMM_SIMD_PAR(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int sparsity = state.range(3);

  auto *A_dense = new swiftware::hpp::DenseMatrix(m, k);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);
  for (int i = 0; i < m && i < k; i++) {
    A_dense->data[i * k + i] = 1.0f;
    if (i + 1 < k) A_dense->data[i * k + (i + 1)] = 0.5f;
    if (i > 0) A_dense->data[i * k + (i - 1)] = 0.5f;
  }
  auto *A_csr = swiftware::hpp::denseToCSR(A_dense);
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);
  for (int i = 0; i < k * n; ++i) B->data[i] = 1.0;
  std::fill(C->data.begin(), C->data.end(), 0.0f);

  for (auto _ : state) {
    swiftware::hpp::spmmCSR(m, n, k, A_csr->Ap.data(), A_csr->Ai.data(),
                            A_csr->Ax.data(), B->data.data(), C->data.data(),
                            swiftware::hpp::ScheduleParams(1, 0));
  }
  delete A_dense; delete A_csr; delete B; delete C;
}

// SPMM Tiled+SIMD+Parallel variant (t1=16, fixed 16x16 tile)
static void BM_SPMM_TILED(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int sparsity = state.range(3);

  auto *A_dense = new swiftware::hpp::DenseMatrix(m, k);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);
  for (int i = 0; i < m && i < k; i++) {
    A_dense->data[i * k + i] = 1.0f;
    if (i + 1 < k) A_dense->data[i * k + (i + 1)] = 0.5f;
    if (i > 0) A_dense->data[i * k + (i - 1)] = 0.5f;
  }
  auto *A_csr = swiftware::hpp::denseToCSR(A_dense);
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);
  for (int i = 0; i < k * n; ++i) B->data[i] = 1.0;
  std::fill(C->data.begin(), C->data.end(), 0.0f);

  for (auto _ : state) {
    swiftware::hpp::spmmCSR(m, n, k, A_csr->Ap.data(), A_csr->Ai.data(),
                            A_csr->Ax.data(), B->data.data(), C->data.data(),
                            swiftware::hpp::ScheduleParams(16, 16));
  }
  delete A_dense; delete A_csr; delete B; delete C;
}

// SPMV Naive variant (t1=-1)
static void BM_SPMV_NAIVE(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int sparsity = state.range(2);

  auto *A_dense = new swiftware::hpp::DenseMatrix(m, n);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);
  for (int i = 0; i < m && i < n; i++) {
    A_dense->data[i * n + i] = 1.0f;
    if (i + 1 < n) A_dense->data[i * n + (i + 1)] = 0.5f;
    if (i > 0) A_dense->data[i * n + (i - 1)] = 0.5f;
  }
  auto *A_csr = swiftware::hpp::denseToCSR(A_dense);
  auto *b = new swiftware::hpp::DenseMatrix(n, 1);
  auto *c = new swiftware::hpp::DenseMatrix(m, 1);
  for (int i = 0; i < n; ++i) b->data[i] = 1.0;
  std::fill(c->data.begin(), c->data.end(), 0.0f);

  for (auto _ : state) {
    swiftware::hpp::spmvCSR(m, n, A_csr->Ap.data(), A_csr->Ai.data(),
                            A_csr->Ax.data(), b->data.data(), c->data.data(),
                            swiftware::hpp::ScheduleParams(-1, 0));
  }
  delete A_dense; delete A_csr; delete b; delete c;
}

// SPMV Parallel variant (t1=0)
static void BM_SPMV_PARALLEL(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int sparsity = state.range(2);

  auto *A_dense = new swiftware::hpp::DenseMatrix(m, n);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);
  for (int i = 0; i < m && i < n; i++) {
    A_dense->data[i * n + i] = 1.0f;
    if (i + 1 < n) A_dense->data[i * n + (i + 1)] = 0.5f;
    if (i > 0) A_dense->data[i * n + (i - 1)] = 0.5f;
  }
  auto *A_csr = swiftware::hpp::denseToCSR(A_dense);
  auto *b = new swiftware::hpp::DenseMatrix(n, 1);
  auto *c = new swiftware::hpp::DenseMatrix(m, 1);
  for (int i = 0; i < n; ++i) b->data[i] = 1.0;
  std::fill(c->data.begin(), c->data.end(), 0.0f);

  for (auto _ : state) {
    swiftware::hpp::spmvCSR(m, n, A_csr->Ap.data(), A_csr->Ai.data(),
                            A_csr->Ax.data(), b->data.data(), c->data.data(),
                            swiftware::hpp::ScheduleParams(0, 0));
  }
  delete A_dense; delete A_csr; delete b; delete c;
}

// SPMV Tiled+Parallel variant (t1=16, fixed 16x16 tile)
static void BM_SPMV_TILED(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int sparsity = state.range(2);

  auto *A_dense = new swiftware::hpp::DenseMatrix(m, n);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);
  for (int i = 0; i < m && i < n; i++) {
    A_dense->data[i * n + i] = 1.0f;
    if (i + 1 < n) A_dense->data[i * n + (i + 1)] = 0.5f;
    if (i > 0) A_dense->data[i * n + (i - 1)] = 0.5f;
  }
  auto *A_csr = swiftware::hpp::denseToCSR(A_dense);
  auto *b = new swiftware::hpp::DenseMatrix(n, 1);
  auto *c = new swiftware::hpp::DenseMatrix(m, 1);
  for (int i = 0; i < n; ++i) b->data[i] = 1.0;
  std::fill(c->data.begin(), c->data.end(), 0.0f);

  for (auto _ : state) {
    swiftware::hpp::spmvCSR(m, n, A_csr->Ap.data(), A_csr->Ai.data(),
                            A_csr->Ax.data(), b->data.data(), c->data.data(),
                            swiftware::hpp::ScheduleParams(16, 16));
  }
  delete A_dense; delete A_csr; delete b; delete c;
}

// GEMV benchmark with tiling parameters

// Variant-specific GEMV benchmarks
static void BM_GEMV_NAIVE(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);
    std::fill(A->data.begin(), A->data.end(), 1.0f);
    std::fill(x->data.begin(), x->data.end(), 1.0f);
    std::fill(y->data.begin(), y->data.end(), 0.0f);
    for (auto _ : state) {
        swiftware::hpp::gemv_naive(m, n, A->data.data(), x->data.data(), y->data.data(),
                                                             swiftware::hpp::ScheduleParams(-1, -1));
    }
    delete A; delete x; delete y;
}

static void BM_GEMV_SIMD(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);
    std::fill(A->data.begin(), A->data.end(), 1.0f);
    std::fill(x->data.begin(), x->data.end(), 1.0f);
    std::fill(y->data.begin(), y->data.end(), 0.0f);
    for (auto _ : state) {
        swiftware::hpp::gemv_simd(m, n, A->data.data(), x->data.data(), y->data.data(),
                                                            swiftware::hpp::ScheduleParams(-1, -1));
    }
    delete A; delete x; delete y;
}

static void BM_GEMV_SIMD_PAR(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);
    std::fill(A->data.begin(), A->data.end(), 1.0f);
    std::fill(x->data.begin(), x->data.end(), 1.0f);
    std::fill(y->data.begin(), y->data.end(), 0.0f);
    for (auto _ : state) {
        swiftware::hpp::gemv_simd_parallel(m, n, A->data.data(), x->data.data(), y->data.data(),
                                                                             swiftware::hpp::ScheduleParams(-1, -1));
    }
    delete A; delete x; delete y;
}



// Variant-specific registrations (naive and simd variants) - sweep across sizes
// These are for the stacked bar plots showing optimization technique comparison
BENCHMARK(BM_GEMM_NAIVE)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
#ifndef QUICK_RUN
    ->Args({1024, 1024, 1024})
    ->Args({4096, 4096, 4096})
#endif
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_GEMM_SIMD)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
#ifndef QUICK_RUN
    ->Args({4096, 4096, 4096})
#endif
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_GEMM_SIMD_PAR)
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
#ifndef QUICK_RUN
    ->Args({4096, 4096, 4096})
#endif
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// GEMM TILED variant - comprehensive tile size sweeps
// For each matrix size, test multiple tile configurations
BENCHMARK(BM_GEMM_TILED_SIMD_PAR)
    // 64x64x64 - tile sizes from 8 to 64
    ->Args({64, 64, 64, 8, 8})
    ->Args({64, 64, 64, 16, 16})
    ->Args({64, 64, 64, 32, 32})
    ->Args({64, 64, 64, 64, 64})
    // Non-square tiles for 64
    ->Args({64, 64, 64, 16, 32})
    ->Args({64, 64, 64, 32, 16})
    // 128x128x128 - tile sizes from 8 to 128
    ->Args({128, 128, 128, 8, 8})
    ->Args({128, 128, 128, 16, 16})
    ->Args({128, 128, 128, 32, 32})
    ->Args({128, 128, 128, 64, 64})
    ->Args({128, 128, 128, 128, 128})
    // Non-square tiles for 128
    ->Args({128, 128, 128, 16, 32})
    ->Args({128, 128, 128, 32, 64})
    ->Args({128, 128, 128, 64, 32})
    // 256x256x256 - tile sizes from 8 to 256
    ->Args({256, 256, 256, 8, 8})
    ->Args({256, 256, 256, 16, 16})
    ->Args({256, 256, 256, 32, 32})
    ->Args({256, 256, 256, 64, 64})
    ->Args({256, 256, 256, 128, 128})
    ->Args({256, 256, 256, 256, 256})
    // Non-square tiles for 256
    ->Args({256, 256, 256, 32, 64})
    ->Args({256, 256, 256, 64, 128})
    ->Args({256, 256, 256, 128, 64})
    // 512x512x512 - tile sizes from 8 to 512
    ->Args({512, 512, 512, 8, 8})
    ->Args({512, 512, 512, 16, 16})
    ->Args({512, 512, 512, 32, 32})
    ->Args({512, 512, 512, 64, 64})
    ->Args({512, 512, 512, 128, 128})
    ->Args({512, 512, 512, 256, 256})
    ->Args({512, 512, 512, 512, 512})
    // Non-square tiles for 512
    ->Args({512, 512, 512, 64, 128})
    ->Args({512, 512, 512, 128, 256})
    ->Args({512, 512, 512, 256, 128})
    // 1024x1024x1024 - tile sizes from 8 to 1024
    ->Args({1024, 1024, 1024, 8, 8})
    ->Args({1024, 1024, 1024, 16, 16})
    ->Args({1024, 1024, 1024, 32, 32})
    ->Args({1024, 1024, 1024, 64, 64})
    ->Args({1024, 1024, 1024, 128, 128})
    ->Args({1024, 1024, 1024, 256, 256})
    ->Args({1024, 1024, 1024, 512, 512})
    ->Args({1024, 1024, 1024, 1024, 1024})
    // Non-square tiles for 1024
    ->Args({1024, 1024, 1024, 128, 256})
    ->Args({1024, 1024, 1024, 256, 512})
    ->Args({1024, 1024, 1024, 512, 256})
#ifndef QUICK_RUN
    // 4096x4096x4096 - tile sizes from 8 to 1024 (don't go beyond 1024 for tiles)
    ->Args({4096, 4096, 4096, 8, 8})
    ->Args({4096, 4096, 4096, 16, 16})
    ->Args({4096, 4096, 4096, 32, 32})
    ->Args({4096, 4096, 4096, 64, 64})
    ->Args({4096, 4096, 4096, 128, 128})
    ->Args({4096, 4096, 4096, 256, 256})
    ->Args({4096, 4096, 4096, 512, 512})
    ->Args({4096, 4096, 4096, 1024, 1024})
    // Non-square tiles for 4096
    ->Args({4096, 4096, 4096, 256, 512})
    ->Args({4096, 4096, 4096, 512, 1024})
    ->Args({4096, 4096, 4096, 1024, 512})
#endif
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// GEMM registrations using new BENCHMARK() format above

// Variant-specific GEMV registrations - sweep across sizes for stacked bar plots
BENCHMARK(BM_GEMV_NAIVE)
    ->Args({64, 64})
    ->Args({128, 128})
    ->Args({256, 256})
    ->Args({512, 512})
#ifndef QUICK_RUN
    ->Args({1024, 1024})
    ->Args({4096, 4096})
#endif
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_GEMV_SIMD)
    ->Args({64, 64})
    ->Args({128, 128})
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
#ifndef QUICK_RUN
    ->Args({4096, 4096})
#endif
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_GEMV_SIMD_PAR)
    ->Args({64, 64})
    ->Args({128, 128})
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
#ifndef QUICK_RUN
    ->Args({4096, 4096})
#endif
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// SPMM Naive variant registrations - sweep sparsity levels (50-95%)
BENCHMARK(BM_SPMM_NAIVE)
    ->Args({256, 256, 256, 50})
    ->Args({256, 256, 256, 55})
    ->Args({256, 256, 256, 60})
    ->Args({256, 256, 256, 65})
    ->Args({256, 256, 256, 70})
    ->Args({256, 256, 256, 75})
    ->Args({256, 256, 256, 80})
    ->Args({256, 256, 256, 85})
    ->Args({256, 256, 256, 90})
    ->Args({256, 256, 256, 95})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// SPMM SIMD variant registrations
BENCHMARK(BM_SPMM_SIMD)
    ->Args({256, 256, 256, 50})
    ->Args({256, 256, 256, 55})
    ->Args({256, 256, 256, 60})
    ->Args({256, 256, 256, 65})
    ->Args({256, 256, 256, 70})
    ->Args({256, 256, 256, 75})
    ->Args({256, 256, 256, 80})
    ->Args({256, 256, 256, 85})
    ->Args({256, 256, 256, 90})
    ->Args({256, 256, 256, 95})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// SPMM SIMD+Parallel variant registrations
BENCHMARK(BM_SPMM_SIMD_PAR)
    ->Args({256, 256, 256, 50})
    ->Args({256, 256, 256, 55})
    ->Args({256, 256, 256, 60})
    ->Args({256, 256, 256, 65})
    ->Args({256, 256, 256, 70})
    ->Args({256, 256, 256, 75})
    ->Args({256, 256, 256, 80})
    ->Args({256, 256, 256, 85})
    ->Args({256, 256, 256, 90})
    ->Args({256, 256, 256, 95})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// SPMM Tiled+SIMD+Parallel variant registrations (fixed 16x16 tile)
BENCHMARK(BM_SPMM_TILED)
    ->Args({256, 256, 256, 50})
    ->Args({256, 256, 256, 55})
    ->Args({256, 256, 256, 60})
    ->Args({256, 256, 256, 65})
    ->Args({256, 256, 256, 70})
    ->Args({256, 256, 256, 75})
    ->Args({256, 256, 256, 80})
    ->Args({256, 256, 256, 85})
    ->Args({256, 256, 256, 90})
    ->Args({256, 256, 256, 95})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// SPMV Naive variant registrations - sweep sparsity levels (50-95%)
BENCHMARK(BM_SPMV_NAIVE)
    ->Args({256, 256, 50})
    ->Args({256, 256, 55})
    ->Args({256, 256, 60})
    ->Args({256, 256, 65})
    ->Args({256, 256, 70})
    ->Args({256, 256, 75})
    ->Args({256, 256, 80})
    ->Args({256, 256, 85})
    ->Args({256, 256, 90})
    ->Args({256, 256, 95})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// SPMV Parallel variant registrations
BENCHMARK(BM_SPMV_PARALLEL)
    ->Args({256, 256, 50})
    ->Args({256, 256, 55})
    ->Args({256, 256, 60})
    ->Args({256, 256, 65})
    ->Args({256, 256, 70})
    ->Args({256, 256, 75})
    ->Args({256, 256, 80})
    ->Args({256, 256, 85})
    ->Args({256, 256, 90})
    ->Args({256, 256, 95})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// SPMV Tiled+Parallel variant registrations (fixed 16x16 tile)
BENCHMARK(BM_SPMV_TILED)
    ->Args({256, 256, 50})
    ->Args({256, 256, 55})
    ->Args({256, 256, 60})
    ->Args({256, 256, 65})
    ->Args({256, 256, 70})
    ->Args({256, 256, 75})
    ->Args({256, 256, 80})
    ->Args({256, 256, 85})
    ->Args({256, 256, 90})
    ->Args({256, 256, 95})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK_MAIN();
