// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "gemm.h"
#include "gemv.h"
#include "spmm.h"
#include "spmv.h"
#include "utils.h"

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
    swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(),
                         C->data.data(),
                         swiftware::hpp::ScheduleParams(t1, t2));
  }
  delete A;
  delete B;
  delete C;
}

static void BM_SPMM(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);
  int k = state.range(2);
  int t1 = state.range(3);
  int t2 = state.range(4);

  // Create a sparse matrix A (m x k) in CSR format
  // Using a simple pattern: diagonal + some off-diagonal elements
  // This creates a sparse matrix with approximately 10% sparsity
  auto *A_dense = new swiftware::hpp::DenseMatrix(m, k);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);

  // Fill with a pattern: diagonal and some adjacent elements
  for (int i = 0; i < m && i < k; i++) {
    A_dense->data[i * k + i] = 1.0f; // Diagonal
    if (i + 1 < k) {
      A_dense->data[i * k + (i + 1)] = 0.5f; // Next column
    }
    if (i > 0) {
      A_dense->data[i * k + (i - 1)] = 0.5f; // Previous column
    }
  }

  auto *A_csr = swiftware::hpp::denseToCSR(A_dense);

  // Create dense matrices B (k x n) and C (m x n)
  auto *B = new swiftware::hpp::DenseMatrix(k, n);
  auto *C = new swiftware::hpp::DenseMatrix(m, n);

  for (int i = 0; i < k * n; ++i) {
    B->data[i] = 1.0;
  }
  std::fill(C->data.begin(), C->data.end(), 0.0f);

  swiftware::hpp::ScheduleParams scheduleParams(t1, t2);

  for (auto _ : state) {
    swiftware::hpp::spmmCSR(m, n, k, A_csr->Ap.data(), A_csr->Ai.data(),
                            A_csr->Ax.data(), B->data.data(), C->data.data(),
                            scheduleParams);
  }

  delete A_dense;
  delete A_csr;
  delete B;
  delete C;
}

static void BM_SPMV(benchmark::State &state) {
  int m = state.range(0);
  int n = state.range(1);

  // Create a sparse matrix A (m x n) in CSR format
  // Using a simple pattern: diagonal + some off-diagonal elements
  auto *A_dense = new swiftware::hpp::DenseMatrix(m, n);
  std::fill(A_dense->data.begin(), A_dense->data.end(), 0.0f);

  // Fill with a pattern: diagonal and some adjacent elements
  for (int i = 0; i < m && i < n; i++) {
    A_dense->data[i * n + i] = 1.0f; // Diagonal
    if (i + 1 < n) {
      A_dense->data[i * n + (i + 1)] = 0.5f; // Next column
    }
    if (i > 0) {
      A_dense->data[i * n + (i - 1)] = 0.5f; // Previous column
    }
  }

  auto *A_csr = swiftware::hpp::denseToCSR(A_dense);

  // Create dense vectors b (n) and c (m)
  auto *b = new swiftware::hpp::DenseMatrix(n, 1);
  auto *c = new swiftware::hpp::DenseMatrix(m, 1);

  for (int i = 0; i < n; ++i) {
    b->data[i] = 1.0;
  }
  std::fill(c->data.begin(), c->data.end(), 0.0f);

  swiftware::hpp::ScheduleParams scheduleParams(-1, -1);

  for (auto _ : state) {
    swiftware::hpp::spmvCSR(m, n, A_csr->Ap.data(), A_csr->Ai.data(),
                            A_csr->Ax.data(), b->data.data(), c->data.data(),
                            scheduleParams);
  }

  delete A_dense;
  delete A_csr;
  delete b;
  delete c;
}

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

  for (auto _ : state) {
    swiftware::hpp::gemv(m, n, A->data.data(), x->data.data(), y->data.data(),
                         swiftware::hpp::ScheduleParams(-1, -1));
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
    ->Args({64, 64, 64, 16, 16})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({64, 64, 64, 32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({64, 64, 64, 64, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({128, 128, 128, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({128, 128, 128, 16, 16})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({128, 128, 128, 32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({128, 128, 128, 64, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({128, 128, 128, 128, 128})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({256, 256, 256, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

    BENCHMARK(BM_GEMM)
    ->Args({256, 256, 256, 32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({512, 512, 512, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

    BENCHMARK(BM_GEMM)
    ->Args({512, 512, 512, 64, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMM)
    ->Args({1024, 1024, 1024, -1, -1})
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

// SPMM benchmarks
BENCHMARK(BM_SPMM)
    ->Args({64, 64, 64, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPMM)
    ->Args({64, 64, 64, 8, 8})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPMM)
    ->Args({128, 128, 128, 16, 16})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPMM)
    ->Args({256, 256, 256, 32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPMM)
    ->Args({512, 512, 512, 64, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPMM)
    ->Args({1024, 1024, 1024, 128, 128})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

// SPMV benchmarks
BENCHMARK(BM_SPMV)
    ->Args({4096, 4096})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPMV)
    ->Args({8192, 8192})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPMV)
    ->Args({16384, 16384})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK_MAIN();
