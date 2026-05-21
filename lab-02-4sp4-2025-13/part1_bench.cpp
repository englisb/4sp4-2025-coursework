// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab
// Distribution of the code is not
// allowed in any form without permission
// from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "cache_size.h"
#include <random>

static void BM_COPY(benchmark::State &state) {
  // Get array size and repetition count from benchmark arguments
  int m = state.range(0);   // Array size
  int REP = state.range(1); // Repetition count

  // Allocate and initialize array A
  double *A = new double[m];

  // Initialize with random data to ensure cache misses
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(1.0, 100.0);

  for (int i = 0; i < m; ++i) {
    A[i] = dis(gen);
  }

  // Set up performance counters
  state.counters["L1_misses"] =
      benchmark::Counter(0, benchmark::Counter::kIsRate);
  state.counters["L2_misses"] =
      benchmark::Counter(0, benchmark::Counter::kIsRate);
  state.counters["L3_misses"] =
      benchmark::Counter(0, benchmark::Counter::kIsRate);

  // Benchmark loop
  for (auto _ : state) {
    // Pause timing for setup
    state.PauseTiming();

    // Resume timing for actual work
    state.ResumeTiming();

    swiftware::hpp::copy_func_2(A, m, REP);
  }

  // Calculate bandwidth (bytes per second)
  state.SetBytesProcessed(state.iterations() * m * sizeof(double) * REP);

  // Clean up
  delete[] A;
}

// Set sizes. Three repetitions should be enough. Make sure not to exceed.
#define MINSIZE_KB 4
#define MAXSIZE_KB 1024

BENCHMARK(BM_COPY)
    ->ArgsProduct({benchmark::CreateRange(MINSIZE_KB * 1024 / 8,
                                          MAXSIZE_KB * 1024 / 8, /*multi=*/2),
                   {1}})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

BENCHMARK_MAIN();
