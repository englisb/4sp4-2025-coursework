// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab

#include "benchmark/benchmark.h"
#include "sort_algorithms.h"
#include "utils.h"

#include <iostream>



static void BM_SORT_STD(benchmark::State &state) {
  auto m = state.range(0);
  auto *A = new int[m]();
  swiftware::hpp::fill_random(A, m, 0, 1000000);

  for (auto _: state) {
    std::sort(A, A + m);
  }

  // verify correctness
  if (!swiftware::hpp::is_sorted(A, m)) {
    std::cerr << state.name() <<  "Array is not sorted!" << std::endl;
    state.SkipWithError("Array is not sorted");
  } else {
    state.SetComplexityN(m);
    state.SetItemsProcessed(m);
  }
  delete[] A;
}

static void BM_SELECTION_SORT(benchmark::State &state) {
  auto m = state.range(0);
  auto *A = new int[m]();
  
  for (auto _: state) {
    swiftware::hpp::fill_random(A, m, 0, 1000000);
    swiftware::hpp::selection_sort(A, m);
  }

  // verify correctness
  if (!swiftware::hpp::is_sorted(A, m)) {
    std::cerr << state.name() <<  "Array is not sorted!" << std::endl;
    state.SkipWithError("Array is not sorted");
  } else {
    state.SetComplexityN(m);
    state.SetItemsProcessed(m);
  }
  delete[] A;
}

static void BM_BUBBLE_SORT(benchmark::State &state) {
  auto m = state.range(0);
  auto *A = new int[m]();
  
  for (auto _: state) {
    swiftware::hpp::fill_random(A, m, 0, 1000000);
    swiftware::hpp::bubble_sort(A, m);
  }

  // verify correctness
  if (!swiftware::hpp::is_sorted(A, m)) {
    std::cerr << state.name() <<  "Array is not sorted!" << std::endl;
    state.SkipWithError("Array is not sorted");
  } else {
    state.SetComplexityN(m);
    state.SetItemsProcessed(m);
  }
  delete[] A;
}

static void BM_QUICKSORT(benchmark::State &state) {
  auto m = state.range(0);
  auto *A = new int[m]();
  swiftware::hpp::fill_random(A, m, 0, 1000000);

  for (auto _: state) {
    swiftware::hpp::quick_sort(A,m);
  }

  // verify correctness
  if (!swiftware::hpp::is_sorted(A, m)) {
    std::cerr << state.name() <<  "Array is not sorted!" << std::endl;
    state.SkipWithError("Array is not sorted");
  } else {
    state.SetComplexityN(m);
    state.SetItemsProcessed(m);
  }
  delete[] A;
}

#define MAXSIZE 32*1024


BENCHMARK(BM_SORT_STD)->ArgsProduct({benchmark::CreateRange(4, MAXSIZE, /*multi=*/2), {1}})
    ->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(50);

BENCHMARK(BM_QUICKSORT)->ArgsProduct({benchmark::CreateRange(4, MAXSIZE, /*multi=*/2), {1}})
    ->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(50);

BENCHMARK(BM_SELECTION_SORT)->ArgsProduct({benchmark::CreateRange(4, MAXSIZE, /*multi=*/2), {1}})
    ->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(50);

BENCHMARK(BM_BUBBLE_SORT)->ArgsProduct({benchmark::CreateRange(4, MAXSIZE, /*multi=*/2), {1}})
    ->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(50);



BENCHMARK_MAIN();



// // Main function to run the benchmarks.
// int main(int argc, char** argv) {
//   benchmark::Initialize(&argc, argv);
//   benchmark::RunSpecifiedBenchmarks();
//   return 0;
// }