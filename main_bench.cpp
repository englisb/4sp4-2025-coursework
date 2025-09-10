//
// Created by kazem on 8/15/25.
//


#include "benchmark/benchmark.h"
#include "utils.h"
#include "vec_op.h"

#include <iostream>




static void BM_VECOP(benchmark::State &state) {
 auto m = state.range(0);
 std::vector<float> a(m, 1.0);
 std::vector<float> b(m, 2.0);

 for(auto _ : state) {
   std::vector<float> c;
   swiftware::hpp::vec_op(a, b, c);
 }
}

static void BM_VECOP_CUSTOM(benchmark::State &state) {
 auto m = state.range(0);
 std::vector<float> a(m, 1.0);
 std::vector<float> b(m, 2.0);

 for(auto _ : state) {
   std::vector<float> c;
   swiftware::hpp::vec_op_custom(a, b, c);
 }
}

BENCHMARK(BM_VECOP)->ArgsProduct({benchmark::CreateRange(2*1024, 1024*1024, /*multi=*/2)})
    ->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(50);

BENCHMARK(BM_VECOP_CUSTOM)->ArgsProduct({benchmark::CreateRange(1000, 100000, /*multi=*/10)})
  ->Unit(benchmark::kMicrosecond)->Iterations(5)->Repetitions(20);

// TODO: add more benchmarks and apply necessary changes


//BENCHMARK_MAIN();


// Main function to run the benchmarks.
int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}