// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab
// Distribution of the code is not
// allowed in any form without permission
// from SwiftWare Lab.


#include "benchmark/benchmark.h"
#include "cache_size.h"
#include <iostream>



static void BM_COPY(benchmark::State &state) {
// TODO implement this benchmark
 state.SkipWithMessage("Not implemented yet");

}

// TODO set sizes. Three repeatitions should be enough. Make sure not to exceed.
#define MAXSIZE 131072

BENCHMARK(BM_COPY)->ArgsProduct({benchmark::CreateRange(1024, MAXSIZE, /*multi=*/2), {1}})
    ->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(3);



BENCHMARK_MAIN();
