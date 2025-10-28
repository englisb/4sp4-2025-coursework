// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "utils.h"
#include "dense_nn.h"
#include <iostream>


static void BM_DENSENN(benchmark::State &state) {
    auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
    auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
    auto *features = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

    //TODO: Extract labels and features from mnist dataset

    auto *weightsOutput = swiftware::hpp::readCSV("./data/model/weights_output.csv");
    auto *weightsHidden = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    auto *biasesHidden = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *biasesOutput = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams scheduleParams(state.range(0), state.range(1));

    //TODO : Implement the benchmark
    int correctPredictions = 0;

    for (auto _: state) {
        // Running the NN function

        // TODO: Calculate accuracy
        state.PauseTiming();

        //state.counters["Accuracy"] = accuracy;

        state.ResumeTiming();
    }

    delete mnistData;
    delete labels;
    delete features;
    delete weightsOutput;
    delete weightsHidden;
    delete biasesHidden;
    delete biasesOutput;
}



static void BM_SPARSENN(benchmark::State &state) {
    auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
    auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
    auto *features = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

    //TODO: Extract labels and features from mnist dataset


    // TODO : load sparse weights and convert to CSR format

    //TODO : Implement the benchmark
    int correctPredictions = 0;

    for (auto _: state) {
        //TODO Running the NN function

        // TODO: Calculate accuracy
        state.PauseTiming();




        state.ResumeTiming();
    }

    delete mnistData;
    delete labels;
    delete features;
}


// For baseline and simd where tile sizes are not used
BENCHMARK(BM_DENSENN)
    ->Args({32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);




BENCHMARK_MAIN();
