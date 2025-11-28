// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "dense_nn.h"
#include "utils.h"
#include <algorithm>

static void BM_DENSENN(benchmark::State &state) {
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);

  // Extract labels and features from mnist dataset
  // Labels are in the first column, features are in the rest
  int numSamples = mnistData->m;
  int numFeatures = mnistData->n - 1;

  // Test only the first 10 samples
  int testSamples = std::min(10, numSamples);

  auto *labels = new swiftware::hpp::DenseMatrix(testSamples, 1);
  auto *features = new swiftware::hpp::DenseMatrix(testSamples, numFeatures);

  // Extract labels (first column) and features (remaining columns)
  for (int i = 0; i < testSamples; i++) {
    labels->data[i] =
        mnistData->data[i * mnistData->n]; // First column is label
    for (int j = 0; j < numFeatures; j++) {
      // Normalize pixel values to [0, 1] range
      features->data[i * numFeatures + j] =
          mnistData->data[i * mnistData->n + j + 1] / 255.0f;
    }
  }

  auto *weightsOutput =
      swiftware::hpp::readCSV("./data/model/weights_output.csv");
  auto *weightsHidden =
      swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
  auto *biasesHidden =
      swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput =
      swiftware::hpp::readCSV("./data/model/biases_output.csv");
  swiftware::hpp::ScheduleParams scheduleParams(state.range(0), state.range(1));

  int correctPredictions = 0;
  double accuracy = 0.0;

  for (auto _ : state) {
    // Running the NN function - use GEMM version
    auto *predictions = swiftware::hpp::dense_nn_gemm(
        features, weightsHidden, weightsOutput, biasesHidden, biasesOutput,
        scheduleParams);

    state.PauseTiming();

    // Calculate accuracy
    correctPredictions = 0;
    for (int i = 0; i < testSamples; i++) {
      int predicted = static_cast<int>(predictions->data[i]);
      int actual = static_cast<int>(labels->data[i]);
      if (predicted == actual) {
        correctPredictions++;
      }
    }
    accuracy = (static_cast<double>(correctPredictions) / testSamples) * 100.0;

    state.counters["Accuracy"] = accuracy;
    state.counters["Correct"] = correctPredictions;
    state.counters["Total"] = testSamples;

    delete predictions;
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
  auto *features =
      new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  // TODO: Extract labels and features from mnist dataset

  // TODO : load sparse weights and convert to CSR format

  // TODO : Implement the benchmark
  int correctPredictions = 0;

  for (auto _ : state) {
    // TODO Running the NN function

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
