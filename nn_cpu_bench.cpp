// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "dense_nn.h"
#include "sparse_nn.h"
#include "utils.h"

static void BM_DENSENN(benchmark::State &state) {
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features =
      new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  // Extract labels and features from mnist dataset
  // Labels are in the first column, features are in the remaining columns
  // Features need to be normalized to [0, 1] range (divide by 255.0)
  for (int i = 0; i < mnistData->m; i++) {
    // Extract label from first column
    labels->data[i] = mnistData->data[i * mnistData->n + 0];

    // Extract features from remaining columns
    for (int j = 1; j < mnistData->n; j++) {
      features->data[i * features->n + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0;
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

  // Implement the benchmark
  int correctPredictions = 0;
  // int totalSamples = features->m;
  int totalSamples = 10;

  for (auto _ : state) {
    // Running the NN function
    auto *predictions = swiftware::hpp::dense_nn_gemm(
        features, weightsHidden, weightsOutput, biasesHidden, biasesOutput,
        scheduleParams);

    // Calculate accuracy
    state.PauseTiming();

    correctPredictions = 0;
    for (int i = 0; i < totalSamples; i++) {
      if (static_cast<int>(predictions->data[i]) ==
          static_cast<int>(labels->data[i])) {
        correctPredictions++;
      }
    }

    double accuracy =
        (static_cast<double>(correctPredictions) / totalSamples) * 100.0;
    state.counters["Accuracy"] = accuracy;

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

  // Extract labels and features from mnist dataset
  // Labels are in the first column, features are in the remaining columns
  // Features need to be normalized to [0, 1] range (divide by 255.0)
  for (int i = 0; i < mnistData->m; i++) {
    // Extract label from first column
    labels->data[i] = mnistData->data[i * mnistData->n + 0];

    // Extract features from remaining columns
    for (int j = 1; j < mnistData->n; j++) {
      features->data[i * features->n + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0;
    }
  }

  // Load sparse weights and convert to CSR format
  // Using 50% sparsity as an example (can be changed to other sparsity levels)
  auto *weightsHiddenCSR =
      swiftware::hpp::loadPrunedWeightsCSR("./data/model/50_W1.csv");
  auto *weightsOutputCSR =
      swiftware::hpp::loadPrunedWeightsCSR("./data/model/50_W2.csv");
  auto *biasesHidden =
      swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput =
      swiftware::hpp::readCSV("./data/model/biases_output.csv");
  swiftware::hpp::ScheduleParams scheduleParams(state.range(0), state.range(1));

  // Implement the benchmark
  int correctPredictions = 0;
  // int totalSamples = features->m;
  int totalSamples = 10;

  for (auto _ : state) {
    // Running the NN function (using SPMM version for batch processing)
    auto *predictions = swiftware::hpp::sparseNNSpmm(
        features, weightsHiddenCSR, weightsOutputCSR, biasesHidden,
        biasesOutput, scheduleParams);

    // Calculate accuracy
    state.PauseTiming();

    correctPredictions = 0;
    for (int i = 0; i < totalSamples; i++) {
      if (static_cast<int>(predictions->data[i]) ==
          static_cast<int>(labels->data[i])) {
        correctPredictions++;
      }
    }

    double accuracy =
        (static_cast<double>(correctPredictions) / totalSamples) * 100.0;
    state.counters["Accuracy"] = accuracy;

    delete predictions;
    state.ResumeTiming();
  }

  delete mnistData;
  delete labels;
  delete features;
  delete weightsHiddenCSR;
  delete weightsOutputCSR;
  delete biasesHidden;
  delete biasesOutput;
}

// Benchmark comparing Magnitude-based pruning vs SparseGPT pruning
static void BM_SPARSITY_COMPARISON_MAGNITUDE(benchmark::State &state) {
  // Load MNIST data
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features =
      new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  // Extract labels and features
  for (int i = 0; i < mnistData->m; i++) {
    labels->data[i] = mnistData->data[i * mnistData->n + 0];
    for (int j = 1; j < mnistData->n; j++) {
      features->data[i * features->n + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0;
    }
  }

  // Load magnitude-based pruned weights (50% sparsity)
  auto *weightsHiddenCSR =
      swiftware::hpp::loadPrunedWeightsCSR("./data/model/90_W1.csv");
  auto *weightsOutputCSR =
      swiftware::hpp::loadPrunedWeightsCSR("./data/model/90_W2.csv");
  auto *biasesHidden =
      swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput =
      swiftware::hpp::readCSV("./data/model/biases_output.csv");
  swiftware::hpp::ScheduleParams scheduleParams(state.range(0), state.range(1));

  int correctPredictions = 0;
  int totalSamples = 10000;

  for (auto _ : state) {
    // Running the sparse NN with magnitude-pruned weights
    auto *predictions = swiftware::hpp::sparseNNSpmm(
        features, weightsHiddenCSR, weightsOutputCSR, biasesHidden,
        biasesOutput, scheduleParams);

    // Calculate accuracy
    state.PauseTiming();
    correctPredictions = 0;
    for (int i = 0; i < totalSamples; i++) {
      if (static_cast<int>(predictions->data[i]) ==
          static_cast<int>(labels->data[i])) {
        correctPredictions++;
      }
    }
    double accuracy =
        (static_cast<double>(correctPredictions) / totalSamples) * 100.0;
    state.counters["Accuracy"] = accuracy;
    delete predictions;
    state.ResumeTiming();
  }

  delete mnistData;
  delete labels;
  delete features;
  delete weightsHiddenCSR;
  delete weightsOutputCSR;
  delete biasesHidden;
  delete biasesOutput;
}

static void BM_SPARSITY_COMPARISON_SPARSEGPT(benchmark::State &state) {
  // Load MNIST data
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features =
      new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  // Extract labels and features
  for (int i = 0; i < mnistData->m; i++) {
    labels->data[i] = mnistData->data[i * mnistData->n + 0];
    for (int j = 1; j < mnistData->n; j++) {
      features->data[i * features->n + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0;
    }
  }

  // Load SparseGPT pruned weights (50% sparsity)
  auto *weightsHiddenCSR =
      swiftware::hpp::loadPrunedWeightsCSR("./sparseGPT/weights_hidden.csv");
  auto *weightsOutputCSR =
      swiftware::hpp::loadPrunedWeightsCSR("./sparseGPT/weights_output.csv");
  auto *biasesHidden = swiftware::hpp::readCSV("./sparseGPT/biases_hidden.csv");
  auto *biasesOutput = swiftware::hpp::readCSV("./sparseGPT/biases_output.csv");
  swiftware::hpp::ScheduleParams scheduleParams(state.range(0), state.range(1));

  int correctPredictions = 0;
  int totalSamples = 10000;

  for (auto _ : state) {
    // Running the sparse NN with SparseGPT-pruned weights
    auto *predictions = swiftware::hpp::sparseNNSpmm(
        features, weightsHiddenCSR, weightsOutputCSR, biasesHidden,
        biasesOutput, scheduleParams);

    // Calculate accuracy
    state.PauseTiming();
    correctPredictions = 0;
    for (int i = 0; i < totalSamples; i++) {
      if (static_cast<int>(predictions->data[i]) ==
          static_cast<int>(labels->data[i])) {
        correctPredictions++;
      }
    }
    double accuracy =
        (static_cast<double>(correctPredictions) / totalSamples) * 100.0;
    state.counters["Accuracy"] = accuracy;
    delete predictions;
    state.ResumeTiming();
  }

  delete mnistData;
  delete labels;
  delete features;
  delete weightsHiddenCSR;
  delete weightsOutputCSR;
  delete biasesHidden;
  delete biasesOutput;
}

// For baseline and simd where tile sizes are not used
BENCHMARK(BM_DENSENN)
    ->Args({32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(1);

BENCHMARK(BM_SPARSENN)
    ->Args({32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(1);

// Sparsity comparison benchmarks: Magnitude-based vs SparseGPT
BENCHMARK(BM_SPARSITY_COMPARISON_MAGNITUDE)
    ->Args({32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPARSITY_COMPARISON_SPARSEGPT)
    ->Args({32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK_MAIN();
