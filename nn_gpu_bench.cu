// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gpu_dense_nn.cuh"
#include "gpu_sparse_nn.cuh"
#include "utils.h"
#include <nvbench/nvbench.cuh>

void nvbench_gpu_dense_nn_gemm(nvbench::state &state) {
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features =
      new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  // Extract labels and features from mnist dataset
  for (int i = 0; i < mnistData->m; i++) {
    labels->data[i] = mnistData->data[i * mnistData->n + 0];
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

  swiftware::hpp::ScheduleParams scheduleParams(32, 32);

  int batchSize = static_cast<int>(state.get_int64("batch_size"));
  auto *batchFeatures = new swiftware::hpp::DenseMatrix(batchSize, features->n);
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < features->n; j++) {
      batchFeatures->data[i * features->n + j] =
          features->data[i * features->n + j];
    }
  }

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) {
    auto *predictions = swiftware::hpp::gpu_dense_nn_gemm(
        batchFeatures, weightsHidden, weightsOutput, biasesHidden, biasesOutput,
        scheduleParams);

    // Calculate accuracy
    int correctPredictions = 0;
    for (int i = 0; i < batchSize; i++) {
      if (static_cast<int>(predictions->data[i]) ==
          static_cast<int>(labels->data[i])) {
        correctPredictions++;
      }
    }
    double accuracy =
        (static_cast<double>(correctPredictions) / batchSize) * 100.0;

    delete predictions;
  });

  delete batchFeatures;
  delete mnistData;
  delete labels;
  delete features;
  delete weightsOutput;
  delete weightsHidden;
  delete biasesHidden;
  delete biasesOutput;
}

void nvbench_gpu_dense_nn_gemv(nvbench::state &state) {
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features =
      new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  for (int i = 0; i < mnistData->m; i++) {
    labels->data[i] = mnistData->data[i * mnistData->n + 0];
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

  swiftware::hpp::ScheduleParams scheduleParams(32, 32);

  int batchSize = 10;
  auto *batchFeatures = new swiftware::hpp::DenseMatrix(batchSize, features->n);
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < features->n; j++) {
      batchFeatures->data[i * features->n + j] =
          features->data[i * features->n + j];
    }
  }

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) {
    auto *predictions = swiftware::hpp::gpu_dense_nn_gemv(
        batchFeatures, weightsHidden, weightsOutput, biasesHidden, biasesOutput,
        scheduleParams);

    int correctPredictions = 0;
    for (int i = 0; i < batchSize; i++) {
      if (static_cast<int>(predictions->data[i]) ==
          static_cast<int>(labels->data[i])) {
        correctPredictions++;
      }
    }
    double accuracy =
        (static_cast<double>(correctPredictions) / batchSize) * 100.0;

    delete predictions;
  });

  delete batchFeatures;
  delete mnistData;
  delete labels;
  delete features;
  delete weightsOutput;
  delete weightsHidden;
  delete biasesHidden;
  delete biasesOutput;
}

void nvbench_gpu_sparse_nn_spmm(nvbench::state &state) {
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features =
      new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  for (int i = 0; i < mnistData->m; i++) {
    labels->data[i] = mnistData->data[i * mnistData->n + 0];
    for (int j = 1; j < mnistData->n; j++) {
      features->data[i * features->n + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0;
    }
  }

  int sparsity = static_cast<int>(state.get_int64("sparsity"));
  std::string w1_file = "./data/model/" + std::to_string(sparsity) + "_W1.csv";
  std::string w2_file = "./data/model/" + std::to_string(sparsity) + "_W2.csv";

  auto *weightsHiddenCSR = swiftware::hpp::loadPrunedWeightsCSR(w1_file);
  auto *weightsOutputCSR = swiftware::hpp::loadPrunedWeightsCSR(w2_file);
  auto *biasesHidden =
      swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput =
      swiftware::hpp::readCSV("./data/model/biases_output.csv");

  swiftware::hpp::ScheduleParams scheduleParams(32, 32);

  int batchSize = static_cast<int>(state.get_int64("batch_size"));
  auto *batchFeatures = new swiftware::hpp::DenseMatrix(batchSize, features->n);
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < features->n; j++) {
      batchFeatures->data[i * features->n + j] =
          features->data[i * features->n + j];
    }
  }

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) {
    auto *predictions = swiftware::hpp::gpu_sparseNNSpmm(
        batchFeatures, weightsHiddenCSR, weightsOutputCSR, biasesHidden,
        biasesOutput, scheduleParams);

    int correctPredictions = 0;
    for (int i = 0; i < batchSize; i++) {
      if (static_cast<int>(predictions->data[i]) ==
          static_cast<int>(labels->data[i])) {
        correctPredictions++;
      }
    }
    double accuracy =
        (static_cast<double>(correctPredictions) / batchSize) * 100.0;

    delete predictions;
  });

  delete batchFeatures;
  delete mnistData;
  delete labels;
  delete features;
  delete weightsHiddenCSR;
  delete weightsOutputCSR;
  delete biasesHidden;
  delete biasesOutput;
}

void nvbench_gpu_sparse_nn_spmv(nvbench::state &state) {
  auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
  auto *labels = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
  auto *features =
      new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

  for (int i = 0; i < mnistData->m; i++) {
    labels->data[i] = mnistData->data[i * mnistData->n + 0];
    for (int j = 1; j < mnistData->n; j++) {
      features->data[i * features->n + (j - 1)] =
          mnistData->data[i * mnistData->n + j] / 255.0;
    }
  }

  int sparsity = 50;
  std::string w1_file = "./data/model/" + std::to_string(sparsity) + "_W1.csv";
  std::string w2_file = "./data/model/" + std::to_string(sparsity) + "_W2.csv";

  auto *weightsHiddenCSR = swiftware::hpp::loadPrunedWeightsCSR(w1_file);
  auto *weightsOutputCSR = swiftware::hpp::loadPrunedWeightsCSR(w2_file);
  auto *biasesHidden =
      swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
  auto *biasesOutput =
      swiftware::hpp::readCSV("./data/model/biases_output.csv");

  swiftware::hpp::ScheduleParams scheduleParams(32, 32);

  int batchSize = 10;
  auto *batchFeatures = new swiftware::hpp::DenseMatrix(batchSize, features->n);
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < features->n; j++) {
      batchFeatures->data[i * features->n + j] =
          features->data[i * features->n + j];
    }
  }

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &) {
    auto *predictions = swiftware::hpp::gpu_sparseNNSpmv(
        batchFeatures, weightsHiddenCSR, weightsOutputCSR, biasesHidden,
        biasesOutput, scheduleParams);

    int correctPredictions = 0;
    for (int i = 0; i < batchSize; i++) {
      if (static_cast<int>(predictions->data[i]) ==
          static_cast<int>(labels->data[i])) {
        correctPredictions++;
      }
    }
    double accuracy =
        (static_cast<double>(correctPredictions) / batchSize) * 100.0;

    delete predictions;
  });

  delete batchFeatures;
  delete mnistData;
  delete labels;
  delete features;
  delete weightsHiddenCSR;
  delete weightsOutputCSR;
  delete biasesHidden;
  delete biasesOutput;
}

NVBENCH_BENCH(nvbench_gpu_dense_nn_gemm)
    .set_name("GPU_Dense_NN_GEMM")
    .add_int64_axis("batch_size", {1, 8});

NVBENCH_BENCH(nvbench_gpu_dense_nn_gemv).set_name("GPU_Dense_NN_GEMV");

NVBENCH_BENCH(nvbench_gpu_sparse_nn_spmm)
    .set_name("GPU_Sparse_NN_SpMM")
    .add_int64_axis("sparsity", {50, 60, 70, 80, 90, 95})
    .add_int64_axis("batch_size", {1, 8});

NVBENCH_BENCH(nvbench_gpu_sparse_nn_spmv).set_name("GPU_Sparse_NN_SpMV");
