// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "sparse_nn.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

namespace swiftware::hpp {
// TODO: Add more test

// Test basic sparse neural network with SPMM
TEST(SparseNNTest, SpmmBasicTest) {
  int batchSize = 2;
  int inputSize = 4;
  int hiddenSize = 3;
  int outputSize = 2;

  // Create input data (2 samples, 4 features each)
  // Input = [[1, 2, 3, 4],
  //          [5, 6, 7, 8]]
  auto *InData = new DenseMatrix(batchSize, inputSize);
  InData->data[0 * inputSize + 0] = 1.0f;
  InData->data[0 * inputSize + 1] = 2.0f;
  InData->data[0 * inputSize + 2] = 3.0f;
  InData->data[0 * inputSize + 3] = 4.0f;
  InData->data[1 * inputSize + 0] = 5.0f;
  InData->data[1 * inputSize + 1] = 6.0f;
  InData->data[1 * inputSize + 2] = 7.0f;
  InData->data[1 * inputSize + 3] = 8.0f;

  // Create sparse weight matrix W1 (hiddenSize x inputSize) = identity-like
  // pattern W1 = [[1, 0, 0, 0],
  //       [0, 1, 0, 0],
  //       [0, 0, 1, 0]]
  auto *W1_dense = new DenseMatrix(hiddenSize, inputSize);
  std::fill(W1_dense->data.begin(), W1_dense->data.end(), 0.0f);
  W1_dense->data[0 * inputSize + 0] = 1.0f;
  W1_dense->data[1 * inputSize + 1] = 1.0f;
  W1_dense->data[2 * inputSize + 2] = 1.0f;
  auto *W1 = denseToCSR(W1_dense);

  // Create sparse weight matrix W2 (outputSize x hiddenSize) = identity-like
  // pattern W2 = [[1, 0, 0],
  //       [0, 1, 0]]
  auto *W2_dense = new DenseMatrix(outputSize, hiddenSize);
  std::fill(W2_dense->data.begin(), W2_dense->data.end(), 0.0f);
  W2_dense->data[0 * hiddenSize + 0] = 1.0f;
  W2_dense->data[1 * hiddenSize + 1] = 1.0f;
  auto *W2 = denseToCSR(W2_dense);

  // Create bias vectors (all zeros for simplicity)
  auto *B1 = new DenseMatrix(1, hiddenSize);
  std::fill(B1->data.begin(), B1->data.end(), 0.0f);
  auto *B2 = new DenseMatrix(1, outputSize);
  std::fill(B2->data.begin(), B2->data.end(), 0.0f);

  ScheduleParams sp(2, 2);
  auto *predictions = sparseNNSpmm(InData, W1, W2, B1, B2, sp);

  // With identity-like weights and zero biases, the network should pass through
  // the first two features. After tanh and sigmoid, we expect reasonable
  // values. The exact values depend on the activations, but predictions should
  // be valid indices
  EXPECT_GE(predictions->data[0], 0.0f);
  EXPECT_LT(predictions->data[0], outputSize);
  EXPECT_GE(predictions->data[1], 0.0f);
  EXPECT_LT(predictions->data[1], outputSize);

  delete InData;
  delete W1_dense;
  delete W1;
  delete W2_dense;
  delete W2;
  delete B1;
  delete B2;
  delete predictions;
}

} // namespace swiftware::hpp