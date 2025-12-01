// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "dense_nn.h"
#include "gemm.h"
#include "gemv.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace swiftware::hpp {

// Implement Dense NN with GEMM
DenseMatrix *dense_nn_gemm(DenseMatrix *InData, DenseMatrix *W1,
                           DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2,
                           ScheduleParams Sp) {
  int batchSize = InData->m;
  int inputSize = InData->n;
  int hiddenSize = W1->m;
  int outputSize = W2->m;

  // Allocate intermediate matrices
  DenseMatrix *H = new DenseMatrix(batchSize, hiddenSize);
  DenseMatrix *Z = new DenseMatrix(batchSize, outputSize);
  DenseMatrix *pred = new DenseMatrix(batchSize, 1);

  // Initialize H and Z to zeros (for accumulation in GEMM)
  std::fill(H->data.begin(), H->data.end(), 0.0f);
  std::fill(Z->data.begin(), Z->data.end(), 0.0f);

  // Layer 1: H = tanh(X * W1^T + b1)
  // X is (batchSize, inputSize), W1 is (hiddenSize, inputSize)
  // We need X * W1^T, which is (batchSize, inputSize) * (inputSize, hiddenSize)
  // Create transposed W1: W1^T is (inputSize, hiddenSize)
  DenseMatrix *W1T = new DenseMatrix(inputSize, hiddenSize);
  for (int i = 0; i < hiddenSize; i++) {
    for (int j = 0; j < inputSize; j++) {
      W1T->data[j * hiddenSize + i] = W1->data[i * inputSize + j];
    }
  }

  // H = X * W1^T (batchSize, inputSize) * (inputSize, hiddenSize) = (batchSize,
  // hiddenSize) GEMM: C = A * B + C, where A is m x k, B is k x n So:
  // gemm(batchSize, hiddenSize, inputSize, X, W1T, H)
  gemm(batchSize, hiddenSize, inputSize, InData->data.data(), W1T->data.data(),
       H->data.data(), Sp);

  // Add bias b1
  add_bias(H->data.data(), B1->data.data(), batchSize, hiddenSize);

  // Apply tanh activation
  apply_tanh(H->data.data(), batchSize * hiddenSize);

  // Layer 2: Z = sigmoid(H * W2^T + b2)
  // H is (batchSize, hiddenSize), W2 is (outputSize, hiddenSize)
  // Create transposed W2: W2^T is (hiddenSize, outputSize)
  DenseMatrix *W2T = new DenseMatrix(hiddenSize, outputSize);
  for (int i = 0; i < outputSize; i++) {
    for (int j = 0; j < hiddenSize; j++) {
      W2T->data[j * outputSize + i] = W2->data[i * hiddenSize + j];
    }
  }

  // Z = H * W2^T (batchSize, hiddenSize) * (hiddenSize, outputSize) =
  // (batchSize, outputSize)
  gemm(batchSize, outputSize, hiddenSize, H->data.data(), W2T->data.data(),
       Z->data.data(), Sp);

  // Add bias b2
  add_bias(Z->data.data(), B2->data.data(), batchSize, outputSize);

  // Apply sigmoid activation
  apply_sigmoid(Z->data.data(), batchSize * outputSize);

  // Find argmax for each sample
  argmax_rows(Z->data.data(), pred->data.data(), batchSize, outputSize);

  delete H;
  delete Z;
  delete W1T;
  delete W2T;

  return pred;
}

// Implement Dense NN with GEMV (process one sample at a time)
DenseMatrix *dense_nn_gemv(DenseMatrix *InData, DenseMatrix *W1,
                           DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2,
                           ScheduleParams Sp) {
  int batchSize = InData->m;
  int inputSize = InData->n;
  int hiddenSize = W1->m;
  int outputSize = W2->m;

  DenseMatrix *pred = new DenseMatrix(batchSize, 1);
  DenseMatrix *H = new DenseMatrix(batchSize, hiddenSize);
  DenseMatrix *Z = new DenseMatrix(batchSize, outputSize);

  // Initialize to zeros
  std::fill(H->data.begin(), H->data.end(), 0.0f);
  std::fill(Z->data.begin(), Z->data.end(), 0.0f);

  // Process each sample one at a time
  for (int i = 0; i < batchSize; i++) {
    const float *x = &InData->data[i * inputSize];
    float *h = &H->data[i * hiddenSize];
    float *z = &Z->data[i * outputSize];

    // Layer 1: h = W1 * x + b1
    // W1 is (hiddenSize, inputSize), x is (inputSize,), h is (hiddenSize,)
    // GEMV: y = A * x + y, where A is m x n
    // So: gemv(hiddenSize, inputSize, W1, x, h)
    gemv(hiddenSize, inputSize, W1->data.data(), x, h, Sp);

    // Add bias
    for (int j = 0; j < hiddenSize; j++) {
      h[j] += B1->data[j];
    }

    // Apply tanh
    apply_tanh(h, hiddenSize);

    // Layer 2: z = W2 * h + b2
    // W2 is (outputSize, hiddenSize), h is (hiddenSize,), z is (outputSize,)
    gemv(outputSize, hiddenSize, W2->data.data(), h, z, Sp);

    // Add bias
    for (int j = 0; j < outputSize; j++) {
      z[j] += B2->data[j];
    }

    // Apply sigmoid
    apply_sigmoid(z, outputSize);
  }

  // Find argmax for all samples
  argmax_rows(Z->data.data(), pred->data.data(), batchSize, outputSize);

  delete H;
  delete Z;

  return pred;
}

#ifdef USE_MKL
void gemmMKL(int m, int n, int k, const float *A, const float *B, float *C,
             ScheduleParams Sp);
void gemvMKL(int m, int n, const float *A, const float *x, float *y,
             ScheduleParams Sp);

DenseMatrix *dense_nn_mkl_gemm(DenseMatrix *InData, DenseMatrix *W1,
                               DenseMatrix *W2, DenseMatrix *B1,
                               DenseMatrix *B2, ScheduleParams Sp) {
  int batchSize = InData->m;
  DenseMatrix *pred = new DenseMatrix(batchSize, 1);

  return pred;
}

DenseMatrix *dense_nn_mkl_gemv(DenseMatrix *InData, DenseMatrix *W1,
                               DenseMatrix *W2, DenseMatrix *B1,
                               DenseMatrix *B2, ScheduleParams Sp) {
  int batchSize = InData->m;
  DenseMatrix *pred = new DenseMatrix(batchSize, 1);

  return pred;
}
#endif

} // namespace swiftware::hpp