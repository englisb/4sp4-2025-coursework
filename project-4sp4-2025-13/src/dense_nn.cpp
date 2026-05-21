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

#ifdef USE_MKL
#include <mkl.h>
#endif

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
DenseMatrix *dense_nn_mkl_gemm(DenseMatrix *InData, DenseMatrix *W1,
                               DenseMatrix *W2, DenseMatrix *B1,
                               DenseMatrix *B2, ScheduleParams Sp) {
  int batchSize = InData->m;
  int inputSize = InData->n;
  int hiddenSize = W1->m;
  int outputSize = W2->m;

  // Allocate intermediate matrices
  DenseMatrix *H = new DenseMatrix(batchSize, hiddenSize);
  DenseMatrix *Z = new DenseMatrix(batchSize, outputSize);
  DenseMatrix *pred = new DenseMatrix(batchSize, 1);

  // Initialize to zeros
  std::fill(H->data.begin(), H->data.end(), 0.0f);
  std::fill(Z->data.begin(), Z->data.end(), 0.0f);

  // Layer 1: H = tanh(X * W1^T + b1)
  // MKL GEMM: C = alpha*op(A)*op(B) + beta*C
  // We want: H = InData * W1^T
  // InData is (batchSize x inputSize), W1 is (hiddenSize x inputSize)
  // W1^T is (inputSize x hiddenSize) when transposed
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              batchSize, hiddenSize, inputSize,
              1.0f,                          // alpha
              InData->data.data(), inputSize, // A: batchSize x inputSize
              W1->data.data(), inputSize,     // B: hiddenSize x inputSize (will be transposed)
              0.0f,                          // beta (overwrite H)
              H->data.data(), hiddenSize);    // C: batchSize x hiddenSize

  // Add bias and apply tanh
  add_bias(H->data.data(), B1->data.data(), batchSize, hiddenSize);
  apply_tanh(H->data.data(), batchSize * hiddenSize);

  // Layer 2: Z = sigmoid(H * W2^T + b2)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              batchSize, outputSize, hiddenSize,
              1.0f,                          // alpha
              H->data.data(), hiddenSize,     // A: batchSize x hiddenSize
              W2->data.data(), hiddenSize,    // B: outputSize x hiddenSize (will be transposed)
              0.0f,                          // beta
              Z->data.data(), outputSize);    // C: batchSize x outputSize

  // Add bias and apply sigmoid
  add_bias(Z->data.data(), B2->data.data(), batchSize, outputSize);
  apply_sigmoid(Z->data.data(), batchSize * outputSize);

  // Argmax
  argmax_rows(Z->data.data(), pred->data.data(), batchSize, outputSize);

  delete H;
  delete Z;
  return pred;
}

DenseMatrix *dense_nn_mkl_gemv(DenseMatrix *InData, DenseMatrix *W1,
                               DenseMatrix *W2, DenseMatrix *B1,
                               DenseMatrix *B2, ScheduleParams Sp) {
  int batchSize = InData->m;
  int inputSize = InData->n;
  int hiddenSize = W1->m;
  int outputSize = W2->m;

  DenseMatrix *pred = new DenseMatrix(batchSize, 1);

  // Process each sample one at a time using GEMV
  for (int i = 0; i < batchSize; i++) {
    // Allocate vectors for this sample
    std::vector<float> h(hiddenSize, 0.0f);
    std::vector<float> z(outputSize, 0.0f);

    // Layer 1: h = tanh(x * W1^T + b1)
    // x is (1 x inputSize), W1 is (hiddenSize x inputSize)
    // We want h = W1 * x^T (matrix-vector product)
    // MKL GEMV: y = alpha*A*x + beta*y
    // We need: h = W1 * x, where W1 is (hiddenSize x inputSize) and x is (inputSize x 1)
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                hiddenSize, inputSize,
                1.0f,                                      // alpha
                W1->data.data(), inputSize,                // A: hiddenSize x inputSize
                &InData->data[i * inputSize], 1,          // x: inputSize vector
                0.0f,                                      // beta
                h.data(), 1);                              // y: hiddenSize vector

    // Add bias and apply tanh
    for (int j = 0; j < hiddenSize; j++) {
      h[j] = std::tanh(h[j] + B1->data[j]);
    }

    // Layer 2: z = sigmoid(h * W2^T + b2)
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                outputSize, hiddenSize,
                1.0f,                                      // alpha
                W2->data.data(), hiddenSize,               // A: outputSize x hiddenSize
                h.data(), 1,                               // x: hiddenSize vector
                0.0f,                                      // beta
                z.data(), 1);                              // y: outputSize vector

    // Add bias and apply sigmoid
    for (int j = 0; j < outputSize; j++) {
      z[j] = 1.0f / (1.0f + std::exp(-(z[j] + B2->data[j])));
    }

    // Find argmax
    int maxIdx = 0;
    float maxVal = z[0];
    for (int j = 1; j < outputSize; j++) {
      if (z[j] > maxVal) {
        maxVal = z[j];
        maxIdx = j;
      }
    }
    pred->data[i] = static_cast<float>(maxIdx);
  }

  return pred;
}
#endif

} // namespace swiftware::hpp