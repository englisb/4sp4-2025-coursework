// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "sparse_nn.h"
#include "spmv.h"
#include "utils.h"
#include <algorithm>
#include <cmath>

namespace swiftware::hpp {

// TODO Implement Sparse NN with SPMM
DenseMatrix *sparseNNSpmm(DenseMatrix *InData, CSR *W1, CSR *W2,
                          DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
  int batchSize = InData->m;
  int inputSize = InData->n;
  int hiddenSize = W1->m;
  int outputSize = W2->m;

  // Allocate intermediate matrices
  DenseMatrix *H = new DenseMatrix(batchSize, hiddenSize);
  DenseMatrix *Z = new DenseMatrix(batchSize, outputSize);
  DenseMatrix *pred = new DenseMatrix(batchSize, 1);

  // Initialize H and Z to zeros (for accumulation in SPMM)
  std::fill(H->data.begin(), H->data.end(), 0.0f);
  std::fill(Z->data.begin(), Z->data.end(), 0.0f);

  // Layer 1: H = tanh(X * W1^T + b1)
  // X is (batchSize, inputSize), W1 is CSR (hiddenSize, inputSize)
  // We need X * W1^T, which is (batchSize, inputSize) * (inputSize, hiddenSize)
  // = (batchSize, hiddenSize) Strategy: Transpose X to X^T (inputSize,
  // batchSize), compute W1 * X^T = (hiddenSize, batchSize), then transpose to
  // (batchSize, hiddenSize)

  // Transpose input: X^T is (inputSize, batchSize)
  DenseMatrix *XT = new DenseMatrix(inputSize, batchSize);
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < inputSize; j++) {
      XT->data[j * batchSize + i] = InData->data[i * inputSize + j];
    }
  }

  // Transpose H to get HT (hiddenSize, batchSize) for SPMM output
  DenseMatrix *HT = new DenseMatrix(hiddenSize, batchSize);
  std::fill(HT->data.begin(), HT->data.end(), 0.0f);

  // Compute HT = W1 * X^T, where W1 is (hiddenSize, inputSize) CSR, X^T is
  // (inputSize, batchSize) dense Result HT is (hiddenSize, batchSize)
  spmmCSR(hiddenSize, batchSize, inputSize, W1->Ap.data(), W1->Ai.data(),
          W1->Ax.data(), XT->data.data(), HT->data.data(), Sp);

  // Transpose HT back to H (batchSize, hiddenSize)
  for (int i = 0; i < hiddenSize; i++) {
    for (int j = 0; j < batchSize; j++) {
      H->data[j * hiddenSize + i] = HT->data[i * batchSize + j];
    }
  }

  // Add bias b1
  add_bias(H->data.data(), B1->data.data(), batchSize, hiddenSize);

  // Apply tanh activation
  apply_tanh(H->data.data(), batchSize * hiddenSize);

  // Layer 2: Z = sigmoid(H * W2^T + b2)
  // H is (batchSize, hiddenSize), W2 is CSR (outputSize, hiddenSize)
  // Transpose H to HT (hiddenSize, batchSize)
  std::fill(HT->data.begin(), HT->data.end(), 0.0f);
  for (int i = 0; i < batchSize; i++) {
    for (int j = 0; j < hiddenSize; j++) {
      HT->data[j * batchSize + i] = H->data[i * hiddenSize + j];
    }
  }

  // Transpose Z to get ZT (outputSize, batchSize) for SPMM output
  DenseMatrix *ZT = new DenseMatrix(outputSize, batchSize);
  std::fill(ZT->data.begin(), ZT->data.end(), 0.0f);

  // Compute ZT = W2 * HT, where W2 is (outputSize, hiddenSize) CSR, HT is
  // (hiddenSize, batchSize) dense Result ZT is (outputSize, batchSize)
  spmmCSR(outputSize, batchSize, hiddenSize, W2->Ap.data(), W2->Ai.data(),
          W2->Ax.data(), HT->data.data(), ZT->data.data(), Sp);

  // Transpose ZT back to Z (batchSize, outputSize)
  for (int i = 0; i < outputSize; i++) {
    for (int j = 0; j < batchSize; j++) {
      Z->data[j * outputSize + i] = ZT->data[i * batchSize + j];
    }
  }

  // Add bias b2
  add_bias(Z->data.data(), B2->data.data(), batchSize, outputSize);

  // Apply sigmoid activation
  apply_sigmoid(Z->data.data(), batchSize * outputSize);

  // Find argmax for each sample
  argmax_rows(Z->data.data(), pred->data.data(), batchSize, outputSize);

  delete H;
  delete Z;
  delete XT;
  delete HT;
  delete ZT;

  return pred;
}

// TODO: Implement Sparse NN with SPMV
DenseMatrix *sparseNNSpmv(DenseMatrix *InData, CSR *W1, CSR *W2,
                          DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
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
    // W1 is CSR (hiddenSize, inputSize), x is (inputSize,), h is (hiddenSize,)
    // SPMV: c = A * b + c, where A is m x n
    // So: spmvCSR(hiddenSize, inputSize, W1, x, h)
    spmvCSR(hiddenSize, inputSize, W1->Ap.data(), W1->Ai.data(), W1->Ax.data(),
            x, h, Sp);

    // Add bias
    for (int j = 0; j < hiddenSize; j++) {
      h[j] += B1->data[j];
    }

    // Apply tanh
    apply_tanh(h, hiddenSize);

    // Layer 2: z = W2 * h + b2
    // W2 is CSR (outputSize, hiddenSize), h is (hiddenSize,), z is
    // (outputSize,)
    spmvCSR(outputSize, hiddenSize, W2->Ap.data(), W2->Ai.data(), W2->Ax.data(),
            h, z, Sp);

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

} // namespace swiftware::hpp