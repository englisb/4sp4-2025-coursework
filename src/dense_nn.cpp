// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "dense_nn.h"
#include "gemm.h"
#include "gemv.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace swiftware::hpp {

// Helper function to apply tanh activation
void apply_tanh(float *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = std::tanh(data[i]);
  }
}

// Helper function to apply sigmoid activation
void apply_sigmoid(float *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1.0f / (1.0f + std::exp(-data[i]));
  }
}

// Helper function to add bias vector to matrix (broadcasting)
void add_bias(float *matrix, const float *bias, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i * cols + j] += bias[j];
    }
  }
}

// Helper function to find argmax for each row
void argmax_rows(const float *matrix, float *indices, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    int max_idx = 0;
    float max_val = matrix[i * cols];
    for (int j = 1; j < cols; j++) {
      if (matrix[i * cols + j] > max_val) {
        max_val = matrix[i * cols + j];
        max_idx = j;
      }
    }
    indices[i] = static_cast<float>(max_idx);
  }
}

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

  // Initialize H to zeros (for accumulation in GEMM)
  std::fill(H->data.begin(), H->data.end(), 0.0f);
  std::fill(Z->data.begin(), Z->data.end(), 0.0f);

  auto start_total = std::chrono::high_resolution_clock::now();

  // Layer 1: H = tanh(X * W1^T + b1)
  // X is (batchSize, inputSize), W1 is (hiddenSize, inputSize)
  // We need X * W1^T, which is (batchSize, inputSize) * (inputSize, hiddenSize)
  // Since W1 is stored as (hiddenSize, inputSize), we compute W1 * X^T and
  // transpose Or we can compute directly: for each row of X, multiply by each
  // row of W1

  // Actually, let's compute it as: H = X * W1^T
  // We can do this by: for each sample in X, compute dot product with each row
  // of W1 Or use GEMM: H = X * W1^T But GEMM computes C = A * B, so we need A=X
  // (batchSize, inputSize) and B=W1^T (inputSize, hiddenSize) Since W1 is
  // (hiddenSize, inputSize), W1^T is (inputSize, hiddenSize) So we can use:
  // gemm(batchSize, hiddenSize, inputSize, X, W1, H) But this would compute X *
  // W1, not X * W1^T

  // Let me think differently: we need to transpose W1 first or use a different
  // approach Create transposed W1: W1^T is (inputSize, hiddenSize)
  DenseMatrix *W1T = new DenseMatrix(inputSize, hiddenSize);
  for (int i = 0; i < hiddenSize; i++) {
    for (int j = 0; j < inputSize; j++) {
      W1T->data[j * hiddenSize + i] = W1->data[i * inputSize + j];
    }
  }

  auto start_gemm1 = std::chrono::high_resolution_clock::now();
  // H = X * W1^T (batchSize, inputSize) * (inputSize, hiddenSize) = (batchSize,
  // hiddenSize)
  gemm(batchSize, hiddenSize, inputSize, InData->data.data(), W1T->data.data(),
       H->data.data(), Sp);
  auto end_gemm1 = std::chrono::high_resolution_clock::now();
  auto gemm1_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_gemm1 - start_gemm1)
                        .count();

  // Add bias b1
  add_bias(H->data.data(), B1->data.data(), batchSize, hiddenSize);

  // Apply tanh activation
  auto start_tanh = std::chrono::high_resolution_clock::now();
  apply_tanh(H->data.data(), batchSize * hiddenSize);
  auto end_tanh = std::chrono::high_resolution_clock::now();
  auto tanh_time = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_tanh - start_tanh)
                       .count();

  // Layer 2: Z = sigmoid(H * W2^T + b2)
  // H is (batchSize, hiddenSize), W2 is (outputSize, hiddenSize)
  // Create transposed W2: W2^T is (hiddenSize, outputSize)
  DenseMatrix *W2T = new DenseMatrix(hiddenSize, outputSize);
  for (int i = 0; i < outputSize; i++) {
    for (int j = 0; j < hiddenSize; j++) {
      W2T->data[j * outputSize + i] = W2->data[i * hiddenSize + j];
    }
  }

  auto start_gemm2 = std::chrono::high_resolution_clock::now();
  // Z = H * W2^T (batchSize, hiddenSize) * (hiddenSize, outputSize) =
  // (batchSize, outputSize)
  gemm(batchSize, outputSize, hiddenSize, H->data.data(), W2T->data.data(),
       Z->data.data(), Sp);
  auto end_gemm2 = std::chrono::high_resolution_clock::now();
  auto gemm2_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_gemm2 - start_gemm2)
                        .count();

  // Add bias b2
  add_bias(Z->data.data(), B2->data.data(), batchSize, outputSize);

  // Apply sigmoid activation
  auto start_sigmoid = std::chrono::high_resolution_clock::now();
  apply_sigmoid(Z->data.data(), batchSize * outputSize);
  auto end_sigmoid = std::chrono::high_resolution_clock::now();
  auto sigmoid_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_sigmoid - start_sigmoid)
                          .count();

  // Find argmax for each sample
  auto start_argmax = std::chrono::high_resolution_clock::now();
  argmax_rows(Z->data.data(), pred->data.data(), batchSize, outputSize);
  auto end_argmax = std::chrono::high_resolution_clock::now();
  auto argmax_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_argmax - start_argmax)
                         .count();

  auto end_total = std::chrono::high_resolution_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_total - start_total)
                        .count();

  // Print timing information
  std::cout << "GEMM Dense NN Timing (microseconds):" << std::endl;
  std::cout << "  GEMM Layer 1: " << gemm1_time << " us" << std::endl;
  std::cout << "  Tanh:         " << tanh_time << " us" << std::endl;
  std::cout << "  GEMM Layer 2: " << gemm2_time << " us" << std::endl;
  std::cout << "  Sigmoid:      " << sigmoid_time << " us" << std::endl;
  std::cout << "  Argmax:       " << argmax_time << " us" << std::endl;
  std::cout << "  Total:        " << total_time << " us" << std::endl;

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

  auto start_total = std::chrono::high_resolution_clock::now();
  long long total_gemv1_time = 0;
  long long total_tanh_time = 0;
  long long total_gemv2_time = 0;
  long long total_sigmoid_time = 0;
  long long total_argmax_time = 0;

  // Process each sample one at a time
  for (int i = 0; i < batchSize; i++) {
    const float *x = &InData->data[i * inputSize];
    float *h = &H->data[i * hiddenSize];
    float *z = &Z->data[i * outputSize];

    // Layer 1: h = tanh(W1 * x + b1)
    // W1 is (hiddenSize, inputSize), x is (inputSize,), h is (hiddenSize,)
    auto start_gemv1 = std::chrono::high_resolution_clock::now();
    gemv(hiddenSize, inputSize, W1->data.data(), x, h, Sp);
    auto end_gemv1 = std::chrono::high_resolution_clock::now();
    total_gemv1_time += std::chrono::duration_cast<std::chrono::microseconds>(
                            end_gemv1 - start_gemv1)
                            .count();

    // Add bias
    for (int j = 0; j < hiddenSize; j++) {
      h[j] += B1->data[j];
    }

    // Apply tanh
    auto start_tanh = std::chrono::high_resolution_clock::now();
    apply_tanh(h, hiddenSize);
    auto end_tanh = std::chrono::high_resolution_clock::now();
    total_tanh_time += std::chrono::duration_cast<std::chrono::microseconds>(
                           end_tanh - start_tanh)
                           .count();

    // Layer 2: z = sigmoid(W2 * h + b2)
    // W2 is (outputSize, hiddenSize), h is (hiddenSize,), z is (outputSize,)
    auto start_gemv2 = std::chrono::high_resolution_clock::now();
    gemv(outputSize, hiddenSize, W2->data.data(), h, z, Sp);
    auto end_gemv2 = std::chrono::high_resolution_clock::now();
    total_gemv2_time += std::chrono::duration_cast<std::chrono::microseconds>(
                            end_gemv2 - start_gemv2)
                            .count();

    // Add bias
    for (int j = 0; j < outputSize; j++) {
      z[j] += B2->data[j];
    }

    // Apply sigmoid
    auto start_sigmoid = std::chrono::high_resolution_clock::now();
    apply_sigmoid(z, outputSize);
    auto end_sigmoid = std::chrono::high_resolution_clock::now();
    total_sigmoid_time += std::chrono::duration_cast<std::chrono::microseconds>(
                              end_sigmoid - start_sigmoid)
                              .count();
  }

  // Find argmax for all samples
  auto start_argmax = std::chrono::high_resolution_clock::now();
  argmax_rows(Z->data.data(), pred->data.data(), batchSize, outputSize);
  auto end_argmax = std::chrono::high_resolution_clock::now();
  total_argmax_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_argmax - start_argmax)
                          .count();

  auto end_total = std::chrono::high_resolution_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_total - start_total)
                        .count();

  // Print timing information
  std::cout << "GEMV Dense NN Timing (microseconds):" << std::endl;
  std::cout << "  GEMV Layer 1 (total): " << total_gemv1_time << " us"
            << std::endl;
  std::cout << "  Tanh (total):         " << total_tanh_time << " us"
            << std::endl;
  std::cout << "  GEMV Layer 2 (total): " << total_gemv2_time << " us"
            << std::endl;
  std::cout << "  Sigmoid (total):      " << total_sigmoid_time << " us"
            << std::endl;
  std::cout << "  Argmax:               " << total_argmax_time << " us"
            << std::endl;
  std::cout << "  Total:                " << total_time << " us" << std::endl;

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

  auto start_total = std::chrono::high_resolution_clock::now();

  // Create transposed W1
  DenseMatrix *W1T = new DenseMatrix(inputSize, hiddenSize);
  for (int i = 0; i < hiddenSize; i++) {
    for (int j = 0; j < inputSize; j++) {
      W1T->data[j * hiddenSize + i] = W1->data[i * inputSize + j];
    }
  }

  auto start_gemm1 = std::chrono::high_resolution_clock::now();
  gemmMKL(batchSize, hiddenSize, inputSize, InData->data.data(),
          W1T->data.data(), H->data.data(), Sp);
  auto end_gemm1 = std::chrono::high_resolution_clock::now();
  auto gemm1_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_gemm1 - start_gemm1)
                        .count();

  add_bias(H->data.data(), B1->data.data(), batchSize, hiddenSize);

  auto start_tanh = std::chrono::high_resolution_clock::now();
  apply_tanh(H->data.data(), batchSize * hiddenSize);
  auto end_tanh = std::chrono::high_resolution_clock::now();
  auto tanh_time = std::chrono::duration_cast<std::chrono::microseconds>(
                       end_tanh - start_tanh)
                       .count();

  // Create transposed W2
  DenseMatrix *W2T = new DenseMatrix(hiddenSize, outputSize);
  for (int i = 0; i < outputSize; i++) {
    for (int j = 0; j < hiddenSize; j++) {
      W2T->data[j * outputSize + i] = W2->data[i * hiddenSize + j];
    }
  }

  auto start_gemm2 = std::chrono::high_resolution_clock::now();
  gemmMKL(batchSize, outputSize, hiddenSize, H->data.data(), W2T->data.data(),
          Z->data.data(), Sp);
  auto end_gemm2 = std::chrono::high_resolution_clock::now();
  auto gemm2_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_gemm2 - start_gemm2)
                        .count();

  add_bias(Z->data.data(), B2->data.data(), batchSize, outputSize);

  auto start_sigmoid = std::chrono::high_resolution_clock::now();
  apply_sigmoid(Z->data.data(), batchSize * outputSize);
  auto end_sigmoid = std::chrono::high_resolution_clock::now();
  auto sigmoid_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_sigmoid - start_sigmoid)
                          .count();

  auto start_argmax = std::chrono::high_resolution_clock::now();
  argmax_rows(Z->data.data(), pred->data.data(), batchSize, outputSize);
  auto end_argmax = std::chrono::high_resolution_clock::now();
  auto argmax_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_argmax - start_argmax)
                         .count();

  auto end_total = std::chrono::high_resolution_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_total - start_total)
                        .count();

  std::cout << "MKL GEMM Dense NN Timing (microseconds):" << std::endl;
  std::cout << "  GEMM Layer 1: " << gemm1_time << " us" << std::endl;
  std::cout << "  Tanh:         " << tanh_time << " us" << std::endl;
  std::cout << "  GEMM Layer 2: " << gemm2_time << " us" << std::endl;
  std::cout << "  Sigmoid:      " << sigmoid_time << " us" << std::endl;
  std::cout << "  Argmax:       " << argmax_time << " us" << std::endl;
  std::cout << "  Total:        " << total_time << " us" << std::endl;

  delete H;
  delete Z;
  delete W1T;
  delete W2T;

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
  DenseMatrix *H = new DenseMatrix(batchSize, hiddenSize);
  DenseMatrix *Z = new DenseMatrix(batchSize, outputSize);

  std::fill(H->data.begin(), H->data.end(), 0.0f);
  std::fill(Z->data.begin(), Z->data.end(), 0.0f);

  auto start_total = std::chrono::high_resolution_clock::now();
  long long total_gemv1_time = 0;
  long long total_tanh_time = 0;
  long long total_gemv2_time = 0;
  long long total_sigmoid_time = 0;
  long long total_argmax_time = 0;

  for (int i = 0; i < batchSize; i++) {
    const float *x = &InData->data[i * inputSize];
    float *h = &H->data[i * hiddenSize];
    float *z = &Z->data[i * outputSize];

    auto start_gemv1 = std::chrono::high_resolution_clock::now();
    gemvMKL(hiddenSize, inputSize, W1->data.data(), x, h, Sp);
    auto end_gemv1 = std::chrono::high_resolution_clock::now();
    total_gemv1_time += std::chrono::duration_cast<std::chrono::microseconds>(
                            end_gemv1 - start_gemv1)
                            .count();

    for (int j = 0; j < hiddenSize; j++) {
      h[j] += B1->data[j];
    }

    auto start_tanh = std::chrono::high_resolution_clock::now();
    apply_tanh(h, hiddenSize);
    auto end_tanh = std::chrono::high_resolution_clock::now();
    total_tanh_time += std::chrono::duration_cast<std::chrono::microseconds>(
                           end_tanh - start_tanh)
                           .count();

    auto start_gemv2 = std::chrono::high_resolution_clock::now();
    gemvMKL(outputSize, hiddenSize, W2->data.data(), h, z, Sp);
    auto end_gemv2 = std::chrono::high_resolution_clock::now();
    total_gemv2_time += std::chrono::duration_cast<std::chrono::microseconds>(
                            end_gemv2 - start_gemv2)
                            .count();

    for (int j = 0; j < outputSize; j++) {
      z[j] += B2->data[j];
    }

    auto start_sigmoid = std::chrono::high_resolution_clock::now();
    apply_sigmoid(z, outputSize);
    auto end_sigmoid = std::chrono::high_resolution_clock::now();
    total_sigmoid_time += std::chrono::duration_cast<std::chrono::microseconds>(
                              end_sigmoid - start_sigmoid)
                              .count();
  }

  auto start_argmax = std::chrono::high_resolution_clock::now();
  argmax_rows(Z->data.data(), pred->data.data(), batchSize, outputSize);
  auto end_argmax = std::chrono::high_resolution_clock::now();
  total_argmax_time = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_argmax - start_argmax)
                          .count();

  auto end_total = std::chrono::high_resolution_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_total - start_total)
                        .count();

  std::cout << "MKL GEMV Dense NN Timing (microseconds):" << std::endl;
  std::cout << "  GEMV Layer 1 (total): " << total_gemv1_time << " us"
            << std::endl;
  std::cout << "  Tanh (total):         " << total_tanh_time << " us"
            << std::endl;
  std::cout << "  GEMV Layer 2 (total): " << total_gemv2_time << " us"
            << std::endl;
  std::cout << "  Sigmoid (total):      " << total_sigmoid_time << " us"
            << std::endl;
  std::cout << "  Argmax:               " << total_argmax_time << " us"
            << std::endl;
  std::cout << "  Total:                " << total_time << " us" << std::endl;

  delete H;
  delete Z;

  return pred;
}
#endif

} // namespace swiftware::hpp