// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gpu_sparse_nn.cuh"
#include "gpu_utils.h"
#include "kernels.cuh"
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)

namespace swiftware::hpp {

DenseMatrix *gpu_sparseNNSpmm(DenseMatrix *InData, CSR *W1, CSR *W2,
                              DenseMatrix *B1, DenseMatrix *B2,
                              ScheduleParams Sp) {
  int batchSize = InData->m;
  int inputSize = InData->n;
  int hiddenSize = W1->m;
  int outputSize = W2->m;

  // Allocate device memory for input and output matrices
  float *d_input, *d_B1, *d_B2, *d_H, *d_Z;
  int *d_W1_row_ptr, *d_W1_col_id, *d_W2_row_ptr, *d_W2_col_id;
  float *d_W1_val, *d_W2_val;

  CUDA_CHECK(cudaMalloc(&d_input, batchSize * inputSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B1, hiddenSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B2, outputSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_H, batchSize * hiddenSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Z, batchSize * outputSize * sizeof(float)));

  // Allocate and copy CSR format for W1
  CUDA_CHECK(cudaMalloc(&d_W1_row_ptr, (W1->m + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_W1_col_id, W1->nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_W1_val, W1->nnz * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_W1_row_ptr, W1->Ap.data(), (W1->m + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W1_col_id, W1->Ai.data(), W1->nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W1_val, W1->Ax.data(), W1->nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Allocate and copy CSR format for W2
  CUDA_CHECK(cudaMalloc(&d_W2_row_ptr, (W2->m + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_W2_col_id, W2->nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_W2_val, W2->nnz * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_W2_row_ptr, W2->Ap.data(), (W2->m + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W2_col_id, W2->Ai.data(), W2->nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W2_val, W2->Ax.data(), W2->nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Copy input data and biases to device
  CUDA_CHECK(cudaMemcpy(d_input, InData->data.data(),
                        batchSize * inputSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B1, B1->data.data(), hiddenSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B2, B2->data.data(), outputSize * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Initialize H and Z to zero
  CUDA_CHECK(cudaMemset(d_H, 0, batchSize * hiddenSize * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_Z, 0, batchSize * outputSize * sizeof(float)));

  int threads = 256;

  // Layer 1: H = tanh(X * W1^T + b1)
  int blocks = (hiddenSize + threads - 1) / threads;
  SpMM<<<blocks, threads>>>(d_W1_row_ptr, d_W1_col_id, d_W1_val, d_input, d_H,
                            hiddenSize, batchSize, inputSize);
  CUDA_CHECK(cudaGetLastError());

  // Add bias
  blocks = (batchSize * hiddenSize + threads - 1) / threads;
  add_bias_kernel<<<blocks, threads>>>(d_H, d_B1, batchSize, hiddenSize);
  CUDA_CHECK(cudaGetLastError());

  // Apply tanh
  apply_tanh_kernel<<<blocks, threads>>>(d_H, batchSize * hiddenSize);
  CUDA_CHECK(cudaGetLastError());

  // Layer 2: Z = sigmoid(H * W2^T + b2)
  blocks = (outputSize + threads - 1) / threads;
  SpMM<<<blocks, threads>>>(d_W2_row_ptr, d_W2_col_id, d_W2_val, d_H, d_Z,
                            outputSize, batchSize, hiddenSize);
  CUDA_CHECK(cudaGetLastError());

  // Add bias
  blocks = (batchSize * outputSize + threads - 1) / threads;
  add_bias_kernel<<<blocks, threads>>>(d_Z, d_B2, batchSize, outputSize);
  CUDA_CHECK(cudaGetLastError());

  // Apply sigmoid
  apply_sigmoid_kernel<<<blocks, threads>>>(d_Z, batchSize * outputSize);
  CUDA_CHECK(cudaGetLastError());

  // Argmax on device
  DenseMatrix *pred = new DenseMatrix(batchSize, 1);
  float *d_pred;
  CUDA_CHECK(cudaMalloc(&d_pred, batchSize * sizeof(float)));

  blocks = (batchSize + threads - 1) / threads;
  argmax_rows_kernel<<<blocks, threads>>>(d_Z, d_pred, batchSize, outputSize);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(pred->data.data(), d_pred, batchSize * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Free device memory
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_B1));
  CUDA_CHECK(cudaFree(d_B2));
  CUDA_CHECK(cudaFree(d_H));
  CUDA_CHECK(cudaFree(d_Z));
  CUDA_CHECK(cudaFree(d_W1_row_ptr));
  CUDA_CHECK(cudaFree(d_W1_col_id));
  CUDA_CHECK(cudaFree(d_W1_val));
  CUDA_CHECK(cudaFree(d_W2_row_ptr));
  CUDA_CHECK(cudaFree(d_W2_col_id));
  CUDA_CHECK(cudaFree(d_W2_val));
  CUDA_CHECK(cudaFree(d_pred));

  return pred;
}

DenseMatrix *gpu_sparseNNSpmv(DenseMatrix *InData, CSR *W1, CSR *W2,
                              DenseMatrix *B1, DenseMatrix *B2,
                              ScheduleParams Sp) {
  int batchSize = InData->m;
  int inputSize = InData->n;
  int hiddenSize = W1->m;
  int outputSize = W2->m;

  DenseMatrix *pred = new DenseMatrix(batchSize, 1);

  // Allocate device memory for weights and biases (reused for all samples)
  int *d_W1_row_ptr, *d_W1_col_id, *d_W2_row_ptr, *d_W2_col_id;
  float *d_W1_val, *d_W2_val, *d_B1, *d_B2;
  float *d_input, *d_H, *d_Z;

  // Allocate and copy CSR format for W1
  CUDA_CHECK(cudaMalloc(&d_W1_row_ptr, (W1->m + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_W1_col_id, W1->nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_W1_val, W1->nnz * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_W1_row_ptr, W1->Ap.data(), (W1->m + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W1_col_id, W1->Ai.data(), W1->nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W1_val, W1->Ax.data(), W1->nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Allocate and copy CSR format for W2
  CUDA_CHECK(cudaMalloc(&d_W2_row_ptr, (W2->m + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_W2_col_id, W2->nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_W2_val, W2->nnz * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_W2_row_ptr, W2->Ap.data(), (W2->m + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W2_col_id, W2->Ai.data(), W2->nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W2_val, W2->Ax.data(), W2->nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_B1, hiddenSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B2, outputSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_H, hiddenSize * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Z, outputSize * sizeof(float)));

  // Copy biases to device
  CUDA_CHECK(cudaMemcpy(d_B1, B1->data.data(), hiddenSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B2, B2->data.data(), outputSize * sizeof(float),
                        cudaMemcpyHostToDevice));

  int threads = 256;

  // Process each sample
  for (int i = 0; i < batchSize; i++) {
    // Copy input sample to device
    CUDA_CHECK(cudaMemcpy(d_input, &InData->data[i * inputSize],
                          inputSize * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize H and Z
    CUDA_CHECK(cudaMemset(d_H, 0, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_Z, 0, outputSize * sizeof(float)));

    // Layer 1: H = tanh(W1 * x + b1)
    int blocks = (hiddenSize + threads - 1) / threads;
    SpMV<<<blocks, threads>>>(d_W1_row_ptr, d_W1_col_id, d_W1_val, d_input, d_H,
                              hiddenSize, inputSize);
    CUDA_CHECK(cudaGetLastError());

    // Add bias
    add_bias_kernel<<<blocks, threads>>>(d_H, d_B1, 1, hiddenSize);
    CUDA_CHECK(cudaGetLastError());

    // Apply tanh
    apply_tanh_kernel<<<blocks, threads>>>(d_H, hiddenSize);
    CUDA_CHECK(cudaGetLastError());

    // Layer 2: Z = sigmoid(W2 * H + b2)
    blocks = (outputSize + threads - 1) / threads;
    SpMV<<<blocks, threads>>>(d_W2_row_ptr, d_W2_col_id, d_W2_val, d_H, d_Z,
                              outputSize, hiddenSize);
    CUDA_CHECK(cudaGetLastError());

    // Add bias
    add_bias_kernel<<<blocks, threads>>>(d_Z, d_B2, 1, outputSize);
    CUDA_CHECK(cudaGetLastError());

    // Apply sigmoid
    apply_sigmoid_kernel<<<blocks, threads>>>(d_Z, outputSize);
    CUDA_CHECK(cudaGetLastError());

    // Argmax on device
    float *d_pred;
    CUDA_CHECK(cudaMalloc(&d_pred, sizeof(float)));
    argmax_rows_kernel<<<1, 1>>>(d_Z, d_pred, 1, outputSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&pred->data[i], d_pred, sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_pred));
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_W1_row_ptr));
  CUDA_CHECK(cudaFree(d_W1_col_id));
  CUDA_CHECK(cudaFree(d_W1_val));
  CUDA_CHECK(cudaFree(d_W2_row_ptr));
  CUDA_CHECK(cudaFree(d_W2_col_id));
  CUDA_CHECK(cudaFree(d_W2_val));
  CUDA_CHECK(cudaFree(d_B1));
  CUDA_CHECK(cudaFree(d_B2));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_H));
  CUDA_CHECK(cudaFree(d_Z));

  return pred;
}

} // namespace swiftware::hpp