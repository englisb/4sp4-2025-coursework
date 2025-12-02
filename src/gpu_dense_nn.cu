// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gpu_dense_nn.cuh"
#include "kernels.cuh"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <algorithm>

#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)

namespace swiftware::hpp {

DenseMatrix *gpu_dense_nn_gemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, 
                                DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
    int batchSize = InData->m;
    int inputSize = InData->n;
    int hiddenSize = W1->m;
    int outputSize = W2->m;

    // Allocate device memory
    float *d_input, *d_W1, *d_W2, *d_B1, *d_B2, *d_H, *d_Z;
    
    CUDA_CHECK(cudaMalloc(&d_input, batchSize * inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, hiddenSize * inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, outputSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B1, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B2, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, batchSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z, batchSize * outputSize * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, InData->data.data(), batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, W1->data.data(), hiddenSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2->data.data(), outputSize * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, B1->data.data(), hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2, B2->data.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize H and Z to zero
    CUDA_CHECK(cudaMemset(d_H, 0, batchSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_Z, 0, batchSize * outputSize * sizeof(float)));

    // Layer 1: H = tanh(X * W1^T + b1)
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid1((hiddenSize + TILE_SIZE - 1) / TILE_SIZE, (batchSize + TILE_SIZE - 1) / TILE_SIZE);
    MM<<<grid1, block>>>(d_input, d_W1, d_H, batchSize, hiddenSize, inputSize);
    CUDA_CHECK(cudaGetLastError());

    // Add bias
    int threads = 256;
    int blocks = (batchSize * hiddenSize + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(d_H, d_B1, batchSize, hiddenSize);
    CUDA_CHECK(cudaGetLastError());

    // Apply tanh
    apply_tanh_kernel<<<blocks, threads>>>(d_H, batchSize * hiddenSize);
    CUDA_CHECK(cudaGetLastError());

    // Layer 2: Z = sigmoid(H * W2^T + b2)
    dim3 grid2((outputSize + TILE_SIZE - 1) / TILE_SIZE, (batchSize + TILE_SIZE - 1) / TILE_SIZE);
    MM<<<grid2, block>>>(d_H, d_W2, d_Z, batchSize, outputSize, hiddenSize);
    CUDA_CHECK(cudaGetLastError());

    // Add bias
    blocks = (batchSize * outputSize + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(d_Z, d_B2, batchSize, outputSize);
    CUDA_CHECK(cudaGetLastError());

    // Apply sigmoid
    apply_sigmoid_kernel<<<blocks, threads>>>(d_Z, batchSize * outputSize);
    CUDA_CHECK(cudaGetLastError());

    // Copy Z back to host for argmax
    DenseMatrix *Z_host = new DenseMatrix(batchSize, outputSize);
    CUDA_CHECK(cudaMemcpy(Z_host->data.data(), d_Z, batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Argmax on device
    DenseMatrix *pred = new DenseMatrix(batchSize, 1);
    float *d_pred;
    CUDA_CHECK(cudaMalloc(&d_pred, batchSize * sizeof(float)));
    
    blocks = (batchSize + threads - 1) / threads;
    argmax_rows_kernel<<<blocks, threads>>>(d_Z, d_pred, batchSize, outputSize);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(pred->data.data(), d_pred, batchSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_B2));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_Z));
    CUDA_CHECK(cudaFree(d_pred));
    
    delete Z_host;
    
    return pred;
}

DenseMatrix *gpu_dense_nn_gemv(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, 
                                DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
    int batchSize = InData->m;
    int inputSize = InData->n;
    int hiddenSize = W1->m;
    int outputSize = W2->m;

    DenseMatrix *pred = new DenseMatrix(batchSize, 1);

    // Allocate device memory for weights and biases (reused for all samples)
    float *d_W1, *d_W2, *d_B1, *d_B2, *d_input, *d_H, *d_Z;
    
    CUDA_CHECK(cudaMalloc(&d_W1, hiddenSize * inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, outputSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B1, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B2, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_H, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z, outputSize * sizeof(float)));

    // Copy weights and biases to device
    CUDA_CHECK(cudaMemcpy(d_W1, W1->data.data(), hiddenSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2->data.data(), outputSize * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B1, B1->data.data(), hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B2, B2->data.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;

    // Process each sample
    for (int i = 0; i < batchSize; i++) {
        // Copy input sample to device
        CUDA_CHECK(cudaMemcpy(d_input, &InData->data[i * inputSize], inputSize * sizeof(float), cudaMemcpyHostToDevice));

        // Initialize H and Z
        CUDA_CHECK(cudaMemset(d_H, 0, hiddenSize * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_Z, 0, outputSize * sizeof(float)));

        // Layer 1: H = tanh(W1 * x + b1)
        int blocks = (hiddenSize + threads - 1) / threads;
        MV<<<blocks, threads>>>(d_W1, d_input, d_H, hiddenSize, inputSize);
        CUDA_CHECK(cudaGetLastError());

        // Add bias
        add_bias_kernel<<<blocks, threads>>>(d_H, d_B1, 1, hiddenSize);
        CUDA_CHECK(cudaGetLastError());

        // Apply tanh
        apply_tanh_kernel<<<blocks, threads>>>(d_H, hiddenSize);
        CUDA_CHECK(cudaGetLastError());

        // Layer 2: Z = sigmoid(W2 * H + b2)
        blocks = (outputSize + threads - 1) / threads;
        MV<<<blocks, threads>>>(d_W2, d_H, d_Z, outputSize, hiddenSize);
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
        CUDA_CHECK(cudaMemcpy(&pred->data[i], d_pred, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_pred));
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_B1));
    CUDA_CHECK(cudaFree(d_B2));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_H));
    CUDA_CHECK(cudaFree(d_Z));

    return pred;
}

} // namespace swiftware