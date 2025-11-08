// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "gtest/gtest.h"

#include "kernel.cuh"

#include <vector>

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>


TEST(MulAddKernelV1Test, SmallFloatArray) {
const size_t n = 1024;
std::vector<float> h_a(n), h_b(n), h_out(n, 1.0f), h_res(n);
std::iota(h_a.begin(), h_a.end(), 0.0f);
std::iota(h_b.begin(), h_b.end(), 1.0f);

float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
cudaError_t err = cudaMalloc(&d_a, n * sizeof(float));
ASSERT_EQ(err, cudaSuccess);
err = cudaMalloc(&d_b, n * sizeof(float));
ASSERT_EQ(err, cudaSuccess);
err = cudaMalloc(&d_out, n * sizeof(float));
ASSERT_EQ(err, cudaSuccess);

err = cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
ASSERT_EQ(err, cudaSuccess);
err = cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
ASSERT_EQ(err, cudaSuccess);
err = cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);
ASSERT_EQ(err, cudaSuccess);

const int block = 256;
const int grid = static_cast<int>((n + block - 1) / block);
swiftware::hpp::mulAddKernel_v1<float><<<grid, block>>>(d_a, d_b, d_out, n);
err = cudaDeviceSynchronize();
ASSERT_EQ(err, cudaSuccess);

err = cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
ASSERT_EQ(err, cudaSuccess);

for (size_t i = 0; i < n; ++i) {
float expected = h_a[i] * h_b[i] + h_out[i];
ASSERT_FLOAT_EQ(h_res[i], expected);
}

cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_out);
}

// TODO: add more tests and apply necessary changes

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}