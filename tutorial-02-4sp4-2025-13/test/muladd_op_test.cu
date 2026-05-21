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

// Additional tests for v1 (grid-stride loop)
TEST(MulAddKernelV1Test, LargeArrayGridStride) {
  const size_t n = 1 << 20;  // 1M elements
  std::vector<float> h_a(n), h_b(n), h_out(n, 1.0f), h_res(n);
  
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i % 100);
    h_b[i] = static_cast<float>((i + 1) % 100);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  swiftware::hpp::mulAddKernel_v1<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; i += 1000) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v1 with fewer blocks than needed (grid-stride should handle)
TEST(MulAddKernelV1Test, FewerBlocksThanElements) {
  const size_t n = 10000;
  std::vector<float> h_a(n), h_b(n), h_out(n, 2.0f), h_res(n);
  std::iota(h_a.begin(), h_a.end(), 0.0f);
  std::iota(h_b.begin(), h_b.end(), 1.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = 10;  // Much fewer blocks than needed (only 2560 threads)
  swiftware::hpp::mulAddKernel_v1<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v1 with non-power-of-2 size
TEST(MulAddKernelV1Test, NonPowerOf2Size) {
  const size_t n = 12345;
  std::vector<float> h_a(n), h_b(n), h_out(n, 3.5f), h_res(n);
  
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i) * 0.1f;
    h_b[i] = static_cast<float>(i) * 0.2f;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  swiftware::hpp::mulAddKernel_v1<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_NEAR(h_res[i], expected, 1e-4) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v1 with single element
TEST(MulAddKernelV1Test, SingleElement) {
  const size_t n = 1;
  std::vector<float> h_a = {7.0f};
  std::vector<float> h_b = {3.0f};
  std::vector<float> h_out = {5.0f};
  std::vector<float> h_res(n);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = 1;
  swiftware::hpp::mulAddKernel_v1<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Expected: 7*3+5 = 26
  ASSERT_FLOAT_EQ(h_res[0], 26.0f);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Tests for v2 (single-pass)
TEST(MulAddKernelV2Test, SmallFloatArray) {
  const size_t n = 1024;
  std::vector<float> h_a(n), h_b(n), h_out(n, 1.0f), h_res(n);
  std::iota(h_a.begin(), h_a.end(), 0.0f);
  std::iota(h_b.begin(), h_b.end(), 1.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  swiftware::hpp::mulAddKernel_v2<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v2 with large array
TEST(MulAddKernelV2Test, LargeArray) {
  const size_t n = 1 << 20;  // 1M elements
  std::vector<float> h_a(n), h_b(n), h_out(n, 1.0f), h_res(n);
  
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i % 100);
    h_b[i] = static_cast<float>((i + 1) % 100);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  swiftware::hpp::mulAddKernel_v2<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; i += 1000) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v2 with non-aligned size
TEST(MulAddKernelV2Test, NonAlignedSize) {
  const size_t n = 1023;  // Not divisible by block size
  std::vector<float> h_a(n), h_b(n), h_out(n, 2.5f), h_res(n);
  std::iota(h_a.begin(), h_a.end(), 0.0f);
  std::iota(h_b.begin(), h_b.end(), 1.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  swiftware::hpp::mulAddKernel_v2<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v2 with single element
TEST(MulAddKernelV2Test, SingleElement) {
  const size_t n = 1;
  std::vector<float> h_a = {4.0f};
  std::vector<float> h_b = {5.0f};
  std::vector<float> h_out = {3.0f};
  std::vector<float> h_res(n);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = 1;
  swiftware::hpp::mulAddKernel_v2<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Expected: 4*5+3 = 23
  ASSERT_FLOAT_EQ(h_res[0], 23.0f);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v2 with negative values
TEST(MulAddKernelV2Test, NegativeValues) {
  const size_t n = 500;
  std::vector<float> h_a(n), h_b(n), h_out(n), h_res(n);
  
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i) - 250.0f;
    h_b[i] = static_cast<float>(i) - 100.0f;
    h_out[i] = static_cast<float>(i) * 0.5f;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  swiftware::hpp::mulAddKernel_v2<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_NEAR(h_res[i], expected, 1e-3) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v2 with very small array
TEST(MulAddKernelV2Test, VerySmallArray) {
  const size_t n = 5;
  std::vector<float> h_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> h_b = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> h_out = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> h_res(n);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = 1;
  swiftware::hpp::mulAddKernel_v2<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Expected: [1*2+1, 2*3+1, 3*4+1, 4*5+1, 5*6+1] = [3, 7, 13, 21, 31]
  ASSERT_FLOAT_EQ(h_res[0], 3.0f);
  ASSERT_FLOAT_EQ(h_res[1], 7.0f);
  ASSERT_FLOAT_EQ(h_res[2], 13.0f);
  ASSERT_FLOAT_EQ(h_res[3], 21.0f);
  ASSERT_FLOAT_EQ(h_res[4], 31.0f);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v3 with size divisible by 4 (perfect alignment)
TEST(MulAddKernelV3Test, AlignedSize) {
  const size_t n = 1024;  // Divisible by 4
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
  const int grid = static_cast<int>((n / 4 + block - 1) / block);
  swiftware::hpp::mulAddKernel_v3<float><<<grid, block>>>(d_a, d_b, d_out, n);
  err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess);

  err = cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v3 with size NOT divisible by 4 (tail case with 1 element)
TEST(MulAddKernelV3Test, TailCase1Element) {
  const size_t n = 1025;  // 1024 + 1
  std::vector<float> h_a(n), h_b(n), h_out(n, 2.0f), h_res(n);
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
  const size_t groups = (n + 3) / 4; // ceil(n/4)
  const int grid = static_cast<int>((groups + block - 1) / block);
  swiftware::hpp::mulAddKernel_v3<float><<<grid, block>>>(d_a, d_b, d_out, n);
  err = cudaDeviceSynchronize();
  ASSERT_EQ(err, cudaSuccess);

  err = cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v3 with tail case of 2 elements
TEST(MulAddKernelV3Test, TailCase2Elements) {
  const size_t n = 1026;  // 1024 + 2
  std::vector<float> h_a(n), h_b(n), h_out(n, 3.0f), h_res(n);
  std::iota(h_a.begin(), h_a.end(), 0.0f);
  std::iota(h_b.begin(), h_b.end(), 1.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block * 4 - 1) / (block * 4));
  swiftware::hpp::mulAddKernel_v3<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v3 with tail case of 3 elements
TEST(MulAddKernelV3Test, TailCase3Elements) {
  const size_t n = 1027;  // 1024 + 3
  std::vector<float> h_a(n), h_b(n), h_out(n, 5.0f), h_res(n);
  std::iota(h_a.begin(), h_a.end(), 0.0f);
  std::iota(h_b.begin(), h_b.end(), 1.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n + block * 4 - 1) / (block * 4));
  swiftware::hpp::mulAddKernel_v3<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v3 with very small array (< 4 elements)
TEST(MulAddKernelV3Test, VerySmallArray) {
  const size_t n = 3;
  std::vector<float> h_a = {1.0f, 2.0f, 3.0f};
  std::vector<float> h_b = {2.0f, 3.0f, 4.0f};
  std::vector<float> h_out = {1.0f, 1.0f, 1.0f};
  std::vector<float> h_res(n);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = 1;
  swiftware::hpp::mulAddKernel_v3<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Expected: [1*2+1, 2*3+1, 3*4+1] = [3, 7, 13]
  ASSERT_FLOAT_EQ(h_res[0], 3.0f);
  ASSERT_FLOAT_EQ(h_res[1], 7.0f);
  ASSERT_FLOAT_EQ(h_res[2], 13.0f);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v3 with single element
TEST(MulAddKernelV3Test, SingleElement) {
  const size_t n = 1;
  std::vector<float> h_a = {5.0f};
  std::vector<float> h_b = {3.0f};
  std::vector<float> h_out = {2.0f};
  std::vector<float> h_res(n);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = 1;
  swiftware::hpp::mulAddKernel_v3<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Expected: 5*3+2 = 17
  ASSERT_FLOAT_EQ(h_res[0], 17.0f);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v3 with large array
TEST(MulAddKernelV3Test, LargeArray) {
  const size_t n = 1 << 20;  // 1M elements
  std::vector<float> h_a(n), h_b(n), h_out(n, 1.0f), h_res(n);
  
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i % 100);
    h_b[i] = static_cast<float>((i + 1) % 100);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n / 4 + block - 1) / block);
  swiftware::hpp::mulAddKernel_v3<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Verify a sampling of results
  for (size_t i = 0; i < n; i += 1000) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

// Test v3 with negative values
TEST(MulAddKernelV3Test, NegativeValues) {
  const size_t n = 100;
  std::vector<float> h_a(n), h_b(n), h_out(n), h_res(n);
  
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i) - 50.0f;  // -50 to 49
    h_b[i] = static_cast<float>(i) - 25.0f;  // -25 to 74
    h_out[i] = static_cast<float>(i) * 0.5f;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_out, n * sizeof(float));

  cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, h_out.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const int block = 256;
  const int grid = static_cast<int>((n / 4 + block - 1) / block);
  swiftware::hpp::mulAddKernel_v3<float><<<grid, block>>>(d_a, d_b, d_out, n);
  cudaDeviceSynchronize();

  cudaMemcpy(h_res.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; ++i) {
    float expected = h_a[i] * h_b[i] + h_out[i];
    ASSERT_FLOAT_EQ(h_res[i], expected) << "Mismatch at index " << i;
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