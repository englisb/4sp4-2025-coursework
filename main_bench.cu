// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gpu_utils.h"
#include "include/gpu_dense_nn.cuh"
#include "kernels.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <numeric>
#include <nvbench/nvbench.cuh>
#include <vector>

#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)

// Simple helper to build a random CSR matrix with given sparsity (fraction of
// zeros). Sparsity is provided as [0,1], e.g., 0.9 means keep 10% of entries.
static void build_random_csr(size_t n, float sparsity, std::vector<int> &row_ptr,
                             std::vector<int> &col_id,
                             std::vector<float> &values) {
  row_ptr.assign(n + 1, 0);
  col_id.clear();
  values.clear();

  int nnz = 0;
  const float keep_threshold = 1.0f - sparsity; // probability to keep entry
  for (size_t i = 0; i < n; i++) {
    row_ptr[i] = nnz;
    for (size_t j = 0; j < n; j++) {
      if (static_cast<float>(rand()) / RAND_MAX < keep_threshold) {
        col_id.push_back(static_cast<int>(j));
        values.push_back(static_cast<float>(rand()) / RAND_MAX);
        nnz++;
      }
    }
  }
  row_ptr[n] = nnz;
}

void report_summary(nvbench::state &state) {
  state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
  // state.get_summary("nv/cold/time/gpu/mean").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/min").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/max").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
  state.get_summary("nv/cold/sm_clock_rate/mean").remove_value("hide");
  state.get_summary("nv/cold/sm_clock_rate/scaling/percent")
      .remove_value("hide");
}

void nvbench_gemm(nvbench::state &state) {
  const size_t n = static_cast<size_t>(state.get_int64("n"));

  // Allocate host memory
  std::vector<float> h_a(n * n);
  std::vector<float> h_b(n * n);
  std::vector<float> h_c(n * n, 0.0f);

  // Initialize with random values
  for (size_t i = 0; i < n * n; i++) {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * n * sizeof(float)));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, n * n * sizeof(float)));

  int threads = 256;
  int blocks = (n * n + threads - 1) / threads;

  state.exec(nvbench::exec_tag::timer,
             [&](nvbench::launch &launch, auto &timer) {
               timer.start();
               swiftware::hpp::gemm_gpu_baseline<<<blocks, threads>>>(
                   n, n, n, d_a, d_b, d_c);
               timer.stop();
             });

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  report_summary(state);
}

void nvbench_gemm_shared(nvbench::state &state) {
  const size_t n = static_cast<size_t>(state.get_int64("n"));

  std::vector<float> h_a(n * n);
  std::vector<float> h_b(n * n);
  std::vector<float> h_c(n * n, 0.0f);

  for (size_t i = 0; i < n * n; i++) {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, n * n * sizeof(float)));

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch,
                                           auto &timer) {
    timer.start();
    swiftware::hpp::gemm_gpu_shared<<<grid, block>>>(n, n, n, d_a, d_b, d_c);
    timer.stop();
  });

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  report_summary(state);
}

void nvbench_gemm_coalesced(nvbench::state &state) {
  const size_t n = static_cast<size_t>(state.get_int64("n"));

  std::vector<float> h_a(n * n);
  std::vector<float> h_b(n * n);
  std::vector<float> h_c(n * n, 0.0f);

  for (size_t i = 0; i < n * n; i++) {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, n * n * sizeof(float)));

  dim3 block(16, 16);
  dim3 grid((n + 15) / 16, (n + 15) / 16);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch,
                                           auto &timer) {
    timer.start();
    swiftware::hpp::gemm_gpu_coalesced<<<grid, block>>>(n, n, n, d_a, d_b, d_c);
    timer.stop();
  });

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  report_summary(state);
}

void nvbench_gemm_combined(nvbench::state &state) {
  const size_t n = static_cast<size_t>(state.get_int64("n"));

  std::vector<float> h_a(n * n);
  std::vector<float> h_b(n * n);
  std::vector<float> h_c(n * n, 0.0f);

  for (size_t i = 0; i < n * n; i++) {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, n * n * sizeof(float)));

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch,
                                           auto &timer) {
    timer.start();
    swiftware::hpp::gemm_gpu_combined<<<grid, block>>>(n, n, n, d_a, d_b, d_c);
    timer.stop();
  });

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  report_summary(state);
}

void nvbench_gemv(nvbench::state &state) {
  const size_t n = static_cast<size_t>(state.get_int64("n"));

  std::vector<float> h_a(n * n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n, 0.0f);

  for (size_t i = 0; i < n * n; i++) {
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (size_t i = 0; i < n; i++) {
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, n * sizeof(float)));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  state.exec(nvbench::exec_tag::timer,
             [&](nvbench::launch &launch, auto &timer) {
               timer.start();
               swiftware::hpp::MV<<<blocks, threads>>>(d_a, d_b, d_c, n, n);
               timer.stop();
             });

  CUDA_CHECK(
      cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  report_summary(state);
}

// ---------------------------------------------
// SpMM variants (BASELINE, COALESCED, SHARED, COMBINED)
// ---------------------------------------------
template <typename Kernel>
void nvbench_spmm_variant(nvbench::state &state, Kernel kernel) {
  const size_t n = static_cast<size_t>(state.get_int64("n"));
  const float sparsity = state.get_float64("sparsity") / 100.0f;

  std::vector<int> row_ptr;
  std::vector<int> col_id;
  std::vector<float> values;
  build_random_csr(n, sparsity, row_ptr, col_id, values);
  const int nnz = static_cast<int>(values.size());

  std::vector<float> h_b(n * n);
  for (size_t i = 0; i < n * n; i++) {
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  int *d_row_ptr = nullptr, *d_col_id = nullptr;
  float *d_values = nullptr, *d_b = nullptr, *d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col_id, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_id, col_id.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, n * n * sizeof(float)));

  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);

  state.exec(nvbench::exec_tag::timer,
             [&](nvbench::launch &, auto &timer) {
               timer.start();
               kernel<<<blocks, threads>>>(d_row_ptr, d_col_id, d_values,
                                           d_b, d_c, n, n, n);
               timer.stop();
             });

  CUDA_CHECK(cudaFree(d_row_ptr));
  CUDA_CHECK(cudaFree(d_col_id));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  report_summary(state);
}

void nvbench_spmm_baseline(nvbench::state &state) {
  nvbench_spmm_variant(state, swiftware::hpp::SpMM_BASELINE);
}

void nvbench_spmm_coalesced(nvbench::state &state) {
  nvbench_spmm_variant(state, swiftware::hpp::SpMM_COALESCED);
}

void nvbench_spmm_shared(nvbench::state &state) {
  const size_t n = static_cast<size_t>(state.get_int64("n"));
  const float sparsity = state.get_float64("sparsity") / 100.0f;

  std::vector<int> row_ptr;
  std::vector<int> col_id;
  std::vector<float> values;
  build_random_csr(n, sparsity, row_ptr, col_id, values);
  const int nnz = static_cast<int>(values.size());

  std::vector<float> h_b(n * n);
  for (size_t i = 0; i < n * n; i++) {
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  int *d_row_ptr = nullptr, *d_col_id = nullptr;
  float *d_values = nullptr, *d_b = nullptr, *d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col_id, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_id, col_id.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, n * n * sizeof(float)));

  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);

  state.exec(nvbench::exec_tag::timer,
             [&](nvbench::launch &, auto &timer) {
               timer.start();
               swiftware::hpp::SpMM_SHARED_MEMORY<<<blocks, threads>>>(
                   d_row_ptr, d_col_id, d_values, d_b, d_c, n, n, n, TILE_SIZE);
               timer.stop();
             });

  CUDA_CHECK(cudaFree(d_row_ptr));
  CUDA_CHECK(cudaFree(d_col_id));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  report_summary(state);
}
void nvbench_spmm_combined(nvbench::state &state) {
  nvbench_spmm_variant(state, swiftware::hpp::SpMM_COMBINED);
}

// ---------------------------------------------
// SpMV variants (BASELINE, COALESCED, WARP_LEVEL, COMBINED)
// ---------------------------------------------
template <typename Kernel>
void nvbench_spmv_variant(nvbench::state &state, Kernel kernel) {
  const size_t n = static_cast<size_t>(state.get_int64("n"));
  const float sparsity = state.get_float64("sparsity") / 100.0f;

  std::vector<int> row_ptr;
  std::vector<int> col_id;
  std::vector<float> values;
  build_random_csr(n, sparsity, row_ptr, col_id, values);
  const int nnz = static_cast<int>(values.size());

  std::vector<float> h_b(n);
  for (size_t i = 0; i < n; i++) {
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  int *d_row_ptr = nullptr, *d_col_id = nullptr;
  float *d_values = nullptr, *d_b = nullptr, *d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col_id, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_id, col_id.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, n * sizeof(float)));

  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);

  state.exec(nvbench::exec_tag::timer,
             [&](nvbench::launch &, auto &timer) {
               timer.start();
               kernel<<<blocks, threads>>>(d_row_ptr, d_col_id, d_values, d_b,
                                           d_c, n, n);
               timer.stop();
             });

  CUDA_CHECK(cudaFree(d_row_ptr));
  CUDA_CHECK(cudaFree(d_col_id));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  report_summary(state);
}

void nvbench_spmv_baseline(nvbench::state &state) {
  nvbench_spmv_variant(state, swiftware::hpp::SpMV_BASELINE);
}

void nvbench_spmv_coalesced(nvbench::state &state) {
  nvbench_spmv_variant(state, swiftware::hpp::SpMV_COALESCED);
}

void nvbench_spmv_warp(nvbench::state &state) {
  nvbench_spmv_variant(state, swiftware::hpp::SpMV_WARP_LEVEL);
}

void nvbench_spmv_combined(nvbench::state &state) {
  nvbench_spmv_variant(state, swiftware::hpp::SpMV_COMBINED);
}

// ============================================================================
// Block Size Tuning Benchmarks - Test different block sizes to find optimal
// ============================================================================
template <typename Kernel>
void nvbench_spmm_blocksize_tuning(nvbench::state &state, Kernel kernel) {
  const size_t n = static_cast<size_t>(state.get_int64("n"));
  const int block_size = static_cast<int>(state.get_int64("block_size"));
  const float sparsity = state.get_float64("sparsity") / 100.0f;

  std::vector<int> row_ptr;
  std::vector<int> col_id;
  std::vector<float> values;
  build_random_csr(n, sparsity, row_ptr, col_id, values);
  const int nnz = static_cast<int>(values.size());

  std::vector<float> h_b(n * n);
  for (size_t i = 0; i < n * n; i++) {
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  int *d_row_ptr = nullptr, *d_col_id = nullptr;
  float *d_values = nullptr, *d_b = nullptr, *d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col_id, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, n * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, n * n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_id, col_id.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, n * n * sizeof(float)));

  const int blocks = static_cast<int>((n + block_size - 1) / block_size);

  state.exec(nvbench::exec_tag::timer,
             [&](nvbench::launch &, auto &timer) {
               timer.start();
               kernel<<<blocks, block_size>>>(d_row_ptr, d_col_id, d_values,
                                              d_b, d_c, n, n, n);
               timer.stop();
             });

  CUDA_CHECK(cudaFree(d_row_ptr));
  CUDA_CHECK(cudaFree(d_col_id));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  report_summary(state);
}

void nvbench_spmm_combined_tuning(nvbench::state &state) {
  nvbench_spmm_blocksize_tuning(state, swiftware::hpp::SpMM_COMBINED);
}

NVBENCH_BENCH(nvbench_gemm)
    .set_name("GEMM_Baseline")
    .add_int64_axis("n", {256, 512, 1024, 2048, 4096});
NVBENCH_BENCH(nvbench_gemm_shared)
    .set_name("GEMM_Shared")
    .add_int64_axis("n", {256, 512, 1024, 2048, 4096});
NVBENCH_BENCH(nvbench_gemm_coalesced)
    .set_name("GEMM_Coalesced")
    .add_int64_axis("n", {256, 512, 1024, 2048, 4096});
NVBENCH_BENCH(nvbench_gemm_combined)
    .set_name("GEMM_Combined")
    .add_int64_axis("n", {256, 512, 1024, 2048, 4096});
NVBENCH_BENCH(nvbench_gemv)
    .set_name("GEMV")
    .add_int64_axis("n", {256, 512, 1024, 2048, 4096});

// Block size tuning for SpMM - test different block sizes (128, 256, 512, 1024)
NVBENCH_BENCH(nvbench_spmm_combined_tuning)
  .set_name("SpMM_Combined_Tuning")
  .add_int64_axis("n", {512, 1024, 2048, 4096})
  .add_int64_axis("block_size", {128, 256, 512, 1024})
  .add_float64_axis("sparsity", {50.0, 70.0, 90.0});

NVBENCH_BENCH(nvbench_spmm_baseline)
  .set_name("SpMM_Baseline")
  .add_int64_axis("n", {128, 256, 512, 1024, 2048, 4096})
  .add_float64_axis("sparsity", {50.0, 70.0, 90.0});
NVBENCH_BENCH(nvbench_spmm_coalesced)
  .set_name("SpMM_Coalesced")
  .add_int64_axis("n", {128, 256, 512, 1024, 2048, 4096})
  .add_float64_axis("sparsity", {50.0, 70.0, 90.0});
NVBENCH_BENCH(nvbench_spmm_shared)
  .set_name("SpMM_Shared")
  .add_int64_axis("n", {128, 256, 512, 1024, 2048, 4096})
  .add_float64_axis("sparsity", {50.0, 70.0, 90.0});
NVBENCH_BENCH(nvbench_spmm_combined)
  .set_name("SpMM_Combined")
  .add_int64_axis("n", {128, 256, 512, 1024, 2048, 4096})
  .add_float64_axis("sparsity", {50.0, 70.0, 90.0});

NVBENCH_BENCH(nvbench_spmv_baseline)
  .set_name("SpMV_Baseline")
  .add_int64_axis("n", {128, 256, 512, 1024, 2048, 4096})
  .add_float64_axis("sparsity", {50.0, 70.0, 90.0});
NVBENCH_BENCH(nvbench_spmv_coalesced)
  .set_name("SpMV_Coalesced")
  .add_int64_axis("n", {128, 256, 512, 1024, 2048, 4096})
  .add_float64_axis("sparsity", {50.0, 70.0, 90.0});
NVBENCH_BENCH(nvbench_spmv_warp)
  .set_name("SpMV_WarpLevel")
  .add_int64_axis("n", {128, 256, 512, 1024, 2048, 4096})
  .add_float64_axis("sparsity", {50.0, 70.0, 90.0});
NVBENCH_BENCH(nvbench_spmv_combined)
  .set_name("SpMV_Combined")
  .add_int64_axis("n", {128, 256, 512, 1024, 2048, 4096})
  .add_float64_axis("sparsity", {50.0, 70.0, 90.0});
