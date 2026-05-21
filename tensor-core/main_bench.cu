// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include <nvbench/nvbench.cuh>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <random>

#include "kernel.cuh"


inline void cuda_check(cudaError_t e, const char* file, int line) {
  if (e != cudaSuccess) {
    //std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
    std::abort();
  }
}


#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)


void generate_random_float_arrays(half* A, int n) {
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  for (int i = 0; i < n; i++) {
    A[i] = distribution(generator);
  }
}


void dense_matmul(half *A, half *B, float* C, int m, int n, int k)
{
  for (int i = 0; i < m; i++){
    for (int j = 0; j < n; j++){
      float prod = 0;
      for (int l = 0; l < k; l++){
        float f_A = A[i * k + l];
        float f_B = B[l * n + j];
        prod += f_A * f_B;
//        C[i * n + j] += A[i * k + l] * B[l * n + j];
      }
      C[i * n + j] = prod;
    }
  }
}

void report_summary(nvbench::state& state)
{
  state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
  state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
  //state.get_summary("nv/cold/time/gpu/mean").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/min").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/max").set_string("hide", "");
  state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
  state.get_summary("nv/cold/sm_clock_rate/mean").remove_value("hide");
  state.get_summary("nv/cold/sm_clock_rate/scaling/percent").remove_value("hide");

}

#define M 16
#define K 16
#define WMMA_N_TILE 16

void nvbench_matmul_cc(nvbench::state& state)
{
  const size_t N = static_cast<size_t>(state.get_int64("n"));
  int total_elements_C = M * N;
  int num_n_tiles = (N + WMMA_N_TILE - 1) / WMMA_N_TILE; // Total 16x16 tiles across the N dimension

  // --- Naive CUDA Core Execution ---
  // Launch one thread per element C[row][col]
  dim3 threads_naive(32, 1, 1); // 32 threads per block, simple for 1D mapping
  dim3 blocks_naive((N + 31) / 32, M, 1); // A block row for each M row, blocks to cover N

  // generate A as 16 by 16 and B as 16 by N
  half *h_A, *h_B;
  float *h_C, *h_correct_C;
  h_C = (float*)malloc(total_elements_C * sizeof(float));
  h_correct_C = (float*)malloc(total_elements_C * sizeof(float));
  for (int i = 0; i < total_elements_C; i++)
  {
    h_C[i] = 0;
    h_correct_C[i] = 0;
  }
  h_B = (half*)malloc(K*N * sizeof(half));
  h_A = (half*)malloc(M * K * sizeof(half));
  generate_random_float_arrays(h_A, M * K);
  generate_random_float_arrays(h_B, K * N);
  dense_matmul(h_A, h_B, h_correct_C, M, N, K); // compute correct result on host


  half *d_A, *d_B;
  float *d_C;
  CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void**)&d_C, total_elements_C * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
  // Device pointers


  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
    CUDA_CHECK(cudaMemcpyAsync(d_C, h_C, N * sizeof(float), cudaMemcpyHostToDevice, launch.get_stream()));
    // start timer
    timer.start();
    // Naive Kernel Call
    swiftware::hpp::Naive_CUDA_Core_GEMM<<<blocks_naive, threads_naive>>>(d_A, d_B, d_C, N);
    // stop timer
    timer.stop();
  });

  CUDA_CHECK(cudaStreamSynchronize(0)); // or pass the stream you used
  std::vector<float> out_host(total_elements_C);
  CUDA_CHECK(cudaMemcpy(out_host.data(), d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify correctness (roughly)
  bool correct = true;
  for (size_t i = 0; i < total_elements_C; ++i) {
    if (fabs(out_host[i] - h_correct_C[i])/fabs(h_correct_C[i]) > 1e-3) {
      correct = false;
      printf("Mismatch at index %zu: GPU result = %f, CPU result = %f\n", i, out_host[i], h_correct_C[i]);
      break;
    }
  }
  if (!correct) {
    throw std::runtime_error("Results do not match!");
  }
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_correct_C);

  report_summary(state);
}


void nvbench_matmul_tc(nvbench::state& state)
{
  const size_t N = static_cast<size_t>(state.get_int64("n"));
  int total_elements_C = M * N;
  int num_n_tiles = (N + WMMA_N_TILE - 1) / WMMA_N_TILE; // Total 16x16 tiles across the N dimension

  // generate A as 16 by 16 and B as 16 by N
  half *h_A, *h_B;
  float *h_C, *h_correct_C;
  h_C = (float*)malloc(total_elements_C * sizeof(float));
  h_correct_C = (float*)malloc(total_elements_C * sizeof(float));
  for (int i = 0; i < total_elements_C; i++)
  {
    h_C[i] = 0;
    h_correct_C[i] = 0;
  }
  h_B = (half*)malloc(K*N * sizeof(half));
  h_A = (half*)malloc(M * K * sizeof(half));
  generate_random_float_arrays(h_A, M * K);
  generate_random_float_arrays(h_B, K * N);
  dense_matmul(h_A, h_B, h_correct_C, M, N, K); // compute correct result on host


  half *d_A, *d_B;
  float *d_C;
  CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void**)&d_C, total_elements_C * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
  // Device pointers

  dim3 threads_wmma(32, 1, 1);
  dim3 blocks_wmma(num_n_tiles, 1, 1); // One block per 16x16 tile in the N dimension

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
    CUDA_CHECK(cudaMemcpyAsync(d_C, h_C, N * sizeof(float), cudaMemcpyHostToDevice, launch.get_stream()));
    // start timer
    timer.start();
    // Naive Kernel Call
    swiftware::hpp::WMMA_Tensor_Core_GEMM<<<blocks_wmma, threads_wmma>>>(d_A, d_B, d_C, N);
    // stop timer
    timer.stop();
  });

  CUDA_CHECK(cudaStreamSynchronize(0)); // or pass the stream you used
  std::vector<float> out_host(total_elements_C);
  CUDA_CHECK(cudaMemcpy(out_host.data(), d_C, total_elements_C * sizeof(float), cudaMemcpyDeviceToHost));

  // Verify correctness
  bool correct = true;
  for (size_t i = 0; i < total_elements_C; ++i) {
    if (fabs(out_host[i] - h_correct_C[i])/fabs(h_correct_C[i]) > 1e-3) {
      correct = false;
      printf("Mismatch at index %zu: GPU result = %f, CPU result = %f\n", i, out_host[i], h_correct_C[i]);
      break;
    }
  }
  if (!correct) {
    throw std::runtime_error("Results do not match!");
  }
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_correct_C);

  report_summary(state);
}



NVBENCH_BENCH(nvbench_matmul_cc).set_name("matmul_cc").add_int64_axis("n", {1<<10, 1<<11, 1<<12, 1<<13});
NVBENCH_BENCH(nvbench_matmul_tc).set_name("matmul_tc").add_int64_axis("n", {1<<10, 1<<11, 1<<12, 1<<13});


