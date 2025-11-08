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
#include "kernel.cuh"
#include "utils.h"


#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)


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

void nvbench_mulAdd_v1(nvbench::state& state)
{
  const size_t n = static_cast<size_t>(state.get_int64("n"));
  std::vector<float> A(n), B(n), Out(n), Ref(n);
  swiftware::hpp::generate_random_float_arrays(A.data(), B.data(), Out.data(), Ref.data(), n);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  swiftware::hpp::allocate_and_copy_to_device(A.data(), B.data(), Out.data(), &d_a, &d_b, &d_out, n);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
    CUDA_CHECK(cudaMemcpyAsync(d_out, Out.data(), n * sizeof(float), cudaMemcpyHostToDevice, launch.get_stream()));
    // start timer
    timer.start();
    swiftware::hpp::mulAddKernel_v1<float><<<grid, block, 0, launch.get_stream()>>>(d_a, d_b, d_out, n);
    // stop timer
    timer.stop();
  });

  //CUDA_CHECK(cudaStreamSynchronize(0)); // or pass the stream you used
  std::vector<float> out_host(n);
  CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));


  // compare to Ref with small epsilon
  const float eps = 1e-5f;
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(out_host[i] - Ref[i]) > eps) {
      std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], Ref[i]);
      std::abort();
    }
  }

  swiftware::hpp::free_buffers(d_a, d_b, d_out);

  report_summary(state);
}

void nvbench_mulAdd_v2(nvbench::state& state)
{
  const size_t n = static_cast<size_t>(state.get_int64("n"));
  std::vector<float> A(n), B(n), Out(n), Ref(n);
  swiftware::hpp::generate_random_float_arrays(A.data(), B.data(), Out.data(), Ref.data(), n);

  float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  swiftware::hpp::allocate_and_copy_to_device(A.data(), B.data(), Out.data(), &d_a, &d_b, &d_out, n);

  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
    CUDA_CHECK(cudaMemcpyAsync(d_out, Out.data(), n * sizeof(float), cudaMemcpyHostToDevice, launch.get_stream()));
    // start timer
    timer.start();
    swiftware::hpp::mulAddKernel_v2<float><<<grid, block, 0, launch.get_stream()>>>(d_a, d_b, d_out, n);
    // stop timer
    timer.stop();
  });

  //CUDA_CHECK(cudaStreamSynchronize(0)); // or pass the stream you used
  std::vector<float> out_host(n);
  CUDA_CHECK(cudaMemcpy(out_host.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));


  // compare to Ref with small epsilon
  const float eps = 1e-5f;
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(out_host[i] - Ref[i]) > eps) {
      std::fprintf(stderr, "Mismatch at %zu: got %f expected %f\n", i, out_host[i], Ref[i]);
      std::abort();
    }
  }

  swiftware::hpp::free_buffers(d_a, d_b, d_out);

  report_summary(state);
}

// TODO: add nvbench_mulAdd_v3
void nvbench_mulAdd_v3(nvbench::state& state)
{
  // TODO: add your benchmarking code here
}

NVBENCH_BENCH(nvbench_mulAdd_v1).set_name("mulAdd_v1").add_int64_axis("n", {1<<10, 1<<15, 1<<20, 1<<25});
NVBENCH_BENCH(nvbench_mulAdd_v2).set_name("mulAdd_v2").add_int64_axis("n", {1<<10, 1<<15, 1<<20, 1<<25});
