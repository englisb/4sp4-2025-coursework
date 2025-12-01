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
#include "kernels.cuh"
#include "gpu_utils.h"
#include "include/gpu_dense_nn.cuh"


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

void nvbench_gemm(nvbench::state& state)
{
    const size_t n = static_cast<size_t>(state.get_int64("n"));


    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    auto *A = new swiftware::hpp::DenseMatrix(n, n);
    auto *B = new swiftware::hpp::DenseMatrix(n, n);
    auto *C = new swiftware::hpp::DenseMatrix(n, n);
    // Allocate and initialize A, B, C on GPU
    CUDA_CHECK(cudaMalloc((void**)&A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&C, n * n * sizeof(float)));
    // Initialize A and B with some values
    // TODO: Initialize A and B with some values

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        // start timer
        timer.start();
        gemm_gpu_baseline<<<grid, block>>>(n, n, n, A, B, C);
        // stop timer
        timer.stop();
    });


    //TODO  compare to Ref with small epsilon

    report_summary(state);
}

void nvbench_gemv(nvbench::state& state)
{
    const size_t n = static_cast<size_t>(state.get_int64("n"));


    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        // start timer
        timer.start();
        // TODO: launch your  kernel here
        // stop timer
        timer.stop();
    });


    //TODO  compare to Ref with small epsilon

    report_summary(state);
}

void nvbench_spmm(nvbench::state& state)
{
    const size_t n = static_cast<size_t>(state.get_int64("n"));


    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        // start timer
        timer.start();
        // TODO: launch your  kernel here
        // stop timer
        timer.stop();
    });


    //TODO  compare to Ref with small epsilon

    report_summary(state);
}

void nvbench_gemv(nvbench::state& state)
{
    const size_t n = static_cast<size_t>(state.get_int64("n"));


    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        // start timer
        timer.start();
        // TODO: launch your  kernel here
        // stop timer
        timer.stop();
    });


    //TODO  compare to Ref with small epsilon

    report_summary(state);
}


NVBENCH_BENCH(nvbench_gemm).set_name("gemm").add_int64_power_of_two_axis("n", nvbench::range(4, 12, 2))
                            .add_string_axis("GemmStrategy", {"BASELINE", "SHARED_MEMORY", "COALESCED_MEMORY", "COMBINED"});


