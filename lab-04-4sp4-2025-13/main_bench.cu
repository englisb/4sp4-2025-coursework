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
#include <chrono>

#include "sparse_io.h"
#include "triangular_solver.cuh"

inline void cuda_check(cudaError_t e, const char* file, int line) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
    std::abort();
  }
}

#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)


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

void nvbench_sptrsv(nvbench::state& state)
{
  auto mat_index = static_cast<size_t>(state.get_int64("n"));
  state.set_blocking_kernel_timeout(160.0);
  std::vector<std::string> file_names;
  file_names.push_back("./data/1138_bus/1138_bus.mtx");
  file_names.push_back("./data/crystm01/crystm01.mtx");
  file_names.push_back("./data/minsurfo/minsurfo.mtx");
  file_names.push_back("./data/apache2/apache2.mtx");
  file_names.push_back("./data/test/test.mtx");

  auto &matrix_summary = state.add_summary("matrix_name");
  matrix_summary.set_string("value", file_names[mat_index]);

  const std::string& file_name = file_names[mat_index];
  swiftware::hpp::Matrix<double> matrix = swiftware::hpp::readMatrixMarket<double>(file_name);
  swiftware::hpp::CSR<double> csr_matrix = swiftware::hpp::COO_to_CSR(matrix);
  
  int n = matrix.cols;
  std::vector<double> rhs(n);
  std::vector<double> solution(n, 0.0f);

  std::vector<double> expected_solution;
  swiftware::hpp::build_rhs_for_triangular_solve(csr_matrix, rhs, expected_solution);
  
  // PREPROCESSING (SCHEDULING TIME) - Measure separately for analysis
  auto sched_start = std::chrono::high_resolution_clock::now();
  std::vector<int> level_ptr, level_set;
  swiftware::hpp::compute_level_sets(csr_matrix.col_indices.data(), 
                                      csr_matrix.row_pointer.data(), 
                                      n, level_ptr, level_set);
  int num_levels = level_ptr.size() - 1;
  auto sched_end = std::chrono::high_resolution_clock::now();
  double scheduling_time_us = std::chrono::duration<double, std::micro>(sched_end - sched_start).count();

  // Allocate device memory
  double *d_val = nullptr;
  int *d_col_ind = nullptr;
  int *d_row_ptr = nullptr;
  double *d_x = nullptr;
  double *d_b = nullptr;
  int *d_level_ptr = nullptr;
  int *d_level_set = nullptr;

  size_t nnz = csr_matrix.values.size();
  CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_level_ptr, level_ptr.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_level_set, level_set.size() * sizeof(int)));

  // Copy data to device (including preprocessed level sets)
  CUDA_CHECK(cudaMemcpy(d_val, csr_matrix.values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_ind, csr_matrix.col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_matrix.row_pointer.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_level_ptr, level_ptr.data(), level_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_level_set, level_set.data(), level_set.size() * sizeof(int), cudaMemcpyHostToDevice));

  // Launch configuration
  int threadsPerBlock = 256;
  
  // BENCHMARK ONLY THE SOLVE TIME (not scheduling)
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch){
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));

    // Launch one kernel per level - all rows in a level execute in parallel
    for (int level = 0; level < num_levels; ++level) {
      int level_size = level_ptr[level + 1] - level_ptr[level];
      int blocksPerGrid = (level_size + threadsPerBlock - 1) / threadsPerBlock;
      
      swiftware::hpp::sparse_csr_parallel_gpu_leveled<double><<<blocksPerGrid, threadsPerBlock>>>(
        d_val, d_col_ind, d_row_ptr, d_x, d_b, n, d_level_ptr, d_level_set, level
      );
      // Don't sync inside loop - let kernels overlap if possible
    }
    // Single sync at end
    CUDA_CHECK(cudaDeviceSynchronize());
  });

  // Ensure device work finished and copy solution back
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(solution.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

  // Verify correctness: solution should match expected_solution
  double max_error = 0.0;
  for (int i = 0; i < n; ++i) {
    double error = std::abs(solution[i] - expected_solution[i]);
    max_error = std::max(max_error, error);
  }
  
  if (max_error > 1e-5) {
    std::cerr << "WARNING: Custom GPU kernel solution has error " << max_error 
              << " for matrix " << file_names[mat_index] << std::endl;
  }

  // Add custom summary values for analysis
  state.add_summary("Scheduling Time (us)").set_string("value", std::to_string(scheduling_time_us));
  state.add_summary("Num Levels").set_string("value", std::to_string(num_levels));
  state.add_summary("Max Error").set_string("value", std::to_string(max_error));

  // Free device memory
  CUDA_CHECK(cudaFree(d_val));
  CUDA_CHECK(cudaFree(d_col_ind));
  CUDA_CHECK(cudaFree(d_row_ptr));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_level_ptr));
  CUDA_CHECK(cudaFree(d_level_set));
}

void nvbench_sptrsv_sequential_gpu(nvbench::state& state)
{
  auto mat_index = static_cast<size_t>(state.get_int64("n"));
  state.set_blocking_kernel_timeout(160.0);
  std::vector<std::string> file_names;
  file_names.push_back("./data/1138_bus/1138_bus.mtx");
  file_names.push_back("./data/crystm01/crystm01.mtx");
  file_names.push_back("./data/minsurfo/minsurfo.mtx");
  file_names.push_back("./data/apache2/apache2.mtx");
  file_names.push_back("./data/test/test.mtx");

  auto &matrix_summary = state.add_summary("matrix_name");
  matrix_summary.set_string("value", file_names[mat_index]);

  const std::string& file_name = file_names[mat_index];
  swiftware::hpp::Matrix<double> matrix = swiftware::hpp::readMatrixMarket<double>(file_name);
  swiftware::hpp::CSR<double> csr_matrix = swiftware::hpp::COO_to_CSR(matrix);
  
  int n = matrix.cols;
  std::vector<double> rhs(n);
  std::vector<double> solution(n, 0.0f);

  std::vector<double> expected_solution;
  swiftware::hpp::build_rhs_for_triangular_solve(csr_matrix, rhs, expected_solution);

  // Allocate device memory
  double *d_val = nullptr;
  int *d_col_ind = nullptr;
  int *d_row_ptr = nullptr;
  double *d_x = nullptr;
  double *d_b = nullptr;

  size_t nnz = csr_matrix.values.size();
  CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_val, csr_matrix.values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_ind, csr_matrix.col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_matrix.row_pointer.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch){
    CUDA_CHECK(cudaMemset(d_x, 0, n * sizeof(double)));
    
    // Launch single-threaded GPU kernel
    swiftware::hpp::sparse_csr_sequential_gpu<double><<<1, 1>>>(
      d_val, d_col_ind, d_row_ptr, d_x, d_b, n
    );
    CUDA_CHECK(cudaDeviceSynchronize());
  });

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(solution.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

  double max_error = 0.0;
  for (int i = 0; i < n; ++i) {
    double error = std::abs(solution[i] - expected_solution[i]);
    max_error = std::max(max_error, error);
  }
  
  if (max_error > 1e-5) {
    std::cerr << "WARNING: GPU Sequential solution has error " << max_error 
              << " for matrix " << file_names[mat_index] << std::endl;
  }

  CUDA_CHECK(cudaFree(d_val));
  CUDA_CHECK(cudaFree(d_col_ind));
  CUDA_CHECK(cudaFree(d_row_ptr));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_b));
}


void nvbench_sptrsv_cusparse(nvbench::state& state)
{
  auto mat_index = static_cast<size_t>(state.get_int64("n"));
  std::vector<std::string> file_names;

  file_names.push_back("./data/1138_bus/1138_bus.mtx");
  file_names.push_back("./data/crystm01/crystm01.mtx");
  file_names.push_back("./data/minsurfo/minsurfo.mtx");
  file_names.push_back("./data/apache2/apache2.mtx");
  file_names.push_back("./data/test/test.mtx");
  // put matrix name into state
  auto &matrix_summary = state.add_summary("matrix_name");
  matrix_summary.set_string("value", file_names[mat_index]);

  const std::string& file_name = file_names[mat_index];
  // Read the matrix from the file
  swiftware::hpp::Matrix<double> matrix = swiftware::hpp::readMatrixMarket<double>(file_name);
  // Convert the matrix to CSR format
  swiftware::hpp::CSR<double> csr_matrix = swiftware::hpp::COO_to_CSR(matrix);
  // build rhs
  int n = matrix.cols;
  std::vector<double> rhs(n);

  // Allocate memory for the solution vector
  std::vector<double> solution(n, 0.0f);

// allocate device buffers for CSR and vectors
  double *d_values = nullptr;
  int *d_col = nullptr;
  int *d_row = nullptr;
  double *d_rhs = nullptr;
  double *d_solution = nullptr;

  size_t nnz = csr_matrix.values.size();
  size_t col_size = csr_matrix.col_indices.size();
  size_t row_ptr_size = csr_matrix.row_pointer.size();

  // Build RHS vector
  std::vector<double> expected_solution;
  swiftware::hpp::build_rhs_for_triangular_solve(csr_matrix, rhs, expected_solution);

  // Allocate and copy data to device
  CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_col, col_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_row, row_ptr_size * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_rhs, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_solution, n * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_values, csr_matrix.values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col, csr_matrix.col_indices.data(), col_size * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_row, csr_matrix.row_pointer.data(), row_ptr_size * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rhs, rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice));

  cusparseHandle_t cusparse;
  cusparseCreate(&cusparse);
  cusparseSetStream(cusparse, 0);
  cusparseSpMatDescr_t matA;
  cusparseCreateCsr(&matA, n, n, (int)nnz, d_row, d_col, d_values,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

  cusparseDnVecDescr_t vecX, vecY;
  cusparseCreateDnVec(&vecX, n, d_rhs, CUDA_R_64F);
  cusparseCreateDnVec(&vecY, n, d_solution, CUDA_R_64F);

  cusparseSpSVDescr_t spsv;
  cusparseSpSV_createDescr(&spsv);

  double alpha = 1.0;
  size_t bufferSize = 0;
  cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, vecX, vecY, CUDA_R_64F,
                          CUSPARSE_SPSV_ALG_DEFAULT, spsv, &bufferSize);

  void* dBuffer = nullptr;
  if (bufferSize > 0) CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
    CUDA_CHECK(cudaMemset(d_solution, 0, n * sizeof(double)));

      cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                           matA, vecX, vecY, CUDA_R_64F,
                           CUSPARSE_SPSV_ALG_DEFAULT, spsv, dBuffer);

    // start timer, launch SPTRSV kernel, stop timer
    timer.start();
      cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                         matA, vecX, vecY, CUDA_R_64F,
                         CUSPARSE_SPSV_ALG_DEFAULT, spsv);

    timer.stop();
    });
  if (dBuffer) CUDA_CHECK(cudaFree(dBuffer));
  cusparseSpSV_destroyDescr(spsv);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroySpMat(matA);
  cusparseDestroy(cusparse);

// ensure device work finished, copy solution back and free device memory
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(solution.data(), d_solution, n * sizeof(double), cudaMemcpyDeviceToHost));

  // Verify correctness: solution should match expected_solution
  double max_error = 0.0;
  for (int i = 0; i < n; ++i) {
    double error = std::abs(solution[i] - expected_solution[i]);
    max_error = std::max(max_error, error);
  }
  
  if (max_error > 1e-5) {
    std::cerr << "WARNING: cuSPARSE solution has error " << max_error 
              << " for matrix " << file_names[mat_index] << std::endl;
  }

  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_col));
  CUDA_CHECK(cudaFree(d_row));
  CUDA_CHECK(cudaFree(d_rhs));
  CUDA_CHECK(cudaFree(d_solution));

  report_summary(state);
}



NVBENCH_BENCH(nvbench_sptrsv).set_name("sptrsv").add_int64_axis("n", {0, 1, 2, 3});
NVBENCH_BENCH(nvbench_sptrsv_sequential_gpu).set_name("sptrsv_sequential_gpu").add_int64_axis("n", {0, 1, 2, 3});
NVBENCH_BENCH(nvbench_sptrsv_cusparse).set_name("sptrsv_cusparse").add_int64_axis("n", {0, 1, 2, 3});

