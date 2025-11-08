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
  state.set_blocking_kernel_timeout(160.0); // set timeout to 60 seconds
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
  // Print the matrix
  // Convert the matrix to CSR format
  swiftware::hpp::CSR<double> csr_matrix = swiftware::hpp::COO_to_CSR(matrix);
  // build rhs
  int n = matrix.cols;
  std::vector<double> rhs(n);
  // Allocate memory for the solution vector
  std::vector<double> solution(n, 0.0f);
  auto SP = swiftware::hpp::ScheduleParams(-1, -1, 20, 1);

  // TODO : add necessary function to allocate memory and copy it to device





  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){

    // start timer, launch SPTRSV kernel, stop timer
    timer.start();

      ///swiftware::hpp::sparse_csr_parallel_gpu<double>;

    timer.stop();
  });

// TODO : ensure device work finished, copy solution back and free device memory


  //TODO:  test the solution vector to be all ones


  report_summary(state);
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

  // TODO allocate and copy data to device


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


  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_col));
  CUDA_CHECK(cudaFree(d_row));
  CUDA_CHECK(cudaFree(d_rhs));
  CUDA_CHECK(cudaFree(d_solution));

  report_summary(state);
}



NVBENCH_BENCH(nvbench_sptrsv).set_name("sptrsv").add_int64_axis("n", {0, 1, 2, 3});
NVBENCH_BENCH(nvbench_sptrsv_cusparse).set_name("sptrsv_cusparse").add_int64_axis("n", {0, 1, 2, 3});

