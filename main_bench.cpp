// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#ifdef USE_MKL
#include "mkl.h"
#include "mkl_spblas.h"
#endif

#include "benchmark/benchmark.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include "sparse_io.h"
#include "triangular_solver.h"



static void BM_SPTRSV(benchmark::State &state) {
  auto mat_index = state.range(0);
  auto threadNo = state.range(1);
  std::vector<std::string> file_names;
    file_names.push_back("./data/1138_bus/1138_bus.mtx");
    file_names.push_back("./data/crystm01/crystm01.mtx");
    file_names.push_back("./data/minsurfo/minsurfo.mtx");
    file_names.push_back("./data/apache2/apache2.mtx");
    file_names.push_back("./data/test/test.mtx");
    // put matrix name in the log
    state.SetLabel("Matrix:" + file_names[mat_index]);

    const std::string& file_name = file_names[mat_index];
    // Read the matrix from the file
    swiftware::hpp::Matrix<double> matrix = swiftware::hpp::readMatrixMarket<double>(file_name);
    // Convert the matrix to CSR format
    swiftware::hpp::CSR<double> csr_matrix = swiftware::hpp::COO_to_CSR(matrix);
    // build rhs such that x = 1 is the solution: b_i = sum_j L_ij
    int n = matrix.cols;
    std::vector<double> rhs(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int idx = csr_matrix.row_pointer[i]; idx < csr_matrix.row_pointer[i+1]; ++idx) {
            rhs[i] += csr_matrix.values[idx];
        }
    }
    // Allocate memory for the solution vector
    std::vector<double> solution(n, 0.0);
    auto SP = swiftware::hpp::ScheduleParams(-1, -1, 1, 1);

    for (auto _: state) {
        (void)_;
        std::fill(solution.begin(), solution.end(), 0.0);
        // Running the triangular solver
        swiftware::hpp::sptrsv_csr<double>(csr_matrix.values.data(), csr_matrix.col_indices.data(), csr_matrix.row_pointer.data(), solution.data(), rhs.data(), n, &SP);
        benchmark::DoNotOptimize(solution.data());
    }
    // test the solution vector to be all ones

}

static void BM_SPTRSV_OMP(benchmark::State &state) {
    auto mat_index = state.range(0);
    auto threadNo = state.range(1);
    std::vector<std::string> file_names;
    file_names.push_back("./data/1138_bus/1138_bus.mtx");
    file_names.push_back("./data/crystm01/crystm01.mtx");
    file_names.push_back("./data/minsurfo/minsurfo.mtx");
    file_names.push_back("./data/apache2/apache2.mtx");
    file_names.push_back("./data/test/test.mtx");
    state.SetLabel("Matrix:" + file_names[mat_index]);
  const std::string& file_name = file_names[mat_index];
  // Read and convert to CSR
  swiftware::hpp::Matrix<double> matrix = swiftware::hpp::readMatrixMarket<double>(file_name);
  swiftware::hpp::CSR<double> csr_matrix = swiftware::hpp::COO_to_CSR(matrix);

  // Build RHS such that x = 1 is the solution: b_i = sum_j L_ij
  int n = matrix.cols;
  std::vector<double> rhs(n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int idx = csr_matrix.row_pointer[i]; idx < csr_matrix.row_pointer[i+1]; ++idx) {
      rhs[i] += csr_matrix.values[idx];
    }
  }

  // Solution vector
  std::vector<double> solution(n, 0.0);
  auto SP = swiftware::hpp::ScheduleParams(-1, -1, static_cast<int>(threadNo), 1);

  for (auto _: state) {
    (void)_;
    std::fill(solution.begin(), solution.end(), 0.0);
    swiftware::hpp::sptrsv_csr_parallel<double>(csr_matrix.values.data(),
                          csr_matrix.col_indices.data(),
                          csr_matrix.row_pointer.data(),
                          solution.data(), rhs.data(), n, &SP);
    benchmark::DoNotOptimize(solution.data());
  }
}

#ifdef USE_MKL
static void BM_SPTRSV_MKL(benchmark::State &state)
{
    auto mat_index = state.range(0);
    auto threadNo = state.range(1);
    std::vector<std::string> file_names;
    file_names.push_back("./data/1138_bus/1138_bus.mtx");
    file_names.push_back("./data/crystm01/crystm01.mtx");
    file_names.push_back("./data/minsurfo/minsurfo.mtx");
    file_names.push_back("./data/apache2/apache2.mtx");
    file_names.push_back("./data/test/test.mtx");
    state.SetLabel("Matrix:" + file_names[mat_index]);

    const std::string& file_name = file_names[mat_index];
    // Read the matrix from the file
    swiftware::hpp::Matrix<double> matrix = swiftware::hpp::readMatrixMarket<double>(file_name);
    // Print the matrix
    // Convert the matrix to CSR format
    swiftware::hpp::CSR<double> csr_matrix = swiftware::hpp::COO_to_CSR(matrix);
    // build rhs
    int n = matrix.cols;
    std::vector<double> rhs(n);
    // TODO add anything needed

    // copy indices to MKL format
    std::vector<MKL_INT> mkl_row_ptr(csr_matrix.row_pointer.begin(), csr_matrix.row_pointer.end());
    std::vector<MKL_INT> mkl_col_indices(csr_matrix.col_indices.begin(), csr_matrix.col_indices.end());

    // Allocate memory for the solution vector
    std::vector<double> solution(n, 0.0f);
            // MKL SPTRSV setup
         sparse_matrix_t A;
         matrix_descr descr;
         descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
         descr.mode = SPARSE_FILL_MODE_LOWER;
         descr.diag = SPARSE_DIAG_NON_UNIT;

         mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, n, n,
                                 mkl_row_ptr.data(),
                                 mkl_row_ptr.data() + 1,
                                 mkl_col_indices.data(),
                                 csr_matrix.values.data());

         for (auto _: state) {
             // Running the triangular solver
             mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, rhs.data(), solution.data());
         }
    // test the solution vector to be all ones

}
#endif



BENCHMARK(BM_SPTRSV)->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), {1}})->Unit(benchmark::kMicrosecond)
  ->Iterations(1)
  ->Repetitions(5);

// Custom thread sweep: 1 plus powers of two up to 128
static void ApplyThreadSweep(benchmark::internal::Benchmark* b) {
  for (int m = 0; m <= 3; ++m) {
    for (int t : {1,2,4,8,16,32,64,128}) {
      b->Args({m, t});
    }
  }
}

BENCHMARK(BM_SPTRSV_OMP)->Apply(ApplyThreadSweep)->Unit(benchmark::kMicrosecond)
  ->Iterations(1)
  ->Repetitions(5);

#ifdef USE_MKL
BENCHMARK(BM_SPTRSV_MKL)->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), {1}})->Unit(benchmark::kMicrosecond)
  ->Iterations(1)
  ->Repetitions(5);
#endif

// Benchmark function to collect level-set timing data
static void BM_SPTRSV_OMP_LEVELSET(benchmark::State &state) {
    auto mat_index = state.range(0);
    auto threadNo = state.range(1);
    std::vector<std::string> file_names;
    file_names.push_back("./data/1138_bus/1138_bus.mtx");
    file_names.push_back("./data/crystm01/crystm01.mtx");
    file_names.push_back("./data/minsurfo/minsurfo.mtx");
    file_names.push_back("./data/apache2/apache2.mtx");
    
    std::vector<std::string> matrix_names = {"1138_bus", "crystm01", "minsurfo", "apache2"};
    state.SetLabel("Matrix:" + matrix_names[mat_index] + ":Threads:" + std::to_string(threadNo));
    
    const std::string& file_name = file_names[mat_index];
    swiftware::hpp::Matrix<double> matrix = swiftware::hpp::readMatrixMarket<double>(file_name);
    swiftware::hpp::CSR<double> csr_matrix = swiftware::hpp::COO_to_CSR(matrix);

    int n = matrix.cols;
    std::vector<double> rhs(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int idx = csr_matrix.row_pointer[i]; idx < csr_matrix.row_pointer[i+1]; ++idx) {
            rhs[i] += csr_matrix.values[idx];
        }
    }

    std::vector<double> solution(n, 0.0);
    auto SP = swiftware::hpp::ScheduleParams(-1, -1, static_cast<int>(threadNo), 1);
    SP.collect_level_times = true;

    for (auto _: state) {
        (void)_;
        std::fill(solution.begin(), solution.end(), 0.0);
        swiftware::hpp::sptrsv_csr_parallel<double>(csr_matrix.values.data(),
                              csr_matrix.col_indices.data(),
                              csr_matrix.row_pointer.data(),
                              solution.data(), rhs.data(), n, &SP);
        benchmark::DoNotOptimize(solution.data());
    }
    
    // Compute and report level-set statistics
    if (!SP.level_times.empty()) {
        double sum = std::accumulate(SP.level_times.begin(), SP.level_times.end(), 0.0);
        double mean = sum / SP.level_times.size();
        double min_time = *std::min_element(SP.level_times.begin(), SP.level_times.end());
        double max_time = *std::max_element(SP.level_times.begin(), SP.level_times.end());
        
        state.counters["num_levels"] = SP.level_times.size();
        state.counters["mean_level_time_us"] = mean * 1e6;
        state.counters["min_level_time_us"] = min_time * 1e6;
        state.counters["max_level_time_us"] = max_time * 1e6;
    }
}

BENCHMARK(BM_SPTRSV_OMP_LEVELSET)->Apply(ApplyThreadSweep)->Unit(benchmark::kMicrosecond)
  ->Iterations(1)
  ->Repetitions(1);

BENCHMARK_MAIN();
