#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "include/sparse_io.h"
#include "include/triangular_solver.cuh"
#include "include/def.h"
#include <cuda_runtime.h>
#include <cusparse.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

void compare_solutions(const std::string& matrix_file) {
    // Load matrix
    auto matrix = swiftware::hpp::readMatrixMarket<double>(matrix_file);
    auto csr_matrix = swiftware::hpp::COO_to_CSR(matrix);
    int n = matrix.cols;
    
    std::vector<double> rhs(n);
    std::vector<double> expected_solution;
    swiftware::hpp::build_rhs_for_triangular_solve(csr_matrix, rhs, expected_solution);
    
    // Allocate device memory
    double *d_val, *d_x_custom, *d_x_cusparse, *d_b;
    int *d_col_ind, *d_row_ptr, *d_ready;
    
    size_t nnz = csr_matrix.values.size();
    CUDA_CHECK(cudaMalloc(&d_val, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_col_ind, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x_custom, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x_cusparse, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ready, n * sizeof(int)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_val, csr_matrix.values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, csr_matrix.col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr_matrix.row_pointer.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, rhs.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Run custom kernel
    CUDA_CHECK(cudaMemset(d_x_custom, 0, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_ready, 0, n * sizeof(int)));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int iter = 0; iter < n; ++iter) {
        swiftware::hpp::sparse_csr_parallel_gpu<double><<<blocksPerGrid, threadsPerBlock>>>(
            d_val, d_col_ind, d_row_ptr, d_x_custom, d_b, n, d_ready
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Run cuSPARSE
    cusparseHandle_t cusparse;
    cusparseCreate(&cusparse);
    
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    
    int* d_row_csr = d_row_ptr;
    cusparseCreateCsr(&matA, n, n, nnz,
                     d_row_csr, d_col_ind, d_val,
                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    
    cusparseCreateDnVec(&vecX, n, d_b, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, n, d_x_cusparse, CUDA_R_64F);
    
    cusparseSpSVDescr_t spsv;
    cusparseSpSV_createDescr(&spsv);
    
    double alpha = 1.0;
    size_t bufferSize = 0;
    cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                           matA, vecX, vecY, CUDA_R_64F,
                           CUSPARSE_SPSV_ALG_DEFAULT, spsv, &bufferSize);
    
    void* dBuffer = nullptr;
    if (bufferSize > 0) CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    
    CUDA_CHECK(cudaMemset(d_x_cusparse, 0, n * sizeof(double)));
    cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                         matA, vecX, vecY, CUDA_R_64F,
                         CUSPARSE_SPSV_ALG_DEFAULT, spsv, dBuffer);
    
    cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                       matA, vecX, vecY, CUDA_R_64F,
                       CUSPARSE_SPSV_ALG_DEFAULT, spsv);
    
    // Copy results back
    std::vector<double> custom_solution(n);
    std::vector<double> cusparse_solution(n);
    
    CUDA_CHECK(cudaMemcpy(custom_solution.data(), d_x_custom, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cusparse_solution.data(), d_x_cusparse, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Compare solutions
    double max_error_custom = 0.0, max_error_cusparse = 0.0;
    double max_diff = 0.0;
    double avg_error_custom = 0.0, avg_error_cusparse = 0.0;
    double avg_diff = 0.0;
    
    for (int i = 0; i < n; ++i) {
        double err_custom = std::abs(custom_solution[i] - expected_solution[i]);
        double err_cusparse = std::abs(cusparse_solution[i] - expected_solution[i]);
        double diff = std::abs(custom_solution[i] - cusparse_solution[i]);
        
        max_error_custom = std::max(max_error_custom, err_custom);
        max_error_cusparse = std::max(max_error_cusparse, err_cusparse);
        max_diff = std::max(max_diff, diff);
        
        avg_error_custom += err_custom;
        avg_error_cusparse += err_cusparse;
        avg_diff += diff;
    }
    
    avg_error_custom /= n;
    avg_error_cusparse /= n;
    avg_diff /= n;
    
    // Print results
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Matrix: " << matrix_file << " (n=" << n << ", nnz=" << nnz << ")\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::setw(30) << "Metric" << std::setw(20) << "Custom Kernel" << std::setw(20) << "cuSPARSE" << "\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::setw(30) << "Max error vs expected:" << std::scientific << std::setw(20) << max_error_custom 
              << std::setw(20) << max_error_cusparse << "\n";
    std::cout << std::setw(30) << "Avg error vs expected:" << std::scientific << std::setw(20) << avg_error_custom 
              << std::setw(20) << avg_error_cusparse << "\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::setw(30) << "Max difference:" << std::scientific << std::setw(20) << max_diff << "\n";
    std::cout << std::setw(30) << "Avg difference:" << std::scientific << std::setw(20) << avg_diff << "\n";
    std::cout << std::string(80, '-') << "\n";
    
    bool custom_good = (max_error_custom < 1e-5);
    bool cusparse_good = (max_error_cusparse < 1e-5);
    bool match = (max_diff < 1e-5);
    
    std::cout << "Custom kernel correct (< 1e-5):  " << (custom_good ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "cuSPARSE correct (< 1e-5):       " << (cusparse_good ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "Solutions match (< 1e-5):        " << (match ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << std::string(80, '=') << "\n\n";
    
    // Cleanup
    if (dBuffer) CUDA_CHECK(cudaFree(dBuffer));
    cusparseSpSV_destroyDescr(spsv);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);
    cusparseDestroy(cusparse);
    
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_col_ind));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_x_custom));
    CUDA_CHECK(cudaFree(d_x_cusparse));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_ready));
}

int main() {
    std::vector<std::string> matrices = {
        "./data/1138_bus/1138_bus.mtx",
        "./data/crystm01/crystm01.mtx",
        "./data/minsurfo/minsurfo.mtx",
        "./data/apache2/apache2.mtx"
    };
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              GPU SpTRSV Correctness Verification Report                    ║\n";
    std::cout << "║                    Custom Kernel vs cuSPARSE                               ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n";
    
    for (const auto& mat : matrices) {
        compare_solutions(mat);
    }
    
    return 0;
}
