// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.
#ifndef PROJECT_DENSE_MATMUL_UTILS_H
#define PROJECT_DENSE_MATMUL_UTILS_H

#include "def.h"
#include <string>

namespace swiftware::hpp {

// TODO add necessary includes

// Helper functions for neural network operations
/// \brief Apply tanh activation function to data
/// \param data Pointer to data array
/// \param size Size of the data array
void apply_tanh(float *data, int size);

/// \brief Apply sigmoid activation function to data
/// \param data Pointer to data array
/// \param size Size of the data array
void apply_sigmoid(float *data, int size);

/// \brief Add bias vector to matrix (broadcasting)
/// \param matrix Pointer to matrix data (row-major)
/// \param bias Pointer to bias vector
/// \param rows Number of rows in matrix
/// \param cols Number of columns in matrix
void add_bias(float *matrix, const float *bias, int rows, int cols);

/// \brief Find argmax for each row of a matrix
/// \param matrix Pointer to matrix data (row-major)
/// \param indices Output array for argmax indices
/// \param rows Number of rows in matrix
/// \param cols Number of columns in matrix
void argmax_rows(const float *matrix, float *indices, int rows, int cols);

// Do not change the following function signatures
/// \brief Read a CSV file and store it in a DenseMatrix
/// \param filename Path to the CSV file
/// \param OutMat Pointer to the DenseMatrix to store the data
/// \param removeFirstRow Whether to remove the first row of the CSV file
DenseMatrix *readCSV(const std::string &filename, bool removeFirstRow = false);

/// \brief Convert a DenseMatrix to CSR format
/// \param dense DenseMatrix to convert
/// \return Pointer to CSR structure containing the sparse representation
CSR *denseToCSR(const DenseMatrix *dense);

/// \brief Load a pruned weight matrix from CSV and convert to CSR format
/// \param filename Path to the CSV file containing the pruned weight matrix
/// \return Pointer to CSR structure containing the sparse representation
CSR *loadPrunedWeightsCSR(const std::string &filename);

} // namespace swiftware::hpp

#endif // PROJECT_DENSE_MATMUL_UTILS_H
