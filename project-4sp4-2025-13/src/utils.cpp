// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "utils.h"
#include <cmath>
#include <fstream>
#include <sstream>

namespace swiftware::hpp {

// Helper function to apply tanh activation
void apply_tanh(float *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = std::tanh(data[i]);
  }
}

// Helper function to apply sigmoid activation
void apply_sigmoid(float *data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1.0f / (1.0f + std::exp(-data[i]));
  }
}

// Helper function to add bias vector to matrix (broadcasting)
void add_bias(float *matrix, const float *bias, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      matrix[i * cols + j] += bias[j];
    }
  }
}

// Helper function to find argmax for each row
void argmax_rows(const float *matrix, float *indices, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    int max_idx = 0;
    float max_val = matrix[i * cols];
    for (int j = 1; j < cols; j++) {
      if (matrix[i * cols + j] > max_val) {
        max_val = matrix[i * cols + j];
        max_idx = j;
      }
    }
    indices[i] = static_cast<float>(max_idx);
  }
}

// Do not change the following function signatures
DenseMatrix *readCSV(const std::string &filename, bool removeFirstRow) {
  std::ifstream file(filename);
  std::string line, word;
  // determine number of columns in file
  std::vector<std::string> lines;
  int cntr = 0;
  while (getline(file, line)) {
    lines.push_back(line);
  }
  if (removeFirstRow) {
    lines.erase(lines.begin());
  }
  std::vector<std::vector<float>> valuesPerLine(lines.size());
  for (int i = 0; i < lines.size(); i++) {
    std::stringstream lineStream(lines[i]);
    while (getline(lineStream, word, ',')) {
      valuesPerLine[i].push_back(std::stof(word));
    }
  }
  auto *OutMat = new DenseMatrix(valuesPerLine.size(), valuesPerLine[0].size());
  auto *data = OutMat->data.data();
  int ncol = OutMat->n;
  for (int i = 0; i < valuesPerLine.size(); i++) {
    size_t cols = valuesPerLine[i].size();
    for (uint j = 0; j < cols; j++) {
      data[i * ncol + j] = valuesPerLine[i][j];
    }
  }
  return OutMat;
}

CSR *denseToCSR(const DenseMatrix *dense) {
  if (dense == nullptr) {
    return nullptr;
  }

  int m = dense->m;
  int n = dense->n;
  auto *csr = new CSR(m, n);

  // Count non-zeros and build CSR arrays
  int nnz = 0;
  csr->Ap[0] = 0;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float val = dense->data[i * n + j];
      // Consider values as non-zero if they are not exactly zero
      // (using a small epsilon to handle floating point precision)
      if (val != 0.0f) {
        csr->Ai.push_back(j);
        csr->Ax.push_back(val);
        nnz++;
      }
    }
    csr->Ap[i + 1] = nnz;
  }

  csr->nnz = nnz;
  return csr;
}

CSR *loadPrunedWeightsCSR(const std::string &filename) {
  // Load the dense matrix from CSV
  DenseMatrix *dense = readCSV(filename, false);
  if (dense == nullptr) {
    return nullptr;
  }

  // Convert to CSR format
  CSR *csr = denseToCSR(dense);

  // Clean up the dense matrix
  delete dense;

  return csr;
}

} // namespace swiftware::hpp