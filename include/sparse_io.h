// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB04_SPARSE_IO_H
#define LAB04_SPARSE_IO_H
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <complex>
#include <algorithm>

#include "def.h"


namespace swiftware::hpp {
  // Enum for different matrix properties
  enum class MatrixType { General, Symmetric, SkewSymmetric, Hermitian };
  enum class DataType { Real, Integer, Complex, Pattern };


  // A more formal data structure to hold the matrix
  template<typename T>
  struct Matrix {
    int rows;
    int cols;
    int non_zeros;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<T> values;
    MatrixType type;
  };

  template<typename T>
  struct CSR {
    int rows;
    int cols;
    int non_zeros;
    std::vector<int> row_pointer;
    std::vector<int> col_indices;
    std::vector<T> values;
    MatrixType type;
  };

  template<typename T>
  CSR<T> COO_to_CSR(const Matrix<T> &coo_matrix) {
    CSR<T> csr_matrix;
    // TODO : implement the conversion from COO to CSR
    return csr_matrix;
  }



// Function to parse the header line
  void parseHeader(const std::string& line, DataType& data_type, MatrixType& matrix_type) {
    std::stringstream ss(line);
    std::string token;
    ss >> token; // %%MatrixMarket
    ss >> token; // matrix
    ss >> token; // coordinate or array

    // Parse data type
    ss >> token;
    if (token == "real") data_type = DataType::Real;
    else if (token == "integer") data_type = DataType::Integer;
    else if (token == "complex") data_type = DataType::Complex;
    else if (token == "pattern") data_type = DataType::Pattern;
    else throw std::runtime_error("Unsupported data type in header.");

    // Parse matrix type
    ss >> token;
    if (token == "general") matrix_type = MatrixType::General;
    else if (token == "symmetric") matrix_type = MatrixType::Symmetric;
    else if (token == "skew-symmetric") matrix_type = MatrixType::SkewSymmetric;
    else if (token == "hermitian") matrix_type = MatrixType::Hermitian;
    else throw std::runtime_error("Unsupported matrix type in header.");
  }


  // Templated function to read the matrix
  template<typename T>
  Matrix<T> readMatrixMarket(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    DataType data_type;
    MatrixType matrix_type;
    bool header_found = false;

    // Skip comments and parse header
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      if (line[0] == '%') {
        if (line.substr(0, 14) == "%%MatrixMarket") {
          parseHeader(line, data_type, matrix_type);
          header_found = true;
        }
        continue;
      }

      // This should be the line with matrix dimensions
      std::stringstream ss(line);
      Matrix<T> matrix;
      ss >> matrix.rows >> matrix.cols >> matrix.non_zeros;
      matrix.type = matrix_type;

      if (std::is_same<T, std::complex<double> >::value) {
        if (data_type != DataType::Complex) {
          throw std::runtime_error("Expected complex data but file specifies a different type.");
        }
      }

      matrix.row_indices.reserve(matrix.non_zeros);
      matrix.col_indices.reserve(matrix.non_zeros);
      if (data_type != DataType::Pattern) {
        matrix.values.reserve(matrix.non_zeros);
      }

      int row, col;
      double real_val, imag_val;

      for (int i = 0; i < matrix.non_zeros; ++i) {
        if (file.eof()) {
          throw std::runtime_error("File ended prematurely.");
        }
        file >> row >> col;
        matrix.row_indices.push_back(row - 1);
        matrix.col_indices.push_back(col - 1);

        if (data_type == DataType::Real) {
          file >> real_val;
          if  (std::is_same<T, double>::value) {
            matrix.values.push_back(real_val);
          } else if (std::is_same<T, float>::value){
            matrix.values.push_back(static_cast<float>(real_val));
          } else {
            throw std::runtime_error("Type mismatch: Expected integer or complex, got real.");
          }
        } else if (data_type == DataType::Integer) {
          int int_val;
          file >> int_val;
          if (std::is_same<T, int>::value) {
            matrix.values.push_back(int_val);
          } else {
            throw std::runtime_error("Type mismatch: Expected real or complex, got integer.");
          }
        } else if (data_type == DataType::Complex) {
          file >> real_val >> imag_val;
          if (std::is_same<T, std::complex<double> >::value) {
            //matrix.values.push_back(std::complex<double>(real_val, imag_val));
          } else {
            throw std::runtime_error("Type mismatch: Expected real or integer, got complex.");
          }
        } else if (data_type == DataType::Pattern) {
          // No value to read, default to 1 for generic processing
          if (std::is_same<T, double>::value) {
            matrix.values.push_back(1.0);
          }
        }
      }
      file.close();
      return matrix;
    }

    if (!header_found) {
      throw std::runtime_error("Invalid Matrix Market file: header not found.");
    }
    throw std::runtime_error("File format error: Dimensions line not found.");
  }

  // TODO : implement necessary function


  template<typename T>
  void PrintCSR(const CSR<T>& csr) {
    std::cout << "CSR Matrix:" << std::endl;
    std::cout << "Rows: " << csr.rows << ", Cols: " << csr.cols << ", Non-zeros: " << csr.non_zeros << std::endl;
    std::cout << "Row Pointer: ";
    for (const auto& rp : csr.row_pointer) {
      std::cout << rp << " ";
    }
    std::cout << std::endl;
    std::cout << "Column Indices: ";
    for (const auto& ci : csr.col_indices) {
      std::cout << ci << " ";
    }
    std::cout << std::endl;
    std::cout << "Values: ";
    for (const auto& v : csr.values) {
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }

  template<typename T>
  void PrintMatrix(const Matrix<T>& matrix) {
    std::cout << "Matrix:" << std::endl;
    std::cout << "Rows: " << matrix.rows << ", Cols: " << matrix.cols << ", Non-zeros: " << matrix.non_zeros << std::endl;
    std::cout << "Row Indices: ";
    for(int i=0; i < matrix.non_zeros; i++ ){
      std::cout<< matrix.row_indices[i] << ","<<matrix.col_indices[i]
      <<","<<matrix.values[i]<<" \n ";
    }
    std::cout<< std::endl;
  }





}

#endif //LAB04_SPARSE_IO_H
