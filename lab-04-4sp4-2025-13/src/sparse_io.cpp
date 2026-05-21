// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <complex>
#include "sparse_io.h"

namespace swiftware::hpp {

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



}