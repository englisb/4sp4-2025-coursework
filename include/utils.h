// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.
#ifndef PROJECT_DENSE_MATMUL_UTILS_H
#define PROJECT_DENSE_MATMUL_UTILS_H

#include <string>
#include "def.h"

namespace swiftware::hpp {

 //TODO add necessary includes



 // Do not change the following function signatures
 /// \brief Read a CSV file and store it in a DenseMatrix
 /// \param filename Path to the CSV file
 /// \param OutMat Pointer to the DenseMatrix to store the data
 /// \param removeFirstRow Whether to remove the first row of the CSV file
 DenseMatrix * readCSV(const std::string &filename, bool removeFirstRow = false);




}

#endif //PROJECT_DENSE_MATMUL_UTILS_H
