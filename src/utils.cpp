// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "utils.h"
#include <fstream>
#include <sstream>


namespace swiftware::hpp {

    //TODO add necessary includes

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


}