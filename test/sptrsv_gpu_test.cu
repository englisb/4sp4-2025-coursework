// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "gtest/gtest.h"
#include "triangular_solver.cuh"
#include <numeric>


TEST(SpTRSVGPUTest, SmallMatrix) {
    const size_t n = 1024;
    // TODO: Initialize a small sparse triangular matrix in CSR format

}

// TODO: add more tests and apply necessary changes

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}