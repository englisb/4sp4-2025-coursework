// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "gemv.h"
#include <gtest/gtest.h>
#include <cmath>


namespace swiftware::hpp {
    //TODO: Add more test

// Test basic gemv
TEST(GemvTest, BasicTest) {
    int m = 4, n = 4;
    float A[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float x[4] = {1,2,3,4};
    float y[4] = {0};
    
    ScheduleParams sp(2, 2);
    gemv(m, n, A, x, y, sp);
    
    EXPECT_NEAR(y[0], 1.0f, 1e-5);
    EXPECT_NEAR(y[1], 2.0f, 1e-5);
    EXPECT_NEAR(y[2], 3.0f, 1e-5);
    EXPECT_NEAR(y[3], 4.0f, 1e-5);
}



}

