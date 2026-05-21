// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab
// Distribution of the code is not
// allowed in any form without permission
// from SwiftWare Lab.



#include "gtest/gtest.h"
#include "heat.h"
#include <vector>
#include <cmath>

// Test that field_create properly initializes fields
TEST(HeatTest, FieldCreation) {
  field heat1, heat2;
  field_create(&heat1, &heat2, 100, 100);
  
  ASSERT_EQ(heat1.nx, 100);
  ASSERT_EQ(heat1.ny, 100);
  ASSERT_EQ(heat2.nx, 100);
  ASSERT_EQ(heat2.ny, 100);
  ASSERT_EQ(heat1.data.size(), 102 * 102);  // Including boundary layers
  ASSERT_EQ(heat2.data.size(), 102 * 102);
}

// Test that evolve maintains boundary conditions
TEST(HeatTest, BoundaryConditions) {
  field current, previous;
  field_create(&current, &previous, 50, 50);
  
  double a = 0.5;
  double dx2 = current.dx * current.dx;
  double dy2 = current.dy * current.dy;
  double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
  
  // Run one iteration
  evolve(&current, &previous, a, dt);
  
  // Check that boundary conditions are preserved (should not change)
  // Top boundary
  for (int j = 0; j < current.ny + 2; j++) {
    ASSERT_EQ(previous.data[j], current.data[j]);
  }
  
  // Bottom boundary
  for (int j = 0; j < current.ny + 2; j++) {
    ASSERT_EQ(previous.data[(current.nx + 1) * (current.ny + 2) + j], 
              current.data[(current.nx + 1) * (current.ny + 2) + j]);
  }
}

// Test that evolve and evolve_tiled produce the same results
TEST(HeatTest, TiledEquivalence) {
  field curr1, prev1, curr2, prev2;
  field_create(&curr1, &prev1, 100, 100);
  field_create(&curr2, &prev2, 100, 100);
  
  // Make sure both start with identical data
  std::copy(prev1.data.begin(), prev1.data.end(), prev2.data.begin());
  
  double a = 0.5;
  double dx2 = curr1.dx * curr1.dx;
  double dy2 = curr1.dy * curr1.dy;
  double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
  
  // Run both versions
  evolve(&curr1, &prev1, a, dt);
  evolve_tiled(&curr2, &prev2, a, dt);
  
  // Compare results - should be identical
  for (size_t i = 0; i < curr1.data.size(); i++) {
    ASSERT_NEAR(curr1.data[i], curr2.data[i], 1e-10) 
      << "Mismatch at index " << i;
  }
}

// Test that heat diffuses (average temperature changes over time)
TEST(HeatTest, HeatDiffusion) {
  field current, previous;
  field_create(&current, &previous, 100, 100);
  
  double initial_avg = field_average(&previous);
  
  double a = 0.5;
  double dx2 = current.dx * current.dx;
  double dy2 = current.dy * current.dy;
  double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
  
  // Run several iterations
  for (int iter = 0; iter < 10; iter++) {
    evolve(&current, &previous, a, dt);
    field_swap(&current, &previous);
  }
  
  double final_avg = field_average(&previous);
  
  // The average temperature should change due to boundary conditions
  ASSERT_NE(initial_avg, final_avg);
}

// Test that evolve_tiled maintains consistency over multiple iterations
TEST(HeatTest, TiledMultipleIterations) {
  field curr1, prev1, curr2, prev2;
  field_create(&curr1, &prev1, 100, 100);
  field_create(&curr2, &prev2, 100, 100);
  
  // Make sure both start with identical data
  std::copy(prev1.data.begin(), prev1.data.end(), prev2.data.begin());
  
  double a = 0.5;
  double dx2 = curr1.dx * curr1.dx;
  double dy2 = curr1.dy * curr1.dy;
  double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
  
  // Run multiple iterations with both versions
  for (int iter = 0; iter < 5; iter++) {
    evolve(&curr1, &prev1, a, dt);
    evolve_tiled(&curr2, &prev2, a, dt);
    field_swap(&curr1, &prev1);
    field_swap(&curr2, &prev2);
  }
  
  // Compare final results
  for (size_t i = 0; i < prev1.data.size(); i++) {
    ASSERT_NEAR(prev1.data[i], prev2.data[i], 1e-9) 
      << "Mismatch at index " << i << " after multiple iterations";
  }
}




int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}