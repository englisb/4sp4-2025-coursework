// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "gtest/gtest.h"
#include "n_body.h"
#include <vector>
#include <cmath>
#include <algorithm>

using namespace swiftware::hpp;

// Helper function to compare forces with tolerance
bool forces_match(const std::vector<double>& fx1, const std::vector<double>& fy1,
                  const std::vector<double>& fx2, const std::vector<double>& fy2,
                  double tolerance = 1e-10) {
    if (fx1.size() != fx2.size() || fy1.size() != fy2.size()) return false;
    
    for (size_t i = 0; i < fx1.size(); ++i) {
        double fx_diff = std::abs(fx1[i] - fx2[i]);
        double fy_diff = std::abs(fy1[i] - fy2[i]);
        double fx_rel = (std::abs(fx1[i]) > 1e-12) ? fx_diff / std::abs(fx1[i]) : fx_diff;
        double fy_rel = (std::abs(fy1[i]) > 1e-12) ? fy_diff / std::abs(fy1[i]) : fy_diff;
        
        if (fx_diff > tolerance && fx_rel > tolerance) return false;
        if (fy_diff > tolerance && fy_rel > tolerance) return false;
    }
    return true;
}

// Helper function to compare particles with tolerance
bool particles_match(const std::vector<Particle>& p1, const std::vector<Particle>& p2,
                     double tolerance = 1e-10) {
    if (p1.size() != p2.size()) return false;
    
    for (size_t i = 0; i < p1.size(); ++i) {
        if (std::abs(p1[i].x - p2[i].x) > tolerance) return false;
        if (std::abs(p1[i].y - p2[i].y) > tolerance) return false;
        if (std::abs(p1[i].vx - p2[i].vx) > tolerance) return false;
        if (std::abs(p1[i].vy - p2[i].vy) > tolerance) return false;
        if (std::abs(p1[i].mass - p2[i].mass) > tolerance) return false;
    }
    return true;
}

// Test: Generate random particles - vectorized matches scalar
TEST(NBodyVectorizedTest, GenerateRandomParticlesMatch) {
    const int N = 100;
    const unsigned int seed = 42;
    
    std::vector<Particle> particles_scalar;
    std::vector<Particle> particles_vectorized;
    
    generate_random_particles(particles_scalar, N, seed);
    generate_random_particles_vectorized(particles_vectorized, N, seed);
    
    ASSERT_EQ(particles_scalar.size(), N);
    ASSERT_EQ(particles_vectorized.size(), N);
    
    // Both should generate identical particles with same seed
    EXPECT_TRUE(particles_match(particles_scalar, particles_vectorized, 1e-12));
}

// Test: Generate random particles - non-zero mass
TEST(NBodyVectorizedTest, GenerateRandomParticlesNonZeroMass) {
    const int N = 50;
    std::vector<Particle> particles;
    
    generate_random_particles_vectorized(particles, N, 123);
    
    for (const auto& p : particles) {
        EXPECT_GT(p.mass, 1.0);  // Mass should be in [1, 11)
        EXPECT_LT(p.mass, 11.0);
    }
}

// Test: Calculate forces - vectorized matches scalar (small N)
TEST(NBodyVectorizedTest, CalculateForcesMatchSmall) {
    const int N = 10;
    const unsigned int seed = 100;
    
    std::vector<Particle> particles;
    generate_random_particles(particles, N, seed);
    
    std::vector<double> fx_scalar(N), fy_scalar(N);
    std::vector<double> fx_vectorized(N), fy_vectorized(N);
    
    calculate_forces(particles, fx_scalar, fy_scalar);
    calculate_forces_vectorized(particles, fx_vectorized, fy_vectorized);
    
    EXPECT_TRUE(forces_match(fx_scalar, fy_scalar, fx_vectorized, fy_vectorized, 1e-10));
}

// Test: Calculate forces - vectorized matches scalar (medium N)
TEST(NBodyVectorizedTest, CalculateForcesMatchMedium) {
    const int N = 50;
    const unsigned int seed = 200;
    
    std::vector<Particle> particles;
    generate_random_particles(particles, N, seed);
    
    std::vector<double> fx_scalar(N), fy_scalar(N);
    std::vector<double> fx_vectorized(N), fy_vectorized(N);
    
    calculate_forces(particles, fx_scalar, fy_scalar);
    calculate_forces_vectorized(particles, fx_vectorized, fy_vectorized);
    
    EXPECT_TRUE(forces_match(fx_scalar, fy_scalar, fx_vectorized, fy_vectorized, 1e-10));
}

// Test: Calculate forces - non-multiple of 4
TEST(NBodyVectorizedTest, CalculateForcesNonMultipleOf4) {
    const int N = 13;  // Not a multiple of 4
    const unsigned int seed = 300;
    
    std::vector<Particle> particles;
    generate_random_particles(particles, N, seed);
    
    std::vector<double> fx_scalar(N), fy_scalar(N);
    std::vector<double> fx_vectorized(N), fy_vectorized(N);
    
    calculate_forces(particles, fx_scalar, fy_scalar);
    calculate_forces_vectorized(particles, fx_vectorized, fy_vectorized);
    
    EXPECT_TRUE(forces_match(fx_scalar, fy_scalar, fx_vectorized, fy_vectorized, 1e-10));
}

// Test: Calculate forces - edge case with N=1
TEST(NBodyVectorizedTest, CalculateForcesSingleParticle) {
    const int N = 1;
    
    std::vector<Particle> particles = {{10.0, 20.0, 0.5, 0.5, 5.0}};
    
    std::vector<double> fx_scalar(N, 999.0), fy_scalar(N, 999.0);
    std::vector<double> fx_vectorized(N, 999.0), fy_vectorized(N, 999.0);
    
    calculate_forces(particles, fx_scalar, fy_scalar);
    calculate_forces_vectorized(particles, fx_vectorized, fy_vectorized);
    
    // Single particle should have zero force
    EXPECT_NEAR(fx_scalar[0], 0.0, 1e-12);
    EXPECT_NEAR(fy_scalar[0], 0.0, 1e-12);
    EXPECT_NEAR(fx_vectorized[0], 0.0, 1e-12);
    EXPECT_NEAR(fy_vectorized[0], 0.0, 1e-12);
}

// Test: Calculate forces - particles at same position (skip condition)
TEST(NBodyVectorizedTest, CalculateForcesSamePosition) {
    const int N = 5;
    
    std::vector<Particle> particles = {
        {0.0, 0.0, 0.0, 0.0, 5.0},
        {0.0, 0.0, 0.0, 0.0, 5.0},  // Same position as particle 0
        {10.0, 10.0, 0.0, 0.0, 5.0},
        {20.0, 20.0, 0.0, 0.0, 5.0},
        {30.0, 30.0, 0.0, 0.0, 5.0}
    };
    
    std::vector<double> fx_scalar(N), fy_scalar(N);
    std::vector<double> fx_vectorized(N), fy_vectorized(N);
    
    calculate_forces(particles, fx_scalar, fy_scalar);
    calculate_forces_vectorized(particles, fx_vectorized, fy_vectorized);
    
    EXPECT_TRUE(forces_match(fx_scalar, fy_scalar, fx_vectorized, fy_vectorized, 1e-10));
}

// Test: Update positions - vectorized matches scalar
TEST(NBodyVectorizedTest, UpdatePositionsMatch) {
    const int N = 20;
    const unsigned int seed = 400;
    const double dt = 0.01;
    
    std::vector<Particle> particles_scalar;
    std::vector<Particle> particles_vectorized;
    generate_random_particles(particles_scalar, N, seed);
    generate_random_particles(particles_vectorized, N, seed);
    
    std::vector<double> fx(N), fy(N);
    calculate_forces(particles_scalar, fx, fy);
    
    update_positions(particles_scalar, fx, fy, dt);
    update_positions_vectorized(particles_vectorized, fx, fy, dt);
    
    EXPECT_TRUE(particles_match(particles_scalar, particles_vectorized, 1e-12));
}

// Test: Full simulation - vectorized matches scalar
TEST(NBodyVectorizedTest, FullSimulationMatch) {
    const int N = 20;
    const int num_steps = 10;
    const double dt = 0.01;
    const unsigned int seed = 500;
    
    std::vector<Particle> particles_scalar;
    std::vector<Particle> particles_vectorized;
    generate_random_particles(particles_scalar, N, seed);
    generate_random_particles(particles_vectorized, N, seed);
    
    run_simulation(particles_scalar, num_steps, dt);
    run_simulation_vectorized(particles_vectorized, num_steps, dt);
    
    EXPECT_TRUE(particles_match(particles_scalar, particles_vectorized, 1e-9));
}

// Test: Performance characteristic - vectorized should process data correctly with large N
TEST(NBodyVectorizedTest, LargeScaleCorrectness) {
    const int N = 100;
    const unsigned int seed = 600;
    
    std::vector<Particle> particles;
    generate_random_particles(particles, N, seed);
    
    std::vector<double> fx_scalar(N), fy_scalar(N);
    std::vector<double> fx_vectorized(N), fy_vectorized(N);
    
    calculate_forces(particles, fx_scalar, fy_scalar);
    calculate_forces_vectorized(particles, fx_vectorized, fy_vectorized);
    
    // Allow slightly more tolerance for large N due to accumulation order differences
    EXPECT_TRUE(forces_match(fx_scalar, fy_scalar, fx_vectorized, fy_vectorized, 1e-9));
}

// Test: Newton's 3rd law - forces are equal and opposite
TEST(NBodyVectorizedTest, NewtonsThirdLaw) {
    const int N = 4;
    
    std::vector<Particle> particles = {
        {0.0, 0.0, 0.0, 0.0, 5.0},
        {10.0, 0.0, 0.0, 0.0, 5.0},
        {0.0, 10.0, 0.0, 0.0, 5.0},
        {10.0, 10.0, 0.0, 0.0, 5.0}
    };
    
    std::vector<double> fx(N), fy(N);
    calculate_forces_vectorized(particles, fx, fy);
    
    // Total force on system should be zero (Newton's 3rd law)
    double total_fx = 0.0, total_fy = 0.0;
    for (int i = 0; i < N; ++i) {
        total_fx += fx[i];
        total_fy += fy[i];
    }
    
    EXPECT_NEAR(total_fx, 0.0, 1e-10);
    EXPECT_NEAR(total_fy, 0.0, 1e-10);
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}