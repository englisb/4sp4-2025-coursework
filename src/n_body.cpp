// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include <iostream>
#include <vector>
#include <cmath>

#include "n_body.h"



void swiftware::hpp::generate_random_particles(std::vector<Particle>& particles, int N, unsigned int seed) {
    srand(seed);
    particles.resize(N);
    for (int i = 0; i < N; ++i) {
        particles[i].x = static_cast<double>(rand()) / RAND_MAX * 100.0;
        particles[i].y = static_cast<double>(rand()) / RAND_MAX * 100.0;
        particles[i].vx = static_cast<double>(rand()) / RAND_MAX * 1.0;
        particles[i].vy = static_cast<double>(rand()) / RAND_MAX * 1.0;
        particles[i].mass = static_cast<double>(rand()) / RAND_MAX * 10.0 + 1.0; // Avoid zero mass
    }
}

  // Function to calculate the gravitational force between two particles
  void swiftware::hpp::calculate_forces(std::vector<swiftware::hpp::Particle> &particles,
                                        std::vector<double> &fx, std::vector<double> &fy) {
    const double G = 6.67430e-11; // Gravitational constant
    int N = particles.size();

    // Reset forces for this time step
    std::fill(fx.begin(), fx.end(), 0.0);
    std::fill(fy.begin(), fy.end(), 0.0);

    // Brute-force N^2 calculation of forces
    for (int i = 0; i < N; ++i) {
      for (int j = i + 1; j < N; ++j) {
        double dx = particles[j].x - particles[i].x;
        double dy = particles[j].y - particles[i].y;
        double dist_sq = dx * dx + dy * dy;
        double dist = std::sqrt(dist_sq);

        // Avoid division by zero for particles at the same position
        if (dist < 1e-4) continue;


        double force = G * particles[i].mass * particles[j].mass / dist_sq;

        fx[i] += force * dx / dist;
        fy[i] += force * dy / dist;

        fx[j] -= force * dx / dist;
        fy[j] -= force * dy / dist;
      }
    }
  }

  // Function to update particle positions and velocities using Verlet integration
  void swiftware::hpp::update_positions(std::vector<swiftware::hpp::Particle> &particles, const std::vector<double> &fx,
                                        const std::vector<double> &fy, double dt) {
    int N = particles.size();
    for (int i = 0; i < N; ++i) {
      double ax = fx[i] / particles[i].mass;
      double ay = fy[i] / particles[i].mass;

      // Update velocity (using Euler for simplicity)
      particles[i].vx += ax * dt;
      particles[i].vy += ay * dt;

      // Update position
      particles[i].x += particles[i].vx * dt;
      particles[i].y += particles[i].vy * dt;
    }
  }


void swiftware::hpp::run_simulation(std::vector<swiftware::hpp::Particle> &particles, int num_steps, double dt) {
  const int N = particles.size();
  std::vector<double> fx(N), fy(N);

  // Main simulation loop
  for (int step = 0; step < num_steps; ++step) {
    calculate_forces(particles, fx, fy);
    update_positions(particles, fx, fy, dt);

    // Optional: print positions periodically
//    if (step % 50 == 0) {
//      std::cout << "Step " << step << ": Particle 0 position = ("
//                << particles[0].x << ", " << particles[0].y << ")" << std::endl;
//    }
  }
}


// TODO: vectorized versions of your defined functions for N-Body simulation




