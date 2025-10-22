// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB03_N_BODY_H
#define LAB03_N_BODY_H

namespace swiftware::hpp
{

    // Define a structure for a particle
    struct Particle {
        double x, y; // Position
        double vx, vy; // Velocity
        double mass; // Mass
    };

    // generate random particles
    void generate_random_particles(std::vector<Particle>& particles, int N,  unsigned int seed=10);

    // Function to calculate the gravitational force between two particles
    void calculate_forces(std::vector<swiftware::hpp::Particle>& , std::vector<double>& , std::vector<double>& );

    // Function to update particle positions and velocities using Verlet integration
    void update_positions(std::vector<Particle>& , const std::vector<double>& , const std::vector<double>& , double );

    void run_simulation(std::vector<swiftware::hpp::Particle> &particles, int num_steps, double dt);


    /// TODO: define necessary functions for the vectorized version



}

#endif //LAB03_N_BODY_H