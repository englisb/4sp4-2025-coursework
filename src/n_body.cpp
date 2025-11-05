// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include <iostream>
#include <vector>
#include <cmath>
#include <immintrin.h>

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

void swiftware::hpp::generate_random_particles_vectorized(std::vector<Particle>& particles, int N,  unsigned int seed) {
  particles.resize(N);

  #ifdef USE_MKL
      // Use Intel MKL's VSL for vectorized random number generation
      VSLStreamStatePtr stream;
      vslNewStream(&stream, VSL_BRNG_MT19937, seed);
      
      std::vector<double> rand_buffer(N * 5); // x, y, vx, vy, mass
      vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, N * 5, rand_buffer.data(), 0.0, 1.0);
      
      for (int i = 0; i < N; ++i) {
          particles[i].x = rand_buffer[i * 5 + 0] * 100.0;
          particles[i].y = rand_buffer[i * 5 + 1] * 100.0;
          particles[i].vx = rand_buffer[i * 5 + 2] * 1.0;
          particles[i].vy = rand_buffer[i * 5 + 3] * 1.0;
          particles[i].mass = rand_buffer[i * 5 + 4] * 10.0 + 1.0;
      }
      
      vslDeleteStream(&stream);
  #else
      // Fallback to manual AVX vectorization with scalar rand()
      srand(seed);
      __m256d scale_100 = _mm256_set1_pd(100.0);
      __m256d scale_1 = _mm256_set1_pd(1.0);
      __m256d scale_10 = _mm256_set1_pd(10.0);
      __m256d rand_max_inv = _mm256_set1_pd(1.0 / RAND_MAX);

      int i = 0;
      for (; i + 4 <= N; i += 4) {
          double rand_x[4], rand_y[4], rand_vx[4], rand_vy[4], rand_mass[4];
          for (int k = 0; k < 4; ++k) {
              rand_x[k] = static_cast<double>(rand());
              rand_y[k] = static_cast<double>(rand());
              rand_vx[k] = static_cast<double>(rand());
              rand_vy[k] = static_cast<double>(rand());
              rand_mass[k] = static_cast<double>(rand());
          }

          __m256d x_vec = _mm256_loadu_pd(rand_x);
          __m256d y_vec = _mm256_loadu_pd(rand_y);
          __m256d vx_vec = _mm256_loadu_pd(rand_vx);
          __m256d vy_vec = _mm256_loadu_pd(rand_vy);
          __m256d mass_vec = _mm256_loadu_pd(rand_mass);

          x_vec = _mm256_mul_pd(_mm256_mul_pd(x_vec, rand_max_inv), scale_100);
          y_vec = _mm256_mul_pd(_mm256_mul_pd(y_vec, rand_max_inv), scale_100);
          vx_vec = _mm256_mul_pd(_mm256_mul_pd(vx_vec, rand_max_inv), scale_1);
          vy_vec = _mm256_mul_pd(_mm256_mul_pd(vy_vec, rand_max_inv), scale_1);
          mass_vec = _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(mass_vec, rand_max_inv), scale_10), scale_1);

          double x_out[4], y_out[4], vx_out[4], vy_out[4], mass_out[4];
          _mm256_storeu_pd(x_out, x_vec);
          _mm256_storeu_pd(y_out, y_vec);
          _mm256_storeu_pd(vx_out, vx_vec);
          _mm256_storeu_pd(vy_out, vy_vec);
          _mm256_storeu_pd(mass_out, mass_vec);

          for (int k = 0; k < 4; ++k) {
              particles[i + k].x = x_out[k];
              particles[i + k].y = y_out[k];
              particles[i + k].vx = vx_out[k];
              particles[i + k].vy = vy_out[k];
              particles[i + k].mass = mass_out[k];
          }
      }

      for (; i < N; ++i) {
          particles[i].x = static_cast<double>(rand()) / RAND_MAX * 100.0;
          particles[i].y = static_cast<double>(rand()) / RAND_MAX * 100.0;
          particles[i].vx = static_cast<double>(rand()) / RAND_MAX * 1.0;
          particles[i].vy = static_cast<double>(rand()) / RAND_MAX * 1.0;
          particles[i].mass = static_cast<double>(rand()) / RAND_MAX * 10.0 + 1.0;
      }
  #endif
}

void swiftware::hpp::calculate_forces_vectorized(std::vector<swiftware::hpp::Particle>& particles, std::vector<double>& fx, std::vector<double>& fy) {
  const double G = 6.67430e-11; // Gravitational constant
  int N = particles.size(), i, j;

  // Reset forces for this time step
  std::fill(fx.begin(), fx.end(), 0.0);
  std::fill(fy.begin(), fy.end(), 0.0);

  const int vec_length = 4; // AVX processes 4 doubles at a time

  __m256d G_vec = _mm256_set1_pd(G);
  __m256d eps_vec = _mm256_set1_pd(1e-4);
  __m256d dx_vec, x_vec, xi_vec, dy_vec, y_vec, yi_vec, dist_sq_vec, dist_vec, dist_vec_safe, inv_dist_vec;
  __m256d force_vec, fx_vec, fy_vec;

  // Iterate all i; inner loop handles vector chunks + scalar tail
  for (int i = 0; i < N; ++i) {
    xi_vec = _mm256_set1_pd(particles[i].x);
    yi_vec = _mm256_set1_pd(particles[i].y);
    for (j = i + 1; j + vec_length <= N; j += vec_length) {
        x_vec = _mm256_set_pd(particles[j + 3].x, particles[j + 2].x, particles[j + 1].x, particles[j].x);
        y_vec = _mm256_set_pd(particles[j + 3].y, particles[j + 2].y, particles[j + 1].y, particles[j].y);
        dx_vec = _mm256_sub_pd(x_vec, xi_vec);
        dy_vec = _mm256_sub_pd(y_vec, yi_vec);
        dist_sq_vec = _mm256_add_pd(_mm256_mul_pd(dx_vec, dx_vec), _mm256_mul_pd(dy_vec, dy_vec));
        dist_vec = _mm256_sqrt_pd(dist_sq_vec);

        // Create mask for valid distances (>= eps)
        __m256d valid_mask = _mm256_cmp_pd(dist_vec, eps_vec, _CMP_GE_OQ);
        
        // Compute force only for valid lanes; zero out invalid lanes
        __m256d mass_j_vec = _mm256_set_pd(particles[j + 3].mass, particles[j + 2].mass, particles[j + 1].mass, particles[j].mass);
        force_vec = _mm256_mul_pd(G_vec, _mm256_mul_pd(_mm256_set1_pd(particles[i].mass), mass_j_vec));
        force_vec = _mm256_div_pd(force_vec, dist_sq_vec);  // May produce INF/NaN for zero lanes; masked next
        
        // Directional force components: force * (dx/dist) and force * (dy/dist)
        // Compute for all lanes, then zero out invalid ones with mask
        inv_dist_vec = _mm256_div_pd(force_vec, dist_vec);
        fx_vec = _mm256_mul_pd(inv_dist_vec, dx_vec);
        fy_vec = _mm256_mul_pd(inv_dist_vec, dy_vec);
        
        // Apply mask to directional forces: zero out invalid lanes (mimics skip behavior)
        fx_vec = _mm256_and_pd(fx_vec, valid_mask);
        fy_vec = _mm256_and_pd(fy_vec, valid_mask);

        // Horizontal sum to accumulate forces on particle i
        // Hadd reduces [a,b,c,d] -> [a+b, c+d, a+b, c+d], then extract and sum
        __m256d fx_sum_vec = _mm256_hadd_pd(fx_vec, fx_vec);  // [fx0+fx1, fx2+fx3, fx0+fx1, fx2+fx3]
        __m256d fy_sum_vec = _mm256_hadd_pd(fy_vec, fy_vec);
        // Extract low and high 128-bit lanes and add
        __m128d fx_low = _mm256_castpd256_pd128(fx_sum_vec);
        __m128d fx_high = _mm256_extractf128_pd(fx_sum_vec, 1);
        __m128d fy_low = _mm256_castpd256_pd128(fy_sum_vec);
        __m128d fy_high = _mm256_extractf128_pd(fy_sum_vec, 1);
        __m128d fx_sum128 = _mm_add_pd(fx_low, fx_high);
        __m128d fy_sum128 = _mm_add_pd(fy_low, fy_high);
        // Extract scalar sum (element 0 contains the sum)
        fx[i] += _mm_cvtsd_f64(fx_sum128);
        fy[i] += _mm_cvtsd_f64(fy_sum128);

        // Apply Newton's 3rd law to the j particles, per lane
        // Extract directional forces to apply to each j particle
        double fx_tmp[4], fy_tmp[4];
        _mm256_storeu_pd(fx_tmp, fx_vec);
        _mm256_storeu_pd(fy_tmp, fy_vec);
        for (int k = 0; k < vec_length; ++k) {
          fx[j + k] -= fx_tmp[k];
          fy[j + k] -= fy_tmp[k];
        }
    }
    
    // Handle remaining particles
    for (; j < N; ++j) {
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

void swiftware::hpp::update_positions_vectorized(std::vector<Particle>& particles, const std::vector<double>& fx, const std::vector<double>& fy, double dt) {
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

void swiftware::hpp::run_simulation_vectorized(std::vector<swiftware::hpp::Particle> &particles, int num_steps, double dt) {
    const int N = particles.size();
    std::vector<double> fx(N), fy(N);

    // Main simulation loop
    for (int step = 0; step < num_steps; ++step) {
        calculate_forces_vectorized(particles, fx, fy);
        update_positions_vectorized(particles, fx, fy, dt);

        // Optional: print positions periodically
        if (step % 50 == 0) {
            std::cout << "Step " << step << ": Particle 0 position = ("
                      << particles[0].x << ", " << particles[0].y << ")" << std::endl;
        }
    }
}
