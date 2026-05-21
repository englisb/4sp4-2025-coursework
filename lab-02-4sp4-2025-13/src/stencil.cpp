// (c) 2023 ENCCS, CSC and the contributors

// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab
// Distribution of the code is not
// allowed in any form without permission
// from SwiftWare Lab.


#include "heat.h"

// Update the temperature values using five-point stencil
// Arguments:
//   curr: current temperature values
//   prev: temperature values from previous time step
//   a: diffusivity
//   dt: time step
void evolve(field *curr, field *prev, double a, double dt)
{
  // Compute heat equation using five-point stencil
  // Loop over interior points (excluding boundary layers)
  double dx2 = prev->dx * prev->dx;
  double dy2 = prev->dy * prev->dy;
  
  for (int i = 1; i <= prev->nx; i++) {
    for (int j = 1; j <= prev->ny; j++) {
      int ind = i * (prev->ny + 2) + j;
      
      // Get the five-point stencil values
      double center = prev->data[ind];
      double right = prev->data[ind + 1];
      double left = prev->data[ind - 1];
      double up = prev->data[ind - (prev->ny + 2)];
      double down = prev->data[ind + (prev->ny + 2)];
      
      // Apply the five-point stencil formula
      curr->data[ind] = center + a * dt * (
        (left - 2.0 * center + right) / dx2 +
        (up - 2.0 * center + down) / dy2
      );
    }
  }
}


void evolve_tiled(field *curr, field *prev, double a, double dt, int tile_i, int tile_j)
{
  // Optimized tiled stencil with better cache locality and data reuse
  const double dx2 = prev->dx * prev->dx;
  const double dy2 = prev->dy * prev->dy;
  const double dt_dx2 = a * dt / dx2;
  const double dt_dy2 = a * dt / dy2;
  
  const int ny_stride = prev->ny + 2;
  const int TILE_I = tile_i;
  const int TILE_J = tile_j;
  
  // Tile the computation for better cache locality
  for (int ii = 1; ii <= prev->nx; ii += TILE_I) {
    const int i_end = (ii + TILE_I <= prev->nx) ? ii + TILE_I : prev->nx + 1;
    
    for (int jj = 1; jj <= prev->ny; jj += TILE_J) {
      const int j_end = (jj + TILE_J <= prev->ny) ? jj + TILE_J : prev->ny + 1;
      
      // Process tile with optimized memory access pattern
      for (int i = ii; i < i_end; i++) {
        const double* __restrict__ prev_row_up = &prev->data[(i-1) * ny_stride + jj];
        const double* __restrict__ prev_row = &prev->data[i * ny_stride + jj];
        const double* __restrict__ prev_row_down = &prev->data[(i+1) * ny_stride + jj];
        double* __restrict__ curr_row = &curr->data[i * ny_stride + jj];
        
        // Vectorizable inner loop with linear memory access
        for (int j = jj; j < j_end; j++) {
          const int local_j = j - jj;
          const double center = prev_row[local_j];
          const double left = prev_row[local_j - 1];
          const double right = prev_row[local_j + 1];
          const double up = prev_row_up[local_j];
          const double down = prev_row_down[local_j];
          
          // Optimized stencil computation
          curr_row[local_j] = center + dt_dx2 * (left + right - 2.0 * center) + 
                                     dt_dy2 * (up + down - 2.0 * center);
        }
      }
    }
  }
}