// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab
// Distribution of the code is not
// allowed in any form without permission
// from SwiftWare Lab.


#include "benchmark/benchmark.h"
#include "heat.h"




static void BM_STENCIL(benchmark::State &state) {
  int rows = state.range(0);
  int cols = state.range(0);
  int nsteps = state.range(1);

  // Set up the solver
  field current, previous;
  field_create(&current, &previous, rows, cols);

  // Output the initial field and its temperature
  field_write(&current, 0);
  double average_temp = field_average(&current);
  printf("Average temperature, start: %f\n", average_temp);

  // Set diffusivity constant
  double a = 0.5;
  // Compute the largest stable time step
  double dx2 = current.dx * current.dx;
  double dy2 = current.dy * current.dy;
  double dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
  // Set output interval
  int output_interval = 1500;

  for (auto _: state)
  {
    // Time evolution
    for (int iter = 1; iter <= nsteps; iter++) {
      evolve(&current, &previous, a, dt);
      if (iter % output_interval == 0) {
        field_write(&current, iter);
      }
      // Swap current and previous fields for next iteration step
      field_swap(&current, &previous);
    }
  }

  // Output the final field and its temperature
  average_temp = field_average(&previous);
  //printf("Average temperature at end: %f\n", average_temp);
  // Compare temperature for reference

  field_write(&previous, nsteps);
}



BENCHMARK(BM_STENCIL)->ArgsProduct({{2500}, {500}})
  ->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(1);



BENCHMARK_MAIN();
