// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "benchmark/benchmark.h"
#include "n_body.h"

#include <iostream>


#include "cholesky.h"


static void BM_CHOLESKY(benchmark::State &state) {
  auto m = state.range(0);

    // Cholesky allocation
    double **A = new double*[m]();
    for (int i = 0; i < m; i++) {
        A[i] = new double[m]();
    }
    double **L = new double*[m]();
    for (int i = 0; i < m; i++) {
        L[i] = new double[m]();
    }
    for (int i = 0; i < m; i++) {
        double diag = 0;
        for (int j = 0; j < m; j++) {
            A[i][j] = static_cast<double>(rand()) / RAND_MAX * 10.0;
            diag += A[i][j] * A[i][j];
        }
        A[i][i] = diag;
    }
    for (auto _: state){
        swiftware::hpp::cholesky_decomposition(A, L, m);
    }

    // Cholesky deallocation
    for (int i = 0; i < m; i++){
        delete[] A[i];
        delete[] L[i];
    }

}

static void BM_CHOLESKY_VEC(benchmark::State &state) {
  auto m = state.range(0);

  // Cholesky allocation
  double **A = new double*[m]();
  for (int i = 0; i < m; i++) {
    A[i] = new double[m]();
  }
  double **L = new double*[m]();
  for (int i = 0; i < m; i++) {
    L[i] = new double[m]();
  }

  for (int i = 0; i < m; i++) {
    double diag = 0;
    for (int j = 0; j < m; j++) {
      A[i][j] = static_cast<double>(rand()) / RAND_MAX * 10.0;
      diag += A[i][j] * A[i][j];
    }
    A[i][i] = diag;
  }
  for (auto _: state)
  {
    swiftware::hpp::cholesky_decomposition_vectorized(A, L, m);
  }

  // Cholesky deallocation
  for (int i = 0; i < m; i++)
  {
    delete[] A[i];
    delete[] L[i];
  }

}

// TODO: you will need another benchmark for the bonus part

static void BM_NBODY(benchmark::State &state) {
    auto m = state.range(0);

    const int N = 1000; // Number of particles
    const double dt = 0.01; // Time step
    const int num_steps = 100;

    std::vector<swiftware::hpp::Particle> particles(N);
    std::vector<double> fx(N), fy(N);
    swiftware::hpp::generate_random_particles(particles, N);


    for( auto _: state) {
      swiftware::hpp::run_simulation(particles, num_steps, dt);
    }
}

static void BM_NBODY_VEC(benchmark::State &state) {
  auto m = state.range(0);

  const int N = 1000; // Number of particles
  const double dt = 0.01; // Time step
  const int num_steps = 100;
  // TODO : benchmark vectorized N-Body version

  for( auto _: state) {

  }

}

#define MAXSIZE 512

BENCHMARK(BM_CHOLESKY)->ArgsProduct({benchmark::CreateRange(64, MAXSIZE, /*multi=*/2), {1}})
    ->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(10);
BENCHMARK(BM_CHOLESKY_VEC)->ArgsProduct({benchmark::CreateRange(64, MAXSIZE, /*multi=*/2), {1}})
    ->Unit(benchmark::kMicrosecond)->Iterations(1)->Repetitions(10);

BENCHMARK(BM_NBODY)->ArgsProduct({{1}, {1}})
    ->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(10);
BENCHMARK(BM_NBODY_VEC)->ArgsProduct({{1}, {1}})
    ->Unit(benchmark::kMillisecond)->Iterations(1)->Repetitions(10);





BENCHMARK_MAIN();

