// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab

#include "utils.h"


namespace swiftware::hpp {

  void print_array(int *A, int n) {
    for (int i = 0; i < n; ++i) {
      std::cout << A[i] << " ";
    }
    std::cout << std::endl;
  }

  bool is_sorted(int *A, int n) {
    for (int i = 1; i < n; ++i) {
      if (A[i - 1] > A[i]) {
        return false;
      }
    }
    return true;
  }

  void fill_random(int *A, int n, int min, int max) {
    for (int i = 0; i < n; ++i) {
      A[i] = min + rand() % (max - min + 1);
    }
  }


} // namespace swiftware::hpp