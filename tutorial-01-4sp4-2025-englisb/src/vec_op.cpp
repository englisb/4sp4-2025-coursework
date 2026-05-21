// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include "vec_op.h"

namespace swiftware::hpp {

  void vec_op(std::vector<float> a, std::vector<float> b, std::vector<float> &c) {
   int n = a.size();
   c.resize(n);
   for (int i = 0; i < n; ++i) {
    c[i] = 0.5 * a[i] * b[i];
   }
  }

    void vec_op_custom(std::vector<float> a, std::vector<float> b, std::vector<float> &c) {
        int n = std::max(a.size(), b.size());
        a.resize(n);
        b.resize(n);
        c.resize(n);
        for (int i = n - 1; i >= 0; i--) {
            c[i] = 0.5 * a[i] * b[i];
        }
    }
}