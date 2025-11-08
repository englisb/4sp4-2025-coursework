// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB03_TRIANGULAR_SOLVER_H
#define LAB03_TRIANGULAR_SOLVER_H
#include "def.h"

namespace swiftware::hpp
{
  template<typename T>
  void sptrsv_csr(T *val, int *col_ind, int *row_ptr, T *x, T *b, int n, ScheduleParams *SP) {
    // TODO: Simple sequential implementation of SpTRSV
  }


  // TODO: implement the parallel version.
  template<typename T>
  void sptrsv_csr_parallel(T *val, int *col_ind, int *row_ptr, T *x, T *b, int n, ScheduleParams *SP) {
    // TODO: Parallel implementation can be added here

  }



}

#endif //LAB03_TRIANGULAR_SOLVER_H