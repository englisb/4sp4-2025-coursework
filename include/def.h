// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.



#ifndef PROJECT_DENSE_MATMUL_DEF_H
#define PROJECT_DENSE_MATMUL_DEF_H

#include <vector>


namespace swiftware::hpp {

 struct ScheduleParams {
  int TileSize1;
  int TileSize2;
  ScheduleParams(int TileSize1, int TileSize2): TileSize1(TileSize1), TileSize2(TileSize2){}
 };

 // please do not change the following struct
 struct DenseMatrix{
  int m;
  int n;
  std::vector<float> data;
  DenseMatrix(int m, int n): m(m), n(n), data(m*n){}
 };


}

#endif //PROJECT_DENSE_MATMUL_DEF_H
