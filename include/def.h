// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB04_DEF_H
#define LAB04_DEF_H

#include <vector>


namespace swiftware::hpp {

  struct ScheduleParams {
    int NumThreads;
    int ChunkSize;
    // TODO: Add more parameters if needed
    int TileSize1;
    int TileSize2;
    int NumWF;
    std::vector<int> WFPointer, WFIterations;
    ScheduleParams(int TileSize1, int TileSize2, int NT, int CS):
      TileSize1(TileSize1), TileSize2(TileSize2), NumThreads(NT), ChunkSize(CS){}
  };

  // please do not change below lines
  struct DenseMatrix{
    int m;
    int n;
    std::vector<float> data;
    DenseMatrix(int m, int n): m(m), n(n), data(m*n){}
  };



}

#endif //LAB04_DEF_H
