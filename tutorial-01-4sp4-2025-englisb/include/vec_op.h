// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#ifndef BASICS_VEC_ADD_H
#define BASICS_VEC_ADD_H

#include <vector>

namespace swiftware::hpp {

   /// \brief a vector operation
   /// \param a The first vector
   /// \param b The second vector
   /// \param c The result vector
   void vec_op(std::vector<float> a, std::vector<float> b, std::vector<float>& c);
   void vec_op_custom(std::vector<float> a, std::vector<float> b, std::vector<float>& c);

  }
#endif //BASICS_VEC_ADD_H
