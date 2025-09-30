// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab

#include "sort_algorithms.h"
#include <algorithm> // for std::swap

namespace swiftware::hpp
{

  int partition(int *A, int low, int high)
  {
    int pivot = A[high];
    int i = low -1;
    for (int j = low; j < high; j++)
    {
      if (A[j] <= pivot)
      {
        i++;
        std::swap(A[i], A[j]);
      }
    }
    std::swap(A[i+1], A[high]);
    return i+1;
  }

  void quick_sort_recursive(int *A, int low, int high)
  {
    if (low < high)
    {
      int p = partition(A, low, high);
      quick_sort_recursive(A, low, p-1);
      quick_sort_recursive(A, p+1, high);
    }
  }

  void quick_sort(int *A, int n){
    quick_sort_recursive(A, 0, n-1);

  }

  void selection_sort(int *A, int n){
    for (int i = 0; i < n - 1; i++) {
      int min_idx = i;
      // Find the minimum element in the remaining unsorted array
      for (int j = i + 1; j < n; j++) {
        if (A[j] < A[min_idx]) {
          min_idx = j;
        }
      }
      // Swap the found minimum element with the first element
      if (min_idx != i) {
        std::swap(A[i], A[min_idx]);
      }
    }
  }

  void bubble_sort(int *A, int n){
    for (int i = 0; i < n - 1; i++) {
      bool swapped = false;
      // Last i elements are already in place
      for (int j = 0; j < n - i - 1; j++) {
        if (A[j] > A[j + 1]) {
          std::swap(A[j], A[j + 1]);
          swapped = true;
        }
      }
      // If no two elements were swapped, array is sorted
      if (!swapped) {
        break;
      }
    }
  }

}
