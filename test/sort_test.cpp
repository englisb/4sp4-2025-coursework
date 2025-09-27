// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab


#include "gtest/gtest.h"
#include "sort_algorithms.h"
#include <vector>

// Test cases for selection sort algorithm

TEST(SelectionSortTest, SortRandomArray) {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int expected[] = {11, 12, 22, 25, 34, 64, 90};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    swiftware::hpp::selection_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(arr[i], expected[i]);
    }
}

TEST(SelectionSortTest, SortAlreadySortedArray) {
    int arr[] = {1, 2, 3, 4, 5};
    int expected[] = {1, 2, 3, 4, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    swiftware::hpp::selection_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(arr[i], expected[i]);
    }
}

TEST(SelectionSortTest, SortReverseSortedArray) {
    int arr[] = {9, 7, 5, 3, 1};
    int expected[] = {1, 3, 5, 7, 9};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    swiftware::hpp::selection_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(arr[i], expected[i]);
    }
}

// Test cases for bubble sort algorithm

TEST(BubbleSortTest, SortArrayWithDuplicates) {
    int arr[] = {5, 2, 8, 2, 9, 1, 5, 4};
    int expected[] = {1, 2, 2, 4, 5, 5, 8, 9};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    swiftware::hpp::bubble_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(arr[i], expected[i]);
    }
}

TEST(BubbleSortTest, SortSingleElementArray) {
    int arr[] = {42};
    int expected[] = {42};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    swiftware::hpp::bubble_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(arr[i], expected[i]);
    }
}

TEST(BubbleSortTest, SortNegativeNumbers) {
    int arr[] = {-3, -1, -7, -2, -5};
    int expected[] = {-7, -5, -3, -2, -1};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    swiftware::hpp::bubble_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(arr[i], expected[i]);
    }
}

// Test cases for quick sort algorithm

TEST(QuickSortTest, SortAllIdenticalElements) {
    int arr[] = {7, 7, 7, 7, 7, 7};
    int expected[] = {7, 7, 7, 7, 7, 7};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    swiftware::hpp::quick_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(arr[i], expected[i]);
    }
}

TEST(QuickSortTest, SortWorstCaseScenario) {
    int arr[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    int expected[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    swiftware::hpp::quick_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(arr[i], expected[i]);
    }
}

TEST(QuickSortTest, SortMixedPositiveNegativeZero) {
    int arr[] = {10, -5, 0, 3, -12, 8, -1};
    int expected[] = {-12, -5, -1, 0, 3, 8, 10};
    int n = sizeof(arr) / sizeof(arr[0]);
    
    swiftware::hpp::quick_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(arr[i], expected[i]);
    }
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}