// Created by SwiftWare Lab on 9/25.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab

#ifndef LAB01_UTILS_H
#define LAB01_UTILS_H

#include <iostream>
#include <cstdlib>
namespace swiftware::hpp{
    /**
     * @brief A utility function to print an array.
     * @param A Pointer to the array.
     * @param n Size of the array.
     */
    void print_array(int *A, int n);

    /**
     * @brief A utility function to check if an array is sorted.
     * @param A Pointer to the array.
     * @param n Size of the array.
     * @return True if the array is sorted, false otherwise.
     */
    bool is_sorted(int *A, int n);

    /**
     * @brief A utility function to fill an array with random integers.
     * @param A Pointer to the array.
     * @param n Size of the array.
     * @param min Minimum value for the random integers.
     * @param max Maximum value for the random integers.
     * @return void
     */
    void fill_random(int *A, int n, int min = 0, int max = 1000);
}
#endif //LAB01_UTILS_H
