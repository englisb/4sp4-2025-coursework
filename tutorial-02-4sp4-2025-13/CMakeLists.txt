# See apt.kitware.com for updated CMake versions on Ubuntu.
cmake_minimum_required(VERSION 3.20)

project(tut02 CUDA CXX)

# Use CPM to fetch NVBench's main branch.
# CPM: https://github.com/cpm-cmake/CPM.cmake
# NVBench: https://github.com/NVIDIA/nvbench
include(cmake/CPM.cmake)
CPMAddPackage("gh:NVIDIA/nvbench#main")

# Ensure CMAKE_CUDA_ARCHITECTURES is set (cannot be empty)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for")
endif()


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx")
include(FetchContent)


FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4
)
# enable BENCHMARK_DOWNLOAD_DEPENDENCIES
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)

# enable google test
set(BENCHMARK_ENABLE_TESTS ON)
set(BENCHMARK_ENABLE_GTEST_TESTS ON)


FetchContent_MakeAvailable(googlebenchmark)



include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(tut02 main_bench.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
)
target_link_libraries(tut02
        nvbench::main)


add_subdirectory(test)
