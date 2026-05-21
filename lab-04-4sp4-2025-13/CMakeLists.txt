CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_LIST_DIR}/cmake
        "${CMAKE_CURRENT_LIST_DIR}/cmake/third_party"
        )
include(DownloadProject)

get_directory_property(HAS_PARENT PARENT_DIRECTORY)

# set top level variable
#set(GOOGLE_BENCHMARK_REQUIRED OFF CACHE BOOL "GBench is needed")
set(PROFILING_WITH_PAPI OFF CACHE BOOL "PAPI is needed")
set(PROFILING_WITH_LIKWID OFF CACHE BOOL "LIKWID is needed")


project(lab04 C CXX)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")



## Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)




# fetch google benchmark
option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)
option(OPENMP "Enables OpenMP" ON)

# Find OpenMP
if(OPENMP)
    find_package(OpenMP REQUIRED)
endif()

# Find Intel MKL
option(USE_MKL "Use Intel MKL for reference implementations" OFF)
if(USE_MKL)
    find_package(MKL CONFIG REQUIRED)
    message(STATUS "MKL found: ${MKL_ROOT}")
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
if (PROFILING_ENABLED)
    # enable PFM
    set(BENCHMARK_ENABLE_LIBPFM ON)
endif()

FetchContent_MakeAvailable(googlebenchmark)

# enable google test
set(BENCHMARK_ENABLE_TESTS ON)
set(BENCHMARK_ENABLE_GTEST_TESTS ON)


include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(lab04-cpu
        main_bench.cpp
)
target_link_libraries(lab04-cpu
        benchmark::benchmark
)


if(OPENMP)
    target_link_libraries(lab04-cpu OpenMP::OpenMP_CXX)
endif()

if(USE_MKL)
    target_link_libraries(lab04-cpu MKL::MKL)
    target_compile_definitions(lab04-cpu PRIVATE USE_MKL)
endif()

# GPU support
if(GPU_ENABLED)
    # Ensure CMAKE_CUDA_ARCHITECTURES is set (cannot be empty)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for")
    endif()

    # Enable CUDA language before setting CUDA-specific properties
    enable_language(CUDA OPTIONAL)
    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Use CPM to fetch NVBench's main branch.
    # CPM: https://github.com/cpm-cmake/CPM.cmake
    # NVBench: https://github.com/NVIDIA/nvbench
    include(cmake/CPM.cmake)
    CPMAddPackage("gh:NVIDIA/nvbench#main")

    add_executable(lab04-gpu
            main_bench.cu)

    target_link_libraries(lab04-gpu
            nvbench::main
            cusparse
    )


endif()

add_subdirectory(test)