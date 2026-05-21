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


project(lab03 C CXX)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# set the c++ standard to c++11
set( CMAKE_CXX_STANDARD 11 )



## Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)




# fetch google benchmark
option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)
#option(OPENMP "Enables OpenMP" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
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

# Find Intel MKL
option(USE_MKL "Use Intel MKL for reference implementations" OFF)
if(USE_MKL)
    find_package(MKL CONFIG REQUIRED)
    message(STATUS "MKL found: ${MKL_ROOT}")
endif()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(lab03
        src/cholesky.cpp
        src/n_body.cpp

        main_bench.cpp
)
target_link_libraries(lab03
        benchmark::benchmark
)
if(USE_MKL)
    target_link_libraries(lab03 MKL::MKL)
    target_compile_definitions(lab03 PRIVATE USE_MKL)
endif()
add_subdirectory(test)

