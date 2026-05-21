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


project(lab02 C CXX)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# set the c++ standard to c++11
set( CMAKE_CXX_STANDARD 11 )



## Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)




# fetch google benchmark
option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)
#option(OPENMP "Enables OpenMP" ON)

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

add_executable(lab02-part1
        src/cache_size.cpp
        part1_bench.cpp
)
target_link_libraries(lab02-part1
        benchmark::benchmark
)



add_executable(lab02-part2
        src/utils.cpp
        src/pngwriter.c
        src/setup.cpp
        src/stencil.cpp
        part2_bench.cpp
)
target_link_libraries(lab02-part2
        benchmark::benchmark
)


if(HAVE_PNG)
    add_definitions(-DHAVE_PNG=1)
    find_package(PNG REQUIRED)

    target_link_libraries(lab02-part2
    PNG::PNG)
endif()


add_subdirectory(test)