

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)


project(sptrsv_test LANGUAGES CXX CUDA)

add_executable(sptrsv_gpu_test
        sptrsv_gpu_test.cu
)

target_link_libraries(sptrsv_gpu_test
        GTest::gmock_main
)

target_include_directories(sptrsv_gpu_test
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)

# Set common compilation options for all targets
target_compile_features(sptrsv_gpu_test  PUBLIC cxx_std_17)

add_executable(sptrsv_omp_test
        sptrsv_openmp_test.cpp
)
target_link_libraries(sptrsv_omp_test
        GTest::gmock_main
)
target_include_directories(sptrsv_omp_test
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(sptrsv_omp_test  PUBLIC cxx_std_17)

if(OPENMP)
    target_link_libraries(sptrsv_omp_test OpenMP::OpenMP_CXX)
endif()

include(GoogleTest)
gtest_discover_tests(sptrsv_gpu_test)
gtest_discover_tests(sptrsv_omp_test)