

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)


project(muladd_op_test LANGUAGES CXX CUDA)

add_executable(muladd_op_test
        muladd_op_test.cu
)

target_link_libraries(muladd_op_test
        GTest::gmock_main
)

target_include_directories(muladd_op_test
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)

# Set common compilation options for all targets
target_compile_features(muladd_op_test  PUBLIC cxx_std_17)

include(GoogleTest)
gtest_discover_tests(muladd_op_test)