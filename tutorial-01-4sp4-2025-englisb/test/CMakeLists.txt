
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)


add_executable(vec_op_test
        ../test/vec_op_test.cpp
    ${CMAKE_SOURCE_DIR}/src/vec_op.cpp
)

target_link_libraries(vec_op_test
        GTest::gmock_main
)

target_include_directories(vec_op_test
PRIVATE ${CMAKE_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(vec_op_test)
