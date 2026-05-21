
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)


add_executable(sort_test
        ../test/sort_test.cpp
        ${CMAKE_SOURCE_DIR}/src/sort_algorithms.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

target_link_libraries(sort_test
        GTest::gmock_main
)

target_include_directories(sort_test
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)

include(GoogleTest)
gtest_discover_tests(sort_test)