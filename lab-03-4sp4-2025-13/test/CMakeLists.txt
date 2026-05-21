
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)


add_executable(cholesky_test
        ../test/cholesky_test.cpp
        ${CMAKE_SOURCE_DIR}/src/cholesky.cpp
)

target_link_libraries(cholesky_test
        GTest::gmock_main
)

target_include_directories(cholesky_test
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)

add_executable(nbody_test
        ../test/nbody_test.cpp
        ${CMAKE_SOURCE_DIR}/src/n_body.cpp
)

target_link_libraries(nbody_test
        GTest::gmock_main
)

target_include_directories(nbody_test
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)


include(GoogleTest)
gtest_discover_tests(cholesky_test)