
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)


add_executable(heat_test
        ../test/heat_test.cpp
        ${CMAKE_SOURCE_DIR}/src/setup.cpp
        ${CMAKE_SOURCE_DIR}/src/stencil.cpp
        ${CMAKE_SOURCE_DIR}/src/pngwriter.c
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

target_link_libraries(heat_test
        GTest::gmock_main
)

target_include_directories(heat_test
        PRIVATE ${CMAKE_SOURCE_DIR}/include
)

if(HAVE_PNG)
    add_definitions(-DHAVE_PNG=1)
    find_package(PNG REQUIRED)
    target_link_libraries(heat_test
            PNG::PNG)
endif()

include(GoogleTest)
gtest_discover_tests(heat_test)