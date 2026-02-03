cmake_minimum_required(VERSION 3.12.0)
project(Project7_MatrixFree LANGUAGES CXX C)

include(cmake-common.cmake)

include_directories(include)
set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${dir}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${dir}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/bin")

add_executable(matrix_free_test
        tests/matrix_free_adr.cpp
        include/adr_problem.hpp
        include/adr_operator.hpp
        include/default_coefficient.hpp
        include/general_definitions.hpp
)
deal_ii_setup_target(matrix_free_test)
