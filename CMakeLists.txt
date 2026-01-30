cmake_minimum_required(VERSION 3.12.0)
project(Project7_MatrixFree LANGUAGES CXX C)

include(cmake-common.cmake)

include_directories(include)
set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${dir}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${dir}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/bin")

file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

list(FILTER SOURCES EXCLUDE REGEX "main\\.cpp$")

add_library(matrix_free_core ${SOURCES} ${HEADERS})
deal_ii_setup_target(matrix_free_core)

add_executable(matrix_free src/main.cpp)
deal_ii_setup_target(matrix_free)

target_link_libraries(matrix_free matrix_free_core ${DEAL_II_LIBRARIES})

add_subdirectory(tests)

