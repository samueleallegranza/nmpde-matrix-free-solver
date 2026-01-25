cmake_minimum_required(VERSION 3.12.0)
project(Project7_MatrixFree LANGUAGES CXX C)

include(cmake-common.cmake)

# Add the include directory to the search path
include_directories(include)

# Collect files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# Define the target first
add_executable(matrix_free src/main.cpp ${SOURCES} ${HEADERS})

# Setup the target with deal.II flags
deal_ii_setup_target(matrix_free)

# Link the libraries
target_link_libraries(matrix_free ${DEAL_II_LIBRARIES})
