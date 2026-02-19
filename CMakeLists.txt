#┌───────────────────────────────────────────────────────────────┐
#│                                                               │
#│   ▄▄▄  ▄    ▄        █             ▄        ▀             ▄   │
#│ ▄▀   ▀ ██  ██  ▄▄▄   █   ▄   ▄▄▄   █      ▄▄▄     ▄▄▄   ▄▄█▄▄ │
#│ █      █ ██ █ ▀   █  █ ▄▀   █▀  █  █        █    █   ▀    █   │
#│ █      █ ▀▀ █ ▄▀▀▀█  █▀█    █▀▀▀▀  █        █     ▀▀▀▄    █   │
#│  ▀▄▄▄▀ █    █ ▀▄▄▀█  █  ▀▄  ▀█▄▄▀  █▄▄▄▄▄ ▄▄█▄▄  ▀▄▄▄▀    ▀▄▄ │
#│                                                               │
#│                                                               │
#└───────────────────────────────────────────────────────────────┘

cmake_minimum_required(VERSION 3.12.0)
project(Project7_MatrixFree LANGUAGES CXX C)

include(cmake-common.cmake)

include_directories(include)
set(dir ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${dir}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${dir}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${dir}/bin")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-float-conversion -frounding-math -O3 -msse")

find_package(fmt)

#┌───────────────────────────────────────────────────────────────┐
#│                                                               │
#│ ▀▀█      ▀    █                             ▀                 │
#│   █    ▄▄▄    █▄▄▄    ▄ ▄▄   ▄▄▄    ▄ ▄▄  ▄▄▄     ▄▄▄    ▄▄▄  │
#│   █      █    █▀ ▀█   █▀  ▀ ▀   █   █▀  ▀   █    █▀  █  █   ▀ │
#│   █      █    █   █   █     ▄▀▀▀█   █       █    █▀▀▀▀   ▀▀▀▄ │
#│   ▀▄▄  ▄▄█▄▄  ██▄█▀   █     ▀▄▄▀█   █     ▄▄█▄▄  ▀█▄▄▀  ▀▄▄▄▀ │
#│                                                               │
#│                                                               │
#└───────────────────────────────────────────────────────────────┘
add_library(prm_handler include/prm_handler.hpp src/adr_prm_handler.cpp)
deal_ii_setup_target(prm_handler)
target_link_libraries(prm_handler fmt::fmt)

add_library(matrix_based_adr_lib include/adr_mb_problem.hpp src/adr_mb_problem.cpp)
deal_ii_setup_target(matrix_based_adr_lib)
target_link_libraries(matrix_based_adr_lib fmt::fmt prm_handler)

add_library(matrix_free_adr_lib include/adr_problem.hpp src/adr_problem.cpp include/adr_operator.hpp)
deal_ii_setup_target(matrix_free_adr_lib)
target_link_libraries(matrix_free_adr_lib fmt::fmt prm_handler)

#┌─────────────────────────────────────────────────────────────────────────────┐
#│                                                                             │
#│ ▄▄▄▄▄▄                               ▄           █      ▀▀█                 │
#│ █      ▄   ▄   ▄▄▄    ▄▄▄   ▄   ▄  ▄▄█▄▄   ▄▄▄   █▄▄▄     █     ▄▄▄    ▄▄▄  │
#│ █▄▄▄▄▄  █▄█   █▀  █  █▀  ▀  █   █    █    ▀   █  █▀ ▀█    █    █▀  █  █   ▀ │
#│ █       ▄█▄   █▀▀▀▀  █      █   █    █    ▄▀▀▀█  █   █    █    █▀▀▀▀   ▀▀▀▄ │
#│ █▄▄▄▄▄ ▄▀ ▀▄  ▀█▄▄▀  ▀█▄▄▀  ▀▄▄▀█    ▀▄▄  ▀▄▄▀█  ██▄█▀    ▀▄▄  ▀█▄▄▀  ▀▄▄▄▀ │
#│                                                                             │
#│                                                                             │
#└─────────────────────────────────────────────────────────────────────────────┘
add_executable(matrix_free_test
        tests/matrix_free_adr.cpp
        include/general_definitions.hpp
)
deal_ii_setup_target(matrix_free_test)
target_link_libraries(matrix_free_test prm_handler matrix_free_adr_lib fmt::fmt)


add_executable(matrix_based_test
        tests/matrix_based_adr.cpp
        include/general_definitions.hpp
)
deal_ii_setup_target(matrix_based_test)
target_link_libraries(matrix_based_test prm_handler matrix_based_adr_lib fmt::fmt)