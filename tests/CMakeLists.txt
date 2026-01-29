# Find all .cpp files in this directory
file(GLOB TEST_SOURCES "*.cpp")

# Create an executable for each test file
foreach(source_file ${TEST_SOURCES})
  # Get the filename without extension (e.g., "test1" from "test1.cpp")
  get_filename_component(executable_name ${source_file} NAME_WE)

  # Create the executable
  add_executable(${executable_name} ${source_file})

  # Setup deal.II variables for this target
  deal_ii_setup_target(${executable_name})

  # Link against the core library (so we can test project classes) and deal.II
  # Note: logic relies on matrix_free_core being defined in the parent directory
  if(TARGET matrix_free_core)
    target_link_libraries(${executable_name} matrix_free_core ${DEAL_II_LIBRARIES})
  else()
    target_link_libraries(${executable_name} ${DEAL_II_LIBRARIES})
  endif()
endforeach()
