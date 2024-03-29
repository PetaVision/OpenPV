# This CMakeLists.txt file bypasses pv_add_test because
# it uses a config file instead of setting the configuration
# on the command line, as pv_add_test has been assuming.
# TODO: cmake/PVAddTest.cmake needs to be config-file aware

set(SRC_CPP
  src/ConfigParserTest.cpp
)

get_filename_component(TEST_BASE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(TEST_BINARY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${TEST_BASE_NAME}")
set(TEST_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(TEST_CONFIG_FILE "input/config.txt")

include_directories("${TEST_TOP_DIR}/../Shared")
pv_add_executable(
  ${TEST_BASE_NAME}
  SRC ${SRC_CPP}
  OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
add_dependencies(${TEST_BASE_NAME} pv)
if (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
  set(TEST_SOURCE_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/input")
  if (EXISTS "${TEST_SOURCE_INPUT}")
    set(TEST_BINARY_INPUT "${CMAKE_CURRENT_BINARY_DIR}/input")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${TEST_SOURCE_INPUT} ${TEST_BINARY_INPUT})
  endif (EXISTS "${TEST_SOURCE_INPUT}")
endif()

set(TEST_NAME "${TEST_BASE_NAME}_1")
if(PV_MPI_SINGLE_PROCESS_TEST)
  set(PV_MPI_SINGLE_COMMAND "${MPIEXEC}")
  set(PV_MPI_SINGLE_OPTIONS "${MPIEXEC_NUMPROC_FLAG};1")
else()
  set(PV_MPI_SINGLE_COMMAND)
  set(PV_MPI_SINGLE_OPTIONS)
endif()
add_test(
  NAME ${TEST_NAME}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${PV_MPI_SINGLE_COMMAND} ${PV_MPI_SINGLE_OPTIONS} ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_CONFIG_FILE})
