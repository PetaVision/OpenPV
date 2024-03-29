# This CMakeLists.txt file bypasses pv_add_test because
# it uses a config file instead of setting the configuration
# on the command line, as pv_add_test has been assuming.
# TODO: cmake/PVAddTest.cmake needs to be config-file aware

if (PV_SYSTEM_TEST_MPIBLOCK)
  set(SRC_CPP
     src/NormProbeOutputterTest.cpp
  )

  get_filename_component(TEST_BASE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set(TEST_BINARY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${TEST_BASE_NAME}")
  set(TEST_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

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

  set(TEST_DESC "8")
  set(TEST_CONFIG_FILE "input/config.txt")
  set(TEST_NAME "${TEST_BASE_NAME}_${TEST_DESC}")
  add_test(
    NAME ${TEST_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 8 ${PV_MPI_OPTIONS_EXTRA} ${MPIEXEC_PREFLAGS} ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} "${TEST_CONFIG_FILE}")
endif (PV_SYSTEM_TEST_MPIBLOCK)
