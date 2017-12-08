# This CMakeLists.txt file bypasses pv_add_test because
# it uses a config file instead of setting the configuration
# on the command line, as pv_add_test has been assuming.
# TODO: cmake/PVAddTest.cmake needs to be config-file aware

if (PV_USE_MPI AND MPI_FOUND)
  set(SRC_CPP
    src/IndexLayer.cpp
    src/IndexWeightConn.cpp
    src/IndexWeightUpdater.cpp
    src/MtoNOutputStateTest.cpp
  )
  
  set(SRC_HPP
    src/IndexLayer.hpp
    src/IndexWeightConn.hpp
    src/IndexWeightUpdater.hpp
  )
  
  get_filename_component(TEST_NAME_BASE ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  set(TEST_BINARY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${TEST_NAME_BASE}")
  set(TEST_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
  set(TEST_CONFIG_FILE_BASE "input/config")
  
  include_directories("${TEST_TOP_DIR}/../Shared")
  pv_add_executable(${TEST_NAME_BASE}
    SRC ${SRC_CPP}
    OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
  add_dependencies(${TEST_NAME_BASE} pv)
  if (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    set(TEST_SOURCE_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/input")
    if (EXISTS "${TEST_SOURCE_INPUT}")
      set(TEST_BINARY_INPUT "${CMAKE_CURRENT_BINARY_DIR}/input")
      execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${TEST_SOURCE_INPUT} ${TEST_BINARY_INPUT})
    endif (EXISTS "${TEST_SOURCE_INPUT}")
  endif()

  set(TEST_NAME "${TEST_NAME_BASE}_multirowcol")
  set(TEST_CONFIG_FILE "${TEST_CONFIG_FILE_BASE}_multirowcol.txt")
  add_test(
    NAME ${TEST_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${MPIEXEC} -np 16 ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_CONFIG_FILE})

  set(TEST_NAME "${TEST_NAME_BASE}_multibatch")
  set(TEST_CONFIG_FILE "${TEST_CONFIG_FILE_BASE}_multibatch.txt")
  add_test(
    NAME ${TEST_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${MPIEXEC} -np 16 ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_CONFIG_FILE})
  
  set(TEST_NAME "${TEST_NAME_BASE}_multirow")
  set(TEST_CONFIG_FILE "${TEST_CONFIG_FILE_BASE}_multirow.txt")
  add_test(
    NAME ${TEST_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${MPIEXEC} -np 16 ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_CONFIG_FILE})
  
  set(TEST_NAME "${TEST_NAME_BASE}_multicolumn")
  set(TEST_CONFIG_FILE "${TEST_CONFIG_FILE_BASE}_multicolumn.txt")
  add_test(
    NAME ${TEST_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${MPIEXEC} -np 16 ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_CONFIG_FILE})
endif (PV_USE_MPI AND MPI_FOUND)
