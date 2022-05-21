# pv_add_test([PARAMS | NO_PARAMS] [MPI_ONLY | NO_MPI] [MIN_MPI_COPIES] [MAX_MPI_COPIES] [FLAGS] [BASE_NAME] [SRCFILES])
#
# All arguments except SRCFILES are optional.
# SRCFILES and PARAMS each take one or more string arguments.
# MIN_MPI_COPIES and MAX_MPI_COPIES each take one numerical argument.
# FLAGS and BASE_NAME each take one string argument.
#
# Add a test case for the OpenPV library. This creates the binary and adds
# test tests. Multiple test cases are added by default.
# --------------------------------------------------------------------------------
# Example CMakeLists.txt files for a PVSystemTest:
#
# Most files will have a single line:
#
#   pv_add_test(SRCFILES main.cpp file1.cpp file2.cpp file1.hpp file2.hpp)
# 
#   This will add a test with a base name based on the test's directory name. The params file
#   will be named input/directory_name.params
#
#   This test will use 1,2, and 4 MPI copies
#
# No params files:
#
#   pv_add_test(NO_PARAMS SRCFILES main.cpp)
#
# Alternate param files:
#
#   pv_add_test(PARAMS test_kernel test_kernel_normalizepost test_kernel_normalizepost_shrunken SRCFILES main.cpp)
#
# No params files, no MPI:
#
#   pv_add_test(NO_PARAMS NO_MPI SRCFILES main.cpp)
#
# Append special flags to the command line
#
#   pv_add_test(FLAGS "-c checkpoints/Checkpoint06 --testall" SRCFILES main.cpp)
#
# Specify the number of MPI copies, only test MPI, and special flags
#
#   pv_add_test(MIN_MPI_COPIES 4 MPI_ONLY FLAGS "-batchwidth 4" SRCFILES main.cpp)
#
# --------------------------------------------------------------------------------
# Options:
#
# PARAMS - by default, this is set to BASE_NAME and a params file of name input/BASE_NAME.params
#   is searched for. If multiple params files are specified, then a separate set of tests
#   will be run for each of the params files
#
# NO_PARAMS - if set, then no params files will be used for this test. Setting PARAMS and
#   NO_PARAMS will result in an error
#
# MPI_ONLY and NO_MPI are mutually exclusive. Both arguments are options.
# By default, one non-MPI test case is created, and two MPI test cases are created, for
# 2 and 4 copies
#
# MIN_MPI_COPIES - minimum number of MPI copies. Default is 2
#
# MAX_MPI_COPIES - maximum number of MPI copies. Test cases are incremented by powers of two,
#  First case is 2, next is 4, and so on, until MAX_MPI_COPIES is reached. Default is 4
#
# FLAGS - by default, the flags are "-t ${PV_SYSTEM_TEST_THREADS}" or
#  "-t ${PV_SYSTEM_TEST_THREADS} -shuffle ${PV_SYSTEM_TEST_SHUFFLE}".
#  Setting FLAGS appends to this
#
# BASE_NAME - by default, this is the last component of the path ${CMAKE_CURRENT_SOURCE_DIR}.
#  Set BASE_NAME to override
#
# SRCFILES - The list source files for the executable that this test runs. There must be
#  at least one source file. This list of files will be passed to pv_add_executable().

function(pv_add_test_error TEST_NAME MESSAGE)
    message(ERROR "Cannot add OpenPV test ${TEST_NAME}. ${MESSAGE}")
endfunction()

macro(pv_add_test)
  cmake_parse_arguments(PARSED_ARGS "MPI_ONLY;NO_MPI;NO_PARAMS" "MIN_MPI_COPIES;MAX_MPI_COPIES;FLAGS;BASE_NAME" "PARAMS;SRCFILES" ${ARGN})

  set(ERROR_STATE OFF)

  get_filename_component(BASE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  if(PARSED_ARGS_BASE_NAME)
    set(BASE_NAME ${PARSED_ARGS_BASE_NAME})
  endif()

  if (PARSED_ARGS_MPI_ONLY AND PARSED_ARGS_NO_MPI)
    pv_add_test_error(${BASE_NAME} "MPI_ONLY and NO_MPI are both set. These options are mutually exclusive.")
    set(ERROR_STATE ON)
  endif()

  set(MIN_MPI_COPIES 2)
  if(PARSED_ARGS_MIN_MPI_COPIES)
    set(MIN_MPI_COPIES ${PARSED_ARGS_MIN_MPI_COPIES})
  endif()

  set(MAX_MPI_COPIES 4)
  if(PARSED_ARGS_MAX_MPI_COPIES)
    set(MAX_MPI_COPIES ${PARSED_ARGS_MAX_MPI_COPIES})
  endif()

  # Set the TEST_FLAGS
  set(TEST_FLAGS "-t;${PV_SYSTEM_TEST_THREADS}")
  if (PV_SYSTEM_TEST_SHUFFLE)
    set(TEST_FLAGS "${TEST_FLAGS};-shuffle;${PV_SYSTEM_TEST_SHUFFLE}")
  endif()
  if (PARSED_ARGS_FLAGS)
    string(REPLACE " " ";" FLAG_LIST ${PARSED_ARGS_FLAGS})
    set(TEST_FLAGS "${TEST_FLAGS};${FLAG_LIST}")
  endif()

  # Set the list of params files, if needed
  set(PARAMS "")
  if(PARSED_ARGS_PARAMS)
    if(PARSED_ARGS_NO_PARAMS)
      set(ERROR_STATE ON)
      pv_add_test_error(${BASE_NAME} "NO_PARAMS and PARAMS are both set. These options are mutually exclusive")
    else()
      set(PARAMS ${PARSED_ARGS_PARAMS})
    endif()
  else()
    if (NOT PARSED_ARGS_NO_PARAMS)
      set(PARAMS ${BASE_NAME})
    endif()
  endif()

  if (NOT ERROR_STATE)
    set(TEST_LOG_DIR ${CMAKE_CURRENT_BINARY_DIR})
    set(TEST_BINARY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${BASE_NAME}")
    set(TEST_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(TEST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
  
    # Add the executable
    
    string(COMPARE EQUAL "${PARSED_ARGS_SRCFILES}" "" SRCFILES_EMPTY)
    if (SRCFILES_EMPTY)
      message(FATAL_ERROR "${BASE_NAME} did not contain any source files")
    endif (SRCFILES_EMPTY)

    include_directories("${TEST_TOP_DIR}/../Shared")
    pv_add_executable(${BASE_NAME} SRC ${PARSED_ARGS_SRCFILES} OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
    add_dependencies(${BASE_NAME} pv)
    if (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
      set(TEST_SOURCE_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/input")
      if (EXISTS "${TEST_SOURCE_INPUT}")
        set(TEST_BINARY_INPUT "${CMAKE_CURRENT_BINARY_DIR}/input")
        execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${TEST_SOURCE_INPUT} ${TEST_BINARY_INPUT})
      endif (EXISTS "${TEST_SOURCE_INPUT}")
    endif()
    
    set(FIRST_TEST ON)
    set(PREV_TEST_NAME "")

    if (PARAMS)
      list(LENGTH PARAMS NUM_PARAMS)
      foreach (PARAM ${PARAMS})
        if(NUM_PARAMS GREATER 1)
          set(TEST_BASE_NAME ${BASE_NAME}_${PARAM})
        else()
          set(TEST_BASE_NAME ${BASE_NAME})
        endif()

        # Run single-process test
        if (NOT PARSED_ARGS_MPI_ONLY)
          # One process, no MPI
          set(TEST_NAME "${TEST_BASE_NAME}_1")
          set(TEST_LOG "${TEST_LOG_DIR}/${TEST_NAME}.log")
          set(TEST_PARAMS "input/${PARAM}.params")
          if (PV_MPI_SINGLE_PROCESS_TEST)
            set(COPIES 1)
            add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
              COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${COPIES} ${MPIEXEC_PREFLAGS} ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_FLAGS} -p ${TEST_PARAMS} -l ${TEST_LOG})
          else()
            add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
              COMMAND ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_FLAGS} -p ${TEST_PARAMS} -l ${TEST_LOG})
          endif()
          set(FIRST_TEST OFF)
          set(PREV_TEST_NAME ${TEST_NAME})
        endif()
        
        # Run MPI tests
        if (PV_USE_MPI AND MPI_FOUND AND NOT PARSED_ARGS_NO_MPI)
          set(COPIES ${MIN_MPI_COPIES})
          while((COPIES EQUAL MIN_MPI_COPIES OR COPIES GREATER MIN_MPI_COPIES) AND
                (COPIES EQUAL MAX_MPI_COPIES OR COPIES LESS    MAX_MPI_COPIES))
            set(TEST_NAME "${TEST_BASE_NAME}_${COPIES}")
            set(TEST_LOG "${TEST_LOG_DIR}/${TEST_NAME}.log")
            set(TEST_PARAMS "input/${PARAM}.params")
            add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
              COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${COPIES} ${MPIEXEC_PREFLAGS} ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_FLAGS} -p ${TEST_PARAMS} -l ${TEST_LOG})

            if (NOT FIRST_TEST)
              set_tests_properties(${TEST_NAME} PROPERTIES DEPENDS ${PREV_TEST_NAME})
              set(FIRST_TEST OFF)
            endif()
            
            set(PREV_TEST_NAME ${TEST_NAME})
            math(EXPR COPIES "${COPIES} * 2")
          endwhile()
        endif()
      endforeach()
    else()
      # This is ugly. The only difference between this block and the if(PARAMS) block
      # is that add_test doesn't have the -p params flag. Commonalities should be found,
      # factored out and code reuse be made to happen.
      set(TEST_BASE_NAME ${BASE_NAME})

      # Run single-process test
      if (NOT PARSED_ARGS_MPI_ONLY)
        # One process, no MPI
        set(TEST_NAME "${TEST_BASE_NAME}_1")
        set(TEST_LOG "${TEST_LOG_DIR}/${TEST_NAME}.log")
        if (PV_MPI_SINGLE_PROCESS_TEST)
          set(COPIES 1)
          add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${COPIES} ${MPIEXEC_PREFLAGS} ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_FLAGS} -l ${TEST_LOG})
        else()
          add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_FLAGS} -l ${TEST_LOG})
        endif()
        set(FIRST_TEST OFF)
        set(PREV_TEST_NAME ${TEST_NAME})
      endif()
      
      # Run MPI tests
      if (PV_USE_MPI AND MPI_FOUND AND NOT PARSED_ARGS_NO_MPI)
        set(COPIES ${MIN_MPI_COPIES})
        while((COPIES EQUAL MIN_MPI_COPIES OR COPIES GREATER MIN_MPI_COPIES) AND
              (COPIES EQUAL MAX_MPI_COPIES OR COPIES LESS    MAX_MPI_COPIES))
          set(TEST_NAME "${TEST_BASE_NAME}_${COPIES}")
          set(TEST_LOG "${TEST_LOG_DIR}/${TEST_NAME}.log")
          
          add_test(NAME ${TEST_NAME} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${COPIES} ${MPIEXEC_PREFLAGS} ${PV_SYSTEM_TEST_COMMAND} ${TEST_BINARY} ${TEST_FLAGS} -l ${TEST_LOG})
          
          if (NOT FIRST_TEST)
            set_tests_properties(${TEST_NAME} PROPERTIES DEPENDS ${PREV_TEST_NAME})
            set(FIRST_TEST OFF)
          endif()
          
          set(PREV_TEST_NAME ${TEST_NAME})
          math(EXPR COPIES "${COPIES} * 2")
        endwhile()
      endif()
    endif()
  endif()
endmacro()
