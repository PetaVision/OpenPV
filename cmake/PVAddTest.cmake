# pv_add_test([ONLY_MPI | NO_MPI] [MAX_MPI_COPIES] [FLAGS] [BASE_NAME])
#
# All arguments are optional
#
# Add a test case for the OpenPV library. This creates the binary and adds
# test tests. Multiple test cases are added by default.
# --------------------------------------------------------------------------------
# Example CMakeLists.txt files for a PVSystemTest:
#
# Most files will have a single line:
#
#   pv_add_test()
# 
#   This will add a test with a base name based on the test's directory name. The params file
#   will be named input/directory_name.params
#
#   This test will use 1,2, and 4 MPI copies
#
# No params files:
#
#   pv_add_test(NO_PARAMS)
#
# Alternate param files:
#
#   pv_add_test(PARAMS test_kernel test_kernel_normalizepost test_kernel_normalizepost_shrunken)
#
# No params files, no MPI:
#
#   pv_add_test(NO_PARAMS NO_MPI)
#
# Append special flags to the command line
#
#   pv_add_test(FLAGS "-c checkpoints/Checkpoint06 --testall")
#
# Specify the number of MPI copies, only test MPI, and special flags
#
#   pv_add_test(MIN_MPI_COPIES 4 MPI_ONLY FLAGS "-batchwidth 4")
#
# --------------------------------------------------------------------------------
# Options:
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
# FLAGS - by default, the flags are "-t ${PV_SYSTEM_TEST_THREADS}". Setting FLAGS appends
#  to this
#
# BASE_NAME - by default, this is the last component of the path ${CMAKE_CURRENT_SOURCE_DIR}.
#  Set BASE_NAME to override
#
# PARAMS - by default, this is set to BASE_NAME and a params file of name input/BASE_NAME.params
#   is searched for. If multiple params files are specified, then a separate set of tests
#   will be run for each of the params files
#
# NO_PARAMS - if set, then no params files will be used for this test. Setting PARAMS and
#   NO_PARAMS will result in an error
#

function(pv_add_test_error TEST_NAME MESSAGE)
    message(ERROR "Cannot add OpenPV test ${TEST_NAME}. ${MESSAGE}")
endfunction()

macro(pv_add_test)
  cmake_parse_arguments(PARSED_ARGS "MPI_ONLY;NO_MPI;NO_PARAMS" "MIN_MPI_COPIES;MAX_MPI_COPIES;FLAGS;BASE_NAME" "PARAMS" ${ARGN})
  
  set(ERROR_STATE OFF)

  get_filename_component(BASE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
  if(PARSED_ARGS_BASE_NAME)
    set(BASE_NAME ${PARSED_ARGS_BASE_NAME})
  endif()

  if (PARSED_ARGS_MPI_ONLY AND PARSED_ARGS_NO_MPI)
    pv_add_test_error(${BASE_NAME} "ONLY_MPI and NO_MPI are both set. These options are mutually exclusive.")
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
  if (PARSED_ARGS_FLAGS)
    string(REPLACE " " ";" FLAG_LIST ${PARSED_ARGS_FLAGS})
    set(TEST_FLAGS ${TEST_FLAGS};${FLAG_LIST})
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
    
    # File globbing is not a recommended best practice in the CMake docs.
    # But it's so darn easy and getting rid of this particular set of file
    # globbing is a bit tedious and the gains aren't that big.
    # 
    # To fix this, add a SRC multi_value_keywords to cmake_parse_arguments and specify
    # the list of sources in the call to pv_add_test. Then the list of sources will be in
    # PARSED_ARGS_SRC
    file(GLOB libSrcCPP ${TEST_SOURCE_DIR}/*.cpp)
    file(GLOB libSrcC ${TEST_SOURCE_DIR}/*.c)
    file(GLOB libSrcHPP ${TEST_SOURCE_DIR}/*.hpp)
    file(GLOB libSrcH ${TEST_SOURCE_DIR}/*.h)

    pv_add_executable(${BASE_NAME} SRC ${libSrcCPP} ${libSrcC} ${libSrcHPP} ${libSrcH} OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
    add_dependencies(${BASE_NAME} pv)
    
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

        # Run non-mpi-test
        if (NOT PARSED_ARGS_MPI_ONLY)
          # One process, no MPI
          set(TEST_NAME "${TEST_BASE_NAME}_1")
          set(TEST_LOG "${TEST_LOG_DIR}/${TEST_NAME}.log")
          set(TEST_PARAMS "input/${PARAM}.params")
          add_test(${TEST_NAME} ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
            ${TEST_BINARY} ${TEST_FLAGS} -p ${TEST_PARAMS} -l ${TEST_LOG})
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
            add_test(${TEST_NAME} ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
              ${MPIEXEC} -np ${COPIES} ${TEST_BINARY} ${TEST_FLAGS} -p ${TEST_PARAMS} -l ${TEST_LOG})

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

      # Run non-mpi-test
      if (NOT PARSED_ARGS_MPI_ONLY)
        # One process, no MPI
        set(TEST_NAME "${TEST_BASE_NAME}_1")
        set(TEST_LOG "${TEST_LOG_DIR}/${TEST_NAME}.log")
        add_test(${TEST_NAME} ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
          ${TEST_BINARY} ${TEST_FLAGS} -l ${TEST_LOG})
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
          
          add_test(${TEST_NAME} ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
            ${MPIEXEC} -np ${COPIES} ${TEST_BINARY} ${TEST_FLAGS} -l ${TEST_LOG})
          
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
