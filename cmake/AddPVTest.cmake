#Define function for adding a test to the test harness
macro(AddPVTest BaseName ParamNames inFlags inMpi)
   if (NOT CMAKE_BUILD_TYPE)
       set(CMAKE_BUILD_TYPE Debug) #Can be: None, Debug, Release, RelWithDebInfo, MinSizeRel
   endif (NOT CMAKE_BUILD_TYPE)

   set(TEST_DIR ${CMAKE_CURRENT_BINARY_DIR})
   set(LOG_DIR ${TEST_DIR})
   set(TEST_BINARY "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${BaseName}")
   set(PARAMS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/input")
   set(TEST_TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

   #Case based on size of paramNames
   list(LENGTH ${ParamNames} numParams)

   #No params case
   if(${numParams} EQUAL 0)
      set(testName ${BaseName})
      #One process
      add_test(${testName}_1 ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
        ${TEST_BINARY} ${inFlags} -l
        ${LOG_DIR}/${testName}_1.log)
      if(${PV_USE_MPI} AND ${inMpi})
         #Two processes
         add_test(${testName}_2 ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
            ${MPIEXEC} -np 2 ${TEST_BINARY} ${inFlags} -l 
            ${LOG_DIR}/${testName}_2.log)
         #Four processes
         add_test(${testName}_4 ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
            ${MPIEXEC} -np 4 ${TEST_BINARY} ${inFlags} -l 
            ${LOG_DIR}/${testName}_4.log)
         #Add dependencies
         set_tests_properties(${testName}_2 PROPERTIES DEPENDS ${testName}_1)
         set_tests_properties(${testName}_4 PROPERTIES DEPENDS ${testName}_2)
      endif(${PV_USE_MPI} AND ${inMpi})
   else(${numParams} EQUAL 0)
      #Multiple params here
      foreach(param IN LISTS ${ParamNames})
         #Set test name based on number of parameters
         if(${numParams} GREATER 1)
            set(testName ${BaseName}_${param})
         else(${numParams} GREATER 1)
            set(testName ${BaseName})
         endif(${numParams} GREATER 1)
         #One process
         add_test(${testName}_1 ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
            ${TEST_BINARY} -p ${PARAMS_DIR}/${param}.params ${inFlags} -l
            ${LOG_DIR}/${testName}_1.log)
         if(${PV_USE_MPI} AND ${inMpi})
            #Two processes
            add_test(${testName}_2 ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
               ${MPIEXEC} -np 2 ${TEST_BINARY} -p ${PARAMS_DIR}/${param}.params
               ${inFlags} -l
               ${LOG_DIR}/${testName}_2.log)
            #Four processes
            add_test(${testName}_4 ${CMAKE_COMMAND} -E chdir ${TEST_TOP_DIR}
               ${MPIEXEC} -np 4 ${TEST_BINARY} -p ${PARAMS_DIR}/${param}.params
               ${inFlags} -l
               ${LOG_DIR}/${testName}_4.log)
            #Add dependencies
            set_tests_properties(${testName}_2 PROPERTIES DEPENDS ${testName}_1)
            set_tests_properties(${testName}_4 PROPERTIES DEPENDS ${testName}_2)
         endif(${PV_USE_MPI} AND ${inMpi})
      endforeach(param)
   endif(${numParams} EQUAL 0)

   PVSystemsTest()
endmacro(AddPVTest)