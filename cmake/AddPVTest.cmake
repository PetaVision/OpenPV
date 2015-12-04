#Define function for adding a test to the test harness
macro(AddPVTest BaseName ParamNames inFlags inMpi)
   #Case based on size of paramNames
   list(LENGTH ${ParamNames} numParams)
   #No params case
   if(${numParams} EQUAL 0)
      set(testName ${BaseName})
      #One process
      add_test(${testName}_1 ${CMAKE_COMMAND} -E chdir ${BaseName}
         Debug/${BaseName} ${inFlags} -l
         ${testName}_1.log)
      if(${PV_USE_MPI} AND ${inMpi})
         #Two processes
         add_test(${testName}_2 ${CMAKE_COMMAND} -E chdir ${BaseName}
            ${MPIEXEC} -np 2 Debug/${BaseName} ${inFlags} -l 
            ${testName}_2.log)
         #Four processes
         add_test(${testName}_4 ${CMAKE_COMMAND} -E chdir ${BaseName}
            ${MPIEXEC} -np 4 Debug/${BaseName} ${inFlags} -l 
            ${testName}_4.log)
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
         add_test(${testName}_1 ${CMAKE_COMMAND} -E chdir ${BaseName}
            Debug/${BaseName} -p input/${param}.params ${inFlags} -l
            ${testName}_1.log)
         if(${PV_USE_MPI} AND ${inMpi})
            #Two processes
            add_test(${testName}_2 ${CMAKE_COMMAND} -E chdir ${BaseName}
               ${MPIEXEC} -np 2 Debug/${BaseName} -p input/${param}.params
               ${inFlags} -l
               ${testName}_2.log)
            #Four processes
            add_test(${testName}_4 ${CMAKE_COMMAND} -E chdir ${BaseName}
               ${MPIEXEC} -np 4 Debug/${BaseName} -p input/${param}.params
               ${inFlags} -l
               ${testName}_4.log)
            #Add dependencies
            set_tests_properties(${testName}_2 PROPERTIES DEPENDS ${testName}_1)
            set_tests_properties(${testName}_4 PROPERTIES DEPENDS ${testName}_2)
         endif(${PV_USE_MPI} AND ${inMpi})
      endforeach(param)
   endif(${numParams} EQUAL 0)
endmacro(AddPVTest)
