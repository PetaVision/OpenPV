if (${PV_USE_MPI})
   set(testBaseName BatchWeightUpdateMpiTest) #The base name of the test
   set(testParams ) #Names of the parameter files in the input directory, leaving out .params
   set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
   set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
   set(testTopDir ${CMAKE_CURRENT_SOURCE_DIR})
   set(testBinDir "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")

   #This addtest is hard coded because this test requires a run with exactly 10 mpi processes
   add_test(${testBaseName}_10 ${CMAKE_COMMAND} -E chdir ${testTopDir}
      ${MPIEXEC} -np 10 ${testBinDir}/${testBaseName} ${testFlags} -l
      ${testBaseName}_10.log)
endif (${PV_USE_MPI})

PVSystemsTest()
