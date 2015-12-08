set(testBaseName BatchSweepTest) #The base name of the test
set(testParams BatchSweepTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS} -batchwidth 4) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
set(testTopDir ${CMAKE_CURRENT_SOURCE_DIR})
set(testBinDir "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")

if (${PV_USE_MPI})
   #This addtest is hard coded because this test requires a run with exactly 4 mpi processes
   add_test(${testBaseName}_4 ${CMAKE_COMMAND} -E chdir ${testTopDir}
      ${MPIEXEC} -np 4 ${testBinDir}/${testBaseName} ${testFlags} -p input/BatchSweepTest.params -l
      ${testBaseName}_4.log)
endif (${PV_USE_MPI})

PVSystemsTest()
