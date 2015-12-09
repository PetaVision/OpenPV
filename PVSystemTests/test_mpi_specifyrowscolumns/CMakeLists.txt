set(testBaseName test_mpi_specifyrowscolumns) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
set(testTopDir ${CMAKE_CURRENT_SOURCE_DIR})
set(testBinDir "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}")

if (${PV_USE_MPI})
    #This addtest is hard coded because this test requires a run with exactly 6 mpi processes
    add_test(${testBaseName}_6 ${CMAKE_COMMAND} -E chdir ${testTopDir}
       ${MPIEXEC} -np 6 ${testBinDir}/${testBaseName} ${testFlags} -l
       ${testBaseName}_6.log)
endif (${PV_USE_MPI})

PVSystemsTest()
