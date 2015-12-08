set(testBaseName BatchMPICheckpointSystemTest) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
set(testTopDir ${CMAKE_CURRENT_SOURCE_DIR})

if (${PV_USE_MPI})
    #This addtest is hard coded because this test requires a run with 2, 4, and 8 mpi processes
    add_test(${testBaseName}_2 ${CMAKE_COMMAND} -E chdir ${testTopDir}
       ${MPIEXEC} -np 2 Debug/${testBaseName} ${testFlags} -l
       ${testBaseName}_2.log)
    add_test(${testBaseName}_4 ${CMAKE_COMMAND} -E chdir ${testTopDir}
       ${MPIEXEC} -np 4 Debug/${testBaseName} ${testFlags} -l
       ${testBaseName}_4.log)
    add_test(${testBaseName}_8 ${CMAKE_COMMAND} -E chdir ${testTopDir}
       ${MPIEXEC} -np 8 Debug/${testBaseName} ${testFlags} -l
       ${testBaseName}_8.log)
    #Add dependencies
    set_tests_properties(${testBaseName}_4 PROPERTIES DEPENDS ${testBaseName}_2)
    set_tests_properties(${testBaseName}_8 PROPERTIES DEPENDS ${testBaseName}_4)
endif (${PV_USE_MPI})

PVSystemsTest()