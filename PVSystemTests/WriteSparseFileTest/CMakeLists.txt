set(testBaseName WriteSparseFileTest) #The base name of the test
set(testParams WriteSparseFileTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -c checkpoints/Checkpoint06 --testall -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})
