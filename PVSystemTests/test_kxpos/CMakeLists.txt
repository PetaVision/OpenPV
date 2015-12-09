set(testBaseName test_kxpos) #The base name of the test
set(testParams ) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi FALSE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})
