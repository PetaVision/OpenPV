set(testBaseName BasicSystemTest) #The base name of the test
set(testParams BasicSystemTest) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest1(${testBaseName} testParams "${testFlags}" ${testMpi})

#include(PVSystemsTest)