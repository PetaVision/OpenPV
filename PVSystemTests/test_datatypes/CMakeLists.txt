set(testBaseName test_datatypes) #The base name of the test
# No test params for this test. AddPVTest will see that testParams has a length
# of zero and will do the right thing
#set(testParams test_datatypes) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})
