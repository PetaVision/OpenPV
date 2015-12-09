set(testBaseName KernelTest) #The base name of the test
set(testParams test_kernel test_kernel_normalizepost test_kernel_normalizepost_shrunken) #Names of the parameter files in the input directory, leaving out .params
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})
