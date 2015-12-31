set(testBaseName LIFTest) #The base name of the test
if (PV_USE_CUDA OR PV_USE_OPENCL)
   set(testParams LIFTest LIFTest_GPU) #Names of the parameter files in the input directory, leaving out .params
else (PV_USE_CUDA OR PV_USE_OPENCL)
   set(testParams LIFTest) #Names of the parameter files in the input directory, leaving out .params
endif (PV_USE_CUDA OR PV_USE_OPENCL)
set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})
