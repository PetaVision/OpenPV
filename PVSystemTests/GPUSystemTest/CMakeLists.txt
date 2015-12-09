if (PV_USE_CUDA OR PV_USE_OPENCL)
   set(testBaseName GPUSystemTest) #The base name of the test
   set(testParams preTest postTest postTestNoTranspose postTestOneToMany
      postTest_linked postTestGpuGroup HyPerLCAGpuTest
      postRecvBatch preRecvBatch) #Names of the parameter files in the input directory, leaving out .params
   set(testFlags -t ${PV_SYSTEM_TEST_THREADS}) #Extra flags to pass to PetaVision test
   set(testMpi TRUE) #Whether to call inside mpiexec with multiple processes
   AddPVTest(${testBaseName} testParams "${testFlags}" ${testMpi})
endif (PV_USE_CUDA OR PV_USE_OPENCL)
