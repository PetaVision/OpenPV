if (PV_USE_CUDA)
  pv_add_test(PARAMS preTest postTest postTestNoTranspose postTestOneToMany
      postTest_linked postTestGpuGroup HyPerLCAGpuTest
      postRecvBatch preRecvBatch diffMarginPostTest)
endif()
