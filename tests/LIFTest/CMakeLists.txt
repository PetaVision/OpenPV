set(TEST_PARAMS "LIFTest")

if (PV_USE_CUDA)
  set(TEST_PARAMS "${TEST_PARAMS} LIFTest_GPU")
endif()

pv_add_test(PARAMS ${testParams})
