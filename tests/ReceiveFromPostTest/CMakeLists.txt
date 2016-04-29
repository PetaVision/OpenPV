set(TEST_PARAMS postTest_margins postTestNoTranspose manyToOnePatchSizeTest oneToManyPatchSizeTest postTest_ManyToOne postTest_OneToMany)

if(PV_USE_CUDA)
   set(TEST_PARAMS "${TEST_PARAMS};postTestNoTranspose_GPU")
endif ()

pv_add_test(PARAMS ${TEST_PARAMS})

