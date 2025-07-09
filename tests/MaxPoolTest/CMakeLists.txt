set(SRC_CPP
   src/MaxPoolTest.cpp
)

set(TEST_PARAMS "maxpoolTestPre;maxpoolTestPost;unpoolTestPre;unpoolTestPost;broadcastPoolingTest")

if (PV_USE_CUDA)
  set(TEST_PARAMS "${TEST_PARAMS};maxpoolTestPostGPU")
endif()

pv_add_test(PARAMS ${TEST_PARAMS} SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
