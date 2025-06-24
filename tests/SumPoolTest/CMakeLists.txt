set(SRC_CPP
  src/SumPoolTest.cpp
)

set(TEST_PARAMS "sumpoolTestPre;sumpoolTestPost;unpoolTestPre;unpoolTestPost;broadcastPoolingTest")

if (PV_USE_CUDA)
  set(TEST_PARAMS "${TEST_PARAMS};sumpoolTestPostGPU")
endif()

pv_add_test(PARAMS ${TEST_PARAMS} SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
