set(SRC_CPP
  src/LIFTestProbe.cpp
  src/main.cpp
)

set(SRC_HPP
  src/LIFTestProbe.hpp
)

set(TEST_PARAMS "LIFTest")

if (PV_USE_CUDA)
  set(TEST_PARAMS "${TEST_PARAMS} LIFTest_GPU")
endif()

pv_add_test(PARAMS ${testParams} SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
